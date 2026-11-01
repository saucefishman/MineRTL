"""Macro flow: find repeated subcircuits, place and route one instance, stamp the rest.

subgraphs.find_repeated finds the repeats (cell type, pin role and which nets leave the instance all
count). build_macro runs the normal
placer and router on one instance, with its external nets as IO pins, and keeps the routed blocks as
a component: an input pin is its entry repeater, an output pin its exit repeater, so the top level
routes to them like to any gate pin. extract_macros swaps the instances for CUSTOM components.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from litemapy import BlockState, Region, Schematic

from minecraft_v.progress import log
from minecraft_v.placement_engine.ir import (
    CURRENT_SCHEMA_VERSION, CardinalDirection, Component, ComponentList, ComponentType, Direction, Footprint,
    NetConnection, NetEndpoint, PinRef,
)
from .block_utils import _is_air
from .subgraphs import RepeatedClass, find_repeated
from .template import _mirror_block_x, _rotate_block_180

Pos = tuple[int, int, int]
_IO_TYPES = (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)


@dataclass
class Macro:
    name: str
    footprint: Footprint
    pins: list[PinRef]
    blocks: list[tuple[Pos, BlockState]]             # relative to the footprint origin
    arcs: dict[str, dict[str, int]]                  # input pin -> output pin -> longest comb path, ticks
    q_out: dict[str, int]                            # output pin -> ticks from an inner register
    setup: dict[str, int]                            # input pin -> ticks to an inner register
    terminals: dict[str, Pos] = field(default_factory=dict)  # "cell:pin" -> inner wire end, relative
    gate_boxes: list[tuple[Pos, Pos]] = field(default_factory=list)  # inner gate footprints (lo, size)
    owner: dict[Pos, str] = field(default_factory=dict)      # inner routed cell -> inner net
    nets: dict[str, tuple[str, list[str]]] = field(default_factory=dict)  # inner net -> source, sinks
    port_net: dict[str, str] = field(default_factory=dict)   # pin -> inner net
    cells: int = 0

    def blocks_oriented(self, mirrored: bool, rotated: bool) -> list[tuple[Pos, BlockState]]:
        out = []
        for rel, block in self.blocks:
            if rotated:
                block = _rotate_block_180(block)
            if mirrored:
                block = _mirror_block_x(block)
            out.append((self.oriented(rel, mirrored, rotated), block))
        return out

    def oriented(self, rel: Pos, mirrored: bool, rotated: bool) -> Pos:
        x, y, z = rel
        if mirrored != rotated:  # half a turn also reverses X, so the two cancel there
            x = self.footprint.width - 1 - x
        if rotated:
            z = self.footprint.depth - 1 - z
        return x, y, z

    def world(self, rel: Pos, origin: Pos, mirrored: bool, rotated: bool = False) -> Pos:
        x, y, z = self.oriented(rel, mirrored, rotated)
        return origin[0] + x, origin[1] + y, origin[2] + z


@dataclass
class Instance:
    macro: str
    component_id: str
    cell_map: dict[str, str]  # macro's cell id -> netlist cell id
    pin_net: dict[str, str]   # macro pin -> netlist net


def _net_index(comp: ComponentList) -> tuple[dict[str, list[tuple[str, str, Direction]]], dict[tuple[str, str], str]]:
    ends: dict[str, list[tuple[str, str, Direction]]] = {}
    pin_net: dict[tuple[str, str], str] = {}
    for net in comp.nets:
        ends[net.net_id] = [(net.source.component_id, net.source.pin_name, Direction.OUT)] + [
            (s.component_id, s.pin_name, Direction.IN) for s in net.sinks]
        for cid, pin, _ in ends[net.net_id]:
            pin_net[(cid, pin)] = net.net_id
    return ends, pin_net


def _port_name(i: int) -> str:
    return f"p{i}"


def macro_netlist(cls: RepeatedClass, comp: ComponentList, back_unique: bool = False,
                  ) -> tuple[ComponentList, dict[str, str]]:
    """The template instance as its own netlist, external nets on IO pins. Also returns inner net -> port."""
    by_id = {c.id: c for c in comp.components}
    ends, _ = _net_index(comp)
    comps = [by_id[cid] for cid in sorted(cls.cells)]
    nets, ports = [], {}
    # Entries on both faces: each needs a Y per pin for its lanes, so this halves the height pins force.
    by_fanout = sorted((n for n, label in cls.nets.items() if label == "input"),
                       key=lambda n: (-len(ends[n]), n))
    if back_unique:  # shared inputs on the lane face, per-tile ones behind with the outputs
        north_inputs = {n: len({m[n] for m in cls.instances}) > 1 for n in by_fanout}
    else:
        north_inputs = {n: k % 2 == 1 for k, n in enumerate(by_fanout)}
    for net in sorted(cls.nets):
        label = cls.nets[net]
        inside = [e for e in ends[net] if e[0] in cls.cells]
        sinks = [NetEndpoint(component_id=c, pin_name=p) for c, p, d in inside if d == Direction.IN]
        if label == "internal":
            source = next((c, p) for c, p, d in inside if d == Direction.OUT)
        else:
            port = ports[net] = _port_name(len(ports))
            if label == "input":
                north = north_inputs.get(net, False)
                comps.append(Component(id=port, type=ComponentType.INPUT_PIN,
                                       pins=[PinRef(name=port, direction=Direction.OUT,
                                                    side=CardinalDirection.NORTH if north else None)],
                                       metadata={"north_face": True} if north else {"lane_pair": True} if back_unique else {}))
                source = (port, port)
            else:
                comps.append(Component(id=port, type=ComponentType.OUTPUT_PIN,
                                       pins=[PinRef(name=port, direction=Direction.IN)]))
                source = next((c, p) for c, p, d in inside if d == Direction.OUT)
                sinks.append(NetEndpoint(component_id=port, pin_name=port))
        nets.append(NetConnection(net_id=net, source=NetEndpoint(component_id=source[0], pin_name=source[1]),
                                  sinks=sinks))
    return ComponentList(schema_version=CURRENT_SCHEMA_VERSION, components=comps, nets=nets), ports


def _digest(sub: ComponentList) -> str:
    return hashlib.sha256(sub.model_dump_json().encode()).hexdigest()[:16]


def macro_timing(sub: ComponentList, net_ticks: dict[str, int], gate_delay: Callable[[Component], int],
                 sequential: frozenset[ComponentType]) -> tuple[dict, dict, dict]:
    """Longest paths through the routed macro: (input -> output arcs, register -> output, input -> register).

    Ticks include the entry repeater of an input pin and the exit repeater of an output pin.
    """
    by_id = {c.id: c for c in sub.components}
    fanout: dict[str, list[NetConnection]] = defaultdict(list)
    for net in sub.nets:
        fanout[net.source.component_id].append(net)

    memo: dict[str, tuple[dict[str, int], int | None]] = {}

    def ahead(cid: str) -> tuple[dict[str, int], int | None]:
        """Ticks from the output of cid to each output pin past it, and to the farthest register."""
        if cid not in memo:
            ports, reg = {}, None
            for net in fanout[cid]:
                wire = net_ticks.get(net.net_id, 0)
                for sink in net.sinks:
                    c = by_id[sink.component_id]
                    if c.type == ComponentType.OUTPUT_PIN:
                        ports[c.id] = max(ports.get(c.id, 0), wire + 1)
                    elif c.type in sequential:
                        reg = wire if reg is None else max(reg, wire)
                    else:
                        delay = wire + gate_delay(c)
                        more, more_reg = ahead(c.id)
                        for port, t in more.items():
                            ports[port] = max(ports.get(port, 0), t + delay)
                        if more_reg is not None:
                            reg = delay + more_reg if reg is None else max(reg, delay + more_reg)
            memo[cid] = ports, reg
        return memo[cid]

    arcs, q_out, setup = {}, {}, {}
    for c in sub.components:
        if c.type == ComponentType.INPUT_PIN:
            ports, reg = ahead(c.id)
            arcs[c.id] = {port: t + 1 for port, t in ports.items()}
            if reg is not None:
                setup[c.id] = reg + 1
        elif c.type in sequential:
            for port, t in ahead(c.id)[0].items():
                q_out[port] = max(q_out.get(port, 0), t + gate_delay(c))
    return arcs, q_out, setup


def macro_from_capture(name: str, sub: ComponentList, capture: dict, ports: dict[str, str], cells: int,
                       gate_delay, sequential) -> Macro:
    """Cut the routed inner build down to its non-air box and turn its IO repeaters into pins."""
    ws: Region = capture["workspace"]
    placed = {p.component.id: p for p in capture["placed"]}
    blocks = [(p, ws[p]) for p in ws.block_positions() if not _is_air(ws[p])]
    lo = tuple(min(p[a] for p, _ in blocks) for a in range(3))
    hi = tuple(max(p[a] for p, _ in blocks) for a in range(3))
    fp = Footprint(width=hi[0] - lo[0] + 1, height=hi[1] - lo[1] + 1, depth=hi[2] - lo[2] + 1)
    rel = lambda p: (p[0] - lo[0], p[1] - lo[1], p[2] - lo[2])  # noqa: E731
    pins = []
    for c in sub.components:
        if c.type not in _IO_TYPES:
            continue
        cell = capture["io_cells"][(c.id, c.id)]
        r = rel(cell)
        is_input = c.type == ComponentType.INPUT_PIN
        south = is_input and not c.metadata.get("north_face")
        # Facing away from the box is what lets the top level reach it.
        if (r[2] != fp.depth - 1) if south else (r[2] != 0):
            raise ValueError(f"pin {c.id} at {r} is not on its face of {fp}")
        pins.append(PinRef(
            name=c.id, direction=Direction.IN if is_input else Direction.OUT,
            side=CardinalDirection.SOUTH if south else CardinalDirection.NORTH,
            offset=(r[0], r[1], fp.depth - 1 - r[2]),
        ))
    arcs, q_out, setup = macro_timing(sub, capture["net_ticks"], gate_delay, sequential)
    nets = {n.net_id: (f"{n.source.component_id}:{n.source.pin_name}",
                       [f"{s.component_id}:{s.pin_name}" for s in n.sinks]) for n in sub.nets}
    return Macro(
        name=name, footprint=fp, pins=pins,
        blocks=[(rel(p), b) for p, b in blocks],
        arcs=arcs, q_out=q_out, setup=setup,
        terminals={f"{cid}:{pin}": rel(t) for (cid, pin), t in capture["pin_terminal"].items()},
        gate_boxes=[(rel(p.origin), (p.component.footprint.width, p.component.footprint.height,
                                     p.component.footprint.depth))
                    for p in placed.values() if p.component.type not in _IO_TYPES],
        owner={rel(p): n for p, n in capture["dust_owner"].items() if not n.startswith("__schematic__")},
        nets=nets, port_net={port: net for net, port in ports.items()}, cells=cells,
    )


def save_macro(macro: Macro, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fp = macro.footprint
    region = Region(0, 0, 0, fp.width, fp.height, fp.depth)
    for p, b in macro.blocks:
        region[p] = b
    Schematic(name=macro.name, author="MineRTL", regions={"macro": region}).save(str(directory / f"{macro.name}.litematic"))
    (directory / f"{macro.name}.json").write_text(json.dumps({
        "footprint": fp.model_dump(), "pins": [p.model_dump(mode="json") for p in macro.pins],
        "arcs": macro.arcs, "q_out": macro.q_out, "setup": macro.setup, "terminals": macro.terminals,
        "gate_boxes": macro.gate_boxes, "owner": [[list(p), n] for p, n in macro.owner.items()],
        "nets": macro.nets, "port_net": macro.port_net, "cells": macro.cells,
    }))


def load_macro(name: str, directory: Path) -> Macro | None:
    meta_path, schem_path = directory / f"{name}.json", directory / f"{name}.litematic"
    if not (meta_path.is_file() and schem_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text())
    region = next(iter(Schematic.load(str(schem_path)).regions.values()))
    blocks = [(p, region[p]) for p in region.block_positions() if not _is_air(region[p])]
    return Macro(
        name=name, footprint=Footprint(**meta["footprint"]), pins=[PinRef(**p) for p in meta["pins"]],
        blocks=blocks, arcs=meta["arcs"], q_out=meta["q_out"], setup=meta["setup"],
        terminals={k: tuple(v) for k, v in meta["terminals"].items()},
        gate_boxes=[(tuple(a), tuple(b)) for a, b in meta["gate_boxes"]],
        owner={tuple(p): n for p, n in meta["owner"]},
        nets={k: (s, list(t)) for k, (s, t) in meta["nets"].items()},
        port_net=meta["port_net"], cells=meta["cells"],
    )


_LANE_FLOOR = BlockState("minecraft:smooth_quartz")
_LANE_DUST = BlockState("minecraft:redstone_wire")
_LANE_REACH = 13   # dust cells between lane repeaters
_TAP_COLUMNS = 8   # shared pins step through this many X positions, so neighbours in Y don't touch
_TAP_PITCH = 3
_TILE_GAP = 3      # X between tiles, so that wires at their edges can't couple


def _lane_repeater(facing: str) -> BlockState:
    return BlockState("minecraft:repeater", facing=facing, delay="1")


def build_array(array_name: str, tile: Macro, cls: RepeatedClass, ports: dict[str, str]) -> tuple[Macro, Instance]:
    """One macro from all instances of a class: tiles in an X row, shared inputs on prebuilt lanes.

    A port every instance ties to one net runs as a straight lane along the input faces, on the port's own Y
    (tiles are built with one input pin per Y, see poisson._IoPlacer), so it never crosses another. Its only
    pin taps the middle of the lane. Other ports stay per tile.
    """
    fp = tile.footprint
    n = len(cls.instances)
    x0 = 0
    pitch = fp.width + _TILE_GAP
    width = n * pitch - _TILE_GAP
    tile_z = 3  # past the north face: pin, lane and spacer planes; the same again past the south face
    faces = {CardinalDirection.SOUTH: (fp.depth + tile_z, fp.depth + tile_z + 1, fp.depth + tile_z + 2, "south"),
             CardinalDirection.NORTH: (tile_z - 1, tile_z - 2, tile_z - 3, "north")}
    depth = fp.depth + 2 * tile_z
    blocks: dict[Pos, BlockState] = {}
    owner: dict[Pos, str] = {}
    terminals: dict[str, Pos] = {}
    gate_boxes: list[tuple[Pos, Pos]] = []
    nets: dict[str, tuple[str, list[str]]] = {}
    port_net: dict[str, str] = {}
    pins: list[PinRef] = []
    pin_net: dict[str, str] = {}
    arcs: dict[str, dict[str, int]] = {}
    q_out: dict[str, int] = {}
    setup: dict[str, int] = {}
    pin_by_name = {p.name: p for p in tile.pins}
    inner_of = {port: net for net, port in ports.items()}

    def tile_rel(k: int, p: Pos) -> Pos:
        return p[0] + x0 + k * pitch, p[1], p[2] + tile_z

    def lay(pos: Pos, block: BlockState, net: str | None = None, floor: bool = True) -> None:
        blocks[pos] = block
        if net is not None:
            owner[pos] = net
        if floor:
            blocks.setdefault((pos[0], pos[1] - 1, pos[2]), _LANE_FLOOR)

    for k, mapping in enumerate(cls.instances):
        for rel, block in tile.blocks:
            blocks[tile_rel(k, rel)] = block
        for rel, inner in tile.owner.items():
            owner[tile_rel(k, rel)] = mapping[inner]
        for lo, size in tile.gate_boxes:
            gate_boxes.append((tile_rel(k, lo), size))
        for key, rel in tile.terminals.items():
            cell, pin = key.rsplit(":", 1)
            if cell in cls.cells:
                terminals[f"{mapping[cell]}:{pin}"] = tile_rel(k, rel)
        for inner, (source, sinks) in tile.nets.items():
            net = mapping[inner]
            mapped = [f"{mapping[s.rsplit(':', 1)[0]]}:{s.rsplit(':', 1)[1]}" for s in sinks
                      if s.rsplit(":", 1)[0] in cls.cells]
            if source.rsplit(":", 1)[0] in cls.cells:
                nets[net] = (f"{mapping[source.rsplit(':', 1)[0]]}:{source.rsplit(':', 1)[1]}", mapped)
            else:
                nets[net] = ("", nets.get(net, ("", []))[1] + mapped)

    inputs = sorted((p for p in tile.pins if p.direction == Direction.IN), key=lambda p: p.offset[1])
    outputs = [p for p in tile.pins if p.direction == Direction.OUT]
    extra: dict[tuple[int, str], tuple[str, int]] = {}  # (tile, tile pin) -> (array pin, ticks before the tile pin)
    n_lanes = 0
    for port in inputs:
        tile_x, y = port.offset[0], port.offset[1]
        spacer_z, lane_z, pin_z, inward = faces[port.side]
        rel_nets = {m[inner_of[port.name]] for m in cls.instances}
        for k in range(n):
            lay((tile_x + x0 + k * pitch, y, spacer_z), _lane_repeater(inward),
                cls.instances[k][inner_of[port.name]])
        if len(rel_nets) == 1 and n > 1:
            net = rel_nets.pop()
            xs = [tile_x + x0 + k * pitch for k in range(n)]
            # Tap the middle: pins sit near each other on the face and the lane runs both ways from there.
            tap = width // 2 + (n_lanes % _TAP_COLUMNS - _TAP_COLUMNS // 2) * _TAP_PITCH
            n_lanes += 1
            lay((tap, y, pin_z), _lane_repeater(inward), net)
            lay((tap, y, lane_z), _LANE_DUST, net)
            ticks_at: dict[int, int] = {tap: 1}
            taps = set(xs)
            for step, facing, end in ((-1, "east", min(xs)), (1, "west", max(xs))):
                since, ticks = 0, 1
                for x in range(tap + step, end + step, step) if (end - tap) * step > 0 else ():
                    if since >= _LANE_REACH and x not in taps:
                        lay((x, y, lane_z), _lane_repeater(facing), net)
                        since, ticks = 0, ticks + 1
                    else:
                        lay((x, y, lane_z), _LANE_DUST, net)
                        since += 1
                    ticks_at[x] = ticks
            pin_name = port.name
            pins.append(PinRef(name=pin_name, direction=Direction.IN, side=port.side,
                               offset=(tap, y, depth - 1 - pin_z)))
            pin_net[pin_name] = net
            port_net[pin_name] = net
            nets[net] = (f"{pin_name}:{pin_name}", nets[net][1])
            for k, x in enumerate(xs):
                extra[(k, port.name)] = (pin_name, ticks_at[x] + 1)
        else:
            for k in range(n):
                net = cls.instances[k][inner_of[port.name]]
                x = tile_x + x0 + k * pitch
                name = f"t{k}_{port.name}"
                lay((x, y, lane_z), _lane_repeater(inward), net)
                lay((x, y, pin_z), _lane_repeater(inward), net)
                pins.append(PinRef(name=name, direction=Direction.IN, side=port.side,
                                   offset=(x, y, depth - 1 - pin_z)))
                pin_net[name] = net
                port_net[name] = net
                nets[net] = (f"{name}:{name}", nets[net][1])
                extra[(k, port.name)] = (name, 3)
    for k, mapping in enumerate(cls.instances):
        for port in outputs:
            net = mapping[inner_of[port.name]]
            name = f"t{k}_{port.name}"
            x = port.offset[0] + x0 + k * pitch
            for z in range(tile_z):  # out through the lane plane: its own Y, so no lane is in the way
                lay((x, port.offset[1], z), _lane_repeater("south"), net)
            pins.append(PinRef(name=name, direction=Direction.OUT, side=CardinalDirection.NORTH,
                               offset=(x, port.offset[1], depth - 1)))
            pin_net[name] = net
            port_net[name] = net
            nets[net] = (nets[net][0], nets[net][1] + [f"{name}:{name}"])
            if port.name in tile.q_out:
                q_out[name] = tile.q_out[port.name] + tile_z
        for port in inputs:
            array_pin, before = extra[(k, port.name)]
            into = arcs.setdefault(array_pin, {})
            for out, ticks in tile.arcs.get(port.name, {}).items():
                into[f"t{k}_{out}"] = max(into.get(f"t{k}_{out}", 0), ticks + before + tile_z)
            if port.name in tile.setup:
                setup[array_pin] = max(setup.get(array_pin, 0), tile.setup[port.name] + before)
    ys = [p[1] for p in blocks]
    low, high = min(ys), max(ys)
    array = Macro(
        name=array_name,
        footprint=Footprint(width=width, height=high - low + 1, depth=depth),
        pins=[p.model_copy(update={"offset": (p.offset[0], p.offset[1] - low, p.offset[2])}) for p in pins],
        blocks=[((x, y - low, z), b) for (x, y, z), b in blocks.items()],
        arcs=arcs, q_out=q_out, setup=setup,
        terminals={k: (p[0], p[1] - low, p[2]) for k, p in terminals.items()},
        gate_boxes=[((p[0], p[1] - low, p[2]), size) for p, size in gate_boxes],
        owner={(p[0], p[1] - low, p[2]): net for p, net in owner.items()},
        nets=nets, port_net=port_net, cells=len(cls.cells) * n,
    )
    identity = {cell: cell for mapping in cls.instances for cell in (mapping[c] for c in cls.cells)}
    return array, Instance(array_name, f"{array_name}_0", identity, pin_net)


# Between a bank's rows, in Z from the end of the first row: a spacer repeater, then two sets of lane planes, then the
# other row's spacer. Lanes on one Y share a set; alternate Ys use the other set, which leaves the Y above a lane free
# for bridges over it. The wider gap between the sets is where a bridge lands clear of the other set's floors.
_PLANE_PITCH = 2
_PIN_PITCH = 12                 # Z between the pins of a bank past its east end
_BANK_LANE_REACH = 9           # dust cells between lane repeaters, leaving signal for the stubs
_DUST_REACH = 14               # dust cells a repeater can drive
_TAP_CLEARANCE = 3             # X between taps of the two rows


def _lane_plane(per_y: int, lane_set: int, k: int) -> int:
    return 3 + lane_set * (_PLANE_PITCH * (per_y - 1) + 3) + _PLANE_PITCH * k


def _row_gap(per_y: int) -> int:
    return _lane_plane(per_y, 1, per_y - 1) + 4


def _row_shift(xs: set[int], width: int, pitch: int, columns: int) -> int:
    """X offset of the turned row that keeps its taps clear of the other row's."""
    upright = {k * pitch + x for k in range(columns) for x in xs}
    for shift in range(pitch):
        turned = {k * pitch + shift + width - 1 - x for k in range(columns) for x in xs}
        if all(abs(a - b) >= _TAP_CLEARANCE for a in upright for b in turned):
            return shift
    raise ValueError("no row offset keeps a bank's taps apart")


def build_bank(bank_name: str, tile: Macro, instances: list[dict[str, str]], cells: set[str], ports: dict[str, str],
               columns: int) -> tuple[Macro, Instance]:
    """One macro from up to 2 x columns tiles: two rows, the second turned half way, pin faces toward one lane plane.

    Every shared input is a lane between the rows, serving both, and fans out past the east end to its pin. Several
    lanes share a Y (as the tile's pins do); a tap to a far lane bridges over the nearer ones. Per-tile pins stay on the
    tiles' back faces, the outside of the box.
    """
    fp = tile.footprint
    pitch = fp.width + _TILE_GAP
    shared_pins = [p for p in tile.pins if p.side == CardinalDirection.SOUTH]
    shared_ys = sorted({p.offset[1] for p in shared_pins})
    per_y = max((sum(p.offset[1] == y for p in shared_pins) for y in shared_ys), default=1)
    row1_z = fp.depth + _row_gap(per_y)
    shift = _row_shift({p.offset[0] for p in shared_pins}, fp.width, pitch, columns)
    row_width = columns * pitch - _TILE_GAP + shift
    east = row_width + 2
    pin_x = east + _PLANE_PITCH * per_y + 1
    depth = row1_z + fp.depth
    width = pin_x + 1
    blocks: dict[Pos, BlockState] = {}
    wired: set[Pos] = set()
    floors: set[Pos] = set()
    owner: dict[Pos, str] = {}
    terminals: dict[str, Pos] = {}
    gate_boxes: list[tuple[Pos, Pos]] = []
    nets: dict[str, tuple[str, list[str]]] = {}
    port_net: dict[str, str] = {}
    pin_net: dict[str, str] = {}
    pins: list[PinRef] = []
    arcs: dict[str, dict[str, int]] = {}
    q_out: dict[str, int] = {}
    setup: dict[str, int] = {}
    inner_of = {port: net for net, port in ports.items()}
    oriented = {False: tile.blocks_oriented(False, False), True: tile.blocks_oriented(False, True)}

    def lay(pos: Pos, block: BlockState, net: str) -> None:
        below = (pos[0], pos[1] - 1, pos[2])
        if pos in wired or pos in floors or below in wired:
            raise ValueError(f"bank wiring collides at {pos}")
        wired.add(pos)
        blocks[pos] = block
        owner[pos] = net
        floors.add(below)
        blocks[below] = _LANE_FLOOR

    def place(k: int) -> tuple[tuple[int, int, int], bool]:
        row = k // columns
        return (k % columns * pitch + row * shift, 0, row * row1_z), row == 1

    def stub(x: int, y: int, lane: int, spacer: int, inward: int, crossed: int, since: int, net: str) -> int:
        """Dust from a lane to a tile's spacer repeater, up over ``crossed`` nearer lanes; returns repeaters on it."""
        def rise(n: int) -> int:  # n cells from the lane
            if crossed == 0 or n > 2 * crossed + 1:
                return 0
            return 1 if n in (1, 2 * crossed + 1) else 2

        cells = abs(spacer - lane) - 1
        ticks = 0
        for n in range(1, cells + 1):
            pos = (x, y + rise(n), lane + inward * n)
            level = rise(n - 1) == rise(n) == rise(n + 1) and n > 1
            if level and since + 1 + cells - n > _DUST_REACH:
                lay(pos, _lane_repeater("south" if inward < 0 else "north"), net)
                since, ticks = 0, ticks + 1
            else:
                lay(pos, _LANE_DUST, net)
                since += 1
        return ticks

    def at(k: int, rel: Pos) -> Pos:
        origin, turned = place(k)
        x, y, z = tile.oriented(rel, False, turned)
        return origin[0] + x, y, origin[2] + z

    for k, mapping in enumerate(instances):
        origin, turned = place(k)
        for (x, y, z), block in oriented[turned]:
            blocks[(origin[0] + x, y, origin[2] + z)] = block
        for rel, inner in tile.owner.items():
            owner[at(k, rel)] = mapping[inner]
        for (x, y, z), (sx, sy, sz) in tile.gate_boxes:
            corners = [at(k, (x, y, z)), at(k, (x + sx - 1, y, z + sz - 1))]
            gate_boxes.append(((min(c[0] for c in corners), y, min(c[2] for c in corners)), (sx, sy, sz)))
        for key, rel in tile.terminals.items():
            cell, pin = key.rsplit(":", 1)
            if cell in cells:
                terminals[f"{mapping[cell]}:{pin}"] = at(k, rel)
        for inner, (source, sinks) in tile.nets.items():
            net = mapping[inner]
            mapped = [f"{mapping[e.rsplit(':', 1)[0]]}:{e.rsplit(':', 1)[1]}" for e in sinks
                      if e.rsplit(":", 1)[0] in cells]
            if source.rsplit(":", 1)[0] in cells:
                nets[net] = (f"{mapping[source.rsplit(':', 1)[0]]}:{source.rsplit(':', 1)[1]}", mapped)
            else:
                nets[net] = ("", nets.get(net, ("", []))[1] + mapped)

    extra: dict[tuple[int, str], tuple[str, int]] = {}
    on_y: dict[int, int] = defaultdict(int)
    lanes_on = {y: sum(p.offset[1] == y for p in shared_pins) for y in shared_ys}
    for port in sorted(tile.pins, key=lambda p: (p.offset[1], p.offset[0])):
        tile_net = inner_of[port.name]
        tile_x, y = port.offset[0], port.offset[1]
        shared = port.side == CardinalDirection.SOUTH
        if not shared:  # stays on the back of its tile
            for k, mapping in enumerate(instances):
                turned = place(k)[1]
                net = mapping[tile_net]
                rel = (tile_x, y, 0 if port.direction == Direction.OUT or not shared else fp.depth - 1)
                x, _, z = at(k, rel)
                name = f"t{k}_{port.name}"
                pins.append(PinRef(name=name, direction=port.direction,
                                   side=CardinalDirection.SOUTH if turned else CardinalDirection.NORTH,
                                   offset=(x, y, depth - 1 - z)))
                pin_net[name] = net
                port_net[name] = net
                if port.direction == Direction.OUT:
                    nets[net] = (nets[net][0], nets[net][1] + [f"{name}:{name}"])
                else:
                    nets[net] = (f"{name}:{name}", nets[net][1])
                    extra[(k, port.name)] = (name, 0)
            continue
        net = instances[0][tile_net]
        lane_set = shared_ys.index(y) % 2
        k_lane = on_y[y]
        on_y[y] += 1
        lane_z = fp.depth + _lane_plane(per_y, lane_set, k_lane)
        pin_z = depth // 2 + (2 * k_lane + lane_set - per_y) * _PIN_PITCH
        # Lanes fan out to the pins, the outermost turning first, so that none crosses another on this Y.
        north = pin_z < lane_z
        turn_x = east + _PLANE_PITCH * (k_lane if north else per_y - 1 - k_lane)
        lay((pin_x, y, pin_z), _lane_repeater("east"), net)
        taps = {at(k, (tile_x, y, fp.depth - 1))[0] for k in range(len(instances))}
        path = [(x, pin_z, "east") for x in range(pin_x - 1, turn_x, -1)]
        step = 1 if lane_z > pin_z else -1
        path += [(turn_x, z, None if z in (pin_z, lane_z) else "north" if step > 0 else "south")
                 for z in range(pin_z, lane_z + step, step)]
        path += [(x, lane_z, "east") for x in range(turn_x - 1, min(taps) - 1, -1)]
        since, ticks = 0, 1
        ticks_at: dict[int, int] = {}
        since_at: dict[int, int] = {}
        for x, z, facing in path:
            if since >= _BANK_LANE_REACH and facing is not None and x not in taps:
                lay((x, y, z), _lane_repeater(facing), net)
                since, ticks = 0, ticks + 1
            else:
                lay((x, y, z), _LANE_DUST, net)
                since += 1
            if z == lane_z:
                ticks_at[x], since_at[x] = ticks, since
        stub_ticks: dict[int, int] = {}
        for k in range(len(instances)):
            x, _, z = at(k, (tile_x, y, fp.depth - 1))
            turned = place(k)[1]
            inward = 1 if turned else -1  # from the lanes toward this tile
            spacer = z - inward
            lay((x, y, spacer), _lane_repeater("south" if inward < 0 else "north"), net)
            crossed = lanes_on[y] - 1 - k_lane if turned else k_lane
            stub_ticks[k] = stub(x, y, lane_z, spacer, inward, crossed, since_at[x], net)
        name = port.name
        pins.append(PinRef(name=name, direction=Direction.IN, side=CardinalDirection.EAST,
                           offset=(pin_x, y, depth - 1 - pin_z)))
        pin_net[name] = net
        port_net[name] = net
        nets[net] = (f"{name}:{name}", nets[net][1])
        for k in range(len(instances)):
            extra[(k, port.name)] = (name, ticks_at[at(k, (tile_x, y, 0))[0]] + stub_ticks[k] + 1)
    for k in range(len(instances)):
        for port in tile.pins:
            if port.direction == Direction.OUT and port.name in tile.q_out:
                q_out[f"t{k}_{port.name}"] = tile.q_out[port.name]
            if port.direction != Direction.IN:
                continue
            array_pin, before = extra[(k, port.name)]
            into = arcs.setdefault(array_pin, {})
            for out, ticks in tile.arcs.get(port.name, {}).items():
                into[f"t{k}_{out}"] = max(into.get(f"t{k}_{out}", 0), ticks + before)
            if port.name in tile.setup:
                setup[array_pin] = max(setup.get(array_pin, 0), tile.setup[port.name] + before)
    low = min(p[1] for p in blocks)
    high = max(p[1] for p in blocks)
    bank = Macro(
        name=bank_name, footprint=Footprint(width=width, height=high - low + 1, depth=depth),
        pins=[p.model_copy(update={"offset": (p.offset[0], p.offset[1] - low, p.offset[2])}) for p in pins],
        blocks=[((x, y - low, z), b) for (x, y, z), b in blocks.items()],
        arcs=arcs, q_out=q_out, setup=setup,
        terminals={k: (p[0], p[1] - low, p[2]) for k, p in terminals.items()},
        gate_boxes=[((p[0], p[1] - low, p[2]), size) for p, size in gate_boxes],
        owner={(p[0], p[1] - low, p[2]): net for p, net in owner.items()},
        nets=nets, port_net=port_net, cells=len(cells) * len(instances),
    )
    identity = {cell: cell for mapping in instances for cell in (mapping[c] for c in cells)}
    return bank, Instance(bank_name, f"{bank_name}_0", identity, pin_net)


def io_column_pins(n_pins: int, y_gap: int, column_gap: int) -> int:
    """Pins per IO column that keep a macro's pin face about square."""
    return max(1, math.ceil(math.sqrt(n_pins * column_gap / y_gap)))


def extract_macros(comp: ComponentList, build: Callable[[str, ComponentList, dict[str, str], int], Macro],
                   max_fanout: int = 2, min_cells: int = 8, row_tiles: int = 0, bank_columns: int = 0, min_instances: int = 2, on_class: Callable[[RepeatedClass, Macro], None] | None = None,
                   ) -> tuple[ComponentList, dict[str, Macro], list[Instance]]:
    """Replace every repeated subcircuit with a CUSTOM component; returns (top netlist, macros, instances)."""
    macros: dict[str, Macro] = {}
    instances: list[Instance] = []
    new_components: list[Component] = []
    hidden_cells: set[str] = set()
    ends, _ = _net_index(comp)
    pin_rows: list[tuple[str, str, str, Direction]] = []

    # Small repeats route worse as macros than as gates.
    classes = [cls for cls in find_repeated(comp, max_fanout, min_cells) if len(cls.instances) >= min_instances]
    for cls in classes:
        banked = bank_columns > 0 and len(cls.instances) >= 2 * bank_columns
        sub, ports = macro_netlist(cls, comp, back_unique=banked)
        tile_name = f"macro_{_digest(sub)}"
        tile = build(tile_name, sub, ports, len(cls.cells))
        if on_class is not None:
            on_class(cls, tile)
        per_array = 2 * bank_columns if banked else row_tiles or len(cls.instances)
        for start in range(0, len(cls.instances), per_array):
            row = replace(cls, instances=cls.instances[start:start + per_array])
            name = f"{'bank' if banked else 'array'}_{len(macros)}"
            if banked:
                macro, instance = build_bank(name, tile, row.instances, cls.cells, ports, bank_columns)
            else:
                macro, instance = build_array(name, tile, row, ports)
            macros[name] = macro
            fp = macro.footprint
            log(f"[macro] {name}: {len(row.instances)} x {tile_name}, footprint {fp.width}x{fp.height}x{fp.depth}, "
                  f"{len(macro.pins)} pins")
            hidden_cells.update(instance.cell_map)
            instances.append(instance)
            new_components.append(Component(id=instance.component_id, type=ComponentType.CUSTOM, pins=macro.pins,
                                            footprint=fp, metadata={"macro": name, **({"bank": True} if banked else {})}))
            for pin in macro.pins:
                pin_rows.append((instance.pin_net[pin.name], instance.component_id, pin.name, pin.direction))
    if not instances:
        return comp, {}, []

    rows: dict[str, list[tuple[str, str, Direction]]] = defaultdict(list)
    for net, cid, pin, direction in pin_rows:
        rows[net].append((cid, pin, direction))
    components = [c for c in comp.components if c.id not in hidden_cells] + new_components
    nets = []
    for net in comp.nets:
        kept = [e for e in ends[net.net_id] if e[0] not in hidden_cells] + rows.get(net.net_id, [])
        sources = [e for e in kept if e[2] == Direction.OUT]
        sinks = [e for e in kept if e[2] == Direction.IN]
        if not sources or not sinks:
            continue
        nets.append(NetConnection(
            net_id=net.net_id, source=NetEndpoint(component_id=sources[0][0], pin_name=sources[0][1]),
            sinks=[NetEndpoint(component_id=c, pin_name=p) for c, p, _ in sinks],
        ))
    top = ComponentList(schema_version=CURRENT_SCHEMA_VERSION, components=components, nets=nets)
    check_equivalent(comp, top, macros, instances)
    return top, macros, instances


def _flat_nets(comp: ComponentList) -> set[tuple[tuple[str, str], frozenset[tuple[str, str]]]]:
    return {((n.source.component_id, n.source.pin_name),
             frozenset((s.component_id, s.pin_name) for s in n.sinks)) for n in comp.nets}


def check_equivalent(original: ComponentList, top: ComponentList, macros: dict[str, Macro],
                     instances: list[Instance]) -> None:
    """Flatten the top netlist through the macros; every net must match the original exactly."""
    by_id = {i.component_id: i for i in instances}

    def expand(cid: str, pin: str, want_source: bool) -> list[tuple[str, str]]:
        inst = by_id.get(cid)
        if inst is None:
            return [(cid, pin)]
        macro = macros[inst.macro]
        source, sinks = macro.nets[macro.port_net[pin]]
        ends = [source] if want_source else [s for s in sinks if s.rsplit(":", 1)[0] not in macro.port_net]
        return [(inst.cell_map[e.rsplit(":", 1)[0]], e.rsplit(":", 1)[1]) for e in ends]

    flat: set = set()
    consumed: set[tuple[str, str]] = set()
    for net in top.nets:
        source = expand(net.source.component_id, net.source.pin_name, True)
        sinks = frozenset(e for s in net.sinks for e in expand(s.component_id, s.pin_name, False))
        if net.source.component_id in by_id:
            inst = by_id[net.source.component_id]
            macro = macros[inst.macro]
            inner = macro.nets[macro.port_net[net.source.pin_name]]
            sinks |= {(inst.cell_map[s.rsplit(":", 1)[0]], s.rsplit(":", 1)[1]) for s in inner[1]
                      if s.rsplit(":", 1)[0] not in macro.port_net}
        flat.add((source[0], sinks))
        consumed.add((net.source.component_id, net.source.pin_name))
    for inst in instances:
        macro = macros[inst.macro]
        exposed = set(macro.port_net.values())
        for net, (source, sinks) in macro.nets.items():
            if net in exposed:
                continue
            flat.add(((inst.cell_map[source.rsplit(":", 1)[0]], source.rsplit(":", 1)[1]),
                      frozenset((inst.cell_map[s.rsplit(":", 1)[0]], s.rsplit(":", 1)[1]) for s in sinks)))
    want = _flat_nets(original)
    if flat != want:
        missing, extra = want - flat, flat - want
        raise ValueError(f"macro netlist is not the original: {len(missing)} nets missing, {len(extra)} extra; "
                         f"e.g. {next(iter(missing or extra))}")
