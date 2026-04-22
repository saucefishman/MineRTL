from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from litemapy import BlockState, Region, Schematic
from minecraft_v.cell_library import SCHEMATIC_MAP
from minecraft_v.placement_ir import (
    CardinalDirection,
    Component,
    ComponentList,
    ComponentType,
    Direction,
    Footprint,
    NetConnection,
    NetEndpoint,
    PinRef,
)

AIR = BlockState("minecraft:air")
STONE = BlockState("minecraft:stone")
REDSTONE = BlockState("minecraft:redstone_wire")

REPEATER_STRIDE = 15

def _block_str(block: BlockState) -> str:
    return str(block)

def _is_air(block: BlockState) -> bool:
    text = _block_str(block)
    return text == "minecraft:air" or text.startswith("minecraft:air[")

def _is_redstone_wire(block: BlockState) -> bool:
    return "redstone_wire" in _block_str(block)

def _is_repeater(block: BlockState) -> bool:
    return "minecraft:repeater" in _block_str(block)


def _needs_support(block: BlockState) -> bool:
    name = _block_str(block)
    return (
        "minecraft:repeater" in name
        or "minecraft:comparator" in name
        or "minecraft:redstone_wire" in name
        or "minecraft:redstone_torch[" in name
        or name == "minecraft:redstone_torch"
    )

_SIDE_NORMAL: dict[str, tuple[int, int, int]] = {
    CardinalDirection.SOUTH.value: (0, 0, 1),
    CardinalDirection.NORTH.value: (0, 0, -1),
    CardinalDirection.EAST.value: (1, 0, 0),
    CardinalDirection.WEST.value: (-1, 0, 0),
}

_OPPOSITE_SIDE: dict[str, str] = {
    CardinalDirection.SOUTH.value: CardinalDirection.NORTH.value,
    CardinalDirection.NORTH.value: CardinalDirection.SOUTH.value,
    CardinalDirection.EAST.value: CardinalDirection.WEST.value,
    CardinalDirection.WEST.value: CardinalDirection.EAST.value,
}


def _io_repeater_facing(component_type: ComponentType, side: CardinalDirection) -> str:
    if component_type == ComponentType.INPUT_PIN:
        return _OPPOSITE_SIDE[side.value]
    return side.value


def _default_io_side(component: Component, pin: PinRef) -> CardinalDirection:
    if pin.side is not None:
        return pin.side
    if component.type == ComponentType.INPUT_PIN:
        return CardinalDirection.SOUTH
    if component.type == ComponentType.OUTPUT_PIN:
        return CardinalDirection.NORTH
    raise ValueError(f"Missing pin.side for {component.id} pin {pin.name}")

def _non_air_bounds(template: Region) -> tuple[int, int, int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    zs: list[int] = []
    for pos in template.block_positions():
        if not _is_air(template[pos]):
            x, y, z = pos
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not xs:
        return (
            template.min_x(),
            template.min_y(),
            template.min_z(),
            template.max_x(),
            template.max_y(),
            template.max_z(),
        )
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

def _ensure_support(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    cell: tuple[int, int, int],
) -> None:
    x, y, z = cell
    if y <= 0:
        return
    below = (x, y - 1, z)
    if below in solid:
        return
    if _is_air(workspace[x, y - 1, z]):
        workspace[x, y - 1, z] = STONE
        solid.add(below)

def _paste_template(
    workspace: Region,
    template: Region,
    dest_origin: tuple[int, int, int],
    solid: set[tuple[int, int, int]],
) -> None:
    tmin_x, tmin_y, tmin_z, _, _, _ = _non_air_bounds(template)
    dx, dy, dz = dest_origin
    placed_cells: list[tuple[tuple[int, int, int], BlockState]] = []
    for pos in template.block_positions():
        block = template[pos]
        if _is_air(block):
            continue
        x, y, z = pos
        wx = dx + (x - tmin_x)
        wy = dy + (y - tmin_y)
        wz = dz + (z - tmin_z)
        workspace[wx, wy, wz] = block
        solid.add((wx, wy, wz))
        placed_cells.append(((wx, wy, wz), block))
    for cell, block in placed_cells:
        if _needs_support(block):
            _ensure_support(workspace, solid, cell)

def _pin_world(
    origin: tuple[int, int, int],
    pin: PinRef,
    footprint: Footprint,
) -> tuple[int, int, int]:
    ox, oy, oz = origin
    px, py, pz = pin.offset
    return (ox + px, oy + py, oz + (footprint.depth - 1 - pz))

def _inside_footprint(
    pos: tuple[int, int, int],
    origin: tuple[int, int, int],
    footprint: Footprint,
) -> bool:
    ox, oy, oz = origin
    x, y, z = pos
    return (
        ox <= x < ox + footprint.width
        and oy <= y < oy + footprint.height
        and oz <= z < oz + footprint.depth
    )

def _io_repeater_cell(
    pin_world: tuple[int, int, int],
    origin: tuple[int, int, int],
    footprint: Footprint,
    side: CardinalDirection,
    bounds: tuple[int, int, int, int, int, int] | None = None,
) -> tuple[int, int, int]:
    nx, ny, nz = _SIDE_NORMAL[side.value]
    x, y, z = pin_world
    cx, cy, cz = x + nx, y + ny, z + nz
    while _inside_footprint((cx, cy, cz), origin, footprint):
        cx += nx
        cy += ny
        cz += nz
    if bounds is not None:
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        if not (
            min_x <= cx <= max_x
            and min_y <= cy <= max_y
            and min_z <= cz <= max_z
        ):
            raise ValueError(
                f"IO repeater cell {(cx, cy, cz)} outside workspace bounds {bounds}; "
                f"pin_world={pin_world} origin={origin} side={side.value}"
            )
    return (cx, cy, cz)

def _expand_multibit_io(comp: ComponentList) -> ComponentList:
    renames: dict[tuple[str, str], tuple[str, str]] = {}
    new_components: list[Component] = []
    for c in comp.components:
        is_io = c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
        if not is_io or len(c.pins) <= 1:
            new_components.append(c)
            continue
        for pin in c.pins:
            new_id = pin.name if pin.name != c.id else f"{c.id}_{pin.name}"
            new_pin = PinRef(
                name=pin.name,
                direction=pin.direction,
                side=pin.side,
                offset=(0, 0, 0),
            )
            new_components.append(
                Component(
                    id=new_id,
                    type=c.type,
                    pins=[new_pin],
                    params=dict(c.params),
                    footprint=Footprint(width=1, height=1, depth=1),
                )
            )
            renames[(c.id, pin.name)] = (new_id, pin.name)

    def rebind(ep: NetEndpoint) -> NetEndpoint:
        key = (ep.component_id, ep.pin_name)
        if key in renames:
            nid, nm = renames[key]
            return NetEndpoint(component_id=nid, pin_name=nm)
        return ep

    new_nets = [
        NetConnection(
            net_id=net.net_id,
            source=rebind(net.source),
            sinks=[rebind(sink) for sink in net.sinks],
        )
        for net in comp.nets
    ]
    return ComponentList(
        schema_version=comp.schema_version,
        components=new_components,
        nets=new_nets,
    )

@dataclass
class _Placed:
    component: Component
    origin: tuple[int, int, int]

@dataclass
class _Layout:
    placed: list[_Placed]
    output_row_z: int
    gate_row_z: int
    input_row_z: int
    max_gate_depth: int
    base_y: int
    outputs_gutter: tuple[int, int]  # inclusive (z_min, z_max)
    inputs_gutter: tuple[int, int]

def _rows_split(components: list[Component]) -> tuple[list[Component], list[Component], list[Component]]:
    inputs = [c for c in components if c.type == ComponentType.INPUT_PIN]
    outputs = [c for c in components if c.type == ComponentType.OUTPUT_PIN]
    gates = [
        c
        for c in components
        if c.type
        not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN, ComponentType.CUSTOM)
    ]
    return inputs, gates, outputs

def _pin_xs_on_face(component: Component, origin_x: int, side_values: set[str]) -> list[int]:
    xs: list[int] = []
    for p in component.pins:
        effective = p.side.value if p.side is not None else None
        if effective in side_values:
            xs.append(origin_x + p.offset[0])
    return xs

def _first_safe_x(cursor_x: int, forbidden: set[int], pin_x_offsets: list[int]) -> int:
    x = cursor_x
    while True:
        ok = True
        for off in pin_x_offsets:
            px = x + off
            if any((px + d) in forbidden for d in (-1, 0, 1)):
                ok = False
                break
        if ok:
            return x
        x += 1

def _layout_components(components: list[Component], nets: list[NetConnection], *, gutter: int, base_y: int, io_margin: int = 2, routing_gutter: int = 4, x_margin: int = 2) -> _Layout:
    inputs, gates, outputs = _rows_split(components)
    output_row_z = io_margin
    gate_row_z = output_row_z + 1 + routing_gutter
    max_gate_depth = max((c.footprint.depth for c in gates), default=1)
    input_row_z = gate_row_z + max_gate_depth + routing_gutter
    placed: list[_Placed] = []
    gate_south_xs: set[int] = set()
    gate_north_xs: set[int] = set()
    cursor_x = x_margin
    gate_placed_by_id: dict[str, _Placed] = {}
    for g in gates:
        origin = (cursor_x, base_y, gate_row_z)
        gp = _Placed(component=g, origin=origin)
        placed.append(gp)
        gate_placed_by_id[g.id] = gp
        gate_south_xs.update(_pin_xs_on_face(g, cursor_x, {CardinalDirection.SOUTH.value}))
        gate_north_xs.update(_pin_xs_on_face(g, cursor_x, {CardinalDirection.NORTH.value}))
        cursor_x += g.footprint.width + gutter
    def _gate_pin_x(cid: str, pin_name: str) -> int | None:
        gp = gate_placed_by_id.get(cid)
        if gp is None:
            return None
        for p in gp.component.pins:
            if p.name == pin_name:
                return _pin_world(gp.origin, p, gp.component.footprint)[0]
        return None

    paired_x: dict[str, int] = {}
    for net in nets:
        src_x = _gate_pin_x(net.source.component_id, net.source.pin_name)
        for sink in net.sinks:
            sink_x = _gate_pin_x(sink.component_id, sink.pin_name)
            if sink_x is not None and net.source.component_id not in gate_placed_by_id:
                current = paired_x.get(net.source.component_id)
                paired_x[net.source.component_id] = (
                    sink_x if current is None else min(current, sink_x)
                )
            if src_x is not None and sink.component_id not in gate_placed_by_id:
                current = paired_x.get(sink.component_id)
                paired_x[sink.component_id] = (
                    src_x if current is None else min(current, src_x)
                )
    inputs_sorted = sorted(inputs, key=lambda c: paired_x.get(c.id, 10**9))
    cursor_x = x_margin
    for inp in inputs_sorted:
        offs = [p.offset[0] for p in inp.pins]
        safe = _first_safe_x(cursor_x, gate_south_xs, offs)
        placed.append(_Placed(component=inp, origin=(safe, base_y, input_row_z)))
        cursor_x = safe + inp.footprint.width + 1
    outputs_sorted = sorted(outputs, key=lambda c: paired_x.get(c.id, 10**9))
    cursor_x = x_margin
    for out in outputs_sorted:
        offs = [p.offset[0] for p in out.pins]
        safe = _first_safe_x(cursor_x, gate_north_xs, offs)
        placed.append(_Placed(component=out, origin=(safe, base_y, output_row_z)))
        cursor_x = safe + out.footprint.width + 1
    outputs_gutter = (output_row_z + 1, gate_row_z - 1)
    inputs_gutter = (gate_row_z + max_gate_depth, input_row_z - 1)
    return _Layout(
        placed=placed,
        output_row_z=output_row_z,
        gate_row_z=gate_row_z,
        input_row_z=input_row_z,
        max_gate_depth=max_gate_depth,
        base_y=base_y,
        outputs_gutter=outputs_gutter,
        inputs_gutter=inputs_gutter,
    )

def _compute_workspace_dims(
    placed: list[_Placed],
    *,
    base_y: int,
    io_margin: int,
    routing_headroom: int,
) -> tuple[int, int, int]:
    max_x = 1
    max_y = base_y + 1
    max_z = 1
    for item in placed:
        ox, oy, oz = item.origin
        fp = item.component.footprint
        max_x = max(max_x, ox + fp.width)
        max_y = max(max_y, oy + fp.height)
        max_z = max(max_z, oz + fp.depth)
    width = max_x + io_margin
    depth = max_z + io_margin
    height = max_y + routing_headroom + io_margin
    return (width, height, depth)

@dataclass
class _PinInfo:
    component: Component
    origin: tuple[int, int, int]
    pin: PinRef
    pin_world: tuple[int, int, int]
    channel_side: str
    channel_terminal: tuple[int, int, int]
    extension_cells: list[tuple[int, int, int]]
    gutter_entry: tuple[int, int, int]
    gutter: str


def _resolve_pin_info(
    placed: _Placed,
    pin: PinRef,
    *,
    layout: _Layout,
) -> _PinInfo:
    comp = placed.component
    origin = placed.origin
    fp = comp.footprint
    pw = _pin_world(origin, pin, fp)
    is_io = comp.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
    pin_side = _default_io_side(comp, pin)
    if is_io:
        channel_side = _OPPOSITE_SIDE[pin_side.value]
    else:
        channel_side = pin_side.value
    nx, ny, nz = _SIDE_NORMAL[channel_side]
    cx, cy, cz = pw[0] + nx, pw[1] + ny, pw[2] + nz
    while _inside_footprint((cx, cy, cz), origin, fp):
        cx += nx
        cy += ny
        cz += nz
    terminal = (cx, cy, cz)
    if is_io and comp.type == ComponentType.INPUT_PIN:
        gutter = "inputs"
    elif is_io and comp.type == ComponentType.OUTPUT_PIN:
        gutter = "outputs"
    elif channel_side == CardinalDirection.SOUTH.value:
        gutter = "inputs"
    elif channel_side == CardinalDirection.NORTH.value:
        gutter = "outputs"
    else:
        gutter = "inputs"

    extension_cells: list[tuple[int, int, int]] = []
    if channel_side in (CardinalDirection.EAST.value, CardinalDirection.WEST.value):
        entry_x = terminal[0]
        entry_y = terminal[1]
        entry_z = terminal[2]
        bend_target_z = layout.inputs_gutter[0]
        step = 1 if bend_target_z > entry_z else -1
        z = entry_z
        x_walk = pw[0] + nx
        y_walk = pw[1] + ny
        z_walk = pw[2] + nz
        while (x_walk, y_walk, z_walk) != terminal:
            extension_cells.append((x_walk, y_walk, z_walk))
            x_walk += nx
            y_walk += ny
            z_walk += nz
        extension_cells.append(terminal)
        while z != bend_target_z:
            z += step
            extension_cells.append((entry_x, entry_y, z))
        gutter_entry = (entry_x, entry_y, bend_target_z)
        gutter = "inputs"
    elif is_io:
        gutter_entry = pw
    else:
        gutter_entry = terminal
    return _PinInfo(
        component=comp,
        origin=origin,
        pin=pin,
        pin_world=pw,
        channel_side=channel_side,
        channel_terminal=terminal,
        extension_cells=extension_cells,
        gutter_entry=gutter_entry,
        gutter=gutter,
    )

def _claim(dust_owner: dict[tuple[int, int, int], str], cell: tuple[int, int, int], net_id: str) -> None:
    prev = dust_owner.get(cell)
    if prev is not None and prev != net_id:
        raise ValueError(f"Routing collision at {cell}: net '{prev}' vs '{net_id}'")
    dust_owner[cell] = net_id

def _lay_dust_cells(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    cells: Iterable[tuple[int, int, int]],
    net_id: str,
) -> None:
    for cell in cells:
        existing = workspace[cell[0], cell[1], cell[2]]
        if _is_repeater(existing):
            _claim(dust_owner, cell, net_id)
            continue
        if _is_redstone_wire(existing):
            _claim(dust_owner, cell, net_id)
            continue
        if not _is_air(existing):
            raise ValueError(f"Cannot lay dust at {cell}: blocked by {existing}")
        workspace[cell[0], cell[1], cell[2]] = REDSTONE
        _claim(dust_owner, cell, net_id)
        _ensure_support(workspace, solid, cell)

def _direction_label(dx: int, dy: int, dz: int) -> str:
    if dz > 0:
        return CardinalDirection.SOUTH.value
    if dz < 0:
        return CardinalDirection.NORTH.value
    if dx > 0:
        return CardinalDirection.EAST.value
    if dx < 0:
        return CardinalDirection.WEST.value
    raise ValueError("zero delta")

def _insert_repeaters_on_path(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    path: Sequence[tuple[int, int, int]],
    net_id: str,
    flow_facing: str,
    stride: int = REPEATER_STRIDE,
) -> None:
    for i in range(stride, len(path), stride):
        cell = path[i]
        existing = workspace[cell[0], cell[1], cell[2]]
        if _is_repeater(existing):
            continue
        workspace[cell[0], cell[1], cell[2]] = BlockState(
            "minecraft:repeater", facing=flow_facing, delay="1"
        )
        _claim(dust_owner, cell, net_id)
        _ensure_support(workspace, solid, cell)

@dataclass
class _ChannelNet:
    net_id: str
    source_entry: tuple[int, int, int]
    sink_entries: list[tuple[int, int, int]]

def _segment_z(x: int, y: int, z0: int, z1: int) -> list[tuple[int, int, int]]:
    if z0 == z1:
        return [(x, y, z0)]
    step = 1 if z1 > z0 else -1
    return [(x, y, z) for z in range(z0, z1 + step, step)]

def _segment_x(y: int, z: int, x0: int, x1: int) -> list[tuple[int, int, int]]:
    if x0 == x1:
        return [(x0, y, z)]
    step = 1 if x1 > x0 else -1
    return [(x, y, z) for x in range(x0, x1 + step, step)]

def _net_range(cn: _ChannelNet) -> tuple[int, int]:
    xs = [cn.source_entry[0]] + [s[0] for s in cn.sink_entries]
    return (min(xs), max(xs))

def _assign_wire_rows(
    nets: list[_ChannelNet],
    wire_rows_ordered: list[int],
) -> dict[str, int]:
    if not nets:
        return {}
    if len(nets) > len(wire_rows_ordered):
        raise ValueError(
            f"channel has {len(wire_rows_ordered)} wire rows but {len(nets)} nets need routing"
        )
    net_range = {cn.net_id: _net_range(cn) for cn in nets}
    edges: dict[str, set[str]] = {cn.net_id: set() for cn in nets}
    in_degree: dict[str, int] = {cn.net_id: 0 for cn in nets}
    for X in nets:
        x_min, x_max = net_range[X.net_id]
        for Y in nets:
            if X.net_id == Y.net_id:
                continue
            y_src_x = Y.source_entry[0]
            y_sink_xs = [s[0] for s in Y.sink_entries]
            src_in = x_min <= y_src_x <= x_max
            sink_in = any(x_min <= sx <= x_max for sx in y_sink_xs)
            if src_in and sink_in:
                raise ValueError(
                    f"Net '{X.net_id}' range {(x_min, x_max)} contains both source and a sink of "
                    f"net '{Y.net_id}'; dogleg routing would be needed"
                )
            if src_in and not sink_in:
                if X.net_id not in edges[Y.net_id]:
                    edges[Y.net_id].add(X.net_id)
                    in_degree[X.net_id] += 1
            elif sink_in and not src_in:
                if Y.net_id not in edges[X.net_id]:
                    edges[X.net_id].add(Y.net_id)
                    in_degree[Y.net_id] += 1
    queue: list[str] = sorted(nid for nid, d in in_degree.items() if d == 0)
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for succ in sorted(edges[nid]):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    if len(order) != len(nets):
        raise ValueError("Cycle in wire-row ordering; dogleg routing needed")
    return {nid: wire_rows_ordered[i] for i, nid in enumerate(order)}

def _route_channel(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    *,
    channel_y: int,
    wire_rows_ordered: list[int],
    nets: list[_ChannelNet],
) -> None:
    assignment = _assign_wire_rows(nets, wire_rows_ordered)
    for cn in nets:
        wire_z = assignment[cn.net_id]
        _route_channel_net(workspace, solid, dust_owner, channel_y, wire_z, cn)


def _route_channel_net(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    channel_y: int,
    wire_z: int,
    cn: _ChannelNet,
) -> None:
    src = cn.source_entry
    src_x = src[0]
    src_z = src[2]
    src_bridge = _segment_z(src_x, channel_y, src_z, wire_z)
    _lay_dust_cells(workspace, solid, dust_owner, src_bridge, cn.net_id)
    if wire_z != src_z:
        src_flow = _direction_label(0, 0, wire_z - src_z)
        _insert_repeaters_on_path(
            workspace, solid, dust_owner, src_bridge, cn.net_id, src_flow
        )
    xs = [src_x] + [s[0] for s in cn.sink_entries]
    x_min, x_max = min(xs), max(xs)
    wire_cells = _segment_x(channel_y, wire_z, x_min, x_max)
    _lay_dust_cells(workspace, solid, dust_owner, wire_cells, cn.net_id)
    east_half = [c for c in wire_cells if c[0] >= src_x]
    east_half.sort(key=lambda c: c[0])
    _insert_repeaters_on_path(
        workspace, solid, dust_owner, east_half, cn.net_id, CardinalDirection.EAST.value
    )
    west_half = [c for c in wire_cells if c[0] <= src_x]
    west_half.sort(key=lambda c: -c[0])
    _insert_repeaters_on_path(
        workspace, solid, dust_owner, west_half, cn.net_id, CardinalDirection.WEST.value
    )
    for sink in cn.sink_entries:
        sk_x = sink[0]
        sk_z = sink[2]
        sink_bridge = _segment_z(sk_x, channel_y, wire_z, sk_z)
        _lay_dust_cells(workspace, solid, dust_owner, sink_bridge, cn.net_id)
        if sk_z != wire_z:
            sk_flow = _direction_label(0, 0, sk_z - wire_z)
            _insert_repeaters_on_path(
                workspace, solid, dust_owner, sink_bridge, cn.net_id, sk_flow
            )
def _bfs_xz(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    net_id: str,
    bounds: tuple[int, int, int, int, int, int],
) -> list[tuple[int, int, int]]:
    min_x, _, min_z, max_x, _, max_z = bounds
    sy = start[1]
    if start[1] != goal[1]:
        raise ValueError("overpass BFS requires same Y")
    def foreign_adjacent(pos: tuple[int, int, int]) -> bool:
        x, y, z = pos
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                npos = (x + dx, y, z + dz)
                owner = dust_owner.get(npos)
                if owner is not None and owner != net_id:
                    return True
        return False
    def walkable(pos: tuple[int, int, int], is_endpoint: bool) -> bool:
        x, y, z = pos
        if not (min_x <= x <= max_x and min_z <= z <= max_z):
            return False
        if pos in solid and not _is_redstone_wire(workspace[x, y, z]):
            return False
        blk = workspace[x, y, z]
        if not _is_air(blk) and not _is_redstone_wire(blk):
            return False
        if _is_redstone_wire(blk):
            if dust_owner.get(pos) not in (None, net_id):
                return False
        if not is_endpoint and foreign_adjacent(pos):
            return False
        return True
    if not walkable(start, True):
        raise ValueError(f"overpass start {start} not walkable for {net_id}")
    queue: deque[tuple[int, int, int]] = deque([start])
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        x, _, z = cur
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            npos = (x + dx, sy, z + dz)
            if npos in came_from:
                continue
            if not walkable(npos, npos == goal):
                continue
            came_from[npos] = cur
            queue.append(npos)
    if goal not in came_from:
        raise ValueError(
            f"overpass BFS for net '{net_id}' from {start} to {goal} found no path"
        )
    path: list[tuple[int, int, int]] = []
    cur: tuple[int, int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path

def _vertical_column_cells(x: int, z: int, y0: int, y1: int) -> list[tuple[int, int, int]]:
    if y0 == y1:
        return [(x, y0, z)]
    step = 1 if y1 > y0 else -1
    return [(x, y, z) for y in range(y0, y1 + step, step)]

def _route_overpass(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    *,
    overpass_y: int,
    source_entry: tuple[int, int, int],
    sink_entries: list[tuple[int, int, int]],
    net_id: str,
    bounds: tuple[int, int, int, int, int, int],
) -> None:
    rise = _vertical_column_cells(source_entry[0], source_entry[2], source_entry[1], overpass_y)
    _lay_dust_cells(workspace, solid, dust_owner, rise, net_id)
    hub = (source_entry[0], overpass_y, source_entry[2])
    for sink in sink_entries:
        target = (sink[0], overpass_y, sink[2])
        path = _bfs_xz(workspace, solid, dust_owner, hub, target, net_id, bounds)
        _lay_dust_cells(workspace, solid, dust_owner, path, net_id)
        if len(path) > 1:
            for i in range(REPEATER_STRIDE, len(path), REPEATER_STRIDE):
                cell = path[i]
                prev = path[i - 1]
                facing = _direction_label(cell[0] - prev[0], 0, cell[2] - prev[2])
                workspace[cell[0], cell[1], cell[2]] = BlockState(
                    "minecraft:repeater", facing=facing, delay="1"
                )
                _claim(dust_owner, cell, net_id)
                _ensure_support(workspace, solid, cell)
        drop = _vertical_column_cells(sink[0], sink[2], overpass_y, sink[1])
        _lay_dust_cells(workspace, solid, dust_owner, drop, net_id)

def _component_row(c: Component) -> str:
    if c.type == ComponentType.INPUT_PIN:
        return "input"
    if c.type == ComponentType.OUTPUT_PIN:
        return "output"
    return "gate"

def _classify_net(
    net: NetConnection,
    comp_by_id: dict[str, Component],
) -> str:
    src_row = _component_row(comp_by_id[net.source.component_id])
    sink_rows = {_component_row(comp_by_id[s.component_id]) for s in net.sinks}
    if src_row == "input" and sink_rows <= {"gate"}:
        return "inputs"
    if src_row == "gate" and sink_rows <= {"output"}:
        return "outputs"
    if src_row == "gate" and sink_rows <= {"gate"}:
        return "overpass"
    if src_row == "input" and sink_rows <= {"output"}:
        return "overpass"
    return "mixed"

def _load_template_region(schematics_dir: Path, prefix: str) -> Region:
    path = schematics_dir / f"{prefix}.litematic"
    if not path.is_file():
        raise FileNotFoundError(f"Missing schematic template: {path}")
    schematic = Schematic.load(str(path))
    if not schematic.regions:
        raise ValueError(f"Schematic has no regions: {path}")
    return next(iter(schematic.regions.values()))

@dataclass
class _Plan:
    component_list: ComponentList
    layout: _Layout
    workspace: Region
    solid: set[tuple[int, int, int]]
    dust_owner: dict[tuple[int, int, int], str]
    pin_infos: dict[tuple[str, str], _PinInfo]
    io_repeater_cell: dict[tuple[str, str], tuple[int, int, int]]
    io_repeater_facing: dict[tuple[str, str], str]
    mc_version: int
    lm_version: int
    lm_subversion: int

def _plan_placement(
    comp: ComponentList,
    schematics_dir: Path,
    *,
    gutter: int = 8,
    workspace_size: tuple[int, int, int] | None = None,
    base_y: int = 40,
    io_margin: int = 2,
    routing_headroom: int = 4,
    routing_gutter: int | None = None,
) -> _Plan:
    comp = _expand_multibit_io(comp)
    n_inputs = sum(1 for c in comp.components if c.type == ComponentType.INPUT_PIN)
    n_outputs = sum(1 for c in comp.components if c.type == ComponentType.OUTPUT_PIN)
    needed = max(n_inputs, n_outputs)
    auto_gutter = max(4, 2 * needed + 1) if needed else 4
    if routing_gutter is None:
        routing_gutter = auto_gutter
    else:
        routing_gutter = max(routing_gutter, auto_gutter)
    layout = _layout_components(
        comp.components,
        comp.nets,
        gutter=gutter,
        base_y=base_y,
        io_margin=io_margin,
        routing_gutter=routing_gutter,
    )
    if workspace_size is None:
        width, height, depth = _compute_workspace_dims(
            layout.placed,
            base_y=base_y,
            io_margin=io_margin,
            routing_headroom=routing_headroom,
        )
    else:
        width, height, depth = workspace_size
    workspace = Region(0, 0, 0, width, height, depth)
    solid: set[tuple[int, int, int]] = set()
    dust_owner: dict[tuple[int, int, int], str] = {}
    ws_bounds = (0, 0, 0, width - 1, height - 1, depth - 1)
    mc_version = 2975
    lm_version = 6
    lm_sub = 1
    for item in layout.placed:
        c = item.component
        if c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN, ComponentType.CUSTOM):
            continue
        if c.type not in SCHEMATIC_MAP:
            raise ValueError(f"No schematic template registered for {c.type}")
        info = SCHEMATIC_MAP[c.type]
        template = _load_template_region(schematics_dir, info.file_prefix)
        ref = Schematic.load(str(schematics_dir / f"{info.file_prefix}.litematic"))
        mc_version = int(ref.mc_version)
        lm_version = int(ref.lm_version)
        lm_sub = int(ref.lm_subversion)
        _paste_template(workspace, template, item.origin, solid)
    pin_infos: dict[tuple[str, str], _PinInfo] = {}
    for item in layout.placed:
        for pin in item.component.pins:
            info = _resolve_pin_info(item, pin, layout=layout)
            pin_infos[(item.component.id, pin.name)] = info
    io_repeater_cell: dict[tuple[str, str], tuple[int, int, int]] = {}
    io_repeater_facing: dict[tuple[str, str], str] = {}
    for item in layout.placed:
        c = item.component
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
            continue
        for pin in c.pins:
            side = _default_io_side(c, pin)
            pw = _pin_world(item.origin, pin, c.footprint)
            cell = _io_repeater_cell(pw, item.origin, c.footprint, side, bounds=ws_bounds)
            facing = _io_repeater_facing(c.type, side)
            workspace[cell[0], cell[1], cell[2]] = BlockState(
                "minecraft:repeater", facing=facing, delay="1"
            )
            solid.add(cell)
            _ensure_support(workspace, solid, cell)
            io_repeater_cell[(c.id, pin.name)] = cell
            io_repeater_facing[(c.id, pin.name)] = facing

    comp_by_id = {c.id: c for c in comp.components}
    pin_to_net: dict[tuple[str, str], str] = {}
    for net in comp.nets:
        pin_to_net[(net.source.component_id, net.source.pin_name)] = net.net_id
        for sink in net.sinks:
            pin_to_net[(sink.component_id, sink.pin_name)] = net.net_id

    for key, info in pin_infos.items():
        if not info.extension_cells:
            continue
        nid = pin_to_net.get(key)
        if nid is None:
            continue
        _lay_dust_cells(workspace, solid, dust_owner, info.extension_cells, nid)
    inputs_channel: list[_ChannelNet] = []
    outputs_channel: list[_ChannelNet] = []
    overpass_nets: list[tuple[NetConnection, _PinInfo, list[_PinInfo]]] = []
    for net in comp.nets:
        src_info = pin_infos[(net.source.component_id, net.source.pin_name)]
        sink_infos = [
            pin_infos[(s.component_id, s.pin_name)] for s in net.sinks
        ]
        kind = _classify_net(net, comp_by_id)
        if kind == "inputs":
            inputs_channel.append(
                _ChannelNet(
                    net_id=net.net_id,
                    source_entry=src_info.gutter_entry,
                    sink_entries=[s.gutter_entry for s in sink_infos],
                )
            )
        elif kind == "outputs":
            outputs_channel.append(
                _ChannelNet(
                    net_id=net.net_id,
                    source_entry=src_info.gutter_entry,
                    sink_entries=[s.gutter_entry for s in sink_infos],
                )
            )
        elif kind == "overpass":
            overpass_nets.append((net, src_info, sink_infos))
        else:
            raise NotImplementedError(
                f"Mixed-gutter nets not yet supported: net {net.net_id}"
            )
    i_gmin, i_gmax = layout.inputs_gutter
    i_wire_rows = list(range(i_gmax, i_gmin - 1, -2))
    _route_channel(
        workspace,
        solid,
        dust_owner,
        channel_y=base_y,
        wire_rows_ordered=i_wire_rows,
        nets=inputs_channel,
    )
    o_gmin, o_gmax = layout.outputs_gutter
    o_wire_rows = list(range(o_gmax, o_gmin - 1, -2))
    _route_channel(
        workspace,
        solid,
        dust_owner,
        channel_y=base_y,
        wire_rows_ordered=o_wire_rows,
        nets=outputs_channel,
    )
    overpass_y = base_y + layout.max_gate_depth + 2
    overpass_y = min(overpass_y, height - 2)
    for net, src_info, sink_infos in overpass_nets:
        _route_overpass(
            workspace,
            solid,
            dust_owner,
            overpass_y=overpass_y,
            source_entry=src_info.gutter_entry,
            sink_entries=[s.gutter_entry for s in sink_infos],
            net_id=net.net_id,
            bounds=ws_bounds,
        )
    return _Plan(
        component_list=comp,
        layout=layout,
        workspace=workspace,
        solid=solid,
        dust_owner=dust_owner,
        pin_infos=pin_infos,
        io_repeater_cell=io_repeater_cell,
        io_repeater_facing=io_repeater_facing,
        mc_version=mc_version,
        lm_version=lm_version,
        lm_subversion=lm_sub,
    )


def build_litematic_from_component_list(
    comp: ComponentList,
    schematics_dir: Path,
    out_path: Path,
    *,
    gutter: int = 8,
    workspace_size: tuple[int, int, int] | None = None,
    base_y: int = 40,
    schematic_name: str = "build",
    io_margin: int = 2,
    routing_headroom: int = 4,
    routing_gutter: int = 4,
) -> Path:
    plan = _plan_placement(
        comp,
        schematics_dir=schematics_dir,
        gutter=gutter,
        workspace_size=workspace_size,
        base_y=base_y,
        io_margin=io_margin,
        routing_headroom=routing_headroom,
        routing_gutter=routing_gutter,
    )
    schematic = plan.workspace.as_schematic(
        name=schematic_name, author="minecraft-v", description="merged placement"
    )
    schematic.mc_version = plan.mc_version
    schematic.lm_version = plan.lm_version
    schematic.lm_subversion = plan.lm_subversion
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schematic.save(str(out_path))
    return out_path
