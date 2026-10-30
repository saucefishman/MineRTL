from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable
from math import ceil
from pathlib import Path

from litemapy import BlockState, Region, Schematic

from minecraft_v.build_utils import save_artifact, save_schematic_artifact, DEFAULT_SNAPSHOT_DIR
from minecraft_v.cell_library import SCHEMATIC_MAP
from minecraft_v.progress import ProgressBar, log
from minecraft_v.placement_engine.ir import (
    Component,
    ComponentList,
    ComponentType,
    NetConnection,
)
from .block_utils import _is_air, _ensure_support, _is_redstone_wire, _is_repeater, _is_torch, _block_str, _is_glass
from .constants import (
    _HORIZ_DIRS, _DIRS_6, _SIDE_NORMAL, _OPPOSITE_SIDE, WOOLS, REDSTONE, STONE, _REPEATER_MAX_DELAY,
    _ROUTE_MAX_PASSES, _PRES_SCHEDULE, _CONFLICT_BASE, _ROUTE_STAGNATION_PASSES, _ROUTE_SEARCH_TIMEOUT,
    _ROUTE_TIMEOUT_ESCALATION, _ROUTE_FINAL_SEARCH_TIMEOUT, _ROUTE_BATCH_PER_WORKER,
)
from .layout import (
    _Placed, _expand_multibit_io, _assign_component_y_levels,
    _layout_components, _compute_workspace_dims,
)
from .anneal import _optimize_placement
from .negotiate import Congestion, OverlayDict, OverlaySet, NetRoute, OverlayWorkspace
from .parallel import RouteWorkers, claims_of
from .pathfinding import IndexedWorkspace, RouteTimeout, _BlockIndex, _find_wire_path, _signal_depths
from .ripup import Recorder, JournaledSet, JournaledDict, JournalingWorkspace, rip_up, rollback
from .template import (
    _pin_world, _io_repeater_cell, _io_repeater_facing,
    _default_io_side, _paste_template, _load_template_region,
)
from .cosmetics import apply_dust_shapes, place_io_fixtures, place_io_signs
from .verify import verify_signals, verify_support
from .wire import LayConflict, _lay_redstone_path, _place_repeaters_for_net

REDSTONE_BLOCK = BlockState("minecraft:redstone_block")
IRON_TRAPDOOR = BlockState("minecraft:iron_trapdoor", open="false", half="bottom", facing="north")
BARREL = BlockState("minecraft:barrel")


def _pin_for_endpoint(
        pin_world: dict[tuple[str, str], tuple[int, int, int]],
        component_id: str,
        pin_name: str,
) -> tuple[int, int, int]:
    key = (component_id, pin_name)
    if key in pin_world:
        return pin_world[key]
    raise ValueError(f"Unknown pin endpoint {component_id}.{pin_name}")


def _place_gate_templates(
        workspace: Region,
        placed: list[_Placed],
        solid: set[tuple[int, int, int]],
        schematics_dir: Path,
        dust_owner: dict[tuple[int, int, int], str],
) -> tuple[int, int, int]:
    """Paste component schematics into workspace. Returns (mc_version, lm_version, lm_subversion)."""
    ref_mc_version = 2975
    ref_lm_version = 6
    ref_lm_sub = 1
    for item in placed:
        c = item.component
        if c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN, ComponentType.CUSTOM):
            continue
        if c.type not in SCHEMATIC_MAP:
            raise ValueError(f"No schematic template for {c.type}")
        info = SCHEMATIC_MAP[c.type]
        template = _load_template_region(schematics_dir, info.file_prefix)
        ref = Schematic.load(str(schematics_dir / f"{info.file_prefix}.litematic"))
        ref_mc_version = int(ref.mc_version)
        ref_lm_version = int(ref.lm_version)
        ref_lm_sub = int(ref.lm_subversion)
        ox, oy, oz = item.origin
        pin_positions = frozenset(
            (ox + p.offset[0],
             oy + p.offset[1],
             oz + (c.footprint.depth - 1 - p.offset[2]))
            for p in c.pins
        )
        _paste_template(workspace, template, item.origin, solid,
                        dust_owner, f"__schematic__{c.id}", pin_positions,
                        mirror_width=c.footprint.width if c.metadata.get("mirror_x") else None)
    return (ref_mc_version, ref_lm_version, ref_lm_sub)


def _place_const_sources(
        workspace: Region,
        placed: list[_Placed],
        solid: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    """Place redstone blocks (const=1) and iron trapdoors (const=0) for const pins."""
    const_positions: set[tuple[int, int, int]] = set()
    for item in placed:
        c = item.component
        if c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
            continue
        for pin in c.pins:
            if pin.const_value not in ('0', '1', 'x', 'z'):
                continue
            pw = _pin_world(item.origin, pin, c.footprint)
            side = pin.side
            if side is not None:
                nx, ny, nz = _SIDE_NORMAL[side.value]
                bx, by, bz = pw[0] + nx, pw[1] + ny, pw[2] + nz
            else:
                bx, by, bz = pw
            if _is_air(workspace[bx, by, bz]):
                if pin.const_value == '1':
                    block = REDSTONE_BLOCK
                elif pin.const_value == '0':
                    block = IRON_TRAPDOOR
                else:
                    block = BARREL
                workspace[bx, by, bz] = block
                solid.add((bx, by, bz))
                if pin.const_value == '1':
                    const_positions.add((bx, by, bz))
    return const_positions


def _compute_max_bridge_y(
        placed: list[_Placed],
        base_y: int,
        bridge_height: int,
) -> int:
    max_comp_top = base_y
    for item in placed:
        top = item.origin[1] + item.component.footprint.height - 1
        max_comp_top = max(max_comp_top, top)
    return max_comp_top + bridge_height


def _compute_pin_maps(
        placed: list[_Placed],
) -> tuple[dict[tuple[str, str], tuple[int, int, int]], dict[tuple[str, str], tuple[int, int, int]],
           dict[tuple[int, int, int], tuple[int, int]]]:
    """Return (pin_world_map, pin_terminal, terminal_approach) for all component pins.

    terminal_approach: gate-pin terminal -> horizontal direction wire leaves it in (the pin's side normal).
    """
    pin_world_map: dict[tuple[str, str], tuple[int, int, int]] = {}
    pin_terminal: dict[tuple[str, str], tuple[int, int, int]] = {}
    terminal_approach: dict[tuple[int, int, int], tuple[int, int]] = {}
    for item in placed:
        c = item.component
        origin = item.origin
        is_io = c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
        for pin in c.pins:
            pw = _pin_world(origin, pin, c.footprint)
            pin_world_map[(c.id, pin.name)] = pw
            if is_io:
                pin_terminal[(c.id, pin.name)] = pw
            else:
                side = pin.side
                if side is None:
                    pin_terminal[(c.id, pin.name)] = pw
                else:
                    nx, ny, nz = _SIDE_NORMAL[side.value]
                    terminal = (pw[0] + nx, pw[1] + ny, pw[2] + nz)
                    pin_terminal[(c.id, pin.name)] = terminal
                    if ny == 0:
                        terminal_approach[terminal] = (nx, nz)
    return pin_world_map, pin_terminal, terminal_approach


def _place_io_repeaters(
        workspace: Region,
        placed: list[_Placed],
        solid: set[tuple[int, int, int]],
        pin_world_map: dict[tuple[str, str], tuple[int, int, int]],
        ws_bounds: tuple[int, int, int, int, int, int],
        clock_pin: str | None = None,
) -> tuple[set[tuple[int, int, int]], dict[str, tuple[tuple[int, int, int], str, tuple[str, str]]],
           list[tuple[tuple[int, int, int], tuple[int, int], str]],
           list[tuple[tuple[int, int, int], tuple[int, int], bool]]]:
    """Place I/O repeaters and return (all repeater cells, output target lookup, sign specs, fixture specs).

    The clock input (`clock_pin`, or a pin named `clk`) gets no lever: it is driven by a clock.
    """
    io_repeater_cells: set[tuple[int, int, int]] = set()
    io_signs: list[tuple[tuple[int, int, int], tuple[int, int], str]] = []
    io_fixtures: list[tuple[tuple[int, int, int], tuple[int, int], bool]] = []
    output_repeater_lookup: dict[str, tuple[tuple[int, int, int], str, tuple[str, str]]] = {}
    for item in placed:
        c = item.component
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
            continue
        origin = item.origin
        for pin in c.pins:
            side = _default_io_side(c, pin)
            pw = pin_world_map[(c.id, pin.name)]
            cell = _io_repeater_cell(pw, origin, c.footprint, side, bounds=ws_bounds)
            facing = _io_repeater_facing(c.type, side)
            workspace[cell[0], cell[1], cell[2]] = BlockState(
                "minecraft:repeater", facing=facing, delay="1",
            )
            # Not added to `solid`: a repeater can't hold dust, and `solid` doubles as "can support".
            _ensure_support(workspace, solid, cell)
            io_repeater_cells.add(cell)
            nx, _, nz = _SIDE_NORMAL[side.value]
            io_signs.append((cell, (nx, nz), pin.name))
            is_input = c.type == ComponentType.INPUT_PIN
            if not (is_input and c.id in (clock_pin, 'clk')):
                io_fixtures.append((cell, (nx, nz), is_input))
            if c.type == ComponentType.OUTPUT_PIN:
                side_name = side.value
                endpoint = (c.id, pin.name)
                for key in (c.id, pin.name, f"{c.id}.{pin.name}"):
                    output_repeater_lookup[key] = (cell, side_name, endpoint)
    return io_repeater_cells, output_repeater_lookup, io_signs, io_fixtures


_EXT_TUBE_DEPTH = 3  # cells of straight approach lane kept clear in front of each output pin target


def _route_output_pin_extensions(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        output_pin_targets: dict[str, tuple[int, int]],
        output_repeater_lookup: dict[str, tuple[tuple[int, int, int], str, tuple[str, str]]],
        net_id_by_output_endpoint: dict[tuple[str, str], str],
        ext_bounds: tuple[int, int, int, int, int, int],
        layer_z: int,
        max_bridge_y: int,
        footprint_blocked: frozenset[tuple[int, int, int]] = frozenset(),
        all_terminals: dict[str, set[tuple[int, int, int]]] | None = None,
        inverted_cells: JournaledSet | None = None,
        rec: Recorder | None = None,
        net_paths: dict[str, tuple[str, list[list[tuple[int, int, int]]]]] | None = None,
) -> list[tuple[str, Exception]]:
    """Route each output pin's net on to its target in the z=layer_z output layer.

    One _GreedyRouter job per pin (journal key "<net>><pin>"), branching off the net's
    existing wire. net_paths: the core router's paths, so repeater placement counts signal
    from the last repeater upstream. Returns (job key, error) for failed extensions.
    """
    rec = Recorder() if rec is None else rec
    if inverted_cells is None:
        inverted_cells = JournaledSet(rec, name="inverted")
    min_x, min_y, min_z, max_x, max_y, max_z = ext_bounds

    pin_target_x_offset = (max_x - min_x) // 2

    def _in_bounds(pos: tuple[int, int, int]) -> bool:
        x, y, z = pos
        return min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z

    # Validate all targets upfront and build target position map for cross-pin protection.
    ext_target_positions: dict[str, tuple[int, int, int]] = {}
    repeater_cells_by_pin: dict[str, tuple[int, int, int]] = {}
    for pin_name, (tx, ty) in output_pin_targets.items():
        if output_repeater_lookup.get(pin_name) is None:
            known = ", ".join(sorted(output_repeater_lookup))
            raise ValueError(f"Unknown output pin target '{pin_name}'. Known keys: {known}")
        etx = tx + pin_target_x_offset
        if not (min_x <= etx <= max_x and min_y <= ty <= max_y):
            raise ValueError(
                f"Output target for '{pin_name}' is out of bounds: ({etx}, {ty}) not in "
                f"x=[{min_x},{max_x}], y=[{min_y},{max_y}]"
            )
        ext_target_positions[pin_name] = (etx, ty, layer_z)
        repeater_cells_by_pin[pin_name] = output_repeater_lookup[pin_name][0]
    if not (min_z <= layer_z <= max_z):
        raise ValueError(f"Output layer z={layer_z} outside workspace z=[{min_z},{max_z}]")

    all_ext_targets = frozenset(ext_target_positions.values())

    def _repeater_protection_zone(rep: tuple[int, int, int]) -> frozenset[tuple[int, int, int]]:
        """Cardinal+diagonal keepout around a single repeater cell — same geometry as _compute_net_protected."""
        rx, ry, rz = rep
        zone: set[tuple[int, int, int]] = {rep, (rx, ry + 1, rz)}
        for dx, dz in _HORIZ_DIRS:
            for dy in range(-2, 3):
                zone.add((rx + dx * 1, ry + dy, rz + dz * 1))
        for dx, dz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            zone.add((rx + dx, ry, rz + dz))
        return frozenset(zone)

    all_terminal_positions = frozenset(
        pos for ts in (all_terminals or {}).values() for pos in ts) | all_ext_targets
    # Nearest-in-height first: a wire to a far target passes the nearer targets' heights, so
    # it goes around their (already laid) approaches instead of the nearer ones being walled
    # in by it (video: sources above the target grid, low bits descended across high bits').
    def _ext_height_gap(pn: str) -> int:
        return abs(ext_target_positions[pn][1] - repeater_cells_by_pin[pn][1])

    net_by_pin: dict[str, str] = {}
    for pin_name in output_pin_targets:
        base_net_id = net_id_by_output_endpoint.get(output_repeater_lookup[pin_name][2])
        if base_net_id is None:
            raise ValueError(f"Output target '{pin_name}' has no mapped net")
        net_by_pin[pin_name] = base_net_id
    # A pin's keepout: every other net's terminal keepout (_compute_net_protected) plus the
    # zones around every other pin's repeater and target. Counted per cell with one owner per
    # net (its terminals plus its pins' zones) so each pin gets a lazy view, not a copy.
    def _target_tube(target: tuple[int, int, int]) -> set[tuple[int, int, int]]:
        """Straight approach lane in front of a target (+z), so wires don't sweep across it."""
        tx, ty, tz = target
        return {(tx + dx, ty + dy, tz + k) for k in range(1, _EXT_TUBE_DEPTH + 1)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)}

    pin_zone = {pn: _repeater_protection_zone(repeater_cells_by_pin[pn])
                | _repeater_protection_zone(ext_target_positions[pn])
                | _target_tube(ext_target_positions[pn]) for pn in output_pin_targets}
    owner_cells: dict[str, set[tuple[int, int, int]]] = {
        net_id: _terminal_keepout(ts) for net_id, ts in (all_terminals or {}).items()}
    for pn, net_id in net_by_pin.items():
        owner_cells.setdefault(net_id, set()).update(pin_zone[pn])
    owner_count = _count_cells(owner_cells)
    jobs: list[_RouteJob] = []
    for pin_name in sorted(output_pin_targets, key=_ext_height_gap):
        repeater_cell, side_name, endpoint = output_repeater_lookup[pin_name]
        base_net_id = net_by_pin[pin_name]
        nx, ny, nz = _SIDE_NORMAL[side_name]
        launch = (repeater_cell[0] + nx, repeater_cell[1] + ny, repeater_cell[2] + nz)
        if not _in_bounds(launch):
            # Output repeaters can sit on workspace boundary; route from the in-bounds side in that case.
            launch = (repeater_cell[0] - nx, repeater_cell[1] - ny, repeater_cell[2] - nz)
        if not _in_bounds(launch):
            raise ValueError(f"No in-bounds launch cell for output target '{pin_name}'")
        target = ext_target_positions[pin_name]
        # Other pins of this same net are foreign terminals too.
        same_net = frozenset(c for pn, net_id in net_by_pin.items()
                             if net_id == base_net_id and pn != pin_name for c in pin_zone[pn])
        protected = _ForeignKeepout(owner_count, owner_cells[base_net_id], extra=same_net)
        jobs.append(_RouteJob(
            key=f"{base_net_id}>{pin_name}", net_id=base_net_id,
            src=launch, sinks=[target], bounds=ext_bounds,
            protected=protected, terminals={launch, target},
            branch_from_net=True,
            search_kwargs={"limit_exclusions": True, "x_h_factor": 2},
        ))

    router = _GreedyRouter(workspace, rec, solid, dust_owner, inverted_cells, footprint_blocked, max_bridge_y,
                           all_terminal_positions, paths=net_paths)
    # Targets sit on a tight grid in one layer: ripping neighbours there cascades (runner SA
    # seed 3: 5 rip rounds, 44s; video: 5 -> 7 failed even ripping just the jobs crossing a
    # failed target's approach column) without routing more pins, so that layer is off here.
    failures = router.run(jobs, "[ext]", rip_blockers=False)
    print(f"[ext] done ({len(jobs)} extensions, {len(failures)} failed)")
    return failures


def _component_interior(placed: list[_Placed]) -> set[tuple[int, int, int]]:
    """Cells inside non-IO component footprints."""
    return {
        (ox + dx, oy + dy, oz + dz)
        for item in placed
        if item.component.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
        for ox, oy, oz in [item.origin]
        for fp in [item.component.footprint]
        for dx in range(fp.width)
        for dy in range(fp.height)
        for dz in range(fp.depth)
    }


def _build_footprint_blocked(
        placed: list[_Placed],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        io_repeater_cells: set[tuple[int, int, int]],
        const_positions: set[tuple[int, int, int]],
        io_fixtures: list[tuple[tuple[int, int, int], tuple[int, int], bool]] = (),
        block_output_lamps: bool = True,
) -> frozenset[tuple[int, int, int]]:
    """Hard-block zone: footprint interiors + expanded borders + IO repeater tops + const air-gaps
    + the lever/lamp cells (and their neighbours) behind input / in front of output repeaters."""
    fp_interior = _component_interior(placed)
    fp_expanded: set[tuple[int, int, int]] = set(fp_interior)
    for cx, cy, cz in fp_interior:
        for ddx, ddy, ddz in _DIRS_6:
            fp_expanded.add((cx + ddx, cy + ddy, cz + ddz))
        fp_expanded.add((cx, cy + 2, cz))  # extra upward block: support placed below wire
        # Vertical diagonals: signal coupling and support-block placement reach footprint through these
        for ddx, ddz in _HORIZ_DIRS:
            fp_expanded.add((cx + ddx, cy + 1, cz + ddz))
            fp_expanded.add((cx + ddx, cy - 1, cz + ddz))
    for rx, ry, rz in io_repeater_cells:
        fp_expanded.add((rx, ry + 1, rz))
    for (rx, ry, rz), (nx, nz), is_input in io_fixtures:
        if not is_input and not block_output_lamps:
            continue  # output extensions start in front of the repeater
        fixture = [(rx + nx, ry, rz + nz)] + ([(rx + nx, ry + 1, rz + nz)] if is_input else [])
        for fx, fy, fz in fixture:
            fp_expanded.add((fx, fy, fz))
            for ddx, ddy, ddz in _DIRS_6:
                fp_expanded.add((fx + ddx, fy + ddy, fz + ddz))
    # Air-gap const redstone blocks: block all 6 immediate neighbors
    for rbx, rby, rbz in const_positions:
        for ddx, ddy, ddz in _DIRS_6:
            fp_expanded.add((rbx + ddx, rby + ddy, rbz + ddz))
    fp_expanded -= set(pin_terminal.values())
    return frozenset(fp_expanded)


def _build_terminal_map(
        work_nets: list[NetConnection],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
) -> dict[str, set[tuple[int, int, int]]]:
    all_terminals: dict[str, set[tuple[int, int, int]]] = {}
    for net in work_nets:
        t: set[tuple[int, int, int]] = set()
        t.add(_pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name))
        for sink in net.sinks:
            t.add(_pin_for_endpoint(pin_terminal, sink.component_id, sink.pin_name))
        all_terminals[net.net_id] = t
    return all_terminals


def _compute_net_protected(
        net_id: str,
        all_terminals: dict[str, set[tuple[int, int, int]]],
) -> frozenset[tuple[int, int, int]]:
    """Cells (terminal + 1-cell horiz buffer) of every other net — hard-blocked for routing."""
    cells: set[tuple[int, int, int]] = set()
    for other_id, terminals in all_terminals.items():
        if other_id != net_id:
            cells |= _terminal_keepout(terminals)
    return frozenset(cells)


def _terminal_keepout(terminals: set[tuple[int, int, int]]) -> set[tuple[int, int, int]]:
    """One net's terminal keepout cells (what _compute_net_protected unions over other nets)."""
    cells: set[tuple[int, int, int]] = set()
    for tx, ty, tz in terminals:
        cells.add((tx, ty, tz))
        cells.add((tx, ty + 1, tz))
        for dx, dz in _HORIZ_DIRS:
            for dy in range(-2, 3):
                cells.add((tx + dx, ty + dy, tz + dz))
        for dz in range(3):
            for dy in range(-1, 1):
                cells.add((tx, ty + dy, tz + dz))
        for dx, dz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            for dy in range(-2, 2):
                cells.add((tx + dx, ty, tz + dz))
    return cells


_APPROACH_CORRIDOR = 2  # cells reserved straight out from each gate pin's terminal


def _terminal_halo(
        terminals: set[tuple[int, int, int]],
        approach: dict[tuple[int, int, int], tuple[int, int]],
) -> set[tuple[int, int, int]]:
    """Cells where foreign wire makes a terminal, or its straight approach corridor,
    unwalkable (_wire_walkable checks 2 and 3)."""
    cells: set[tuple[int, int, int]] = set()
    for t in terminals:
        tx, ty, tz = t
        adx, adz = approach.get(t, (0, 0))
        n = _APPROACH_CORRIDOR if t in approach else 0
        for k in range(n + 1):
            cx, cz = tx + k * adx, tz + k * adz
            if k:
                cells.add((cx, ty, cz))
            cells.add((cx, ty + 1, cz))
            cells.add((cx, ty - 1, cz))
            for dx, dz in _HORIZ_DIRS:
                for dy in (-1, 0, 1):
                    cells.add((cx + dx, ty + dy, cz + dz))
    return cells


def _count_cells(per_net: dict[str, set[tuple[int, int, int]]]) -> dict[tuple[int, int, int], int]:
    count: dict[tuple[int, int, int], int] = {}
    for cells in per_net.values():
        for pos in cells:
            count[pos] = count.get(pos, 0) + 1
    return count


class _ForeignKeepout:
    """_compute_net_protected(net_id, ...) as a lazy view over a per-cell net count.

    Materializing every other net's keepout per net is quadratic in net count; this answers
    `pos in protected` (and `protected - cells`, as _find_wire_path uses it) without copying.
    Subtraction (goal relief) never frees a cell in another net's terminal halo: wire there
    would make that terminal unreachable, e.g. a slope into a goal two cells from another pin.
    """
    __slots__ = ("_count", "_own", "_excluded", "_extra", "_halo", "_own_halo")

    def __init__(self, count: dict[tuple[int, int, int], int], own: set[tuple[int, int, int]],
                 excluded: frozenset[tuple[int, int, int]] = frozenset(),
                 extra: frozenset[tuple[int, int, int]] = frozenset(),
                 halo: dict[tuple[int, int, int], int] | None = None,
                 own_halo: set[tuple[int, int, int]] = frozenset()):
        self._count = count
        self._own = own
        self._excluded = excluded
        self._extra = extra  # always protected, regardless of excluded (see with_extra)
        self._halo = {} if halo is None else halo
        self._own_halo = own_halo

    def __contains__(self, pos) -> bool:
        if pos in self._extra:
            return True
        n = self._count.get(pos, 0)
        if n == 0 or (n == 1 and pos in self._own):
            return False
        if pos not in self._excluded:
            return True
        h = self._halo.get(pos, 0)
        return h > 1 or (h == 1 and pos not in self._own_halo)

    def _copy(self, excluded, extra) -> _ForeignKeepout:
        return _ForeignKeepout(self._count, self._own, excluded, extra, self._halo, self._own_halo)

    def __sub__(self, other) -> _ForeignKeepout:
        return self._copy(self._excluded | frozenset(other), self._extra)

    def with_extra(self, cells) -> _ForeignKeepout:
        """Also protect cells, surviving later subtraction (goal relief)."""
        return self._copy(self._excluded, self._extra | frozenset(cells))

    def __iter__(self):
        return (pos for pos in self._count if pos in self)


class _KeepoutMap:
    """Per-net terminal keepouts, counted per cell (see _ForeignKeepout)."""

    def __init__(self, all_terminals: dict[str, set[tuple[int, int, int]]],
                 terminal_approach: dict[tuple[int, int, int], tuple[int, int]] | None = None):
        approach = terminal_approach or {}
        self._own_halo = {net_id: _terminal_halo(t, approach) for net_id, t in all_terminals.items()}
        self._halo = _count_cells(self._own_halo)
        self._own = {net_id: _terminal_keepout(t) | self._own_halo[net_id] for net_id, t in all_terminals.items()}
        self._count = _count_cells(self._own)

    def for_net(self, net_id: str) -> _ForeignKeepout:
        return _ForeignKeepout(self._count, self._own.get(net_id, set()),
                               halo=self._halo, own_halo=self._own_halo.get(net_id, set()))


def _net_sort_key(
        net: NetConnection,
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
) -> int:
    src = _pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name)
    return sum(
        abs(src[0] - d[0]) + abs(src[1] - d[1]) + abs(src[2] - d[2])
        for sink in net.sinks
        for d in [_pin_for_endpoint(pin_terminal, sink.component_id, sink.pin_name)]
    )


_LAY_RETRIES = 4  # per sink: re-searches after a path that couldn't be laid (LayConflict)
_NET_RETRIES = 2  # per net: re-routes after no repeater fits a stretch
_BLOCKER_RADIUS = 3  # nets with cells this close to a failed sink get ripped up and re-queued
_BLOCKER_MAX = 4
_RIP_ROUNDS = 3  # per job: rip-and-retry rounds, each for a sink that hasn't had one yet

Pos = tuple[int, int, int]


@dataclass
class _RouteJob:
    """One greedy routing unit: a net, or one output-pin extension of a net."""
    key: str                       # journal / queue identity
    net_id: str                    # owner of the laid cells
    src: Pos
    sinks: list[Pos]
    bounds: tuple[int, int, int, int, int, int]
    protected: object              # _ForeignKeepout or a set
    terminals: set[Pos]            # kept clear when blocking a stretch with no repeater slot
    support_block: BlockState = STONE
    branch_from_net: bool = False  # first sink may also branch off the net's existing wire
    search_kwargs: dict = field(default_factory=dict)


class _GreedyRouter:
    """Greedy, journaled routing of _RouteJobs with the retry layers that make it robust:

    - per sink: a path the search accepted can still fail to lay when it crosses its own
      earlier cells/supports outside the search's local snapshot (LayConflict). Undo that
      sink's partial path and search again without the conflicting cell.
    - per job: no repeater slot on a stretch (all slopes/turns): undo the job and route it
      again with that stretch blocked, keeping clear of the job's terminals.
    - per run: a sink search that fails is usually walled in by jobs routed earlier. Rip the
      jobs around the failed sink, route it again, then re-queue them. Up to _RIP_ROUNDS
      times per job, once per failed sink (a repeat failure at the same sink is final).

    A job that still fails is ripped up so its partial wire / support / repeaters don't
    block later jobs or show up as debris.
    """

    def __init__(self, region: Region, rec: Recorder, solid: JournaledSet, dust_owner: JournaledDict,
                 inverted_cells: JournaledSet, footprint_blocked: frozenset[Pos], max_bridge_y: int,
                 terminal_positions: frozenset[Pos],
                 paths: dict[str, tuple[str, list[list[Pos]]]] | None = None):
        self.region = region
        self.rec = rec
        self.solid = solid
        self.dust_owner = dust_owner
        self.inverted_cells = inverted_cells
        self.footprint_blocked = footprint_blocked
        self.max_bridge_y = max_bridge_y
        self.terminal_positions = terminal_positions
        self.indexed = IndexedWorkspace(region)  # all writes go through this, keeping its index current
        self.ws = JournalingWorkspace(self.indexed, rec)
        # job key -> (net id, ordered sink paths): what _place_repeaters_for_net walks.
        self.paths: dict[str, tuple[str, list[list[Pos]]]] = {} if paths is None else paths
        self.failed_goal: dict[str, Pos] = {}  # job key -> sink whose search/laying raised

    def net_paths(self, net_id: str) -> list[list[Pos]]:
        return [p for nid, ps in self.paths.values() if nid == net_id for p in ps]

    def rip(self, key: str) -> None:
        rip_up(self.rec, key, self.indexed, self.solid, self.dust_owner)
        self.paths.pop(key, None)

    def route_job(self, job: _RouteJob, job_blocked: frozenset[Pos]) -> None:
        rec, ws = self.rec, self.ws
        paths: list[list[Pos]] = []
        net_seeds = [pos for pos, owner in self.dust_owner.items()
                     if owner == job.net_id and _is_redstone_wire(self.region[pos])] if job.branch_from_net else []
        sinks = sorted(job.sinks, key=lambda p: abs(p[0] - job.src[0]) + abs(p[1] - job.src[1]) + abs(p[2] - job.src[2]))
        for dst in sinks:
            tree_seeds = net_seeds + [pos for path in paths for pos in path]
            seed_depths = _signal_depths((self.net_paths(job.net_id) if net_seeds else []) + paths)
            self.failed_goal[job.key] = dst  # cleared when the sink is laid
            blocked: set[Pos] = set(job_blocked - {job.src, dst})
            keep = {job.src, dst, *tree_seeds}
            while True:
                mark = len(rec.journals[job.key])
                try:
                    path = _find_wire_path(
                        ws, self.solid, self.dust_owner,
                        job.src, dst, job.net_id,
                        job.bounds, self.max_bridge_y,
                        tree_seeds=tree_seeds,
                        protected=job.protected,
                        footprint_blocked=self.footprint_blocked,
                        inverted_cells=frozenset(self.inverted_cells),
                        terminal_positions=self.terminal_positions,
                        blocked=frozenset(blocked),
                        seed_depths=seed_depths,
                        **job.search_kwargs,
                    )
                    _lay_redstone_path(ws, self.solid, self.dust_owner, path, job.net_id,
                                       opaque_support_block=job.support_block,
                                       inverted_cells=self.inverted_cells,
                                       goal=dst,
                                       terminal_positions=self.terminal_positions)
                    break
                except LayConflict as e:
                    cell = next((c for c in e.cells if c not in keep and c not in blocked), None)
                    if cell is None or len(blocked) >= len(job_blocked) + _LAY_RETRIES:
                        raise
                    rollback(rec, job.key, mark)
                    blocked.add(cell)
            self.failed_goal.pop(job.key, None)
            paths.append(path)
        # The net's earlier paths first, so a branch starting mid-net counts signal from the
        # last repeater upstream rather than from where it joins.
        _place_repeaters_for_net(ws, self.dust_owner, job.net_id, self.net_paths(job.net_id) + paths)
        self.paths[job.key] = (job.net_id, paths)

    def route_job_retrying(self, job: _RouteJob) -> None:
        near_terminal = {(x + dx, y + dy, z + dz) for x, y, z in job.terminals
                         for dx, dy, dz in ((0, 0, 0), *_DIRS_6)}
        job_blocked: frozenset[Pos] = frozenset()
        for attempt in range(_NET_RETRIES + 1):
            mark = len(self.rec.journals[job.key])
            try:
                self.route_job(job, job_blocked)
                return
            except LayConflict as e:
                stretch = frozenset(e.cells) - near_terminal - job_blocked
                if attempt == _NET_RETRIES or not stretch or "cannot place repeater" not in str(e):
                    raise
                rollback(self.rec, job.key, mark)
                job_blocked |= stretch

    def blockers(self, goal: Pos, key: str, keys: set[str]) -> list[str]:
        """Jobs (of keys) with wire or blocks within _BLOCKER_RADIUS of goal, nearest first."""
        gx, gy, gz = goal
        r = _BLOCKER_RADIUS
        dist: dict[str, int] = {}
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    pos = (gx + dx, gy + dy, gz + dz)
                    d = abs(dx) + abs(dy) + abs(dz)
                    for owner in (self.dust_owner.get(pos), self.rec.writer.get(pos)):
                        if owner is not None and owner != key and owner in keys:
                            dist[owner] = min(dist.get(owner, d), d)
        return sorted(dist, key=lambda n: dist[n])

    def run(self, jobs: list[_RouteJob], label: str, deadline: float | None = None, fail_fast: bool = False,
            on_done: Callable[[int, _RouteJob], None] | None = None,
            rip_blockers: bool = True) -> list[tuple[str, Exception]]:
        """Route jobs in order; returns (job key, error) for the ones that failed.

        fail_fast: raise on the first job that fails instead of collecting failures.
        rip_blockers: False skips the rip-around-a-failed-sink layer.
        on_done(i, job): called after each job's attempt, i its 1-based position in jobs.
        """
        failures: list[tuple[str, Exception]] = []
        keys = {job.key for job in jobs}
        queue = deque(enumerate(jobs, 1))
        cleared_for: dict[str, set[Pos]] = {}  # job key -> sinks whose blockers were ripped
        requeued: set[str] = set()
        ripped: set[str] = set()
        with ProgressBar(label, len(jobs)) as bar:
            while queue:
                i, job = queue.popleft()
                bar.update(n=0 if job.key in ripped or job.key in requeued else 1, info=job.key)
                if deadline is not None and time.monotonic() > deadline:
                    if fail_fast:
                        raise RuntimeError(f"Routing failed for net {job.key}: route deadline reached")
                    failures.append((job.key, ValueError("route deadline reached")))
                    continue
                self.failed_goal.pop(job.key, None)
                try:
                    with self.rec.recording(job.key):
                        self.route_job_retrying(job)
                except Exception as e:
                    self.rip(job.key)
                    goal = self.failed_goal.get(job.key)
                    cleared = cleared_for.get(job.key, set())
                    victims = [] if goal is None or goal in cleared or len(cleared) >= _RIP_ROUNDS \
                        or not rip_blockers else [
                        v for v in self.blockers(goal, job.key, keys) if v not in ripped][:_BLOCKER_MAX]
                    if victims and not isinstance(e, LayConflict):
                        cleared_for.setdefault(job.key, set()).add(goal)
                        requeued.add(job.key)
                        log(f"{label} {job.key}: {e}; ripping {', '.join(victims)} and retrying")
                        for v in victims:
                            self.rip(v)
                            ripped.add(v)
                        queue.appendleft((i, job))
                        queue.extend(entry for entry in enumerate(jobs, 1) if entry[1].key in victims)
                        continue
                    if fail_fast:
                        raise RuntimeError(f"Routing failed for net {job.key}: {e}") from e
                    failures.append((job.key, e))
                    log(f"[error] skipped net {job.key}: {e}")
                if on_done is not None:
                    on_done(i, job)
        return failures


def _route_all_nets(
        workspace: Region,
        rec: Recorder,
        solid: JournaledSet,
        dust_owner: JournaledDict,
        sorted_nets: list[NetConnection],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        ws_bounds: tuple[int, int, int, int, int, int],
        max_bridge_y: int,
        footprint_blocked: frozenset[tuple[int, int, int]],
        all_terminals: dict[str, set[tuple[int, int, int]]],
        snapshot_dir: Path | None = None,
        deadline: float | None = None,
        fail_fast: bool = False,
        terminal_approach: dict[tuple[int, int, int], tuple[int, int]] | None = None,
) -> tuple[list[tuple[str, Exception]], set[tuple[int, int, int]], dict[str, tuple[str, list[list[Pos]]]]]:
    """Greedy routing of every net (_GreedyRouter); returns (failures, inverted cells, paths).

    fail_fast: raise on the first net that fails instead of collecting failures.
    """
    inverted_cells = JournaledSet(rec, name="inverted")
    keepouts = _KeepoutMap(all_terminals, terminal_approach)
    router = _GreedyRouter(workspace, rec, solid, dust_owner, inverted_cells, footprint_blocked, max_bridge_y,
                           frozenset(pin_terminal.values()))
    jobs = [
        _RouteJob(
            key=net.net_id, net_id=net.net_id,
            src=_pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name),
            sinks=[_pin_for_endpoint(pin_terminal, s.component_id, s.pin_name) for s in net.sinks],
            bounds=ws_bounds,
            protected=keepouts.for_net(net.net_id),
            terminals=all_terminals.get(net.net_id, set()),
            support_block=WOOLS[i % len(WOOLS)],
        )
        for i, net in enumerate(sorted_nets, 1)
    ]

    def snapshot(i: int, job: _RouteJob) -> None:
        if snapshot_dir is not None and i % 10 == 0:
            safe_id = job.key.replace("/", "_").replace("\\", "_")
            save_schematic_artifact(f"net_{i:03d}_{safe_id}.litematic", workspace, snapshot_dir)

    failures = router.run(jobs, "[wire]", deadline=deadline, fail_fast=fail_fast, on_done=snapshot)
    print(f"[wire] done ({len(jobs)} nets, {len(failures)} failed)")
    return failures, inverted_cells, router.paths


def _commit_route(
        workspace: Region,
        rec: Recorder,
        solid: JournaledSet,
        dust_owner: JournaledDict,
        inverted_cells: JournaledSet,
        net_id: str,
        route: NetRoute,
        net_ids: set[str],
) -> bool:
    """Write a virtual route into the real workspace; rip it back out if it clashes.

    Clash = a block lands on a non-air cell (other than sharing an existing solid as
    support), or one of its signal cells ends up next to (halo) another committed net's
    signal — the same adjacency _wire_walkable forbids.
    """
    ws = JournalingWorkspace(workspace, rec)
    ok = True
    with rec.recording(net_id):
        for pos, block in route.overlay.items():
            if not _is_air(workspace[pos]):
                if pos in solid and pos in route.support:
                    continue  # shared support block
                ok = False
                break
            ws[pos] = block
        if ok:
            for pos in route.solid:
                solid.add(pos)
            for pos, owner in route.owner.items():
                dust_owner[pos] = owner
            for pos in route.inverted:
                inverted_cells.add(pos)
    if ok:
        for pos in route.halo:
            owner = dust_owner.get(pos)
            if owner is not None and owner != net_id and owner in net_ids:
                ok = False
                break
    if not ok:
        rip_up(rec, net_id, workspace, solid, dust_owner)
    return ok


def _route_all_nets_negotiated(
        workspace: Region,
        rec: Recorder,
        solid: JournaledSet,
        dust_owner: JournaledDict,
        sorted_nets: list[NetConnection],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        ws_bounds: tuple[int, int, int, int, int, int],
        max_bridge_y: int,
        footprint_blocked: frozenset[tuple[int, int, int]],
        all_terminals: dict[str, set[tuple[int, int, int]]],
        passes: int,
        component_cells: frozenset[tuple[int, int, int]] = frozenset(),
        snapshot_dir: Path | None = None,
        search_timeout: float | None = _ROUTE_SEARCH_TIMEOUT,
        final_search_timeout: float | None = _ROUTE_FINAL_SEARCH_TIMEOUT,
        route_workers: int = 1,
        deadline: float | None = None,
        fail_fast: bool = False,
        terminal_approach: dict[tuple[int, int, int], tuple[int, int]] | None = None,
) -> tuple[list[tuple[str, Exception]], set[tuple[int, int, int]]]:
    """Negotiated-congestion routing (PathFinder), then commit, then a hard cleanup pass.

    Negotiation passes: every net (pass 0) or every net touching overuse / unrouted (later
    passes) is ripped up and rerouted against the static world plus its own blocks only.
    Other nets add cost, not walls: Congestion.cost = history + pres * users, with pres
    rising per pass and history growing on cells still overused after each pass. Stops
    when a pass leaves no overuse and no unrouted nets, or after _ROUTE_STAGNATION_PASSES
    passes without improvement, or passes-1 passes.

    Per-net search limit starts at search_timeout and is multiplied by
    _ROUTE_TIMEOUT_ESCALATION each time that net times out (capped at final_search_timeout),
    so hard nets aren't retried with a limit they already failed.

    Commit: the best pass's routes are written into the real workspace, clean nets first;
    any that clash, or that verify_signals finds open/shorted afterwards (claims don't model
    every interaction), are ripped out. Leftovers get a final hard (overlap-free) A* pass
    with final_search_timeout per search.

    route_workers > 1 runs negotiation passes in forked worker processes (parallel.py):
    nets are routed in batches against congestion frozen at the start of each batch.

    deadline (time.monotonic()): past it, no new net/batch is started; the best pass so far
    is committed and the final hard pass skips its remaining nets (reported as failures),
    so a capped build still writes a partial result.

    fail_fast: raise on the first net that fails the final hard pass (negotiation-pass
    failures are retried, so they don't count).
    """
    base_index = _BlockIndex(workspace)  # static world; unchanged until commit
    inverted_cells = JournaledSet(rec, name="inverted")
    net_ids = {net.net_id for net in sorted_nets}
    net_by_id = {net.net_id: net for net in sorted_nets}
    net_order = {net.net_id: i for i, net in enumerate(sorted_nets)}
    all_terminal_positions = frozenset(pin_terminal.values())
    cong = Congestion()
    routes: dict[str, NetRoute] = {}
    last_error: dict[str, Exception] = {}
    net_timeout: dict[str, float | None] = {}

    def terminals(net_id: str) -> tuple[tuple[int, int, int], list[tuple[int, int, int]]]:
        net = net_by_id[net_id]
        return (_pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name),
                [_pin_for_endpoint(pin_terminal, s.component_id, s.pin_name) for s in net.sinks])

    keepouts = _KeepoutMap(all_terminals, terminal_approach)

    def route_net(net: NetConnection, ws, solid_view, owner_view, inverted, congestion_cost, time_limit):
        src_pin = _pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name)
        protected = keepouts.for_net(net.net_id)
        net_ordered_paths: list[list[tuple[int, int, int]]] = []
        dst_pins = (_pin_for_endpoint(pin_terminal, sink.component_id, sink.pin_name) for sink in net.sinks)
        sorted_pins = sorted(dst_pins,
                             key=lambda p: abs(p[0] - src_pin[0]) + abs(p[1] - src_pin[1]) + abs(p[2] - src_pin[2]))
        support_block = WOOLS[net_order[net.net_id] % len(WOOLS)]
        for dst_pin in sorted_pins:
            tree_seeds = [pos for _path in net_ordered_paths for pos in _path]
            path = _find_wire_path(
                ws, solid_view, owner_view,
                src_pin, dst_pin, net.net_id,
                ws_bounds, max_bridge_y,
                tree_seeds=tree_seeds,
                protected=protected,
                footprint_blocked=footprint_blocked,
                inverted_cells=frozenset(inverted),
                terminal_positions=all_terminal_positions,
                congestion_cost=congestion_cost,
                time_limit=time_limit,
                seed_depths=_signal_depths(net_ordered_paths),
            )
            _lay_redstone_path(ws, solid_view, owner_view, path, net.net_id,
                               opaque_support_block=support_block,
                               inverted_cells=inverted,
                               goal=dst_pin,
                               terminal_positions=all_terminal_positions)
            net_ordered_paths.append(path)
        _place_repeaters_for_net(ws, owner_view, net.net_id, net_ordered_paths)

    def route_virtual(net_id: str, pres: float, cong: Congestion, time_limit: float | None) -> NetRoute:
        view = OverlayWorkspace(workspace, base_index)
        solid_view = OverlaySet(solid)
        owner_view = OverlayDict(dust_owner)
        inverted: set[tuple[int, int, int]] = set()
        route_net(net_by_id[net_id], view, solid_view, owner_view, inverted,
                  lambda pos: cong.cost(pos, pres, _CONFLICT_BASE), time_limit)
        return NetRoute.from_view(view, solid_view, owner_view, inverted)

    def settle(net_id: str, old: NetRoute | None, route: NetRoute | None, error: Exception | None) -> NetRoute | None:
        """Record a reroute's outcome; a failed one (e.g. timeout) keeps the previous route."""
        if error is None:
            last_error.pop(net_id, None)
            return route
        last_error[net_id] = error
        limit = net_timeout.get(net_id, search_timeout)
        if isinstance(error, RouteTimeout) and limit is not None:
            cap = final_search_timeout if final_search_timeout is not None else limit * _ROUTE_TIMEOUT_ESCALATION
            net_timeout[net_id] = min(limit * _ROUTE_TIMEOUT_ESCALATION, cap)
        return old

    def past_deadline() -> bool:
        return deadline is not None and time.monotonic() > deadline

    def route_pass_sequential(bar: ProgressBar, pres: float) -> None:
        for net_id in to_route:
            if past_deadline():
                return
            bar.update(info=net_id)
            old = routes.pop(net_id, None)
            if old is not None:
                cong.remove(net_id, old)
            try:
                route, error = route_virtual(net_id, pres, cong, net_timeout.get(net_id, search_timeout)), None
            except Exception as e:
                route, error = None, e
            route = settle(net_id, old, route, error)
            if route is not None:
                routes[net_id] = route
                cong.add(net_id, route)

    def route_pass_parallel(bar: ProgressBar, pres: float) -> None:
        # Batches of _ROUTE_BATCH_PER_WORKER * workers nets route against the same frozen
        # congestion; within a batch, slowest-last-time nets go out first.
        nonlocal deltas
        batch_size = _ROUTE_BATCH_PER_WORKER * route_workers

        def progress(net_id: str) -> None:
            bar.update(info=net_id)

        for b in range(0, len(to_route), batch_size):
            if past_deadline():
                return
            batch = to_route[b:b + batch_size]
            olds = {}
            for net_id in batch:
                old = olds[net_id] = routes.pop(net_id, None)
                if old is not None:
                    cong.remove(net_id, old)
                    deltas.append(("rm", net_id, claims_of(old)))
            tasks = [(net_id, pres, net_timeout.get(net_id, search_timeout))
                     for net_id in sorted(batch, key=lambda n: -net_seconds.get(n, 0.0))]
            results = workers.run_batch(deltas, tasks, progress)
            deltas = []
            for net_id in batch:
                route, error, net_seconds[net_id] = results[net_id]
                route = settle(net_id, olds[net_id], route, error)
                if route is not None:
                    routes[net_id] = route
                    cong.add(net_id, route)
                    deltas.append(("add", net_id, claims_of(route)))

    order = [net.net_id for net in sorted_nets]
    to_route = list(order)
    best: tuple[tuple[int, int], dict[str, NetRoute]] | None = None
    stale = 0
    deltas: list = []  # congestion changes not yet sent to the workers
    net_seconds: dict[str, float] = {}
    workers = RouteWorkers(route_workers, route_virtual) if route_workers > 1 and passes > 1 else None
    try:
        for k in range(max(passes - 1, 0)):
            pres = _PRES_SCHEDULE[min(k, len(_PRES_SCHEDULE) - 1)]
            with ProgressBar(f"[wire] pass {k}", len(to_route)) as bar:
                if workers is None:
                    route_pass_sequential(bar, pres)
                else:
                    route_pass_parallel(bar, pres)
            over = cong.overused()
            for pos in over:
                cong.hist[pos] += 1
            if workers is not None:
                deltas.append(("hist", None, list(over)))
            conflicted = cong.nets_at(over) & routes.keys()
            unrouted = [net_id for net_id in order if net_id not in routes]
            score = (len(unrouted) + len(conflicted), len(over))
            if best is None or score < best[0]:
                best = (score, dict(routes))
                stale = 0
            else:
                stale += 1
            if not over and not unrouted:
                break
            if stale >= _ROUTE_STAGNATION_PASSES:
                break
            if past_deadline():
                break
            to_route = [net_id for net_id in order if net_id in conflicted or net_id not in routes]
    finally:
        if workers is not None:
            workers.close()

    # Commit the best pass. Rebuild congestion for exactly those routes to know which are clean.
    final_routes = best[1] if best is not None else {}
    final_cong = Congestion()
    for net_id, route in final_routes.items():
        final_cong.add(net_id, route)
    conflicted = final_cong.nets_at(final_cong.overused())
    commit_order = sorted(final_routes, key=lambda n: (n in conflicted, net_order[n]))
    leftovers = [net_id for net_id in order if net_id not in final_routes]
    for net_id in commit_order:
        if not _commit_route(workspace, rec, solid, dust_owner, inverted_cells, net_id,
                             final_routes[net_id], net_ids):
            last_error[net_id] = ValueError("route clashed with another net at commit")
            leftovers.append(net_id)
    # Claims only model same-cell use and adjacency; check the committed result for real.
    committed = [net_id for net_id in commit_order if net_id not in leftovers]
    bad: set[str] = set()
    for violation in verify_signals(workspace, {n: terminals(n) for n in committed}, dust_owner,
                                    component_cells):
        bad |= violation.nets & set(committed)
    for net_id in sorted(bad, key=net_order.get):
        rip_up(rec, net_id, workspace, solid, dust_owner)
        last_error[net_id] = ValueError("open/short after commit (verify_signals)")
        leftovers.append(net_id)
    if snapshot_dir is not None:
        save_schematic_artifact("negotiated_commit.litematic", workspace, snapshot_dir)

    # Final hard pass in the real workspace: overlaps forbidden, no search timeout.
    ws = JournalingWorkspace(workspace, rec)
    failures: list[tuple[str, Exception]] = []
    with ProgressBar("[wire] final", len(leftovers)) as bar:
        for net_id in sorted(leftovers, key=net_order.get):
            bar.update(info=net_id)
            if past_deadline():
                if fail_fast:
                    raise RuntimeError(f"Routing failed for net {net_id}: route deadline reached")
                failures.append((net_id, ValueError("route deadline reached")))
                continue
            try:
                with rec.recording(net_id):
                    route_net(net_by_id[net_id], ws, solid, dust_owner, inverted_cells, None, final_search_timeout)
            except Exception as e:
                rip_up(rec, net_id, workspace, solid, dust_owner)
                if fail_fast:
                    raise RuntimeError(f"Routing failed for net {net_id}: {e}") from e
                failures.append((net_id, e))
    for net_id, e in failures:
        print(f"[error] unrouted net {net_id}: {e}")
    print(f"[wire] done ({len(order)} nets, {len(failures)} failed)")
    return failures, inverted_cells


def _report_signal_violations(
        workspace: Region,
        nets: list[NetConnection],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        component_cells: frozenset[tuple[int, int, int]],
        failed: set[str],
) -> None:
    """Print verify_signals opens/shorts for the routed nets (routing can 'succeed' yet be broken)."""
    spec = {
        net.net_id: (_pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name),
                     [_pin_for_endpoint(pin_terminal, s.component_id, s.pin_name) for s in net.sinks])
        for net in nets if net.net_id not in failed
    }
    violations = verify_signals(workspace, spec, dust_owner, component_cells)
    for violation in violations:
        print(f"[verify] {violation.kind} {sorted(violation.nets)} at {violation.pos}")
    if violations:
        print(f"[verify] {len(violations)} violation(s) in {len(spec)} routed nets")


def _report_unsupported_blocks(
        workspace: Region,
        dust_owner: dict[tuple[int, int, int], str],
        component_cells: frozenset[tuple[int, int, int]],
) -> None:
    """Print verify_support hits: dust/repeaters/torches with nothing under (or behind) them."""
    violations = verify_support(workspace, dust_owner, component_cells)
    for violation in violations:
        block = workspace[violation.pos]
        print(f"[verify] unsupported {block.id} {sorted(violation.nets)} at {violation.pos}")
    if violations:
        print(f"[verify] {len(violations)} unsupported block(s)")


def _compute_critical_path(
        comp: ComponentList,
        dust_owner: dict[tuple[int, int, int], str],
        workspace: Region,
) -> int:
    """Count repeaters/torches per net, topo-sort DP; returns the critical path in ticks."""
    from collections import deque

    # Wire ticks per net: repeaters and torches placed in dust_owner
    net_ticks: dict[str, int] = {n.net_id: 0 for n in comp.nets}
    for pos, nid in dust_owner.items():
        if nid not in net_ticks:
            continue
        blk = workspace[pos[0], pos[1], pos[2]]
        if _is_repeater(blk) or _is_torch(blk):
            net_ticks[nid] += 1

    # IO repeater on nets driven by INPUT_PINs (+1 per such net)
    comp_by_id: dict[str, Component] = {c.id: c for c in comp.components}
    for n in comp.nets:
        if comp_by_id[n.source.component_id].type == ComponentType.INPUT_PIN:
            net_ticks[n.net_id] += 1

    _SEQUENTIAL = frozenset({ComponentType.DFF, ComponentType.DFFE, ComponentType.DLATCH})

    # Sequential components are split into two virtual graph nodes to break cycles:
    #   "{id}\x00q" — virtual Q-output source: in_degree=0, out_time=0 (registered output)
    #   "{id}"      — real D/C/E-input sink: receives edges from combinational logic, no
    #                 outgoing edges; arrival tracks setup-time constraint
    def _q_node(cid: str) -> str:
        return cid + "\x00q"

    seq_ids = {c.id for c in comp.components if c.type in _SEQUENTIAL}
    all_nodes: set[str] = {c.id for c in comp.components} | {_q_node(cid) for cid in seq_ids}

    edges: dict[str, list[tuple[str, str]]] = {n: [] for n in all_nodes}
    in_degree: dict[str, int] = {n: 0 for n in all_nodes}

    for n in comp.nets:
        src = n.source.component_id
        src_node = _q_node(src) if src in seq_ids else src
        for sink in n.sinks:
            sid = sink.component_id
            edges[src_node].append((n.net_id, sid))
            in_degree[sid] += 1

    arrival: dict[str, int] = {n: 0 for n in all_nodes}

    queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    processed = 0
    while queue:
        nid = queue.popleft()
        processed += 1
        if nid.endswith("\x00q"):
            out_t = 0  # registered Q: always t=0
        else:
            c = comp_by_id[nid]
            delay = SCHEMATIC_MAP[c.type].propagation_delay_ticks if c.type in SCHEMATIC_MAP else 0
            out_t = arrival[nid] + delay
        for net_id, sink_id in edges[nid]:
            t = out_t + net_ticks[net_id]
            if t > arrival[sink_id]:
                arrival[sink_id] = t
            in_degree[sink_id] -= 1
            if in_degree[sink_id] == 0:
                queue.append(sink_id)

    if processed < len(all_nodes):
        print("[timing] WARNING: cycle detected in component graph — path may be incomplete")

    # Critical endpoints: OUTPUT_PINs and DFF/DFFE/DLATCH real nodes (setup-time sinks).
    # Fall back to all real component nodes if none exist.
    endpoint_ids = [
                       c.id for c in comp.components
                       if c.type == ComponentType.OUTPUT_PIN or c.type in _SEQUENTIAL
                   ] or [c.id for c in comp.components]

    return max(arrival[n] for n in endpoint_ids)


def _construct_clock_region(
        output_target: tuple[int, int, int],
        critical_path_ticks: int,
        arm_wire_length: int = 4
) -> Region:
    num_repeaters = ceil(critical_path_ticks // _REPEATER_MAX_DELAY / 2) * 2
    region_depth = num_repeaters + arm_wire_length + 2  # repeaters split across sides, 2 blocks for each repeater for connecting
    region_width = 4
    region_height = 2
    ws = Region(output_target[0], output_target[1] - 1, output_target[2], region_width, region_height, region_depth)
    for z in range(region_depth):
        ws[0, 1, z] = REDSTONE
        for x in range(region_width):
            ws[x, 0, z] = BlockState("minecraft:sandstone")

    for x in range(2, 4):
        ws[x, 1, arm_wire_length - 1] = BlockState("minecraft:redstone_wire", power='15')

    for idx in range(num_repeaters // 2):
        z = arm_wire_length + idx * 2
        ws[1, 1, z] = BlockState("minecraft:repeater", facing='east', delay='1')
        ws[2, 1, z] = REDSTONE
        ws[3, 1, z] = BlockState("minecraft:repeater", facing='north', delay=str(_REPEATER_MAX_DELAY))
        ws[2, 1, z + 1] = BlockState("minecraft:repeater", facing='south', delay=str(_REPEATER_MAX_DELAY))
        ws[3, 1, z + 1] = REDSTONE
        if (idx + 3) % 8 == 0:
            ws[0, 1, z + 1] = BlockState('minecraft:repeater', facing='south', delay='1')
    for x in range(2, 4):
        ws[x, 1, arm_wire_length + num_repeaters] = REDSTONE

    return ws


def build_litematic_from_component_list(
        comp: ComponentList,
        schematics_dir: Path,
        out_path: Path,
        *,
        gutter: int | None = None,
        workspace_size: tuple[int, int, int] | None = None,
        base_y: int = 5,
        schematic_name: str = "build",
        component_width_spacing: int = 7,
        component_depth_spacing: int | None = None,
        additional_component_height_spacing: int = 3,
        routing_headroom: int = 5,
        bridge_height: int = 1,
        allow_routing_failures: bool = False,
        output_pin_targets: dict[str, tuple[int, int]] | None = None,
        pin_targets_space=50,
        clock_pin: str = 'clk',
        route_passes: int = _ROUTE_MAX_PASSES,
        wire_timeout: float | None = _ROUTE_SEARCH_TIMEOUT,
        route_workers: int = 1,
        route_deadline: float | None = None,
        snapshots: bool = False,
        placer: str = "sa",
        place_y_moves: bool = True,
        place_z_moves: bool = True,
        place_flow_rule: bool = True,
        place_mirror: bool = True,
        place_seed: int = 0,
        place_workers: int = 1,
        flat: bool = False,
) -> Path:
    comp = _expand_multibit_io(comp)
    max_comp_height = max((c.footprint.height for c in comp.components), default=1)
    y_stride = max_comp_height + additional_component_height_spacing
    y_level = _assign_component_y_levels(comp.components, comp.nets, y_stride=y_stride, clock_pin=clock_pin)
    if flat:
        y_level = {cid: 0 for cid in y_level}
        place_y_moves = False
    # Order output pins like their extension targets, so extensions don't have to cross.
    output_x_order = {pn: float(tx) for pn, (tx, _ty) in (output_pin_targets or {}).items()}
    if placer == "legacy":
        placed = _layout_components(
            comp.components, comp.nets,
            gutter=10 if gutter is None else gutter, base_y=base_y, y_level=y_level,
            io_margin=component_width_spacing,
            routing_gutter=6 if component_depth_spacing is None else component_depth_spacing, flat=flat,
            output_x_order=output_x_order,
        )
    else:
        placed = _optimize_placement(
            comp.components, comp.nets, y_level,
            placer=placer, base_y=base_y, gutter=gutter,
            io_margin=component_width_spacing, routing_gutter=component_depth_spacing,
            y_moves=place_y_moves, z_moves=place_z_moves, flow_rule=place_flow_rule,
            mirror_moves=place_mirror, seed=place_seed,
            clock_pin=clock_pin, workers=place_workers, flat=flat, output_x_order=output_x_order,
        )
    if output_pin_targets:
        # Reserve z=0 for output extensions by shifting the existing build one layer forward.
        placed = [
            _Placed(component=p.component, origin=(p.origin[0], p.origin[1], p.origin[2] + pin_targets_space))
            for p in placed
        ]
    else:
        # One free layer in front of the output repeaters, for their lamps.
        placed = [
            _Placed(component=p.component, origin=(p.origin[0], p.origin[1], p.origin[2] + 1))
            for p in placed
        ]
    work_nets = comp.nets

    if workspace_size is None:
        width, height, depth = _compute_workspace_dims(
            placed, base_y=base_y, io_margin=component_width_spacing, routing_headroom=routing_headroom,
        )
    else:
        width, height, depth = workspace_size
    print(f"[layout] workspace {width}x{height}x{depth}")

    output_layer_z = 0
    workspace_depth = depth

    workspace = Region(0, 0, 0, width, height, workspace_depth)
    rec = Recorder()
    solid: set[tuple[int, int, int]] = JournaledSet(rec, name="solid")
    dust_owner: dict[tuple[int, int, int], str] = JournaledDict(rec)
    # Keep core net routing out of the reserved extension layer at z=0.
    ws_min_z = pin_targets_space if output_pin_targets else 0
    ws_bounds = (0, 0, ws_min_z, width - 1, height - 1, depth - 1)
    io_bounds = ws_bounds
    ext_bounds = (0, 0, 0, width - 1, height - 1, workspace_depth - 1)

    ref_mc_version, ref_lm_version, ref_lm_sub = _place_gate_templates(
        workspace, placed, solid, schematics_dir, dust_owner,
    )
    const_positions = _place_const_sources(workspace, placed, solid)
    max_bridge_y = _compute_max_bridge_y(placed, base_y, bridge_height)
    pin_world_map, pin_terminal, terminal_approach = _compute_pin_maps(placed)
    io_repeater_cells, output_repeater_lookup, io_signs, io_fixtures = _place_io_repeaters(
        workspace, placed, solid, pin_world_map, io_bounds, clock_pin)
    output_component_ids = {c.id for c in comp.components if c.type == ComponentType.OUTPUT_PIN}
    net_id_by_output_endpoint: dict[tuple[str, str], str] = {
        (sink.component_id, sink.pin_name): net.net_id
        for net in work_nets
        for sink in net.sinks
        if sink.component_id in output_component_ids
    }
    footprint_blocked = _build_footprint_blocked(
        placed, pin_terminal, io_repeater_cells, const_positions, io_fixtures, block_output_lamps=not output_pin_targets,
    )
    all_terminals = _build_terminal_map(work_nets, pin_terminal)

    save_artifact("component_layout.json", [
        {
            "id": p.component.id,
            "type": p.component.type.value,
            "origin": list(p.origin),
            "footprint": {
                "width": p.component.footprint.width,
                "height": p.component.footprint.height,
                "depth": p.component.footprint.depth,
            },
            "pins": [
                {"name": pin.name, "direction": pin.direction.value, "offset": list(pin.offset),
                 "const_value": pin.const_value}
                for pin in p.component.pins
            ],
        }
        for p in placed
    ])

    # Snapshots are slow debug output (litematic serialization); stale ones are cleared either way.
    if DEFAULT_SNAPSHOT_DIR.exists():
        for f in DEFAULT_SNAPSHOT_DIR.iterdir():
            f.unlink()
    snapshot_dir = DEFAULT_SNAPSHOT_DIR if snapshots else None
    sorted_nets = sorted(work_nets, key=lambda net: _net_sort_key(net, pin_terminal))
    deadline = None if route_deadline is None else time.monotonic() + route_deadline
    if route_passes <= 1:
        if route_workers > 1:
            print("[wire] --route-workers ignored: needs --route-passes > 1")
        routing_failures, inverted_cells, net_paths = _route_all_nets(
            workspace, rec, solid, dust_owner,
            sorted_nets, pin_terminal, ws_bounds, max_bridge_y,
            footprint_blocked, all_terminals,
            snapshot_dir=snapshot_dir,
            deadline=deadline,
            fail_fast=not allow_routing_failures,
            terminal_approach=terminal_approach,
        )
    else:
        net_paths = {}  # negotiated mode keeps no ordered paths
        routing_failures, inverted_cells = _route_all_nets_negotiated(
            workspace, rec, solid, dust_owner,
            sorted_nets, pin_terminal, ws_bounds, max_bridge_y,
            footprint_blocked, all_terminals, route_passes,
            component_cells=frozenset(_component_interior(placed)),
            snapshot_dir=snapshot_dir,
            search_timeout=wire_timeout,
            route_workers=route_workers,
            deadline=deadline,
            fail_fast=not allow_routing_failures,
            terminal_approach=terminal_approach,
        )

    _report_signal_violations(workspace, work_nets, pin_terminal, dust_owner,
                              frozenset(_component_interior(placed)), {nid for nid, _ in routing_failures})

    if routing_failures:
        failed = ", ".join(nid for nid, _ in routing_failures)
        if allow_routing_failures:
            print(f"IGNORING - Routing failed for {len(routing_failures)} net(s): {failed}")
        else:
            raise RuntimeError(f"Routing failed for {len(routing_failures)} net(s): {failed}")

    if output_pin_targets:
        try:
            ext_failures = _route_output_pin_extensions(
                workspace,
                solid,
                dust_owner,
                output_pin_targets,
                output_repeater_lookup,
                net_id_by_output_endpoint,
                ext_bounds,
                output_layer_z,
                max_bridge_y,
                footprint_blocked=footprint_blocked,
                all_terminals=all_terminals,
                inverted_cells=inverted_cells,
                rec=rec,
                net_paths=net_paths,
            )
        except ValueError as e:  # bad target spec
            ext_failures = [("output pin targets", e)]
        if ext_failures:
            msg = "; ".join(f"{key}: {e}" for key, e in ext_failures)
            if allow_routing_failures:
                print(f"IGNORING - Pin target routing failed: {msg}")
            else:
                raise RuntimeError(f"Pin target routing failed: {msg}")
        elif snapshot_dir is not None:
            save_schematic_artifact("extensions.litematic", workspace, snapshot_dir)

    _report_unsupported_blocks(workspace, dust_owner, frozenset(_component_interior(placed)))

    critical_path_ticks = _compute_critical_path(comp, dust_owner, workspace)

    # Cosmetic only, after all verification: dust shapes, then label signs above I/O repeaters.
    apply_dust_shapes(workspace)
    place_io_signs(workspace, io_signs)
    place_io_fixtures(workspace, io_fixtures)

    regions = {'main': workspace}

    for p in placed:
        if p.component.id == clock_pin:
            target = (p.origin[0], p.origin[1], p.origin[2] + 2)
            clock_region = _construct_clock_region(target, critical_path_ticks)
            regions['clock'] = clock_region
            break

    schematic = Schematic(
        name=schematic_name,
        author="MineRTL",
        regions=regions
    )
    # schematic = workspace.as_schematic(
    #     name=schematic_name, author="minecraft-v", description="merged placement"
    # )
    #
    schematic.mc_version = ref_mc_version
    schematic.lm_version = ref_lm_version
    schematic.lm_subversion = ref_lm_sub
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schematic.save(str(out_path))
    return out_path
