from __future__ import annotations

import bisect
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Callable
from math import ceil

import numpy as np
from pathlib import Path

from litemapy import BlockState, Region, Schematic

from minecraft_v.placement_engine.probes import probe_components, resolve_probes
from minecraft_v.placement_engine.server_file import SERVER_FILE_SUFFIX, schematic_bounds, write_server_file
from minecraft_v.placement_engine.resume import ResumeStore
from minecraft_v.build_utils import save_artifact, save_schematic_artifact, DEFAULT_SNAPSHOT_DIR
from minecraft_v.cell_library import INIT_HIGH_STATES, SCHEMATIC_MAP
from minecraft_v.progress import ProgressBar, debug, log, phase, status
from minecraft_v.placement_engine.ir import (
    Component,
    ComponentList,
    ComponentType,
    NetConnection,
)
from .block_utils import _is_air, _ensure_support, _is_redstone_wire, _is_repeater, _is_torch, _block_str, _is_glass
from .constants import (
    _HORIZ_DIRS, _DIRS_6, _SIDE_NORMAL, _OPPOSITE_SIDE, WOOLS, REDSTONE, STONE, _REPEATER_MAX_DELAY,
    _ROUTE_STAGNATION_FIRST, _ROUTE_MAX_PASSES, _PRES_SCHEDULE, _CONFLICT_BASE, _ROUTE_STAGNATION_PASSES, _ROUTE_SEARCH_TIMEOUT,
    _ROUTE_TIMEOUT_ESCALATION, _ROUTE_FINAL_SEARCH_TIMEOUT, _ROUTE_BATCH_PER_WORKER,
)
from .layout import (
    _Placed, _expand_multibit_io, _assign_component_y_levels,
    _layout_components, _compute_workspace_dims,
)
from .anneal import _optimize_placement
from .poisson import _poisson_placement
from .negotiate import Congestion, OverlayDict, OverlaySet, NetRoute, OverlayWorkspace
from .parallel import RouteWorkers, claims_of, run_forked
from .propagate import settle as settle_blocks
from .pathfinding import IndexedWorkspace, RouteTimeout, _BlockIndex, _find_wire_path, _signal_depths
from .ripup import Recorder, JournaledSet, JournaledDict, JournalingWorkspace, rip_up, rollback
from .template import (
    _pin_world, _io_repeater_cell, _io_repeater_facing,
    _default_io_side, _paste_template, _load_template_region,
)
from .hold import fix_hold, flat_terminals, repeater_delay
from .macro import Macro, io_column_pins, load_macro, macro_from_capture, save_macro
from .cosmetics import apply_dust_shapes, place_io_fixtures, place_io_signs
from .verify import find_gate_contacts, find_inverted_contacts, find_latch, verify_signals, verify_support
from .wire import LayConflict, _lay_redstone_path, _place_repeaters_for_net

REDSTONE_BLOCK = BlockState("minecraft:redstone_block")
IRON_TRAPDOOR = BlockState("minecraft:iron_trapdoor", open="false", half="bottom", facing="north")
BARREL = BlockState("minecraft:barrel")
_WALL_VIA_EDGE = 16  # how far from a macro's far edge its waypoints spread
_WALL_VIA_LIFT = 4  # above a macro: clear of its keepout, with room for the wire's support
_VIA_TRIES, _VIA_STEP = 8, 3  # search around a blocked waypoint
_MACRO_SEEDS = 4  # placement seeds to try before giving up on a macro


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
        macros: dict[str, Macro] | None = None,
) -> tuple[int, int, int]:
    """Paste component schematics into workspace. Returns (mc_version, lm_version, lm_subversion)."""
    ref_mc_version = 2975
    ref_lm_version = 6
    ref_lm_sub = 1
    oriented_blocks: dict[tuple[str, bool, bool], list] = {}
    for item in placed:
        c = item.component
        if c.type == ComponentType.CUSTOM:
            _stamp_macro(workspace, macros[c.metadata["macro"]], item.origin, solid,
                         bool(c.metadata.get("mirror_x")), bool(c.metadata.get("rotate_180")), oriented_blocks)
            continue
        if c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
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
                        mirror_width=c.footprint.width if c.metadata.get("mirror_x") else None,
                        block_states=INIT_HIGH_STATES.get(c.type) if c.metadata.get("init") else None)
    return (ref_mc_version, ref_lm_version, ref_lm_sub)


def _stamp_macro(
        workspace: Region,
        macro: Macro,
        origin: tuple[int, int, int],
        solid: set[tuple[int, int, int]],
        mirrored: bool,
        rotated: bool,
        oriented_blocks: dict[tuple[str, bool, bool], list],
) -> None:
    """Paste a routed macro. Its footprint is walled off, so `solid` only has to know the hull."""
    key = (macro.name, mirrored, rotated)
    if key not in oriented_blocks:
        oriented_blocks[key] = macro.blocks_oriented(mirrored, rotated)
    blocks = oriented_blocks[key]
    ox, oy, oz = origin
    for (x, y, z), block in blocks:
        workspace[ox + x, oy + y, oz + z] = block


def _place_const_sources(
        workspace: Region,
        placed: list[_Placed],
        solid: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    """Place redstone blocks (const=1) and iron trapdoors (const=0) for const pins."""
    const_positions: set[tuple[int, int, int]] = set()
    for item in placed:
        c = item.component
        if c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN, ComponentType.CUSTOM):
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
        clock_lever: bool = False,
) -> tuple[set[tuple[int, int, int]], dict[str, tuple[tuple[int, int, int], str, tuple[str, str]]],
           list[tuple[tuple[int, int, int], tuple[int, int], str]],
           list[tuple[tuple[int, int, int], tuple[int, int], bool]]]:
    """Place I/O repeaters and return (all repeater cells, output target lookup, sign specs, fixture specs).

    The clock input (`clock_pin`, or a pin named `clk`) gets no lever: it is driven by a clock.
    With `clock_lever` it does, for the server to toggle.
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
            if not (is_input and c.id in (clock_pin, 'clk')) or (clock_lever and c.id == clock_pin):
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
    log(f"[ext] done ({len(jobs)} extensions, {len(failures)} failed)")
    return failures


def _component_interior(placed: list[_Placed]) -> set[tuple[int, int, int]]:
    """Cells inside non-IO component footprints."""
    return {
        (ox + dx, oy + dy, oz + dz)
        for item in placed
        if item.component.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN, ComponentType.CUSTOM)
        for ox, oy, oz in [item.origin]
        for fp in [item.component.footprint]
        for dx in range(fp.width)
        for dy in range(fp.height)
        for dz in range(fp.depth)
    }


_MACRO_WALL = 4  # keepout thickness inside a macro's box: thicker than any routing step, so the rest can be hollow


def _macro_shell(item: _Placed, margin: tuple[int, int, int, int] = (0, 0, 0, 0)) -> set[tuple[int, int, int]]:
    """Hull of a macro's box grown by margin (sides, below, above, unused); the middle stays out of the sets."""
    ox, oy, oz = item.origin
    fp = item.component.footprint
    side, below, above, _ = margin
    wall = _MACRO_WALL
    xs = np.arange(ox - side, ox + fp.width + side)
    ys = np.arange(oy - below, oy + fp.height + above)
    zs = np.arange(oz - side, oz + fp.depth + side)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    keep = ((gx < ox + wall) | (gx >= ox + fp.width - wall) | (gy < oy + wall) | (gy >= oy + fp.height - wall)
            | (gz < oz + wall) | (gz >= oz + fp.depth - wall))
    return set(zip(gx[keep].tolist(), gy[keep].tolist(), gz[keep].tolist()))


def _hidden_cells(placed: list[_Placed]) -> frozenset[tuple[int, int, int]]:
    """Cells verification treats as air: gate interiors and macro walls (nothing reaches past those)."""
    cells = _component_interior(placed)
    for item in placed:
        if item.component.type == ComponentType.CUSTOM:
            cells |= _macro_shell(item)
    return frozenset(cells)


def _build_footprint_blocked(
        placed: list[_Placed],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        io_repeater_cells: set[tuple[int, int, int]],
        const_positions: set[tuple[int, int, int]],
        io_fixtures: list[tuple[tuple[int, int, int], tuple[int, int], bool]] = (),
        block_output_lamps: bool = True,
        envelope: set[tuple[int, int, int]] = frozenset(),
) -> frozenset[tuple[int, int, int]]:
    """Hard-block zone: footprint interiors + expanded borders + IO repeater tops + const air-gaps
    + the lever/lamp cells (and their neighbours) behind input / in front of output repeaters."""
    fp_interior = _component_interior(placed)
    fp_expanded: set[tuple[int, int, int]] = set(fp_interior)
    for item in placed:
        if item.component.type == ComponentType.CUSTOM:
            fp_expanded |= _macro_shell(item, (1, 1, 2, 0))  # same margins as the gate expansion below
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
    fp_expanded |= envelope
    fp_expanded -= set(pin_terminal.values())
    return frozenset(fp_expanded)


_ENVELOPE_GATE_GAP = 6   # room kept past a gate's far face, where its pins may need wire
_ENVELOPE_THICKNESS = 3  # thicker than any single routing step, so no move hops the barrier


def _io_barrier(
        columns: dict[int, int],
        gate_far: list[tuple[int, int, int]],
        width: int,
) -> list[tuple[int, int]]:
    """(x, z) cells of a barrier past IO ``columns`` (x -> z of the pin's outer cell), far side = higher z.

    A column and its neighbours end at the pin. Between columns the limit is the farther of the two
    nearest, so no pocket opens behind a pin, and gates (x0, x1, far z) push it out only where they are.
    """
    if not columns:
        return []
    xs = sorted(columns)
    limit = []
    for x in range(width):
        lane = [columns[c] for c in (x - 1, x, x + 1) if c in columns]
        if lane:
            limit.append(max(lane))
            continue
        i = bisect.bisect(xs, x)
        near = max(columns[c] for c in xs[max(i - 1, 0):i + 1])
        limit.append(max([near, *(far + _ENVELOPE_GATE_GAP for x0, x1, far in gate_far if x0 <= x <= x1)]))
    cells = []
    for x, z in enumerate(limit):
        top = max(limit[max(x - 1, 0):x + 2]) + _ENVELOPE_THICKNESS
        cells.extend((x, far) for far in range(z + 1, top + 1))
    return cells


def _io_envelope_blocked(
        placed: list[_Placed],
        bounds: tuple[int, int, int, int, int, int],
        block_outputs: bool,
) -> set[tuple[int, int, int]]:
    """Wire keepout: nothing routes behind an input (+Z) or ahead of an output (-Z), at any Y.

    The barrier starts just past each pin's repeater; lever and lamp are placed after routing.
    """
    _, _, min_z, max_x, max_y, max_z = bounds
    inputs: dict[int, int] = {}
    outputs: dict[int, int] = {}
    gates = []
    for item in placed:
        x, _, z = item.origin
        if item.component.type == ComponentType.INPUT_PIN and not item.component.metadata.get("north_face"):
            inputs[x] = max(inputs.get(x, z + 1), z + 1)
        elif item.component.type == ComponentType.INPUT_PIN:
            outputs[x] = min(outputs.get(x, z - 1), z - 1)
        elif item.component.type == ComponentType.OUTPUT_PIN:
            outputs[x] = min(outputs.get(x, z - 1), z - 1)
        else:
            gates.append((x, x + item.component.footprint.width - 1, z, z + item.component.footprint.depth - 1))
    width = max_x + 1
    columns = [(x, z) for x, z in _io_barrier(inputs, [(x0, x1, z1) for x0, x1, _, z1 in gates], width)]
    if block_outputs:  # outputs mirrored: far side = lower z
        columns += [(x, -z) for x, z in _io_barrier({x: -z for x, z in outputs.items()},
                                                    [(x0, x1, -z0) for x0, x1, z0, _ in gates], width)]
    return {(x, y, z) for x, z in columns if min_z <= z <= max_z for y in range(max_y + 1)}


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


_LAY_RETRIES = 24  # per sink: re-searches after a path that couldn't be laid (LayConflict)
_NET_RETRIES = 6  # per net: re-routes after no repeater fits a stretch, or a latch or inverted contact
_SINK_RETRIES = 4  # per net: re-routes with a sink it couldn't reach moved to the front
_BLOCKER_RADIUS = 3  # nets with cells this close to a failed sink get ripped up and re-queued
_BLOCKER_MAX = 4
_RIP_ROUNDS = 3  # per job: rip-and-retry rounds, each for a sink that hasn't had one yet
_ESCALATE_RADIUS = 6  # a job out of retries gets this wider ring ripped, once, even nets ripped before
_ESCALATE_MAX = 8
_REPAIR_ROUNDS = 3  # re-routes of nets verify still finds fault with
_SPEC_CLEARANCE = 2  # speculative jobs clash with blocks this close (per axis): dust through a block to dust
_SINK_BATCH_MIN = 8  # sinks left in a job for searching them at once to be worth the forks
_SPEC_MAX_SPAN = 200  # jobs longer than this (_RouteJob.span) aren't worth speculating: a redo costs too much

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
    via: list[Pos] = field(default_factory=list)  # reached first, so the sinks branch from there
    lead: list[Pos] = field(default_factory=list)  # short of a pin, for a straight last stretch
    avoid: frozenset[Pos] = frozenset()  # cells verify found a fault at, kept out of re-routes
    first: tuple[Pos, ...] = ()    # sinks verify found a fault at, routed first in re-routes

    @property
    def span(self) -> int:
        """Total Manhattan distance to the sinks: what sorts nets, and a rough guess at the cost of routing this."""
        return sum(abs(self.src[0] - s[0]) + abs(self.src[1] - s[1]) + abs(self.src[2] - s[2]) for s in self.sinks)


@dataclass
class _KeptBranches:
    """A job's laid branches and where it stopped, while the jobs around a failed sink get ripped."""
    paths: list[list[Pos]]
    reached_vias: list[Pos]
    vias: list[Pos]
    leads: list[Pos]
    sinks: list[Pos]
    next_target: int
    targets: list[Pos]  # per path: the via, lead or sink it was laid for
    cells: list[frozenset[Pos]]  # per path: the blocks laying it wrote


@dataclass
class _Speculated:
    """What routing a job in a forked worker left: enough to redo it here without searching."""
    blocks: dict[Pos, tuple[str, tuple[tuple[str, str], ...]]]  # final block at each cell it wrote
    solid: dict[Pos, bool]  # final membership of what it touched
    inverted: dict[Pos, bool]
    owner: dict[Pos, str | None]
    paths: list[list[Pos]]
    laid: _KeptBranches


_LOG_EVERY_SECONDS = 30.0     # routing progress in the log
_LOG_SLOW_SEARCH_SECONDS = 5.0
_LOG_SLOW_JOB_SECONDS = 10.0


class _GreedyRouter:
    """Greedy, journaled routing of _RouteJobs with the retry layers that make it robust:

    - per sink: a path the search accepted can still fail to lay when it crosses its own
      earlier cells/supports outside the search's local snapshot (LayConflict). Undo that
      sink's partial path and search again without the conflicting cell.
    - per job: no repeater slot on a stretch (all slopes/turns): undo the job and route it
      again with that stretch blocked, keeping clear of the job's terminals.
    - per job: a sink can be walled in by the job's own earlier wire: route it again with that
      sink first.
    - per sink: a sink search that fails is usually walled in by jobs routed earlier. Keep the
      job's branches, rip the jobs around that sink, and search for it again. Once per sink.
      Searches start on a tight limit (_ROUTE_STAGNATION_FIRST) so that this comes early; a
      sink past its rip, or with no rip to come, gets the full limit if the tight one fails.
    - per run: if that didn't help, or there was nothing to rip, the whole job goes. Rip the
      jobs around the failed sink, route it again, then re-queue them. Up to _RIP_ROUNDS
      times per job, once per failed sink (a repeat failure at the same sink is final). A job
      out of those gets one wider ring ripped, whatever already happened to those jobs.
    - per batch, with workers: the next jobs route at once in forked workers, unable to see each
      other. Taken in queue order, a result is kept unless it comes within _SPEC_CLEARANCE of a
      block laid since the fork; otherwise, or if the worker failed, the job routes here as usual.
      A rip throws away what's left of the batch, as a result may rest on what it removed.

    Ripping a job around a failure takes only its branch nearest that failure, and the branches
    that hang off it (rip_branch). The job is re-queued to search for those sinks alone.

    A job that still fails is ripped up so its partial wire / support / repeaters don't
    block later jobs or show up as debris.
    """

    def __init__(self, region: Region, rec: Recorder, solid: JournaledSet, dust_owner: JournaledDict,
                 inverted_cells: JournaledSet, footprint_blocked: frozenset[Pos], max_bridge_y: int,
                 terminal_positions: frozenset[Pos],
                 paths: dict[str, tuple[str, list[list[Pos]]]] | None = None,
                 resume: ResumeStore | None = None,
                 gate_cells: frozenset[Pos] = frozenset()):
        self.resume = resume
        self.gate_cells = gate_cells
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
        self.kept: dict[str, _KeptBranches] = {}  # jobs to carry on from their failed sink
        self.has_kept_wire: set[str] = set()  # those, and jobs whose kept wire a whole retry has to clear
        self.sinks_cleared: dict[str, set[Pos]] = {}  # job key -> sinks that have had their one per-sink rip
        self.laid: dict[str, _KeptBranches] = {}  # finished jobs' branches, for rip_branch
        self.rips_blockers = True  # whether a failed sink can get room by ripping (run's rip_blockers)
        self.speculated: dict[str, _Speculated | None] = {}  # this batch's results still to take; None: worker failed
        self.laid_since_fork: set[Pos] = set()
        self.spec_max_span = _SPEC_MAX_SPAN
        self.sink_workers = 1  # searches at once for a long job's next sinks; 1 is one at a time
        self.sink_stats = dict.fromkeys(("batches", "searched", "kept", "clashed", "worse", "failed"), 0)
        self.sink_seconds = 0.0

    def net_paths(self, net_id: str) -> list[list[Pos]]:
        return [p for nid, ps in self.paths.values() if nid == net_id for p in ps]

    def rip(self, key: str) -> None:
        self.kept.pop(key, None)
        self.laid.pop(key, None)
        self.has_kept_wire.discard(key)
        rip_up(self.rec, key, self.indexed, self.solid, self.dust_owner)
        self.paths.pop(key, None)

    def rip_branch(self, job: _RouteJob, around: list[Pos]) -> None:
        """Rip the job's branch nearest any of around, with the branches that hang off it; keep the rest.

        Goes through a whole rip and lays the rest again, as other jobs' supports can be mixed up in the journal.
        The job carries on from the ripped sinks when it's next routed, which also puts its repeaters back.
        A job with no record of its branches, or that would lose a via its other branches cross on, goes whole.
        """
        laid = self.laid.get(job.key)
        if laid is None:
            self.rip(job.key)
            return

        def distance(i: int) -> int:
            return min((abs(c[0] - a[0]) + abs(c[1] - a[1]) + abs(c[2] - a[2]) for c in laid.cells[i] for a in around),
                       default=1 << 30)

        # On a tie the later branch: fewer can hang off it.
        nearest = min(reversed(range(len(laid.paths))), key=distance, default=None)
        gone, keep = set(laid.paths[nearest]) if nearest is not None else set(), []
        kept_cells = {job.src}  # sibling branches start here too
        for i, path in enumerate(laid.paths):
            if i == nearest:
                continue
            if path[0] in gone and path[0] not in kept_cells:
                gone.update(path)
            else:
                keep.append(i)
                kept_cells.update(path)
        ripped_via = any(laid.targets[i] in laid.vias for i in range(len(laid.paths)) if i not in keep)
        self.rip(job.key)
        if ripped_via or not keep:
            return
        paths, cells = [], []
        try:
            with self.rec.recording(job.key):
                for i in keep:
                    mark = len(self.rec.journals[job.key])
                    _lay_redstone_path(self.ws, self.solid, self.dust_owner, laid.paths[i], job.net_id,
                                       opaque_support_block=job.support_block,
                                       inverted_cells=self.inverted_cells,
                                       goal=laid.targets[i],
                                       terminal_positions=self.terminal_positions)
                    paths.append(laid.paths[i])
                    cells.append(self._cells_since(job.key, mark))
        except ValueError:  # the world moved on around it
            self.rip(job.key)
            return
        targets = [laid.targets[i] for i in keep]
        sinks = [t for t in targets if t in laid.sinks]
        self.kept[job.key] = _KeptBranches(paths, laid.vias, laid.vias, laid.leads,
                                           [*sinks, *(s for s in laid.sinks if s not in sinks)],
                                           len(laid.vias) + len(sinks), targets, cells)
        self.has_kept_wire.add(job.key)

    def _cells_since(self, key: str, mark: int) -> frozenset[Pos]:
        return frozenset(e[2] for e in self.rec.journals[key][mark:] if e[0] == "block")

    def route_job(self, job: _RouteJob, job_blocked: frozenset[Pos], first: tuple[Pos, ...] = ()) -> None:
        rec, ws = self.rec, self.ws
        kept = self.kept.pop(job.key, None)
        paths: list[list[Pos]] = kept.paths if kept else []
        path_targets: list[Pos] = kept.targets if kept else []
        path_cells: list[frozenset[Pos]] = kept.cells if kept else []
        net_seeds = [pos for pos, owner in self.dust_owner.items()
                     if owner == job.net_id and _is_redstone_wire(self.region[pos])] if job.branch_from_net else []
        reached_vias: list[Pos] = kept.reached_vias if kept else []
        if kept:
            vias, leads, sinks = kept.vias, kept.leads, kept.sinks
        else:
            vias = [v for v in (self._open_via(v, job) for v in job.via) if v is not None]
            leads = [v for v in (self._open_via(v, job) for v in job.lead) if v is not None]
            # Nearest first; a lead comes before its pin, which is further on the same line.
            sinks = sorted([*job.sinks, *leads],
                           key=lambda p: abs(p[0] - job.src[0]) + abs(p[1] - job.src[1]) + abs(p[2] - job.src[2]))
            sinks = [*first, *(p for p in sinks if p not in first)]
        targets = [*vias, *sinks]
        forked_paths: dict[Pos, list[Pos] | None] = {}
        laid_since_fork: set[Pos] = set()
        for dst_index, dst in enumerate(targets):
            if kept and dst_index < kept.next_target:
                continue
            status(f"{job.key} target {dst_index + 1}/{len(targets)}")
            pending = frozenset(job.sinks) & frozenset(sinks[max(0, dst_index - len(vias)):])
            tree_seeds = net_seeds + [pos for path in paths for pos in path]
            start = job.src
            far_via = next((v for v in reached_vias if (dst[2] - v[2]) * (job.src[2] - v[2]) < 0), None)
            trunk: frozenset[Pos] = frozenset()
            if far_via is not None:  # searching from the near side too would only flood the wall's face
                # Back along the wire to the via would branch off every cell of it, leaving no repeater slot.
                start = far_via
                near = [c for c in tree_seeds if (c[2] - far_via[2]) * (job.src[2] - far_via[2]) >= 0 and c != far_via]
                trunk = frozenset(near)
                tree_seeds = [far_via, *(c for c in tree_seeds if c not in trunk and c != far_via)]
            seed_depths = _signal_depths((self.net_paths(job.net_id) if net_seeds else []) + paths)
            self.failed_goal[job.key] = dst  # cleared when the sink is laid
            blocked: set[Pos] = set(job_blocked - {job.src, dst})
            ends = {job.src, dst}
            attempts = 0
            forked = None
            if self.sink_workers > 1 and far_via is None and dst_index >= len(vias) and not set(leads) & set(targets[dst_index:]):
                if dst not in forked_paths and len(targets) - dst_index >= _SINK_BATCH_MIN:
                    forked_paths = self._search_sinks(job, targets, sinks, dst_index, tree_seeds, seed_depths, job_blocked)
                    laid_since_fork = set()
                if dst in forked_paths:
                    forked = self._usable_sink_path(forked_paths.pop(dst), dst, laid_since_fork)
            while True:
                mark = len(rec.journals[job.key])
                search_started = time.monotonic()
                saved = forked or (self.resume.path_for(job.key, dst) if self.resume else None)
                forked = None  # a path that can't be laid gets a search of its own
                try:
                    try:
                        def search(stagnation: int = -1) -> list[Pos]:
                            return _find_wire_path(
                                ws, self.solid, self.dust_owner,
                                start, dst, job.net_id,
                                job.bounds, self.max_bridge_y,
                                tree_seeds=tree_seeds,
                                protected=job.protected,
                                footprint_blocked=self.footprint_blocked,
                                inverted_cells=frozenset(self.inverted_cells),
                                terminal_positions=self.terminal_positions,
                                blocked=frozenset(blocked | trunk),
                                seed_depths=seed_depths,
                                pending_sinks=pending,
                                stagnation=stagnation,
                                **job.search_kwargs,
                            )

                        def search_tight_first() -> list[Pos]:
                            # Failing fast is cheaper than searching hard while a rip could still make room.
                            try:
                                return search(_ROUTE_STAGNATION_FIRST)
                            except ValueError:
                                if dst in leads or (self.rips_blockers and dst in job.sinks
                                                    and dst not in self.sinks_cleared.get(job.key, ())):
                                    raise
                                rollback(rec, job.key, mark)  # a failed search marks its way with lamps
                                return search()

                        path = saved
                        if path is None:
                            try:
                                path = search_tight_first()
                            except ValueError:
                                if start == job.src:
                                    raise
                                start, trunk = job.src, frozenset()  # past the via there's no way on: from the whole tree
                                tree_seeds = net_seeds + [pos for p in paths for pos in p]
                                rollback(rec, job.key, mark)
                                path = search_tight_first()
                    finally:
                        if time.monotonic() - search_started >= _LOG_SLOW_SEARCH_SECONDS:
                            debug(f"[wire] {job.key}: slow search, {time.monotonic() - search_started:.1f}s, "
                                f"target {dst_index + 1}/{len(vias) + len(sinks)}")
                    beside_start = saved is None and start != job.src and path[0] != start and path[0] not in tree_seeds
                    if beside_start and path[0][0] == start[0] and path[0][2] == start[2]:
                        # Began right above or below the via: no move links that, so search from the tree instead.
                        start, trunk = job.src, frozenset()
                        tree_seeds = net_seeds + [pos for p in paths for pos in p]
                        continue
                    if beside_start:
                        # A search may begin beside its start. Past a via, the repeater walk needs
                        # the link back to it, or it counts signal from nothing.
                        path = [start, *path]
                    _lay_redstone_path(ws, self.solid, self.dust_owner, path, job.net_id,
                                       opaque_support_block=job.support_block,
                                       inverted_cells=self.inverted_cells,
                                       goal=dst,
                                       terminal_positions=self.terminal_positions)
                    if self.resume:
                        self.resume.remember(job.key, dst, path)
                    break
                except LayConflict as e:
                    if self.resume:
                        self.resume.forget(job.key, dst)
                    cell = next((c for c in e.cells if c not in ends and c not in blocked), None)
                    if cell is None or len(blocked) >= len(job_blocked) + _LAY_RETRIES:
                        if dst not in leads:
                            raise
                        rollback(rec, job.key, mark)
                        path = None  # only a hint
                        break
                    rollback(rec, job.key, mark)
                    if cell in tree_seeds:  # can't branch here; the wire stays
                        tree_seeds = [t for t in tree_seeds if t != cell]
                        attempts += 1
                        if attempts > _LAY_RETRIES:
                            raise
                    else:
                        blocked.add(cell)
                except ValueError:
                    if self.resume:
                        self.resume.forget(job.key, dst)
                    if dst not in leads:
                        if dst in job.sinks:
                            rollback(rec, job.key, mark)
                            self.kept[job.key] = _KeptBranches(paths, reached_vias, vias, leads, sinks, dst_index,
                                                               path_targets, path_cells)
                            self.has_kept_wire.add(job.key)
                        raise
                    rollback(rec, job.key, mark)
                    path = None  # only a hint: the pin itself may still be reachable
                    break
            self.failed_goal.pop(job.key, None)
            if path is None:
                continue
            if dst in vias:
                reached_vias.append(dst)
            paths.append(path)
            path_targets.append(dst)
            path_cells.append(self._cells_since(job.key, mark))
            if forked_paths:
                laid_since_fork |= path_cells[-1]
        # The net's earlier paths first, so a branch starting mid-net counts signal from the
        # last repeater upstream rather than from where it joins.
        _place_repeaters_for_net(ws, self.dust_owner, job.net_id, self.net_paths(job.net_id) + paths)
        # Wire branches off the net's own sinks, so signal has to be followed through those.
        latch = find_latch(self.region, job.src, self.terminal_positions - {job.src, *job.sinks})
        if latch and all(self.dust_owner.get(cell, job.net_id) == job.net_id for cell in latch):
            raise LayConflict(f"Net '{job.net_id}': wire past a repeater feeds back into it (latch)", tuple(latch))
        # The search only sees its own path's last few cells, and a tower or drop can land beside wire laid earlier.
        own_cells = frozenset(e[2] for e in rec.journals[job.key] if e[0] == "block")
        inverted = find_inverted_contacts(self.region, job.src, job.sinks,
                                          self.terminal_positions - {job.src, *job.sinks},
                                          self.footprint_blocked, self.terminal_positions | own_cells)
        if inverted:
            raise LayConflict(f"Net '{job.net_id}': wire touches its own inverted signal at {inverted[:4]} (inverted)", tuple(inverted))
        touching = find_gate_contacts(self.region, own_cells, self.gate_cells)
        if touching:
            raise LayConflict(f"Net '{job.net_id}': wire powers a gate's block at {touching[:4]} (gate contact)", tuple(touching))
        self.paths[job.key] = (job.net_id, paths)
        self.laid[job.key] = _KeptBranches(paths, reached_vias, vias, leads, sinks, len(targets), path_targets, path_cells)
        self.has_kept_wire.discard(job.key)

    def _search_sinks(self, job: _RouteJob, targets: list[Pos], sinks: list[Pos], first_index: int,
                      tree_seeds: list[Pos], seed_depths, blocked: frozenset[Pos]) -> dict[Pos, list[Pos] | None]:
        """Search for the next sinks at once in forked workers, against the wire as it is now."""
        started = time.monotonic()
        batch = targets[first_index:first_index + self.sink_workers]

        def search(index: int, dst: Pos) -> list[Pos] | None:
            try:
                return _find_wire_path(
                    self.ws, self.solid, self.dust_owner,
                    job.src, dst, job.net_id,
                    job.bounds, self.max_bridge_y,
                    tree_seeds=tree_seeds,
                    protected=job.protected,
                    footprint_blocked=self.footprint_blocked,
                    inverted_cells=frozenset(self.inverted_cells),
                    terminal_positions=self.terminal_positions,
                    blocked=blocked - {job.src, dst},
                    seed_depths=seed_depths,
                    pending_sinks=frozenset(job.sinks) & frozenset(targets[index:]),
                    stagnation=_ROUTE_STAGNATION_FIRST,
                    **job.search_kwargs,
                )
            except ValueError:
                return None

        results = run_forked([lambda i=first_index + n, dst=dst: search(i, dst) for n, dst in enumerate(batch)])
        self.sink_stats["batches"] += 1
        self.sink_stats["searched"] += len(batch)
        self.sink_seconds += time.monotonic() - started
        return dict(zip(batch, results))

    def _usable_sink_path(self, path: list[Pos] | None, dst: Pos, laid: set[Pos]) -> list[Pos] | None:
        """A forked search's path, unless wire laid since the fork is too close or would have made a shorter branch."""
        if path is None:
            self.sink_stats["failed"] += 1
            return None
        if laid:
            r = range(-_SPEC_CLEARANCE, _SPEC_CLEARANCE + 1)
            near = [(dx, dy, dz) for dx in r for dy in r for dz in r]
            if any((x + dx, y + dy, z + dz) in laid for x, y, z in path for dx, dy, dz in near):
                self.sink_stats["clashed"] += 1
                return None
            if min(abs(c[0] - dst[0]) + abs(c[1] - dst[1]) + abs(c[2] - dst[2]) for c in laid) < len(path):
                self.sink_stats["worse"] += 1
                return None
        self.sink_stats["kept"] += 1
        return path

    def _via_open(self, via: Pos, job: _RouteJob) -> bool:
        """Whether a waypoint is free: air with air around it, so no other net's wire is near, and outside other
        nets' terminal keepouts, which a search can never enter."""
        x, y, z = via
        if not (0 <= x < self.region.width and 0 <= y < self.region.height - 2 and 0 <= z < self.region.length):
            return False
        if via in job.protected:
            return False
        cells = [(x + dx, y + dy, z + dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        return all(c not in self.footprint_blocked and self.dust_owner.get(c) in (None, job.net_id)
                   and _is_air(self.region[c]) for c in cells)

    def _open_via(self, via: Pos, job: _RouteJob) -> Pos | None:
        """The free cell nearest a wanted waypoint, trying along X and Z and then a layer up. None if there's none."""
        x, y, z = via
        for dy in range(0, 3):
            for step in range(_VIA_TRIES):
                for dx, dz in ((step, 0), (-step, 0), (0, step), (0, -step), (step, step), (-step, -step)):
                    cell = (x + _VIA_STEP * dx, y + 2 * dy, z + _VIA_STEP * dz)
                    if self._via_open(cell, job):
                        return cell
        return None

    def route_job_retrying(self, job: _RouteJob) -> None:
        near_terminal = {(x + dx, y + dy, z + dz) for x, y, z in job.terminals
                         for dx, dy, dz in ((0, 0, 0), *_DIRS_6)}
        job_blocked = job.avoid
        first = job.first
        lay_attempts = 0
        while True:
            mark = len(self.rec.journals[job.key])
            try:
                self.route_job(job, job_blocked, first)
                return
            except LayConflict as e:
                stretch = frozenset(e.cells) - near_terminal - job_blocked
                lay_attempts += 1
                if lay_attempts > _NET_RETRIES or not stretch or not any(m in str(e) for m in ("cannot place repeater", "(latch)", "(inverted)")):
                    raise
                self._undo_job(job.key, mark)
                job_blocked |= stretch
            except ValueError:
                goal = self.failed_goal.get(job.key)
                already_first = False  # nothing routed before it, so routing it first is the same search again
                if job.key in self.kept:
                    cleared = self.sinks_cleared.setdefault(job.key, set())
                    if self.rips_blockers and goal not in cleared:
                        cleared.add(goal)  # run() rips around it and carries on from this sink
                        raise
                    already_first = not self.kept.pop(job.key).paths
                if already_first or goal is None or goal not in job.sinks or goal in first or len(first) >= len(job.first) + _SINK_RETRIES:
                    raise
                self._undo_job(job.key, mark)
                first = (*first, goal)

    def _undo_job(self, key: str, mark: int) -> None:
        """Back to routing the job from nothing. Wire kept across a rip predates mark and goes too."""
        if key in self.has_kept_wire:
            self.rip(key)
        else:
            rollback(self.rec, key, mark)

    def blockers(self, goal: Pos, key: str, keys: set[str], r: int = _BLOCKER_RADIUS) -> list[str]:
        """Jobs (of keys) with wire or blocks within r of goal, nearest first."""
        gx, gy, gz = goal
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

    def _speculate_job(self, job: _RouteJob) -> _Speculated | None:
        """In a forked worker: route the job, and report what that wrote."""
        self.sink_workers = 1
        mark = len(self.rec.journals[job.key])
        try:
            with self.rec.recording(job.key):
                self.route_job_retrying(job)
        except Exception:
            return None
        blocks, solid, inverted, owner = {}, {}, {}, {}
        for kind, target, key, _, _ in self.rec.journals[job.key][mark:]:
            if kind == "block":
                block = self.region[key]
                blocks[key] = (block.id, tuple(sorted(block.properties())))
            elif kind == "dict":
                owner[key] = self.dust_owner.get(key)
            else:
                (solid if target is self.solid else inverted)[key] = key in target
        return _Speculated(blocks, solid, inverted, owner, self.paths[job.key][1], self.laid[job.key])

    def _speculate(self, queue: deque, workers: int) -> int:
        """Route the jobs at the head of the queue in forked workers; returns how many.

        A job past the span cutoff ends the batch, as does one carrying on from kept wire (its journal
        predates the fork) or another job of the same net (its repeaters and signal depths are shared).
        """
        batch: list[_RouteJob] = []
        for _, job in queue:
            if (len(batch) == workers or job.span > self.spec_max_span
                    or job.key in self.kept or job.key in self.has_kept_wire
                    or any(job.net_id == other.net_id for other in batch)):
                break
            batch.append(job)
        if len(batch) < 2:
            return 0
        results = run_forked([lambda job=job: self._speculate_job(job) for job in batch])
        self.speculated = {job.key: result for job, result in zip(batch, results)}
        self.laid_since_fork = set()
        return len(batch)

    def _take_speculated(self, job: _RouteJob, spec: _Speculated) -> bool:
        """Redo a worker's result here, unless something laid since the fork is too close to it."""
        r = range(-_SPEC_CLEARANCE, _SPEC_CLEARANCE + 1)
        near = [(dx, dy, dz) for dx in r for dy in r for dz in r]
        laid = self.laid_since_fork
        if laid and any((x + dx, y + dy, z + dz) in laid for x, y, z in spec.blocks for dx, dy, dz in near):
            return False
        states: dict[tuple, BlockState] = {}
        for pos, key in spec.blocks.items():
            block = states.get(key)
            if block is None:
                block = states[key] = BlockState(key[0], **dict(key[1]))
            self.ws[pos] = block
        for target, cells in ((self.solid, spec.solid), (self.inverted_cells, spec.inverted)):
            for pos, present in cells.items():
                if present:
                    target.add(pos)
                else:
                    target.discard(pos)
        for pos, net_id in spec.owner.items():
            if net_id is None:
                self.dust_owner.pop(pos, None)
            else:
                self.dust_owner[pos] = net_id
        self.paths[job.key] = (job.net_id, spec.paths)
        self.laid[job.key] = spec.laid
        if self.resume:
            for target, path in zip(spec.laid.targets, spec.paths):
                self.resume.remember(job.key, target, path)
        return True

    def run(self, jobs: list[_RouteJob], label: str, deadline: float | None = None, fail_fast: bool = False,
            on_done: Callable[[int, _RouteJob], None] | None = None,
            rip_blockers: bool = True, workers: int = 1, sink_workers: int = 1) -> list[tuple[str, Exception]]:
        """Route jobs in order; returns (job key, error) for the ones that failed.

        fail_fast: raise on the first job that fails instead of collecting failures.
        rip_blockers: False skips the rip-around-a-failed-sink layer.
        on_done(i, job): called after each job's attempt, i its 1-based position in jobs.
        workers: above 1, route short jobs speculatively in batches of this many.
        sink_workers: above 1, jobs routed here search for their sinks in batches of this many.
        """
        self.rips_blockers = rip_blockers
        self.sink_workers = sink_workers
        failures: list[tuple[str, Exception]] = []
        keys = {job.key for job in jobs}
        by_key = {job.key: job for job in jobs}
        queue = deque(enumerate(jobs, 1))
        cleared_for: dict[str, set[Pos]] = {}  # job key -> sinks whose blockers were ripped
        requeued: set[str] = set()
        ripped: set[str] = set()
        escalated: set[str] = set()
        started = last_log = time.monotonic()
        attempts = 0
        self.speculated = {}
        spec_stats = dict.fromkeys(("batches", "jobs", "kept", "clashed", "worker_failed", "discarded", "here"), 0)
        spec_seconds = spec_take_seconds = spec_here_seconds = 0.0
        with ProgressBar(label, len(jobs)) as bar:
            while queue:
                now = time.monotonic()
                if now - last_log >= _LOG_EVERY_SECONDS:
                    last_log = now
                    if self.resume:
                        self.resume.save_paths()
                    debug(f"{label} {bar.n}/{len(jobs)} started, {attempts} attempts, {len(ripped)} ripped, "
                        f"{len(failures)} failed, {len(queue)} queued ({now - started:.0f}s)")
                attempts += 1
                if workers > 1 and not self.speculated and (deadline is None or now <= deadline):
                    forked = self._speculate(queue, workers)
                    spec_stats["batches"] += forked > 0
                    spec_stats["jobs"] += forked
                    spec_seconds += time.monotonic() - now
                i, job = queue.popleft()
                bar.update(n=0 if job.key in ripped or job.key in requeued else 1, info=job.key)
                if deadline is not None and time.monotonic() > deadline:
                    if fail_fast:
                        raise RuntimeError(f"Routing failed for net {job.key}: route deadline reached")
                    failures.append((job.key, ValueError("route deadline reached")))
                    self.rip(job.key)  # may hold branches from before a rip
                    continue
                self.failed_goal.pop(job.key, None)
                job_started = time.monotonic()
                batched = job.key in self.speculated
                spec = self.speculated.pop(job.key, None)
                mark = len(self.rec.journals[job.key])
                try:
                    with self.rec.recording(job.key):
                        taken = spec is not None and self._take_speculated(job, spec)
                        spec_take_seconds += time.monotonic() - job_started
                        if not taken:
                            here_started = time.monotonic()
                            self.route_job_retrying(job)
                            dt = time.monotonic() - here_started
                            spec_here_seconds += dt
                            if workers > 1 and dt > 0.3:
                                debug(f"{label} sequential {job.key}: {dt:.2f}s, span {job.span}, batched {batched}")
                    if batched:
                        spec_stats["kept" if taken else "clashed" if spec is not None else "worker_failed"] += 1
                    else:
                        spec_stats["here"] += 1
                    if self.speculated:
                        self.laid_since_fork |= self._cells_since(job.key, mark)
                    if time.monotonic() - job_started >= _LOG_SLOW_JOB_SECONDS:
                        debug(f"{label} {job.key}: {time.monotonic() - job_started:.0f}s, {len(job.sinks)} sinks")
                except Exception as e:
                    spec_stats["discarded"] += len(self.speculated)
                    self.speculated = {}
                    debug(f"{label} {job.key}: failed after {time.monotonic() - job_started:.0f}s, {len(job.sinks)} sinks")
                    if job.key in self.kept:
                        goal = self.failed_goal.get(job.key)
                        victims = [v for v in self.blockers(goal, job.key, keys) if v not in ripped][:_BLOCKER_MAX] \
                            if rip_blockers else []
                        requeued.add(job.key)
                        if victims:
                            log(f"{label} {job.key}: {e}; keeping its branches, ripping a branch of "
                                f"{', '.join(victims)} and retrying that sink")
                            for v in victims:
                                self.rip_branch(by_key[v], [goal])
                                ripped.add(v)
                            queue = deque(entry for entry in queue if entry[1].key not in victims)
                            queue.extend(entry for entry in enumerate(jobs, 1) if entry[1].key in victims)
                        else:
                            self.rip(job.key)
                        queue.appendleft((i, job))
                        continue
                    self.rip(job.key)
                    goal = self.failed_goal.get(job.key)
                    cleared = cleared_for.get(job.key, set())
                    # Around the source too: it can be the walled-in end, and nothing around the goal would help.
                    victims = [] if goal is None or goal in cleared or len(cleared) >= _RIP_ROUNDS \
                        or not rip_blockers else list(dict.fromkeys(
                            [v for v in self.blockers(goal, job.key, keys) if v not in ripped][:_BLOCKER_MAX]
                            + [v for v in self.blockers(job.src, job.key, keys) if v not in ripped][:_BLOCKER_MAX]))
                    if victims and not isinstance(e, LayConflict):
                        cleared_for.setdefault(job.key, set()).add(goal)
                        requeued.add(job.key)
                        log(f"{label} {job.key}: {e}; ripping a branch of {', '.join(victims)} and retrying")
                        for v in victims:
                            self.rip_branch(by_key[v], [goal, job.src])
                            ripped.add(v)
                        queue.appendleft((i, job))
                        queue.extend(entry for entry in enumerate(jobs, 1) if entry[1].key in victims)
                        continue
                    where = goal if goal is not None else next(iter(getattr(e, "cells", ())), None)
                    wider = [] if where is None or job.key in escalated or not rip_blockers else \
                        self.blockers(where, job.key, keys, _ESCALATE_RADIUS)[:_ESCALATE_MAX]
                    if wider:
                        escalated.add(job.key)
                        requeued.add(job.key)
                        log(f"{label} {job.key}: {e}; ripping a branch of {len(wider)} nets around {where} and retrying")
                        for v in wider:
                            self.rip_branch(by_key[v], [where])
                            ripped.add(v)
                        queue = deque(entry for entry in queue if entry[1].key not in wider)
                        queue.appendleft((i, job))
                        queue.extend(entry for entry in enumerate(jobs, 1) if entry[1].key in wider)
                        continue
                    if fail_fast:
                        raise RuntimeError(f"Routing failed for net {job.key}: {e}") from e
                    failures.append((job.key, e))
                    log(f"[error] skipped net {job.key}: {e}")
                if on_done is not None:
                    on_done(i, job)
        if workers > 1:
            log(f"{label} speculation: {spec_stats}, {spec_seconds:.1f}s in batches, "
                f"{spec_take_seconds:.1f}s taking, {spec_here_seconds:.1f}s sequential, "
                f"{time.monotonic() - started:.1f}s in all")
        if sink_workers > 1:
            log(f"{label} sink batches: {self.sink_stats}, {self.sink_seconds:.1f}s")
        if self.resume:
            self.resume.save_paths()
            log(f"{label} resume: {self.resume.reused} paths reused, {self.resume.searched} searched "
                f"({time.monotonic() - started:.0f}s)")
        return failures


_BANK_PIN_LEAD = 6


def _bank_pin_leads(
        net: NetConnection,
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        walls: list[_Placed],
        macros: list[_Placed],
) -> list[tuple[int, int, int]]:
    """Cells short of a bank's east pins, and at the mouth of the channel to pins on its outer faces.

    Wire running along Z right against east pins walls off the ones routed last, and a search heading straight for a
    pin round the corner floods the east face.
    """
    src = _pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name)
    sinks = [_pin_for_endpoint(pin_terminal, s.component_id, s.pin_name) for s in net.sinks]
    leads = []
    for wall in walls:
        (ox, oy, oz), fp = wall.origin, wall.component.footprint
        if wall.component.metadata.get("bank"):
            east = ox + fp.width
            hash_ = zlib.crc32(net.net_id.encode())
            lead = _BANK_PIN_LEAD + 2 * (hash_ % 8)  # spread, or wire running along Z just beyond them is the same wall
            at_pins = [p for p in sinks if east - 1 <= p[0] <= east + 1 and oz <= p[2] < oz + fp.depth
                       and oy <= p[1] < oy + fp.height and src[0] > p[0] + lead]
            leads += [(p[0] + lead, p[1], p[2]) for p in at_pins]
            for other in macros:  # small macros are not walls, but one in the way still floods: a hint over it
                (mx, my, mz), mf = other.origin, other.component.footprint
                if other.component.metadata.get("bank"):
                    continue
                leads += [(p[0], my + mf.height + _WALL_VIA_LIFT, mz + mf.depth // 2) for p in sinks
                          if mx <= p[0] < mx + mf.width
                          and ((src[2] < mz and p[2] >= mz + mf.depth - 1) or (src[2] >= mz + mf.depth and p[2] <= mz))]
                # Straight in to a pin on its faces: wire along the face walls off the rest.
                back = 4 + 2 * (zlib.crc32(net.net_id.encode()) % 3)
                leads += [(p[0], p[1], p[2] - back if p[2] < mz else p[2] + back) for p in sinks
                          if mx <= p[0] < mx + mf.width and (p[2] < mz or p[2] >= mz + mf.depth)
                          and p[2] - mz < mf.depth + 2 and mz - p[2] < 3]
            for other in macros:  # in the way of the straight line to a pin: over it
                (mx, my, mz), mf = other.origin, other.component.footprint
                if other.component.metadata.get("bank") or src[0] < mx + mf.width:
                    continue
                leads += [(mx + mf.width // 2, my + mf.height + _WALL_VIA_LIFT, p[2]) for p in at_pins
                          if mz <= p[2] < mz + mf.depth and p[1] < my + mf.height]
            if src[0] >= east:
                spread = 2 + 2 * ((hash_ >> 8) % 8)  # away from the face, as nets share a Y across tiles
                # Sinks across the bank from the source have a waypoint over it instead.
                leads += [(east + lead, p[1], p[2] - spread) for p in sinks
                          if p[0] < east and p[2] < oz and src[2] < oz + fp.depth]
                leads += [(east + lead, p[1], p[2] + spread) for p in sinks
                          if p[0] < east and p[2] >= oz + fp.depth and src[2] >= oz]
    return list(dict.fromkeys(leads))


def _wall_vias(
        net: NetConnection,
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        walls: list[_Placed],
) -> list[tuple[int, int, int]]:
    """A cell above each macro the net has to cross. A* heads straight at a goal behind one and floods its face."""
    src = _pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name)
    sinks = [_pin_for_endpoint(pin_terminal, s.component_id, s.pin_name) for s in net.sinks]
    via = []
    for wall in walls:
        (ox, oy, oz), fp = wall.origin, wall.component.footprint
        z0, z1 = oz, oz + fp.depth - 1
        crossing = [p for p in sinks if (src[2] < z0 and p[2] > z1) or (src[2] > z1 and p[2] < z0)]
        over = next((p[0] for p in (src, *crossing) if ox <= p[0] < ox + fp.width), None) if crossing else None
        if over is not None:
            # Spread by net: nets from one IO column share an X, and a cell only fits one.
            spread = zlib.crc32(net.net_id.encode())
            # Over whatever is stacked on this wall too.
            top = max(w.origin[1] + w.component.footprint.height for w in walls
                      if w.origin[0] < ox + fp.width and ox < w.origin[0] + w.component.footprint.width
                      and w.origin[2] < oz + fp.depth and oz < w.origin[2] + w.component.footprint.depth)
            # By the far edge: searches start here, and from near the first edge they drop back off it.
            along = (spread >> 8) % min(_WALL_VIA_EDGE, fp.depth)
            via.append((over, top + _WALL_VIA_LIFT + 2 * (spread % 4), z1 - along if src[2] < z0 else z0 + along))
    return via


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
        placed: list[_Placed] = (),
        resume: ResumeStore | None = None,
        route_workers: int = 1,
        spec_max_span: int = _SPEC_MAX_SPAN,
        sink_workers: int = 1,
) -> tuple[list[tuple[str, Exception]], set[tuple[int, int, int]], dict[str, tuple[str, list[list[Pos]]]]]:
    """Greedy routing of every net (_GreedyRouter); returns (failures, inverted cells, paths).

    fail_fast: raise on the first net that fails instead of collecting failures.
    route_workers: above 1, nets of up to spec_max_span route speculatively in batches (_GreedyRouter).
    """
    inverted_cells = JournaledSet(rec, name="inverted")
    keepouts = _KeepoutMap(all_terminals, terminal_approach)
    macros = [p for p in placed if p.component.type == ComponentType.CUSTOM]
    walls = [p for p in placed if p.component.type == ComponentType.CUSTOM
             and (p.component.metadata.get("bank")
                  or p.component.footprint.width >= 2 * p.component.footprint.depth)]  # others are easy to go around
    with phase("[wire] preparing nets"):
        router = _GreedyRouter(workspace, rec, solid, dust_owner, inverted_cells, footprint_blocked, max_bridge_y,
                               frozenset(pin_terminal.values()), resume=resume,
                               gate_cells=frozenset(_component_interior(placed)))
        router.spec_max_span = spec_max_span
        jobs = [
            _RouteJob(
                key=net.net_id, net_id=net.net_id,
                src=_pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name),
                sinks=[_pin_for_endpoint(pin_terminal, s.component_id, s.pin_name) for s in net.sinks],
                bounds=ws_bounds,
                protected=keepouts.for_net(net.net_id),
                terminals=all_terminals.get(net.net_id, set()),
                support_block=WOOLS[i % len(WOOLS)],
                via=_wall_vias(net, pin_terminal, walls),
                lead=_bank_pin_leads(net, pin_terminal, walls, macros),
            )
            for i, net in enumerate(sorted_nets, 1)
        ]

    def snapshot(i: int, job: _RouteJob) -> None:
        if snapshot_dir is not None and i % 10 == 0:
            safe_id = job.key.replace("/", "_").replace("\\", "_")
            save_schematic_artifact(f"net_{i:03d}_{safe_id}.litematic", workspace, snapshot_dir)

    started = time.monotonic()
    failures = router.run(jobs, "[wire]", deadline=deadline, fail_fast=fail_fast, on_done=snapshot,
                          workers=route_workers, sink_workers=sink_workers)
    wire = sum(len(path) for _, net_paths in router.paths.values() for path in net_paths)
    log(f"[wire] done ({len(jobs)} nets, {len(failures)} failed, {time.monotonic() - started:.1f}s, {wire} path cells)")
    if not macros:
        failures += _repair_faulty_nets(router, jobs, {key for key, _ in failures},
                                        frozenset(_component_interior(placed)), deadline)
    return failures, inverted_cells, router.paths


def _repair_faulty_nets(
        router: _GreedyRouter,
        jobs: list[_RouteJob],
        failed: set[str],
        component_cells: frozenset[Pos],
        deadline: float | None,
) -> list[tuple[str, Exception]]:
    """Re-route nets verify finds fault with, away from the fault. Routing can succeed yet leave one:
    a later net's wire can touch an earlier one in a way only the two together show."""
    order = {job.key: i for i, job in enumerate(jobs)}
    by_key = {job.key: job for job in jobs}
    failures: list[tuple[str, Exception]] = []
    for _ in range(_REPAIR_ROUNDS):
        spec = {job.key: (job.src, job.sinks) for job in jobs if job.key not in failed}
        faults: dict[str, set[Pos]] = {}
        with ProgressBar("[repair] verify", len(spec)) as bar:
            violations = verify_signals(router.region, spec, router.dust_owner, component_cells,
                                        on_net=lambda net_id: bar.update(info=net_id))
        for v in violations:
            # Of a short's two nets, moving the later one disturbs the least.
            key = max((n for n in v.nets if n in by_key), key=order.__getitem__, default=None)
            if key is not None and v.kind != "unsupported":
                faults.setdefault(key, set()).add(v.pos)
        if not faults:
            return failures
        log(f"[repair] re-routing {', '.join(sorted(faults))}")
        requeue = sorted(faults, key=order.__getitem__)
        for key in requeue:
            router.rip(key)
            job = by_key[key]
            job.avoid |= frozenset(faults[key]) - job.terminals
            job.first += tuple(p for p in sorted(faults[key]) if p in job.sinks and p not in job.first)
        new = router.run([by_key[key] for key in requeue], "[repair]", deadline=deadline, rip_blockers=False)
        failures += new
        failed |= {key for key, _ in new}
    return failures


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
        log(f"[error] unrouted net {net_id}: {e}")
    log(f"[wire] done ({len(order)} nets, {len(failures)} failed)")
    return failures, inverted_cells


def _report_signal_violations(
        workspace: Region,
        nets: list[NetConnection],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        component_cells: frozenset[tuple[int, int, int]],
        failed: set[str],
) -> list:
    """Print verify_signals opens/shorts for the routed nets (routing can 'succeed' yet be broken)."""
    spec = {
        net.net_id: (_pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name),
                     [_pin_for_endpoint(pin_terminal, s.component_id, s.pin_name) for s in net.sinks])
        for net in nets if net.net_id not in failed
    }
    with ProgressBar("[verify] signals", len(spec)) as bar:
        violations = verify_signals(workspace, spec, dust_owner, component_cells,
                                    on_net=lambda net_id: bar.update(info=net_id))
    for violation in violations:
        log(f"[verify] {violation.kind} {sorted(violation.nets)} at {violation.pos}")
    if violations:
        log(f"[verify] {len(violations)} violation(s) in {len(spec)} routed nets")
    return violations


def _report_flat_violations(
        workspace: Region,
        placed: list[_Placed],
        nets: list[NetConnection],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        macros: dict[str, Macro],
        failed: set[str],
) -> list:
    """verify_signals on the stamped layout as if flat: every net of the original netlist, from gate pin to gate
    pin, through the entry and exit repeaters and the wires inside macros. Only gate interiors are hidden."""
    stamps = {p.component.id: (macros[p.component.metadata["macro"]], p.origin, bool(p.component.metadata.get("mirror_x")),
                                  bool(p.component.metadata.get("rotate_180")))
              for p in placed if p.component.type == ComponentType.CUSTOM}
    hidden = _component_interior(placed)
    owner = {pos: net for pos, net in dust_owner.items()}
    spec: dict[str, tuple[tuple[int, int, int], list[tuple[int, int, int]]]] = {}

    def inner_ends(cid: str, pin: str) -> list[tuple[int, int, int]]:
        macro, origin, mirrored, rotated = stamps[cid]
        source, sinks = macro.nets[macro.port_net[pin]]
        ports = set(macro.port_net)
        cells = [source] if source.rsplit(":", 1)[0] not in ports else []
        cells += [s for s in sinks if s.rsplit(":", 1)[0] not in ports]
        return [macro.world(macro.terminals[e], origin, mirrored, rotated) for e in cells]

    exposed: dict[str, dict[str, str]] = {cid: {} for cid in stamps}
    for net in nets:
        endpoints = [net.source, *net.sinks]
        if net.net_id in failed:
            continue
        ends: list[tuple[int, int, int]] = []
        source = None
        for i, ep in enumerate(endpoints):
            if ep.component_id in stamps:
                macro = stamps[ep.component_id][0]
                exposed[ep.component_id][macro.port_net[ep.pin_name]] = net.net_id
                cells = inner_ends(ep.component_id, ep.pin_name)
                if i == 0:  # driven from inside: the first inner end is the driving gate pin
                    source, cells = cells[0], cells[1:]
                ends += cells
            elif i == 0:
                source = _pin_for_endpoint(pin_terminal, ep.component_id, ep.pin_name)
            else:
                ends.append(_pin_for_endpoint(pin_terminal, ep.component_id, ep.pin_name))
        spec[net.net_id] = (source, ends)
    for cid, (macro, origin, mirrored, rotated) in stamps.items():
        ports = set(macro.port_net.values())
        for inner, (source, sinks) in macro.nets.items():
            name = exposed[cid].get(inner, f"{cid}/{inner}")
            if inner not in ports:
                spec[name] = (macro.world(macro.terminals[source], origin, mirrored, rotated),
                              [macro.world(macro.terminals[s], origin, mirrored, rotated) for s in sinks])
        for rel, inner in macro.owner.items():
            owner[macro.world(rel, origin, mirrored, rotated)] = exposed[cid].get(inner, f"{cid}/{inner}")
        w = macro.footprint.width
        for (x, y, z), (sx, sy, sz) in macro.gate_boxes:
            for dx in range(sx):
                for dy in range(sy):
                    for dz in range(sz):
                        hidden.add(macro.world((x + dx, y + dy, z + dz), origin, mirrored, rotated))
    with ProgressBar("[verify] flat signals", len(spec)) as bar:
        violations = verify_signals(workspace, spec, owner, frozenset(hidden),
                                    on_net=lambda net_id: bar.update(info=net_id))
    for violation in violations:
        log(f"[verify] {violation.kind} {sorted(violation.nets)} at {violation.pos}")
    log(f"[verify] {len(violations)} violation(s) in {len(spec)} flattened nets through {len(stamps)} macros")
    return violations


def _report_unsupported_blocks(
        workspace: Region,
        dust_owner: dict[tuple[int, int, int], str],
        component_cells: frozenset[tuple[int, int, int]],
) -> None:
    """Print verify_support hits: dust/repeaters/torches with nothing under (or behind) them."""
    with phase("[verify] support"):
        violations = verify_support(workspace, dust_owner, component_cells)
    for violation in violations:
        block = workspace[violation.pos]
        log(f"[verify] unsupported {block.id} {sorted(violation.nets)} at {violation.pos}")
    if violations:
        log(f"[verify] {len(violations)} unsupported block(s)")


_SEQUENTIAL = frozenset({ComponentType.DFF, ComponentType.DFFE, ComponentType.DLATCH})


_WIRE_NEIGHBOURS = tuple(
    (dx, dy, dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)) for dy in (-1, 0, 1)
) + tuple((0, dy, 0) for dy in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5))


def _net_wire_ticks(
        comp: ComponentList,
        dust_owner: dict[tuple[int, int, int], str],
        workspace: Region,
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
) -> dict[str, int]:
    """Repeaters and torches on each net's deepest branch from its source."""
    cells: dict[str, dict[tuple[int, int, int], int]] = {n.net_id: {} for n in comp.nets}
    for pos, nid in dust_owner.items():
        if nid in cells:
            blk = workspace[pos[0], pos[1], pos[2]]
            cells[nid][pos] = repeater_delay(blk) if _is_repeater(blk) else 1 if _is_torch(blk) else 0
    ticks: dict[str, int] = {}
    for n in comp.nets:
        net = cells[n.net_id]
        src = _pin_for_endpoint(pin_terminal, n.source.component_id, n.source.pin_name)
        depth: dict[tuple[int, int, int], int] = {}
        floor = 0
        while len(depth) < len(net):  # wire not joined to the rest (a hop the neighbours miss) continues from the deepest
            start = min((p for p in net if p not in depth),
                        key=lambda p: abs(p[0] - src[0]) + abs(p[1] - src[1]) + abs(p[2] - src[2]))
            depth[start] = floor + net[start]
            queue = deque([start])
            while queue:
                x, y, z = pos = queue.popleft()
                for dx, dy, dz in _WIRE_NEIGHBOURS:
                    next_pos = (x + dx, y + dy, z + dz)
                    cost = net.get(next_pos)
                    if cost is None or depth.get(next_pos, 1 << 30) <= depth[pos] + cost:
                        continue
                    depth[next_pos] = depth[pos] + cost
                    if cost:
                        queue.append(next_pos)
                    else:
                        queue.appendleft(next_pos)
            floor = max(depth.values())
        ticks[n.net_id] = floor
    return ticks


_SETUP_TICKS = 1  # data settled at a flip-flop before its clock rises


def _gate_delay(c: Component) -> int:
    return SCHEMATIC_MAP[c.type].propagation_delay_ticks if c.type in SCHEMATIC_MAP else 0


def _compute_critical_path(
        comp: ComponentList,
        dust_owner: dict[tuple[int, int, int], str],
        workspace: Region,
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        macros: dict[str, Macro] | None = None,
        probed: set[str] = frozenset(),
        flat_period: int = 0,
) -> int:
    """Repeaters/torches on each net's deepest branch, topo-sort DP; returns the critical path in ticks."""
    from collections import deque

    # Wire ticks per net: repeaters and torches placed in dust_owner
    net_ticks = _net_wire_ticks(comp, dust_owner, workspace, pin_terminal)

    # IO repeater on nets driven by INPUT_PINs (+1 per such net)
    comp_by_id: dict[str, Component] = {c.id: c for c in comp.components}
    for n in comp.nets:
        if comp_by_id[n.source.component_id].type == ComponentType.INPUT_PIN:
            net_ticks[n.net_id] += 1

    # Sequential components are split into two virtual graph nodes to break cycles:
    #   "{id}\x00q" — virtual Q-output source: in_degree=0, out_time=0 (registered output)
    #   "{id}"      — real D/C/E-input sink: receives edges from combinational logic, no
    #                 outgoing edges; arrival tracks setup-time constraint
    # A macro has a node per pin, joined by its timing arcs.
    def _q_node(cid: str) -> str:
        return cid + "\x00q"

    def _macro_pin(cid: str, pin: str) -> str:
        return f"{cid}\x00pin\x00{pin}"

    macro_of = {c.id: macros[c.metadata["macro"]] for c in comp.components if c.type == ComponentType.CUSTOM}
    seq_ids = {c.id for c in comp.components if c.type in _SEQUENTIAL}
    all_nodes: set[str] = ({c.id for c in comp.components if c.id not in macro_of}
                           | {_q_node(cid) for cid in seq_ids}
                           | {_macro_pin(cid, p.name) for cid, m in macro_of.items() for p in m.pins})

    # (ticks, sink) per node; a macro's arcs are plain edges with the arc's ticks
    edges: dict[str, list[tuple[int, str]]] = {n: [] for n in all_nodes}
    in_degree: dict[str, int] = {n: 0 for n in all_nodes}

    for n in comp.nets:
        src = n.source.component_id
        if src in macro_of:
            src_node = _macro_pin(src, n.source.pin_name)
        else:
            src_node = _q_node(src) if src in seq_ids else src
        for sink in n.sinks:
            sid = _macro_pin(sink.component_id, sink.pin_name) if sink.component_id in macro_of else sink.component_id
            edges[src_node].append((net_ticks[n.net_id], sid))
            in_degree[sid] += 1
    arrival: dict[str, int] = {n: 0 for n in all_nodes}
    for cid, m in macro_of.items():
        for pin, arcs in m.arcs.items():
            for out, ticks in arcs.items():
                edges[_macro_pin(cid, pin)].append((ticks, _macro_pin(cid, out)))
                in_degree[_macro_pin(cid, out)] += 1
        for out, ticks in m.q_out.items():
            arrival[_macro_pin(cid, out)] = ticks

    queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    processed = 0
    while queue:
        nid = queue.popleft()
        processed += 1
        if nid.endswith("\x00q"):
            out_t = _gate_delay(comp_by_id[nid[:-2]])  # clock to Q, from the flip-flop's own clock edge
        elif nid in comp_by_id:
            out_t = arrival[nid] + _gate_delay(comp_by_id[nid])
        else:
            out_t = arrival[nid]  # macro pin: its arcs carry the delay
        for ticks, sink_id in edges[nid]:
            t = out_t + ticks
            if t > arrival[sink_id]:
                arrival[sink_id] = t
            in_degree[sink_id] -= 1
            if in_degree[sink_id] == 0:
                queue.append(sink_id)

    if processed < len(all_nodes):
        log("[timing] WARNING: cycle detected in component graph — path may be incomplete")

    # Critical endpoints: OUTPUT_PINs and DFF/DFFE/DLATCH real nodes (setup-time sinks), and registers in macros.
    # Fall back to all real component nodes if none exist.
    endpoints = [
                       arrival[c.id] for c in comp.components
                       if c.type == ComponentType.OUTPUT_PIN or c.type in _SEQUENTIAL
                   ] or [arrival[c.id] for c in comp.components if c.id not in macro_of]
    for cid, m in macro_of.items():
        endpoints += [arrival[_macro_pin(cid, pin)] + ticks for pin, ticks in m.setup.items()]
    # A print reads these at the clock edge, even if nothing else uses them. Those inside macros count in flat_period.
    endpoints += [arrival[cid] + _gate_delay(comp_by_id[cid]) for cid in probed if cid in comp_by_id
                  and comp_by_id[cid].type not in _SEQUENTIAL and comp_by_id[cid].type != ComponentType.INPUT_PIN]
    # Launch and capture clocks differ by at most the clock wire: the slowest branch against none.
    skew = max((net_ticks[n.net_id] for n in comp.nets
                if any(s.pin_name == "C" and comp_by_id[s.component_id].type in _SEQUENTIAL for s in n.sinks)), default=0)
    # Measured flat too: the top level can't see macros' flip-flops, their clocks, or padding inside them.
    return max(max(endpoints) + skew, flat_period) + _SETUP_TICKS


def _construct_clock_region(
        output_target: tuple[int, int, int],
        critical_path_ticks: int,
        arm_wire_length: int = 4
) -> Region:
    num_repeaters = ceil(critical_path_ticks / _REPEATER_MAX_DELAY / 2) * 2  # period = 4 ticks per repeater
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
        route_spec_max_span: int = _SPEC_MAX_SPAN,
        route_sink_workers: int = 1,
        route_deadline: float | None = None,
        snapshots: bool = False,
        placer: str = "poisson",
        place_y_moves: bool = True,
        place_z_moves: bool = True,
        place_flow_rule: bool = True,
        place_mirror: bool = True,
        place_seed: int = 0,
        place_workers: int = 1,
        flat: bool = False,
        settled: bool = True,
        macros: dict[str, Macro] | None = None,
        flat_comp: ComponentList | None = None,
        capture: dict | None = None,
        io_columns: int | None = None,
        macro_channel: int = 16,
        macro_east_room: int = 20,
        macro_headroom: int = 32,
        io_lane_pins: bool = True,
        resume: ResumeStore | None = None,
        probes: list[dict] | None = None,
        server_clock: int | None = None,
) -> Path:
    """capture: fill this dict with the routed layout and stop before cosmetics and the clock (see build_macro).

    server_clock: instead of generating `clock_pin`'s clock, give it a lever for MCHPRS to toggle every this many
    ticks (0: the generated clock's period for this critical path).
    """
    if macros:  # an array is a wall across the layout: wires get past it overhead
        routing_headroom = max(routing_headroom, macro_headroom)
        bridge_height = max(bridge_height, macro_headroom - 4)
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
    elif resume is not None and resume.load_placed(place_seed) is not None:
        placed = resume.load_placed(place_seed)
        log(f"[resume] placement from {resume.dir}")
    elif placer == "poisson":
        with phase("[place]"):
            placed = _poisson_placement(
                comp.components, comp.nets,
                base_y=base_y, gutter=gutter, io_margin=component_width_spacing,
                routing_gutter=component_depth_spacing, y_spacing=additional_component_height_spacing,
                seed=place_seed, flat=flat, output_x_order=output_x_order, io_column_pins=io_columns,
                macro_channel=macro_channel, macro_east_room=macro_east_room, io_lane_pins=io_lane_pins,
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
    if resume is not None:
        resume.save_placed(place_seed, placed)
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
        # Room past a bank's pin end, for the wire to its pins.
        width = max([width, *(p.origin[0] + p.component.footprint.width + macro_channel for p in placed
                              if p.component.metadata.get("bank"))])
    else:
        width, height, depth = workspace_size
    log(f"[layout] workspace {width}x{height}x{depth}")

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

    with phase("[layout] stamping gates"):
        ref_mc_version, ref_lm_version, ref_lm_sub = _place_gate_templates(
            workspace, placed, solid, schematics_dir, dust_owner, macros,
        )
        const_positions = _place_const_sources(workspace, placed, solid)
        max_bridge_y = _compute_max_bridge_y(placed, base_y, bridge_height)
        pin_world_map, pin_terminal, terminal_approach = _compute_pin_maps(placed)
        io_repeater_cells, output_repeater_lookup, io_signs, io_fixtures = _place_io_repeaters(
            workspace, placed, solid, pin_world_map, io_bounds, clock_pin, clock_lever=server_clock is not None)
        output_component_ids = {c.id for c in comp.components if c.type == ComponentType.OUTPUT_PIN}
        net_id_by_output_endpoint: dict[tuple[str, str], str] = {
            (sink.component_id, sink.pin_name): net.net_id
            for net in work_nets
            for sink in net.sinks
            if sink.component_id in output_component_ids
        }
        footprint_blocked = _build_footprint_blocked(
            placed, pin_terminal, io_repeater_cells, const_positions, io_fixtures, block_output_lamps=not output_pin_targets,
            envelope=_io_envelope_blocked(placed, ws_bounds, block_outputs=not output_pin_targets),
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
    with phase("[wire] ordering nets"):
        sorted_nets = sorted(work_nets, key=lambda net: _net_sort_key(net, pin_terminal))
    deadline = None if route_deadline is None else time.monotonic() + route_deadline
    if route_passes <= 1:
        routing_failures, inverted_cells, net_paths = _route_all_nets(
            workspace, rec, solid, dust_owner,
            sorted_nets, pin_terminal, ws_bounds, max_bridge_y,
            footprint_blocked, all_terminals,
            snapshot_dir=snapshot_dir,
            deadline=deadline,
            fail_fast=not allow_routing_failures,
            terminal_approach=terminal_approach,
            placed=placed,
            resume=resume,
            route_workers=route_workers,
            spec_max_span=route_spec_max_span,
            sink_workers=route_sink_workers,
        )
    else:
        net_paths = {}  # negotiated mode keeps no ordered paths
        routing_failures, inverted_cells = _route_all_nets_negotiated(
            workspace, rec, solid, dust_owner,
            sorted_nets, pin_terminal, ws_bounds, max_bridge_y,
            footprint_blocked, all_terminals, route_passes,
            component_cells=_hidden_cells(placed),
            snapshot_dir=snapshot_dir,
            search_timeout=wire_timeout,
            route_workers=route_workers,
            deadline=deadline,
            fail_fast=not allow_routing_failures,
            terminal_approach=terminal_approach,
        )

    failed_nets = {nid for nid, _ in routing_failures}
    flat_period = 0
    if capture is None:
        flat = flat_comp or comp
        inner = {(cell, pin): macro.world(rel, p.origin, bool(p.component.metadata.get("mirror_x")),
                                          bool(p.component.metadata.get("rotate_180")))
                 for p in placed if p.component.type == ComponentType.CUSTOM
                 for macro in [macros[p.component.metadata["macro"]]]
                 for cell, pin, rel in ((*key.rsplit(":", 1), rel) for key, rel in macro.terminals.items())}
        # Macro flip-flops aren't in the layout, and settling has to cut their clocks too.
        save_artifact("macro_clocks.json", [list(pos) for (_, pin), pos in inner.items() if pin == "C"])
        # Placement splits multi-bit IO into one component per bit, named after the bit.
        flat_period = fix_hold(workspace, flat, flat_terminals(flat, lambda cid, pin: pin_terminal.get((cid, pin), pin_terminal.get((pin, pin))),
                                                 inner), failed_nets, probe_components(probes) if probes else frozenset())
    if macros:
        signal_violations = _report_flat_violations(workspace, placed, work_nets, pin_terminal, dust_owner, macros,
                                                    failed_nets)
    else:
        signal_violations = _report_signal_violations(workspace, work_nets, pin_terminal, dust_owner,
                                                      frozenset(_component_interior(placed)), failed_nets)

    if routing_failures:
        failed = ", ".join(nid for nid, _ in routing_failures)
        if allow_routing_failures:
            log(f"IGNORING - Routing failed for {len(routing_failures)} net(s): {failed}")
        else:
            raise RuntimeError(f"Routing failed for {len(routing_failures)} net(s): {failed}")
    if signal_violations and not macros and not allow_routing_failures:
        raise RuntimeError(f"Verify still finds {len(signal_violations)} fault(s) after repair, first: {signal_violations[0]}")

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
                log(f"IGNORING - Pin target routing failed: {msg}")
            else:
                raise RuntimeError(f"Pin target routing failed: {msg}")
        elif snapshot_dir is not None:
            save_schematic_artifact("extensions.litematic", workspace, snapshot_dir)

    _report_unsupported_blocks(workspace, dust_owner, _hidden_cells(placed))

    if capture is not None:
        capture.update(
            workspace=workspace, placed=placed, dust_owner=dict(dust_owner), pin_terminal=pin_terminal,
            pin_world=pin_world_map,
            net_ticks=_net_wire_ticks(comp, dust_owner, workspace, pin_terminal),
            violations=signal_violations + verify_support(workspace, dust_owner, _hidden_cells(placed)),
            io_cells={(p.component.id, pin.name): _io_repeater_cell(pin_world_map[(p.component.id, pin.name)], p.origin,
                                                                    p.component.footprint,
                                                                    _default_io_side(p.component, pin))
                      for p in placed if p.component.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
                      for pin in p.component.pins},
        )
        return out_path

    with phase("[timing] critical path"):
        critical_path_ticks = _compute_critical_path(comp, dust_owner, workspace, pin_terminal, macros,
                                                     probe_components(probes) if probes else frozenset(), flat_period)
    log(f"[timing] critical path {critical_path_ticks} ticks")

    # Cosmetic only, after all verification: dust shapes, then label signs above I/O repeaters.
    with phase("[finish] dust shapes and signs"):
        apply_dust_shapes(workspace)
        place_io_signs(workspace, io_signs)
        place_io_fixtures(workspace, io_fixtures)

    regions = {'main': workspace}

    for p in placed:
        if p.component.id == clock_pin and server_clock is None:
            target = (p.origin[0], p.origin[1], p.origin[2] + 2)
            clock_region = _construct_clock_region(target, critical_path_ticks)
            regions['clock'] = clock_region
            # The first clock row is inside main, and loaders paste regions in no fixed order (air included).
            for x in range(clock_region.width):
                for y in range(clock_region.height):
                    wx, wy = clock_region.x + x, clock_region.y + y
                    if wx < width and target[2] < depth and clock_region[x, y, 0].id != "minecraft:air":
                        workspace[wx, wy, target[2]] = clock_region[x, y, 0]
            break

    low, size = schematic_bounds(list(regions.values()))
    clock_entry = None
    if server_clock is not None and clock_pin is not None:
        p = next(p for p in placed if p.component.id == clock_pin)
        pin = p.component.pins[0]
        cell = _io_repeater_cell(pin_world_map[(clock_pin, pin.name)], p.origin, p.component.footprint,
                                 _default_io_side(p.component, pin), bounds=io_bounds)
        nx, _, nz = _SIDE_NORMAL[_default_io_side(p.component, pin).value]
        lever = (cell[0] + nx, cell[1] + 1, cell[2] + nz)
        if workspace[lever].id != "minecraft:lever":
            raise RuntimeError(f"no room for the server clock lever of '{clock_pin}' at {lever}")
        half_period = server_clock or max(1, _REPEATER_MAX_DELAY * ceil(critical_path_ticks / _REPEATER_MAX_DELAY / 2))
        clock_entry = {"pin": clock_pin, "pos": [c - lo for c, lo in zip(lever, low)], "half_period_ticks": half_period}
        log(f"[clock] server clock on '{clock_pin}': toggles every {half_period} ticks")

    resolved_probes: list[dict] = []
    if probes:
        positions = dict(pin_world_map)
        positions.update({(p.component.id, pin.name): _io_repeater_cell(pin_world_map[(p.component.id, pin.name)], p.origin,
                                                                        p.component.footprint,
                                                                        _default_io_side(p.component, pin), bounds=io_bounds)
                          for p in placed if p.component.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
                          for pin in p.component.pins})
        positions.update({tuple(key.rsplit(":", 1)): macro.world(rel, p.origin, bool(p.component.metadata.get("mirror_x")),
                                                                 bool(p.component.metadata.get("rotate_180")))
                          for p in placed if p.component.type == ComponentType.CUSTOM
                          for macro in [macros[p.component.metadata["macro"]]]
                          for key, rel in macro.drivers.items()})
        resolved_probes, tapped = resolve_probes(probes, positions, workspace, low)
        log(f"[probes] {len(probes)} prints, {tapped} blocks tapped")
    if resolved_probes or clock_entry:
        server_path = out_path.with_suffix(SERVER_FILE_SUFFIX)
        write_server_file(server_path, size, resolved_probes, clock_entry)
        log(f"[server] {server_path}")

    if settled:
        with phase("[settle] propagating"):
            settle_blocks(workspace)

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
    with phase("[write] saving litematic"):
        schematic.save(str(out_path))
    return out_path


def build_macro(name: str, sub: ComponentList, ports: dict[str, str], cells: int, schematics_dir: Path,
                cache_dir: Path | None = None, route_deadline: float | None = None, **build_kwargs) -> Macro:
    """Place and route one instance into a Macro, trying placement seeds until it verifies clean."""
    if cache_dir is not None and (cached := load_macro(name, cache_dir)) is not None:
        log(f"[macro] {name}: cached, footprint {cached.footprint.width}x{cached.footprint.height}x{cached.footprint.depth}")
        return cached
    n_pins = len(ports)
    gutter, routing_gutter = 6, 6
    for seed in range(_MACRO_SEEDS):
        t0 = time.monotonic()
        capture: dict = {}
        try:
            build_litematic_from_component_list(
                sub, schematics_dir, Path("unused.litematic"), schematic_name=name, clock_pin=None,
                route_passes=1, place_seed=seed, capture=capture, route_deadline=route_deadline,
                gutter=gutter, component_depth_spacing=routing_gutter,
                io_columns=io_column_pins(n_pins, 4, gutter + 1), **build_kwargs,
            )
            if capture["violations"]:
                raise ValueError(f"{len(capture['violations'])} violations, first {capture['violations'][0]}")
            macro = macro_from_capture(name, sub, capture, ports, cells, _gate_delay, _SEQUENTIAL)
        except (RuntimeError, ValueError) as e:
            log(f"[macro] {name} seed {seed} failed: {e}")
            continue
        fp = macro.footprint
        log(f"[macro] {name}: {cells} cells, {n_pins} pins, footprint {fp.width}x{fp.height}x{fp.depth} "
              f"(seed {seed}, {time.monotonic() - t0:.1f}s)")
        if cache_dir is not None:
            save_macro(macro, cache_dir)
        return macro
    raise RuntimeError(f"macro {name} did not route clean in {_MACRO_SEEDS} seeds")
