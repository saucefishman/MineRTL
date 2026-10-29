from __future__ import annotations

import time
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
    _HORIZ_DIRS, _DIRS_6, _SIDE_NORMAL, _OPPOSITE_SIDE, WOOLS, REDSTONE, _REPEATER_MAX_DELAY,
    _ROUTE_MAX_PASSES, _PRES_SCHEDULE, _CONFLICT_BASE, _ROUTE_STAGNATION_PASSES, _ROUTE_SEARCH_TIMEOUT,
    _ROUTE_TIMEOUT_ESCALATION, _ROUTE_FINAL_SEARCH_TIMEOUT, _ROUTE_BATCH_PER_WORKER,
)
from .layout import (
    _Placed, _expand_multibit_io, _assign_component_y_levels,
    _layout_components, _compute_workspace_dims,
)
from .negotiate import Congestion, OverlayDict, OverlaySet, NetRoute, OverlayWorkspace
from .parallel import RouteWorkers, claims_of
from .pathfinding import IndexedWorkspace, RouteTimeout, _BlockIndex, _find_wire_path
from .ripup import Recorder, JournaledSet, JournaledDict, JournalingWorkspace, rip_up
from .template import (
    _pin_world, _io_repeater_cell, _io_repeater_facing,
    _default_io_side, _paste_template, _load_template_region,
)
from .verify import verify_signals, verify_support
from .wire import _lay_redstone_path, _place_repeaters_for_net

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
                        dust_owner, f"__schematic__{c.id}", pin_positions)
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
) -> tuple[dict[tuple[str, str], tuple[int, int, int]], dict[tuple[str, str], tuple[int, int, int]]]:
    """Return (pin_world_map, pin_terminal) for all component pins."""
    pin_world_map: dict[tuple[str, str], tuple[int, int, int]] = {}
    pin_terminal: dict[tuple[str, str], tuple[int, int, int]] = {}
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
                    pin_terminal[(c.id, pin.name)] = (pw[0] + nx, pw[1] + ny, pw[2] + nz)
    return pin_world_map, pin_terminal


def _place_io_repeaters(
        workspace: Region,
        placed: list[_Placed],
        solid: set[tuple[int, int, int]],
        pin_world_map: dict[tuple[str, str], tuple[int, int, int]],
        ws_bounds: tuple[int, int, int, int, int, int],
) -> tuple[set[tuple[int, int, int]], dict[str, tuple[tuple[int, int, int], str, tuple[str, str]]]]:
    """Place I/O repeaters and return (all repeater cells, output target lookup)."""
    io_repeater_cells: set[tuple[int, int, int]] = set()
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
            if c.type == ComponentType.OUTPUT_PIN:
                side_name = side.value
                endpoint = (c.id, pin.name)
                for key in (c.id, pin.name, f"{c.id}.{pin.name}"):
                    output_repeater_lookup[key] = (cell, side_name, endpoint)
    return io_repeater_cells, output_repeater_lookup


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
        inverted_cells: set[tuple[int, int, int]] | None = None,
) -> None:
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

    for pin_name in sorted(output_pin_targets, key=lambda pn: ext_target_positions[pn][1]):
        lookup = output_repeater_lookup[pin_name]
        repeater_cell, side_name, endpoint = lookup
        base_net_id = net_id_by_output_endpoint.get(endpoint)
        if base_net_id is None:
            raise ValueError(f"Output target '{pin_name}' has no mapped net")
        nx, ny, nz = _SIDE_NORMAL[side_name]
        launch = (repeater_cell[0] + nx, repeater_cell[1] + ny, repeater_cell[2] + nz)
        if not _in_bounds(launch):
            # Output repeaters can sit on workspace boundary; route from the in-bounds side in that case.
            launch = (repeater_cell[0] - nx, repeater_cell[1] - ny, repeater_cell[2] - nz)
        if not _in_bounds(launch):
            raise ValueError(f"No in-bounds launch cell for output target '{pin_name}'")
        tree_seeds = [
            pos for pos, owner in dust_owner.items()
            if owner == base_net_id and _is_redstone_wire(workspace[pos[0], pos[1], pos[2]])
        ]
        target = ext_target_positions[pin_name]
        protected = _compute_net_protected(base_net_id, all_terminals) if all_terminals else frozenset()
        # Other output pin targets and their source repeaters act as foreign terminals.
        other_zone: frozenset[tuple[int, int, int]] = frozenset()
        for other_pin, rc in repeater_cells_by_pin.items():
            if other_pin != pin_name:
                other_zone = other_zone | _repeater_protection_zone(rc)
                other_zone = other_zone | _repeater_protection_zone(ext_target_positions[other_pin])
        protected = protected | other_zone
        all_terminal_positions = (frozenset(
            pos for s in all_terminals.values() for pos in s) if all_terminals else frozenset()) | all_ext_targets
        inv_snap = frozenset(inverted_cells) if inverted_cells is not None else frozenset()
        path = _find_wire_path(
            workspace,
            solid,
            dust_owner,
            launch,
            target,
            base_net_id,
            ext_bounds,
            max_bridge_y=max_bridge_y,
            tree_seeds=tree_seeds,
            protected=protected,
            footprint_blocked=footprint_blocked,
            inverted_cells=inv_snap,
            terminal_positions=all_terminal_positions,
            limit_exclusions=True,
            x_h_factor=2
        )
        _lay_redstone_path(workspace, solid, dust_owner, path, base_net_id,
                           inverted_cells=inverted_cells,
                           goal=target,
                           terminal_positions=all_terminal_positions)
        _place_repeaters_for_net(workspace, dust_owner, base_net_id, [path])


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
) -> frozenset[tuple[int, int, int]]:
    """Hard-block zone: footprint interiors + expanded borders + IO repeater tops + const air-gaps."""
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


class _ForeignKeepout:
    """_compute_net_protected(net_id, ...) as a lazy view over a per-cell net count.

    Materializing every other net's keepout per net is quadratic in net count; this answers
    `pos in protected` (and `protected - cells`, as _find_wire_path uses it) without copying.
    """
    __slots__ = ("_count", "_own", "_excluded")

    def __init__(self, count: dict[tuple[int, int, int], int], own: set[tuple[int, int, int]],
                 excluded: frozenset[tuple[int, int, int]] = frozenset()):
        self._count = count
        self._own = own
        self._excluded = excluded

    def __contains__(self, pos) -> bool:
        n = self._count.get(pos, 0)
        if n == 0 or (n == 1 and pos in self._own):
            return False
        return pos not in self._excluded

    def __sub__(self, other) -> _ForeignKeepout:
        return _ForeignKeepout(self._count, self._own, self._excluded | frozenset(other))

    def __iter__(self):
        return (pos for pos in self._count if pos in self)


class _KeepoutMap:
    """Per-net terminal keepouts, computed once; for_net(n) == _compute_net_protected(n, ...)."""

    def __init__(self, all_terminals: dict[str, set[tuple[int, int, int]]]):
        self._own = {net_id: _terminal_keepout(t) for net_id, t in all_terminals.items()}
        self._count: dict[tuple[int, int, int], int] = {}
        for cells in self._own.values():
            for pos in cells:
                self._count[pos] = self._count.get(pos, 0) + 1

    def for_net(self, net_id: str) -> _ForeignKeepout:
        return _ForeignKeepout(self._count, self._own.get(net_id, set()))


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


def _route_all_nets(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        sorted_nets: list[NetConnection],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        ws_bounds: tuple[int, int, int, int, int, int],
        max_bridge_y: int,
        footprint_blocked: frozenset[tuple[int, int, int]],
        all_terminals: dict[str, set[tuple[int, int, int]]],
        snapshot_dir: Path | None = None,
        deadline: float | None = None,
        fail_fast: bool = False,
) -> tuple[list[tuple[str, Exception]], set[tuple[int, int, int]]]:
    """fail_fast: raise on the first net that fails instead of collecting failures."""
    total_nets = len(sorted_nets)
    routing_failures: list[tuple[str, Exception]] = []
    inverted_cells: set[tuple[int, int, int]] = set()
    keepouts = _KeepoutMap(all_terminals)
    region = workspace
    workspace = IndexedWorkspace(region)  # all writes below go through this, keeping its index current
    with ProgressBar("[wire]", total_nets) as bar:
        for net_idx, net in enumerate(sorted_nets, 1):
            bar.update(info=net.net_id)
            if deadline is not None and time.monotonic() > deadline:
                if fail_fast:
                    raise RuntimeError(f"Routing failed for net {net.net_id}: route deadline reached")
                routing_failures.append((net.net_id, ValueError("route deadline reached")))
                continue
            try:
                src_pin = _pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name)
                protected = keepouts.for_net(net.net_id)
                net_ordered_paths: list[list[tuple[int, int, int]]] = []
                dst_pins = (_pin_for_endpoint(pin_terminal, sink.component_id, sink.pin_name) for sink in net.sinks)
                sorted_pins = sorted(dst_pins,
                                     key=lambda p: abs(p[0] - src_pin[0]) + abs(p[1] - src_pin[1]) + abs(p[2] - src_pin[2]))
                for dst_pin in sorted_pins:
                    tree_seeds = [
                        pos for _path in net_ordered_paths for pos in _path
                    ]
                    all_terminal_positions = frozenset(pin_terminal.values())
                    path = _find_wire_path(
                        workspace, solid, dust_owner,
                        src_pin, dst_pin, net.net_id,
                        ws_bounds, max_bridge_y,
                        tree_seeds=tree_seeds,
                        protected=protected,
                        footprint_blocked=footprint_blocked,
                        inverted_cells=frozenset(inverted_cells),
                        terminal_positions=all_terminal_positions,
                    )
                    support_block = WOOLS[net_idx % len(WOOLS)]

                    _lay_redstone_path(workspace, solid, dust_owner, path, net.net_id,
                                       opaque_support_block=support_block,
                                       inverted_cells=inverted_cells,
                                       goal=dst_pin,
                                       terminal_positions=all_terminal_positions)
                    net_ordered_paths.append(path)
                _place_repeaters_for_net(workspace, dust_owner, net.net_id, net_ordered_paths)
            except Exception as e:
                if fail_fast:
                    raise RuntimeError(f"Routing failed for net {net.net_id}: {e}") from e
                routing_failures.append((net.net_id, e))
                log(f"[error] skipped net {net.net_id}: {e}")
            if snapshot_dir is not None and net_idx % 10 == 0:
                safe_id = net.net_id.replace("/", "_").replace("\\", "_")
                save_schematic_artifact(f"net_{net_idx:03d}_{safe_id}.litematic", region, snapshot_dir)
    print(f"[wire] done ({total_nets} nets, {len(routing_failures)} failed)")
    return routing_failures, inverted_cells


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

    keepouts = _KeepoutMap(all_terminals)

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
        gutter: int = 10,
        workspace_size: tuple[int, int, int] | None = None,
        base_y: int = 5,
        schematic_name: str = "build",
        component_width_spacing: int = 7,
        component_depth_spacing: int = 6,
        additional_component_height_spacing: int = 4,
        routing_headroom: int = 5,
        bridge_height: int = 1,
        allow_routing_failures: bool = False,
        output_pin_targets: dict[str, tuple[int, int]] | None = None,
        pin_targets_space=15,
        clock_pin: str = 'clk',
        route_passes: int = _ROUTE_MAX_PASSES,
        wire_timeout: float | None = _ROUTE_SEARCH_TIMEOUT,
        route_workers: int = 1,
        route_deadline: float | None = None,
        snapshots: bool = False,
) -> Path:
    comp = _expand_multibit_io(comp)
    max_comp_height = max((c.footprint.height for c in comp.components), default=1)
    y_stride = max_comp_height + additional_component_height_spacing
    y_level = _assign_component_y_levels(comp.components, comp.nets, y_stride=y_stride)
    placed = _layout_components(
        comp.components, comp.nets,
        gutter=gutter, base_y=base_y, y_level=y_level,
        io_margin=component_width_spacing, routing_gutter=component_depth_spacing,
    )
    if output_pin_targets:
        # Reserve z=0 for output extensions by shifting the existing build one layer forward.
        placed = [
            _Placed(component=p.component, origin=(p.origin[0], p.origin[1], p.origin[2] + pin_targets_space))
            for p in placed
        ]
    work_nets = comp.nets

    if workspace_size is None:
        width, height, depth = _compute_workspace_dims(
            placed, base_y=base_y, io_margin=component_width_spacing, routing_headroom=routing_headroom,
        )
    else:
        width, height, depth = workspace_size

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
    pin_world_map, pin_terminal = _compute_pin_maps(placed)
    io_repeater_cells, output_repeater_lookup = _place_io_repeaters(workspace, placed, solid, pin_world_map, io_bounds)
    output_component_ids = {c.id for c in comp.components if c.type == ComponentType.OUTPUT_PIN}
    net_id_by_output_endpoint: dict[tuple[str, str], str] = {
        (sink.component_id, sink.pin_name): net.net_id
        for net in work_nets
        for sink in net.sinks
        if sink.component_id in output_component_ids
    }
    footprint_blocked = _build_footprint_blocked(placed, pin_terminal, io_repeater_cells, const_positions)
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
        routing_failures, inverted_cells = _route_all_nets(
            workspace, solid, dust_owner,
            sorted_nets, pin_terminal, ws_bounds, max_bridge_y,
            footprint_blocked, all_terminals,
            snapshot_dir=snapshot_dir,
            deadline=deadline,
            fail_fast=not allow_routing_failures,
        )
    else:
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
            _route_output_pin_extensions(
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
            )
        except Exception as e:
            if allow_routing_failures:
                print(f"IGNORING - Pin target routing failed: {e}")
            else:
                raise RuntimeError(f"Pin target routing failed: {e}")
        else:
            if snapshot_dir is not None:
                save_schematic_artifact("extensions.litematic", workspace, snapshot_dir)

    _report_unsupported_blocks(workspace, dust_owner, frozenset(_component_interior(placed)))

    critical_path_ticks = _compute_critical_path(comp, dust_owner, workspace)

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
