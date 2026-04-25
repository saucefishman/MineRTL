from __future__ import annotations

from pathlib import Path

from litemapy import BlockState, Region, Schematic

from minecraft_v.build_utils import save_artifact
from minecraft_v.cell_library import SCHEMATIC_MAP
from minecraft_v.placement_engine.ir import (
    ComponentList,
    ComponentType,
    NetConnection,
)
from .block_utils import _is_air, _ensure_support
from .constants import (
    _HORIZ_DIRS, _DIRS_6, _SIDE_NORMAL, WOOLS,
)
from .layout import (
    _Placed, _expand_multibit_io, _assign_component_y_levels,
    _layout_components, _compute_workspace_dims,
)
from .pathfinding import _find_wire_path
from .template import (
    _pin_world, _io_repeater_cell, _io_repeater_facing,
    _default_io_side, _paste_template, _load_template_region,
)
from .wire import _lay_redstone_path, _place_repeaters_for_net

REDSTONE_BLOCK = BlockState("minecraft:redstone_block")
IRON_TRAPDOOR = BlockState("minecraft:iron_trapdoor", open="false", half="bottom", facing="north")


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
        _paste_template(workspace, template, item.origin, solid)
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
            if pin.const_value not in ('0', '1'):
                continue
            pw = _pin_world(item.origin, pin, c.footprint)
            side = pin.side
            if side is not None:
                nx, ny, nz = _SIDE_NORMAL[side.value]
                bx, by, bz = pw[0] + nx, pw[1] + ny, pw[2] + nz
            else:
                bx, by, bz = pw
            if _is_air(workspace[bx, by, bz]):
                block = REDSTONE_BLOCK if pin.const_value == '1' else IRON_TRAPDOOR
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
) -> set[tuple[int, int, int]]:
    """Place I/O repeaters for input/output pins. Returns set of placed repeater cells."""
    io_repeater_cells: set[tuple[int, int, int]] = set()
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
            solid.add(cell)
            _ensure_support(workspace, solid, cell)
            io_repeater_cells.add(cell)
    return io_repeater_cells


def _build_footprint_blocked(
        placed: list[_Placed],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        io_repeater_cells: set[tuple[int, int, int]],
        const_positions: set[tuple[int, int, int]],
) -> frozenset[tuple[int, int, int]]:
    """Hard-block zone: footprint interiors + expanded borders + IO repeater tops + const air-gaps."""
    fp_interior: set[tuple[int, int, int]] = {
        (ox + dx, oy + dy, oz + dz)
        for item in placed
        if item.component.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
        for ox, oy, oz in [item.origin]
        for fp in [item.component.footprint]
        for dx in range(fp.width)
        for dy in range(fp.height)
        for dz in range(fp.depth)
    }
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
        if other_id == net_id:
            continue
        for tx, ty, tz in terminals:
            cells.add((tx, ty, tz))
            for dx, dz in _HORIZ_DIRS:
                for dy in range(3):
                    cells.add((tx + dx, ty + dy, tz + dz))
            for dx, dz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                cells.add((tx + dx, ty, tz + dz))
    return frozenset(cells)


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
        torch_cells: set[tuple[int, int, int]],
        sorted_nets: list[NetConnection],
        pin_terminal: dict[tuple[str, str], tuple[int, int, int]],
        ws_bounds: tuple[int, int, int, int, int, int],
        max_bridge_y: int,
        footprint_blocked: frozenset[tuple[int, int, int]],
        all_terminals: dict[str, set[tuple[int, int, int]]],
) -> list[tuple[str, Exception]]:
    total_nets = len(sorted_nets)
    routing_failures: list[tuple[str, Exception]] = []
    for net_idx, net in enumerate(sorted_nets, 1):
        print(f"\r[wire] {net_idx}/{total_nets} — {net.net_id:<40}", end="", flush=True)
        try:
            src_pin = _pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name)
            protected = _compute_net_protected(net.net_id, all_terminals)
            for sink in net.sinks:
                dst_pin = _pin_for_endpoint(pin_terminal, sink.component_id, sink.pin_name)
                tree_seeds = [pos for pos, owner in dust_owner.items() if owner == net.net_id]
                path = _find_wire_path(
                    workspace, solid, dust_owner,
                    src_pin, dst_pin, net.net_id,
                    ws_bounds, max_bridge_y,
                    tree_seeds=tree_seeds,
                    protected=protected,
                    footprint_blocked=footprint_blocked,
                )
                _lay_redstone_path(workspace, solid, dust_owner, path, net.net_id,
                                   opaque_support_block=WOOLS[net_idx % len(WOOLS)])
            _place_repeaters_for_net(workspace, dust_owner, torch_cells, net.net_id, src_pin)
        except Exception as e:
            routing_failures.append((net.net_id, e))
            print(f"\n[error] skipped net {net.net_id}: {e}")
    print(f"\r[wire] done ({total_nets} nets, {len(routing_failures)} failed)        ")
    return routing_failures


def build_litematic_from_component_list(
        comp: ComponentList,
        schematics_dir: Path,
        out_path: Path,
        *,
        gutter: int = 10,
        workspace_size: tuple[int, int, int] | None = None,
        base_y: int = 1,
        schematic_name: str = "build",
        io_margin: int = 10,
        routing_gutter: int = 10,
        routing_headroom: int = 5,
        bridge_height: int = 1,
        allow_routing_failures: bool = False,
) -> Path:
    comp = _expand_multibit_io(comp)
    max_comp_height = max((c.footprint.height for c in comp.components), default=1)
    y_stride = max_comp_height + 5
    y_level = _assign_component_y_levels(comp.components, comp.nets, y_stride=y_stride)
    placed = _layout_components(
        comp.components, comp.nets,
        gutter=gutter, base_y=base_y, y_level=y_level,
        io_margin=io_margin, routing_gutter=routing_gutter,
    )
    work_nets = comp.nets

    if workspace_size is None:
        width, height, depth = _compute_workspace_dims(
            placed, base_y=base_y, io_margin=io_margin, routing_headroom=routing_headroom,
        )
    else:
        width, height, depth = workspace_size

    workspace = Region(0, 0, 0, width, height, depth)
    solid: set[tuple[int, int, int]] = set()
    dust_owner: dict[tuple[int, int, int], str] = {}
    torch_cells: set[tuple[int, int, int]] = set()
    ws_bounds = (0, 0, 0, width - 1, height - 1, depth - 1)

    ref_mc_version, ref_lm_version, ref_lm_sub = _place_gate_templates(
        workspace, placed, solid, schematics_dir,
    )
    const_positions = _place_const_sources(workspace, placed, solid)
    max_bridge_y = _compute_max_bridge_y(placed, base_y, bridge_height)
    pin_world_map, pin_terminal = _compute_pin_maps(placed)
    io_repeater_cells = _place_io_repeaters(workspace, placed, solid, pin_world_map, ws_bounds)
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

    sorted_nets = sorted(work_nets, key=lambda net: _net_sort_key(net, pin_terminal))
    routing_failures = _route_all_nets(
        workspace, solid, dust_owner, torch_cells,
        sorted_nets, pin_terminal, ws_bounds, max_bridge_y,
        footprint_blocked, all_terminals,
    )

    if routing_failures:
        failed = ", ".join(nid for nid, _ in routing_failures)
        if allow_routing_failures:
            print(f"IGNORING - Routing failed for {len(routing_failures)} net(s): {failed}")
        else:
            raise RuntimeError(f"Routing failed for {len(routing_failures)} net(s): {failed}")

    schematic = workspace.as_schematic(
        name=schematic_name, author="minecraft-v", description="merged placement"
    )
    schematic.mc_version = ref_mc_version
    schematic.lm_version = ref_lm_version
    schematic.lm_subversion = ref_lm_sub
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schematic.save(str(out_path))
    return out_path
