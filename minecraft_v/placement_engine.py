from __future__ import annotations
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from litemapy import BlockState, Region, Schematic
from minecraft_v.cell_library import SCHEMATIC_MAP
from minecraft_v.placement_ir import (
    CardinalDirection,
    Component,
    ComponentList,
    ComponentType,
    Footprint,
    NetConnection,
    NetEndpoint,
    PinRef,
)

AIR = BlockState("minecraft:air")
STONE = BlockState("minecraft:stone")
REDSTONE = BlockState("minecraft:redstone_wire")

_REPEATER_INTERVAL = 15  # redstone signal range; repeater placed every N dust blocks

_HORIZ_DIRS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DELTA_TO_FACING: dict[tuple[int, int], str] = {
    (1, 0): CardinalDirection.EAST.value,
    (-1, 0): CardinalDirection.WEST.value,
    (0, 1): CardinalDirection.SOUTH.value,
    (0, -1): CardinalDirection.NORTH.value,
}
_DIRS_6: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


def _block_str(block: BlockState) -> str:
    return str(block)


def _is_air(block: BlockState) -> bool:
    text = _block_str(block)
    return text == "minecraft:air" or text.startswith("minecraft:air[")


def _is_redstone_wire(block: BlockState) -> bool:
    return "redstone_wire" in _block_str(block)


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
            template.min_x(), template.min_y(), template.min_z(),
            template.max_x(), template.max_y(), template.max_z(),
        )
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _needs_support(block: BlockState) -> bool:
    name = _block_str(block)
    return (
        "minecraft:repeater" in name
        or "minecraft:comparator" in name
        or "minecraft:redstone_wire" in name
        or "minecraft:redstone_torch[" in name
        or name == "minecraft:redstone_torch"
    )


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
    placed: list[tuple[tuple[int, int, int], BlockState]] = []
    for pos in template.block_positions():
        block = template[pos]
        if _is_air(block):
            continue
        x, y, z = pos
        wx, wy, wz = dx + (x - tmin_x), dy + (y - tmin_y), dz + (z - tmin_z)
        workspace[wx, wy, wz] = block
        solid.add((wx, wy, wz))
        placed.append(((wx, wy, wz), block))
    for cell, block in placed:
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
    if component_type == ComponentType.OUTPUT_PIN:
        return _OPPOSITE_SIDE[side.value]
    return side.value


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
        if not (min_x <= cx <= max_x and min_y <= cy <= max_y and min_z <= cz <= max_z):
            raise ValueError(
                f"IO repeater cell {(cx, cy, cz)} outside bounds {bounds}; "
                f"pin_world={pin_world} origin={origin} side={side.value}"
            )
    return (cx, cy, cz)


def _default_io_side(component: Component, pin: PinRef) -> CardinalDirection:
    if pin.side is not None:
        return pin.side
    if component.type == ComponentType.INPUT_PIN:
        return CardinalDirection.SOUTH
    if component.type == ComponentType.OUTPUT_PIN:
        return CardinalDirection.NORTH
    raise ValueError(f"Missing pin.side for {component.id} pin {pin.name}")


# ---------------------------------------------------------------------------
# Pathfinding
# ---------------------------------------------------------------------------

def _in_bounds(pos: tuple[int, int, int], bounds: tuple[int, int, int, int, int, int]) -> bool:
    x, y, z = pos
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    return min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z


def _can_be_support(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    pos: tuple[int, int, int],
    bounds: tuple[int, int, int, int, int, int],
) -> bool:
    """Return True if pos is already solid or can have stone placed there."""
    if not _in_bounds(pos, bounds):
        return False
    if pos in solid:
        return True
    return _is_air(workspace[pos[0], pos[1], pos[2]])


def _wire_walkable(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    pos: tuple[int, int, int],
    net_id: str,
    bounds: tuple[int, int, int, int, int, int],
) -> bool:
    x, y, z = pos
    if not _in_bounds(pos, bounds):
        return False
    if pos in solid:
        return False
    block = workspace[x, y, z]
    if not _is_air(block):
        if _is_redstone_wire(block):
            owner = dust_owner.get(pos)
            if owner is not None and owner != net_id:
                return False
        else:
            return False
    # 3-wide foreign redstone: no foreign wire at horizontal cardinal neighbors
    # at same Y or +-1 Y (covers slope connections in Minecraft).
    for dx, dz in _HORIZ_DIRS:
        for dy in (-1, 0, 1):
            np = (x + dx, y + dy, z + dz)
            if not _in_bounds(np, bounds):
                continue
            nb = workspace[np[0], np[1], np[2]]
            if _is_redstone_wire(nb):
                owner = dust_owner.get(np)
                if owner is not None and owner != net_id:
                    return False
    return True


def _find_wire_path(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    net_id: str,
    bounds: tuple[int, int, int, int, int, int],
    max_bridge_y: int,
    tree_seeds: list[tuple[int, int, int]] | None = None,
    protected: frozenset[tuple[int, int, int]] = frozenset(),
    footprint_blocked: frozenset[tuple[int, int, int]] = frozenset(),
) -> list[tuple[int, int, int]]:
    """A* pathfinder with Minecraft redstone movement model and bridge support.

    Movement:
      - Flat: wire at (x,y,z) -> (x+-1,y,z) or (x,y,z+-1); needs solid support below.
      - Slope-up: wire at (x,y,z) -> (nx,y+1,nz) where (nx,y,nz) is/can-be stone ramp;
        only if y < max_bridge_y.
      - Slope-down: wire at (x,y,z) -> (nx,y-1,nz); (nx,y,nz) must be non-solid so
        the wire can drape into the lower cell.

    Foreign redstone at a position blocks a 3-wide channel (itself + 1 each side)
    to prevent unintended connections.

    `protected` is a hard-block set of cells reserved for other nets' terminals
    (terminal + 1-cell horizontal buffer). Routing for this net will not enter
    these cells, keeping approach channels clear for other nets.
    """
    min_y = bounds[1]

    def walkable(pos: tuple[int, int, int]) -> bool:
        return pos not in protected and pos not in footprint_blocked and _wire_walkable(
            workspace, solid, dust_owner, pos, net_id, bounds
        )

    def neighbors(pos: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], int]]:
        x, y, z = pos
        result: list[tuple[tuple[int, int, int], int]] = []
        for dx, dz in _HORIZ_DIRS:
            nx, nz = x + dx, z + dz

            # Flat move
            flat = (nx, y, nz)
            if walkable(flat):
                if y <= min_y or _can_be_support(workspace, solid, (nx, y - 1, nz), bounds):
                    result.append((flat, 1))

            # Slope-up
            if y < max_bridge_y:
                up = (nx, y + 1, nz)
                if walkable(up):
                    ramp = (nx, y, nz)
                    if ramp in solid:
                        result.append((up, 2))
                    elif (_in_bounds(ramp, bounds)
                            and _is_air(workspace[ramp[0], ramp[1], ramp[2]])):
                        result.append((up, 3))

            # Slope-down
            if y > min_y:
                down = (nx, y - 1, nz)
                above_down = (nx, y, nz)
                above_clear = (
                    above_down not in solid
                    and _in_bounds(above_down, bounds)
                    and _is_air(workspace[above_down[0], above_down[1], above_down[2]])
                )
                if above_clear and walkable(down):
                    if y - 1 <= min_y or _can_be_support(
                        workspace, solid, (nx, y - 2, nz), bounds
                    ):
                        result.append((down, 2))

        return result

    def heuristic(pos: tuple[int, int, int]) -> int:
        # penalize vertical to prefer flat routes.
        return abs(pos[0] - goal[0]) + abs(pos[2] - goal[2]) + abs(pos[1] - goal[1]) * 2

    # Find walkable seeds near start. Prefer start itself; fall back to neighbors
    # only when start is blocked (e.g. occupied by a component block).
    seeds: list[tuple[int, int, int]] = []
    if walkable(start):
        seeds.append(start)
    else:
        for dx, dy, dz in _DIRS_6:
            cand = (start[0] + dx, start[1] + dy, start[2] + dz)
            if walkable(cand):
                seeds.append(cand)
    # Fan-out tree routing: existing own-net wire cells are valid branch seeds.
    # They bypass the 3-wide check since they're already committed own-net wire.
    if tree_seeds:
        seed_set = set(seeds)
        for pos in tree_seeds:
            if pos not in seed_set and _in_bounds(pos, bounds) and pos not in solid:
                seeds.append(pos)
                seed_set.add(pos)
    if not seeds:
        raise ValueError(f"No walkable cell near start {start} for net {net_id}")

    # Find walkable goal cells. Prefer goal itself; fall back to neighbors only
    # when goal is blocked (e.g. occupied by a component block).
    goal_set: set[tuple[int, int, int]] = set()
    if walkable(goal):
        goal_set.add(goal)
    else:
        for dx, dy, dz in _DIRS_6:
            cand = (goal[0] + dx, goal[1] + dy, goal[2] + dz)
            if walkable(cand):
                goal_set.add(cand)
    if not goal_set:
        raise ValueError(f"No walkable cell near goal {goal} for net {net_id}")

    # A* search — heap stores (f, g, tie_break, pos).
    counter = 0
    open_heap: list[tuple[int, int, int, tuple[int, int, int]]] = []
    g_score: dict[tuple[int, int, int], int] = {}
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for seed in seeds:
        g_score[seed] = 0
        came_from[seed] = None
        heapq.heappush(open_heap, (heuristic(seed), 0, counter, seed))
        counter += 1

    reached: tuple[int, int, int] | None = None
    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        if g > g_score.get(current, 10**9):
            continue
        if current in goal_set:
            reached = current
            break
        for neighbor, cost in neighbors(current):
            new_g = g + cost
            if new_g < g_score.get(neighbor, 10**9):
                g_score[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(
                    open_heap,
                    (new_g + heuristic(neighbor), new_g, counter, neighbor),
                )
                counter += 1

    if reached is None:
        raise ValueError(f"No route for net {net_id} from {start} to {goal}")
    if walkable(goal) and reached != goal:
        raise ValueError(f"No route reached goal {goal} for net {net_id}; stopped at {reached}")

    path: list[tuple[int, int, int]] = []
    cur: tuple[int, int, int] | None = reached
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    if not tree_seeds and path[0] != start:
        raise ValueError(f"Path for net {net_id} does not begin at start {start}; begins at {path[0]}")
    return path


def _lay_redstone_path(
    workspace: Region,
    solid: set[tuple[int, int, int]],
    dust_owner: dict[tuple[int, int, int], str],
    path: Iterable[tuple[int, int, int]],
    net_id: str,
) -> None:
    for x, y, z in path:
        existing = workspace[x, y, z]
        if _is_redstone_wire(existing):
            owner = dust_owner.get((x, y, z))
            if owner is not None and owner != net_id:
                raise ValueError(
                    f"Dust collision at {(x, y, z)} for nets {owner} vs {net_id}"
                )
        workspace[x, y, z] = REDSTONE
        dust_owner[(x, y, z)] = net_id
        # Stone support below; for slope-up this also serves as the ramp.
        if y > 0:
            below = (x, y - 1, z)
            if below not in solid and _is_air(workspace[x, y - 1, z]):
                workspace[x, y - 1, z] = STONE
                solid.add(below)


def _place_repeaters_for_net(
    workspace: Region,
    dust_owner: dict[tuple[int, int, int], str],
    net_id: str,
    source: tuple[int, int, int],
) -> None:
    """DFS from source through this net's wire tree, placing repeaters as needed.

    Carries (path_from_source, reset_index) per branch so fan-out points correctly
    inherit their signal distance rather than assuming full strength.
    """
    def wire_neighbors(pos: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        x, y, z = pos
        result = []
        for dx, dz in _HORIZ_DIRS:
            for dy in (0, 1, -1):
                nb = (x + dx, y + dy, z + dz)
                if dust_owner.get(nb) == net_id:
                    result.append(nb)
        return result

    def is_viable(path: list[tuple[int, int, int]], j: int) -> bool:
        if j <= 0:
            return False
        x, y, z = path[j]
        px, py, pz = path[j - 1]
        if y != py:
            return False  # slope
        if j < len(path):
            nx, ny, nz = path[j + 1]
            if ny != y:
                return False # slope
            if (x - px, z - pz) != (nx - x, nz - z):
                return False  # turn after repeater
        dx, dz = x - px, z - pz
        if (dx, dz) not in _DELTA_TO_FACING:
            return False
        return True

    # DFS stack: (pos, path_from_source_to_pos, reset_index_in_path)
    # Full path stored per branch so fan-out branches don't share mutable state.
    visited: set[tuple[int, int, int]] = {source}
    stack: list[tuple[tuple[int, int, int], list[tuple[int, int, int]], int]] = [
        (source, [source], 0)
    ]
    while stack:
        pos, path, reset_idx = stack.pop()
        depth = len(path) - 1
        dist = depth - reset_idx
        new_reset = reset_idx
        if dist >= _REPEATER_INTERVAL:
            cap = min(depth - 1, reset_idx + _REPEATER_INTERVAL)
            for j in range(cap, reset_idx, -1):
                if is_viable(path, j):
                    rx, ry, rz = path[j]
                    px, _, pz = path[j - 1]
                    facing = _OPPOSITE_SIDE[_DELTA_TO_FACING[(rx - px, rz - pz)]]
                    dust_owner.pop((rx, ry, rz), None)
                    workspace[rx, ry, rz] = BlockState(
                        "minecraft:repeater", facing=facing, delay="1"
                    )
                    new_reset = j
                    break
        for nb in wire_neighbors(pos):
            if nb not in visited:
                visited.add(nb)
                stack.append((nb, path + [nb], new_reset))


@dataclass
class _Placed:
    component: Component
    origin: tuple[int, int, int]


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
            new_id, new_name = renames[key]
            return NetEndpoint(component_id=new_id, pin_name=new_name)
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


def _layout_components(
    components: list[Component],
    *,
    gutter: int,
    base_y: int,
    io_margin: int = 4,
    routing_gutter: int = 10,
) -> list[_Placed]:
    inputs = [c for c in components if c.type == ComponentType.INPUT_PIN]
    outputs = [c for c in components if c.type == ComponentType.OUTPUT_PIN]
    gates = [
        c for c in components
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
    ]

    output_row_z = io_margin
    gate_row_z = output_row_z + routing_gutter + 1
    max_gate_depth = max((c.footprint.depth for c in gates), default=0)
    input_row_z = gate_row_z + max_gate_depth + routing_gutter

    placed: list[_Placed] = []
    for row, row_z in ((inputs, input_row_z), (gates, gate_row_z), (outputs, output_row_z)):
        cursor_x = io_margin
        for component in row:
            origin = (cursor_x, base_y, row_z)
            placed.append(_Placed(component=component, origin=origin))
            cursor_x += component.footprint.width + gutter
    return placed


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


def _load_template_region(schematics_dir: Path, prefix: str) -> Region:
    path = schematics_dir / f"{prefix}.litematic"
    if not path.is_file():
        raise FileNotFoundError(f"Missing schematic template: {path}")
    schematic = Schematic.load(str(path))
    if not schematic.regions:
        raise ValueError(f"Schematic has no regions: {path}")
    return next(iter(schematic.regions.values()))


def build_litematic_from_component_list(
    comp: ComponentList,
    schematics_dir: Path,
    out_path: Path,
    *,
    gutter: int = 6,
    workspace_size: tuple[int, int, int] | None = None,
    base_y: int = 0,
    schematic_name: str = "build",
    io_margin: int = 1,
    routing_gutter: int = 5,
    routing_headroom: int = 5,
    bridge_height: int = 1,
) -> Path:
    comp = _expand_multibit_io(comp)
    placed = _layout_components(
        comp.components,
        gutter=gutter,
        base_y=base_y,
        io_margin=io_margin,
        routing_gutter=routing_gutter,
    )
    if workspace_size is None:
        width, height, depth = _compute_workspace_dims(
            placed,
            base_y=base_y,
            io_margin=io_margin,
            routing_headroom=routing_headroom,
        )
    else:
        width, height, depth = workspace_size

    workspace = Region(0, 0, 0, width, height, depth)
    solid: set[tuple[int, int, int]] = set()
    dust_owner: dict[tuple[int, int, int], str] = {} # dust coord to net id
    ws_bounds = (0, 0, 0, width - 1, height - 1, depth - 1)

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

    # Bridge layer is one fixed level above the highest placed component top.
    max_comp_top = base_y
    for item in placed:
        top = item.origin[1] + item.component.footprint.height - 1
        max_comp_top = max(max_comp_top, top)
    max_bridge_y = max_comp_top + bridge_height

    pin_world: dict[tuple[str, str], tuple[int, int, int]] = {}  # exact block coord of pin on component surface
    pin_terminal: dict[tuple[str, str], tuple[int, int, int]] = {}  # coord where wire connects; sided pins offset one step outward from surface
    for item in placed:
        c = item.component
        origin = item.origin
        is_io = c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
        for pin in c.pins:
            pw = _pin_world(origin, pin, c.footprint)
            pin_world[(c.id, pin.name)] = pw
            if is_io:
                pin_terminal[(c.id, pin.name)] = pw
            else:
                side = pin.side
                if side is None:
                    pin_terminal[(c.id, pin.name)] = pw
                else:
                    nx, ny, nz = _SIDE_NORMAL[side.value]
                    pin_terminal[(c.id, pin.name)] = (pw[0] + nx, pw[1] + ny, pw[2] + nz)

    for item in placed:
        c = item.component
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
            continue
        origin = item.origin
        for pin in c.pins:
            side = _default_io_side(c, pin)
            pw = pin_world[(c.id, pin.name)]
            cell = _io_repeater_cell(pw, origin, c.footprint, side, bounds=ws_bounds)
            facing = _io_repeater_facing(c.type, side)
            workspace[cell[0], cell[1], cell[2]] = BlockState(
                "minecraft:repeater",
                facing=facing,
                delay="1",
            )
            solid.add(cell)
            _ensure_support(workspace, solid, cell)

    # All cells inside any component footprint, expanded 1 block in all directions.
    # Wire routing may not enter this zone. Pin terminals are exempted below.
    _fp_interior: set[tuple[int, int, int]] = {
        (ox + dx, oy + dy, oz + dz)
        for item in placed
        if item.component.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
        for ox, oy, oz in [item.origin]
        for fp in [item.component.footprint]
        for dx in range(fp.width)
        for dy in range(fp.height)
        for dz in range(fp.depth)
    }
    _fp_expanded: set[tuple[int, int, int]] = set(_fp_interior)
    for cx, cy, cz in _fp_interior:
        for ddx, ddy, ddz in _DIRS_6:
            _fp_expanded.add((cx + ddx, cy + ddy, cz + ddz))
        _fp_expanded.add((cx, cy + 2, cz))  # extra upward block: support placed below wire
        # Vertical diagonals (1 horiz + 1 vert): redstone signal coupling and
        # support-block placement can reach the footprint through these cells.
        for ddx, ddz in _HORIZ_DIRS:
            _fp_expanded.add((cx + ddx, cy + 1, cz + ddz))
            _fp_expanded.add((cx + ddx, cy - 1, cz + ddz))
    # Exempt all pin terminals so routing can start/end at component edges.
    _fp_expanded -= set(pin_terminal.values())
    footprint_blocked: frozenset[tuple[int, int, int]] = frozenset(_fp_expanded)

    # Pre-compute all terminal positions per net for keepout calculation.
    all_terminals: dict[str, set[tuple[int, int, int]]] = {}
    for net in comp.nets:
        t: set[tuple[int, int, int]] = set()
        t.add(_pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name))
        for sink in net.sinks:
            t.add(_pin_for_endpoint(pin_terminal, sink.component_id, sink.pin_name))
        all_terminals[net.net_id] = t

    def _net_protected(net_id: str) -> frozenset[tuple[int, int, int]]:
        """Cells (terminal + 1-cell horiz buffer) of every other net — hard-blocked
        so routing for net_id cannot surround another net's terminal approach."""
        cells: set[tuple[int, int, int]] = set()
        for other_id, terminals in all_terminals.items():
            if other_id == net_id:
                continue
            for tx, ty, tz in terminals:
                cells.add((tx, ty, tz))
                for dx, dz in _HORIZ_DIRS:
                    cells.add((tx + dx, ty, tz + dz))
        return frozenset(cells)

    for net in comp.nets:
        src_pin = _pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name)
        protected = _net_protected(net.net_id)
        for sink in net.sinks:
            dst_pin = _pin_for_endpoint(pin_terminal, sink.component_id, sink.pin_name)
            # Tree routing: subsequent sinks branch from the nearest existing wire cell
            # rather than always re-routing from the source terminal. This prevents the
            # fan-out path from traversing unrelated regions of the board.
            tree_seeds = [
                pos for pos, owner in dust_owner.items() if owner == net.net_id
            ]
            path = _find_wire_path(
                workspace, solid, dust_owner,
                src_pin, dst_pin, net.net_id,
                ws_bounds, max_bridge_y,
                tree_seeds=tree_seeds,
                protected=protected,
                footprint_blocked=footprint_blocked,
            )
            _lay_redstone_path(workspace, solid, dust_owner, path, net.net_id)
        # All sinks routed — now place repeaters with correct signal accounting
        # across the full wire tree (fan-out branches inherit distance from source).
        _place_repeaters_for_net(workspace, dust_owner, net.net_id, src_pin)

    schematic = workspace.as_schematic(
        name=schematic_name, author="minecraft-v", description="merged placement"
    )
    schematic.mc_version = ref_mc_version
    schematic.lm_version = ref_lm_version
    schematic.lm_subversion = ref_lm_sub

    out_path.parent.mkdir(parents=True, exist_ok=True)
    schematic.save(str(out_path))
    return out_path


def _pin_for_endpoint(
    pin_world: dict[tuple[str, str], tuple[int, int, int]],
    component_id: str,
    pin_name: str,
) -> tuple[int, int, int]:
    key = (component_id, pin_name)
    if key in pin_world:
        return pin_world[key]
    raise ValueError(f"Unknown pin endpoint {component_id}.{pin_name}")
