from __future__ import annotations
import heapq
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from litemapy import BlockState, Region, Schematic
from minecraft_v.build_utils import save_artifact
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
GLASS = BlockState("minecraft:glass")
WOOLS = list(BlockState(f"minecraft:{color}_wool") for color in (
    'white',
    'orange',
    'magenta',
    'light_blue',
    'yellow',
    'lime',
    'pink',
    'gray',
    # 'light_gray',
    'cyan',
    'purple',
    'blue',
    'brown',
    'green',
    'red',
    'black'
))

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


def _is_repeater(block: BlockState) -> bool:
    return "minecraft:repeater" in _block_str(block)


def _is_torch(block: BlockState) -> bool:
    return "minecraft:redstone_torch" in _block_str(block)


REDSTONE_TORCH = BlockState("minecraft:redstone_torch")


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
        exempt_foreign: frozenset[tuple[int, int, int]] = frozenset(),
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
    # 3-wide foreign signal: no foreign-owned cell (wire, torch, or tower stone)
    # at horizontal cardinal neighbors at same Y or +-1 Y.
    for dx, dz in _HORIZ_DIRS:
        for dy in (-1, 0, 1):
            np = (x + dx, y + dy, z + dz)
            if np in exempt_foreign:
                continue
            if not _in_bounds(np, bounds):
                continue
            owner = dust_owner.get(np)
            if owner is not None and owner != net_id:
                return False
    # Vertical same-XZ: powered blocks distribute signal on all 6 faces including
    # top and bottom, so wire directly above or below a foreign-owned block is unsafe.
    for vdy in (1, -1):
        vp = (x, y + vdy, z)
        if vp in exempt_foreign:
            continue
        if not _in_bounds(vp, bounds):
            continue
        # block must be powered directly by a torch or repeater
        if not (any(
                _in_bounds((vp[0] + dx, vp[1], vp[2] + dz), bounds) and _is_repeater(workspace[vp[0] + dx, vp[1], vp[2] + dz]) for dx, dz in _HORIZ_DIRS
                # todo check repeater is pointing into block
        ) or (_in_bounds((vp[0], vp[1] - 1, vp[2]), bounds) and _is_torch(workspace[vp[0], vp[1] - 1, vp[2]]))):
            continue
        owner = dust_owner.get(vp)
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

    def walkable(pos: tuple[int, int, int], exempt: frozenset[tuple[int, int, int]] = frozenset(set())) -> bool:
        return pos not in protected and pos not in footprint_blocked and _wire_walkable(
            workspace, solid, dust_owner, pos, net_id, bounds, exempt_foreign=exempt
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
                    if ramp in solid or (_in_bounds(ramp, bounds) and _is_air(workspace[ramp[0], ramp[1], ramp[2]])):
                        # TODO: this discourages ramping so we get torch towers for vertical signal extension, but we should guarantee that vertical extension no matter what path is chosen
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
                exempt = set()
                # current dust might generate support, which would cut off connections to foreign dust below
                if _can_be_support(workspace, solid, (x, y - 1, z), bounds):
                    exempt.add((x, y - 2, z))
                if above_clear and walkable(down, frozenset(exempt)):
                    if y - 1 <= min_y or _can_be_support(
                            workspace, solid, (nx, y - 2, nz), bounds
                    ):
                        result.append((down, 2))

        # 2x slope moves: ±2 horizontal + ±2 vertical.
        # The intermediate cell (dx,±1,dz) is not an A* node but placed during laying.
        # Exempt cell: "2 below top" — one step horizontal + 1 below the intermediate,
        # which normally triggers the 3-wide foreign check on the intermediate.
        for dx, dz in _HORIZ_DIRS:
            nx2, nz2 = x + 2 * dx, z + 2 * dz
            mid = (x + dx, y + 1, z + dz)
            top = (nx2, y + 2, nz2)
            exempt = frozenset([(nx2, y, nz2)])
            if y + 2 <= max_bridge_y:
                mid_ok = (
                        mid not in protected
                        and mid not in footprint_blocked
                        and _in_bounds(mid, bounds)
                        and not (mid in solid)
                        and _is_air(workspace[mid[0], mid[1], mid[2]])
                        and _wire_walkable(workspace, solid, dust_owner, mid, net_id, bounds, exempt_foreign=exempt)
                )
                if mid_ok and walkable(top):
                    if _can_be_support(workspace, solid, (x + dx, y, z + dz), bounds):
                        result.append((top, 7))

            if y - 2 >= min_y:
                mid_dn = (x + dx, y - 1, z + dz)
                top_dn = (nx2, y - 2, nz2)
                exempt_dn = frozenset([(nx2, y, nz2)])
                above_mid = (x + dx, y, z + dz)
                above_clear = (
                        above_mid not in solid
                        and _in_bounds(above_mid, bounds)
                        and _is_air(workspace[above_mid[0], above_mid[1], above_mid[2]])
                )
                mid_dn_ok = (
                        above_clear
                        and mid_dn not in protected
                        and mid_dn not in footprint_blocked
                        and _in_bounds(mid_dn, bounds)
                        and not (mid_dn in solid)
                        and _is_air(workspace[mid_dn[0], mid_dn[1], mid_dn[2]])
                        and _wire_walkable(workspace, solid, dust_owner, mid_dn, net_id, bounds,
                                           exempt_foreign=exempt_dn)
                )
                if mid_dn_ok and walkable(top_dn):
                    if y - 2 <= min_y or _can_be_support(workspace, solid, (nx2, y - 3, nz2), bounds):
                        result.append((top_dn, 7))

        # +1Y from tower top: wire step up off the top stone of a tower.
        # Only valid when this position was reached via a tower move.
        _tower_2block_dirs = frozenset([(2, 0), (-2, 0), (0, 2), (0, -2)])
        parent = came_from.get(pos)
        if parent is not None:
            px, py, pz = parent
            if py + 4 == y and (x - px, z - pz) in _tower_2block_dirs:
                up = (x, y + 1, z)
                if y + 1 <= max_bridge_y and walkable(up):
                    result.append((up, 1))

        # Tower-up: 2 blocks to the side + 4 blocks up.
        # Layout: wire at (x,y,z) → repeater at (+1 side) → stone/torch column at (+2 side, +4 up).
        for tdx, tdz in _HORIZ_DIRS:
            if y + 4 > max_bridge_y:
                continue
            tower_top = (x + 2 * tdx, y + 4, z + 2 * tdz)
            if not walkable(tower_top):
                continue
            # Repeater cell (1 block over) must be free and have solid support below
            rep = (x + tdx, y, z + tdz)
            if not _in_bounds(rep, bounds) or rep in footprint_blocked or rep in protected or rep in came_from:
                continue
            if not _is_air(workspace[rep[0], rep[1], rep[2]]):
                continue
            if not _can_be_support(workspace, solid, (x + tdx, y - 1, z + tdz), bounds):
                continue
            # Tower column (2 blocks over, y..y+4) must be free
            col_clear = True
            for cdy in range(5):
                cp = (x + 2 * tdx, y + cdy, z + 2 * tdz)
                if not _in_bounds(cp, bounds) or cp in footprint_blocked or cp in protected:
                    col_clear = False
                    break
                if not _is_air(workspace[cp[0], cp[1], cp[2]]):
                    col_clear = False
                    break
            if not col_clear:
                continue
            # Tower base will be powered → check cell below base has no foreign wire
            below_base = (x + 2 * tdx, y - 1, z + 2 * tdz)
            if _in_bounds(below_base, bounds):
                below_owner = dust_owner.get(below_base)
                if below_owner is not None and below_owner != net_id:
                    continue
            result.append((tower_top, 8))

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
    explored = set()
    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        explored.add(current)
        if g > g_score.get(current, 10 ** 9):
            continue
        if current in goal_set:
            reached = current
            break
        for neighbor, cost in neighbors(current):
            new_g = g + cost
            if new_g < g_score.get(neighbor, 10 ** 9):
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


def _place_support(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        net_id: str,
        cell: tuple[int, int, int],
        opaque_support_block: BlockState = STONE
) -> None:
    x, y, z = cell
    if y <= 0:
        return
    below = (x, y - 1, z)
    if below in solid or not _is_air(workspace[x, y - 1, z]):
        return
    is_stair = (
            y >= 2
            and _is_redstone_wire(workspace[x, y - 2, z])
            and dust_owner.get((x, y - 2, z)) == net_id
            and any(_is_redstone_wire(workspace[x + dx, y - 1, z + dz]) for dx, dz in _HORIZ_DIRS)
    )
    workspace[x, y - 1, z] = GLASS if is_stair else opaque_support_block
    solid.add(below)


def _lay_redstone_path(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        path: Iterable[tuple[int, int, int]],
        net_id: str,
        opaque_support_block: BlockState = STONE
) -> None:
    cells = list(path)
    n = len(cells)

    # Identify tower moves: cell[i] → cell[i+1] is 2 blocks to the side + 4 up.
    _tower_2block = frozenset([(2, 0), (-2, 0), (0, 2), (0, -2)])
    tower_bottom: set[int] = set()
    tower_top: set[int] = set()
    for i in range(n - 1):
        ax, ay, az = cells[i]
        bx, by, bz = cells[i + 1]
        if by == ay + 4 and (bx - ax, bz - az) in _tower_2block:
            tower_bottom.add(i)
            tower_top.add(i + 1)

    # Identify 2x slope moves: cell[i] → cell[i+1] is 2 blocks to the side + ±2 in Y.
    slope2_bottom: set[int] = set()
    for i in range(n - 1):
        ax, ay, az = cells[i]
        bx, by, bz = cells[i + 1]
        if abs(by - ay) == 2 and (bx - ax, bz - az) in _tower_2block:
            slope2_bottom.add(i)

    for i, (x, y, z) in enumerate(cells):
        if i in tower_top:
            # Stone already placed at this position by the tower build — skip.
            continue

        if i in slope2_bottom:
            bx, by, bz = cells[i + 1]
            sdx = (bx - x) // 2
            sdz = (bz - z) // 2
            sdy = (by - y) // 2  # +1 or -1
            mid = (x + sdx, y + sdy, z + sdz)

            # Launch cell: regular wire
            existing = workspace[x, y, z]
            if _is_redstone_wire(existing) or _is_torch(existing):
                owner = dust_owner.get((x, y, z))
                if owner is not None and owner != net_id:
                    raise ValueError(f"Dust collision at {(x, y, z)} for nets {owner} vs {net_id}")
            else:
                workspace[x, y, z] = REDSTONE
                dust_owner[(x, y, z)] = net_id
                _place_support(workspace, solid, dust_owner, net_id, (x, y, z), opaque_support_block)

            # Intermediate cell: redstone dust + support (glass only if stair pattern detected)
            existing_mid = workspace[mid[0], mid[1], mid[2]]
            if _is_redstone_wire(existing_mid) or _is_torch(existing_mid):
                owner = dust_owner.get(mid)
                if owner is not None and owner != net_id:
                    raise ValueError(f"Dust collision at {mid} for nets {owner} vs {net_id}")
            else:
                workspace[mid[0], mid[1], mid[2]] = REDSTONE
                dust_owner[mid] = net_id
                _place_support(workspace, solid, dust_owner, net_id, mid, opaque_support_block)
            # Top cell falls through to the regular else handler below on next iteration.

        elif i in tower_bottom:
            bx, by, bz = cells[i + 1]
            # Unit direction toward tower (cells[i+1] is 2 blocks over)
            tdx = (bx - x) // 2
            tdz = (bz - z) // 2

            # Launch cell: regular wire
            existing = workspace[x, y, z]
            if _is_redstone_wire(existing) or _is_torch(existing):
                owner = dust_owner.get((x, y, z))
                if owner is not None and owner != net_id:
                    raise ValueError(f"Dust collision at {(x, y, z)} for nets {owner} vs {net_id}")
            else:
                workspace[x, y, z] = REDSTONE
                dust_owner[(x, y, z)] = net_id
                _place_support(workspace, solid, dust_owner, net_id, (x, y, z), opaque_support_block)

            # Repeater 1 block to the side: input faces back toward wire (opposite of tdx/tdz)
            rx, ry, rz = x + tdx, y, z + tdz
            facing = _DELTA_TO_FACING[(-tdx, -tdz)]
            workspace[rx, ry, rz] = BlockState("minecraft:repeater", facing=facing, delay="1")
            dust_owner[(rx, ry, rz)] = net_id
            _place_support(workspace, solid, dust_owner, net_id, (rx, ry, rz), opaque_support_block)

            # Tower column 2 blocks to the side: stone-torch-stone-torch-stone
            col = [
                (bx, y, bz, opaque_support_block),
                (bx, y + 1, bz, REDSTONE_TORCH),
                (bx, y + 2, bz, opaque_support_block),
                (bx, y + 3, bz, REDSTONE_TORCH),
                (bx, y + 4, bz, opaque_support_block),
            ]
            for cx, cy, cz, blk in col:
                existing_owner = dust_owner.get((cx, cy, cz))
                if existing_owner is not None and existing_owner != net_id:
                    raise ValueError(
                        f"Tower collision at {(cx, cy, cz)} for nets {existing_owner} vs {net_id}"
                    )
                if existing_owner == net_id:
                    continue
                workspace[cx, cy, cz] = blk
                if blk is STONE:
                    solid.add((cx, cy, cz))
                dust_owner[(cx, cy, cz)] = net_id

        else:
            existing = workspace[x, y, z]
            if _is_redstone_wire(existing) or _is_torch(existing):
                owner = dust_owner.get((x, y, z))
                if owner is not None and owner != net_id:
                    raise ValueError(
                        f"Dust collision at {(x, y, z)} for nets {owner} vs {net_id}"
                    )
                continue
            workspace[x, y, z] = REDSTONE
            dust_owner[(x, y, z)] = net_id
            _place_support(workspace, solid, dust_owner, net_id, (x, y, z), opaque_support_block)


def _place_repeaters_for_net(
        workspace: Region,
        dust_owner: dict[tuple[int, int, int], str],
        torch_cells: set[tuple[int, int, int]],
        net_id: str,
        source: tuple[int, int, int],
) -> None:
    """DFS from source through this net's wire tree, placing repeaters as needed.

    Carries (path_from_source, reset_index) per branch so fan-out points correctly
    inherit their signal distance rather than assuming full strength.

    Torches and existing repeaters are treated as full-power resets (signal = 15).
    Tower blocks (stone cells in dust_owner) are traversed via pure-vertical neighbors.
    """

    def wire_neighbors(pos: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        x, y, z = pos
        result = []
        # Horizontal + ±1 vertical (slope connections)
        for dx, dz in _HORIZ_DIRS:
            for dy in (0, 1, -1):
                nb = (x + dx, y + dy, z + dz)
                if dust_owner.get(nb) == net_id:
                    result.append(nb)
        # Pure vertical: torch/tower-block stacking (same XZ)
        for dy in (1, -1, 2, -2):
            nb = (x, y + dy, z)
            if dust_owner.get(nb) == net_id:
                result.append(nb)
        return result

    def is_viable(path: list[tuple[int, int, int]], j: int) -> bool:
        if j <= 0:
            return False
        x, y, z = path[j]
        px, py, pz = path[j - 1]
        if y != py:
            return False  # slope or tower — not a viable repeater position
        dx, dz = x - px, z - pz
        if (dx, dz) not in _DELTA_TO_FACING:
            return False
        # Check ALL forward wire neighbors (not just current DFS branch's next cell).
        # A repeater has one output direction — branch points are not viable.
        fwd = [nb for nb in wire_neighbors((x, y, z)) if nb != (px, py, pz)]
        if len(fwd) > 1:
            return False  # branch point
        if len(fwd) == 1:
            nx, ny, nz = fwd[0]
            if ny != y:
                return False  # slope or tower after repeater
            if (nx - x, nz - z) != (dx, dz):
                return False  # turn after repeater
        return True

    # DFS stack: (pos, path_from_source_to_pos, reset_index_in_path)
    # Full path stored per branch so fan-out branches don't share mutable state.
    visited: set[tuple[int, int, int]] = {source}
    stack: list[tuple[tuple[int, int, int], list[tuple[int, int, int]], int]] = [
        (source, [source], 0)
    ]
    while stack:
        pos, path, reset_idx = stack.pop()
        # Torch or existing repeater: signal resets to full power here.
        px_, py_, pz_ = pos
        cell_block = workspace[px_, py_, pz_]
        if pos in torch_cells or _is_repeater(cell_block) or _is_torch(cell_block):
            reset_idx = len(path) - 1
        depth = len(path) - 1
        dist = depth - reset_idx
        new_reset = reset_idx
        if dist >= _REPEATER_INTERVAL:
            cap = min(depth - 1, reset_idx + _REPEATER_INTERVAL)
            for j in range(cap, reset_idx, -1):
                if is_viable(path, j):
                    rx, ry, rz = path[j]
                    rpx, _, rpz = path[j - 1]
                    facing = _OPPOSITE_SIDE[_DELTA_TO_FACING[(rx - rpx, rz - rpz)]]
                    if not _is_repeater(workspace[rx, ry, rz]):
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


_BIT_INDEX_RE = re.compile(r'\[(\d+)\]$')
_BIT_NAME_RE = re.compile(r'^(.*)\[(\d+)\]$')


def _io_bit_index(c: Component) -> int | None:
    """Return bit index from an IO component's ID (e.g. 'a[2]' → 2), or None."""
    if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
        return None
    m = _BIT_INDEX_RE.search(c.id)
    return int(m.group(1)) if m else None


def _majority_y(ys: list[int]) -> int:
    """Plurality-vote Y level; tiebreak = max Y."""
    count: dict[int, int] = defaultdict(int)
    for y in ys:
        count[y] += 1
    max_count = max(count.values())
    # TODO: tiebreak to lowest Y value
    return min(y for y, c in count.items() if c == max_count)


def _assign_component_y_levels(
        components: list[Component],
        nets: list[NetConnection],
        *,
        y_stride: int,
) -> dict[str, int]:
    """Return Y offset per component ID.

    Indexed IO (e.g. ``a[2]``) → bit_index * y_stride (fixed).
    Gates → least-loaded Y among candidate levels drawn from both input
    sources and output sinks already assigned. Using both directions lets
    gates near the output boundary (e.g. DFFEs driving indexed outputs)
    land on the correct bit layer even when their inputs are unconstrained.
    Non-indexed IO (scalar shared) → majority Y of connected gates.
    """
    y_level: dict[str, int] = {}

    # Step 1: indexed IO → fixed layer
    for c in components:
        idx = _io_bit_index(c)
        if idx is not None:
            y_level[c.id] = idx * y_stride

    gates = [c for c in components if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)]
    gate_ids = {c.id for c in gates}

    net_source_cid: dict[str, str] = {}
    gate_input_nets: dict[str, list[str]] = defaultdict(list)
    gate_output_nets: dict[str, list[str]] = defaultdict(list)
    net_sink_cids: dict[str, list[str]] = defaultdict(list)
    for net in nets:
        src = net.source.component_id
        net_source_cid[net.net_id] = src
        if src in gate_ids:
            gate_output_nets[src].append(net.net_id)
        for sink in net.sinks:
            net_sink_cids[net.net_id].append(sink.component_id)
            if sink.component_id in gate_ids:
                gate_input_nets[sink.component_id].append(net.net_id)

    # Step 2: forward + backward passes.
    # Forward (depth 0 → max): gates near outputs (e.g. DFFEs) see indexed output
    # pin Y levels and spread correctly. Depth-0 gates have no assigned sinks yet.
    # Backward (depth max → 0): MUXes now see DFFE Y levels; FAs see MUX Y levels.
    # gate_count_per_y is rebuilt each pass so load balancing stays accurate.
    #
    # Use only IO-connected nets for depth ordering. Purely gate-to-gate nets like
    # carry chains (FA[i].Cout → FA[i+1].Cin) cross bit-layer boundaries and corrupt
    # the processing order, causing gates to be pulled to the wrong Y level.
    io_ids = {c.id for c in components if c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)}
    io_nets = [n for n in nets if n.source.component_id in io_ids or any(s.component_id in io_ids for s in n.sinks)]
    depths = _build_dependency_layers(gates, io_nets)
    max_depth = max(depths.values(), default=0)
    gates_by_depth: dict[int, list[Component]] = defaultdict(list)
    for c in gates:
        gates_by_depth[depths[c.id]].append(c)

    def _run_pass(depth_order: list[int]) -> None:
        gate_count_per_y: dict[int, int] = defaultdict(int)
        for existing_y in y_level.values():
            gate_count_per_y[existing_y] += 0  # ensure key exists; actual counts come from gates
        # Seed counts from already-assigned gates so load balancing is accurate.
        for c in gates:
            if c.id in y_level:
                gate_count_per_y[y_level[c.id]] += 1
        for d in depth_order:
            for c in gates_by_depth[d]:
                # Remove current contribution before reassigning.
                if c.id in y_level:
                    gate_count_per_y[y_level[c.id]] -= 1
                input_ys = [
                    y_level[net_source_cid[nid]]
                    for nid in gate_input_nets[c.id]
                    if net_source_cid[nid] in y_level
                ]
                output_ys = [
                    y_level[sink_cid]
                    for nid in gate_output_nets[c.id]
                    for sink_cid in net_sink_cids[nid]
                    if sink_cid in y_level
                ]
                candidates = set(input_ys + output_ys)
                assigned = min(candidates, key=lambda y: gate_count_per_y[y]) if candidates else 0
                y_level[c.id] = assigned
                gate_count_per_y[assigned] += 1

    _run_pass(list(range(max_depth + 1)))  # forward
    _run_pass(list(range(max_depth, -1, -1)))  # backward

    # Step 3: non-indexed IO (e.g. scalar `b`) → majority Y of connected gates
    for c in components:
        if c.id in y_level:
            continue
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
            continue
        connected_ys: list[int] = []
        for net in nets:
            if net.source.component_id == c.id:
                for sink in net.sinks:
                    if sink.component_id in y_level:
                        connected_ys.append(y_level[sink.component_id])
            else:
                for sink in net.sinks:
                    if sink.component_id == c.id:
                        src = net.source.component_id
                        if src in y_level:
                            connected_ys.append(y_level[src])
        y_level[c.id] = _majority_y(connected_ys) if connected_ys else 0

    return y_level


def _build_dependency_layers(
        gates: list[Component],
        nets: list[NetConnection],
) -> dict[str, int]:
    """Return depth per gate: 0 = driven directly from inputs, N = N hops into graph.

    Cycles are broken by removing back edges detected during DFS.
    """
    gate_ids = {c.id for c in gates}

    successors: dict[str, set[str]] = {c.id: set() for c in gates}
    predecessors: dict[str, set[str]] = {c.id: set() for c in gates}
    for net in nets:
        src = net.source.component_id
        for sink in net.sinks:
            dst = sink.component_id
            if src in gate_ids and dst in gate_ids and src != dst:
                successors[src].add(dst)
                predecessors[dst].add(src)

    # Iterative DFS — detect back edges (cycle edges) and remove them.
    # Sort gate_ids and successor lists for deterministic traversal regardless of hash seed.
    visited: set[str] = set()
    in_stack: set[str] = set()
    for start in sorted(gate_ids):
        if start in visited:
            continue
        stack: list[tuple[str, list[str], int]] = [(start, sorted(successors[start]), 0)]
        in_stack.add(start)
        visited.add(start)
        while stack:
            node, nbrs, idx = stack[-1]
            if idx < len(nbrs):
                stack[-1] = (node, nbrs, idx + 1)
                nb = nbrs[idx]
                if nb not in visited:
                    visited.add(nb)
                    in_stack.add(nb)
                    stack.append((nb, sorted(successors[nb]), 0))
                elif nb in in_stack:
                    # Back edge — remove to break cycle.
                    successors[node].discard(nb)
                    predecessors[nb].discard(node)
            else:
                in_stack.discard(node)
                stack.pop()

    # depth = longest path from any root.
    in_deg = {gid: len(predecessors[gid]) for gid in gate_ids}
    depth: dict[str, int] = {gid: 0 for gid in gate_ids}
    queue: deque[str] = deque(sorted(gid for gid in gate_ids if in_deg[gid] == 0))
    while queue:
        node = queue.popleft()
        for nb in successors[node]:
            depth[nb] = max(depth[nb], depth[node] + 1)
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    return depth


def _layout_components(
        components: list[Component],
        nets: list[NetConnection],
        *,
        gutter: int,
        base_y: int,
        io_margin: int = 4,
        routing_gutter: int = 10,
        y_level: dict[str, int] | None = None,
) -> list[_Placed]:
    inputs = [c for c in components if c.type == ComponentType.INPUT_PIN]
    outputs = [c for c in components if c.type == ComponentType.OUTPUT_PIN]
    gates = [
        c for c in components
        if c.type not in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
    ]

    def comp_y(c: Component) -> int:
        if y_level is not None and c.id in y_level:
            return base_y + y_level[c.id]
        return base_y

    # Compute per-Y-level local depths, ignoring cross-Y connections (e.g. carry
    # chains). This keeps each bit-layer's Z layout consistent: FA always at depth 0,
    # DFF at depth 1, regardless of which bit they serve.
    gates_by_y: dict[int, list[Component]] = defaultdict(list)
    for c in gates:
        gates_by_y[comp_y(c)].append(c)

    # Exclude register outputs from local depth nets: DFF/DFFE/DLATCH Q outputs
    # feed back into combinational inputs forming cycles. Keeping them would make
    # cycle-breaking arbitrary (DFS-order-dependent), inverting FA/DFF Z ordering
    # in some layers. Dropping them ensures combinational gates always precede
    # registers in the local depth order.
    _REGISTER_TYPES = {ComponentType.DFF, ComponentType.DFFE, ComponentType.DLATCH}
    _non_register_ids = {c.id for c in gates if c.type not in _REGISTER_TYPES}
    _feedforward_nets = [n for n in nets if n.source.component_id in _non_register_ids]

    local_depth: dict[str, int] = {}
    for y_gates in gates_by_y.values():
        # _build_dependency_layers only creates edges between gates in y_gates,
        # so cross-Y carry connections are naturally excluded.
        ld = _build_dependency_layers(y_gates, _feedforward_nets)
        local_depth.update(ld)

    max_depth = max(local_depth.values(), default=0)

    layer_gates: dict[int, list[Component]] = defaultdict(list)
    for c in gates:
        layer_gates[local_depth[c.id]].append(c)

    # Layout along Z: outputs at small Z, inputs at large Z.
    # Outputs at Z=1 so their repeaters land at Z=0 (workspace edge).
    # Gate layers ordered highest local depth (closest to outputs) → lowest.
    output_row_z = 1
    z_cursor = output_row_z + routing_gutter
    layer_z: dict[int, int] = {}
    for layer_idx in range(max_depth, -1, -1):
        layer_z[layer_idx] = z_cursor
        layer_depth = max((c.footprint.depth for c in layer_gates.get(layer_idx, [])), default=0)
        z_cursor += layer_depth + routing_gutter
    input_row_z = z_cursor

    placed: list[_Placed] = []

    # (local_depth, y) -> x cursor
    x_cursors: dict[tuple[int, int], int] = {}

    for c in outputs:
        y = comp_y(c)
        key = (max_depth + 1, y)
        cursor_x = x_cursors.get(key, io_margin)
        placed.append(_Placed(component=c, origin=(cursor_x, y, output_row_z)))
        x_cursors[key] = cursor_x + c.footprint.width + gutter

    for layer_idx in range(max_depth, -1, -1):
        z = layer_z[layer_idx]
        for c in layer_gates.get(layer_idx, []):
            y = comp_y(c)
            key = (layer_idx, y)
            cursor_x = x_cursors.get(key, io_margin)
            placed.append(_Placed(component=c, origin=(cursor_x, y, z)))
            x_cursors[key] = cursor_x + c.footprint.width + gutter

    for c in inputs:
        y = comp_y(c)
        key = (-1, y)
        cursor_x = x_cursors.get(key, io_margin)
        placed.append(_Placed(component=c, origin=(cursor_x, y, input_row_z)))
        x_cursors[key] = cursor_x + c.footprint.width + gutter

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
    depth = max_z + 1  # +1 so the I/O repeater fits at depth-1; shadow falls out of bounds
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
        comp.components,
        comp.nets,
        gutter=gutter,
        base_y=base_y,
        y_level=y_level,
        io_margin=io_margin,
        routing_gutter=routing_gutter,
    )
    work_nets = comp.nets
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
    dust_owner: dict[tuple[int, int, int], str] = {}  # dust coord to net id
    torch_cells: set[tuple[int, int, int]] = set()  # upright torch positions in towers
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

    # Place constant sources adjacent to const pins.
    # const_value='1' → redstone block (powered)
    # const_value='0' → iron trapdoor (signal blocker)
    REDSTONE_BLOCK = BlockState("minecraft:redstone_block")
    IRON_TRAPDOOR = BlockState("minecraft:iron_trapdoor", open="false", half="bottom", facing="north")
    _const_signal_positions: set[tuple[int, int, int]] = set()
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
                    _const_signal_positions.add((bx, by, bz))

    # Bridge layer is one fixed level above the highest placed component top.
    max_comp_top = base_y
    for item in placed:
        top = item.origin[1] + item.component.footprint.height - 1
        max_comp_top = max(max_comp_top, top)
    max_bridge_y = max_comp_top + bridge_height

    pin_world: dict[tuple[str, str], tuple[int, int, int]] = {}  # exact block coord of pin on component surface
    pin_terminal: dict[tuple[str, str], tuple[
        int, int, int]] = {}  # coord where wire connects; sided pins offset one step outward from surface
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

    _io_repeater_cells: set[tuple[int, int, int]] = set()
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
            _io_repeater_cells.add(cell)

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
    # Block cell directly above each I/O repeater.
    for rx, ry, rz in _io_repeater_cells:
        _fp_expanded.add((rx, ry + 1, rz))
    # Air-gap redstone blocks (const=1): block all 6 immediate neighbors so wiring
    # cannot be placed adjacent to a powered block and receive unintended signal.
    for rbx, rby, rbz in _const_signal_positions:
        for ddx, ddy, ddz in _DIRS_6:
            _fp_expanded.add((rbx + ddx, rby + ddy, rbz + ddz))
    # Exempt all pin terminals so routing can start/end at component edges.
    _fp_expanded -= set(pin_terminal.values())
    footprint_blocked: frozenset[tuple[int, int, int]] = frozenset(_fp_expanded)

    # Pre-compute all terminal positions per net for keepout calculation.
    all_terminals: dict[str, set[tuple[int, int, int]]] = {}
    _io_terminals = set()  # io terminals get additional keepouts
    for net in work_nets:
        t: set[tuple[int, int, int]] = set()
        source_coord = _pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name)
        t.add(source_coord)
        component = next(c for c in comp.components if c.id == net.source.component_id)
        if component.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
            _io_terminals.add(source_coord)
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
                    for dy in range(3):
                        cells.add((tx + dx, ty + dy, tz + dz))
                for dx, dz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    cells.add((tx + dx, ty, tz + dz))

        return frozenset(cells)

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

    def _net_sort_key(net: NetConnection) -> int:
        src = _pin_for_endpoint(pin_terminal, net.source.component_id, net.source.pin_name)
        return sum(
            abs(src[0] - d[0]) + abs(src[1] - d[1]) + abs(src[2] - d[2])
            for sink in net.sinks
            for d in [_pin_for_endpoint(pin_terminal, sink.component_id, sink.pin_name)]
        )

    sorted_nets = sorted(work_nets, key=_net_sort_key)
    total_nets = len(sorted_nets)
    routing_failures: list[tuple[str, Exception]] = []
    for net_idx, net in enumerate(sorted_nets, 1):
        print(f"\r[wire] {net_idx}/{total_nets} — {net.net_id:<40}", end="", flush=True)
        try:
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
                _lay_redstone_path(workspace, solid, dust_owner, path, net.net_id,
                                   opaque_support_block=WOOLS[net_idx % len(WOOLS)])
            # All sinks routed — now place repeaters with correct signal accounting
            # across the full wire tree (fan-out branches inherit distance from source).
            _place_repeaters_for_net(workspace, dust_owner, torch_cells, net.net_id, src_pin)
        except Exception as e:
            routing_failures.append((net.net_id, e))
            print(f"\n[error] skipped net {net.net_id}: {e}")
    print(f"\r[wire] done ({total_nets} nets, {len(routing_failures)} failed)        ")
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


def _pin_for_endpoint(
        pin_world: dict[tuple[str, str], tuple[int, int, int]],
        component_id: str,
        pin_name: str,
) -> tuple[int, int, int]:
    key = (component_id, pin_name)
    if key in pin_world:
        return pin_world[key]
    raise ValueError(f"Unknown pin endpoint {component_id}.{pin_name}")
