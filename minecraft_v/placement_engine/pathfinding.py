from __future__ import annotations
import heapq
from litemapy import BlockState, Region
from .constants import _HORIZ_DIRS, _DIRS_6, _TOWER_2BLOCK, GLASS, _ROUTE_MAX_NODES, _ROUTE_STAGNATION
from .block_utils import _is_air, _is_repeater, _is_torch, _is_redstone_wire


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
        inverted_cells: frozenset[tuple[int, int, int]] = frozenset(),
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
        if not (any(
                _in_bounds((vp[0] + dx, vp[1], vp[2] + dz), bounds)
                and _is_repeater(workspace[vp[0] + dx, vp[1], vp[2] + dz])
                for dx, dz in _HORIZ_DIRS
        ) or (
            _in_bounds((vp[0], vp[1] - 1, vp[2]), bounds)
            and _is_torch(workspace[vp[0], vp[1] - 1, vp[2]])
        )):
            continue
        owner = dust_owner.get(vp)
        if owner is not None and owner != net_id:
            return False
    # Inverted tower cells (torch y+1 and stone y+2) emit wrong-polarity quasi-power
    # on their 4 horizontal faces. Quasi-power doesn't propagate stairwise so only
    # check same-Y horizontal neighbors.
    for dx, dz in _HORIZ_DIRS:
        if (x + dx, y, z + dz) in inverted_cells:
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
        inverted_cells: frozenset[tuple[int, int, int]] = frozenset(),
) -> list[tuple[int, int, int]]:
    min_y = bounds[1]
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    current_path: set[tuple[int, int, int]] = set()

    # Strip the goal terminal's own keepout zone from protected so the approach
    # path is not blocked by the destination's own neighborhood.
    gx, gy, gz = goal
    goal_exclusion: set[tuple[int, int, int]] = {(gx, gy, gz)}
    for _dx, _dz in _HORIZ_DIRS:
        for _dy in range(-2, 3):
            goal_exclusion.add((gx + _dx, gy + _dy, gz + _dz))
    for _dx, _dz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        goal_exclusion.add((gx + _dx, gy, gz + _dz))
    effective_protected = protected - goal_exclusion

    # Shrink footprint blocking on the side the goal terminal is on: remove the
    # 1-cell expansion cells immediately adjacent to the goal (including +2Y for
    # the IO repeater top clearance) so slope/flat approaches can reach the terminal.
    goal_fp_relief: set[tuple[int, int, int]] = set()
    for _ddx, _ddy, _ddz in _DIRS_6:
        goal_fp_relief.add((gx + _ddx, gy + _ddy, gz + _ddz))
    goal_fp_relief.add((gx, gy + 2, gz))
    effective_footprint_blocked = footprint_blocked - goal_fp_relief

    def walkable(
            pos: tuple[int, int, int],
            exempt: frozenset[tuple[int, int, int]] = frozenset(),
    ) -> bool:
        return (
            pos not in effective_protected
            and pos not in effective_footprint_blocked
            and _wire_walkable(workspace, solid, dust_owner, pos, net_id, bounds, exempt_foreign=exempt, inverted_cells=inverted_cells)
        )

    def _horiz_neighbors(pos: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], int]]:
        x, y, z = pos
        result: list[tuple[tuple[int, int, int], int]] = []
        for dx, dz in _HORIZ_DIRS:
            nx, nz = x + dx, z + dz

            # Flat move
            flat = (nx, y, nz)
            if walkable(flat) and (nx, y + 1, nz) not in current_path:
                if y <= min_y or (
                    (nx, y - 1, nz) not in current_path
                    and _can_be_support(workspace, solid, (nx, y - 1, nz), bounds)
                ):
                    result.append((flat, 1))

            # Slope-up
            if y < max_bridge_y:
                up = (nx, y + 1, nz)
                if walkable(up):
                    ramp = (nx, y, nz)
                    if ramp not in current_path and (
                        ramp in solid or (_in_bounds(ramp, bounds) and _is_air(workspace[ramp[0], ramp[1], ramp[2]]))
                    ):
                        # TODO: this discourages ramping so we get torch towers for vertical signal extension, but we should guarantee that vertical extension no matter what path is chosen
                        result.append((up, 3))

            # Slope-down
            if y > min_y:
                down = (nx, y - 1, nz)
                above_down = (nx, y, nz)
                above_clear = (
                    above_down not in solid
                    and above_down not in current_path
                    and (nx, y + 1, nz) not in current_path
                    and _in_bounds(above_down, bounds)
                    and _is_air(workspace[above_down[0], above_down[1], above_down[2]])
                )
                exempt: set[tuple[int, int, int]] = set()
                # current dust might generate support, which would cut off connections to foreign dust below
                if _can_be_support(workspace, solid, (x, y - 1, z), bounds):
                    exempt.add((x, y - 2, z))
                if above_clear and walkable(down, frozenset(exempt)):
                    if y - 1 <= min_y or _can_be_support(workspace, solid, (nx, y - 2, nz), bounds):
                        result.append((down, 2))
        return result

    def _double_slope_neighbors(pos: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], int]]:
        x, y, z = pos
        result: list[tuple[tuple[int, int, int], int]] = []
        for dx, dz in _HORIZ_DIRS:
            nx2, nz2 = x + 2 * dx, z + 2 * dz
            exempt = frozenset([(nx2, y, nz2)])

            # 2x slope-up
            if y + 2 <= max_bridge_y:
                mid = (x + dx, y + 1, z + dz)
                top = (nx2, y + 2, nz2)
                mid_support = (x + dx, y, z + dz)
                mid_ok = (
                    mid not in protected
                    and mid not in footprint_blocked
                    and mid not in current_path
                    and _in_bounds(mid, bounds)
                    and mid not in solid
                    and _is_air(workspace[mid[0], mid[1], mid[2]])
                    and _wire_walkable(workspace, solid, dust_owner, mid, net_id, bounds, exempt_foreign=exempt)
                )
                if mid_ok and walkable(top):
                    if mid_support not in current_path and _can_be_support(workspace, solid, mid_support, bounds):
                        result.append((top, 7))

            # 2x slope-down
            if y - 2 >= min_y:
                mid_dn = (x + dx, y - 1, z + dz)
                top_dn = (nx2, y - 2, nz2)
                exempt_dn = frozenset([(nx2, y, nz2)])
                above_mid = (x + dx, y, z + dz)
                above_clear = (
                    above_mid not in solid
                    and above_mid not in current_path
                    and _in_bounds(above_mid, bounds)
                    and _is_air(workspace[above_mid[0], above_mid[1], above_mid[2]])
                )
                mid_dn_ok = (
                    above_clear
                    and mid_dn not in protected
                    and mid_dn not in footprint_blocked
                    and mid_dn not in current_path
                    and _in_bounds(mid_dn, bounds)
                    and mid_dn not in solid
                    and _is_air(workspace[mid_dn[0], mid_dn[1], mid_dn[2]])
                    and _wire_walkable(workspace, solid, dust_owner, mid_dn, net_id, bounds, exempt_foreign=exempt_dn)
                )
                if mid_dn_ok and walkable(top_dn):
                    if y - 2 <= min_y or _can_be_support(workspace, solid, (nx2, y - 3, nz2), bounds):
                        result.append((top_dn, 7))
        return result

    def _tower_neighbors(pos: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], int]]:
        x, y, z = pos
        result: list[tuple[tuple[int, int, int], int]] = []

        # +1Y from tower top: only valid when this position was reached via a tower move
        parent = came_from.get(pos)
        if parent is not None:
            px, py, pz = parent
            if py + 4 == y and (x - px, z - pz) in _TOWER_2BLOCK:
                up = (x, y + 1, z)
                if y + 1 <= max_bridge_y and walkable(up):
                    result.append((up, 1))

        # Tower-up: 2 blocks to the side + 4 blocks up
        for tdx, tdz in _HORIZ_DIRS:
            if y + 4 > max_bridge_y:
                continue
            tower_top = (x + 2 * tdx, y + 4, z + 2 * tdz)
            if not walkable(tower_top):
                continue
            rep = (x + tdx, y, z + tdz)
            if not _in_bounds(rep, bounds) or rep in footprint_blocked or rep in protected or rep in came_from:
                continue
            if not _is_air(workspace[rep[0], rep[1], rep[2]]):
                continue
            if not _can_be_support(workspace, solid, (x + tdx, y - 1, z + tdz), bounds):
                continue
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
            # Tower base will be powered — check cell below base has no foreign wire
            below_base = (x + 2 * tdx, y - 1, z + 2 * tdz)
            if _in_bounds(below_base, bounds):
                below_owner = dust_owner.get(below_base)
                if below_owner is not None and below_owner != net_id:
                    continue
            result.append((tower_top, 8))
        return result

    def neighbors(pos: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], int]]:
        result = _horiz_neighbors(pos)
        result.extend(_double_slope_neighbors(pos))
        result.extend(_tower_neighbors(pos))
        return result

    def heuristic(pos: tuple[int, int, int]) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[2] - goal[2]) + abs(pos[1] - goal[1]) * 2

    # Seed cells: prefer start itself; fall back to neighbors when blocked
    seeds: list[tuple[int, int, int]] = []
    if walkable(start):
        seeds.append(start)
    else:
        for ddx, ddy, ddz in _DIRS_6:
            cand = (start[0] + ddx, start[1] + ddy, start[2] + ddz)
            if walkable(cand):
                seeds.append(cand)
    # Fan-out tree routing: existing own-net wire cells are valid branch seeds
    if tree_seeds:
        seed_set = set(seeds)
        for ts in tree_seeds:
            if ts not in seed_set and _in_bounds(ts, bounds) and ts not in solid:
                seeds.append(ts)
                seed_set.add(ts)
    if not seeds:
        raise ValueError(f"No walkable cell near start {start} for net {net_id}")

    # Goal cells: goal exclusion already removed from effective_protected above,
    # so walkable() correctly admits the goal and its approach cells.
    goal_set: set[tuple[int, int, int]] = set()
    if walkable(goal):
        goal_set.add(goal)
    else:
        for ddx, ddy, ddz in _DIRS_6:
            cand = (goal[0] + ddx, goal[1] + ddy, goal[2] + ddz)
            if walkable(cand):
                goal_set.add(cand)
    if not goal_set:
        raise ValueError(f"No walkable cell near goal {goal} for net {net_id}")

    # A* search
    counter = 0
    open_heap: list[tuple[int, int, int, tuple[int, int, int]]] = []
    g_score: dict[tuple[int, int, int], int] = {}
    for seed in seeds:
        g_score[seed] = 0
        came_from[seed] = None
        heapq.heappush(open_heap, (heuristic(seed), 0, counter, seed))
        counter += 1

    reached: tuple[int, int, int] | None = None
    best_node: tuple[int, int, int] = seeds[0]
    best_h: int = min(heuristic(s) for s in seeds)
    explored = set()
    stagnation = 0
    early_stop_reason: str | None = None
    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        if g > g_score.get(current, 10 ** 9):
            continue
        h = heuristic(current)
        if h < best_h:
            best_h = h
            best_node = current
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= _ROUTE_STAGNATION:
                early_stop_reason = f"stagnated ({stagnation} expansions without progress)"
                break
        explored.add(current)
        if len(explored) >= _ROUTE_MAX_NODES:
            early_stop_reason = f"node cap ({_ROUTE_MAX_NODES})"
            break
        if current in goal_set:
            reached = current
            break
        current_path.clear()
        node: tuple[int, int, int] | None = current
        while node is not None:
            current_path.add(node)
            node = came_from.get(node)
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
        lamp = BlockState("minecraft:redstone_lamp")
        cur: tuple[int, int, int] | None = best_node
        while cur is not None:
            cx, cy, cz = cur
            workspace[cx, cy, cz] = lamp
            cur = came_from.get(cur)
        for coord in sorted(explored, key=heuristic)[:100]:
            if _is_air(workspace[*coord]):
                workspace[*coord] = GLASS
        reason = f"; {early_stop_reason}" if early_stop_reason else ""
        raise ValueError(f"No route for net {net_id} from {start} to {goal} (closest reached: {best_node}{reason})")
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
