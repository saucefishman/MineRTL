from __future__ import annotations
from typing import Iterable
from litemapy import BlockState, Region
from .constants import (
    GLASS, STONE, REDSTONE, REDSTONE_TORCH,
    _HORIZ_DIRS, _DELTA_TO_FACING, _OPPOSITE_SIDE, _REPEATER_INTERVAL, _TOWER_2BLOCK,
)
from .block_utils import _is_air, _is_redstone_wire, _is_torch, _is_repeater, _block_str


def _place_support(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        net_id: str,
        cell: tuple[int, int, int],
        opaque_support_block: BlockState = STONE,
) -> bool: # True if placed support block or there was a solid, false if something got in the way
    x, y, z = cell
    if y <= 0:
        return True # assuming placed on ground
    below = (x, y - 1, z)
    if below in solid:
        return True
    elif not _is_air(workspace[x, y - 1, z]):
        return False
    is_stair = (
        y >= 2
        and _is_redstone_wire(workspace[x, y - 2, z])
        and dust_owner.get((x, y - 2, z)) == net_id
        and any(_is_redstone_wire(workspace[x + dx, y - 1, z + dz]) for dx, dz in _HORIZ_DIRS)
    )
    workspace[x, y - 1, z] = GLASS if is_stair else opaque_support_block
    solid.add(below)
    return True


def _lay_dust_cell(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        net_id: str,
        pos: tuple[int, int, int],
        opaque_support_block: BlockState,
) -> None:
    """Place a single redstone dust cell, raising on foreign collision."""
    x, y, z = pos
    existing = workspace[x, y, z]
    if _is_redstone_wire(existing) or _is_torch(existing):
        owner = dust_owner.get(pos)
        if owner is not None and owner != net_id:
            raise ValueError(f"Dust collision at {pos} for nets {owner} vs {net_id}")
        return
    elif not _is_air(existing):
        raise ValueError(f"Tried to redstone over an existing block ({existing}) at {pos}")
    workspace[x, y, z] = REDSTONE
    dust_owner[pos] = net_id
    _place_support(workspace, solid, dust_owner, net_id, pos, opaque_support_block)


def _lay_slope2_move(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        net_id: str,
        launch: tuple[int, int, int],
        top: tuple[int, int, int],
        opaque_support_block: BlockState,
) -> None:
    """Lay the launch and intermediate cells of a 2x-slope move. Top falls through to regular handler."""
    x, y, z = launch
    bx, by, bz = top
    sdx, sdz, sdy = (bx - x) // 2, (bz - z) // 2, (by - y) // 2
    mid = (x + sdx, y + sdy, z + sdz)
    _lay_dust_cell(workspace, solid, dust_owner, net_id, launch, opaque_support_block)
    _lay_dust_cell(workspace, solid, dust_owner, net_id, mid, opaque_support_block)


def _lay_powered_minus4_move(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        net_id: str,
        launch: tuple[int, int, int],
        dest: tuple[int, int, int],
        opaque_support_block: BlockState,
        inverted_cells: set[tuple[int, int, int]] | None = None,
        goal: tuple[int, int, int] | None = None,
        terminal_positions: frozenset[tuple[int, int, int]] = frozenset(),
) -> None:
    """Lay a powered -4 move via two wall torch inversions.

    Signal: wire→block(y-1)→inverted torch(nx,y-1)→inverted dust(nx,y-2)
            →unpowered block(nx,y-3)→active torch(x,y-3)→output dust(x,y-4).
    Cells (nx,y-1,nz) and (nx,y-2,nz) carry opposite-polarity signal and are
    added to inverted_cells so future routing avoids them.
    """
    x, y, z = launch  # dest == (x, y-4, z)

    # Find the valid side direction (verified clear during A*, should still be free)
    direction: tuple[int, int] | None = None
    for dx, dz in _HORIZ_DIRS:
        nx_c, nz_c = x + dx, z + dz
        side_cells = [
            (nx_c, y - 1, nz_c),
            (nx_c, y - 2, nz_c),
            (nx_c, y - 3, nz_c),
            (x,    y - 3, z),
            (nx_c, y,     nz_c),  # clearance above first torch
            (x,    y - 2, z),     # clearance above second torch
        ]
        if not all(
            _is_air(workspace[cx, cy, cz]) and (cx, cy, cz) not in solid
            for cx, cy, cz in side_cells
        ):
            continue
        if (goal is not None and (nx_c, y - 1, nz_c) == goal) \
                or (nx_c, y - 1, nz_c) in terminal_positions \
                or (nx_c, y - 2, nz_c) in terminal_positions \
                or (nx_c, y - 3, nz_c) in terminal_positions \
                or (x,    y - 3, z)    in terminal_positions \
                or (x,    y - 1, z)    in terminal_positions \
                or (nx_c, y,     nz_c) in terminal_positions \
                or (x,    y - 2, z)    in terminal_positions:
            continue
        # Mirror A* adjacency check: no foreign-owned wire at ±1 XZ ±1 Y of any
        # side cell — matches _powered_minus4_neighbors lines 378-387.
        adj_ok = True
        for cx, cy, cz in side_cells:
            for adx, adz in _HORIZ_DIRS:
                for ady in (-1, 0, 1):
                    sp = (cx + adx, cy + ady, cz + adz)
                    owner = dust_owner.get(sp)
                    if owner is not None and owner != net_id:
                        adj_ok = False
                        break
                if not adj_ok:
                    break
            if not adj_ok:
                break
        if not adj_ok:
            continue
        direction = (dx, dz)
        break
    if direction is None:
        raise ValueError(f"No valid direction for powered_minus4 at {launch}")
    tdx, tdz = direction
    nx, nz = x + tdx, z + tdz

    # Ensure opaque support at (x, y-1, z) — glass cannot conduct signal to torch
    below_launch = (x, y - 1, z)
    if below_launch not in solid:
        workspace[x, y - 1, z] = opaque_support_block
        solid.add(below_launch)
    elif _block_str(workspace[x, y - 1, z]) == "minecraft:glass":
        workspace[x, y - 1, z] = opaque_support_block
    dust_owner[below_launch] = net_id

    # Launch wire
    if _is_air(workspace[x, y, z]):
        workspace[x, y, z] = REDSTONE
        dust_owner[(x, y, z)] = net_id

    # Wall torch at (nx, y-1, nz) — inverted; attached to block at (x, y-1, z)
    facing1 = _DELTA_TO_FACING[(tdx, tdz)]
    workspace[nx, y - 1, nz] = BlockState("minecraft:redstone_wall_torch", facing=facing1)
    dust_owner[(nx, y - 1, nz)] = net_id
    if inverted_cells is not None:
        inverted_cells.add((nx, y - 1, nz))

    # Inverted dust at (nx, y-2, nz)
    workspace[nx, y - 2, nz] = REDSTONE
    dust_owner[(nx, y - 2, nz)] = net_id
    if inverted_cells is not None:
        inverted_cells.add((nx, y - 2, nz))

    # Support block at (nx, y-3, nz) — wall anchor for second torch; also supports dust above
    workspace[nx, y - 3, nz] = opaque_support_block
    solid.add((nx, y - 3, nz))
    dust_owner[(nx, y - 3, nz)] = net_id

    # Wall torch at (x, y-3, z) — active output; attached to block at (nx, y-3, nz)
    facing2 = _DELTA_TO_FACING[(-tdx, -tdz)]
    workspace[x, y - 3, z] = BlockState("minecraft:redstone_wall_torch", facing=facing2)
    dust_owner[(x, y - 3, z)] = net_id

    # Output dust at dest = (x, y-4, z)
    _lay_dust_cell(workspace, solid, dust_owner, net_id, dest, opaque_support_block)


def _lay_tower_move(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        net_id: str,
        launch: tuple[int, int, int],
        tower_top: tuple[int, int, int],
        opaque_support_block: BlockState,
        inverted_cells: set[tuple[int, int, int]] | None = None,
) -> None:
    """Lay launch wire, repeater, and stone/torch column for a tower move."""
    x, y, z = launch
    bx, by, bz = tower_top
    tdx = (bx - x) // 2
    tdz = (bz - z) // 2

    if _is_air(workspace[*launch]):
        _lay_dust_cell(workspace, solid, dust_owner, net_id, launch, opaque_support_block)

    rx, ry, rz = x + tdx, y, z + tdz
    facing = _DELTA_TO_FACING[(-tdx, -tdz)]
    workspace[rx, ry, rz] = BlockState("minecraft:repeater", facing=facing, delay="1")
    dust_owner[(rx, ry, rz)] = net_id
    placed_support = _place_support(workspace, solid, dust_owner, net_id, (rx, ry, rz), opaque_support_block)
    if not placed_support:
        raise ValueError(f"Repeater needs support but cannot place it for tower move at launch cell {launch}")

    col = [
        (bx, y,     bz, opaque_support_block),
        (bx, y + 1, bz, REDSTONE_TORCH),
        (bx, y + 2, bz, opaque_support_block),
        (bx, y + 3, bz, REDSTONE_TORCH),
        (bx, y + 4, bz, opaque_support_block),
    ]
    for cx, cy, cz, blk in col:
        existing_owner = dust_owner.get((cx, cy, cz))
        if existing_owner is not None and existing_owner != net_id:
            raise ValueError(f"Tower collision at {(cx, cy, cz)} for nets {existing_owner} vs {net_id}")
        if existing_owner == net_id:
            continue
        workspace[cx, cy, cz] = blk
        if blk == opaque_support_block:
            solid.add((cx, cy, cz))
        dust_owner[(cx, cy, cz)] = net_id
    if inverted_cells is not None:
        inverted_cells.add((bx, y + 1, bz))
        inverted_cells.add((bx, y + 2, bz))


def _lay_redstone_path(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        path: Iterable[tuple[int, int, int]],
        net_id: str,
        opaque_support_block: BlockState = STONE,
        inverted_cells: set[tuple[int, int, int]] | None = None,
        goal: tuple[int, int, int] | None = None,
        terminal_positions: frozenset[tuple[int, int, int]] = frozenset(),
) -> None:
    cells = list(path)
    n = len(cells)

    tower_bottom: set[int] = set()
    tower_top: set[int] = set()
    slope2_bottom: set[int] = set()
    p4_bottom: set[int] = set()
    p4_top: set[int] = set()
    for i in range(n - 1):
        ax, ay, az = cells[i]
        bx, by, bz = cells[i + 1]
        delta = (bx - ax, bz - az)
        if by == ay + 4 and delta in _TOWER_2BLOCK:
            tower_bottom.add(i)
            tower_top.add(i + 1)
        elif abs(by - ay) == 2 and delta in _TOWER_2BLOCK:
            slope2_bottom.add(i)
        elif by == ay - 4 and bx == ax and bz == az:
            p4_bottom.add(i)
            p4_top.add(i + 1)

    for i, cell in enumerate(cells):
        if (i in tower_top and i not in tower_bottom) or (i in p4_top and i not in p4_bottom):
            continue  # already placed by their respective move handler

        if i in slope2_bottom:
            _lay_slope2_move(workspace, solid, dust_owner, net_id, cell, cells[i + 1], BlockState("minecraft:sandstone"))

        elif i in tower_bottom:
            _lay_tower_move(workspace, solid, dust_owner, net_id, cell, cells[i + 1], opaque_support_block, inverted_cells)

        elif i in p4_bottom:
            _lay_powered_minus4_move(workspace, solid, dust_owner, net_id, cell, cells[i + 1], opaque_support_block, inverted_cells, goal, terminal_positions)

        else:
            _lay_dust_cell(workspace, solid, dust_owner, net_id, cell, opaque_support_block)


def _place_repeaters_for_net(
        workspace: Region,
        dust_owner: dict[tuple[int, int, int], str],
        net_id: str,
        ordered_paths: list[list[tuple[int, int, int]]],
) -> None:
    """Walk each routed A* path linearly, placing repeaters as needed.

    Uses the ordered paths directly so wrap-around detours are measured at
    their true signal length, not the shorter adjacency-graph shortcut.
    Torches and existing repeaters reset the signal counter.
    Shared prefixes are re-walked per path; repeaters placed on path N are
    detected via _is_repeater on path N+1, resetting the counter correctly.
    """
    # Cells with >1 distinct successor across all paths are branch points;
    # a repeater there can only drive one direction so it is never viable.
    successors: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
    for path in ordered_paths:
        for i in range(len(path) - 1):
            cell = path[i]
            if cell not in successors:
                successors[cell] = set()
            successors[cell].add(path[i + 1])
    branch_points: set[tuple[int, int, int]] = {
        cell for cell, succs in successors.items() if len(succs) > 1
    }

    def is_viable(path: list[tuple[int, int, int]], j: int) -> bool:
        if j <= 0 or j >= len(path) - 1:
            return False
        x, y, z = path[j]
        px, py, pz = path[j - 1]
        nx, ny, nz = path[j + 1]
        if y != py:
            return False  # on slope or tower segment
        dx, dz = x - px, z - pz
        if (dx, dz) not in _DELTA_TO_FACING:
            return False
        if ny != y:
            return False  # next cell is slope or tower
        if (nx - x, nz - z) != (dx, dz):
            return False  # turn after repeater
        if path[j] in branch_points:
            return False  # branch point — repeater drives only one output
        return True

    # upstream_seg[pos] = path segment [last_reset_cell, ..., pos] from the most recent
    # reset point to pos. Prepended to tree-seed paths so the backward cap search can
    # reach viable repeater positions in the already-laid upstream wire.
    upstream_seg: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}

    for orig_path in ordered_paths:
        path: list[tuple[int, int, int]] = list(orig_path)
        start = path[0]
        # Prepend upstream segment so the cap search can look back into prior wire.
        if start in upstream_seg and len(upstream_seg[start]) > 1:
            path = upstream_seg[start][:-1] + path  # upstream[:-1] avoids duplicating start
        reset_depth = 0 # path will always start at reset location
        reset_idx = 0
        depth = 0
        depths = []
        for pos_idx, pos in enumerate(path):
            px_, py_, pz_ = pos
            cell_block = workspace[px_, py_, pz_]
            if pos_idx > 0:
                prev = path[pos_idx - 1]
                depth += abs(pos[0] - prev[0]) + abs(pos[2] - prev[2]) # dy does not decrease power
                dy = pos[1] - prev[1]
                dxz = (pos[0] - prev[0], pos[2] - prev[2])
                # Tower top: repeater in column outputs full signal.
                # Powered-minus-4 dest: second wall torch outputs full signal.
                is_special_reset = (
                    (dy == 4 and dxz in _TOWER_2BLOCK)
                    or (dy == -4 and dxz == (0, 0))
                )
                if is_special_reset or _is_repeater(cell_block) or _is_torch(cell_block):
                    reset_depth = depth - 1
                    reset_idx = pos_idx

            depths.append(depth)
            dist = depth - reset_depth
            upstream_seg[pos] = path[reset_idx:pos_idx + 1]
            if dist >= _REPEATER_INTERVAL:
                # Scan backward in signal-depth units, not path-index units.
                # cap = highest j where the repeater can actually receive signal.
                cap = pos_idx - 1
                while cap > reset_idx and depths[cap] - reset_depth >= _REPEATER_INTERVAL:
                    cap -= 1
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
                        reset_depth = depths[j]
                        reset_idx = j
                        break
                else:
                    raise ValueError(
                        f"Net {net_id!r}: cannot place repeater to extend signal at path depth {pos_idx} "
                        f"(signal dist {dist} >= {_REPEATER_INTERVAL}, no viable position in "
                        f"{path[reset_idx+1:cap+1]})"
                    )
