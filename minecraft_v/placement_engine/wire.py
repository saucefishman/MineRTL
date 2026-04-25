from __future__ import annotations
from typing import Iterable
from litemapy import BlockState, Region
from .constants import (
    GLASS, STONE, REDSTONE, REDSTONE_TORCH,
    _HORIZ_DIRS, _DELTA_TO_FACING, _OPPOSITE_SIDE, _REPEATER_INTERVAL, _TOWER_2BLOCK,
)
from .block_utils import _is_air, _is_redstone_wire, _is_torch, _is_repeater


def _place_support(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        net_id: str,
        cell: tuple[int, int, int],
        opaque_support_block: BlockState = STONE,
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


def _lay_tower_move(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        net_id: str,
        launch: tuple[int, int, int],
        tower_top: tuple[int, int, int],
        opaque_support_block: BlockState,
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
    _place_support(workspace, solid, dust_owner, net_id, (rx, ry, rz), opaque_support_block)

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
        if blk is STONE:
            solid.add((cx, cy, cz))
        dust_owner[(cx, cy, cz)] = net_id


def _lay_redstone_path(
        workspace: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        path: Iterable[tuple[int, int, int]],
        net_id: str,
        opaque_support_block: BlockState = STONE,
) -> None:
    cells = list(path)
    n = len(cells)

    tower_bottom: set[int] = set()
    tower_top: set[int] = set()
    slope2_bottom: set[int] = set()
    for i in range(n - 1):
        ax, ay, az = cells[i]
        bx, by, bz = cells[i + 1]
        delta = (bx - ax, bz - az)
        if by == ay + 4 and delta in _TOWER_2BLOCK:
            tower_bottom.add(i)
            tower_top.add(i + 1)
        elif abs(by - ay) == 2 and delta in _TOWER_2BLOCK:
            slope2_bottom.add(i)

    for i, cell in enumerate(cells):
        if i in tower_top and i not in tower_bottom:
            continue  # stone already placed by _lay_tower_move

        if i in slope2_bottom:
            _lay_slope2_move(workspace, solid, dust_owner, net_id, cell, cells[i + 1], opaque_support_block)

        elif i in tower_bottom:
            _lay_tower_move(workspace, solid, dust_owner, net_id, cell, cells[i + 1], opaque_support_block)

        else:
            _lay_dust_cell(workspace, solid, dust_owner, net_id, cell, opaque_support_block)
            # Alternating down-staircase: (x,y,z)→(x+dx,y-1,z+dz)→(x,y-2,z)
            # Support at (x,y-1,z) is directly above (x,y-2,z) wire — use glass.
            if i + 2 < n:
                ax, ay, az = cell
                bx, by, bz = cells[i + 1]
                cx, cy, cz = cells[i + 2]
                if (by == ay - 1 and abs(bx - ax) + abs(bz - az) == 1
                        and cx == ax and cy == ay - 2 and cz == az):
                    support = (ax, ay - 1, az)
                    if support in solid:
                        workspace[ax, ay - 1, az] = GLASS


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
        for dx, dz in _HORIZ_DIRS:
            for dy in (0, 1, -1):
                nb = (x + dx, y + dy, z + dz)
                if dust_owner.get(nb) == net_id:
                    result.append(nb)
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

    visited: set[tuple[int, int, int]] = {source}
    stack: list[tuple[tuple[int, int, int], list[tuple[int, int, int]], int]] = [
        (source, [source], 0)
    ]
    while stack:
        pos, path, reset_idx = stack.pop()
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
            else:
                raise ValueError(
                    f"Net {net_id!r}: cannot place repeater to extend signal at path depth {depth} "
                    f"(signal dist {dist} >= {_REPEATER_INTERVAL}, no viable position in "
                    f"{path[reset_idx+1:cap+1]})"
                )
        for nb in wire_neighbors(pos):
            if nb not in visited:
                visited.add(nb)
                stack.append((nb, path + [nb], new_reset))
