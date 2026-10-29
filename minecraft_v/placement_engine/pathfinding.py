from __future__ import annotations

import heapq
import time
from typing import Callable

import numpy as np
from litemapy import BlockState, Region

from .block_utils import _is_air, _is_repeater, _is_torch, _is_redstone_wire
from .constants import _HORIZ_DIRS, _DIRS_6, _TOWER_2BLOCK, GLASS, _ROUTE_MAX_NODES, _ROUTE_STAGNATION, \
    _ROUTE_BUDGET_FACTOR, _DELTA_TO_FACING, _HORIZ_DIRS_8, _H_RADIUS_FACTOR, _HORIZ_RADIUS_MIN, \
    _Y_RADIUS_MIN


class RouteTimeout(ValueError):
    """A* gave up because its time_limit expired (as opposed to exhausting the search)."""


# A* local path snapshot as a bitmask: one bit per cell of a node's 3x3x3 window minus its
# own column (8 horizontal neighbours x Y-1/0/+1) — exactly the cells move conditions ask
# about. Cells outside the window are never in a snapshot.
_SNAP_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (ox, oy, oz) for oy in (-1, 0, 1) for ox in (-1, 0, 1) for oz in (-1, 0, 1) if ox or oz)
_SNAP_BIT: dict[tuple[int, int, int], int] = {o: 1 << i for i, o in enumerate(_SNAP_OFFSETS)}
_SNAP_MASK = (1 << len(_SNAP_OFFSETS)) - 1
_NO_PARENT = -1  # parent node of A* seeds


def _snap_bit(ox: int, oy: int, oz: int) -> int:
    return _SNAP_BIT.get((ox, oy, oz), 0)


# Per horizontal direction (in _HORIZ_DIRS order): dx, dz, then the snapshot bits of the
# neighbour cell at Y-1, Y and Y+1.
_HORIZ_SNAP_BITS: tuple[tuple[int, int, int, int, int], ...] = tuple(
    (dx, dz, _snap_bit(dx, -1, dz), _snap_bit(dx, 0, dz), _snap_bit(dx, 1, dz)) for dx, dz in _HORIZ_DIRS)


_snap_shift_cache: dict[tuple[int, int, int], tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]] = {}


def _snap_shift_tables(d: tuple[int, int, int]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    """Re-centre a snapshot from a node onto its neighbour at offset d: per-byte lookup
    tables mapping the node's bits to the neighbour's, plus the bit of the node itself."""
    tables = _snap_shift_cache.get(d)
    if tables is None:
        dx, dy, dz = d
        moved = [_snap_bit(ox - dx, oy - dy, oz - dz) for ox, oy, oz in _SNAP_OFFSETS]
        t0, t1, t2 = (tuple(sum(moved[8 * k + i] for i in range(8) if b >> i & 1) for b in range(256))
                      for k in range(3))
        tables = _snap_shift_cache[d] = (t0, t1, t2, _snap_bit(-dx, -dy, -dz))
    return tables


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


class _BlockIndex:
    """Sparse sets of signal-relevant cells, built from the Region's raw block array.

    litemapy's Region.__getitem__ is slow (coordinate conversion + palette lookup per
    read); _wire_walkable does ~40 reads per call. Valid only while the workspace is
    unchanged, i.e. for the duration of one A* search.
    """
    __slots__ = ("non_air", "wire", "torch", "repeater")

    def __init__(self, region: Region):
        # Positive-size regions store blocks at region coordinates directly.
        assert region.width > 0 and region.height > 0 and region.length > 0
        blocks = region._Region__blocks
        kinds = np.array([
            (0 if _is_air(b) else 1) | (2 if _is_redstone_wire(b) else 0)
            | (4 if _is_torch(b) else 0) | (8 if _is_repeater(b) else 0)
            for b in region._Region__palette
        ], dtype=np.uint8)
        k = kinds[blocks]

        def cells(mask: int) -> set[tuple[int, int, int]]:
            return set(map(tuple, np.argwhere(k & mask).tolist()))

        self.non_air = cells(1)
        self.wire = cells(2)
        self.torch = cells(4)
        self.repeater = cells(8)

    @classmethod
    def layered(cls, base: _BlockIndex, overlay: dict[tuple[int, int, int], BlockState]) -> _BlockIndex:
        """base (static world) plus a net's overlay blocks. Overlay only ever fills air in base."""
        idx = cls.__new__(cls)
        idx.non_air = set(base.non_air)
        idx.wire = set(base.wire)
        idx.torch = set(base.torch)
        idx.repeater = set(base.repeater)
        for pos, b in overlay.items():
            if _is_air(b):
                continue
            idx.non_air.add(pos)
            if _is_redstone_wire(b):
                idx.wire.add(pos)
            elif _is_torch(b):
                idx.torch.add(pos)
            elif _is_repeater(b):
                idx.repeater.add(pos)
        return idx

    def update(self, pos: tuple[int, int, int], old: BlockState, new: BlockState) -> None:
        """Reflect one block write (old → new); same classification as __init__."""
        for cells, test in ((self.non_air, lambda b: not _is_air(b)), (self.wire, _is_redstone_wire),
                            (self.torch, _is_torch), (self.repeater, _is_repeater)):
            was, now = test(old), test(new)
            if was and not now:
                cells.discard(pos)
            elif now and not was:
                cells.add(pos)


class IndexedWorkspace:
    """Write-through Region proxy that keeps a _BlockIndex current across searches.

    Building a _BlockIndex scans the whole region; with one search per sink that dominated
    the greedy pass on big designs. Every write must go through this proxy, or the index
    goes stale. Searches read the live index, so the region must not change mid-search.
    """

    def __init__(self, region: Region):
        self.region = region
        self._index = _BlockIndex(region)

    def __getitem__(self, pos) -> BlockState:
        return self.region[pos]

    def __setitem__(self, pos, block: BlockState) -> None:
        pos = tuple(pos)
        self._index.update(pos, self.region[pos], block)
        self.region[pos] = block

    def __getattr__(self, name):
        return getattr(self.region, name)

    def block_index(self) -> _BlockIndex:
        return self._index


def _wire_walkable(
        blocks: _BlockIndex | Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
        pos: tuple[int, int, int],
        net_id: str,
        bounds: tuple[int, int, int, int, int, int],
        exempt_foreign: frozenset[tuple[int, int, int]] = frozenset(),
        inverted_cells: frozenset[tuple[int, int, int]] = frozenset(),
        footprint_blocked: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """Pass a _BlockIndex when calling repeatedly against an unchanged workspace."""
    if not isinstance(blocks, _BlockIndex):
        blocks = _BlockIndex(blocks)
    wire, torch, repeater = blocks.wire, blocks.torch, blocks.repeater
    x, y, z = pos
    # Bounds checks inlined: this runs millions of times per routing run.
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    if not (min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z):
        return False
    if pos in solid:
        return False
    if pos in blocks.non_air:
        if pos in wire:
            owner = dust_owner.get(pos)
            if owner is not None and owner != net_id:
                return False
        else:
            return False
    # 3-wide foreign signal: no foreign-owned cell (wire, torch, or tower stone)
    # at horizontal cardinal neighbors at same Y or +-1 Y.
    for dx, dz in _HORIZ_DIRS:
        for dy in (-1, 0, 1):
            ny = y + dy
            np = (x + dx, ny, z + dz)
            if np in exempt_foreign:
                continue
            if not (min_x <= x + dx <= max_x and min_y <= ny <= max_y and min_z <= z + dz <= max_z):
                continue
            owner = dust_owner.get(np)
            if owner is not None and owner != net_id:
                return False
            if owner is None and np not in footprint_blocked:
                if np in torch or np in wire:
                    return False
    # Vertical same-XZ: powered blocks distribute signal on all 6 faces including
    # top and bottom, so wire directly above or below a foreign-owned block is unsafe.
    for vdy in (1, -1):
        vp = (x, y + vdy, z)
        if vp in exempt_foreign:
            continue
        if not min_y <= y + vdy <= max_y:
            continue
        # No per-neighbour bounds guard: the index only holds cells inside the Region, and
        # routing bounds only exclude Region cells that are empty while core nets route.
        vx, vy, vz = vp
        if not ((vx + 1, vy, vz) in repeater or (vx - 1, vy, vz) in repeater
                or (vx, vy, vz + 1) in repeater or (vx, vy, vz - 1) in repeater
                or (vx, vy - 1, vz) in torch):
            continue
        owner = dust_owner.get(vp)
        if owner is not None and owner != net_id:
            return False
    # Inverted cells (tower torch/stone, powered-minus4 torch/dust) must not be
    # horizontally adjacent at same Y or ±1 Y — the ±1 check covers slope connections
    # into inverted redstone dust, which wire can reach slopingly.
    for dx, dz in _HORIZ_DIRS:
        for dy in (-1, 0, 1):
            if (x + dx, y + dy, z + dz) in inverted_cells:
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
        terminal_positions: frozenset[tuple[int, int, int]] = frozenset(),
        weight: float = 1.5,
        limit_exclusions=False,
        x_h_factor=1,
        congestion_cost: Callable[[tuple[int, int, int]], float] | None = None,
        time_limit: float | None = None,
) -> list[tuple[int, int, int]]:
    """A* wire route from start to goal.

    congestion_cost: extra cost per path cell (negotiated-congestion routing: other nets'
    usage + history). Excluded from the f-cap budget.
    time_limit: wall-clock seconds for the search; on expiry it fails like any other early stop.
    workspace may be a view exposing block_index() (see negotiate.OverlayWorkspace).
    """
    min_y = bounds[1]
    # came_from is write-only during search (set at expansion, not push) and used
    # only for final path reconstruction. All history-dependent move checks use
    # path_snapshot carried per heap entry instead, so different paths to the same
    # cell don't corrupt each other's move conditions.
    came_from: dict[int, int] = {}  # node -> parent node (see the A* search below)

    # Strip the goal terminal's own keepout zone from protected so the approach
    # path is not blocked by the destination's own neighborhood.
    gx, gy, gz = goal
    goal_exclusion: set[tuple[int, int, int]] = {(gx, gy, gz)}
    for _dx, _dz in (_HORIZ_DIRS if not limit_exclusions else ((0, 1), (0, 2), (0, 3))):
        for _dy in range(-2, 3):
            goal_exclusion.add((gx + _dx, gy + _dy, gz + _dz))
    for _dx, _dz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for _dy in range(-2, 2):
            goal_exclusion.add((gx + _dx, gy + _dy, gz + _dz))
    goal_exclusion.add((gx, gy - 1, gz))
    effective_protected = protected - goal_exclusion

    sx, sy, sz = start

    # Shrink footprint blocking on the side the goal terminal is on: remove the
    # 1-cell expansion cells immediately adjacent to the goal (including +2Y for
    # the IO repeater top clearance) so slope/flat approaches can reach the terminal.
    fp_relief: set[tuple[int, int, int]] = set()
    if not limit_exclusions:
        for _ddx, _ddz in ((1, 0), (-1, 0), (0, 1)):
            for _ddy in (-1, 1):
                fp_relief.add((gx + _ddx, gy + _ddy, gz + _ddz))
        for _ddx in range(1, 3):
            for _ddy in range(-1, 2):
                fp_relief.add((sx + _ddx, sy + _ddy, sz))
                fp_relief.add((sx - _ddx, sy + _ddy, sz))
        fp_relief.add((gx, gy + 2, gz))
        fp_relief.add((gx, gy - 1, gz))
    effective_footprint_blocked = footprint_blocked - fp_relief

    # Workspace is static during a search, so per-cell walkability is cached. A* asks
    # about the same cell many times (multiple local snapshots, multiple move types).
    walk_cache: dict[tuple[tuple[int, int, int], frozenset[tuple[int, int, int]]], bool] = {}
    _no_exempt: frozenset[tuple[int, int, int]] = frozenset()
    block_index = workspace.block_index() if hasattr(workspace, "block_index") else _BlockIndex(workspace)
    ix_non_air, ix_wire, ix_torch, ix_repeater = (
        block_index.non_air, block_index.wire, block_index.torch, block_index.repeater)

    def can_support(pos: tuple[int, int, int]) -> bool:
        # Same as _can_be_support, via the index.
        return _in_bounds(pos, bounds) and (pos in solid or pos not in ix_non_air)

    def walkable(
            pos: tuple[int, int, int],
            exempt: frozenset[tuple[int, int, int]] = _no_exempt,
    ) -> bool:
        if pos in effective_protected or pos in effective_footprint_blocked:
            return False
        key = (pos, exempt)
        if key not in walk_cache:
            walk_cache[key] = _wire_walkable(
                block_index, solid, dust_owner, pos, net_id, bounds, exempt_foreign=exempt,
                inverted_cells=inverted_cells, footprint_blocked=effective_footprint_blocked,
            )
        return walk_cache[key]

    mid_cache: dict[tuple[tuple[int, int, int], frozenset[tuple[int, int, int]]], bool] = {}

    def mid_walkable(pos: tuple[int, int, int], exempt: frozenset[tuple[int, int, int]]) -> bool:
        # 2x-slope intermediate cell: plain _wire_walkable (no inverted/footprint args), cached.
        key = (pos, exempt)
        if key not in mid_cache:
            mid_cache[key] = _wire_walkable(block_index, solid, dust_owner, pos, net_id, bounds, exempt_foreign=exempt)
        return mid_cache[key]

    def _is_powered_block(pos: tuple[int, int, int]):
        if not _in_bounds(pos, bounds) or pos not in solid:
            return False
        has_input_repeater = False
        for dx, dz in _HORIZ_DIRS:
            b = (pos[0] + dx, pos[1], pos[2] + dz)
            if not _in_bounds(b, bounds):
                continue
            if b not in ix_repeater:
                continue
            block = workspace[*b]
            if block['facing'] == _DELTA_TO_FACING[(dx, dz)]:
                has_input_repeater = True
                break

        return has_input_repeater

    def _torch_below(support_pos: tuple[int, int, int]) -> bool:
        """True if a torch at support_pos-1Y would power the support block → unintended signal."""
        bx, by, bz = support_pos
        below = (bx, by - 1, bz)
        if not _in_bounds(below, bounds):
            return False
        return below in ix_torch

    def heuristic(pos: tuple[int, int, int]) -> int:
        dx = abs(pos[0] - goal[0]) * x_h_factor
        dz = abs(pos[2] - goal[2])
        dy = abs(pos[1] - goal[1])
        return dx + dy * 4 + dz

    # Moves from a cell depend on the path only through the local snapshot (and whether the
    # cell was entered by a tower move); everything else reads the static world. A cell is
    # expanded once per distinct snapshot (~15x on mid designs), so each cell's moves are
    # worked out once, recording per move the snapshot bits that would forbid it.
    # Move entry: (dest_id, cost, forbid_bits, tower_gate, extra_cell, snap_tables, flat_dir, h, near)
    #   tower_gate: 0 always, 1 not right after a tower move, 2 only right after one
    #   extra_cell: 2x-slope intermediate, added to every neighbour's snapshot when the move is taken
    #   flat_dir: (dx, dz) for same-Y moves of at most one XZ step (direction-change penalty), else None
    # Powered -4 is one move with alternatives: the first side whose bits are clear wins.
    #
    # Cells get small int ids (pos_list[id] = cell) so a search node is one int,
    # id << 24 | snapshot, instead of a nested tuple that is re-hashed on every lookup.
    pos_list: list[tuple[int, int, int]] = []
    pos_ids: dict[tuple[int, int, int], int] = {}
    move_cache: list[tuple[list[tuple], tuple | None] | None] = []  # by cell id

    def _pid(p: tuple[int, int, int]) -> int:
        i = pos_ids.get(p)
        if i is None:
            i = pos_ids[p] = len(pos_list)
            pos_list.append(p)
            move_cache.append(None)
        return i

    def _entry(pos, dest, cost, forbid, gate=0, extra=None):
        x, y, z = pos
        dx, dy, dz = dest[0] - x, dest[1] - y, dest[2] - z
        flat_dir = (dx, dz) if dy == 0 and abs(dx) <= 1 and abs(dz) <= 1 else None
        h = heuristic(dest)
        near = ((abs(goal[0] - dest[0]) + abs(goal[2] - dest[2])) < _HORIZ_RADIUS_MIN
                and abs(goal[1] - dest[1]) < _Y_RADIUS_MIN)
        return _pid(dest), cost, forbid, gate, extra, _snap_shift_tables((dx, dy, dz)), flat_dir, h, near

    def _static_moves(pos: tuple[int, int, int]) -> tuple[list[tuple], tuple | None]:
        if pos in inverted_cells:
            return [], None
        x, y, z = pos
        moves: list[tuple] = []

        for dx, dz, b_lo, b_mid, b_hi in _HORIZ_SNAP_BITS:
            nx, nz = x + dx, z + dz

            # Flat move
            flat = (nx, y, nz)
            if walkable(flat) and not _is_powered_block((nx, y + 1, nz)):
                if y <= min_y:
                    moves.append(_entry(pos, flat, 1, b_hi | b_mid))
                elif can_support((nx, y - 1, nz)) and not _torch_below((nx, y - 1, nz)):
                    moves.append(_entry(pos, flat, 1, b_hi | b_mid | b_lo))

            # Slope-up (disallowed immediately after a tower move — tower top already
            # has a dedicated +1Y move)
            if y < max_bridge_y:
                up = (nx, y + 1, nz)
                if (x, y + 1, z) not in ix_non_air and walkable(up) and not _is_powered_block((nx, y + 2, nz)): # ensure no block will cut off slope connection and up pos is valid
                    ramp = (nx, y, nz)
                    if (
                        ramp in solid or (_in_bounds(ramp, bounds) and ramp not in ix_non_air)
                    ) and not _torch_below(ramp):
                        # TODO: this discourages ramping so we get torch towers for vertical signal extension, but we should guarantee that vertical extension no matter what path is chosen
                        moves.append(_entry(pos, up, 3, b_hi | b_mid, gate=1))

            # Slope-down
            if y > min_y:
                down = (nx, y - 1, nz)
                above_down = (nx, y, nz)
                # (nx, y+1, nz) is the ancestor wire that would place a support at above_down,
                # blocking the slope — hence its bit in the forbid mask.
                above_clear = (
                    above_down not in solid
                    and _in_bounds(above_down, bounds)
                    and above_down not in ix_non_air
                    and not _is_powered_block(above_down)
                )
                exempt: set[tuple[int, int, int]] = set()
                # current dust might generate support, which would cut off connections to foreign dust below
                if can_support((x, y - 1, z)):
                    exempt.add((x, y - 2, z))
                if above_clear and walkable(down, frozenset(exempt)):
                    if y - 1 <= min_y or (
                        can_support((nx, y - 2, nz))
                        and not _torch_below((nx, y - 2, nz))
                    ):
                        below_owner = dust_owner.get((nx, y - 3, nz))
                        breaks_slope = (
                            below_owner is not None
                            and below_owner != net_id
                            and any(
                                dust_owner.get((nx + sdx, y - 2, nz + sdz)) == below_owner
                                for sdx, sdz in _HORIZ_DIRS
                            )
                        )
                        if not breaks_slope:
                            moves.append(_entry(pos, down, 2, b_lo | b_mid | b_hi))

        for dx, dz, b_lo, b_mid, b_hi in _HORIZ_SNAP_BITS:
            nx2, nz2 = x + 2 * dx, z + 2 * dz
            exempt_top = frozenset([(nx2, y, nz2)])

            # 2x slope-up (disallowed immediately after tower move)
            if y + 2 <= max_bridge_y:
                mid = (x + dx, y + 1, z + dz)
                above_mid = (x + dx, y + 2, z + dz)
                top = (nx2, y + 2, nz2)
                mid_support = (x + dx, y, z + dz)
                mid_ok = (
                    (x, y + 1, z) not in ix_non_air
                    and mid not in protected
                    and mid not in footprint_blocked
                    and _in_bounds(mid, bounds)
                    and above_mid not in solid
                    and mid not in solid
                    and mid not in ix_non_air
                    and mid_walkable(mid, exempt_top)
                )
                if mid_ok and walkable(top):
                    if can_support(mid_support) and not _torch_below(mid_support):
                        moves.append(_entry(pos, top, 7, b_hi | b_mid, gate=1, extra=mid))

            # 2x slope-down
            if y - 2 >= min_y:
                mid_dn = (x + dx, y - 1, z + dz)
                top_dn = (nx2, y - 2, nz2)
                exempt_dn = frozenset([(x, y - 2, z)])
                above_mid = (x + dx, y, z + dz)
                above_clear = (
                    above_mid not in solid
                    and _in_bounds(above_mid, bounds)
                    and above_mid not in ix_non_air
                )
                mid_dn_ok = (
                    above_clear
                    and mid_dn not in protected
                    and mid_dn not in footprint_blocked
                    and _in_bounds(mid_dn, bounds)
                    and mid_dn not in solid
                    and mid_dn not in ix_non_air
                    and mid_walkable(mid_dn, exempt_dn)
                )
                if mid_dn_ok and walkable(top_dn):
                    if y - 2 <= min_y or (
                        can_support((nx2, y - 3, nz2))
                        and not _torch_below((nx2, y - 3, nz2))
                    ):
                        moves.append(_entry(pos, top_dn, 7, b_lo | b_mid | b_hi, extra=mid_dn))

        # +1Y from tower top: only valid when this position was reached via a tower move
        up = (x, y + 1, z)
        if y + 1 <= max_bridge_y and walkable(up):
            moves.append(_entry(pos, up, 1, 0, gate=2))

        # Tower-up: 2 blocks to the side + 4 blocks up.
        for tower_top, snap_bits in _tower_candidates(pos):
            moves.append(_entry(pos, tower_top, 8, snap_bits))

        return moves, _powered_minus4_move(pos)

    def _tower_candidates(pos: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], int]]:
        """Tower-up moves from pos legal against the static world: (tower_top, snapshot bits of
        the repeater and its support). (The cell below the tower base is outside the snapshot window.)"""
        x, y, z = pos
        out: list[tuple[tuple[int, int, int], int]] = []
        if y + 4 > max_bridge_y:
            return out
        for tdx, tdz in _HORIZ_DIRS:
            tower_top = (x + 2 * tdx, y + 4, z + 2 * tdz)
            if not walkable(tower_top):
                continue
            rep = (x + tdx, y, z + tdz)
            if not _in_bounds(rep, bounds) or rep in footprint_blocked or rep in protected:
                continue
            if rep in ix_non_air:
                continue
            rep_support = (x + tdx, y - 1, z + tdz)
            if not can_support(rep_support):
                continue
            below_rep_support = (x + tdx, y - 2, z + tdz)
            rep_support_cuts_slope = False
            if _in_bounds(below_rep_support, bounds) and below_rep_support in ix_wire:
                for dx, dz in _HORIZ_DIRS:
                    maybe_next = (below_rep_support[0] + dx, below_rep_support[1] + 1, below_rep_support[2] + dz)
                    if _in_bounds(maybe_next, bounds) and maybe_next in ix_wire:
                        rep_support_cuts_slope = True
                        break
            if rep_support_cuts_slope:
                continue

            col_clear = True
            for cdy in range(5):
                cp = (x + 2 * tdx, y + cdy, z + 2 * tdz)
                if not _in_bounds(cp, bounds) or cp in footprint_blocked or cp in protected:
                    col_clear = False
                    break
                if cp in ix_non_air:
                    col_clear = False
                    break
                for dx, dz in _HORIZ_DIRS:
                    cp_side = (cp[0] + dx, cp[1], cp[2] + dz)
                    if _in_bounds(cp_side, bounds) and cp_side in ix_wire:
                        # If it's our own connection, we create a loop, if its others', we're interfering
                        col_clear = False
                        break
                if not col_clear:
                    break

            if not col_clear:
                continue
            # Tower base will be powered — check cell below base has no foreign wire
            below_base = (x + 2 * tdx, y - 1, z + 2 * tdz)
            if _in_bounds(below_base, bounds):
                below_owner = dust_owner.get(below_base)
                if below_owner is not None and below_owner != net_id:
                    continue
                # If it's our own connection, we create a loop, if its others', we're interfering
                if below_base in ix_wire:
                    continue
            out.append((tower_top, _snap_bit(tdx, 0, tdz) | _snap_bit(tdx, -1, tdz)))
        return out

    m4_clear_cache: dict[tuple[int, int, int], bool] = {}

    def m4_cell_clear(cp: tuple[int, int, int]) -> bool:
        """Path-independent part of a -4 side cell check: no protected cell or foreign
        owner among its cardinal neighbours at Y-1/0/+1, and not foreign-owned itself."""
        if cp not in m4_clear_cache:
            ok = True
            for ddx, ddz in _HORIZ_DIRS:
                for ddy in (-1, 0, 1):
                    sp = (cp[0] + ddx, cp[1] + ddy, cp[2] + ddz)
                    owner = dust_owner.get(sp)
                    if sp in protected or (owner is not None and owner != net_id):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                owner = dust_owner.get(cp)
                ok = owner is None or owner == net_id
            m4_clear_cache[cp] = ok
        return m4_clear_cache[cp]

    def _powered_minus4_move(pos: tuple[int, int, int]) -> tuple | None:
        """(move entry, [(forbid_bits, move_cells)] per side legal against the static world),
        or None. Taken through the first side whose bits are clear of the snapshot."""
        x, y, z = pos
        if y - 4 < min_y:
            return None
        dest = (x, y - 4, z)
        if not walkable(dest):
            return None

        # Support block at (x,y-1,z): permanent solid block — must not land in any
        # footprint or protected zone (goal relief is for wire approach, not solid structures).
        below_launch = (x, y - 1, z)  # own column: never in the snapshot window
        if below_launch == goal or below_launch in terminal_positions or below_launch in footprint_blocked or below_launch in protected:
            return None
        if below_launch in solid:
            if workspace[below_launch[0], below_launch[1], below_launch[2]] == GLASS:
                return None
        elif not (_in_bounds(below_launch, bounds) and below_launch not in ix_non_air):
            return None

        sides: list[tuple[int, tuple[tuple[int, int, int], ...]]] = []
        for dx, dz, b_lo, b_mid, b_hi in _HORIZ_SNAP_BITS:
            nx, nz = x + dx, z + dz

            # All side/clearance cells must be air, unblocked, and unowned by foreign net.
            # Use full protected (not effective_protected) — goal relief is for wire approach only,
            # not for permanent solid blocks or torches placed by this move.
            side_cells = (
                (nx, y - 1, nz),  # first wall torch (inverted)
                (nx, y - 2, nz),  # inverted dust
                (nx, y - 3, nz),  # second support block
                (x,  y - 3, z),   # second wall torch (active)
                (nx, y,     nz),  # clearance above first torch
                (x,  y - 2, z),   # clearance above second torch
            )
            ok = True
            for cp in side_cells:  # cheap checks first
                if (not _in_bounds(cp, bounds) or cp == goal or cp in terminal_positions
                        or cp in footprint_blocked or cp in protected
                        or cp in solid or cp in ix_non_air):
                    ok = False
                    break
            if not ok or not all(m4_cell_clear(cp) for cp in side_cells):
                continue

            # Inverted dust at (nx,y-2,nz) must have no wire at slope-adjacent positions
            # (horizontal cardinal ±1 Y) — dust connects slopingly and would pick up wrong signal.
            for ddx, ddz in _HORIZ_DIRS:
                for ddy in (-1, 1):
                    sp = (nx + ddx, y - 2 + ddy, nz + ddz)
                    if dust_owner.get(sp) is not None:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                # Side cells (and the cells above them) that fall in the snapshot window are
                # (nx, y-1..y+1, nz). move_cells: cells this move fills besides the path cells.
                # They go into the dest's path snapshot so later moves don't step into them
                # (e.g. sloping up onto the second support block).
                sides.append((b_lo | b_mid | b_hi, (below_launch, *side_cells)))

        return (_entry(pos, dest, 8, 0), sides) if sides else None

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
    if not goal_set:
        raise ValueError(f"No walkable cell near goal {goal} for net {net_id}")

    # A* search
    # Heap entry: (f, -h, g, counter, g_wire, node, parent_node). g_wire excludes conflict
    # penalties so the f-cap only bounds real wire length, not negotiation cost.
    # counter is unique per push so comparison never falls through to node/parent.
    # Nodes are (cell id << 24 | local_snapshot), so paths with different local histories
    # coexist — a cheaper path with a restrictive snapshot does not block a more
    # expensive path whose snapshot enables moves the cheaper one can't make.
    # A seed's parent is _NO_PARENT.
    counter = 0
    open_heap = []
    g_score: dict[int, int] = {}
    for seed in seeds:
        # Local snap of seed is empty — seed is never in its own window.
        seed_key = _pid(seed) << 24
        g_score[seed_key] = 0
        h0 = heuristic(seed)
        heapq.heappush(open_heap, (weight * h0, -h0, 0, counter, 0, seed_key, _NO_PARENT))
        counter += 1
    goal_ids = {_pid(gc) for gc in goal_set}

    def node_str(node: int) -> str:
        return str((pos_list[node >> 24], node & _SNAP_MASK))

    reached: int | None = None
    best_node = _pid(seeds[0]) << 24
    best_h: int = min(heuristic(s) for s in seeds)
    max_f: int = max(best_h, 10) * _ROUTE_BUDGET_FACTOR
    explored: set[int] = set()
    explored_xyz: set[int] = set()  # cell ids
    stagnation = 0
    h_radius_pruned = 0
    early_stop_reason: str | None = None
    deadline = time.monotonic() + time_limit if time_limit is not None else None
    pops = 0
    while open_heap:
        pops += 1
        if deadline is not None and pops & 255 == 0 and time.monotonic() > deadline:
            early_stop_reason = f"timeout ({time_limit}s)"
            break
        f, _, g, _, g_wire, current_node, parent = heapq.heappop(open_heap)
        # f-cap: prune nodes whose total estimated cost exceeds budget.
        # Catches runaway detours around impossible obstacles without burning _ROUTE_MAX_NODES.
        f_wire = f - g + g_wire
        if f_wire > max_f:
            early_stop_reason = f"f-cap exceeded (f={f_wire} > max_f={max_f})"
            break
        if g > g_score.get(current_node, 10 ** 9):
            continue
        # Record parent for path reconstruction. First expansion wins: subsequent
        # expansions of the same cell (different local_snap) don't overwrite, so
        # came_from forms a tree with strictly decreasing g toward the seed — no cycles.
        if current_node not in came_from:
            came_from[current_node] = parent
        cid = current_node >> 24
        cur_pos = pos_list[cid]
        h = heuristic(cur_pos)
        if h < best_h:
            best_h = h
            best_node = current_node
        # Stagnation = consecutive re-expansions of already-explored (pos,snap) pairs.
        # New territory always resets it; detours away from goal don't penalize.
        if current_node in explored:
            stagnation += 1
            if stagnation >= _ROUTE_STAGNATION:
                early_stop_reason = f"stagnated ({stagnation} re-expansions without new territory)"
                break
        else:
            stagnation = 0
            explored.add(current_node)
            explored_xyz.add(cid)
            # Unique-XYZ cap: more meaningful than (pos,snap) pair count.
            if len(explored_xyz) >= _ROUTE_MAX_NODES:
                early_stop_reason = f"node cap ({_ROUTE_MAX_NODES} unique positions)"
                break
        if cid in goal_ids:
            reached = current_node
            break
        cur_snap = current_node & _SNAP_MASK
        par_pos = pos_list[parent >> 24] if parent != _NO_PARENT else None
        came_from_tower = (
            par_pos is not None
            and par_pos[1] + 4 == cur_pos[1]
            and (cur_pos[0] - par_pos[0], cur_pos[2] - par_pos[2]) in _TOWER_2BLOCK
        )
        static = move_cache[cid]
        if static is None:
            static = move_cache[cid] = _static_moves(cur_pos)
        moves, m4 = static
        taken: list[tuple] = []
        additional_blocked: list[tuple[int, int, int]] = []
        for move in moves:
            if cur_snap & move[2]:
                continue
            gate = move[3]
            if gate and (gate == 1) == came_from_tower:
                continue
            taken.append(move)
            if move[4] is not None:
                additional_blocked.append(move[4])
        m4_cells: tuple[tuple[int, int, int], ...] = ()
        if m4 is not None:
            for forbid, cells in m4[1]:
                if not cur_snap & forbid:
                    m4_cells = cells
                    break
        # Re-centre the snapshot (plus current itself and the cells this expansion's moves
        # fill) onto each neighbour's window.
        cb0, cb1, cb2 = cur_snap & 255, cur_snap >> 8 & 255, cur_snap >> 16
        in_dir = (cur_pos[0] - par_pos[0], cur_pos[2] - par_pos[2]) if par_pos is not None else None
        h_limit = best_h * _H_RADIUS_FACTOR
        if m4_cells:
            taken.append(m4[0])
        last = len(taken) - 1
        for i, (nid, cost, _, _, _, (t0, t1, t2, self_bit), flat_dir, h, near) in enumerate(taken):
            # Penalise XZ direction change on flat moves only (avoids disrupting slope/tower costs).
            if in_dir is not None and flat_dir is not None and flat_dir != in_dir:
                cost += 1
            new_g_wire = g_wire + cost
            if congestion_cost is not None:
                cost += congestion_cost(pos_list[nid])
            new_g = g + cost
            neighbor_snap = t0[cb0] | t1[cb1] | t2[cb2] | self_bit
            move_cells = m4_cells if i == last else ()
            if additional_blocked or move_cells:
                nx, ny, nz = pos_list[nid]
                for c in additional_blocked:
                    neighbor_snap |= _SNAP_BIT.get((c[0] - nx, c[1] - ny, c[2] - nz), 0)
                for c in move_cells:
                    neighbor_snap |= _SNAP_BIT.get((c[0] - nx, c[1] - ny, c[2] - nz), 0)
            nkey = nid << 24 | neighbor_snap
            if new_g < g_score.get(nkey, 10 ** 9):
                g_score[nkey] = new_g
                if h <= h_limit or near:
                    heapq.heappush(
                        open_heap,
                        (new_g + weight * h, -h, new_g, counter, new_g_wire, nkey, current_node),
                    )
                    counter += 1
                else:
                    h_radius_pruned += 1

    if reached is None:
        lamp = BlockState("minecraft:redstone_lamp")
        cur = best_node
        # Seeds only enter came_from once expanded; an early stop (f-cap) can precede that.
        while cur is not None and cur != _NO_PARENT:
            cx, cy, cz = pos_list[cur >> 24]
            workspace[cx, cy, cz] = lamp
            cur = came_from.get(cur)
        # for prot in sorted(effective_footprint_blocked | effective_protected, key=heuristic)[:100]:
        #     if _is_air(workspace[*prot]):
        #         workspace[*prot] = GLASS
        reason = f"; {early_stop_reason}" if early_stop_reason else ""
        error = RouteTimeout if early_stop_reason and early_stop_reason.startswith("timeout") else ValueError
        raise error(f"No route for net {net_id} from {start} to {goal} (closest reached: {node_str(best_node)}{reason}; h_radius_pruned={h_radius_pruned})")
    if walkable(goal) and pos_list[reached >> 24] != goal:
        raise ValueError(f"No route reached goal {goal} for net {net_id}; stopped at {node_str(reached)}")

    path: list[tuple[int, int, int]] = []
    back_visited = set()
    cur = reached
    duplicate = False
    while cur != _NO_PARENT:
        if cur in back_visited:
            raise ValueError("Path tracing resulted in loop")
        cur_pos = pos_list[cur >> 24]
        if cur_pos in path:
            duplicate = True
        path.append(cur_pos)
        back_visited.add(cur)
        cur = came_from[cur]
    if duplicate:
        print("patch contains duplicates:")
        print(path)
    path.reverse()
    if not tree_seeds and path[0] != start:
        raise ValueError(f"Path for net {net_id} does not begin at start {start}; begins at {path[0]}")
    return path
