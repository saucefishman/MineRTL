from __future__ import annotations

from typing import Callable

import functools
import numpy as np
from litemapy import BlockState, Region

from .block_utils import _is_air, _is_repeater, _is_torch, _is_redstone_wire, lookup, palette_writer
from .wire import LayConflict, _foot_repeaters
from .constants import _HORIZ_DIRS, _DIRS_6, _TOWER_2BLOCK, GLASS, _ROUTE_MAX_NODES, _ROUTE_STAGNATION, _ROUTE_CELL_STATES, _ROUTE_TIGHT_POSITIONS, \
    _ROUTE_MAX_EXPANSIONS, _ROUTE_BUDGET_FACTOR, _ROUTE_BUDGET_FLOOR, _DELTA_TO_FACING, _H_RADIUS_FACTOR, _HORIZ_RADIUS_MIN, \
    _Y_RADIUS_MIN, _REPEATER_INTERVAL


_FACING = [_DELTA_TO_FACING[d] for d in _HORIZ_DIRS]


class RouteTimeout(ValueError):
    """A* gave up because its time_limit expired (as opposed to exhausting the search)."""


# A* local path snapshot as a bitmask: one bit per cell of a node's 3x3x3 window minus its
# own column (8 horizontal neighbours x Y-1/0/+1) — exactly the cells move conditions ask
# about. Cells outside the window are never in a snapshot.
_SNAP_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (ox, oy, oz) for oy in (-1, 0, 1) for ox in (-1, 0, 1) for oz in (-1, 0, 1) if ox or oz)
_SNAP_BIT: dict[tuple[int, int, int], int] = {o: 1 << i for i, o in enumerate(_SNAP_OFFSETS)}
_SNAP_MASK = (1 << len(_SNAP_OFFSETS)) - 1
# Node key bit set when the node was entered by a tower move. Tower-gated moves depend on it,
# and the launch cell is outside the window, so the snapshot alone can't tell a tower arrival
# from a +1Y arrival at the same cell.
_TOWER_FLAG = 1 << len(_SNAP_OFFSETS)
# Signal state (see _signal_depths): 4 bits of signal depth since the last cell a repeater
# could take, then 3 bits of the repeater-candidate direction (index into _HORIZ_DIRS + 1, 0 none).
_SIG_SHIFT = len(_SNAP_OFFSETS) + 1
_SIG_DEPTH_MASK = 15
_NODE_SHIFT = _SIG_SHIFT + 7  # node = cell id << _NODE_SHIFT | candidate dir | depth | tower flag | snapshot
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


def set_foot_repeaters(on: bool) -> None:
    """Allow repeaters at a slope's foot. They can read a glass or powered block under the slope, which verify and repair
    catch, but without them a design of routed banks can fail routing on every seed. Off for
    designs with RAM blocks, where nearly every net has to be right first time."""
    _foot_repeaters[0] = on
    _astar.set_foot_repeaters(on)


def _signal_step(depth: int, cand: int, d: tuple[int, int, int]) -> tuple[int, int]:
    """Signal state after move d from a cell with (depth, cand); the kernel's rule.

    depth: signal distance since the last cell where _place_repeaters_for_net could put a
    repeater (or a tower / powered -4 reset). cand: 1 + the _HORIZ_DIRS index of the move
    into the cell when that move lets it hold a repeater (flat unit step), else 0;
    the cell becomes a repeater slot once the next move is flat in that same direction.
    The search prunes moves whose depth reaches _REPEATER_INTERVAL, so every stretch of a
    path has a slot in reach and repeater placement can't fail for want of one.
    """
    dx, dy, dz = d
    if (dy == 4 and (dx, dz) in _TOWER_2BLOCK) or (dy == -4 and dx == 0 and dz == 0):
        return 1, 0  # laying resets one cell short of the new source, like the repeater walk
    step = abs(dx) + abs(dz)
    if cand and dy == 0 and step == 1 and _HORIZ_DIRS[cand - 1] == (dx, dz):
        depth = 0
    unit = step == 1 and (dy == 0 or (_foot_repeaters[0] and dy == -1))
    return depth + step, (_HORIZ_DIRS.index((dx, dz)) + 1) if unit else 0


def _signal_depths(
        paths: list[list[tuple[int, int, int]]],
        known: dict[tuple[int, int, int], int] | None = None,
        in_place: bool = False,
) -> dict[tuple[int, int, int], int]:
    """Signal depth (_signal_step) at each cell of a net's ordered paths, for tree seeds.

    Each path starts at the source (depth 0) or at a cell of an earlier path or of known
    (depths of the net's wire routed before). A branch start is never a repeater slot.
    in_place: add to known instead of copying it.
    """
    depths: dict[tuple[int, int, int], int] = known if in_place else dict(known or {})
    for path in paths:
        depth, cand = depths.get(path[0], 0), 0
        depths.setdefault(path[0], depth)
        for prev, pos in zip(path, path[1:]):
            depth, cand = _signal_step(depth, cand, (pos[0] - prev[0], pos[1] - prev[1], pos[2] - prev[2]))
            depths.setdefault(pos, depth)
    return depths


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


_KINDS: dict[str, int] = {}  # block id -> _BlockIndex.kinds bits


def _kind_of(block_id: str) -> int:
    """_BlockIndex.kinds' bits for a block id: every test it makes reads the id alone."""
    kind = _KINDS.get(block_id)
    if kind is None:
        block = BlockState(block_id)
        kind = _KINDS[block_id] = ((0 if _is_air(block) else 1) | (2 if _is_redstone_wire(block) else 0)
                                   | (4 if _is_torch(block) else 0) | (8 if _is_repeater(block) else 0)
                                   | (16 if block == GLASS else 0))
    return kind


class _BlockIndex:
    """Sparse sets of signal-relevant cells, built from the Region's raw block array.

    litemapy's Region.__getitem__ is slow (coordinate conversion + palette lookup per
    read); _wire_walkable does ~40 reads per call. Valid only while the workspace is
    unchanged, i.e. for the duration of one A* search.
    """
    __slots__ = ("non_air", "wire", "torch", "repeater", "grid", "grid_bits")

    def __init__(self, region: Region):
        # Positive-size regions store blocks at region coordinates directly.
        assert region.width > 0 and region.height > 0 and region.length > 0
        k = self.kinds(region)

        def cells(mask: int) -> set[tuple[int, int, int]]:
            return set(map(tuple, np.argwhere(k & mask).tolist()))

        self.non_air = cells(1)
        self.wire = cells(2)
        self.torch = cells(4)
        self.repeater = cells(8)
        self.grid = None  # _astar.CellGrid mirroring these sets; update keeps it current

    @classmethod
    def empty(cls) -> _BlockIndex:
        """An index of nothing, for an _astar.CellGrid to fill with views of itself."""
        idx = cls.__new__(cls)
        idx.non_air, idx.wire, idx.torch, idx.repeater = set(), set(), set(), set()
        idx.grid = None
        return idx

    @staticmethod
    def kinds_table(region: Region) -> np.ndarray:
        """Per palette entry: the kind bits (see kinds)."""
        return np.array([
            (0 if _is_air(b) else 1) | (2 if _is_redstone_wire(b) else 0)
            | (4 if _is_torch(b) else 0) | (8 if _is_repeater(b) else 0) | (16 if b == GLASS else 0)
            for b in region._Region__palette
        ], dtype=np.uint8)

    @staticmethod
    def kinds(region: Region, lo: int = 0, hi: int | None = None) -> np.ndarray:
        """Per cell of region (of the x slab lo:hi): 1 not air | 2 wire | 4 torch | 8 repeater | 16 GLASS (the grid's alone)."""
        return lookup(region._Region__blocks[lo:hi], _BlockIndex.kinds_table(region))

    @classmethod
    def layered(cls, base: _BlockIndex, overlay: dict[tuple[int, int, int], BlockState]) -> _BlockIndex:
        """base (static world) plus a net's overlay blocks. Overlay only ever fills air in base."""
        idx = cls.__new__(cls)
        idx.non_air = set(base.non_air)
        idx.wire = set(base.wire)
        idx.torch = set(base.torch)
        idx.repeater = set(base.repeater)
        idx.grid = None
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
        was, now = _kind_of(old.id), _kind_of(new.id)
        if was == now:
            return
        grid = self.grid
        if grid is not None:  # the sets are views of it
            grid.put_kind(pos, was ^ now, now)
            return
        for i, cells in enumerate((self.non_air, self.wire, self.torch, self.repeater)):
            bit = 1 << i
            if was & bit and not now & bit:
                cells.discard(pos)
            elif now & bit and not was & bit:
                cells.add(pos)


class IndexedWorkspace:
    """Write-through Region proxy that keeps a _BlockIndex current across searches.

    Building a _BlockIndex scans the whole region; with one search per sink that dominated
    the greedy pass on big designs. Every write must go through this proxy, or the index
    goes stale. Searches read the live index, so the region must not change mid-search.
    """

    def __init__(self, region: Region):
        self.region = region
        self._built: _BlockIndex | None = None  # built when first used: attach_grid needs none of its sets
        self._write = palette_writer(region)

    @property
    def _index(self) -> _BlockIndex:
        if self._built is None:
            self._built = _BlockIndex(self.region)
        return self._built

    def __getitem__(self, pos) -> BlockState:
        # Region.__getitem__ without its coordinate conversion, a no-op for a positive-size region. Saving a schematic
        # replaces the palette, so it's looked up each time.
        region = self.region
        return region._Region__palette[region._Region__blocks[pos]]

    def __setitem__(self, pos, block: BlockState) -> None:
        self.put(tuple(pos), block)

    def put(self, pos: tuple[int, int, int], block: BlockState) -> BlockState:
        """self[pos] = block, returning the block it replaced."""
        region = self.region
        old = region._Region__palette[region._Region__blocks[pos]]
        (self._built or self._index).update(pos, old, block)
        self._write(pos, block)
        return old

    def __getattr__(self, name):
        return getattr(self.region, name)

    def block_index(self) -> _BlockIndex:
        return self._index

    def attach_grid(self, solid, dust_owner, inverted_cells, footprint_blocked, terminal_positions) -> None:
        """Have searches over this workspace read an _astar.CellGrid of these containers, which
        from here on keep it current. Containers that don't (plain sets) leave searches reading
        them directly. _ASTAR_GRID: 0 no grid, check compares every cell's moves."""
        if (_ASTAR_GRID == "0"
                or not all(hasattr(c, "_grid") for c in (solid, dust_owner, inverted_cells))):
            return
        region = self.region
        if self._built is None:  # the grid replaces the sets with views of itself: millions of cells not to collect
            self._built = _BlockIndex.empty()
        _astar.CellGrid(region.width, region.height, region.length, functools.partial(_BlockIndex.kinds, region), self._built, solid, dust_owner,
                        inverted_cells, footprint_blocked, terminal_positions)


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
        footprint_relief: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """Pass a _BlockIndex when calling repeatedly against an unchanged workspace.

    footprint_blocked minus footprint_relief exempts cells from the neighbour wire/torch check.

    The rule in Python, for reading: searches use the kernel's World.wire_walkable (change both when the rule
    changes)."""
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
            if owner is None and (np not in footprint_blocked or np in footprint_relief):
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
        weight: float = 1.75,
        limit_exclusions=False,
        x_h_factor=1,
        congestion_cost: Callable[[tuple[int, int, int]], float] | None = None,
        time_limit: float | None = None,
        blocked: frozenset[tuple[int, int, int]] = frozenset(),
        seed_depths: dict[tuple[int, int, int], int] | None = None,
        pending_sinks: frozenset[tuple[int, int, int]] = frozenset(),
        stagnation: int = -1,
        budget: int = 1,
        start_depth: int = 0,
        goal_depth: int = 15,
        mark_failure: bool = True,
) -> list[tuple[int, int, int]]:
    """A* wire route from start to goal.

    start_depth: signal depth (_signal_step) at start; the path is good from any cell reached at this depth or less.
    goal_depth: the path reaches goal at this signal depth or less.
    mark_failure: a failed search marks its way with lamps in workspace.

    stagnation: this search's stagnation limit (negative: _ROUTE_STAGNATION). Kernel only.

    budget: multiplies the search's stagnation, position and expansion limits, and the f-cap: a walled-in short net's
    way round can be many times its straight distance. Kernel only.

    seed_depths: signal depth (_signal_depths) of tree seeds; missing seeds count as 0.

    pending_sinks: this net's sinks still to route; inverted cells stay clear of their approaches.

    blocked: extra cells the path must not use, even inside the goal's relief zone (a
    retry after a path that couldn't be laid, see wire.LayConflict).

    congestion_cost: extra cost per path cell. Excluded from the f-cap budget.
    time_limit: wall-clock seconds for the search; on expiry it fails like any other early stop.
    workspace may be a view exposing block_index() (see IndexedWorkspace).
    """
    min_y = bounds[1]
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
    if blocked:
        protected = _with_blocked(protected, blocked)
        effective_protected = _with_blocked(effective_protected, blocked)

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
    # Effective footprint blocking = footprint_blocked minus fp_relief. Kept as the two sets
    # (footprint_blocked can be ~200k cells; copying it per search dominated big designs).

    block_index = workspace.block_index() if hasattr(workspace, "block_index") else _BlockIndex(workspace)
    grid = getattr(block_index, "grid", None)
    ix_non_air, ix_wire, ix_torch, ix_repeater = grid.kind_sets() if grid is not None else (
        block_index.non_air, block_index.wire, block_index.torch, block_index.repeater)

    def heuristic(pos: tuple[int, int, int]) -> int:
        dx = abs(pos[0] - goal[0]) * x_h_factor
        dz = abs(pos[2] - goal[2])
        dy = abs(pos[1] - goal[1])
        return dx + dy * 4 + dz

    # Move generation is the compiled _astar.World. Cells get small int ids (pos_list[id] = cell) so a search node is one int,
    # id << _NODE_SHIFT | tower flag | snapshot, instead of a nested tuple that is re-hashed on every lookup.
    pos_list: list[tuple[int, int, int]] = []
    h_list: list[int] = []  # heuristic(pos_list[id])
    pos_ids: dict[tuple[int, int, int], int] = {}
    move_cache: list[tuple[list[tuple], tuple | None] | None] = []  # by cell id

    def _pid(p: tuple[int, int, int]) -> int:
        i = pos_ids.get(p)
        if i is None:
            i = pos_ids[p] = len(pos_list)
            pos_list.append(p)
            h_list.append(heuristic(p))
            move_cache.append(None)
        return i

    def make_world():
        """The kernel's move generation over this search's world."""
        eprot, prot = _keepout_spec(effective_protected), _keepout_spec(protected)
        if eprot is None or prot is None or not isinstance(dust_owner, dict):
            raise TypeError("the A* kernel reads protected cells as a set or a _ForeignKeepout and dust_owner as a dict, "
                            f"not {type(effective_protected).__name__}, {type(protected).__name__}, {type(dust_owner).__name__}")
        return _astar.World(
            ix_non_air, ix_wire, ix_torch, ix_repeater, solid, dust_owner, net_id, bounds, max_bridge_y,
            goal, x_h_factor, eprot, prot, fp_relief, footprint_blocked, inverted_cells,
            terminal_positions, workspace, GLASS, _FACING,
            pos_ids, pos_list, h_list, move_cache, pending_sinks,
            getattr(block_index, "grid", None), _ASTAR_GRID == "check")

    # The first search's World answers walkable() for the start and goal too: the same rule, compiled.
    first_world = [make_world()]
    walk_at = first_world[0].walkable_at

    def next_world():
        return first_world.pop() if first_world else make_world()

# Seed cells: prefer start itself; fall back to neighbors when blocked
    seeds: list[tuple[int, int, int]] = []
    if walk_at(start):
        seeds.append(start)
    else:
        for ddx, ddy, ddz in _DIRS_6:
            cand = (start[0] + ddx, start[1] + ddy, start[2] + ddz)
            if walk_at(cand):
                seeds.append(cand)
    # Seeds in order, with their group (0 = start seeds, then one per tree seed): existing own-net wire cells are
    # valid branch seeds. The World keeps their ids, which the search's cells continue from pos_list's, unless moves
    # go through Python (a grid check, congestion).
    in_world = congestion_cost is None and _ASTAR_GRID != "check"
    if in_world:
        n_seeds, seed_nodes, seed_groups, group_best_h, best_id, best_h, best_seed = first_world[0].add_seeds(
            seeds, tree_seeds, seed_depths or {})
    else:
        seed_ids, seed_nodes, seed_groups, group_best_h, best = _astar.prepare_seeds(
            seeds, tree_seeds, solid, tuple(bounds), pos_ids, pos_list, h_list, move_cache, gx, gy, gz, x_h_factor,
            seed_depths or {})
        n_seeds, best_id, best_seed = len(seeds), seed_ids[best], seeds[best]
        best_h = h_list[best_id]
    if start_depth:
        seed_nodes = [n | (start_depth << _SIG_SHIFT) if g == 0 else n for n, g in zip(seed_nodes, seed_groups)]
    if not n_seeds:
        raise ValueError(f"No walkable cell near start {start} for net {net_id}")

    # Goal cells: goal exclusion already removed from effective_protected above,
    # so walkable() correctly admits the goal and its approach cells.
    goal_set: set[tuple[int, int, int]] = set()
    goal_walkable = walk_at(goal)
    if goal_walkable:
        goal_set.add(goal)
    if not goal_set:
        raise ValueError(f"No walkable cell near goal {goal} for net {net_id}")

    goal_ids = {first_world[0].cell_id(gc) if in_world else _pid(gc) for gc in goal_set}

    def node_str(node: int) -> str:
        return str((pos_list[node >> _NODE_SHIFT], node & _SNAP_MASK))

    best_node = best_id << _NODE_SHIFT
    group_max_f: list[int] = [max(gh, _ROUTE_BUDGET_FLOOR) * _ROUTE_BUDGET_FACTOR * budget for gh in group_best_h]
    initial_group_best_h = list(group_best_h)
    search_args = (seed_nodes, seed_groups, goal_ids, group_best_h, group_max_f, best_node, best_h,
                   weight, time_limit, pos_list, h_list, None, congestion_cost, goal_depth)

    # Most of a long search's expansions are its many states per cell (snapshots, signal depths): the first few of
    # a cell almost always route as well. A search that fails that way searches again with all of them, unless
    # it's a tight first try, whose failure only means ripping what's around the sink and trying again, or it ran
    # out of time, which a second search would only spend again.
    reached, best_node, h_radius_pruned, early_stop_reason, came_from = _run_search(
        search_args, net_id, next_world, stagnation=stagnation, budget=budget, cell_cap=_ROUTE_CELL_STATES)
    timed_out = early_stop_reason is not None and early_stop_reason.startswith("timeout")
    if reached is None and _ROUTE_CELL_STATES and stagnation < 0 and not timed_out:
        group_best_h[:] = initial_group_best_h
        reached, best_node, h_radius_pruned, early_stop_reason, came_from = _run_search(
            search_args, net_id, next_world, stagnation=stagnation, budget=budget)
    if reached is None and early_stop_reason is None and h_radius_pruned:
        # The open list ran dry only because the h-radius cull discarded detours (typically a
        # blocked approach close to the goal): search again without the cull.
        group_best_h[:] = initial_group_best_h
        reached, best_node, _, early_stop_reason, came_from = _run_search(
            search_args, net_id, next_world, h_radius_factor=0, stagnation=stagnation, budget=budget)

    if reached is None:
        lamp = BlockState("minecraft:redstone_lamp")
        cur = best_node if mark_failure else _NO_PARENT
        # Seeds only enter came_from once expanded; an early stop (f-cap) can precede that.
        while cur is not None and cur != _NO_PARENT:
            cx, cy, cz = pos_list[cur >> _NODE_SHIFT]
            workspace[cx, cy, cz] = lamp
            cur = came_from.get(cur)
        # for prot in sorted((footprint_blocked - fp_relief) | effective_protected, key=heuristic)[:100]:
        #     if _is_air(workspace[*prot]):
        #         workspace[*prot] = GLASS
        reason = f"; {early_stop_reason}" if early_stop_reason else ""
        error = RouteTimeout if early_stop_reason and early_stop_reason.startswith("timeout") else ValueError
        raise error(f"No route for net {net_id} from {start} to {goal} (closest reached: {node_str(best_node)}{reason}; h_radius_pruned={h_radius_pruned})")
    def pos_of(cid: int) -> tuple[int, int, int]:
        """A successful kernel search leaves the cells it made in its own vectors, not pos_list."""
        return pos_list[cid] if cid < len(pos_list) else came_from.pos(cid)

    if goal_walkable and pos_of(reached >> _NODE_SHIFT) != goal:
        raise ValueError(f"No route reached goal {goal} for net {net_id}; stopped at {pos_of(reached >> _NODE_SHIFT)}")

    path: list[tuple[int, int, int]] = came_from.path(reached, pos_list)
    came_from.release()
    if not tree_seeds and path[0] != start:
        raise ValueError(f"Path for net {net_id} does not begin at start {start}; begins at {path[0]}")
    _check_tower_self_overlap(path, net_id)
    return path


def _check_tower_self_overlap(path: list[tuple[int, int, int]], net_id: str) -> None:
    """Raise if a tower move's repeater (or the cell above it) or torch/stone column lands on
    another cell of the same path. The local snapshot can't see the column (2 cells out) or
    path cells that have left the window, and laying would silently drop the torch."""
    path_cells = set(path)
    for (x, y, z), (bx, by, bz) in zip(path, path[1:]):
        if by - y != 4 or (bx - x, bz - z) not in _TOWER_2BLOCK:
            continue
        rx, rz = (x + bx) // 2, (z + bz) // 2
        cells = [(rx, y, rz), (rx, y + 1, rz)] + [(bx, y + cdy, bz) for cdy in range(4)]
        for c in cells:
            if c in path_cells:
                # A LayConflict, so the router searches again without the overlapped cell.
                raise LayConflict(f"Tower move {(x, y, z)} -> {(bx, by, bz)} for net {net_id} overlaps its own path at {c}",
                                  (c, (bx, by, bz)))


_ASTAR_GRID = "1"  # "0": searches read the containers, not a CellGrid; "check": compare every cell's moves
try:
    from . import _astar
except ImportError as exc:
    raise ImportError(f"the compiled A* kernel (minecraft_v.placement_engine._astar) is missing: {exc}. "
                      "Run `uv sync` to build it.") from exc
_astar.set_subset_dominance(True)
_astar.init(_SNAP_OFFSETS, _TOWER_FLAG, _NODE_SHIFT, _ROUTE_STAGNATION, _ROUTE_MAX_NODES,
            _ROUTE_MAX_EXPANSIONS, _H_RADIUS_FACTOR, _HORIZ_DIRS, _HORIZ_RADIUS_MIN, _Y_RADIUS_MIN,
            _SIG_SHIFT, _REPEATER_INTERVAL)


def _with_blocked(keepout, blocked: frozenset[tuple[int, int, int]]):
    if hasattr(keepout, "with_extra"):  # builder._ForeignKeepout
        return keepout.with_extra(blocked)
    return frozenset(keepout) | blocked


def _keepout_spec(keepout) -> tuple | None:
    """A protected-cells container in a form the kernel reads directly, or None if unsupported."""
    if isinstance(keepout, (set, frozenset)):
        return ("set", keepout)
    if hasattr(keepout, "_count"):  # builder._ForeignKeepout
        return ("keepout", keepout._count, keepout._own, keepout._excluded, keepout._extra,
                keepout._halo, keepout._own_halo)
    return None


def _run_search(args: tuple, net_id: str, make_world, h_radius_factor: float = -1, stagnation: int = -1,
                budget: int = 1, cell_cap: int = 0):
    """h_radius_factor: the kernel's h-radius cull for this search (0: off; negative: the default)."""
    *args, goal_depth = args
    # A tight first try (stagnation set) gives up sooner on positions too: past a few thousand it almost never succeeds,
    # and a failure there only means ripping what's around the sink and trying again.
    return _astar.search(*args, make_world(), h_radius_factor=h_radius_factor, stagnation=stagnation, budget=budget,
                         position_cap=_ROUTE_TIGHT_POSITIONS if stagnation >= 0 else 0,
                         cell_cap=cell_cap, goal_depth=goal_depth)
