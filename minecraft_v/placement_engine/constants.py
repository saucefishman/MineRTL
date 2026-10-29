from __future__ import annotations

from litemapy import BlockState

from minecraft_v.placement_engine.ir import CardinalDirection

AIR = BlockState("minecraft:air")
STONE = BlockState("minecraft:stone")
REDSTONE = BlockState("minecraft:redstone_wire")
GLASS = BlockState("minecraft:glass")
REDSTONE_TORCH = BlockState("minecraft:redstone_torch")
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
_ROUTE_MAX_NODES = 100_000   # hard cap on unique XYZ positions expanded before declaring failure
_ROUTE_STAGNATION = 60_000    # consecutive re-expansions of same (pos,snap) → early cut
_ROUTE_BUDGET_FACTOR = 5     # f-score cap: max_f = initial_h * factor; prunes runaway detours
_H_RADIUS_FACTOR = 2        # h-radius cull: discard nodes where h > best_h * factor
_HORIZ_RADIUS_MIN, _Y_RADIUS_MIN = 10, 6

# Negotiated-congestion routing (PathFinder)
_ROUTE_MAX_PASSES = 1              # 1 = single greedy hard pass (default); N>1 = N-1 negotiation passes + final hard pass
_ROUTE_BATCH_PER_WORKER = 4        # parallel negotiation: nets per batch = this × --route-workers
_PRES_SCHEDULE = (2.0, 4.0, 8.0, 16.0, 32.0)  # present-congestion factor per negotiation pass (last value repeats)
_CONFLICT_BASE = 4.0               # per-cell congestion cost = base * (hist + (1 + hist) * pres * other_users)
_ROUTE_STAGNATION_PASSES = 2       # stop negotiating after this many passes without a better score
_ROUTE_SEARCH_TIMEOUT = 0.3        # per-A*-search wall-clock limit (s) in a net's first negotiation attempt
_ROUTE_TIMEOUT_ESCALATION = 3.0    # after a timeout, that net's limit is multiplied by this (up to the final limit)
_ROUTE_FINAL_SEARCH_TIMEOUT = 3.0  # per-search limit in the final hard pass (and the escalation cap)

_HORIZ_DIRS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_HORIZ_DIRS_8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
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

# XZ offsets that constitute a 2-block horizontal step (used by tower and 2x-slope moves)
_TOWER_2BLOCK: frozenset[tuple[int, int]] = frozenset([(2, 0), (-2, 0), (0, 2), (0, -2)])

_REPEATER_MAX_DELAY = 4
