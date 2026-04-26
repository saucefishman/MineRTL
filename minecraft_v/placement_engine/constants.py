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
_ROUTE_MAX_NODES = 100_000   # hard cap on A* expansions before declaring failure
_ROUTE_STAGNATION = 3_000    # expansions without best_h improvement → early cut

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
