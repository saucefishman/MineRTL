from __future__ import annotations
from litemapy import BlockState, Region
from .constants import STONE


def _block_str(block: BlockState) -> str:
    return str(block)


# Classifiers use BlockState.id (block name only): str(block) formats every property and
# dominated routing time when called millions of times from _wire_conflicts.
def _is_air(block: BlockState) -> bool:
    return block.id == "minecraft:air"


def _is_redstone_wire(block: BlockState) -> bool:
    return "redstone_wire" in block.id


def _is_repeater(block: BlockState) -> bool:
    return block.id == "minecraft:repeater"


_TORCH_IDS = frozenset(("minecraft:redstone_torch", "minecraft:redstone_wall_torch"))


def _is_torch(block: BlockState) -> bool:
    return block.id in _TORCH_IDS

def _is_glass(block: BlockState) -> bool:
    return "minecraft:glass" in block.id


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
