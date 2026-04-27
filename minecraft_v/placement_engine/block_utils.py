from __future__ import annotations
from litemapy import BlockState, Region
from .constants import STONE


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
    s = _block_str(block)
    return "minecraft:redstone_torch" in s or "minecraft:redstone_wall_torch" in s


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
