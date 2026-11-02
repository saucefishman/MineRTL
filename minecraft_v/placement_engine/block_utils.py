from __future__ import annotations
from collections.abc import Callable, Iterable
from math import ceil, log

import numpy as np
from litemapy import BlockState, Region
from nbtlib.tag import Compound, Int, List, LongArray

from .constants import STONE

Box = tuple[tuple[int, int, int], tuple[int, int, int]]  # lowest cell, one past the highest


def palette_writer(region: Region) -> Callable[[tuple[int, int, int], BlockState], None]:
    """region[pos] = block without Region's linear search of its palette, which a big build pays per block."""
    palette, blocks = region._Region__palette, region._Region__blocks
    ids = {block: i for i, block in reversed(list(enumerate(palette)))}
    known = len(palette)

    def write(pos: tuple[int, int, int], block: BlockState) -> None:
        nonlocal known
        if len(palette) != known:  # written to directly since
            ids.update((b, i) for i, b in enumerate(palette[known:], known))
            known = len(palette)
        i = ids.get(block)
        if i is None:
            palette.append(block)
            i = ids[block] = known
            known += 1
        blocks[pos] = i

    return write


def _pack_states(blocks: np.ndarray, nbits: int) -> LongArray:
    """Palette ids as Litematica's bit array: y, then z, then x, lowest bits first, values crossing long boundaries."""
    n_x, n_y, n_z = blocks.shape
    out = np.zeros(ceil(n_x * n_y * n_z * nbits / 64), dtype="<u8")
    done = 0  # longs written
    held = np.zeros(0, dtype=np.uint8)  # bits short of a whole long
    for y in range(n_y):
        ids = blocks[:, y, :].T.ravel().astype("<u4")
        bits = np.unpackbits(ids.view(np.uint8).reshape(-1, 4), axis=1, bitorder="little")[:, :nbits].ravel()
        if held.size:
            bits = np.concatenate((held, bits))
        whole = bits.size // 64
        out[done:done + whole] = np.frombuffer(np.packbits(bits[:whole * 64], bitorder="little").tobytes(), dtype="<u8")
        done += whole
        held = bits[whole * 64:]
    if held.size:
        out[done] = np.frombuffer(np.packbits(np.concatenate((held, np.zeros(64 - held.size, np.uint8))),
                                              bitorder="little").tobytes(), dtype="<u8")[0]
    return LongArray(out.astype("<i8"))


class BulkRegion(Region):
    """A Region with what a large build does in bulk done in numpy: Region.to_nbt loops over every cell of its volume."""

    def paste(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, ids: np.ndarray, blocks: list[BlockState]) -> None:
        """Writes blocks[ids[i]] at each (xs[i], ys[i], zs[i])."""
        palette = self._Region__palette
        known = {block: i for i, block in reversed(list(enumerate(palette)))}
        table = np.empty(len(blocks), dtype=self._Region__blocks.dtype)
        for i, block in enumerate(blocks):
            at = known.get(block)
            if at is None:
                at = known[block] = len(palette)
                palette.append(block)
            table[i] = at
        self._Region__blocks[xs, ys, zs] = table[ids]

    def _optimize_palette(self) -> None:
        palette, blocks = self._Region__palette, self._Region__blocks
        used = np.zeros(len(palette), dtype=bool)
        used[np.unique(blocks)] = True
        used[0] = True  # air stays at 0
        kept: dict[BlockState, int] = {}
        table = np.empty(len(palette), dtype=blocks.dtype)
        for i in np.flatnonzero(used).tolist():
            table[i] = kept.setdefault(palette[i], len(kept))
        blocks[...] = table[blocks]
        self._Region__palette = list(kept)

    def to_nbt(self) -> Compound:
        self._optimize_palette()
        palette = self._Region__palette
        root = Compound()
        root["Position"] = Compound({"x": Int(self.x), "y": Int(self.y), "z": Int(self.z)})
        root["Size"] = Compound({"x": Int(self.width), "y": Int(self.height), "z": Int(self.length)})
        root["BlockStatePalette"] = List[Compound]([block.to_nbt() for block in palette])
        root["Entities"] = List[Compound]([entity.to_nbt() for entity in self.entities])
        root["TileEntities"] = List[Compound]([entity.to_nbt() for entity in self.tile_entities])
        root["PendingBlockTicks"] = List[Compound](self.block_ticks)
        root["PendingFluidTicks"] = List[Compound](self.fluid_ticks)
        root["BlockStates"] = _pack_states(self._Region__blocks, max(ceil(log(len(palette), 2)), 2))
        return root


def cells_around_boxes(region: Region, boxes: Iterable[Box], reach: int) -> list[tuple[int, int, int]]:
    """Non-air cells outside each box and within reach of its faces, the only cells that can touch what's inside it."""
    blocks = region._Region__blocks
    air = [i for i, b in enumerate(region._Region__palette) if b.id in ("minecraft:air", "minecraft:cave_air")]
    found: set[tuple[int, int, int]] = set()
    for lo, hi in boxes:
        grown_lo = [max(c - reach, 0) for c in lo]
        grown_hi = [min(c + reach, n) for c, n in zip(hi, blocks.shape)]
        for axis in range(3):
            for start, stop in ((grown_lo[axis], lo[axis]), (hi[axis], grown_hi[axis])):
                if start >= stop:
                    continue
                corner = list(grown_lo)
                corner[axis] = start
                slab = tuple(slice(a, b) for a, b in zip(corner, (*grown_hi[:axis], stop, *grown_hi[axis + 1:])))
                hits = np.argwhere(~np.isin(blocks[slab], air)) + corner
                found.update(map(tuple, hits.tolist()))
    return sorted(found)


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
