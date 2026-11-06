from __future__ import annotations
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from math import ceil, log
import struct
import time
import zlib

import nbtlib
import numpy as np
from litemapy import BlockState, Region, Schematic
from nbtlib.tag import Compound, Int, List, LongArray

from .constants import STONE

try:
    from . import _signals
except ImportError:
    _signals = None  # not built: numpy does the same, slower

_GZIP_CHUNK = 1 << 23
_GZIP_LEVEL = 6  # 9 takes ten times as long for an 11% smaller file
_GZIP_THREADS = 8

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
    if _signals is not None and blocks.dtype == np.uint32 and blocks.size and blocks.flags.c_contiguous:
        _signals.pack_states(blocks, nbits, out)
        return LongArray(out.astype("<i8"))
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


def _gzip_threaded(data: memoryview, path: str) -> None:
    """One gzip member, deflated in pieces on threads, each piece ending on a byte boundary as pigz does."""
    pieces = [data[i:i + _GZIP_CHUNK] for i in range(0, len(data), _GZIP_CHUNK)] or [data]

    def deflate(index: int) -> bytes:
        c = zlib.compressobj(_GZIP_LEVEL, zlib.DEFLATED, -zlib.MAX_WBITS)
        last = index == len(pieces) - 1
        return c.compress(pieces[index]) + c.flush(zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH)

    with ThreadPoolExecutor(_GZIP_THREADS) as pool, open(path, "wb") as out:
        out.write(struct.pack("<BBBBIBB", 0x1f, 0x8b, 8, 0, int(time.time()), 2, 255))
        for piece in pool.map(deflate, range(len(pieces))):
            out.write(piece)
        out.write(struct.pack("<II", zlib.crc32(data), len(data) & 0xffffffff))
        for piece in pieces:
            piece.release()


def save_schematic(schematic: Schematic, path: str) -> None:
    """Schematic.save, the same NBT, but compressed on several threads."""
    schematic.update_metadata()
    buffer = BytesIO()
    nbtlib.File(schematic.to_nbt(save_soft=True), gzipped=True, byteorder="big").write(buffer, "big")
    with buffer.getbuffer() as data:
        _gzip_threaded(data, path)


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
        kernel = _signals is not None and blocks.dtype == np.uint32 and blocks.size and blocks.flags.c_contiguous
        if kernel:
            used = _signals.used_ids(blocks, len(palette))
        else:
            used = np.zeros(len(palette), dtype=bool)
            used[np.unique(blocks)] = True
        used[0] = True  # air stays at 0
        kept: dict[BlockState, int] = {}
        table = np.empty(len(palette), dtype=blocks.dtype)
        for i in np.flatnonzero(used).tolist():
            table[i] = kept.setdefault(palette[i], len(kept))
        if kernel:
            _signals.remap_ids(blocks, table)
        else:
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
    return list(map(tuple, _around_boxes(region, boxes, reach).tolist()))


def _around_boxes(region: Region, boxes: Iterable[Box], reach: int) -> np.ndarray:
    """cells_around_boxes as a sorted n x 3 array."""
    blocks = region._Region__blocks
    air = [i for i, b in enumerate(region._Region__palette) if b.id in ("minecraft:air", "minecraft:cave_air")]
    found: list[np.ndarray] = []
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
                found.append(np.argwhere(~np.isin(blocks[slab], air)) + corner)
    if not found:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.concatenate(found), axis=0) if len(found) > 1 else found[0]


def cells_in_boxes_near(region: Region, boxes: Iterable[Box], reach: int, offsets: Iterable[tuple[int, int, int]]) -> np.ndarray:
    """Cells inside the boxes, sorted, an offset away from a cell around them: what those cells could touch."""
    boxes = list(boxes)
    around = _around_boxes(region, boxes, reach)
    sx, sy, sz = region._Region__blocks.shape
    moved = (around[:, None, :] + np.array(list(offsets), dtype=np.int64)[None, :, :]).reshape(-1, 3)
    inside = np.zeros(len(moved), dtype=bool)
    for lo, hi in boxes:
        inside |= ((moved >= lo) & (moved < hi)).all(axis=1)
    moved = moved[inside]
    moved = moved[((moved >= 0) & (moved < (sx, sy, sz))).all(axis=1)]
    return np.unique(moved, axis=0)


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
