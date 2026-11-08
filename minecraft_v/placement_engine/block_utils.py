from __future__ import annotations
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from math import ceil, log
import struct
import time
import zlib

import nbtlib
import numpy as np
from litemapy import BlockState, Region, Schematic
from nbtlib.tag import INT, Compound, Int, List, LongArray, write_numeric

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
    # By object first: most writes are a few shared constants, and a BlockState hashes in Python. Holds what it saw.
    by_object: dict[int, tuple[BlockState, int]] = {}

    def write(pos: tuple[int, int, int], block: BlockState) -> None:
        nonlocal known
        hit = by_object.get(id(block))
        if hit is not None and hit[0] is block:
            blocks[pos] = hit[1]
            return
        if len(palette) != known:  # written to directly since
            ids.update((b, i) for i, b in enumerate(palette[known:], known))
            known = len(palette)
        i = ids.get(block)
        if i is None:
            palette.append(block)
            i = ids[block] = known
            known += 1
        by_object[id(block)] = (block, i)
        blocks[pos] = i

    return write


def palette_table(region: Region, values) -> np.ndarray:
    """A per palette entry list of small ints (or bools) as the uint8 table the lookups below take."""
    return np.asarray(values, dtype=np.uint8).reshape(len(region._Region__palette))


def _compiled(blocks: np.ndarray, *functions: str) -> bool:
    """Whether the compiled module has these functions and can take blocks (an older build may lack some)."""
    return _signals is not None and all(hasattr(_signals, f) for f in functions) and blocks.dtype == np.uint32 and blocks.flags.c_contiguous and blocks.size > 0


def _check_ids(blocks: np.ndarray, size: int) -> None:
    """The compiled helpers index their tables unchecked: every id in blocks must be below size."""
    assert int(blocks.max()) < size, f"block id {int(blocks.max())} outside a table of {size}"


def lookup(blocks: np.ndarray, table: np.ndarray) -> np.ndarray:
    """table[blocks], table a uint8 array per palette entry."""
    if _compiled(blocks, "table_gather"):
        _check_ids(blocks, len(table))
        return _signals.table_gather(blocks, np.ascontiguousarray(table, dtype=np.uint8))
    return table[blocks]


def cells_where(blocks: np.ndarray, table: np.ndarray, skip: Iterable[Box] = ()) -> np.ndarray:
    """np.argwhere(table[blocks]) less the cells in skip's boxes, as an n x 3 int64 array in argwhere's order."""
    skip = [(tuple(int(c) for c in lo), tuple(int(c) for c in hi)) for lo, hi in skip]
    if _compiled(blocks, "table_where"):
        _check_ids(blocks, len(table))
        return _signals.table_where(blocks, np.ascontiguousarray(table, dtype=np.uint8), skip)
    mask = np.asarray(table, dtype=bool)[blocks]
    for lo, hi in skip:
        mask[max(lo[0], 0):max(hi[0], 0), max(lo[1], 0):max(hi[1], 0), max(lo[2], 0):max(hi[2], 0)] = False
    return np.argwhere(mask)


class _ChunkedLongArray(LongArray):
    """A LongArray that writes itself in pieces: Array.write byteswaps and copies the whole array twice over."""
    _WRITE_LONGS = 1 << 20

    def write(self, fileobj, byteorder="big"):
        write_numeric(INT, len(self), fileobj, byteorder)
        for i in range(0, len(self), self._WRITE_LONGS):
            piece = np.asarray(self[i:i + self._WRITE_LONGS])
            fileobj.write((piece if self.item_type[byteorder] == piece.dtype else piece.byteswap()).tobytes())


def _big_endian_longs(packed: np.ndarray) -> LongArray:
    """The packed little-endian words as a big-endian NBT long array, converted in place (packed is spent)."""
    packed.byteswap(inplace=True)
    return packed.view(">i8").view(_ChunkedLongArray)


def _pack_states(blocks: np.ndarray, nbits: int) -> LongArray:
    """Palette ids as Litematica's bit array: y, then z, then x, lowest bits first, values crossing long boundaries."""
    n_x, n_y, n_z = blocks.shape
    out = np.zeros(ceil(n_x * n_y * n_z * nbits / 64), dtype="<u8")
    if _compiled(blocks, "pack_states"):
        _signals.pack_states(blocks, nbits, out)
        return _big_endian_longs(out)
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
    return _big_endian_longs(out)


class _GzipSink:
    """File-like for nbtlib to write into: one gzip member, deflated in pieces on threads as the bytes arrive, each piece
    ending on a byte boundary as pigz does. Pieces are cut at fixed sizes, so the output doesn't depend on how it was written."""

    def __init__(self, path: str):
        self.out = open(path, "wb")
        self.out.write(struct.pack("<BBBBIBB", 0x1f, 0x8b, 8, 0, int(time.time()), 2, 255))
        self.pool = ThreadPoolExecutor(_GZIP_THREADS)
        self.pending = bytearray()
        self.crc = 0
        self.size = 0

    def write(self, data) -> int:
        self.pending += data
        while len(self.pending) >= (_GZIP_THREADS + 1) * _GZIP_CHUNK:  # the last piece stays back: it ends the member
            self._emit([self._take() for _ in range(_GZIP_THREADS)], last=False)
        return len(data)

    def _take(self) -> bytes:
        piece = bytes(self.pending[:_GZIP_CHUNK])
        del self.pending[:_GZIP_CHUNK]
        return piece

    def _emit(self, pieces: list[bytes], last: bool) -> None:
        def deflate(index: int) -> bytes:
            c = zlib.compressobj(_GZIP_LEVEL, zlib.DEFLATED, -zlib.MAX_WBITS)
            return c.compress(pieces[index]) + c.flush(zlib.Z_FINISH if last and index == len(pieces) - 1 else zlib.Z_SYNC_FLUSH)

        for piece in self.pool.map(deflate, range(len(pieces))):
            self.out.write(piece)
        for piece in pieces:
            self.crc = zlib.crc32(piece, self.crc)
            self.size += len(piece)

    def close(self) -> None:
        pieces = []
        while len(self.pending) > _GZIP_CHUNK:
            pieces.append(self._take())
        pieces.append(bytes(self.pending))  # may be empty: the member still ends
        self.pending = bytearray()
        for i in range(0, len(pieces), _GZIP_THREADS):
            batch = pieces[i:i + _GZIP_THREADS]
            self._emit(batch, last=i + _GZIP_THREADS >= len(pieces))
        self.out.write(struct.pack("<II", self.crc, self.size & 0xffffffff))
        self.pool.shutdown()
        self.out.close()


def save_schematic(schematic: Schematic, path: str) -> None:
    """Schematic.save, the same NBT, but streamed through a gzip that compresses on several threads: no whole copy of the
    NBT, nor of the packed block array."""
    schematic.update_metadata()
    sink = _GzipSink(path)
    try:
        nbtlib.File(schematic.to_nbt(save_soft=True), gzipped=True, byteorder="big").write(sink, "big")
    finally:
        sink.close()


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
        kernel = _compiled(blocks, "used_ids", "remap_ids")
        if kernel:
            _check_ids(blocks, len(palette))
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
