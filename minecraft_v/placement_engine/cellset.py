"""Sets of world cells as sorted arrays of packed integer keys: 8 bytes a cell, where a frozenset of position tuples
costs about 150 (the tuple, its ints, a table slot)."""
from __future__ import annotations

from array import array
from bisect import bisect_left
from collections.abc import Iterable, Iterator

import numpy as np

POS_BITS = 21                    # coordinates up to a million either side of 0
POS_BIAS = 1 << (POS_BITS - 1)
POS_MASK = (1 << POS_BITS) - 1


def pack(x: int, y: int, z: int) -> int:
    return (x + POS_BIAS) | (y + POS_BIAS) << POS_BITS | (z + POS_BIAS) << 2 * POS_BITS


def unpack(key: int) -> tuple[int, int, int]:
    return (key & POS_MASK) - POS_BIAS, (key >> POS_BITS & POS_MASK) - POS_BIAS, (key >> 2 * POS_BITS) - POS_BIAS


class CellSet:
    """An immutable set of (x, y, z) cells. Membership is a binary search: for sets read rarely, and big."""
    __slots__ = ("keys",)

    def __init__(self, keys: np.ndarray | array):
        self.keys = keys if isinstance(keys, array) else array("q", np.ascontiguousarray(keys, dtype=np.int64).tobytes())

    @classmethod
    def from_cells(cls, cells: Iterable[tuple[int, int, int]]) -> CellSet:
        return cls(np.unique(np.fromiter((pack(*c) for c in cells), dtype=np.int64)))

    @classmethod
    def from_boxes(cls, boxes: Iterable[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> CellSet:
        """The cells of boxes given as (origin, size); a box with a size of 0 on any axis has none."""
        by_size: dict[tuple[int, int, int], list[int]] = {}
        for (ox, oy, oz), size in boxes:
            if min(size) > 0:
                by_size.setdefault(tuple(size), []).append(pack(ox, oy, oz))
        parts = []
        for (w, h, d), origins in by_size.items():
            # Keys add like coordinates while no axis carries into the next: they are all far from 2**21.
            dx, dy, dz = np.meshgrid(np.arange(w), np.arange(h), np.arange(d), indexing="ij")
            offsets = (dx | dy << POS_BITS | dz << 2 * POS_BITS).ravel().astype(np.int64)
            parts.append((np.asarray(origins, dtype=np.int64)[:, None] + offsets[None, :]).ravel())
        return cls(np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64))

    def __contains__(self, cell) -> bool:
        x, y, z = cell
        key = (x + POS_BIAS) | (y + POS_BIAS) << POS_BITS | (z + POS_BIAS) << 2 * POS_BITS
        keys = self.keys
        i = bisect_left(keys, key)
        return i < len(keys) and keys[i] == key

    def __len__(self) -> int:
        return len(self.keys)

    def __bool__(self) -> bool:
        return len(self.keys) > 0

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        keys = np.frombuffer(self.keys, dtype=np.int64)
        for start in range(0, len(keys), 1 << 16):
            chunk = keys[start:start + (1 << 16)]
            xs = (chunk & POS_MASK) - POS_BIAS
            ys = (chunk >> POS_BITS & POS_MASK) - POS_BIAS
            zs = (chunk >> 2 * POS_BITS) - POS_BIAS
            yield from zip(xs.tolist(), ys.tolist(), zs.tolist())
