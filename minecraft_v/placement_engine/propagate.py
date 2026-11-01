"""The states a built region settles to, found by propagating signal from its inputs without simulating ticks.

Follows MCHPRS's redstone code (crates/redstone), with no delays. Which blocks read which is compiled into a graph
once, then each torch, repeater, comparator, lamp and connected run of dust is re-evaluated when something it reads
changes, until nothing changes. Repeaters locked by their sides keep their saved state, which is how flip-flops hold
their values.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np
from litemapy import BlockState, Region

Pos = tuple[int, int, int]

_NON_SOLID_MARKERS = ("air", "redstone_wire", "repeater", "comparator", "torch", "lever", "sign", "redstone_block", "glass")
_FACES: tuple[Pos, ...] = ((0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0))
_TOP, _BOTTOM = _FACES[:2]
_HORIZONTAL = _FACES[2:]
_FACING: dict[str, Pos] = {"north": (0, 0, -1), "south": (0, 0, 1), "west": (-1, 0, 0), "east": (1, 0, 0)}
_SIDE_PROPERTY = {(0, 0, -1): "north", (0, 0, 1): "south", (-1, 0, 0): "west", (1, 0, 0): "east"}

_DUST, _REPEATER, _COMPARATOR, _TORCH, _WALL_TORCH, _LAMP, _LEVER, _REDSTONE_BLOCK = range(8)
_KINDS = {
    "minecraft:redstone_wire": _DUST,
    "minecraft:repeater": _REPEATER,
    "minecraft:comparator": _COMPARATOR,
    "minecraft:redstone_torch": _TORCH,
    "minecraft:redstone_wall_torch": _WALL_TORCH,
    "minecraft:redstone_lamp": _LAMP,
    "minecraft:lever": _LEVER,
    "minecraft:redstone_block": _REDSTONE_BLOCK,
}


def _is_solid(block: BlockState) -> bool:
    if block.id.endswith("_slab"):
        return block["type"] == "double"
    return not any(marker in block.id for marker in _NON_SOLID_MARKERS)


def _is_transparent(block: BlockState) -> bool:
    if block.id.endswith("_slab"):
        return block["type"] != "double"
    return block.id in ("minecraft:glass", "minecraft:redstone_block")


def _add(p: Pos, d: Pos) -> Pos:
    return p[0] + d[0], p[1] + d[1], p[2] + d[2]


def _negate(d: Pos) -> Pos:
    return -d[0], -d[1], -d[2]


def _property(block: BlockState, name: str, default: str) -> str:
    try:
        return block[name]
    except KeyError:
        return default


def _cells(mask: np.ndarray) -> list[Pos]:
    return [(x, y, z) for x, y, z in np.argwhere(mask).tolist()]


class _Propagation:
    def __init__(self, region: Region, cuts: Iterable[Pos]):
        palette = region._Region__palette
        blocks = region._Region__blocks
        cut = set(cuts)
        solid = np.array([_is_solid(b) for b in palette])[blocks]
        transparent = np.array([_is_transparent(b) for b in palette])[blocks]
        self.solid = set(_cells(solid)) - cut
        self.transparent = set(_cells(transparent)) - cut
        kinds = np.array([_KINDS.get(b.id, -1) for b in palette])[blocks]
        self.cells = [p for p in _cells(kinds >= 0) if p not in cut]
        self.index = {p: i for i, p in enumerate(self.cells)}
        self.kind = [int(kinds[p]) for p in self.cells]
        self.block = [palette[blocks[p]] for p in self.cells]
        self.facing = [_FACING[_property(b, "facing", "north")] if k in (_REPEATER, _COMPARATOR, _WALL_TORCH) else _TOP
                       for b, k in zip(self.block, self.kind)]
        self.power = [self._saved_power(b, k) for b, k in zip(self.block, self.kind)]  # dust level, or signal emitted
        self.locked = [_property(b, "locked", "false") == "true" for b in self.block]
        self.sides = {i: self._dust_sides(i) for i, k in enumerate(self.kind) if k == _DUST}
        self.dust_runs, self.feeds = self._find_dust_runs()

    @staticmethod
    def _saved_power(block: BlockState, kind: int) -> int:
        if kind in (_REPEATER, _LEVER):
            return 15 if _property(block, "powered", "false") == "true" else 0
        if kind in (_TORCH, _WALL_TORCH):
            return 15 if _property(block, "lit", "true") == "true" else 0
        if kind == _LAMP:
            return 15 if _property(block, "lit", "false") == "true" else 0
        return 15 if kind == _REDSTONE_BLOCK else 0

    def _kind_at(self, p: Pos) -> int | None:
        i = self.index.get(p)
        return None if i is None else self.kind[i]

    # ---- wire shape and connections

    def _connects(self, p: Pos, d: Pos) -> bool:
        i = self.index.get(p)
        if i is None or self.kind[i] == _LAMP:
            return False
        return self.kind[i] != _REPEATER or self.facing[i] in (d, _negate(d))

    def _dust_sides(self, i: int) -> tuple[Pos, ...]:
        """Directions dust points, which it powers blocks in."""
        p = self.cells[i]
        connected = []
        above = _add(p, _TOP)
        for d in _HORIZONTAL:
            side = _add(p, d)
            if (self._connects(side, d)
                    or above not in self.solid and self._kind_at(_add(side, _TOP)) == _DUST
                    or side not in self.solid and self._kind_at(_add(side, _BOTTOM)) == _DUST):
                connected.append(d)
        if len(connected) == 1:
            return connected[0], _negate(connected[0])
        if connected:
            return tuple(connected)
        dot = all(_property(self.block[i], _SIDE_PROPERTY[d], "none") == "none" for d in _HORIZONTAL)
        return () if dot else _HORIZONTAL

    def _dust_links(self, p: Pos) -> list[int]:
        """Dust p takes level from."""
        links = []
        above = _add(p, _TOP)
        for d in _FACES:
            q = _add(p, d)
            if self._kind_at(q) == _DUST:
                links.append(self.index[q])
            if d in _HORIZONTAL:
                up = _add(q, _TOP)
                if above not in self.solid and q not in self.transparent and self._kind_at(up) == _DUST:
                    links.append(self.index[up])
                down = _add(q, _BOTTOM)
                if q not in self.solid and self._kind_at(down) == _DUST:
                    links.append(self.index[down])
        return links

    def _find_dust_runs(self) -> tuple[list[list[int]], dict[int, list[int]]]:
        feeds: dict[int, list[int]] = {i: [] for i in self.sides}  # one way, over glass
        for i in self.sides:
            for q in self._dust_links(self.cells[i]):
                feeds[q].append(i)
        joined = {i: {*feeds[i]} for i in self.sides}
        for i, fed in feeds.items():
            for q in fed:
                joined[q].add(i)
        runs: list[list[int]] = []
        seen: set[int] = set()
        for start in self.sides:
            if start in seen:
                continue
            seen.add(start)
            run = [start]
            for i in run:
                for q in joined[i]:
                    if q not in seen:
                        seen.add(q)
                        run.append(q)
            runs.append(run)
        return runs, feeds

    # ---- what emits into a cell, as get_weak_power and get_strong_power: `side` is the direction from the reader to p

    def _weak(self, p: Pos, side: Pos, dust_power: bool) -> list[int]:
        i = self.index.get(p)
        if i is None:
            return []
        k = self.kind[i]
        if k == _TORCH:
            emits = side != _TOP
        elif k == _WALL_TORCH:
            emits = self.facing[i] != side
        elif k in (_REDSTONE_BLOCK, _LEVER):
            emits = True
        elif k in (_REPEATER, _COMPARATOR):
            emits = self.facing[i] == side
        elif k == _DUST:
            emits = dust_power and side != _BOTTOM and (side == _TOP or _negate(side) in self.sides[i])
        else:
            emits = False
        return [i] if emits else []

    def _strong(self, p: Pos, side: Pos, dust_power: bool) -> list[int]:
        i = self.index.get(p)
        if i is None:
            return []
        k = self.kind[i]
        if k in (_TORCH, _WALL_TORCH):
            return [i] if side == _BOTTOM else []
        if k == _LEVER:
            block = self.block[i]
            face, facing = _property(block, "face", "wall"), _FACING[_property(block, "facing", "north")]
            attached = side == _TOP if face == "floor" else side == _BOTTOM if face == "ceiling" else facing == side
            return [i] if attached else []
        return self._weak(p, side, dust_power) if k in (_DUST, _REPEATER, _COMPARATOR) else []

    def _signal(self, p: Pos, side: Pos, dust_power: bool = True) -> list[int]:
        if p in self.solid:
            return [i for d in _FACES for i in self._strong(_add(p, d), d, dust_power)]
        return self._weak(p, side, dust_power)

    def _diode_input(self, i: int) -> list[int]:
        d = self.facing[i]
        q = _add(self.cells[i], d)
        found = self._signal(q, d)
        return found or ([self.index[q]] if self._kind_at(q) == _DUST else [])

    def _diode_sides(self, i: int, wire_counts: bool) -> list[int]:
        dx, _, dz = self.facing[i]
        found = []
        for d in ((dz, 0, -dx), (-dz, 0, dx)):
            q = _add(self.cells[i], d)
            k = self._kind_at(q)
            if k in (_REPEATER, _COMPARATOR):
                found += self._weak(q, d, False)
            elif wire_counts and k in (_DUST, _REDSTONE_BLOCK):
                found.append(self.index[q])
        return found

    def _inputs(self, i: int) -> list[int]:
        k = self.kind[i]
        p = self.cells[i]
        if k == _TORCH:
            return self._signal(_add(p, _BOTTOM), _TOP)
        if k == _WALL_TORCH:
            d = _negate(self.facing[i])
            return self._signal(_add(p, d), d)
        if k == _LAMP:
            return [e for d in _FACES for e in self._signal(_add(p, d), d)]
        return self._diode_input(i)

    # ---- evaluation

    def run(self) -> None:
        power = self.power
        kind = self.kind
        locked = self.locked
        steppers = [i for i, k in enumerate(kind) if k in (_REPEATER, _COMPARATOR, _TORCH, _WALL_TORCH, _LAMP)]
        inputs = {i: self._inputs(i) for i in steppers}
        side_inputs = {i: self._diode_sides(i, kind[i] == _COMPARATOR) for i in steppers if kind[i] in (_REPEATER, _COMPARATOR)}
        subtracts = {i for i in steppers if kind[i] == _COMPARATOR and _property(self.block[i], "mode", "compare") == "subtract"}
        dust_sources = {i: [e for d in _FACES for e in self._signal(_add(self.cells[i], d), d, False)] for i in self.sides}

        # Who to re-evaluate when an emitter changes: a block's index, or len(cells) + a dust run's number.
        readers: dict[int, set[int]] = {}
        for i in steppers:
            for e in inputs[i]:
                readers.setdefault(e, set()).add(i)
            for e in side_inputs.get(i, ()):
                readers.setdefault(e, set()).add(i)
        offset = len(self.cells)
        for number, run in enumerate(self.dust_runs):
            for i in run:
                for e in dust_sources[i]:
                    readers.setdefault(e, set()).add(offset + number)

        def output(i: int) -> int:
            k = kind[i]
            if k == _REPEATER:
                locked[i] = any(power[e] for e in side_inputs[i])
                if locked[i]:
                    return power[i]
                return 15 if any(power[e] for e in inputs[i]) else 0
            if k == _COMPARATOR:
                level = max((power[e] for e in inputs[i]), default=0)
                sides = max((power[e] for e in side_inputs[i]), default=0)
                if i in subtracts:
                    return max(level - sides, 0)
                return level if level >= sides else 0
            if k == _LAMP:
                return 15 if any(power[e] for e in inputs[i]) else 0
            return 0 if any(power[e] for e in inputs[i]) else 15

        def dust_levels(run: list[int]) -> dict[int, int]:
            """Each cell's level: what blocks feed it, less a step per dust from the strongest."""
            levels = dict.fromkeys(run, 0)
            buckets: list[list[int]] = [[] for _ in range(16)]
            for i in run:
                level = max((power[e] for e in dust_sources[i]), default=0)
                if level:
                    levels[i] = level
                    buckets[level].append(i)
            for level in range(15, 1, -1):
                for i in buckets[level]:
                    if levels[i] != level:
                        continue
                    for q in self.feeds[i]:
                        if levels[q] < level - 1:
                            levels[q] = level - 1
                            buckets[level - 1].append(q)
            return levels

        # Latches wait until everything else is still, so one can't unlock on a lock signal that's only passing through.
        latches = {i for i, found in side_inputs.items() if found}
        queue = deque(offset + number for number in range(len(self.dust_runs)))
        queue.extend(i for i in steppers if i not in latches)
        waiting = deque(latches)
        queued = {*queue, *waiting}
        while queue or waiting:
            item = queue.popleft() if queue else waiting.popleft()
            queued.discard(item)
            if item >= offset:
                changed = [(i, level) for i, level in dust_levels(self.dust_runs[item - offset]).items() if power[i] != level]
            else:
                level = output(item)
                changed = [(item, level)] if level != power[item] else []
            for i, level in changed:
                power[i] = level
            for i, _ in changed:
                for reader in readers.get(i, ()):
                    if reader not in queued:
                        queued.add(reader)
                        (waiting if reader in latches else queue).append(reader)

    def states(self) -> list[str]:
        """One line per block, as `mchprs_headless settle` writes them."""
        lines = []
        for i, (x, y, z) in enumerate(self.cells):
            k = self.kind[i]
            if k == _REPEATER:
                lines.append(f"{x} {y} {z} powered {int(self.power[i] > 0)} {int(self.locked[i])}")
            elif k == _DUST:
                lines.append(f"{x} {y} {z} power {self.power[i]}")
            elif k in (_TORCH, _WALL_TORCH, _LAMP):
                lines.append(f"{x} {y} {z} lit {int(self.power[i] > 0)}")
        return lines


def propagate_states(region: Region, cuts: Iterable[Pos]) -> list[str]:
    """Settled states of region's blocks, with `cuts` cells taken out."""
    propagation = _Propagation(region, cuts)
    propagation.run()
    return propagation.states()
