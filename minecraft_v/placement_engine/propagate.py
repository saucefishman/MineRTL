"""The states a built region settles to, found by propagating signal from its inputs without simulating ticks.

Follows MCHPRS's redstone code (crates/redstone), with no delays. Which blocks read which is compiled into a graph
once, then each torch, repeater, comparator, lamp and connected run of dust is re-evaluated when something it reads
changes, until nothing changes. Repeaters locked by their sides keep their saved state, which is how flip-flops hold
their values. Latches go last, once what locks them is still, so flip-flops settle at whatever level their clock rests,
and nothing they hold is overwritten on the way.

What reads what is worked out on first use, so a settled block's insides cost nothing until signal gets in.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Collection, Iterable

import numpy as np
from litemapy import BlockState, Region
from litemapy.minecraft import TileEntity
from nbtlib.tag import Compound, Int, String

from minecraft_v.progress import log

from .block_utils import Box, cells_around_boxes, palette_writer

Pos = tuple[int, int, int]

_NON_SOLID_MARKERS = ("air", "redstone_wire", "repeater", "comparator", "torch", "lever", "sign", "redstone_block", "glass")
_FACES: tuple[Pos, ...] = ((0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0))
_TOP, _BOTTOM = _FACES[:2]
_HORIZONTAL = _FACES[2:]
_FACING: dict[str, Pos] = {"north": (0, 0, -1), "south": (0, 0, 1), "west": (-1, 0, 0), "east": (1, 0, 0)}
_SIDE_PROPERTY = {(0, 0, -1): "north", (0, 0, 1): "south", (-1, 0, 0): "west", (1, 0, 0): "east"}

_MAX_STEPS_PER_ITEM = 1000
# Who can read a block: through a solid block next to both, so at most two steps away.
_REACH_TWO: tuple[Pos, ...] = tuple((x, y, z) for x in range(-2, 3) for y in range(-2, 3) for z in range(-2, 3)
                                    if abs(x) + abs(y) + abs(z) <= 2)

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
_STEPPERS = (_REPEATER, _COMPARATOR, _TORCH, _WALL_TORCH, _LAMP)


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


class _Propagation:
    def __init__(self, region: Region, opaque: Collection[Box] = ()):
        self.region = region
        self.opaque = list(opaque)
        palette = region._Region__palette
        self.palette = palette
        self.blocks = region._Region__blocks
        self.view = memoryview(self.blocks)  # single cells; numpy's scalars are slow
        self.size = self.blocks.shape
        self.solid_id = [_is_solid(b) for b in palette]
        self.transparent_id = [_is_transparent(b) for b in palette]
        self.kind_id = [_KINDS.get(b.id, -1) for b in palette]
        self.index: dict[Pos, int] = {}
        self.cells: list[Pos] = []
        self.kind: list[int] = []
        self.block: list[BlockState] = []
        self.facing: list[Pos] = []
        self.power: list[int] = []  # dust level, or signal emitted
        self.locked: list[bool] = []
        self.saved: list[tuple[int, bool]] = []
        self.inside: list[bool] = []
        self.sides: dict[int, tuple[Pos, ...]] = {}
        self.link_cache: dict[int, list[int]] = {}
        self.input_cache: dict[int, list[int]] = {}
        self.side_cache: dict[int, list[int]] = {}
        self.source_cache: dict[int, list[int]] = {}
        self.reader_cache: dict[int, list[int]] = {}
        self.run_of: dict[int, int] = {}
        self.dust_runs: list[list[int]] = []
        self.feeds: dict[int, list[int]] = {}
        self.subtracts: set[int] = set()
        self.comparator_signal = {e.position: int(e.to_nbt()["OutputSignal"]) for e in region.tile_entities
                                  if "OutputSignal" in e.to_nbt()}

    def _id(self, p: Pos) -> int:
        x, y, z = p
        w, h, l = self.size
        return self.view[x, y, z] if 0 <= x < w and 0 <= y < h and 0 <= z < l else 0

    def _is_in_box(self, p: Pos) -> bool:
        x, y, z = p
        for (lx, ly, lz), (hx, hy, hz) in self.opaque:
            if lx <= x < hx and ly <= y < hy and lz <= z < hz:
                return True
        return False

    def _at(self, p: Pos) -> int | None:
        """p's number if it's a redstone part."""
        i = self.index.get(p)
        if i is None and self.kind_id[self._id(p)] >= 0:
            i = self._load(p)
        return i

    def _load(self, p: Pos) -> int:
        block = self.palette[self._id(p)]
        k = self.kind_id[self._id(p)]
        i = len(self.cells)
        self.index[p] = i
        self.cells.append(p)
        self.kind.append(k)
        self.block.append(block)
        self.facing.append(_FACING[_property(block, "facing", "north")] if k in (_REPEATER, _COMPARATOR, _WALL_TORCH) else _TOP)
        level = self.comparator_signal.get(p, 0) if k == _COMPARATOR else self._saved_power(block, k)
        locked = _property(block, "locked", "false") == "true"
        self.power.append(level)
        self.locked.append(locked)
        self.saved.append((level, locked))
        inside = self._is_in_box(p)
        self.inside.append(inside)
        if k == _COMPARATOR:
            if _property(block, "mode", "compare") == "subtract":
                self.subtracts.add(i)
        return i

    @staticmethod
    def _saved_power(block: BlockState, kind: int) -> int:
        if kind == _DUST:
            return int(_property(block, "power", "0"))
        if kind in (_REPEATER, _LEVER):
            return 15 if _property(block, "powered", "false") == "true" else 0
        if kind in (_TORCH, _WALL_TORCH):
            return 15 if _property(block, "lit", "true") == "true" else 0
        if kind == _LAMP:
            return 15 if _property(block, "lit", "false") == "true" else 0
        return 15 if kind == _REDSTONE_BLOCK else 0

    def _is_solid_at(self, p: Pos) -> bool:
        return self.solid_id[self._id(p)]

    def _kind_at(self, p: Pos) -> int | None:
        i = self._at(p)
        return None if i is None else self.kind[i]

    # ---- wire shape and connections

    def _connects(self, p: Pos, d: Pos) -> bool:
        block = self._id(p)
        k = self.kind_id[block]
        if k < 0 or k == _LAMP:
            return False
        return k != _REPEATER or _FACING[self.palette[block]["facing"]] in (d, _negate(d))

    def _dust_sides(self, i: int) -> tuple[Pos, ...]:
        """Directions dust points, which it powers blocks in."""
        found = self.sides.get(i)
        if found is None:
            found = self.sides[i] = self._find_dust_sides(i)
        return found

    def _find_dust_sides(self, i: int) -> tuple[Pos, ...]:
        p = self.cells[i]
        connected = []
        above = _add(p, _TOP)
        for d in _HORIZONTAL:
            side = _add(p, d)
            if (self._connects(side, d)
                    or not self._is_solid_at(above) and self._kind_id_at(_add(side, _TOP)) == _DUST
                    or not self._is_solid_at(side) and self._kind_id_at(_add(side, _BOTTOM)) == _DUST):
                connected.append(d)
        if len(connected) == 1:
            return connected[0], _negate(connected[0])
        if connected:
            return tuple(connected)
        dot = all(_property(self.block[i], _SIDE_PROPERTY[d], "none") == "none" for d in _HORIZONTAL)
        return () if dot else _HORIZONTAL

    def _kind_id_at(self, p: Pos) -> int:
        return self.kind_id[self._id(p)]

    def _dust_links(self, i: int) -> list[int]:
        """Dust i takes level from."""
        links = self.link_cache.get(i)
        if links is not None:
            return links
        links = self.link_cache[i] = []
        p = self.cells[i]
        above_solid = self._is_solid_at(_add(p, _TOP))
        for d in _FACES:
            q = _add(p, d)
            if self._kind_id_at(q) == _DUST:
                links.append(self._at(q))
            if d in _HORIZONTAL:
                up = _add(q, _TOP)
                q_id = self._id(q)
                if not above_solid and not self.transparent_id[q_id] and self._kind_id_at(up) == _DUST:
                    links.append(self._at(up))
                down = _add(q, _BOTTOM)
                if not self.solid_id[q_id] and self._kind_id_at(down) == _DUST:
                    links.append(self._at(down))
        return links

    def _dust_takers(self, i: int) -> list[int]:
        """Dust that takes level from i: one way, over glass."""
        takers = self.feeds.get(i)
        if takers is None:
            p = self.cells[i]
            takers = self.feeds[i] = []
            near = [_add(p, d) for d in _FACES]
            near += [_add(_add(p, d), e) for d in _HORIZONTAL for e in (_TOP, _BOTTOM)]
            for q in near:
                if self._kind_id_at(q) == _DUST:
                    j = self._at(q)
                    if i in self._dust_links(j):
                        takers.append(j)
        return takers

    def _run(self, start: int) -> int:
        """The number of the connected run of dust that start is in."""
        number = self.run_of.get(start)
        if number is not None:
            return number
        number = len(self.dust_runs)
        self.run_of[start] = number
        run = [start]
        for i in run:
            for q in (*self._dust_links(i), *self._dust_takers(i)):
                if q not in self.run_of:
                    self.run_of[q] = number
                    run.append(q)
        self.dust_runs.append(run)
        return number

    # ---- what emits into a cell, as get_weak_power and get_strong_power: `side` is the direction from the reader to p

    def _weak(self, p: Pos, side: Pos, dust_power: bool) -> list[int]:
        i = self._at(p)
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
            emits = dust_power and side != _BOTTOM and (side == _TOP or _negate(side) in self._dust_sides(i))
        else:
            emits = False
        return [i] if emits else []

    def _strong(self, p: Pos, side: Pos, dust_power: bool) -> list[int]:
        i = self._at(p)
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
        if self._is_solid_at(p):
            return [i for d in _FACES for i in self._strong(_add(p, d), d, dust_power)]
        return self._weak(p, side, dust_power)

    def _diode_input(self, i: int) -> list[int]:
        d = self.facing[i]
        q = _add(self.cells[i], d)
        found = self._signal(q, d)
        return found or ([self._at(q)] if self._kind_at(q) == _DUST else [])

    def _diode_sides(self, i: int, wire_counts: bool) -> list[int]:
        dx, _, dz = self.facing[i]
        found = []
        for d in ((dz, 0, -dx), (-dz, 0, dx)):
            q = _add(self.cells[i], d)
            k = self._kind_at(q)
            if k in (_REPEATER, _COMPARATOR):
                found += self._weak(q, d, False)
            elif wire_counts and k in (_DUST, _REDSTONE_BLOCK):
                found.append(self._at(q))
        return found

    def _inputs(self, i: int) -> list[int]:
        found = self.input_cache.get(i)
        if found is not None:
            return found
        k = self.kind[i]
        p = self.cells[i]
        if k == _TORCH:
            found = self._signal(_add(p, _BOTTOM), _TOP)
        elif k == _WALL_TORCH:
            d = _negate(self.facing[i])
            found = self._signal(_add(p, d), d)
        elif k == _LAMP:
            found = [e for d in _FACES for e in self._signal(_add(p, d), d)]
        else:
            found = self._diode_input(i)
        self.input_cache[i] = found
        return found

    def _side_inputs(self, i: int) -> list[int]:
        k = self.kind[i]
        if k not in (_REPEATER, _COMPARATOR):
            return []
        found = self.side_cache.get(i)
        if found is None:
            found = self.side_cache[i] = self._diode_sides(i, k == _COMPARATOR)
        return found

    def _dust_sources(self, i: int) -> list[int]:
        found = self.source_cache.get(i)
        if found is None:
            p = self.cells[i]
            found = self.source_cache[i] = [e for d in _FACES for e in self._signal(_add(p, d), d, False)]
        return found

    def _readers(self, e: int) -> list[int]:
        """What to re-evaluate when e changes: a block's number, or ~ a dust run's number."""
        found = self.reader_cache.get(e)
        if found is not None:
            return found
        readers: set[int] = set()
        p = self.cells[e]
        for d in _REACH_TWO:
            q = _add(p, d)
            k = self._kind_id_at(q)
            if k == _DUST:
                c = self._at(q)
                if e in self._dust_sources(c):
                    readers.add(~self._run(c))
            elif k in _STEPPERS:
                c = self._at(q)
                if e in self._inputs(c) or e in self._side_inputs(c):
                    readers.add(c)
        found = self.reader_cache[e] = list(readers)
        return found

    # ---- evaluation

    def _output(self, i: int) -> int:
        k = self.kind[i]
        power = self.power
        inputs = self._inputs(i)
        if k == _REPEATER:
            sides = self._side_inputs(i)
            self.locked[i] = any(power[e] for e in sides)
            if self.locked[i]:
                return power[i]
            return 15 if any(power[e] for e in inputs) else 0
        if k == _COMPARATOR:
            level = max((power[e] for e in inputs), default=0)
            sides = max((power[e] for e in self._side_inputs(i)), default=0)
            if i in self.subtracts:
                return max(level - sides, 0)
            return level if level >= sides else 0
        if k == _LAMP:
            return 15 if any(power[e] for e in inputs) else 0
        return 0 if any(power[e] for e in inputs) else 15

    def _dust_levels(self, run: list[int]) -> dict[int, int]:
        """Each cell's level: what blocks feed it, less a step per dust from the strongest."""
        power = self.power
        levels = dict.fromkeys(run, 0)
        buckets: list[list[int]] = [[] for _ in range(16)]
        for i in run:
            level = max((power[e] for e in self._dust_sources(i)), default=0)
            if level:
                levels[i] = level
                buckets[level].append(i)
        for level in range(15, 1, -1):
            for i in buckets[level]:
                if levels[i] != level:
                    continue
                for q in self._dust_takers(i):
                    if levels[q] < level - 1:
                        levels[q] = level - 1
                        buckets[level - 1].append(q)
        return levels

    def _starting_cells(self) -> list[Pos]:
        """Redstone outside the boxes, and redstone in them that something outside could feed."""
        table = np.array([k >= 0 for k in self.kind_id])
        mask = table[self.blocks]
        for lo, hi in self.opaque:
            mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = False
        cells = [(x, y, z) for x, y, z in np.argwhere(mask).tolist()]
        del mask
        near = {_add(p, d) for p in cells_around_boxes(self.region, self.opaque, 2) for d in _REACH_TWO}
        cells += sorted(p for p in near if self._is_in_box(p) and self.kind_id[self._id(p)] >= 0)
        return cells

    def run(self) -> None:
        power = self.power
        queue: deque[int] = deque()
        steppers: list[int] = []
        for p in self._starting_cells():
            i = self._at(p)
            if self.kind[i] == _DUST:
                number = self._run(i)
                if self.dust_runs[number][0] == i:
                    queue.append(~number)
            elif self.kind[i] in _STEPPERS:
                steppers.append(i)
        # Latches wait until everything else is still, so one can't unlock on a lock signal that's only passing through.
        # A box waits for the wire outside it: what's passing through there would only stir up what's settled in it.
        latches = {i for i in steppers if self._side_inputs(i)}
        queue.extend(i for i in steppers if i not in latches)
        in_box: deque[int] = deque()
        waiting = deque(i for i in steppers if i in latches)
        queued = {*queue, *waiting}
        # A loop of torches never goes still.
        steps = 0
        limit = _MAX_STEPS_PER_ITEM * max(len(queued), 1)
        while queue or in_box or waiting:
            steps += 1
            if steps > limit:
                limit = _MAX_STEPS_PER_ITEM * max(len(self.cells), 1)  # more may have come in since
                if steps > limit:
                    log("[settle] signal never went still, is there an oscillator? Saving the states as they are.")
                    return
            item = queue.popleft() if queue else in_box.popleft() if in_box else waiting.popleft()
            queued.discard(item)
            if item < 0:
                changed = [(i, level) for i, level in self._dust_levels(self.dust_runs[~item]).items() if power[i] != level]
            else:
                level = self._output(item)
                changed = [(item, level)] if level != power[item] else []
            for i, level in changed:
                power[i] = level
            for i, _ in changed:
                for reader in self._readers(i):
                    if reader not in queued:
                        queued.add(reader)
                        if reader >= 0 and self._side_inputs(reader):
                            waiting.append(reader)
                        elif self.inside[reader if reader >= 0 else self.dust_runs[~reader][0]]:
                            in_box.append(reader)
                        else:
                            queue.append(reader)

    def states(self) -> list[str]:
        """One line per block, as `mchprs_headless settle` writes them. Inside a box, only the ones that moved."""
        lines = []
        for i in sorted(range(len(self.cells)), key=self.cells.__getitem__):
            if self.inside[i] and (self.power[i], self.locked[i]) == self.saved[i]:
                continue
            x, y, z = self.cells[i]
            k = self.kind[i]
            if k == _REPEATER:
                lines.append(f"{x} {y} {z} powered {int(self.power[i] > 0)} {int(self.locked[i])}")
            elif k == _DUST:
                lines.append(f"{x} {y} {z} power {self.power[i]}")
            elif k in (_TORCH, _WALL_TORCH, _LAMP):
                lines.append(f"{x} {y} {z} lit {int(self.power[i] > 0)}")
            elif k == _COMPARATOR:
                lines.append(f"{x} {y} {z} output {self.power[i]}")
        return lines


def propagate_states(region: Region, opaque: Collection[Box] = ()) -> list[str]:
    """Settled states of region's blocks."""
    propagation = _Propagation(region, opaque)
    propagation.run()
    return propagation.states()


def comparator_entity(pos: Pos, level: int) -> TileEntity:
    """What a comparator's signal is saved in. Without it one pasted on reads 0: its output doesn't look changed when what
    it reads goes low, so wire it was holding up stays up."""
    entity = TileEntity(Compound({"id": String("minecraft:comparator"), "OutputSignal": Int(level)}))
    entity.position = pos
    return entity


def apply_states(region: Region, lines: Iterable[str]) -> None:
    """Writes states in the form propagate_states and `mchprs_headless settle` give them."""
    flag = lambda v: "true" if v == "1" else "false"
    write = palette_writer(region)
    palette, blocks = region._Region__palette, region._Region__blocks
    view = memoryview(blocks)
    results: dict[tuple, BlockState] = {}  # few distinct results, however many blocks
    entities = {entity.position: entity for entity in region.tile_entities}
    for line in lines:
        x, y, z, kind, *values = line.split()
        pos = (int(x), int(y), int(z))
        if kind == "output":
            old = entities.pop(pos, None)
            if old is not None:
                region.tile_entities.remove(old)
            if values[0] != "0":
                entities[pos] = comparator_entity(pos, int(values[0]))
                region.tile_entities.append(entities[pos])
            values = ["1" if values[0] != "0" else "0"]
        key = (view[pos], kind, *values)
        block = results.get(key)
        if block is None:
            if kind == "powered":
                props = {"powered": flag(values[0]), "locked": flag(values[1])}
            elif kind == "output":
                props = {"powered": flag(values[0])}
            elif kind == "power":
                props = {"power": values[0]}
            else:
                props = {"lit": flag(values[0])}
            block = results[key] = palette[key[0]].with_properties(**props)
        write(pos, block)


def settle(region: Region, opaque: Collection[Box] = ()) -> None:
    """Puts region's blocks in the states they settle to, so nothing transitions when it's pasted. Otherwise the
    pulses while blocks settle can clock flip-flops.

    opaque: boxes already settled on their own with nothing driving them, whose insides are only visited if signal
    reaches them."""
    apply_states(region, propagate_states(region, opaque))
