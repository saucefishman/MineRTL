"""Blocks' timing, measured: their redstone simulated tick by tick, as MCHPRS's redstone code does it (crates/redstone).

What reads what comes from _Propagation, so only cells that signal reaches cost anything. Dust changes within a tick;
repeaters, comparators and torches take their delay, and keep MCHPRS's rules: a pending tick isn't scheduled twice, and a
pulse shorter than a repeater's delay comes out as long as the delay. tests/test_tick_sim.py compares it with mchprs_headless.

An Edges maps (input went up, output went up) to the longest ticks from one to the other. Ticks start at the input
repeater's output changing, and end at the output repeater's output changing.
"""
from __future__ import annotations

import heapq

from .propagate import _COMPARATOR, _LAMP, _REPEATER, _Propagation, Pos

Edges = dict[tuple[bool, bool], int]

_MAX_EVENTS = 20_000_000
_OFF, _ON, _OTHER = 1, 2, 3  # MCHPRS's tick priorities, roughly: only order within a tick depends on them


class TickSim(_Propagation):
    def __init__(self, region):
        super().__init__(region)
        self.now = 0
        self.heap: list[tuple[int, int, int, int]] = []
        self.pending: set[int] = set()
        self.forced: dict[int, int] = {}
        self.sequence = 0
        self.watching: dict[int, str] = {}
        self.seen: dict[tuple[str, bool], int] = {}
        self.last_event = 0
        self.events = 0
        self.delays: dict[int, int] = {}

    def cell(self, p: Pos) -> int:
        i = self._at(p)
        if i is None:
            raise ValueError(f"nothing to simulate at {p}")
        return i

    def _schedule(self, i: int, delay: int, priority: int) -> None:
        self.sequence += 1
        heapq.heappush(self.heap, (self.now + delay, priority, self.sequence, i))
        self.pending.add(i)

    def _delay(self, i: int) -> int:
        found = self.delays.get(i)
        if found is None:
            found = self.delays[i] = int(self.block[i]["delay"]) if "delay" in self.block[i] else 1
        return found

    def _input_on(self, i: int) -> bool:
        power = self.power
        return any(power[e] for e in self._inputs(i))

    def _comparator_output(self, i: int) -> int:
        power = self.power
        level = max((power[e] for e in self._inputs(i)), default=0)
        sides = max((power[e] for e in self._side_inputs(i)), default=0)
        if i in self.subtracts:
            return max(level - sides, 0)
        return level if level >= sides else 0

    def _neighbor_changed(self, i: int) -> None:
        if i in self.forced:
            return
        k = self.kind[i]
        if k == _REPEATER:
            power = self.power
            self.locked[i] = any(power[e] for e in self._side_inputs(i))
            if not self.locked[i] and i not in self.pending:
                on = self._input_on(i)
                if on != (power[i] > 0):
                    self._schedule(i, self._delay(i), _ON if on else _OFF)
        elif i in self.pending or k == _LAMP:
            return
        elif k == _COMPARATOR:
            if self._comparator_output(i) != self.power[i]:
                self._schedule(i, 1, _OTHER)
        elif self._input_on(i) == (self.power[i] > 0):  # a torch
            self._schedule(i, 1, _OTHER)

    def _set(self, i: int, level: int) -> None:
        work = [(i, level)]
        power = self.power
        while work:
            cell, value = work.pop()
            if power[cell] == value:
                continue
            name = self.watching.get(cell)
            if name is not None and (power[cell] > 0) != (value > 0):
                self.seen[name, value > 0] = self.now
            power[cell] = value
            for reader in self._readers(cell):
                if reader >= 0:
                    self._neighbor_changed(reader)
                else:
                    work.extend((c, v) for c, v in self._dust_levels(self.dust_runs[~reader]).items() if power[c] != v)

    def _tick(self, i: int) -> None:
        self.pending.discard(i)
        if i in self.forced:
            return
        k = self.kind[i]
        if k == _REPEATER:
            if self.locked[i]:
                return
            on, powered = self._input_on(i), self.power[i] > 0
            if powered and not on:
                self._set(i, 0)
            elif not powered:
                if not on:
                    self._schedule(i, self._delay(i), _OFF)
                self._set(i, 15)
        elif k == _COMPARATOR:
            self._set(i, self._comparator_output(i))
        else:
            self._set(i, 0 if self._input_on(i) else 15)

    def drive(self, i: int, on: bool) -> None:
        """A cell's output changes now, whatever it reads."""
        self.forced[i] = 15 if on else 0
        self._set(i, self.forced[i])

    def run(self, limit: int = 1 << 20) -> int:
        """Until nothing is pending; returns the tick the last thing happened."""
        start = self.now
        self.last_event = start
        heap = self.heap
        while heap:
            when, _, _, i = heapq.heappop(heap)
            if when - start > limit or self.events > _MAX_EVENTS:
                raise RuntimeError(f"signal still moving {when - start} ticks in, after {self.events} events: an oscillator?")
            self.now = when
            self.events += 1
            before = self.power[i]
            self._tick(i)
            if self.power[i] != before:
                self.last_event = when
        return self.last_event

    def toggle(self, i: int, on: bool, watch: dict[str, int] | None = None, into: dict[str, Edges] | None = None) -> int:
        """Drive a cell and run until still. Every rise and fall of a watched output, pulses too, records its ticks in
        `into`, under (the input went up, the output went up). Returns ticks until the block is still."""
        start = self.now
        self.watching = {c: name for name, c in (watch or {}).items()}
        self.seen = {}
        self.drive(i, on)
        last = self.run()
        self.watching = {}
        if into is not None:
            for (name, up), when in self.seen.items():
                edges = into.setdefault(name, {})
                edges[on, up] = max(edges.get((on, up), 0), when - start)
        return last - start


def merge(into: dict[str, Edges], more: dict[str, Edges]) -> None:
    for name, edges in more.items():
        have = into.setdefault(name, {})
        for key, ticks in edges.items():
            have[key] = max(have.get(key, 0), ticks)


def longest(edges: dict[str, Edges]) -> dict[str, int]:
    return {name: max(found.values()) for name, found in edges.items() if found}
