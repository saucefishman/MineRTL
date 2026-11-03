"""Hold fixing on the routed layout: delay data that reaches a flip-flop before that flip-flop's own clock.

Clock skew is large here (a wire with hundreds of sinks), and new data getting in on the same edge breaks a design
at any clock speed. Each short path gets delay nearest its flip-flop first: repeaters already on the wire go up to their
longest setting, then straight dust becomes repeaters. Delay only ever moves data later, so it can't make another
hold check worse. It does lengthen the critical path.

Everything is in the stamped world, so flip-flops inside macros count too. Wire is edited per cell, so instances of a
macro can end up with different delays.
"""
from __future__ import annotations

import heapq
from collections import defaultdict, deque
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from math import ceil

import numpy as np
from litemapy import BlockState, Region

from minecraft_v.cell_library import SCHEMATIC_MAP
from .block_utils import Box
from .ir import ComponentList, ComponentType
from .verify import (Excluding, HiddenBoxes, Violation, _DELTA_TO_FACING, _DUST, _HORIZ_DIRS, _OPAQUE, _REPEATER, _STRONG, _TORCH,
                     _WALL_TORCH, _World, _signals, cell_byte, signal_grid)

Pos = tuple[int, int, int]
_SEQUENTIAL = (ComponentType.DFF, ComponentType.DFFE, ComponentType.DLATCH)
_MAX_DELAY = 4          # a repeater's longest setting
_HOLD_MARGIN = 6        # what padding aims for: ticks data arrives after the clock pin, with every launch taken as _FASTEST_GATE
_FASTEST_GATE = 1       # ticks through any cell's fastest path; the library's delay is its slowest, too late for hold
# Measured from the cells (tests/test_hold.py). Ticks after C reaches its pin's cell: when D may change without getting in,
# and when Q changes. A margin is safe if data that left Q this late, then through gates at _FASTEST_GATE, still clears the
# hold time. So padding's target has slack: DFF to DFF fails below -1, DFF to DFFE below 2, DFFE to DFF below -3.
_HOLD_TICKS = {ComponentType.DFF: 2, ComponentType.DFFE: 5}
_CLOCK_TO_Q = {ComponentType.DFF: 4, ComponentType.DFFE: 6}
_ROUNDS = 40            # passes over the failing pins; each one pads every failing path
_UNREACHED = 1 << 30
_SETUP_TICKS = 1        # data settled at a flip-flop before its clock rises

# An arrival, per what launched it and which way the signal went: rising clock edge falling, rising; falling clock edge ...
_RISE_DOWN, _RISE_UP, _FALL_DOWN, _FALL_UP = range(4)
_NONE = -1
_POSITIVE = (ComponentType.AND, ComponentType.OR)
_NEGATIVE = (ComponentType.NOT,)


@dataclass
class ClockRequirement:
    """The least half period, from static timing. Times are ticks after the launching edge enters the clock pin.

    full: paths launched and captured by the same edge, which get a whole period.
    half_cycle: paths between opposite edges, which get half of one: the clock itself used as data falling, as RAM write
    windows are, or a flip-flop on an inverted clock.
    write_floor: a RAM write window launched by the falling edge has to have closed by the next rising edge.
    unclocked: flip-flops no input pin clocks, which this doesn't cover."""
    full: int = 0
    half_cycle: int = 0
    write_floor: int = 0
    unclocked: int = 0
    worst: dict[str, str] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)  # per kind: the worst path, cell by cell, with arrival ticks

    @property
    def half_period(self) -> int:
        return max(ceil(self.full / 2), self.half_cycle, self.write_floor)


@dataclass
class HoldShort:
    """A pin padding left under _HOLD_MARGIN. slack is against what the flip-flop really needs: negative can latch wrong data."""
    pin: tuple[str, str]
    pos: Pos
    margin: int
    slack: int
    path: list[tuple[str, Pos]]  # (net, sink cell) from the pin back to the launching flip-flop, clock nets left out


def repeater_delay(block: BlockState) -> int:
    return int(block["delay"]) if "delay" in block else 1


def _step(world: _World, p: Pos) -> int:
    if world.kind(p) == _REPEATER:
        return repeater_delay(world.block(p))
    return 1 if world.is_torch(p) else 0


def wire_arrivals(world: _World, source: Pos, stop: frozenset[Pos]) -> tuple[dict[Pos, int], dict[tuple, tuple]]:
    """Least ticks from source to each cell its signal reaches, and the search's parent of each node."""
    start = (source, _STRONG if world.kind(source) == _OPAQUE else "")
    best = {start: 0}
    parent: dict[tuple, tuple] = {}
    heap = [(0, start)]
    arrival: dict[Pos, int] = {}
    while heap:
        t, node = heapq.heappop(heap)
        if t > best[node]:
            continue
        arrival.setdefault(node[0], t)
        if node[0] in stop and node[0] != source:
            continue
        for nxt in world.successors(node):
            at = t + _step(world, nxt[0])
            if at < best.get(nxt, _UNREACHED):
                best[nxt] = at
                parent[nxt] = node
                heapq.heappush(heap, (at, nxt))
    return arrival, parent


class _HoldTiming:
    def __init__(self, world: _World, comp: ComponentList, terminals: dict[tuple[str, str], Pos], skip: set[str]):
        self.world = world
        self.by_id = {c.id: c for c in comp.components}
        self.terminals = terminals
        self.nets = {n.net_id: n for n in comp.nets if n.net_id not in skip}
        self.nets_from: dict[str, list] = defaultdict(list)
        for n in self.nets.values():
            self.nets_from[n.source.component_id].append(n)
        self.all_terminals = frozenset(terminals.values())
        self.cache: dict[str, tuple] = {}
        clock_nets = [n for n in self.nets.values() if any(s.pin_name == "C" for s in n.sinks)]
        self.clock_net_ids = {n.net_id for n in clock_nets}
        self.roots = self._clock_inputs(clock_nets)

    def _clock_inputs(self, clock_nets: list) -> set[str]:
        """Input pins whose edges can reach a clock pin: directly, or through gates such as clk & enable."""
        drivers: dict[str, list[str]] = defaultdict(list)
        for n in self.nets.values():
            for s in n.sinks:
                drivers[s.component_id].append(n.source.component_id)
        roots: set[str] = set()
        seen: set[str] = set()
        todo = [n.source.component_id for n in clock_nets]
        while todo:
            cid = todo.pop()
            if cid in seen:
                continue
            seen.add(cid)
            kind = self.by_id[cid].type
            if kind == ComponentType.INPUT_PIN:
                roots.add(cid)
            elif kind not in _SEQUENTIAL:
                todo.extend(drivers[cid])
        return roots

    def _clock_pins(self, root: str) -> dict[tuple[str, str], int]:
        return self.propagate({root: 1}, self.gate)[0]

    def clock_arrivals(self) -> dict[str, int]:
        """Each flip-flop's clock arrival. Through a gate, any of several inputs may be the clock, so the latest counts:
        an arrival taken from a quick enable path would understate how long data has to hold."""
        clock: dict[str, int] = {}
        for root in self.roots:
            for (cid, pin), t in self._clock_pins(root).items():
                if pin == "C":
                    clock[cid] = max(clock.get(cid, 0), t)
        return clock

    def forget(self, net_id: str) -> None:
        """A net's wire changed. Other nets keep what they found until they change too."""
        self.cache.pop(net_id, None)

    def wire_ticks(self, n, s) -> int | None:
        return self.net_search(n.net_id)[0].get(self.terminals[s.component_id, s.pin_name])

    def gate(self, cid: str) -> int:
        c = self.by_id[cid]
        return SCHEMATIC_MAP[c.type].propagation_delay_ticks if c.type in SCHEMATIC_MAP else 0

    def net_search(self, net_id: str) -> tuple:
        if net_id not in self.cache:
            n = self.nets[net_id]
            src = self.terminals[n.source.component_id, n.source.pin_name]
            sinks = {self.terminals[s.component_id, s.pin_name] for s in n.sinks}
            self.cache[net_id] = wire_arrivals(self.world, src, Excluding(self.all_terminals, sinks | {src}))
        return self.cache[net_id]

    def propagate(self, starts: dict[str, int], gate: Callable[[str], int]) -> tuple[dict, dict]:
        """Earliest time at each flip-flop pin, and (net, driver) of each flip-flop pin or gate on the way there."""
        into: dict[tuple[str, str], int] = {}
        via: dict = {}
        best = dict(starts)
        heap = sorted((t, cid) for cid, t in starts.items())
        heapq.heapify(heap)
        while heap:
            t, cid = heapq.heappop(heap)
            if t > best.get(cid, _UNREACHED):
                continue
            for n in self.nets_from.get(cid, ()):
                arrival = self.net_search(n.net_id)[0]
                for s in n.sinks:
                    pos = self.terminals[s.component_id, s.pin_name]
                    if pos not in arrival:
                        continue
                    at = t + arrival[pos]
                    kind = self.by_id[s.component_id].type
                    if kind in _SEQUENTIAL:
                        key = (s.component_id, s.pin_name)
                        if at < into.get(key, _UNREACHED):
                            into[key] = at
                            via[key] = (n.net_id, cid, pos)
                    elif kind != ComponentType.OUTPUT_PIN:
                        out = at + gate(s.component_id)
                        if out < best.get(s.component_id, _UNREACHED):
                            best[s.component_id] = out
                            via[s.component_id] = (n.net_id, cid, pos)
                            heapq.heappush(heap, (out, s.component_id))
        return into, via

    def margins(self) -> tuple[dict[str, int], dict[tuple[str, str], int], dict]:
        clock = self.clock_arrivals()
        launches = {cid: t + _FASTEST_GATE for cid, t in clock.items()}
        data, via = self.propagate(launches, lambda _: _FASTEST_GATE)
        margins = {(c, p): t - clock[c] for (c, p), t in data.items() if c in clock and p != "C"}
        return clock, margins, via

    def _data_arrivals(self, launches: dict[str, int]) -> dict[tuple[str, str], int]:
        return self.propagate(launches, lambda _: _FASTEST_GATE)[0]

    def slacks(self, clock: dict[str, int]) -> dict[tuple[str, str], int]:
        """Per pin, ticks past what its flip-flop really needs, launches at their measured clock to Q. Margins take every
        launch as the fastest gate, so this is what can truly fail. Other kinds of flip-flop keep the padding target."""
        launches = {cid: t + _CLOCK_TO_Q.get(self.by_id[cid].type, _FASTEST_GATE) for cid, t in clock.items()}
        return {(c, p): t - clock[c] - _HOLD_TICKS.get(self.by_id[c].type, _HOLD_MARGIN)
                for (c, p), t in self._data_arrivals(launches).items() if c in clock and p != "C"}

    def min_period(self, clock: dict[str, int], probed: Collection[str] = ()) -> int:
        """Longest flip-flop to flip-flop time, with each one's own clock arrival: the slowest safe clock period.
        A print in `probed` reads its cell as the edge enters the clock pin, so that cell has to settle by then."""
        inputs: dict[str, list] = defaultdict(list)
        for n in self.nets.values():
            for s in n.sinks:
                inputs[s.component_id].append((n, s))

        wire = self.wire_ticks

        memo: dict[str, int] = {}

        def out_time(cid: str) -> int:
            """Latest a cell's output changes after the clock edge reaches the first flip-flop. -1: not from a flip-flop."""
            if cid in clock:
                return clock[cid] + self.gate(cid)
            if cid in memo:
                return memo[cid]
            memo[cid] = -1  # stands for no path, should combinational logic ever loop
            late = max((t + d for n, s in inputs[cid]
                        for t, d in [(out_time(n.source.component_id), wire(n, s))] if t >= 0 and d is not None), default=-1)
            memo[cid] = late + self.gate(cid) if late >= 0 else -1
            return memo[cid]

        worst = 0
        for cid, capture in clock.items():
            for n, s in inputs[cid]:
                t, d = out_time(n.source.component_id), wire(n, s)
                if s.pin_name != "C" and t >= 0 and d is not None:
                    worst = max(worst, t + d - capture)
        return max([worst, *(out_time(cid) for cid in probed)])

    def clock_requirement(self, macros: dict | None = None, probed: Collection[str] = ()) -> ClockRequirement:
        """Static timing of the flat netlist: every path from a launching edge to a capturing one, per clock edge and per
        direction, so that a path a cell can't pass isn't counted, like a write window opening from the rising edge.
        Each flip-flop has its own clock arrival, so skew counts path by path rather than the worst everywhere.
        Inputs change as the clock rises, and outputs and prints are read as it rises next."""
        macros = macros or {}
        by_id = self.by_id
        macro_of = {cid: macros[c.metadata["macro"]] for cid, c in by_id.items()
                    if c.type == ComponentType.CUSTOM and c.metadata.get("macro") in macros}
        seq = {cid for cid, c in by_id.items() if c.type in _SEQUENTIAL}
        out_pin = {cid for cid, c in by_id.items() if c.type == ComponentType.OUTPUT_PIN}

        def source_node(n) -> object:
            cid = n.source.component_id
            return (cid, n.source.pin_name) if cid in macro_of else cid

        def sink_node(s) -> object:
            cid = s.component_id
            return (cid, s.pin_name) if cid in macro_of or cid in seq else cid

        # Wire edges, and a macro's arcs: (ticks, sink, input up or None for any, output up).
        wires: dict[object, list[tuple[int, object]]] = defaultdict(list)
        fed: dict[object, int] = defaultdict(int)
        for n in self.nets.values():
            src = source_node(n)
            fed[src] += 0
            for s in n.sinks:
                ticks = self.wire_ticks(n, s)
                if ticks is None:
                    continue
                sink = sink_node(s)
                wires[src].append((ticks, sink))
                fed[sink] += 1
        arcs: dict[object, list[tuple[object, dict]]] = defaultdict(list)
        for cid, macro in macro_of.items():
            for pin, outs in macro.edges.items():
                for out, edges in outs.items():
                    if (cid, out) in fed or (cid, out) in wires:
                        arcs[cid, pin].append(((cid, out), edges))
                        fed[cid, out] += 1
                        fed[cid, pin] += 0
        order: list[object] = []
        left = dict(fed)
        queue = deque(node for node, count in left.items() if count == 0)
        while queue:
            node = queue.popleft()
            order.append(node)
            for _, sink in wires.get(node, ()):
                left[sink] -= 1
                if left[sink] == 0:
                    queue.append(sink)
            for sink, _ in arcs.get(node, ()):
                left[sink] -= 1
                if left[sink] == 0:
                    queue.append(sink)
        looped = len(left) - len(order)
        if looped:
            print(f"[timing] WARNING: {looped} cells on combinational loops; paths through them aren't counted")

        came: dict[tuple[object, int], object] = {}

        def trace(node: object, launch: int) -> str:
            """Back from a node along the latest arrivals of an edge's launch."""
            steps = []
            for _ in range(10000):
                times = data.get(node)
                if times is None:
                    break
                k = launch + (times[launch + 1] >= times[launch])
                steps.append(f"{node[0]}.{node[1]}@{times[k]}" if isinstance(node, tuple) else f"{node}@{times[k]}")
                node = came.get((node, k))
                if node is None:
                    break
            return " <- ".join(steps)

        def gate_out(cid: object, at: list[int]) -> list[int]:
            c = by_id.get(cid)
            if c is None or c.type in (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN):
                return at
            d = self.gate(cid)
            out = [_NONE] * 4
            for launch in (0, 2):
                down, up = at[launch], at[launch + 1]
                if c.type in _POSITIVE:
                    pair = down, up
                elif c.type in _NEGATIVE:
                    pair = up, down
                else:
                    pair = (max(down, up),) * 2
                out[launch] = pair[0] + d if pair[0] >= 0 else _NONE
                out[launch + 1] = pair[1] + d if pair[1] >= 0 else _NONE
            return out

        def arrivals(starts: dict[object, list[int]]) -> dict[object, list[int]]:
            at: dict[object, list[int]] = {}

            def reach(node: object, times: list[int], source: object = None) -> None:
                have = at.setdefault(node, [_NONE] * 4)
                for k in range(4):
                    if times[k] > have[k]:
                        have[k] = times[k]
                        came[node, k] = source

            for node, times in starts.items():
                reach(node, times)
            for node in order:
                here = at.get(node)
                if here is None:
                    continue
                if node in seq:
                    out = here
                else:
                    out = gate_out(node, here) if not isinstance(node, tuple) else here
                for ticks, sink in wires.get(node, ()):
                    reach(sink, [t + ticks if t >= 0 else _NONE for t in out], node)
                for sink, edges in arcs.get(node, ()):
                    times = [_NONE] * 4
                    for launch in (0, 2):
                        for (in_up, out_up), arc in edges.items():
                            t = here[launch + in_up]
                            if t >= 0:
                                k = launch + out_up
                                times[k] = max(times[k], t + arc)
                    reach(sink, times, node)
            return at

        # Which input pins are clocks: those a flip-flop's clock pin leads back to.
        back: dict[object, list[object]] = defaultdict(list)
        for src, sinks in wires.items():
            for _, sink in sinks:
                back[sink].append(src)
        roots: set[str] = set()
        stack = [(cid, "C") for cid in seq]
        seen: set = set(stack)
        while stack:
            for src in back.get(stack.pop(), ()):
                if src in seen or src in seq:
                    continue
                seen.add(src)
                if src in by_id and by_id[src].type == ComponentType.INPUT_PIN:
                    roots.add(src)
                else:
                    stack.append(src)
        result = ClockRequirement()
        edge = {cid: [_NONE, 1, 1, _NONE] for cid in roots}  # 1: through the input's own repeater
        clock_at = arrivals(edge)
        came.clear()
        capture: dict[str, list[tuple[int, int]]] = {}
        starts: dict[object, list[int]] = dict(edge)
        for cid, c in by_id.items():
            if c.type == ComponentType.INPUT_PIN and cid not in roots:
                starts[cid] = [1, 1, _NONE, _NONE]
        for cid in seq:
            at = clock_at.get((cid, "C"))
            rising = [(launch, at[launch + 1]) for launch in (0, 2) if at is not None and at[launch + 1] >= 0]
            if not rising:
                result.unclocked += 1
                continue
            capture[cid] = rising
            q = [_NONE] * 4
            for launch, t in rising:
                q[launch] = q[launch + 1] = t + self.gate(cid)
            starts[cid] = q
        data = arrivals(starts)

        def check(name: str, times: list[int], captures: list[tuple[int, int]], setup: int, node: object = None) -> None:
            for launch in (0, 2):
                latest = max(times[launch], times[launch + 1])
                if latest < 0:
                    continue
                for edge_launch, at in captures:
                    need = latest - at + setup
                    kind = "full" if edge_launch == launch else "half_cycle"
                    if need > getattr(result, kind):
                        setattr(result, kind, need)
                        result.worst[kind] = name
                        if node is not None:
                            result.paths[kind] = trace(node, launch)

        for cid, rising in capture.items():
            for pin in ("D", "E"):
                if (cid, pin) in data:
                    check(f"{cid}.{pin}", data[cid, pin], rising, _SETUP_TICKS, (cid, pin))
        at_rise = [(0, 0)]
        for cid in out_pin:
            if cid in data:
                # The pin is read past its exit repeater, which the wire stops short of.
                check(cid, [t + 1 if t >= 0 else _NONE for t in data[cid]], at_rise, 0, cid)
        for cid in probed:
            if cid in seq:
                times = starts.get(cid)
            else:
                times = gate_out(cid, data[cid]) if cid in data and cid in by_id else None
            if times is not None:
                check(f"print {cid}", times, at_rise, 0)
        # A write the falling edge opened has to be over before what it writes changes, after the next rising edge.
        soonest: dict[object, int] = {cid: 1 for cid in starts if cid in by_id and by_id[cid].type == ComponentType.INPUT_PIN}
        for cid, rising in capture.items():
            soonest[cid] = min(t for _, t in rising) + _FASTEST_GATE
        for node in order:
            t = soonest.get(node)
            if t is None:
                continue
            out = t + _FASTEST_GATE if node in by_id and node not in seq and by_id[node].type != ComponentType.INPUT_PIN else t
            for ticks, sink in wires.get(node, ()):
                soonest[sink] = min(soonest.get(sink, _UNREACHED), out + ticks)
            for sink, edges in arcs.get(node, ()):
                soonest[sink] = min(soonest.get(sink, _UNREACHED), t + min(edges.values()))
        for cid, macro in macro_of.items():
            for pin, still in macro.closing.items():
                times = data.get((cid, pin))
                if times is None or times[_FALL_DOWN] < 0:
                    continue
                changes = min((soonest[cid, p] for p in macro.guarded.get(pin, ()) if (cid, p) in soonest), default=0)
                need = times[_FALL_DOWN] + still - changes
                if need > result.write_floor:
                    result.write_floor = need
                    result.worst["write_floor"] = f"{cid}.{pin}"
                    result.paths["write_floor"] = trace((cid, pin), 2)
        return result

    def path_nets(self, pin: tuple[str, str], via: dict) -> list[tuple[str, Pos]]:
        """The earliest data path into a pin, as (net, sink cell), from the pin back to the launching flip-flop."""
        out = []
        key: object = pin
        while key in via:
            net, driver, pos = via[key]
            out.append((net, pos))
            if self.by_id[driver].type in _SEQUENTIAL:
                break
            key = driver
        return out

    def cells_back(self, net_id: str, sink: Pos) -> list[Pos]:
        """A net's wire from its sink back to its source, ends left out."""
        parent = self.net_search(net_id)[1]
        node = next(((sink, mode) for mode in ("", _STRONG, "W") if (sink, mode) in parent), None)
        cells = []
        while node in parent:
            node = parent[node]
            cells.append(node[0])
        return list(dict.fromkeys(cells[:-1]))


class _GridWorld(_World):
    """A _World whose writes reach the kernel's grid too."""
    grid = None

    def changed(self, p: Pos) -> None:
        super().changed(p)
        self.grid.set_byte(p, cell_byte(self.block(p)))


class _KernelTiming(_HoldTiming):
    """_HoldTiming on the kernel, which does what its Python does: margins are worked out after every padded pin."""

    def __init__(self, world: _GridWorld, comp: ComponentList, terminals: dict[tuple[str, str], Pos], skip: set[str]):
        super().__init__(world, comp, terminals, skip)
        self.grid = world.grid
        ids = {c for c in self.by_id} | {e.component_id for n in self.nets.values() for e in (n.source, *n.sinks)}
        self.gate_ids = sorted(ids)
        self.gate_of = {cid: i for i, cid in enumerate(self.gate_ids)}
        self.net_list = list(self.nets.values())
        self.net_of = {n.net_id: i for i, n in enumerate(self.net_list)}
        self.pin_keys: list[tuple[str, str]] = []
        self.pin_of: dict[tuple[str, str], int] = {}
        self.sink_of: dict[tuple[str, str, str], int] = {}
        net_sinks = []
        for n in self.net_list:
            sinks = []
            for index, s in enumerate(n.sinks):
                self.sink_of.setdefault((n.net_id, s.component_id, s.pin_name), index)
                kind = self.by_id[s.component_id].type
                klass = 0 if kind in _SEQUENTIAL else 1 if kind == ComponentType.OUTPUT_PIN else 2
                pin = -1
                if klass == 0:
                    pin = self.pin_of.setdefault((s.component_id, s.pin_name), len(self.pin_keys))
                    if pin == len(self.pin_keys):
                        self.pin_keys.append((s.component_id, s.pin_name))
                sinks.append((self.gate_of[s.component_id], pin, klass))
            net_sinks.append(sinks)
        gates_from = [[self.net_of[n.net_id] for n in self.nets_from.get(cid, ())] for cid in self.gate_ids]
        self.kernel = _signals.HoldKernel(net_sinks, gates_from, len(self.gate_ids), len(self.pin_keys), self._search)
        self.real_delay = np.array([self.gate(cid) if cid in self.by_id else 0 for cid in self.gate_ids], dtype=np.intc)
        self.fast_delay = np.full(len(self.gate_ids), _FASTEST_GATE, dtype=np.intc)

    def _search(self, index: int) -> list[int]:
        n = self.net_list[index]
        src = self.terminals[n.source.component_id, n.source.pin_name]
        ends = [self.terminals[s.component_id, s.pin_name] for s in n.sinks]
        keys, parents, ticks = self.grid.search(src, {*ends, src}, ends)
        self.cache[n.net_id] = (keys, parents)
        return ticks

    def forget(self, net_id: str) -> None:
        super().forget(net_id)
        self.kernel.forget(self.net_of[net_id])

    def wire_ticks(self, n, s) -> int | None:
        t = self.kernel.arrival(self.net_of[n.net_id], self.sink_of[n.net_id, s.component_id, s.pin_name])
        return None if t < 0 else t

    def _pins(self, starts: dict[str, int], delays: np.ndarray) -> dict[tuple[str, str], int]:
        found = self.kernel.propagate([(self.gate_of[cid], t) for cid, t in starts.items()], delays)
        return {self.pin_keys[pin]: t for pin, t in found}

    def _clock_pins(self, root: str) -> dict[tuple[str, str], int]:
        return self._pins({root: 1}, self.real_delay)

    def margins(self) -> tuple[dict[str, int], dict[tuple[str, str], int], dict]:
        clock = self.clock_arrivals()
        launches = {cid: t + _FASTEST_GATE for cid, t in clock.items()}
        data = self._pins(launches, self.fast_delay)
        margins = {(c, p): t - clock[c] for (c, p), t in data.items() if c in clock and p != "C"}
        return clock, margins, {}  # path_nets reads the kernel's last run

    def _data_arrivals(self, launches: dict[str, int]) -> dict[tuple[str, str], int]:
        return self._pins(launches, self.fast_delay)

    def path_nets(self, pin: tuple[str, str], via: dict) -> list[tuple[str, Pos]]:
        out = []
        step = self.kernel.via_of_pin(self.pin_of[pin])
        while step is not None:
            net, driver, sink = step
            n = self.net_list[net]
            s = n.sinks[sink]
            out.append((n.net_id, self.terminals[s.component_id, s.pin_name]))
            cid = self.gate_ids[driver]
            if self.by_id[cid].type in _SEQUENTIAL:
                break
            step = self.kernel.via_of_gate(driver)
        return out

    def cells_back(self, net_id: str, sink: Pos) -> list[Pos]:
        self.kernel.ensure(self.net_of[net_id])
        keys, parents = self.cache[net_id]
        at = None
        for mode in range(3):  # "", strong, weak, as _HoldTiming tries them
            found = np.flatnonzero(keys == self.grid.node_key(sink, mode))
            if len(found) and parents[found[0]] >= 0:
                at = int(found[0])
                break
        cells = []
        while at is not None and parents[at] >= 0:
            at = int(parents[at])
            cells.append(self.grid.pos_of_key(int(keys[at])))
        return list(dict.fromkeys(cells[:-1]))


def _powers_torch_below(world: _World, p: Pos) -> bool:
    """Whether dust at p drives a torch on the block under it, as a powered -4 drop does. A repeater wouldn't power that block."""
    x, y, z = p
    if world.kind((x, y - 2, z)) == _TORCH:
        return True
    for (dx, dz), facing in _DELTA_TO_FACING.items():
        q = (x + dx, y - 1, z + dz)
        if world.kind(q) == _WALL_TORCH and world.block(q)["facing"] == facing:
            return True
    return False


def _straight_run(world: _World, p: Pos) -> tuple[Pos, Pos] | None:
    """A level dust cell with wire on exactly two opposite sides and nothing else touching: (one side, the other)."""
    x, y, z = p
    links = world.dust_links(p)
    if len(links) > 2 or _powers_torch_below(world, p):
        return None
    ends = []
    for dx, dz in _HORIZ_DIRS:
        q = (x + dx, y, z + dz)
        kind = world.kind(q)
        if kind == _REPEATER:
            if p not in (world.repeater_input(q), world.repeater_output(q)):
                return None  # it would lock the new repeater
            ends.append(q)
        elif kind == _DUST:
            ends.append(q)
        elif world.is_torch(q) or kind == _OPAQUE:
            return None
    if len(ends) != 2 or any(link not in ends for link in links):
        return None
    a, b = ends
    return (a, b) if a[0] + b[0] == 2 * x and a[2] + b[2] == 2 * z else None


def _pad_net(timing: _HoldTiming, net_id: str, sink: Pos, need: int, terminals: frozenset[Pos], counts: dict,
             edits: list[tuple[Pos, BlockState]] | None = None) -> int:
    """Add up to need ticks on a net, nearest the sink first; returns what was added. edits gets each cell's old block."""
    world = timing.world
    region = world.region
    cells = [p for p in timing.cells_back(net_id, sink) if p not in terminals]
    added = 0
    for p in cells:  # no new blocks until these run out
        if added >= need:
            break
        if world.kind(p) == _REPEATER and repeater_delay(world.block(p)) < _MAX_DELAY:
            ticks = min(_MAX_DELAY - repeater_delay(world.block(p)), need - added)
            if edits is not None:
                edits.append((p, world.block(p)))
            region[p] = world.block(p).with_properties(delay=str(repeater_delay(world.block(p)) + ticks))
            world.changed(p)
            added += ticks
            counts["raised"] += 1
    for i, p in enumerate(cells):
        if added >= need:
            break
        if world.kind(p) != _DUST or (i + 1 >= len(cells)):
            continue
        upstream = cells[i + 1]  # cells run back toward the source
        run = _straight_run(world, p)
        if run is None or upstream not in run or any(q in terminals for q in run):
            continue
        delta = (upstream[0] - p[0], upstream[2] - p[2])
        ticks = min(_MAX_DELAY, need - added)
        if edits is not None:
            edits.append((p, world.block(p)))
        region[p] = BlockState("minecraft:repeater", facing=_DELTA_TO_FACING[delta], delay=str(ticks))
        world.changed(p)
        added += ticks
        counts["new"] += 1
    if added:
        timing.forget(net_id)
    return added


def fix_hold(workspace: Region, comp: ComponentList, terminals: dict[tuple[str, str], Pos], failed: set[str],
             probed: Collection[str] = (), opaque: Collection[Box] = (), macros: dict | None = None,
             requirement: list[ClockRequirement] | None = None, shorts: list[HoldShort] | None = None,
             edits: list[tuple[Pos, BlockState]] | None = None) -> int:
    """Pad every data path that beats its flip-flop's clock; returns the slowest flip-flop to flip-flop time. comp is the
    flat netlist, terminals its gate pins' cells, probed the cells prints read, opaque boxes that net searches stop at.
    `requirement` gets what the clock has to be, from the wire as padded: macros time the blocks in the netlist.
    `shorts` gets the pins padding couldn't bring to target, `edits` each changed cell's old block, to undo this."""
    if not any(c.type in _SEQUENTIAL for c in comp.components):
        return 0
    protected = frozenset(terminals.values())
    blocked = HiddenBoxes(frozenset(), opaque)
    grid = signal_grid(workspace, blocked, protected)
    if grid is None:
        world = _World(workspace, blocked, protected)
        timing = _HoldTiming(world, comp, terminals, failed)
    else:
        world = _GridWorld(workspace, blocked, protected)
        world.grid = grid
        timing = _KernelTiming(world, comp, terminals, failed)
    counts = {"raised": 0, "new": 0}
    clock, margins, via = timing.margins()
    if not clock:
        return 0
    first_bad = sum(m < _HOLD_MARGIN for m in margins.values())
    first_worst = min(margins.values(), default=0)
    total = 0
    for _ in range(_ROUNDS):
        bad = sorted((m, pin) for pin, m in margins.items() if m < _HOLD_MARGIN)
        if not bad:
            break
        progress = 0
        for margin, pin in bad:
            need = _HOLD_MARGIN - margin
            for net_id, sink in timing.path_nets(pin, via):
                if net_id in timing.clock_net_ids:
                    continue
                got = _pad_net(timing, net_id, sink, need, protected, counts, edits)
                need -= got
                progress += got
                if need <= 0:
                    break
            # Another pin's path may share these nets, and its numbers are now stale.
            clock, margins, via = timing.margins()
        total += progress
        if not progress:
            break
    left = {pin: m for pin, m in margins.items() if m < _HOLD_MARGIN}
    print(f"[hold] clock arrives {min(clock.values())}..{max(clock.values())} ticks at {len(clock)} flip-flops; "
          f"{first_bad} of {len(margins)} pins short (worst {first_worst}), added {total} ticks "
          f"({counts['raised']} repeaters slowed, {counts['new']} new); {len(left)} left")
    found = [HoldShort(pin, terminals[pin], m, 0, [e for e in timing.path_nets(pin, via) if e[0] not in timing.clock_net_ids])
             for pin, m in sorted(left.items(), key=lambda kv: (kv[1], kv[0]))]
    if found:
        slack = timing.slacks(clock)
        for short in found:
            short.slack = slack[short.pin]
    for short in found[:10]:
        print(f"[hold] still short: {short.pin} margin {short.margin}, slack {short.slack}")
    if shorts is not None:
        shorts.extend(found)
    period = timing.min_period(clock, probed)
    print(f"[hold] slowest flip-flop to flip-flop or print path {period} ticks")
    if requirement is not None:
        needed = timing.clock_requirement(macros, probed)
        requirement.append(needed)
        print(f"[timing] half period {needed.half_period}: paths of a whole period {needed.full}, half a period {needed.half_cycle}, "
              f"RAM writes closing {needed.write_floor}; worst {needed.worst}"
              + (f"; {needed.unclocked} flip-flops on clocks made by logic not counted" if needed.unclocked else ""))
        for kind, route in needed.paths.items():
            print(f"[timing] {kind} path: {route}")
    return period


def hold_violations(shorts: Collection[HoldShort]) -> list[Violation]:
    """Pins whose data can beat their clock even at the flip-flop's measured timing."""
    return [Violation("hold", frozenset({f"{short.pin[0]}.{short.pin[1]}"}), short.pos) for short in shorts if short.slack < 0]


def flat_terminals(comp: ComponentList, top_terminal: Callable[[str, str], Pos],
                   macro_terminals: dict[tuple[str, str], Pos]) -> dict[tuple[str, str], Pos]:
    """Every flat net end's wire cell, whether it stayed at the top level or went into a macro."""
    return {(e.component_id, e.pin_name): macro_terminals.get((e.component_id, e.pin_name))
            or top_terminal(e.component_id, e.pin_name)
            for n in comp.nets for e in [n.source, *n.sinks]}
