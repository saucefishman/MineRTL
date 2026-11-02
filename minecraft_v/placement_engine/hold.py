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
from collections import defaultdict
from collections.abc import Callable, Collection

from litemapy import BlockState, Region

from minecraft_v.cell_library import SCHEMATIC_MAP
from .ir import ComponentList, ComponentType
from .verify import _DELTA_TO_FACING, _DUST, _HORIZ_DIRS, _OPAQUE, _REPEATER, _STRONG, _TORCH, _WALL_TORCH, _World

Pos = tuple[int, int, int]
_SEQUENTIAL = (ComponentType.DFF, ComponentType.DFFE, ComponentType.DLATCH)
_MAX_DELAY = 4          # a repeater's longest setting
_HOLD_MARGIN = 6        # ticks data must arrive after the clock pin: a flip-flop locks its master a few ticks after C
_FASTEST_GATE = 1       # ticks through any cell's fastest path; the library's delay is its slowest, too late for hold
_ROUNDS = 40            # passes over the failing pins; each one pads every failing path
_UNREACHED = 1 << 30


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
        self.roots = {n.source.component_id for n in clock_nets
                      if self.by_id[n.source.component_id].type == ComponentType.INPUT_PIN}

    def gate(self, cid: str) -> int:
        c = self.by_id[cid]
        return SCHEMATIC_MAP[c.type].propagation_delay_ticks if c.type in SCHEMATIC_MAP else 0

    def net_search(self, net_id: str) -> tuple:
        if net_id not in self.cache:
            n = self.nets[net_id]
            src = self.terminals[n.source.component_id, n.source.pin_name]
            sinks = {self.terminals[s.component_id, s.pin_name] for s in n.sinks}
            self.cache[net_id] = wire_arrivals(self.world, src, self.all_terminals - sinks - {src})
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
        pins, _ = self.propagate({cid: 1 for cid in self.roots}, self.gate)
        clock = {cid: t for (cid, pin), t in pins.items() if pin == "C"}
        launches = {cid: t + _FASTEST_GATE for cid, t in clock.items()}
        data, via = self.propagate(launches, lambda _: _FASTEST_GATE)
        margins = {(c, p): t - clock[c] for (c, p), t in data.items() if c in clock and p != "C"}
        return clock, margins, via

    def min_period(self, clock: dict[str, int], probed: Collection[str] = ()) -> int:
        """Longest flip-flop to flip-flop time, with each one's own clock arrival: the slowest safe clock period.
        A print in `probed` reads its cell as the edge enters the clock pin, so that cell has to settle by then."""
        inputs: dict[str, list] = defaultdict(list)
        for n in self.nets.values():
            for s in n.sinks:
                inputs[s.component_id].append((n, s))

        def wire(n, s) -> int | None:
            return self.net_search(n.net_id)[0].get(self.terminals[s.component_id, s.pin_name])

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


def _pad_net(timing: _HoldTiming, net_id: str, sink: Pos, need: int, terminals: frozenset[Pos], counts: dict) -> int:
    """Add up to need ticks on a net, nearest the sink first; returns what was added."""
    world = timing.world
    region = world.region
    cells = [p for p in timing.cells_back(net_id, sink) if p not in terminals]
    added = 0
    for p in cells:  # no new blocks until these run out
        if added >= need:
            break
        if world.kind(p) == _REPEATER and repeater_delay(world.block(p)) < _MAX_DELAY:
            ticks = min(_MAX_DELAY - repeater_delay(world.block(p)), need - added)
            region[p] = world.block(p).with_properties(delay=str(repeater_delay(world.block(p)) + ticks))
            world._kind.pop(p, None)
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
        region[p] = BlockState("minecraft:repeater", facing=_DELTA_TO_FACING[delta], delay=str(ticks))
        world._kind.pop(p, None)
        added += ticks
        counts["new"] += 1
    if added:
        timing.cache.pop(net_id, None)
    return added


def fix_hold(workspace: Region, comp: ComponentList, terminals: dict[tuple[str, str], Pos], failed: set[str],
             probed: Collection[str] = ()) -> int:
    """Pad every data path that beats its flip-flop's clock; returns the slowest safe clock period. comp is the flat
    netlist, terminals its gate pins' cells, probed the cells prints read."""
    if not any(c.type in _SEQUENTIAL for c in comp.components):
        return 0
    world = _World(workspace, frozenset(), frozenset(terminals.values()))
    timing = _HoldTiming(world, comp, terminals, failed)
    protected = frozenset(terminals.values())
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
                got = _pad_net(timing, net_id, sink, need, protected, counts)
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
    for pin, m in sorted(left.items(), key=lambda kv: kv[1])[:10]:
        print(f"[hold] still short: {pin} margin {m}")
    period = timing.min_period(clock, probed)
    print(f"[hold] slowest flip-flop to flip-flop or print path {period} ticks")
    return period


def flat_terminals(comp: ComponentList, top_terminal: Callable[[str, str], Pos],
                   macro_terminals: dict[tuple[str, str], Pos]) -> dict[tuple[str, str], Pos]:
    """Every flat net end's wire cell, whether it stayed at the top level or went into a macro."""
    return {(e.component_id, e.pin_name): macro_terminals.get((e.component_id, e.pin_name))
            or top_terminal(e.component_id, e.pin_name)
            for n in comp.nets for e in [n.source, *n.sinks]}
