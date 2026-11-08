"""Find repeated subcircuits: colour-seeded lockstep growth, then greedy packing of disjoint instances.

Not a complete search (frequent subgraph mining is NP-hard). Growth follows SUBDUE's shape (grow a pattern with its
embedding list, keep the best, take it out, go again), with these deviations: pin roles make most extensions
unambiguous, so one greedy path replaces the beam; instances stay disjoint while growing; and the score is cells saved
less a pin penalty, not MDL. Growth builds the bijection between instances, so no isomorphism test is needed.

- Nets above max_fanout are never followed. Cells on them still record which net it is, so they become ports.
- CUSTOM cells (RAM and ROM blocks, other macros) never join an instance: their contents live in the block, which the
  label can't see, so instances holding different blocks would pass as equal and be stamped with the first one's. They
  stay in the top netlist and the nets to them become ports.
- Seeds are groups of cells with equal colour-refinement colours.
- Every instance adds its matching neighbour together, taking the extension the most instances support first. A step
  can lose instances before later steps pay it back (a mux tree halves per level), so growth runs until under two are
  left and the best prefix is kept.
"""
from __future__ import annotations

import heapq
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from minecraft_v.placement_engine.ir import ComponentList, ComponentType, Direction

_IO_TYPES = (ComponentType.INPUT_PIN, ComponentType.OUTPUT_PIN)
_COLOR_ROUNDS = 2       # colour radius of a seed group
_PIN_PENALTY = 0.5      # score lost per port of the pattern
_PREFIXES_SCORED = 8    # prefixes of a growth to score exactly, by cells saved
_NO_CELL = -1           # an IO pin at the end of a net


@dataclass
class RepeatedClass:
    cells: set[str]                  # the first instance's cells, which name the template
    nets: dict[str, str]             # template net -> "internal", "input" or "output"
    instances: list[dict[str, str]]  # template cell or net -> instance's; the first maps to itself


class _Circuit:
    """The netlist as integers, cells in component order, so that results don't depend on string hashing."""

    def __init__(self, comp: ComponentList, max_fanout: int):
        cells = [c for c in comp.components if c.type not in _IO_TYPES]
        self.ids = [c.id for c in cells]
        self.is_custom = [c.type == ComponentType.CUSTOM for c in cells]
        index = {cid: i for i, cid in enumerate(self.ids)}
        self.net_ids = [n.net_id for n in comp.nets]
        self.driver: list[tuple[int, str]] = []
        self.sinks: list[list[tuple[int, str]]] = []
        self.pins: list[dict[str, int | None]] = [{} for _ in cells]
        for k, net in enumerate(comp.nets):
            self.driver.append((index.get(net.source.component_id, _NO_CELL), net.source.pin_name))
            self.sinks.append([(index.get(s.component_id, _NO_CELL), s.pin_name) for s in net.sinks])
            for cell, pin in [self.driver[k], *self.sinks[k]]:
                if cell != _NO_CELL:
                    self.pins[cell][pin] = k
        self.followed = [len(s) <= max_fanout for s in self.sinks]
        self.is_output: list[dict[str, bool]] = []
        interned: dict[tuple, int] = {}
        self.label: list[int] = []
        for i, c in enumerate(cells):
            consts = tuple(sorted((p.name, p.const_value) for p in c.pins if p.const_value is not None))
            # Stamps share their blocks, so flip-flops starting at different values can't share a macro.
            self.label.append(interned.setdefault((c.type.value, consts, c.metadata.get("init", 0)), len(interned)))
            self.is_output.append({p.name: p.direction == Direction.OUT for p in c.pins})
            for p in c.pins:
                self.pins[i].setdefault(p.name, None)
            self.pins[i] = dict(sorted(self.pins[i].items()))

    def neighbours(self, cell: int, pin: str) -> list[tuple[int, str]]:
        """Cells across a pin's net, if it's one to follow."""
        net = self.pins[cell][pin]
        if net is None or not self.followed[net]:
            return []
        return self.sinks[net] if self.is_output[cell][pin] else [self.driver[net]]


def _colors(circuit: _Circuit, free: list[bool]) -> list[int]:
    """Colour refinement over followed nets. Nets not followed count only by the pin on them."""
    colors = list(circuit.label)
    for _ in range(_COLOR_ROUNDS):
        interned: dict[tuple, int] = {}
        next_colors = []
        for cell in range(len(colors)):
            around = []
            for pin in circuit.pins[cell]:
                near = sorted((p, colors[c]) for c, p in circuit.neighbours(cell, pin) if c != _NO_CELL and free[c])
                around.append((pin, tuple(near)))
            key = (colors[cell], tuple(around))
            next_colors.append(interned.setdefault(key, len(interned)))
        colors = next_colors
    return colors


class _Embedding:
    __slots__ = ("cells", "inside", "nets", "alive")

    def __init__(self, seed: int):
        self.cells = [seed]
        self.inside = {seed}
        self.nets: dict[int, int] = {}  # net -> template net, numbered by first appearance
        self.alive = True


def _pin_pattern(circuit: _Circuit, e: _Embedding, cell: int) -> tuple[int, ...]:
    """Which template net each of a cell's pins is on, recording new ones. Equal patterns keep pin sharing exact."""
    pattern = []
    for net in circuit.pins[cell].values():
        if net is None:
            pattern.append(-1)
        else:
            pattern.append(e.nets.setdefault(net, len(e.nets)))
    return tuple(pattern)


def _grow(circuit: _Circuit, seeds: list[int], colors: list[int], free: list[bool], deadline: float,
          ) -> tuple[list[_Embedding], list[list[_Embedding]]]:
    """Grow from a seed group. Returns every embedding, and the ones alive once the pattern has k + 1 cells, per k."""
    embeddings = [_Embedding(s) for s in seeds]
    owner = {s: e for s, e in zip(seeds, embeddings)}
    alive = list(embeddings)
    for e in alive:
        _pin_pattern(circuit, e, e.cells[0])
    steps = [list(alive)]
    heap: list[tuple[int, int, tuple]] = []
    seen = 0

    def candidates(e: _Embedding, option: tuple) -> list[int]:
        at, pin, label, far_pin = option
        return [c for c, p in circuit.neighbours(e.cells[at], pin)
                if c != _NO_CELL and p == far_pin and free[c] and circuit.label[c] == label and c not in e.inside]

    def offer(at: int) -> None:
        nonlocal seen
        options: Counter[tuple] = Counter()
        for e in alive:
            for pin in circuit.pins[e.cells[at]]:
                near = {(circuit.label[c], p) for c, p in circuit.neighbours(e.cells[at], pin)
                        if c != _NO_CELL and free[c] and c not in e.inside}
                options.update((at, pin, *n) for n in near)
        for option, support in sorted(options.items()):
            if support >= 2:
                heapq.heappush(heap, (-support, seen, option))
                seen += 1

    def around(cell: int) -> dict[str, int | None]:
        return {p: n for p, n in circuit.pins[cell].items() if n is not None and not circuit.followed[n]}

    def choose(options: list[int], model: int | None) -> int:
        """The candidate most like the first instance's: symmetric parts then agree on shared nets, and so share lanes."""
        if len(options) == 1 or model is None:
            return options[0]
        shared = around(model)
        return max(options, key=lambda c: (colors[c] == colors[model],
                                           sum(shared.get(p) == n for p, n in around(c).items()), -c))

    def survivors(option: tuple) -> int:
        """Instances that would keep an option's pin sharing, without changing anything. Only ever falls."""
        model = None
        patterns: Counter[tuple] = Counter()
        for e in alive:
            found = candidates(e, option)
            if not found:
                continue
            cell = choose(found, model)
            if model is None:
                model = cell
            fresh: dict[int, int] = {}
            patterns[tuple(-1 if n is None else e.nets.get(n, fresh.setdefault(n, len(e.nets) + len(fresh)))
                           for n in circuit.pins[cell].values())] += 1
        return max(patterns.values(), default=0)

    offer(0)
    while heap and len(alive) >= 2 and time.monotonic() < deadline:
        _, _, option = heapq.heappop(heap)
        support = survivors(option)
        if support < 2:
            continue
        if heap and support < -heap[0][0]:  # supports only fall, so this entry is known to be stale
            heapq.heappush(heap, (-support, seen, option))
            seen += 1
            continue
        # Claimed by an instance that hasn't moved yet: that one gives way, so a chain ends up with every other instance.
        model = None
        acted: set[int] = set()
        added: list[tuple[_Embedding, int]] = []
        for e in alive:
            if not e.alive:
                continue
            found = candidates(e, option)
            if not found:
                e.alive = False
                continue
            cell = choose(found, model)
            if model is None:
                model = cell
            claimant = owner.get(cell)
            if claimant is not None and claimant.alive and claimant is not e:
                if id(claimant) in acted:
                    e.alive = False
                    continue
                claimant.alive = False
            acted.add(id(e))
            owner[cell] = e
            added.append((e, cell))
        groups: dict[tuple, list[tuple[_Embedding, int]]] = defaultdict(list)
        for e, cell in added:
            if e.alive:
                groups[_pin_pattern(circuit, e, cell)].append((e, cell))
        best = max(groups.values(), key=len, default=[])
        kept = {id(e) for e, _ in best}
        for e in alive:
            if id(e) not in kept and e.alive:
                e.alive = False
        for e, cell in best:
            e.cells.append(cell)
            e.inside.add(cell)
        for e in alive:
            if not e.alive:
                for c in e.cells:
                    if owner.get(c) is e:
                        del owner[c]
        alive = [e for e, _ in best]
        if len(alive) < 2:
            break
        steps.append(alive)
        heapq.heappush(heap, (-support, seen, option))  # a net's next sink is the same option
        seen += 1
        offer(len(steps) - 1)
    return embeddings, steps


@dataclass
class _Candidate:
    cells: list[list[int]]         # per instance, in template order
    nets: list[list[int]]          # per instance, by template net
    labels: list[str]              # by template net

    def macro_pins(self) -> int:
        """Pins on the built macro: one for a port all instances share, else one per instance (macro.build_bank)."""
        n = len(self.cells)
        return sum(1 if len({nets[k] for nets in self.nets}) == 1 else n
                   for k, label in enumerate(self.labels) if label != "internal")

    def score(self) -> float:
        return (len(self.cells) - 1) * len(self.cells[0]) - _PIN_PENALTY * self.macro_pins()


def _lane_groups(group: list[tuple[list[int], list[int]]]) -> list[list[tuple[list[int], list[int]]]]:
    """Split instances so that a port shared by some is shared by all of a part; ports unique to each instance stay."""
    n_ports = len(group[0][1])
    partly = [k for k in range(n_ports) if 1 < len({nets[k] for _, nets in group}) < len(group)]
    parts: dict[tuple, list] = defaultdict(list)
    for member in group:
        parts[tuple(member[1][k] for k in partly)].append(member)
    return [p for p in parts.values() if len(p) >= 2]


def _materialize(circuit: _Circuit, members: list[list[int]]) -> list[_Candidate]:
    """Instances grouped by which nets are ports, then by which nets they share."""
    groups: dict[tuple, list[tuple[list[int], list[int]]]] = defaultdict(list)
    for cells in members:
        inside = set(cells)
        nets: dict[int, int] = {}
        for cell in cells:
            for net in circuit.pins[cell].values():
                if net is not None:
                    nets.setdefault(net, len(nets))
        labels = []
        for net in nets:
            ends = [circuit.driver[net][0], *(c for c, _ in circuit.sinks[net])]
            if all(c in inside for c in ends):
                labels.append("internal")
            else:
                labels.append("output" if circuit.driver[net][0] in inside else "input")
        groups[tuple(labels)].append((cells, list(nets)))
    return [_Candidate([m[0] for m in part], [m[1] for m in part], list(labels))
            for labels, group in groups.items() if len(group) >= 2 for part in _lane_groups(group)]


def _best_prefixes(circuit: _Circuit, steps: list[list[_Embedding]], min_cells: int) -> list[_Candidate]:
    """The parts of a growth's best prefix, one prefix per instance count: more cells at that count only help."""
    last_with: dict[int, int] = {}  # instance count -> largest pattern it reached
    for k, alive in enumerate(steps):
        if k + 1 >= min_cells:
            last_with[len(alive)] = k + 1
    best: list[_Candidate] = []
    ranked = sorted(last_with.items(), key=lambda kv: -(kv[0] - 1) * kv[1])[:_PREFIXES_SCORED]
    for _, size in ranked:
        parts = _materialize(circuit, [e.cells[:size] for e in steps[size - 1]])
        if sum(max(c.score(), 0) for c in parts) > sum(max(c.score(), 0) for c in best):
            best = parts
    return best


def _pack(found: list[_Candidate], taken: list[bool]) -> list[_Candidate]:
    """Best candidates first, without instances that overlap ones already kept; lazy, as scores only fall."""
    kept = []
    heap = [(-c.score(), k) for k, c in enumerate(found)]
    heapq.heapify(heap)
    while heap:
        _, k = heapq.heappop(heap)
        c = found[k]
        keep = [j for j, cells in enumerate(c.cells) if not any(taken[x] for x in cells)]
        if len(keep) < 2:
            continue
        disjoint = _Candidate([c.cells[j] for j in keep], [c.nets[j] for j in keep], c.labels)
        if disjoint.score() <= 0:
            continue
        found[k] = disjoint
        if heap and disjoint.score() < -heap[0][0]:
            heapq.heappush(heap, (-disjoint.score(), k))
            continue
        for cells in disjoint.cells:
            for x in cells:
                taken[x] = True
        kept.append(disjoint)
    return kept


def find_repeated(comp: ComponentList, max_fanout: int = 2, min_cells: int = 8, time_limit: float = 30.0,
                  ) -> list[RepeatedClass]:
    """Classes of disjoint, exactly isomorphic subcircuits, largest coverage first. Stops early at time_limit."""
    deadline = time.monotonic() + time_limit
    circuit = _Circuit(comp, max_fanout)
    taken = list(circuit.is_custom)  # blocks are never part of an instance
    kept: list[_Candidate] = []
    while time.monotonic() < deadline:  # each round mines what the last one left
        free = [not t for t in taken]
        colors = _colors(circuit, free)
        groups: dict[int, list[int]] = defaultdict(list)
        for cell, color in enumerate(colors):
            if free[cell]:
                groups[color].append(cell)
        found: list[_Candidate] = []
        covered = [False] * len(colors)
        for seeds in sorted((g for g in groups.values() if len(g) >= 2), key=lambda g: (-len(g), g[0])):
            if time.monotonic() >= deadline:
                break
            if all(covered[s] for s in seeds):  # a pattern through these cells is already in
                continue
            _, steps = _grow(circuit, seeds, colors, free, deadline)
            for best in _best_prefixes(circuit, steps, min_cells):
                found.append(best)
                for cells in best.cells:
                    for x in cells:
                        covered[x] = True
        new = _pack(found, taken)
        if not new:
            break
        kept.extend(new)
    kept.sort(key=lambda c: -len(c.cells) * len(c.cells[0]))
    classes = []
    for c in kept:
        names = [circuit.net_ids[n] for n in c.nets[0]]
        first = [circuit.ids[x] for x in c.cells[0]]
        instances = [{**{t: circuit.ids[x] for t, x in zip(first, cells)},
                      **{t: circuit.net_ids[n] for t, n in zip(names, nets)}}
                     for cells, nets in zip(c.cells, c.nets)]
        classes.append(RepeatedClass(set(first), dict(zip(names, c.labels)), instances))
    return classes
