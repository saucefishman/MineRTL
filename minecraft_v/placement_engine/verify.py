"""Electrical verification of a routed layout: per-net opens and cross-net shorts.

Builds a directed "powers" relation over blocks (Java redstone, simplified to the
structures the router builds) and searches from each net's source:
- open:  a sink of the net that its source can't reach;
- short: a cell reached from two nets' sources, or a cell owned by another net;
- weak: a sink the source reaches, but with no signal strength left;
- latch: a repeater or torch whose own output reaches its input, so that it holds itself on;
- inverted: wire fed by both the net's signal and its opposite (a tower's middle blocks, the
  inverted stage of a powered drop), or a sink that only gets the opposite.
`verify_support` separately flags dust/repeaters/torches with nothing to stand on.

Rules (see tests/test_verify.py):
- Dust connects to horizontally adjacent dust; diagonally up/down one Y unless an opaque
  block sits directly above the lower dust. It weakly powers the block below it and the
  blocks it points into (no connections = cross, one = line both ways, 2+ = those only).
- Weakly powered blocks feed only torches/repeaters reading them; strongly powered blocks
  (repeater output, torch underneath) also feed dust on all six faces.
- Repeater (`facing` = input direction): input from behind, output in front.
- Torch: input from its attachment block; strongly powers the block above, feeds adjacent
  dust and repeaters (and dust below, unless that's its attachment).
"""
from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from litemapy import BlockState, Region
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .block_utils import cells_where, lookup
from .constants import _DELTA_TO_FACING, _HORIZ_DIRS
from .forks import fork_map

try:
    from . import _signals
except ImportError:
    _signals = None  # not built: the Python below does the same, slower

Pos = tuple[int, int, int]

_FACING_TO_DELTA = {facing: delta for delta, facing in _DELTA_TO_FACING.items()}
_NON_OPAQUE_MARKERS = ("glass", "trapdoor", "redstone_wire", "torch", "repeater", "comparator", "slab", "sign")

_AIR, _DUST, _REPEATER, _TORCH, _WALL_TORCH, _OPAQUE, _OTHER = range(7)
_STRONG, _WEAK = "S", "W"
_MAX_SIGNAL = 15


_LIVE_PIN_SAFE = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:glass"})


@dataclass(frozen=True)
class Violation:
    kind: str                 # "open" | "short" | "weak" | "latch" | "inverted" | "unsupported" | "live_pin" | "hold"
    nets: frozenset[str]
    pos: Pos


def find_live_pin_wire(region, pins: dict[str, Pos]) -> list[Violation]:
    """Anything but air or glass on the cell of a pin that drives it but has no net (a block's unread output, a top-level
    input nothing uses). The pin's repeater powers what is there, and no net's walk starts from it, so verify_signals
    can't see the short. Glass and air don't conduct."""
    size = (abs(region.width), abs(region.height), abs(region.length))
    return [Violation("live_pin", frozenset({name}), pos) for name, pos in pins.items()
            if all(0 <= c < n for c, n in zip(pos, size)) and region[pos].id not in _LIVE_PIN_SAFE]


@lru_cache(maxsize=None)
def _block_kind(bid: str) -> int:
    if bid in ("minecraft:air", "minecraft:cave_air"):
        return _AIR
    if bid == "minecraft:redstone_wire":
        return _DUST
    if bid == "minecraft:repeater":
        return _REPEATER
    if bid == "minecraft:redstone_torch":
        return _TORCH
    if bid == "minecraft:redstone_wall_torch":
        return _WALL_TORCH
    return _OTHER if any(m in bid for m in _NON_OPAQUE_MARKERS) else _OPAQUE


_NOTHING = (_AIR, None)  # a world entry with no block to ask about


def classify_block(block: BlockState) -> tuple[int, int, int, bool]:
    """What _astar.NetChecks needs of a palette entry: kind, _FACING_TO_DELTA of what has a meaningful facing, whether it's
    a redstone block."""
    kind = _block_kind(block.id)
    dx, dz = _FACING_TO_DELTA[block["facing"]] if kind in (_REPEATER, _WALL_TORCH) else (0, 0)
    return kind, dx, dz, block.id == "minecraft:redstone_block"


def _kind_of(block: BlockState) -> tuple[int, BlockState | None]:
    kind = _block_kind(block.id)
    return _NOTHING if kind == _AIR else (kind, block)


class Excluding:
    """Cells of a set, less a few: copying a whole design's terminals per net adds up."""

    def __init__(self, cells, excluded):
        self.cells, self.excluded = cells, excluded

    def __contains__(self, p) -> bool:
        return p in self.cells and p not in self.excluded


class Either:
    """Cells of either of two sets."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __contains__(self, p) -> bool:
        return p in self.a or p in self.b


class HiddenBoxes:
    """Cells of a set, and of a few boxes that would be too many cells to list."""

    def __init__(self, cells, boxes):
        self.cells, self.boxes = cells, boxes

    def __contains__(self, p) -> bool:
        x, y, z = p
        for (lx, ly, lz), (hx, hy, hz) in self.boxes:
            if lx <= x < hx and ly <= y < hy and lz <= z < hz:
                return True
        return p in self.cells


def cell_byte(block: BlockState | None) -> int:
    """What _signals.Grid holds for a block: kind, then facing and delay where it has them."""
    if block is None:
        return 0
    kind = _block_kind(block.id)
    if kind == _AIR:
        return 0
    if block.id == "minecraft:redstone_block":
        return 7
    byte = kind
    if kind in (_REPEATER, _WALL_TORCH):
        byte |= _HORIZ_DIRS.index(_FACING_TO_DELTA[block["facing"]]) << 3
    if kind == _REPEATER:
        byte |= (int(block["delay"]) - 1 if "delay" in block else 0) << 5
    return byte


def signal_grid(region: Region, blocked: Collection[Pos], terminals: Collection[Pos]):
    """The kernel's view of region as _World(region, blocked, terminals) has it, terminals marked as stops. None without
    the kernel, or for what it can't hold: a region not backed by an array, a blocked it can't list, a cell outside."""
    grid = signal_bytes(region, blocked, terminals)
    return None if grid is None else _signals.Grid(grid)


def signal_bytes(region: Region, blocked: Collection[Pos], terminals: Collection[Pos]) -> np.ndarray | None:
    """signal_grid's cell bytes, before they go in a Grid."""
    palette, blocks = getattr(region, "_Region__palette", None), getattr(region, "_Region__blocks", None)
    if _signals is None or blocks is None or not isinstance(blocked, (HiddenBoxes, set, frozenset)):
        return None
    table = np.array([cell_byte(b) for b in palette], dtype=np.uint8)
    size = blocks.shape
    ends = np.array([p for p in terminals if p is not None], dtype=np.int64).reshape(-1, 3)
    if len(ends) and ((ends < 0).any() or (ends >= size).any()):
        return None
    grid = np.asarray(lookup(blocks, table), order="C")
    cells = blocked
    if isinstance(blocked, HiddenBoxes):
        for lo, hi in blocked.boxes:
            grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 0
        cells = blocked.cells
    if cells:
        hidden = np.fromiter(itertools.chain.from_iterable(cells), dtype=np.int64, count=3 * len(cells)).reshape(-1, 3)
        hidden = hidden[((hidden >= 0) & (hidden < size)).all(axis=1)]
        grid[hidden[:, 0], hidden[:, 1], hidden[:, 2]] = 0
    if len(ends):
        at = ends[:, 0], ends[:, 1], ends[:, 2]
        grid[at] = table[blocks[at]] | 128
    return grid


class _World:
    def __init__(self, region: Region, blocked: Collection[Pos], terminals: frozenset[Pos]):
        self.region = region
        self.blocked = blocked
        self.terminals = terminals
        self._kind: dict[Pos, tuple[int, BlockState | None]] = {}
        self._successors: dict[tuple[Pos, str], list[tuple[Pos, str]]] = {}
        self._size = (region.width, region.height, region.length)
        self._palette = getattr(region, "_Region__palette", None)  # straight there, this is read so often
        blocks = getattr(region, "_Region__blocks", None)
        self._blocks = None if blocks is None else memoryview(blocks)  # scalar reads are faster than numpy's
        self._by_palette: dict[int, tuple[int, BlockState | None]] = {}  # what _lookup gives per palette entry

    def changed(self, p: Pos) -> None:
        """The block at p was written. Neighbours' successors depend on it too, so they all go."""
        self._kind.pop(p, None)
        self._successors.clear()

    def kind(self, p: Pos) -> int:
        cached = self._kind.get(p)
        return cached[0] if cached is not None else self._lookup(p)[0]

    def block(self, p: Pos) -> BlockState | None:
        return self._lookup(p)[1]

    def _lookup(self, p: Pos) -> tuple[int, BlockState | None]:
        cached = self._kind.get(p)
        if cached is not None:
            return cached
        x, y, z = p
        w, h, l = self._size
        if not (0 <= x < w and 0 <= y < h and 0 <= z < l):
            result = _NOTHING
        else:
            if self._blocks is not None:
                entry = self._blocks[p]
                result = self._by_palette.get(entry)
                if result is None:
                    b = self._palette[entry]
                    result = self._by_palette[entry] = _kind_of(b)
            else:
                result = _kind_of(self.region[p])
            # Air first: blocked can be millions of cells, and most lookups here are air.
            if result is not _NOTHING and p in self.blocked and p not in self.terminals:
                result = _NOTHING  # component internals are invisible
        self._kind[p] = result
        return result

    # ---- geometry helpers

    def repeater_input(self, p: Pos) -> Pos:
        dx, dz = _FACING_TO_DELTA[self.block(p)["facing"]]
        return p[0] + dx, p[1], p[2] + dz

    def repeater_output(self, p: Pos) -> Pos:
        dx, dz = _FACING_TO_DELTA[self.block(p)["facing"]]
        return p[0] - dx, p[1], p[2] - dz

    def torch_attachment(self, p: Pos) -> Pos:
        if self.kind(p) == _TORCH:
            return p[0], p[1] - 1, p[2]
        dx, dz = _FACING_TO_DELTA[self.block(p)["facing"]]  # wall torch faces away from its wall
        return p[0] - dx, p[1], p[2] - dz

    def is_torch(self, p: Pos) -> bool:
        return self.kind(p) in (_TORCH, _WALL_TORCH)

    def passes_by(self, terminal: Pos) -> bool:
        """A repeater on a pin's wire cell that sends its signal along the wire: the gate's input beside it gets none."""
        if self.kind(terminal) != _REPEATER:
            return False
        out = self.repeater_output(terminal)
        if out in self.blocked or (self.kind(out) == _REPEATER and self.repeater_input(out) == terminal):
            return False
        x, y, z = terminal
        beside = [(x + dx, y, z + dz) for dx, dz in _HORIZ_DIRS]
        return any(q in self.blocked or (self.kind(q) == _REPEATER and self.repeater_input(q) == terminal) for q in beside)

    def dust_links(self, p: Pos) -> list[Pos]:
        """Dust cells p is wired to (same Y, slope up, slope down)."""
        x, y, z = p
        above_open = self.kind((x, y + 1, z)) != _OPAQUE
        links = []
        for dx, dz in _HORIZ_DIRS:
            side = (x + dx, y, z + dz)
            if self.kind(side) == _DUST:
                links.append(side)
                continue
            up = (x + dx, y + 1, z + dz)
            if above_open and self.kind(up) == _DUST:
                links.append(up)
            down = (x + dx, y - 1, z + dz)
            if self.kind(side) != _OPAQUE and self.kind(down) == _DUST:
                links.append(down)
        return links

    def dust_directions(self, p: Pos) -> set[tuple[int, int]]:
        """Horizontal directions dust at p points (its shape)."""
        x, y, z = p
        connected = set()
        for link in self.dust_links(p):
            connected.add((link[0] - x, link[2] - z))
        for dx, dz in _HORIZ_DIRS:
            side = (x + dx, y, z + dz)
            k = self.kind(side)
            if k == _REPEATER:
                if p in (self.repeater_input(side), self.repeater_output(side)):
                    connected.add((dx, dz))
            elif k in (_TORCH, _WALL_TORCH) or (k == _OPAQUE and self.block(side).id == "minecraft:redstone_block"):
                connected.add((dx, dz))
        if not connected:
            return set(_HORIZ_DIRS)  # isolated dust is a cross
        if len(connected) == 1:
            (dx, dz), = connected
            return {(dx, dz), (-dx, -dz)}
        return connected

    # ---- the powers relation: node = (pos, mode); mode only matters for opaque blocks

    def successors(self, node: tuple[Pos, str]) -> list[tuple[Pos, str]]:
        out = self._successors.get(node)
        if out is None:
            out = self._successors[node] = self._find_successors(node)
        return out

    def _find_successors(self, node: tuple[Pos, str]) -> list[tuple[Pos, str]]:
        p, mode = node
        k = self.kind(p)
        out: list[tuple[Pos, str]] = []
        x, y, z = p
        if k == _DUST:
            out.extend((q, "") for q in self.dust_links(p))
            for dx, dz in _HORIZ_DIRS:
                q = (x + dx, y, z + dz)
                if self.kind(q) == _REPEATER and self.repeater_input(q) == p:
                    out.append((q, ""))
            below = (x, y - 1, z)
            if self.kind(below) == _OPAQUE:
                out.append((below, _WEAK))
            for dx, dz in self.dust_directions(p):
                q = (x + dx, y, z + dz)
                if self.kind(q) == _OPAQUE:
                    out.append((q, _WEAK))
        elif k == _REPEATER:
            o = self.repeater_output(p)
            ko = self.kind(o)
            if ko == _DUST:
                out.append((o, ""))
            elif ko == _REPEATER and self.repeater_input(o) == p:
                out.append((o, ""))
            elif ko == _OPAQUE:
                out.append((o, _STRONG))
        elif k in (_TORCH, _WALL_TORCH):
            attachment = self.torch_attachment(p)
            above = (x, y + 1, z)
            if self.kind(above) == _OPAQUE:
                out.append((above, _STRONG))
            for q in [(x + dx, y, z + dz) for dx, dz in _HORIZ_DIRS] + [(x, y - 1, z)]:
                if q == attachment:
                    continue
                kq = self.kind(q)
                if kq == _DUST:
                    out.append((q, ""))
                elif kq == _REPEATER and self.repeater_input(q) == p:
                    out.append((q, ""))
        elif k == _OPAQUE:
            for q in [(x + dx, y, z + dz) for dx, dz in _HORIZ_DIRS] + [(x, y + 1, z), (x, y - 1, z)]:
                kq = self.kind(q)
                if kq in (_TORCH, _WALL_TORCH) and self.torch_attachment(q) == p:
                    out.append((q, ""))
                elif kq == _REPEATER and self.repeater_input(q) == p:
                    out.append((q, ""))
                elif kq == _DUST and mode == _STRONG:
                    out.append((q, ""))
        return out


def _reach(world: _World, source: Pos, stop: frozenset[Pos]) -> set[Pos]:
    """Cells powered (directly or transitively) by source. Doesn't expand past `stop` cells."""
    start_mode = _STRONG if world.kind(source) == _OPAQUE else ""
    start: tuple[Pos, str] = (source, start_mode)
    seen_nodes: set[tuple[Pos, str]] = {start}
    reached = {source}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node[0] in stop and node[0] != source:
            continue
        for nxt in world.successors(node):
            if nxt in seen_nodes:
                continue
            # A strong visit subsumes a weak one.
            if nxt[1] == _WEAK and (nxt[0], _STRONG) in seen_nodes:
                continue
            seen_nodes.add(nxt)
            reached.add(nxt[0])
            queue.append(nxt)
    return reached


def _strengths(world: _World, source: Pos, stop: frozenset[Pos]) -> set[Pos]:
    """Cells source's signal reaches with strength left: dust loses 1 a step, others restore 15."""
    start: tuple[Pos, str] = (source, _STRONG if world.kind(source) == _OPAQUE else "")
    best = {start: _MAX_SIGNAL}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        level = best[node]
        if node[0] in stop and node[0] != source:
            continue
        is_dust = world.kind(node[0]) == _DUST
        for nxt in world.successors(node):
            next_level = level - 1 if is_dust and world.kind(nxt[0]) == _DUST else _MAX_SIGNAL
            if next_level > best.get(nxt, 0):
                best[nxt] = next_level
                queue.append(nxt)
    return {pos for pos, _ in best}


def find_latch(region: Region, source: Pos, stop: frozenset[Pos], world: _World | None = None) -> list[Pos]:
    """The cells of a signal cycle from source through a repeater or torch, if any. world: one to reuse, with nothing hidden."""
    return _latch_cycle(world or _World(region, frozenset(), frozenset()), source, stop)


def _latch_cycle(world: _World, source: Pos, stop: frozenset[Pos]) -> list[Pos]:
    """The cells of a signal cycle from source through a repeater or torch, if any: it holds itself on, as when a
    drop's two torches feed back into the wire that drives them."""
    start: tuple[Pos, str] = (source, _STRONG if world.kind(source) == _OPAQUE else "")
    ids = {start: 0}
    nodes = [start]
    tails: list[int] = []
    heads: list[int] = []
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node[0] in stop and node[0] != source:
            continue
        tail = ids[node]
        for nxt in world.successors(node):
            head = ids.get(nxt)
            if head is None:
                head = ids[nxt] = len(nodes)
                nodes.append(nxt)
                queue.append(nxt)
            tails.append(tail)
            heads.append(head)
    drivers = [i for i, (pos, _) in enumerate(nodes) if world.kind(pos) in (_REPEATER, _TORCH, _WALL_TORCH)]
    if not drivers:
        return []
    graph = csr_matrix((np.ones(len(tails), dtype=np.int8), (tails, heads)), shape=(len(nodes), len(nodes)))
    _, labels = connected_components(graph, directed=True, connection="strong")
    sizes = np.bincount(labels)
    for label in sorted({labels[i] for i in drivers if sizes[labels[i]] > 1}):
        return sorted({nodes[i][0] for i in np.flatnonzero(labels == label).tolist()})
    return []


def find_gate_contacts(region: Region, cells: frozenset[Pos], gate_cells: frozenset[Pos],
                       world: _World | None = None) -> list[Pos]:
    """Dust among cells that powers a solid block inside a gate. That block can carry the gate's torch, which then
    switches on this wire. Verify can't see it: gate interiors are hidden from it."""
    world = world or _World(region, frozenset(), frozenset())
    hits = []
    for p in cells:
        if world.kind(p) != _DUST:
            continue
        x, y, z = p
        below = (x, y - 1, z)
        pointed = [(x + dx, y, z + dz) for dx, dz in world.dust_directions(p)]
        if any(q in gate_cells and world.kind(q) == _OPAQUE for q in (*pointed, below)):
            hits.append(p)
    return hits


def find_inverted_contacts(region: Region, source: Pos, sinks: Iterable[Pos], stop: frozenset[Pos],
                           blocked: frozenset[Pos] = frozenset(), visible: frozenset[Pos] = frozenset()) -> list[Pos]:
    """_inverted_contacts for a net being routed; `blocked` cells are air unless `visible`."""
    return _inverted_contacts(_World(region, blocked, visible), source, frozenset(sinks), stop)


def _inverted_contacts(world: _World, source: Pos, sinks: frozenset[Pos], stop: frozenset[Pos]) -> list[Pos]:
    """Where the opposite of source's signal gets into its wire. Follows signal with its polarity
    (torches flip it): wire reached both ways where opposite signal enters from a block, a torch
    or opposite-only wire, and sinks reached only opposite."""
    start = ((source, _STRONG if world.kind(source) == _OPAQUE else ""), False)
    seen = {start}
    queue = deque([start])
    polarities: dict[Pos, set[bool]] = {source: {False}}
    feeds: list[tuple[Pos, Pos]] = []
    while queue:
        node, flipped = queue.popleft()
        p = node[0]
        if p in stop and p != source:
            continue
        out = flipped ^ (world.is_torch(p) and p != source)
        for nxt in world.successors(node):
            if out:
                feeds.append((p, nxt[0]))
            polarities.setdefault(nxt[0], set()).add(out)
            if (nxt, out) not in seen:
                seen.add((nxt, out))
                queue.append((nxt, out))
    wired = (_DUST, _REPEATER)
    contacts = {q for p, q in feeds if len(polarities[q]) == 2 and world.kind(q) in wired
                and (world.kind(p) not in wired or polarities[p] == {True})}
    contacts |= {q for q in sinks if polarities.get(q) == {True}}
    return sorted(contacts)


class VerifyMemo:
    """What a verify_signals found net by net, for the next one over the same nets to search again only the nets near
    cells that changed in between (a repair round re-routes a few nets). Results are the same as searching them all."""

    def __init__(self):
        self.cells: np.ndarray | None = None  # the cell bytes searched
        self.terminals: frozenset[Pos] | None = None
        self.blocked: object = None
        self.found: dict[str, tuple[tuple[Pos, list[Pos]], list[Violation], np.ndarray]] = {}  # net -> spec, faults, keys


_MEMO_REACH = 3  # cells a search reads around those it reaches: a dust's shape reads its neighbours' links, 2 away
_MEMO_MAX_CHANGED = 20_000  # changed cells past which searching everything again is quicker


def verify_signals(
        region: Region,
        nets: dict[str, tuple[Pos, list[Pos]]],
        owner: dict[Pos, str],
        blocked: Collection[Pos] = frozenset(),
        on_net: Callable[[str], None] | None = None,
        memo: VerifyMemo | None = None,
) -> list[Violation]:
    """Opens (unreached sinks) and shorts (cross-net reach) for the given nets.

    nets: net_id -> (source cell, sink cells). owner: cell -> owning net (e.g. dust_owner).
    blocked: component-internal cells, treated as air unless they are a terminal.
    on_net: called with each net id once it has been checked.
    memo: kept between calls over much the same wire (see VerifyMemo).
    """
    terminals = frozenset(p for src, sinks in nets.values() for p in (src, *sinks))
    world = _World(region, blocked, terminals)
    cells = signal_bytes(region, blocked, terminals) if _signals is not None else None
    grid = None if cells is None else _signals.Grid(cells)
    if grid is not None and hasattr(grid, "check_net_keys"):
        return _verify_on_grid(region, nets, owner, blocked, on_net, terminals, world, grid, memo, cells)
    violations: list[Violation] = []
    reached_by: dict[Pos, str] = {}
    short_pairs: set[frozenset[str]] = set()

    def short(a: str, b: str, pos: Pos) -> None:
        pair = frozenset((a, b))
        if pair not in short_pairs:
            short_pairs.add(pair)
            violations.append(Violation("short", pair, pos))

    for net_id, (source, sinks) in nets.items():
        # Stop at other nets' terminals (their pins lead into components); past this net's
        # own sinks there is only component interior, which `blocked` already hides.
        own = {source, *sinks}
        stop = Excluding(terminals, own)
        if grid is not None:
            order, strong, has_latch, inverted = grid.check_net(source, sinks, own)
            reached = set(order)  # built in the same order as _reach's: which cell a short reports depends on it
            strong_enough = {sink for sink, ok in zip(sinks, strong) if ok}
        else:
            reached = _reach(world, source, stop)
            strong_enough = _strengths(world, source, stop)
        for sink in sinks:
            if sink not in reached or world.passes_by(sink):
                violations.append(Violation("open", frozenset({net_id}), sink))
        for sink in sinks:
            if sink in reached and sink not in strong_enough:
                violations.append(Violation("weak", frozenset({net_id}), sink))
        latch = _latch_cycle(world, source, stop) if grid is None or has_latch else []
        if latch:
            violations.append(Violation("latch", frozenset({net_id}), latch[0]))
        contacts = _inverted_contacts(world, source, frozenset(sinks), stop)[:1] if grid is None else \
            ([] if inverted is None else [inverted])
        for pos in contacts:
            violations.append(Violation("inverted", frozenset({net_id}), pos))
        for pos in reached:
            cell_owner = owner.get(pos)
            if cell_owner is not None and cell_owner != net_id and pos not in blocked:
                short(net_id, cell_owner, pos)
            other = reached_by.get(pos)
            if other is not None and other != net_id and world.kind(pos) != _OPAQUE:
                short(net_id, other, pos)
            reached_by.setdefault(pos, net_id)
        if on_net is not None:
            on_net(net_id)
    return violations


def _unchanged_nets(memo: VerifyMemo, nets: dict[str, tuple[Pos, list[Pos]]], cells: np.ndarray, blocked,
                    terminals: frozenset[Pos]) -> set[str]:
    """Nets whose last search memo has and that no changed cell comes near: the same search would find the same."""
    if (memo.cells is None or memo.cells.shape != cells.shape or memo.blocked is not blocked
            or memo.terminals != terminals):
        return set()
    changed = np.flatnonzero(memo.cells.reshape(-1) != cells.reshape(-1))
    if len(changed) > _MEMO_MAX_CHANGED:
        return set()
    names = [n for n, spec in nets.items() if n in memo.found and memo.found[n][0] == spec]
    if not len(changed):
        return set(names)
    _, h, l = cells.shape
    r = np.arange(-_MEMO_REACH, _MEMO_REACH + 1, dtype=np.int64)
    offsets = (r[:, None, None] * (h * l) + r[None, :, None] * l + r[None, None, :]).reshape(-1)
    near = np.unique((changed[:, None] + offsets[None, :]).reshape(-1))  # wrapping at an edge only adds cells
    watched: list[np.ndarray] = []
    for n in names:
        (source, sinks), _, keys = memo.found[n]
        ends = np.array([source, *sinks], dtype=np.int64).reshape(-1, 3)
        watched.append(np.concatenate((keys, (ends[:, 0] * h + ends[:, 1]) * l + ends[:, 2])))
    if not names:
        return set()
    owner_of = np.repeat(np.arange(len(names)), [len(w) for w in watched])
    hit = np.isin(np.concatenate(watched), near)
    dirty = set(owner_of[hit].tolist())
    return {n for i, n in enumerate(names) if i not in dirty}


def _verify_on_grid(region: Region, nets: dict[str, tuple[Pos, list[Pos]]], owner: dict[Pos, str],
                    blocked: Collection[Pos], on_net: Callable[[str], None] | None, terminals: frozenset[Pos],
                    world: _World, grid, memo: VerifyMemo | None = None,
                    cells: np.ndarray | None = None) -> list[Violation]:
    """verify_signals on the kernel, the same violations in the same order. Every net is searched first; then the
    cells reached are checked for shorts in bulk, and only nets with a suspect cell go through them one by one."""
    _, h, l = region._Region__blocks.shape

    def check(item: tuple[str, tuple[Pos, list[Pos]]]) -> tuple[list[Violation], np.ndarray]:
        net_id, (source, sinks) = item
        own = {source, *sinks}
        stop = Excluding(terminals, own)
        keys, strong, has_latch, inverted, in_reach = grid.check_net_keys(source, sinks, own)
        strong_enough = {sink for sink, ok in zip(sinks, strong) if ok}
        net: list[Violation] = []
        for sink, hit in zip(sinks, in_reach):
            if not hit or world.passes_by(sink):
                net.append(Violation("open", frozenset({net_id}), sink))
        for sink, hit in zip(sinks, in_reach):
            if hit and sink not in strong_enough:
                net.append(Violation("weak", frozenset({net_id}), sink))
        latch = _latch_cycle(world, source, stop) if has_latch else []
        if latch:
            net.append(Violation("latch", frozenset({net_id}), latch[0]))
        if inverted is not None:
            net.append(Violation("inverted", frozenset({net_id}), inverted))
        return net, keys

    # Nets are searched one by one, each on its own: forked processes take a share each.
    same = _unchanged_nets(memo, nets, cells, blocked, terminals) if memo is not None and cells is not None else set()
    searched = iter(fork_map(check, [item for item in nets.items() if item[0] not in same]))
    checked = [memo.found[n][1:] if n in same else next(searched) for n in nets]
    found: list[list[Violation]] = [net for net, _ in checked]
    reached: list[np.ndarray] = [keys for _, keys in checked]
    if memo is not None and cells is not None:
        memo.cells, memo.terminals, memo.blocked = cells, terminals, blocked
        memo.found = {n: (spec, net, keys) for (n, spec), (net, keys) in zip(nets.items(), checked)}
    del checked
    if on_net is not None:
        for net_id in nets:
            on_net(net_id)

    names = list(nets)
    ids = {net_id: i for i, net_id in enumerate(names)}
    all_keys = np.concatenate(reached) if reached else np.empty(0, dtype=np.int64)
    offsets = np.cumsum([0, *map(len, reached)])
    net_of = np.repeat(np.arange(len(reached)), np.diff(offsets))
    suspect = np.zeros(len(reached), dtype=bool)
    first_net = np.empty(len(all_keys), dtype=np.int64)
    if len(all_keys):
        by_key = np.argsort(all_keys, kind="stable")
        sorted_keys = all_keys[by_key]
        sorted_nets = net_of[by_key]
        # Owned by another net. Which cells are blocked is left to the check one by one.
        if owner:
            cells = np.fromiter(itertools.chain.from_iterable(owner), dtype=np.int64, count=3 * len(owner)).reshape(-1, 3)
            values = list(owner.values())
            for name in set(values):
                ids.setdefault(name, len(ids))
            owners = np.fromiter(map(ids.__getitem__, values), dtype=np.int64, count=len(values))
            inside = ((cells >= 0) & (cells < region._Region__blocks.shape)).all(axis=1)
            cell_keys = (cells[inside, 0] * h + cells[inside, 1]) * l + cells[inside, 2]
            owners = owners[inside]
            if len(cell_keys):
                order = np.argsort(cell_keys)
                cell_keys, owners = cell_keys[order], owners[order]
                at = np.minimum(np.searchsorted(cell_keys, sorted_keys), len(cell_keys) - 1)
                owned = (cell_keys[at] == sorted_keys) & (owners[at] != sorted_nets)
                suspect[sorted_nets[owned]] = True
        # Reached by an earlier net: the first net to reach each cell, against each net reaching it.
        starts = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
        group_start = np.maximum.accumulate(np.where(starts, np.arange(len(sorted_keys)), 0))
        first_net[by_key] = sorted_nets[group_start]
        suspect[net_of[first_net != net_of]] = True

    violations: list[Violation] = []
    short_pairs: set[frozenset[str]] = set()

    def short(a: str, b: str, pos: Pos) -> None:
        pair = frozenset((a, b))
        if pair not in short_pairs:
            short_pairs.add(pair)
            violations.append(Violation("short", pair, pos))

    for k, net_id in enumerate(names):
        violations += found[k]
        if not suspect[k]:
            continue
        keys = all_keys[offsets[k]:offsets[k + 1]]
        cells = list(zip((keys // (h * l)).tolist(), ((keys // l) % h).tolist(), (keys % l).tolist()))
        earlier = {pos: names[first] for pos, first in zip(cells, first_net[offsets[k]:offsets[k + 1]].tolist()) if first != k}
        for pos in set(cells):  # the order verify_signals goes through them in, which decides the cell a short reports
            cell_owner = owner.get(pos)
            if cell_owner is not None and cell_owner != net_id and pos not in blocked:
                short(net_id, cell_owner, pos)
            other = earlier.get(pos)
            if other is not None and world.kind(pos) != _OPAQUE:
                short(net_id, other, pos)
    return violations


_NEEDS_FLOOR = ("minecraft:redstone_wire", "minecraft:repeater", "minecraft:comparator", "minecraft:redstone_torch")
_NO_FLOOR_MARKERS = ("air", "redstone_wire", "torch", "repeater", "comparator", "lever", "button", "pressure_plate")


def _gives_support(block: BlockState) -> bool:
    """Whether `block` has a sturdy face to hold dust/repeaters/torches (simplified)."""
    bid = block.id
    if any(m in bid for m in _NO_FLOOR_MARKERS):
        return False
    if bid.endswith("_slab"):
        return block["type"] in ("top", "double")
    if bid.endswith("_trapdoor"):
        return block["half"] == "top" and block["open"] != "true"
    return True


def verify_support(
        region: Region,
        owner: dict[Pos, str],
        blocked: Collection[Pos] = frozenset(),
        skip: Iterable[tuple[Pos, Pos]] = (),
) -> list[Violation]:
    """Dust, repeaters, comparators and torches outside `blocked` with no supporting block.

    Floor-standing blocks need a sturdy block below; a wall torch needs one behind it.
    In-game such blocks pop off (or can't be pasted), so the circuit silently breaks.
    skip: boxes (lowest cell, one past the highest) not to look in at all.
    """
    palette = region._Region__palette
    blocks = region._Region__blocks
    # Per palette entry: whether it needs support, from where, and whether it gives some.
    offsets = np.zeros((len(palette), 3), dtype=np.int64)
    needs = np.zeros(len(palette), dtype=bool)
    for i, b in enumerate(palette):
        if b.id == "minecraft:redstone_wall_torch":
            dx, dz = _FACING_TO_DELTA[b["facing"]]  # faces away from its wall
            offsets[i] = (-dx, 0, -dz)
            needs[i] = True
        elif b.id in _NEEDS_FLOOR:
            offsets[i] = (0, -1, 0)
            needs[i] = True
    if not needs.any():
        return []
    gives = np.array([_gives_support(b) for b in palette], dtype=bool)
    cells = cells_where(blocks, needs, skip)
    support = cells + offsets[blocks[cells[:, 0], cells[:, 1], cells[:, 2]]]
    in_region = ((support >= 0) & (support < (region.width, region.height, region.length))).all(axis=1)
    held = np.zeros(len(cells), dtype=bool)
    inside = support[in_region]
    held[in_region] = gives[blocks[inside[:, 0], inside[:, 1], inside[:, 2]]]
    violations: list[Violation] = []
    for p in map(tuple, cells[~held].tolist()):
        if p in blocked:
            continue
        net = owner.get(p)
        violations.append(Violation("unsupported", frozenset({net} if net else ()), p))
    return violations
