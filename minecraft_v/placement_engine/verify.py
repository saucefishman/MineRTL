"""Electrical verification of a routed layout: per-net opens and cross-net shorts.

Builds a directed "powers" relation over blocks (Java redstone, simplified to the
structures the router builds) and searches from each net's source:
- open:  a sink of the net that its source can't reach;
- short: a cell reached from two nets' sources, or a cell owned by another net.

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

from collections import deque
from dataclasses import dataclass

from litemapy import BlockState, Region

from .constants import _DELTA_TO_FACING, _HORIZ_DIRS

Pos = tuple[int, int, int]

_FACING_TO_DELTA = {facing: delta for delta, facing in _DELTA_TO_FACING.items()}
_NON_OPAQUE_MARKERS = ("glass", "trapdoor", "redstone_wire", "torch", "repeater", "comparator", "slab")

_AIR, _DUST, _REPEATER, _TORCH, _WALL_TORCH, _OPAQUE, _OTHER = range(7)
_STRONG, _WEAK = "S", "W"


@dataclass(frozen=True)
class Violation:
    kind: str                 # "open" | "short"
    nets: frozenset[str]
    pos: Pos


class _World:
    def __init__(self, region: Region, blocked: frozenset[Pos], terminals: frozenset[Pos]):
        self.region = region
        self.blocked = blocked
        self.terminals = terminals
        self._kind: dict[Pos, tuple[int, BlockState | None]] = {}

    def _in_region(self, p: Pos) -> bool:
        r = self.region
        return 0 <= p[0] < r.width and 0 <= p[1] < r.height and 0 <= p[2] < r.length

    def kind(self, p: Pos) -> int:
        return self._lookup(p)[0]

    def block(self, p: Pos) -> BlockState | None:
        return self._lookup(p)[1]

    def _lookup(self, p: Pos) -> tuple[int, BlockState | None]:
        cached = self._kind.get(p)
        if cached is not None:
            return cached
        if not self._in_region(p) or (p in self.blocked and p not in self.terminals):
            result = (_AIR, None)  # component internals are invisible
        else:
            b = self.region[p]
            bid = b.id
            if bid in ("minecraft:air", "minecraft:cave_air"):
                k = _AIR
            elif bid == "minecraft:redstone_wire":
                k = _DUST
            elif bid == "minecraft:repeater":
                k = _REPEATER
            elif bid == "minecraft:redstone_torch":
                k = _TORCH
            elif bid == "minecraft:redstone_wall_torch":
                k = _WALL_TORCH
            elif any(m in bid for m in _NON_OPAQUE_MARKERS):
                k = _OTHER
            else:
                k = _OPAQUE
            result = (k, b)
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


def verify_signals(
        region: Region,
        nets: dict[str, tuple[Pos, list[Pos]]],
        owner: dict[Pos, str],
        blocked: frozenset[Pos] = frozenset(),
) -> list[Violation]:
    """Opens (unreached sinks) and shorts (cross-net reach) for the given nets.

    nets: net_id -> (source cell, sink cells). owner: cell -> owning net (e.g. dust_owner).
    blocked: component-internal cells, treated as air unless they are a terminal.
    """
    terminals = frozenset(p for src, sinks in nets.values() for p in (src, *sinks))
    world = _World(region, blocked, terminals)
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
        reached = _reach(world, source, frozenset(t for t in terminals if t not in own))
        for sink in sinks:
            if sink not in reached:
                violations.append(Violation("open", frozenset({net_id}), sink))
        for pos in reached:
            cell_owner = owner.get(pos)
            if cell_owner is not None and cell_owner != net_id and pos not in blocked:
                short(net_id, cell_owner, pos)
            other = reached_by.get(pos)
            if other is not None and other != net_id and world.kind(pos) != _OPAQUE:
                short(net_id, other, pos)
            reached_by.setdefault(pos, net_id)
    return violations
