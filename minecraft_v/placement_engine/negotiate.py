"""Negotiated-congestion routing state (PathFinder, McMurchie & Ebeling 1995).

During negotiation passes each net is routed *virtually*: A* and the laying code see the
static world (components, I/O repeaters, const sources) plus only that net's own blocks,
via OverlayWorkspace / OverlaySet / OverlayDict. Other nets are invisible to legality
checks; they only show up as cost through `Congestion`, so overlaps ("shorts") are
allowed and get more expensive each pass. Once a pass ends with no overuse (or passes
run out), routes are committed into the real workspace (builder._commit_route).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from litemapy import BlockState, Region

from .block_utils import _is_redstone_wire, _is_repeater, _is_torch
from .constants import _HORIZ_DIRS
from .pathfinding import _BlockIndex

Pos = tuple[int, int, int]


class OverlaySet(set):
    """Copy of a static base set that records what gets added on top.

    A real set (not a read-through wrapper) so A*'s hot `in` checks stay native; the
    copy is made once per net reroute.
    """

    def __init__(self, base):
        super().__init__(base)
        self.top: set[Pos] = set()

    def add(self, key) -> None:
        if not set.__contains__(self, key):
            self.top.add(key)
        set.add(self, key)

    def discard(self, key) -> None:
        self.top.discard(key)
        set.discard(self, key)


class OverlayDict(dict):
    """Copy of a static base dict that records what gets set on top (see OverlaySet)."""

    def __init__(self, base):
        super().__init__(base)
        self.top: dict[Pos, str] = {}

    def __setitem__(self, key, value) -> None:
        self.top[key] = value
        dict.__setitem__(self, key, value)

    def pop(self, key, *default):
        self.top.pop(key, None)
        return dict.pop(self, key, *default)


class OverlayWorkspace:
    """Region view: writes land in `overlay`, reads fall through to the static base."""

    def __init__(self, base: Region, base_index: _BlockIndex):
        self.base = base
        self.overlay: dict[Pos, BlockState] = {}
        self._base_index = base_index

    def __getitem__(self, pos) -> BlockState:
        pos = tuple(pos)
        block = self.overlay.get(pos)
        return self.base[pos] if block is None else block

    def __setitem__(self, pos, block: BlockState) -> None:
        self.overlay[tuple(pos)] = block

    def block_index(self) -> _BlockIndex:
        return _BlockIndex.layered(self._base_index, self.overlay)


@dataclass(frozen=True)
class NetRoute:
    """One net's virtual routing result and the cells it claims."""
    overlay: dict[Pos, BlockState]
    solid: frozenset[Pos]           # solid cells it placed (supports, tower stones, ...)
    owner: dict[Pos, str]           # dust_owner entries it set
    inverted: frozenset[Pos]
    signal: frozenset[Pos]          # cells carrying its signal (wire, repeaters, torches, owned blocks)
    support: frozenset[Pos]         # solid cells that don't carry signal
    halo: frozenset[Pos]            # cells where another net's signal would short against it

    @classmethod
    def from_view(cls, view: OverlayWorkspace, solid: OverlaySet, owner: OverlayDict,
                  inverted: set[Pos]) -> NetRoute:
        signal = {
            p for p, b in view.overlay.items()
            if _is_redstone_wire(b) or _is_repeater(b) or _is_torch(b)
        }
        signal |= owner.top.keys()
        signal |= inverted
        halo: set[Pos] = set()
        for x, y, z in signal:
            # Mirrors _wire_walkable: cardinal neighbours at Y-1/0/+1, plus directly above/below.
            for dx, dz in _HORIZ_DIRS:
                for dy in (-1, 0, 1):
                    halo.add((x + dx, y + dy, z + dz))
            halo.add((x, y + 1, z))
            halo.add((x, y - 1, z))
        halo -= signal
        return cls(
            overlay=dict(view.overlay),
            solid=frozenset(solid.top),
            owner=dict(owner.top),
            inverted=frozenset(inverted),
            signal=frozenset(signal),
            support=frozenset(solid.top - signal),
            halo=frozenset(halo),
        )


class Congestion:
    """Which nets use each cell, plus PathFinder history cost."""

    def __init__(self) -> None:
        self.signal: dict[Pos, set[str]] = defaultdict(set)
        self.support: dict[Pos, set[str]] = defaultdict(set)
        self.halo: dict[Pos, set[str]] = defaultdict(set)
        self.hist: dict[Pos, float] = defaultdict(float)

    def _apply(self, net_id: str, route: NetRoute, add: bool) -> None:
        for table, cells in ((self.signal, route.signal), (self.support, route.support), (self.halo, route.halo)):
            for pos in cells:
                if add:
                    table[pos].add(net_id)
                else:
                    users = table.get(pos)
                    if users is not None:
                        users.discard(net_id)
                        if not users:
                            del table[pos]

    def add(self, net_id: str, route: NetRoute) -> None:
        self._apply(net_id, route, True)

    def remove(self, net_id: str, route: NetRoute) -> None:
        self._apply(net_id, route, False)

    def users(self, pos: Pos) -> int:
        """Other nets a new signal cell at pos would conflict with (caller's net is ripped up)."""
        return len(self.signal.get(pos, ())) + len(self.support.get(pos, ())) + len(self.halo.get(pos, ()))

    def cost(self, pos: Pos, pres: float, base: float) -> float:
        """Extra cost for using pos: history always, present congestion scaled by pres.

        PathFinder c(n) = (b(n) + h(n)) * p(n), p(n) = 1 + pres * users; the A* move
        cost plays b(n), so this returns the part beyond it (with b(n) taken as `base`).
        """
        h = self.hist.get(pos, 0.0)
        users = self.users(pos)
        return base * (h + (1 + h) * pres * users)

    def overused(self) -> set[Pos]:
        """Cells where a net's signal shares a cell with, or sits in the halo of, another net."""
        over = set()
        for pos, sig_nets in self.signal.items():
            if (len(sig_nets) > 1
                    or self.support.get(pos, set()) - sig_nets
                    or self.halo.get(pos, set()) - sig_nets):
                over.add(pos)
        return over

    def nets_at(self, cells) -> set[str]:
        nets: set[str] = set()
        for pos in cells:
            nets |= self.signal.get(pos, set())
            nets |= self.support.get(pos, set())
            nets |= self.halo.get(pos, set())
        return nets
