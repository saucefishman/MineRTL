"""Per-net journaling of routing mutations so a net can be ripped up and rerouted.

Laying code (wire.py) mutates the workspace, `solid`, `dust_owner` and `inverted_cells`
directly. Instead of threading undo logic through it, those containers are replaced by
journaling wrappers: while `Recorder.recording(net_id)` is active, every mutation logs
the previous state into that net's journal. `rip_up` replays the journal in reverse.

A* reads the raw Region and the containers' native (unoverridden) read methods, so
search speed is unaffected; only the laying calls go through `JournalingWorkspace`.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

from litemapy import BlockState, Region

from .block_utils import _is_redstone_wire, _is_repeater, _is_torch, _block_str
from .constants import _HORIZ_DIRS
from .pathfinding import _BlockIndex

_MISSING = object()

# Journal entry: (kind, target, key, prev, prev_writer)
#   kind 'block': target=Region, prev=BlockState, prev_writer=net that wrote prev (or None)
#   kind 'set':   target=JournaledSet, prev=bool (was present)
#   kind 'dict':  target=JournaledDict, prev=value or _MISSING
_Entry = tuple[str, Any, Any, Any, Any]


class Recorder:
    def __init__(self) -> None:
        self.journals: dict[str, list[_Entry]] = defaultdict(list)
        self.active: str | None = None
        # Latest net to write each block via JournalingWorkspace — identifies who depends on a support.
        self.writer: dict[tuple[int, int, int], str] = {}

    @contextmanager
    def recording(self, net_id: str) -> Iterator[None]:
        prev = self.active
        self.active = net_id
        try:
            yield
        finally:
            self.active = prev

    def log(self, entry: _Entry) -> None:
        self.journals[self.active].append(entry)


class JournaledSet(set):
    def __init__(self, recorder: Recorder, iterable=(), name: str = ""):
        super().__init__(iterable)
        self._rec = recorder
        self.name = name

    def add(self, key) -> None:
        if self._rec.active is not None and not set.__contains__(self, key):
            self._rec.log(("set", self, key, False, None))
        set.add(self, key)

    def discard(self, key) -> None:
        if self._rec.active is not None and set.__contains__(self, key):
            self._rec.log(("set", self, key, True, None))
        set.discard(self, key)

    def _restore(self, key, present: bool) -> None:
        if present:
            set.add(self, key)
        else:
            set.discard(self, key)


class JournaledDict(dict):
    def __init__(self, recorder: Recorder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rec = recorder

    def __setitem__(self, key, value) -> None:
        if self._rec.active is not None:
            self._rec.log(("dict", self, key, dict.get(self, key, _MISSING), None))
        dict.__setitem__(self, key, value)

    def pop(self, key, *default):
        if self._rec.active is not None and dict.__contains__(self, key):
            self._rec.log(("dict", self, key, dict.__getitem__(self, key), None))
        return dict.pop(self, key, *default)

    def _restore(self, key, prev) -> None:
        if prev is _MISSING:
            dict.pop(self, key, None)
        else:
            dict.__setitem__(self, key, prev)


class JournalingWorkspace:
    """Write-through Region proxy that journals block writes for the active net."""

    def __init__(self, region: Region, recorder: Recorder):
        self.region = region
        self._rec = recorder

    def __getitem__(self, pos) -> BlockState:
        return self.region[pos]

    def block_index(self):
        # Stacked on an IndexedWorkspace: reuse its live index instead of rescanning.
        inner = getattr(self.region, "block_index", None)
        return inner() if inner is not None else _BlockIndex(self.region)

    def __setitem__(self, pos, block: BlockState) -> None:
        pos = tuple(pos)
        rec = self._rec
        if rec.active is not None:
            rec.log(("block", self.region, pos, self.region[pos], rec.writer.get(pos)))
            rec.writer[pos] = rec.active
        self.region[pos] = block


def _support_dependent(
        region: Region,
        rec: Recorder,
        dust_owner: dict[tuple[int, int, int], str],
        pos: tuple[int, int, int],
        own_written: set[tuple[int, int, int]],
) -> str | None:
    """Net (other than the one being ripped) whose blocks rest on / attach to the block at pos."""
    x, y, z = pos
    above = (x, y + 1, z)
    if above not in own_written:
        try:
            blk = region[above]
        except IndexError:
            blk = None
        if blk is not None and (_is_redstone_wire(blk) or _is_repeater(blk) or _is_torch(blk)):
            owner = dust_owner.get(above) or rec.writer.get(above)
            if owner is not None:
                return owner
    for dx, dz in _HORIZ_DIRS:
        side = (x + dx, y, z + dz)
        if side in own_written:
            continue
        try:
            blk = region[side]
        except IndexError:
            continue
        if _block_str(blk) == "minecraft:redstone_wall_torch":
            owner = dust_owner.get(side) or rec.writer.get(side)
            if owner is not None:
                return owner
    return None


def rip_up(
        rec: Recorder,
        net_id: str,
        region: Region,
        solid: JournaledSet,
        dust_owner: JournaledDict,
) -> None:
    """Revert every mutation journaled for net_id.

    Support preservation: a solid block this net placed that another net's dust /
    repeater / torch now rests on (shared support via `below in solid`) is kept, and its
    block + solid entries are handed to that net's journal so it is removed with it.
    """
    entries = rec.journals.pop(net_id, [])
    if not entries:
        return
    own_written = {e[2] for e in entries if e[0] == "block"}
    keep: dict[tuple[int, int, int], str | None] = {}  # pos -> dependent net (None = revert)
    transfers: dict[str, list[_Entry]] = defaultdict(list)
    for entry in reversed(entries):
        kind, target, key, prev, prev_writer = entry
        if kind == "block" or (kind == "set" and target is solid):
            # Decided on first (newest) encounter, while this net's own blocks above still exist.
            if key not in keep:
                keep[key] = _support_dependent(region, rec, dust_owner, key, own_written) if key in solid else None
            dependent = keep[key]
            if dependent is not None:
                transfers[dependent].append(entry)
                if kind == "block":
                    rec.writer[key] = dependent
                continue
        if kind == "block":
            target[key] = prev
            if prev_writer is None:
                rec.writer.pop(key, None)
            else:
                rec.writer[key] = prev_writer
        else:
            target._restore(key, prev)
    for dependent, moved in transfers.items():
        # moved is newest-first; journals replay newest-last, so restore original order.
        rec.journals[dependent].extend(reversed(moved))


def rollback(rec: Recorder, net_id: str, mark: int) -> None:
    """Revert net_id's mutations journaled after position mark (its journal length then).

    For undoing the net's own latest attempt, so no support preservation: nothing laid
    since mark can have been built on by another net.
    """
    entries = rec.journals.get(net_id)
    if not entries or len(entries) <= mark:
        return
    tail = entries[mark:]
    del entries[mark:]
    for kind, target, key, prev, prev_writer in reversed(tail):
        if kind == "block":
            target[key] = prev
            if prev_writer is None:
                rec.writer.pop(key, None)
            else:
                rec.writer[key] = prev_writer
        else:
            target._restore(key, prev)


def invalid_supports(
        region: Region,
        solid: set[tuple[int, int, int]],
        dust_owner: dict[tuple[int, int, int], str],
) -> list[tuple[int, int, int]]:
    """Owned dust cells with nothing solid below (y > 0). Debug/test helper."""
    bad = []
    for x, y, z in dust_owner:
        if y > 0 and _is_redstone_wire(region[x, y, z]) and (x, y - 1, z) not in solid:
            bad.append((x, y, z))
    return bad
