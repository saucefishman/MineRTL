"""Saved placement and routed paths, so a test build can skip the work it isn't testing.

Jobs and their branches are laid again in the order they last finished, without searching, which rebuilds the same world
for the nets that are searched. A net ripped and routed late goes late again, or its final path could block what was laid
before it; a branch keeps its place because repeaters are placed walking the branches in order. Waypoints keep their
cells too, as other paths hang off them.
Nothing here notices a change to the placer or the netlist: use a new directory after one.
"""
from __future__ import annotations

import copyreg
import os
import pickle
from collections.abc import Callable
from pathlib import Path

from litemapy import BlockState

Pos = tuple[int, int, int]


def _pickle_block(block: BlockState):
    return _make_block, (block.id, dict(block.properties()))


def _make_block(block_id: str, properties: dict) -> BlockState:
    return BlockState(block_id, **properties)


copyreg.pickle(BlockState, _pickle_block)


class ResumeStore:
    def __init__(self, directory: str | Path, reroute: set[str] | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.reroute = reroute or set()
        self.saved: dict[tuple[str, Pos], list[Pos]] = self._load("paths.pkl") or {}  # from the last build
        self.paths: dict[tuple[str, Pos], list[Pos]] = {}  # from this one
        self.saved_rank = {entry: i for i, entry in enumerate(self.saved)}
        self.saved_order: list[str] = self._load("order.pkl") or []  # job keys as the last build finished them
        self.order: dict[str, None] = {}  # as this one has, oldest first
        self.saved_cells: dict[tuple[str, Pos], Pos | None] = self._load("cells.pkl") or {}  # waypoint -> cell it got
        self.cells: dict[tuple[str, Pos], Pos | None] = {}
        self.reused = self.searched = 0

    def _load(self, name: str):
        path = self.dir / name
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _dump(self, name: str, value) -> None:
        tmp = self.dir / f"{name}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(value, f)
        os.replace(tmp, self.dir / name)

    def load_placed(self, seed: int):
        return self._load(f"placed_{seed}.pkl")

    def save_placed(self, seed: int, placed) -> None:
        self._dump(f"placed_{seed}.pkl", placed)

    def path_for(self, key: str, dst: Pos) -> list[Pos] | None:
        # Each once: a second ask is a retry, which needs a new search.
        path = self.saved.pop((key, dst), None)
        if key in self.reroute:
            path = None
        if path is None:
            self.searched += 1
        else:
            self.reused += 1
        return path

    def remember(self, key: str, dst: Pos, path: list[Pos]) -> None:
        # One cell: a second ask for a target already on the net's wire. The path that laid it is the one to keep.
        if len(path) > 1 or (key, dst) not in self.paths:
            self.paths.pop((key, dst), None)  # in the order laid
            self.paths[(key, dst)] = path

    def forget(self, key: str, dst: Pos) -> None:
        self.paths.pop((key, dst), None)

    def in_saved_order(self, jobs: list) -> list:
        """jobs in the order the last build finished them; any it didn't finish keep their order, last."""
        rank = {key: i for i, key in enumerate(self.saved_order)}
        return sorted(jobs, key=lambda job: rank.get(job.key, len(rank)))

    def targets_in_saved_order(self, key: str, targets: list[Pos]) -> list[Pos]:
        """A job's targets in the order the last build laid them; any it didn't lay keep their order, last."""
        return sorted(targets, key=lambda dst: self.saved_rank.get((key, dst), len(self.saved_rank)))

    def waypoint_cell(self, key: str, wanted: Pos, find: Callable[[], Pos | None]) -> Pos | None:
        """The cell the last build gave a wanted waypoint, or find()'s if it had none or the net is being rerouted."""
        entry = (key, wanted)
        cell = self.saved_cells[entry] if entry in self.saved_cells and key not in self.reroute else find()
        self.cells[entry] = cell
        return cell

    def finished(self, key: str) -> None:
        self.order.pop(key, None)
        self.order[key] = None

    def save_paths(self) -> None:
        self._dump("paths.pkl", {**self.saved, **self.paths})
        self._dump("order.pkl", list(self.order))
        self._dump("cells.pkl", {**self.saved_cells, **self.cells})
