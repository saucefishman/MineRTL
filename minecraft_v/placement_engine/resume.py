"""Saved placement and routed paths, so a test build can skip the work it isn't testing.

Paths are laid again in the same order without searching, which rebuilds the same world for the nets that are searched.
Nothing here notices a change to the placer or the netlist: use a new directory after one.
"""
from __future__ import annotations

import copyreg
import os
import pickle
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
        self.paths[(key, dst)] = path

    def forget(self, key: str, dst: Pos) -> None:
        self.paths.pop((key, dst), None)

    def save_paths(self) -> None:
        self._dump("paths.pkl", {**self.saved, **self.paths})
