"""Work split across forked processes, which see this one's memory as it is when they start."""
from __future__ import annotations

import gc
import os
import pickle
import sys
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

_MIN_PER_WORKER = 64  # fewer items than this per process don't pay for a fork


def workers() -> int:
    """Processes for fork_map: MINECRAFT_V_WORKERS, or one per core."""
    return max(1, int(os.environ.get("MINECRAFT_V_WORKERS", os.cpu_count() or 1)))


def fork_map(fn: Callable[[T], R], items: Sequence[T], processes: int | None = None) -> list[R]:
    """[fn(item) for item in items], in order, the items split in runs across forked processes. Results come back
    pickled, so fn should be cheap to return from. Without fork, or for few items, it all runs here."""
    n = min(processes or workers(), len(items) // _MIN_PER_WORKER)
    if n <= 1 or not hasattr(os, "fork"):
        return [fn(item) for item in items]
    bounds = [len(items) * k // n for k in range(n + 1)]
    sys.stdout.flush()
    sys.stderr.flush()
    children: list[tuple[int, int]] = []
    try:
        for k in range(n):
            read, write = os.pipe()
            pid = os.fork()
            if pid == 0:  # pragma: no cover - the child's coverage isn't collected
                os.close(read)
                gc.disable()  # a collection would walk, and so copy, the parent's whole heap
                status = 0
                try:
                    data = pickle.dumps([fn(item) for item in items[bounds[k]:bounds[k + 1]]], protocol=5)
                except BaseException as e:  # noqa: BLE001 - goes back to the parent
                    status = 1
                    try:
                        data = pickle.dumps(e, protocol=5)
                    except Exception:  # noqa: BLE001
                        data = pickle.dumps(RuntimeError(repr(e)), protocol=5)
                with os.fdopen(write, "wb") as f:
                    f.write(bytes([status]))
                    f.write(data)
                os._exit(0)
            os.close(write)
            children.append((pid, read))
        results: list[R] = []
        error: BaseException | None = None
        while children:
            pid, read = children.pop(0)
            with os.fdopen(read, "rb") as f:
                data = f.read()
            os.waitpid(pid, 0)
            if not data:
                error = error or RuntimeError(f"worker {pid} died without a result")
            elif data[0]:
                error = error or pickle.loads(data[1:])
            elif error is None:
                results.extend(pickle.loads(data[1:]))
        if error is not None:
            raise error
        return results
    finally:
        for pid, read in children:  # only if something failed part way
            os.close(read)
            os.waitpid(pid, 0)
