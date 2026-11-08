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
# What a child dirties of the parent's memory, as a share of the parent's footprint: x512 measured 15-19% for each of
# fork_map's workers, and a signal check that holds its results grew to 70%. Forking is skipped when the children would
# not fit in what the machine has free, as the work runs here instead (and gives the same answer, only one at a time).
_WORKER_SHARE = 0.2
_FORKED_SHARE = 0.75
_USE_OF_FREE = 0.7  # of the memory free and inactive, which children may take


def parent_bytes() -> int:
    """The larger of this process's physical footprint (macOS counts compressed and swapped pages, which RSS doesn't) and
    its resident size (in a fork, the footprint is only what it has written so far; the resident size has what it shares)."""
    import re
    import subprocess
    best = 0
    try:
        out = subprocess.run(["footprint", "-p", str(os.getpid())], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*(\w+)", out)
        if m:
            best = int(float(m.group(1)) * {"B": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}[m.group(2)])
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, timeout=30).stdout
        best = max(best, int(rss.strip() or 0) * 1024)
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        pass
    if not best:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        best = peak if sys.platform == "darwin" else peak * 1024
    return best


def free_bytes() -> int | None:
    """Memory free or cheap to take back (free, inactive, speculative), or None where that can't be read."""
    import re
    import subprocess
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                m = re.search(r"MemAvailable:\s+(\d+) kB", f.read())
            return int(m.group(1)) * 1024 if m else None
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=30).stdout
        size = int(re.search(r"page size of (\d+) bytes", out).group(1))
        pages = sum(int(m.group(1)) for key in ("Pages free", "Pages inactive", "Pages speculative")
                    for m in [re.search(key + r":\s+(\d+)", out)] if m)
        return pages * size
    except (OSError, subprocess.SubprocessError, AttributeError, ValueError):
        return None


def children_that_fit(wanted: int, share: float) -> int:
    """How many of `wanted` children of this process, each dirtying `share` of its size, the free memory takes."""
    free = free_bytes()
    if wanted < 1 or free is None:
        return wanted
    each = max(parent_bytes() * share, 1)
    return max(0, min(wanted, int(free * _USE_OF_FREE / each)))


def workers() -> int:
    """Processes for fork_map: MINECRAFT_V_WORKERS, or one per core."""
    return max(1, int(os.environ.get("MINECRAFT_V_WORKERS", os.cpu_count() or 1)))


def fork_map(fn: Callable[[T], R], items: Sequence[T], processes: int | None = None) -> list[R]:
    """[fn(item) for item in items], in order, the items split in runs across forked processes. Results come back
    pickled, so fn should be cheap to return from. Without fork, or for few items, it all runs here."""
    runs = fork_batches(lambda run: [fn(item) for item in run], items, processes)
    return [r for run in runs for r in run]


def fork_batches(fn: Callable[[Sequence[T]], R], items: Sequence[T], processes: int | None = None) -> list[R]:
    """[fn(run) for each run of items], the runs consecutive and as even as can be, one forked process each; one run,
    here, for few items or without fork. fn gets a whole run, so what it returns can be one compact value (arrays) rather
    than a pickled object per item."""
    n = min(processes or workers(), len(items) // _MIN_PER_WORKER)
    if n > 1 and hasattr(os, "fork"):
        n = children_that_fit(n, _WORKER_SHARE)
    if n <= 1 or not hasattr(os, "fork"):
        return [fn(items)]
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
                    data = pickle.dumps(fn(items[bounds[k]:bounds[k + 1]]), protocol=5)
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
                results.append(pickle.loads(data[1:]))
        if error is not None:
            raise error
        return results
    finally:
        for pid, read in children:  # only if something failed part way
            os.close(read)
            os.waitpid(pid, 0)


class Forked:
    """fn() run in a forked process from now until result(), which returns what it returned, or raises what it raised.
    What it prints is held and printed by result(), so it doesn't land in the middle of this process's lines. Without
    fork (or with one worker) fn runs right away, here, and result() hands back what it returned or raises what it
    raised: what fn sees is the state at construction either way, so the outcome doesn't depend on worker count."""

    def __init__(self, fn: Callable[[], R]):
        self._done = False
        self._value: R | None = None
        self._error: BaseException | None = None
        self._pid = 0
        self._read = -1
        if not hasattr(os, "fork") or workers() <= 1 or not children_that_fit(1, _FORKED_SHARE):
            self._done = True
            try:
                self._value = fn()
            except BaseException as e:  # noqa: BLE001 - raised again from result()
                self._error = e
            return
        sys.stdout.flush()
        sys.stderr.flush()
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - the child's coverage isn't collected
            os.close(read)
            gc.disable()
            import io
            out = io.StringIO()
            sys.stdout = sys.stderr = out
            status = 0
            try:
                result = fn()
            except BaseException as e:  # noqa: BLE001 - goes back to the parent
                status, result = 1, e
            try:
                data = pickle.dumps((result, out.getvalue()), protocol=5)
            except Exception as e:  # noqa: BLE001
                status, data = 1, pickle.dumps((RuntimeError(repr(e)), out.getvalue()), protocol=5)
            with os.fdopen(write, "wb") as f:
                f.write(bytes([status]))
                f.write(data)
            os._exit(0)
        os.close(write)
        self._pid, self._read = pid, read

    def result(self) -> R:
        if self._done:
            if self._error is not None:
                raise self._error
            return self._value  # type: ignore[return-value]
        with os.fdopen(self._read, "rb") as f:
            data = f.read()
        os.waitpid(self._pid, 0)
        if not data:
            raise RuntimeError(f"forked process {self._pid} died without a result")
        result, printed = pickle.loads(data[1:])
        sys.stdout.write(printed)
        sys.stdout.flush()
        if data[0]:
            raise result
        return result
