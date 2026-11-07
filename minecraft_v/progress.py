"""Single-line progress on stderr: a spinner, or a bar when the total is known.

Drawn only when stderr is a TTY, so logs stay clean. A background thread redraws the line, so the
spinner and clock keep moving while the main thread works. Compiled code holding the GIL calls
pulse() to let it run. Phases nest; the innermost one shows.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time

_REDRAW_INTERVAL = 0.05
_TICK_SECONDS = 0.1
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_stack: list[ProgressBar] = []
_lock = threading.RLock()
_ticker: threading.Thread | None = None
_forked = False  # worker processes stay off the line; the parent draws it
_debug = False
_phase_times = os.environ.get("MINECRAFT_V_PHASE_TIMES") == "1"  # log each phase's seconds when it ends


def _load_counters():
    """This process's instructions retired and cycles, all threads (macOS only): instructions barely move with
    machine load, unlike wall or CPU time, and instructions per cycle shows whether a phase waits on memory.
    Forked workers count in their own process, not here."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    except OSError:
        return None
    info = (ctypes.c_uint64 * 64)()  # rusage_info_v4: 16-byte uuid, then uint64 fields; ri_instructions, ri_cycles are the 30th, 31st

    def counters() -> tuple[int, int]:
        if libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
            return 0, 0
        return info[2 + 29], info[2 + 30]
    return counters


_counters = _load_counters() if _phase_times else None


def _cost() -> tuple[float, float, int, int]:
    times = os.times()
    return (time.process_time(), times.children_user + times.children_system,
            *(_counters() if _counters else (0, 0)))


def _tick() -> None:
    while True:
        time.sleep(_TICK_SECONDS)
        with _lock:
            if _stack:
                _stack[-1]._draw()


def _after_fork_in_child() -> None:
    global _lock, _forked
    _lock = threading.RLock()  # the ticker that could hold the old one doesn't exist here
    _forked = True
    _stack.clear()


# Hold the lock across fork, so a child never inherits a half-written line's buffer state.
os.register_at_fork(before=lambda: _lock.acquire(), after_in_parent=lambda: _lock.release(),
                    after_in_child=_after_fork_in_child)


class ProgressBar:
    def __init__(self, label: str, total: int | None = None):
        self.label = label
        self.total = total
        self.n = 0
        self.info = ""
        self.detail = ""
        self.live = sys.stderr.isatty() and not _forked
        self._t0 = time.monotonic()
        self._cost0 = _cost() if _phase_times else None
        self._last_draw = 0.0

    def __enter__(self) -> ProgressBar:
        global _ticker
        with _lock:
            _stack.append(self)
            if self.live and _ticker is None:
                _ticker = threading.Thread(target=_tick, daemon=True)
                _ticker.start()
            self._draw()
        return self

    def __exit__(self, *_) -> None:
        with _lock:
            _stack.remove(self)
            self._clear()
            if self._cost0 is not None and not _forked:
                cpu, children, instructions, cycles = (b - a for a, b in zip(self._cost0, _cost()))
                print(f"[phase] {self.label}: {time.monotonic() - self._t0:.1f}s (cpu {cpu:.1f}s"
                      + (f", reaped workers {children:.1f}s" if children >= 0.05 else "")
                      + (f", {instructions / 1e9:.1f}G instr" if _counters else "")
                      + (f", IPC {instructions / cycles:.1f}" if cycles > 0 else "") + ")", flush=True)
            if _stack:
                _stack[-1]._draw()

    def update(self, n: int = 1, info: str = "") -> None:
        self.n += n
        self.info = info
        self.detail = ""
        if time.monotonic() - self._last_draw >= _REDRAW_INTERVAL or (self.total and self.n >= self.total):
            self._draw()

    def _draw(self) -> None:
        if not self.live:
            return
        with _lock:
            now = time.monotonic()
            self._last_draw = now
            cols = shutil.get_terminal_size().columns
            elapsed = now - self._t0
            frame = _FRAMES[int(now / _TICK_SECONDS) % len(_FRAMES)]
            line = f"{frame} {self.label} "
            if self.total is not None:
                frac = min(self.n / self.total, 1.0) if self.total else 1.0
                width = 30
                fill = int(width * frac)
                line += f"[{'#' * fill}{'.' * (width - fill)}] {self.n}/{self.total} "
            line += f"{elapsed:.0f}s {self.info}"
            if self.detail:
                line += f" {self.detail}"
            sys.stderr.write("\r" + line[:cols - 1] + "\x1b[K")
            sys.stderr.flush()

    def _clear(self) -> None:
        if self.live:
            sys.stderr.write("\r\x1b[K")
            sys.stderr.flush()


def phase(label: str) -> ProgressBar:
    """A spinner for work with no known total; status() says what it's up to."""
    return ProgressBar(label)


def status(info: str) -> None:
    """Say what the current phase is doing, keeping its count."""
    if _stack:
        bar = _stack[-1]
        bar.info = info
        bar.detail = ""


def pulse(nodes: int) -> None:
    """From inside an A* search: show how far it has got, and let the ticker run."""
    if _stack:
        bar = _stack[-1]
        bar.detail = f"{nodes // 1000}k nodes"
        if time.monotonic() - bar._last_draw >= _TICK_SECONDS:
            bar._draw()


def set_debug(on: bool) -> None:
    global _debug
    _debug = on


def is_debug() -> bool:
    return _debug


def debug(msg: str) -> None:
    """log(), only with --debug."""
    if _debug:
        log(msg)


def log(msg: str) -> None:
    """Print a line without mangling the active progress line."""
    with _lock:
        if _stack:
            _stack[-1]._clear()
        print(msg, flush=True)
        if _stack:
            _stack[-1]._draw()
