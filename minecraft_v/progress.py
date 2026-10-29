"""Minimal single-line progress bar on stderr. Drawn only when stderr is a TTY, so logs stay clean."""
from __future__ import annotations

import shutil
import sys
import time

_REDRAW_INTERVAL = 0.05
_active: ProgressBar | None = None


class ProgressBar:
    def __init__(self, label: str, total: int):
        self.label = label
        self.total = total
        self.n = 0
        self.info = ""
        self.live = sys.stderr.isatty()
        self._t0 = time.monotonic()
        self._last_draw = 0.0

    def __enter__(self) -> ProgressBar:
        global _active
        _active = self
        self._draw()
        return self

    def __exit__(self, *_) -> None:
        global _active
        _active = None
        self._clear()

    def update(self, n: int = 1, info: str = "") -> None:
        self.n += n
        self.info = info
        if time.monotonic() - self._last_draw >= _REDRAW_INTERVAL or self.n >= self.total:
            self._draw()

    def _draw(self) -> None:
        if not self.live:
            return
        self._last_draw = time.monotonic()
        cols = shutil.get_terminal_size().columns
        frac = self.n / self.total if self.total else 1.0
        width = 30
        fill = int(width * frac)
        line = (f"{self.label} [{'#' * fill}{'.' * (width - fill)}] {self.n}/{self.total} "
                f"{self._last_draw - self._t0:.0f}s {self.info}")
        sys.stderr.write("\r" + line[:cols - 1] + "\x1b[K")
        sys.stderr.flush()

    def _clear(self) -> None:
        if self.live:
            sys.stderr.write("\r\x1b[K")
            sys.stderr.flush()


def log(msg: str) -> None:
    """Print a line without mangling the active progress bar."""
    if _active is not None:
        _active._clear()
    print(msg, flush=True)
    if _active is not None:
        _active._draw()
