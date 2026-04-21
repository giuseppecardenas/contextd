"""Debounced change queue — aggregates FS events into batches.

The idle window (default 30 s per spec §5.2) starts on the first add()
and resets on each subsequent add(). drain_if_ready() returns the
aggregated paths once the window has elapsed with no new additions.
"""

from __future__ import annotations

import time
from pathlib import Path


class DebouncedQueue:
    def __init__(self, window_seconds: float) -> None:
        self._window = window_seconds
        self._pending: set[Path] = set()
        self._last_add: float | None = None

    def add(self, path: Path) -> None:
        self._pending.add(path)
        self._last_add = time.monotonic()

    def drain_if_ready(self) -> list[Path]:
        if not self._pending or self._last_add is None:
            return []
        if time.monotonic() - self._last_add < self._window:
            return []
        out = sorted(self._pending)
        self._pending.clear()
        self._last_add = None
        return out
