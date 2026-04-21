"""Debounced change queue — aggregates FS events into batches.

The idle window (default 30 s per spec §5.2) starts on the first add()
and resets on each subsequent add(). drain_if_ready() returns the
aggregated paths once the window has elapsed with no new additions.

Not thread-safe: ``add`` and ``drain_if_ready`` must be called from the
same thread. The CorpusWatcher callback fires on watchdog's dispatch
thread, so wiring it directly to ``add`` requires either (a) a lock
around both methods, or (b) a thread-safe relay (queue.Queue) that the
main thread drains into ``add``. The M5 pipeline uses (b).
"""

from __future__ import annotations

import time
from pathlib import Path


class DebouncedQueue:
    def __init__(self, window_seconds: float) -> None:
        if window_seconds <= 0:
            raise ValueError(
                f"window_seconds must be > 0; got {window_seconds}. "
                "A zero/negative window makes drain_if_ready() fire on every poll."
            )
        self._window = window_seconds
        self._pending: set[Path] = set()
        self._last_add: float | None = None

    def add(self, path: Path) -> None:
        # Resolve symlinks / collapse ".." so Path("./a") and Path("a") dedup
        # to the same entry in _pending.
        self._pending.add(path.resolve())
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
