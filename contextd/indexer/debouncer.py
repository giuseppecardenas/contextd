"""Debounced change queue — aggregates FS events into batches.

The idle window (default 30 s per spec §5.2) starts on the first add()
and resets on each subsequent add(). drain_if_ready() returns the
aggregated paths once the window has elapsed with no new additions.

Because each add() resets the window, a path receiving events more often than
the window is long would never drain: an editor autosaving every few seconds, a
generator rewriting a file in a loop, or a busy corpus can starve the batch
indefinitely while the index goes stale with no error. ``max_age_seconds``
bounds that by forcing a drain once the oldest pending event reaches the cap,
regardless of ongoing activity.

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
    def __init__(self, window_seconds: float, max_age_seconds: float | None = None) -> None:
        """Aggregate paths, draining after an idle window or a hard age cap.

        ``max_age_seconds`` defaults to ten idle windows and must be at least as
        long as one, since a cap shorter than the window would pre-empt the
        debounce on every batch and defeat the aggregation entirely. Pass a
        non-positive value to disable the cap and restore pure idle-window
        behaviour.
        """
        if window_seconds <= 0:
            raise ValueError(
                f"window_seconds must be > 0; got {window_seconds}. "
                "A zero/negative window makes drain_if_ready() fire on every poll."
            )
        resolved_max_age = window_seconds * 10 if max_age_seconds is None else max_age_seconds
        if resolved_max_age > 0 and resolved_max_age < window_seconds:
            raise ValueError(
                f"max_age_seconds ({resolved_max_age}) must be >= window_seconds "
                f"({window_seconds}); a shorter cap pre-empts the debounce window "
                "on every batch."
            )
        self._window = window_seconds
        self._max_age = resolved_max_age
        self._pending: set[Path] = set()
        self._last_add: float | None = None
        self._first_add: float | None = None

    def add(self, path: Path) -> None:
        # Resolve symlinks / collapse ".." so Path("./a") and Path("a") dedup
        # to the same entry in _pending.
        self._pending.add(path.resolve())
        now = time.monotonic()
        self._last_add = now
        if self._first_add is None:
            self._first_add = now

    def drain_if_ready(self) -> list[Path]:
        if not self._pending or self._last_add is None:
            return []
        now = time.monotonic()
        idle = now - self._last_add >= self._window
        aged_out = (
            self._max_age > 0
            and self._first_add is not None
            and now - self._first_add >= self._max_age
        )
        if not (idle or aged_out):
            return []
        out = sorted(self._pending)
        self._pending.clear()
        self._last_add = None
        self._first_add = None
        return out
