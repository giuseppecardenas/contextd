import time
from pathlib import Path

from contextd.indexer.debouncer import DebouncedQueue


def test_collects_within_window() -> None:
    q = DebouncedQueue(window_seconds=0.1)
    q.add(Path("a"))
    q.add(Path("b"))
    q.add(Path("a"))  # duplicate suppressed
    time.sleep(0.15)
    batch = q.drain_if_ready()
    assert set(batch) == {Path("a"), Path("b")}


def test_no_drain_before_window() -> None:
    q = DebouncedQueue(window_seconds=1.0)
    q.add(Path("a"))
    batch = q.drain_if_ready()
    assert batch == []


def test_drain_resets_window() -> None:
    q = DebouncedQueue(window_seconds=0.05)
    q.add(Path("a"))
    time.sleep(0.1)
    q.drain_if_ready()
    q.add(Path("b"))
    # Immediately after drain, window restarts.
    assert q.drain_if_ready() == []
