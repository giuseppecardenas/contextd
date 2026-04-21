import time
from pathlib import Path

import pytest

from contextd.indexer.debouncer import DebouncedQueue


def test_collects_within_window(tmp_path: Path) -> None:
    q = DebouncedQueue(window_seconds=0.1)
    a = tmp_path / "a"
    b = tmp_path / "b"
    q.add(a)
    q.add(b)
    q.add(a)  # duplicate suppressed
    time.sleep(0.15)
    batch = q.drain_if_ready()
    assert set(batch) == {a.resolve(), b.resolve()}


def test_no_drain_before_window(tmp_path: Path) -> None:
    q = DebouncedQueue(window_seconds=1.0)
    q.add(tmp_path / "a")
    batch = q.drain_if_ready()
    assert batch == []


def test_drain_resets_window(tmp_path: Path) -> None:
    q = DebouncedQueue(window_seconds=0.05)
    q.add(tmp_path / "a")
    time.sleep(0.1)
    q.drain_if_ready()
    q.add(tmp_path / "b")
    # Immediately after drain, window restarts.
    assert q.drain_if_ready() == []


def test_init_rejects_non_positive_window() -> None:
    # Zero or negative window_seconds causes drain_if_ready() to fire
    # on every poll — defeats the debounce contract.
    with pytest.raises(ValueError, match="must be > 0"):
        DebouncedQueue(window_seconds=0)
    with pytest.raises(ValueError, match="must be > 0"):
        DebouncedQueue(window_seconds=-0.5)


def test_add_normalises_path(tmp_path: Path) -> None:
    # Path("./a") and Path("a") hash differently before .resolve(); after
    # the normalisation added for M5.1 they dedup in the pending set.
    q = DebouncedQueue(window_seconds=0.05)
    target = tmp_path / "a"
    target.touch()
    q.add(target)
    q.add(Path(str(target)))  # same absolute path, same object
    time.sleep(0.1)
    batch = q.drain_if_ready()
    assert len(batch) == 1
