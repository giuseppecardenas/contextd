import time
from pathlib import Path

import pytest

from contextd.indexer.watcher import CorpusWatcher


def test_watcher_fires_on_file_write(tmp_path: Path) -> None:
    changes: list[Path] = []
    w = CorpusWatcher(tmp_path, lambda p: changes.append(p))
    w.start()
    try:
        time.sleep(0.1)  # let observer attach
        (tmp_path / "a.md").write_text("hello", encoding="utf-8")
        # Poll briefly for the event.
        for _ in range(20):
            if changes:
                break
            time.sleep(0.05)
    finally:
        w.stop()
    assert any(p.name == "a.md" for p in changes)


def _wait_for(changes: list[Path], name: str, tries: int = 40) -> bool:
    for _ in range(tries):
        if any(p.name == name for p in changes):
            return True
        time.sleep(0.05)
    return False


def test_watcher_fires_on_intra_directory_rename(tmp_path: Path) -> None:
    """An atomic save (write temp beside target, rename over it) must be seen.

    watchdog reports such a save as a single FileMovedEvent naming the
    destination; the accompanying DirModifiedEvent is a directory event and is
    filtered out. Forwarding only created/modified therefore dropped every
    atomically-saved file silently, which is how three files went unindexed.
    """
    changes: list[Path] = []
    w = CorpusWatcher(tmp_path, lambda p: changes.append(p))
    w.start()
    try:
        time.sleep(0.1)  # let observer attach
        tmp = tmp_path / ".target.md.tmp"
        tmp.write_text("hello", encoding="utf-8")
        tmp.rename(tmp_path / "target.md")
        assert _wait_for(changes, "target.md")
    finally:
        w.stop()


def test_watcher_fires_on_delete(tmp_path: Path) -> None:
    """Deletions must be forwarded so the indexer can reap the node."""
    changes: list[Path] = []
    victim = tmp_path / "victim.md"
    victim.write_text("hello", encoding="utf-8")
    w = CorpusWatcher(tmp_path, lambda p: changes.append(p))
    w.start()
    try:
        time.sleep(0.1)
        victim.unlink()
        assert _wait_for(changes, "victim.md")
    finally:
        w.stop()


def test_is_alive_tracks_observer_lifecycle(tmp_path: Path) -> None:
    w = CorpusWatcher(tmp_path, lambda _p: None)
    assert not w.is_alive()  # never started
    w.start()
    try:
        assert w.is_alive()
    finally:
        w.stop()
    assert not w.is_alive()  # stopped


def test_double_start_raises(tmp_path: Path) -> None:
    w = CorpusWatcher(tmp_path, lambda _p: None)
    w.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            w.start()
    finally:
        w.stop()


def test_stop_is_idempotent_without_start(tmp_path: Path) -> None:
    # Calling stop() before start() must not error — consistent with the
    # `None`-guard pattern used for CheckpointStore.clear and elsewhere.
    w = CorpusWatcher(tmp_path, lambda _p: None)
    w.stop()
    w.stop()


def test_start_after_stop_resumes(tmp_path: Path) -> None:
    # Verifies the double-start guard releases after stop() — callers that
    # rotate watchers (e.g. `contextd down` → `contextd up`) aren't blocked.
    w = CorpusWatcher(tmp_path, lambda _p: None)
    w.start()
    w.stop()
    w.start()  # must not raise
    w.stop()
