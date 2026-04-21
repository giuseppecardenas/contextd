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
        (tmp_path / "a.md").write_text("hello")
        # Poll briefly for the event.
        for _ in range(20):
            if changes:
                break
            time.sleep(0.05)
    finally:
        w.stop()
    assert any(p.name == "a.md" for p in changes)


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
