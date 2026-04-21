import time
from pathlib import Path

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
