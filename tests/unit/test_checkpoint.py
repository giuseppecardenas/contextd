from pathlib import Path

from contextd.indexer.checkpoint import Checkpoint, CheckpointStore


def test_save_and_load(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    cp = Checkpoint(phase="embed", last_committed_batch=3, last_committed_file="f.md")
    store.save("corpus1", cp)
    loaded = store.load("corpus1")
    assert loaded == cp


def test_load_returns_none_when_absent(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    assert store.load("nope") is None


def test_clear(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.save("c", Checkpoint(phase="embed", last_committed_batch=0, last_committed_file=None))
    store.clear("c")
    assert store.load("c") is None
