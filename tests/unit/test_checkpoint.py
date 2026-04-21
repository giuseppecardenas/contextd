import json
from pathlib import Path

import pytest

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


def test_save_is_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.save("c", Checkpoint(phase="embed", last_committed_batch=1, last_committed_file=None))
    # After a clean save the .tmp sidecar must not linger.
    assert not (tmp_path / "c.json.tmp").exists()
    assert (tmp_path / "c.json").exists()


def test_save_overwrites_atomically(tmp_path: Path) -> None:
    # Second save must fully replace the first payload, not leave a partial.
    store = CheckpointStore(tmp_path)
    store.save("c", Checkpoint(phase="embed", last_committed_batch=1, last_committed_file="a"))
    store.save("c", Checkpoint(phase="relate", last_committed_batch=7, last_committed_file="b"))
    loaded = store.load("c")
    assert loaded is not None
    assert loaded.phase == "relate"
    assert loaded.last_committed_batch == 7
    assert loaded.last_committed_file == "b"


@pytest.mark.parametrize("bad", ["../escape", "a/b", "a\\b", ".", "..", ".hidden", ""])
def test_invalid_corpus_names_rejected(tmp_path: Path, bad: str) -> None:
    store = CheckpointStore(tmp_path)
    cp = Checkpoint(phase="embed", last_committed_batch=0, last_committed_file=None)
    with pytest.raises(ValueError):
        store.save(bad, cp)
    with pytest.raises(ValueError):
        store.load(bad)
    with pytest.raises(ValueError):
        store.clear(bad)


def test_load_rejects_non_dict_json(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    (tmp_path / "c.json").write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(ValueError, match="must be a JSON object"):
        store.load("c")


def test_load_rejects_wrong_field_types(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    # Wrong type on last_committed_batch.
    (tmp_path / "c.json").write_text(
        json.dumps({"phase": "embed", "last_committed_batch": "three", "last_committed_file": None})
    )
    with pytest.raises(ValueError, match="last_committed_batch"):
        store.load("c")


def test_load_rejects_bool_as_int(tmp_path: Path) -> None:
    # bool is a subclass of int in Python; JSON true/false must not satisfy
    # the last_committed_batch integer contract.
    store = CheckpointStore(tmp_path)
    (tmp_path / "c.json").write_text(
        json.dumps({"phase": "embed", "last_committed_batch": True, "last_committed_file": None})
    )
    with pytest.raises(ValueError, match="last_committed_batch"):
        store.load("c")
