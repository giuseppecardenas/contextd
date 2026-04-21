import json
from pathlib import Path

import pytest

from contextd.indexer.hasher import FileHasher


def test_hash_stable_for_unchanged_file(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("hello")
    hasher = FileHasher()
    assert hasher.hash(f) == hasher.hash(f)


def test_hash_differs_on_content_change(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("hello")
    hasher = FileHasher()
    h1 = hasher.hash(f)
    f.write_text("world")
    h2 = hasher.hash(f)
    assert h1 != h2


def test_is_changed_uses_persistent_state(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("hello")
    hasher = FileHasher(state_path=tmp_path / "index-state.json")
    assert hasher.is_changed(f) is True  # new file
    hasher.mark_seen(f)
    assert hasher.is_changed(f) is False  # unchanged
    f.write_text("world")
    assert hasher.is_changed(f) is True  # content change


def test_load_state_rejects_non_dict_json(tmp_path: Path) -> None:
    state = tmp_path / "index-state.json"
    state.write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(ValueError, match="must be a JSON object"):
        FileHasher(state_path=state)


def test_load_state_rejects_non_string_values(tmp_path: Path) -> None:
    state = tmp_path / "index-state.json"
    state.write_text(json.dumps({"path/to/file": 42}))
    with pytest.raises(ValueError, match="must be str→str"):
        FileHasher(state_path=state)
