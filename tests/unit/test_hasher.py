import json
import os
from pathlib import Path
from unittest.mock import patch

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


def test_stored_distinguishes_unknown_from_identical(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("x")
    h = FileHasher()
    assert h.stored(f) is None
    h.mark_seen(f)
    assert h.stored(f) == h.hash(f)


def test_mark_seen_accepts_precomputed_digest(tmp_path: Path) -> None:
    # Callers pass the digest observed before indexing. Re-reading at completion
    # time would record content newer than what was indexed, and the newer edit
    # would then be filtered out as unchanged and lost.
    f = tmp_path / "a.md"
    f.write_text("old")
    h = FileHasher()
    digest_before = h.hash(f)
    f.write_text("new during indexing")
    h.mark_seen(f, digest_before)
    assert h.stored(f) == digest_before
    assert h.is_changed(f)  # the newer content is still pending


def test_forget_drops_entry_so_restored_file_reindexes(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("x")
    h = FileHasher()
    h.mark_seen(f)
    h.forget(f)
    assert h.stored(f) is None
    assert h.is_changed(f)


def test_forget_is_noop_for_unknown_path(tmp_path: Path) -> None:
    FileHasher().forget(tmp_path / "never-seen.md")


def test_persist_replaces_state_file_atomically(tmp_path: Path) -> None:
    # A truncated write leaves JSON that _load_state rejects, which stops the
    # daemon from starting at all.
    state = tmp_path / "state" / "s-index-state.json"
    f = tmp_path / "a.md"
    f.write_text("x")
    h = FileHasher(state_path=state)
    h.mark_seen(f)
    assert not state.with_suffix(".tmp").exists()
    assert FileHasher(state_path=state).stored(f) == h.hash(f)


def test_forget_many_persists_once(tmp_path: Path) -> None:
    # Reconciliation can forget a whole corpus; per-path persists would rewrite
    # the entire state file once per entry.
    state = tmp_path / "state" / "s-index-state.json"
    files = []
    h = FileHasher(state_path=state)
    for i in range(3):
        f = tmp_path / f"{i}.md"
        f.write_text(str(i))
        h.mark_seen(f)
        files.append(f)

    unknown = tmp_path / "unknown.md"
    writes = 0
    real_replace = os.replace

    def counting_replace(src: object, dst: object) -> None:
        nonlocal writes
        writes += 1
        real_replace(src, dst)  # type: ignore[arg-type]

    with patch("contextd.indexer.hasher.os.replace", side_effect=counting_replace):
        removed = h.forget_many([*files, unknown])

    assert removed == 3
    assert writes == 1
    assert all(h.stored(f) is None for f in files)


def test_forget_many_skips_persist_when_nothing_removed(tmp_path: Path) -> None:
    h = FileHasher(state_path=tmp_path / "state" / "s-index-state.json")
    assert h.forget_many([tmp_path / "never-seen.md"]) == 0
