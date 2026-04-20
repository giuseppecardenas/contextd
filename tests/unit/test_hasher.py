from pathlib import Path

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
