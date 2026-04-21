from pathlib import Path

from contextd.indexer.git_lock import is_git_busy


def test_false_when_no_git_dir(tmp_path: Path) -> None:
    assert is_git_busy(tmp_path) is False


def test_true_when_index_lock_exists(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "index.lock").touch()
    assert is_git_busy(tmp_path) is True
