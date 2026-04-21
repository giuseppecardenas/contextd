from pathlib import Path

from contextd.indexer.git_lock import is_git_busy


def test_false_when_no_git_dir(tmp_path: Path) -> None:
    assert is_git_busy(tmp_path) is False


def test_true_when_index_lock_exists(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "index.lock").touch()
    assert is_git_busy(tmp_path) is True


def test_true_when_head_lock_exists(tmp_path: Path) -> None:
    """Covers the OR-branch that test_true_when_index_lock_exists misses —
    a regression replacing `or` with `and` would still pass the other test."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD.lock").touch()
    assert is_git_busy(tmp_path) is True


def test_false_when_git_dir_exists_without_locks(tmp_path: Path) -> None:
    """Happy-path negative: a fully-present .git dir with neither lock is
    not busy. Bug-bait for a refactor that returns True as a default."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert is_git_busy(tmp_path) is False


def test_resolves_gitfile_to_real_gitdir(tmp_path: Path) -> None:
    """Worktrees and submodules use a ``.git`` regular file containing
    ``gitdir: <path>`` rather than a directory. is_git_busy must probe the
    referenced path, not the gitfile itself."""
    # Real gitdir lives in a sibling path (mirrors `git worktree add` layout).
    real_gitdir = tmp_path / "real-gitdir"
    real_gitdir.mkdir()
    (real_gitdir / "index.lock").touch()

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    # .git is a *file*, not a directory.
    (worktree / ".git").write_text(f"gitdir: {real_gitdir}\n")

    assert is_git_busy(worktree) is True


def test_resolves_relative_gitfile_path(tmp_path: Path) -> None:
    """git init --separate-git-dir writes a relative path in the gitfile."""
    real_gitdir = tmp_path / "shared-gitdir"
    real_gitdir.mkdir()
    (real_gitdir / "HEAD.lock").touch()

    worktree = tmp_path / "sub"
    worktree.mkdir()
    # Relative path — must be resolved relative to the gitfile's directory.
    (worktree / ".git").write_text("gitdir: ../shared-gitdir\n")

    assert is_git_busy(worktree) is True


def test_malformed_gitfile_returns_false(tmp_path: Path) -> None:
    """A .git file without a parseable 'gitdir:' line is not a busy repo."""
    (tmp_path / ".git").write_text("not a gitfile\n")
    assert is_git_busy(tmp_path) is False
