"""Detect whether a corpus root's .git has an active lock.

Spec §5.3: while ``git checkout`` or ``git rebase`` is rewriting
hundreds of files, indexing should pause — otherwise a single
rebase triggers hundreds of Gemini calls.

Handles three ``.git`` shapes:

- Directory (standard clone): probe ``.git/index.lock`` and ``.git/HEAD.lock``.
- Gitfile (worktree / submodule / ``git init --separate-git-dir``): the
  ``.git`` entry is a regular file containing ``gitdir: <path>``; we resolve
  that path and probe the same locks there.
- Absent: not a git repo — never busy.
"""

from __future__ import annotations

from pathlib import Path


def _resolve_gitdir(git_entry: Path) -> Path | None:
    """Return the real gitdir when ``.git`` is either a directory or a gitfile.

    Returns None if ``git_entry`` is neither a directory nor a parseable gitfile.
    """
    if git_entry.is_dir():
        return git_entry
    if not git_entry.is_file():
        return None
    try:
        content = git_entry.read_text()
    except OSError:
        return None
    prefix = "gitdir:"
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            raw_path = stripped[len(prefix) :].strip()
            if not raw_path:
                return None
            resolved = Path(raw_path)
            if not resolved.is_absolute():
                resolved = (git_entry.parent / resolved).resolve()
            return resolved if resolved.is_dir() else None
    return None


def is_git_busy(corpus_root: Path) -> bool:
    git_dir = _resolve_gitdir(corpus_root / ".git")
    if git_dir is None:
        return False
    return (git_dir / "index.lock").exists() or (git_dir / "HEAD.lock").exists()
