"""Detect whether a corpus root's .git has an active lock.

Spec §5.3: while ``git checkout`` or ``git rebase`` is rewriting
hundreds of files, indexing should pause — otherwise a single
rebase triggers hundreds of Gemini calls.
"""

from __future__ import annotations

from pathlib import Path


def is_git_busy(corpus_root: Path) -> bool:
    git_dir = corpus_root / ".git"
    if not git_dir.is_dir():
        return False
    return (git_dir / "index.lock").exists() or (git_dir / "HEAD.lock").exists()
