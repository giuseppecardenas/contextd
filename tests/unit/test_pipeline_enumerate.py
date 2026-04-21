"""Unit tests for enumerate_corpus_files — the dotfile / symlink guards."""

from __future__ import annotations

import os
from pathlib import Path

from contextd.corpus_config import CorpusConfig
from contextd.indexer.pipeline import enumerate_corpus_files


def _cfg(root: Path, include: list[str] | None = None) -> CorpusConfig:
    return CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "t",
                "root": str(root),
                "include": include or ["**/*.md"],
            }
        }
    )


def test_excludes_dotgit_contents_by_default(tmp_path: Path) -> None:
    """A corpus root that contains .git/ must not have its object files
    walked by the indexer. Prevents accidental 10k-file fan-out when a
    user points at a clone root with no explicit exclude config."""
    (tmp_path / "real.md").write_text("body")
    dot_git = tmp_path / ".git" / "objects"
    dot_git.mkdir(parents=True)
    (dot_git / "inside.md").write_text("should-not-be-enumerated")

    files = enumerate_corpus_files(_cfg(tmp_path))
    names = {p.name for p in files}
    assert "real.md" in names
    assert "inside.md" not in names


def test_excludes_venv_pycache_node_modules(tmp_path: Path) -> None:
    """Other conventional exclude dirs — any part of the path matching
    the default exclude set drops the file from enumeration."""
    (tmp_path / "kept.md").write_text("x")
    for bad_dir in (".venv", "__pycache__", "node_modules"):
        sub = tmp_path / bad_dir
        sub.mkdir()
        (sub / "dropped.md").write_text("x")

    files = enumerate_corpus_files(_cfg(tmp_path))
    paths = {str(p) for p in files}
    assert any(p.endswith("/kept.md") for p in paths)
    assert not any("/.venv/" in p or "/__pycache__/" in p or "/node_modules/" in p for p in paths)


def test_skips_symlinks(tmp_path: Path) -> None:
    """Symlinked files are skipped to avoid walking into cycle-forming
    targets (e.g. a symlink back into the corpus root)."""
    (tmp_path / "real.md").write_text("x")
    link_target = tmp_path / "target.md"
    link_target.write_text("y")
    # Create a symlink inside the corpus root.
    link = tmp_path / "link.md"
    os.symlink(link_target, link)

    files = enumerate_corpus_files(_cfg(tmp_path))
    names = {p.name for p in files}
    assert "real.md" in names
    assert "target.md" in names
    # The symlink is skipped even though it would glob to *.md.
    assert "link.md" not in names
