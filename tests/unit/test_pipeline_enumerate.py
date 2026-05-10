"""Unit tests for enumerate_corpus_files and _partition_markdown."""

from __future__ import annotations

import os
from pathlib import Path

from contextd.corpus_config import CorpusConfig
from contextd.indexer.pipeline import _partition_markdown, enumerate_corpus_files


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
    # Use ``Path.parts`` for separator-agnostic membership tests; ``str(p)``
    # uses backslashes on Windows so substring checks on "/.venv/" miss.
    names = {p.name for p in files}
    assert "kept.md" in names
    assert not any({".venv", "__pycache__", "node_modules"} & set(p.parts) for p in files)


def test_skips_symlinks(tmp_path: Path) -> None:
    """Symlinked files are skipped to avoid walking into cycle-forming
    targets (e.g. a symlink back into the corpus root)."""
    import pytest

    (tmp_path / "real.md").write_text("x")
    link_target = tmp_path / "target.md"
    link_target.write_text("y")
    link = tmp_path / "link.md"
    # On Windows, os.symlink requires SeCreateSymbolicLinkPrivilege (only
    # held by admins or accounts with Developer Mode enabled); skip the
    # test rather than fail when the privilege is absent.
    try:
        os.symlink(link_target, link)
    except OSError as exc:
        pytest.skip(f"cannot create symlinks in this environment: {exc}")

    files = enumerate_corpus_files(_cfg(tmp_path))
    names = {p.name for p in files}
    assert "real.md" in names
    assert "target.md" in names
    # The symlink is skipped even though it would glob to *.md.
    assert "link.md" not in names


# ---------------------------------------------------------------------------
# _partition_markdown
# ---------------------------------------------------------------------------


def test_partition_markdown_splits_on_suffix(tmp_path: Path) -> None:
    """Markdown files go into the first bucket; everything else into second."""
    a = tmp_path / "a.md"
    b = tmp_path / "b.lua"
    c = tmp_path / "c.md"
    d = tmp_path / "d.toml"
    for p in (a, b, c, d):
        p.write_text("x")

    md, other = _partition_markdown([a, b, c, d])
    assert md == [a, c]
    assert other == [b, d]


def test_partition_markdown_empty_input() -> None:
    """Empty list returns two empty lists."""
    md, other = _partition_markdown([])
    assert md == []
    assert other == []


def test_partition_markdown_all_markdown(tmp_path: Path) -> None:
    """All .md → first bucket; second bucket is empty."""
    files = [tmp_path / f"f{i}.md" for i in range(3)]
    for f in files:
        f.write_text("x")

    md, other = _partition_markdown(files)
    assert md == files
    assert other == []


def test_partition_markdown_no_markdown(tmp_path: Path) -> None:
    """No .md files → first bucket is empty; second has all files."""
    files = [tmp_path / "a.lua", tmp_path / "b.toml", tmp_path / "c.rs"]
    for f in files:
        f.write_text("x")

    md, other = _partition_markdown(files)
    assert md == []
    assert other == files


def test_partition_markdown_preserves_order(tmp_path: Path) -> None:
    """Relative order within each bucket matches the input order."""
    names = ["z.md", "a.lua", "m.md", "b.lua", "k.md"]
    paths = [tmp_path / n for n in names]
    for p in paths:
        p.write_text("x")

    md, other = _partition_markdown(paths)
    assert [p.name for p in md] == ["z.md", "m.md", "k.md"]
    assert [p.name for p in other] == ["a.lua", "b.lua"]
