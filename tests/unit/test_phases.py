"""Tests for _gc_sections_for_file — per-file section GC helper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from contextd.corpus_config import CorpusConfig


def test_gc_sections_for_file_deletes_stale_section(
    tmp_path: Path,
) -> None:
    from contextd.indexer.phases import _gc_sections_for_file

    md_file = tmp_path / "doc.md"
    md_file.write_text("# Heading A\n\nbody\n")

    store = MagicMock()
    file_path = str(md_file)
    store.exec_read.return_value = [
        {"id": f"{file_path}#heading-a"},
        {"id": f"{file_path}#heading-b"},  # stale
    ]

    corpus_cfg = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "test",
                "root": str(tmp_path),
                "heading_min_level": 1,
            }
        }
    )
    count = _gc_sections_for_file(md_file, corpus_cfg, store)

    assert count == 1
    store.exec_write.assert_called_once()
    call_args = store.exec_write.call_args[0]
    assert "heading-b" in str(call_args)


def test_gc_sections_for_file_preserves_current_sections(
    tmp_path: Path,
) -> None:
    from contextd.indexer.phases import _gc_sections_for_file

    md_file = tmp_path / "doc.md"
    md_file.write_text("# Heading A\n\nbody\n")

    store = MagicMock()
    file_path = str(md_file)
    store.exec_read.return_value = [{"id": f"{file_path}#heading-a"}]

    corpus_cfg = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "test",
                "root": str(tmp_path),
                "heading_min_level": 1,
            }
        }
    )
    count = _gc_sections_for_file(md_file, corpus_cfg, store)

    assert count == 0
    store.exec_write.assert_not_called()


def test_gc_sections_for_file_noop_on_non_md(tmp_path: Path) -> None:
    from contextd.indexer.phases import _gc_sections_for_file

    lua_file = tmp_path / "mod.lua"
    lua_file.write_text("-- code\n")
    store = MagicMock()
    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "test", "root": str(tmp_path)}})
    count = _gc_sections_for_file(lua_file, corpus_cfg, store)

    assert count == 0
    store.exec_read.assert_not_called()
    store.exec_write.assert_not_called()
