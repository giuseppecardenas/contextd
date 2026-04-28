"""Tests for gc_sections_for_file and derive_file_level_for_path — per-file helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from contextd.corpus_config import CorpusConfig
from contextd.indexer.phases import derive_file_level_for_path, gc_sections_for_file


def testgc_sections_for_file_deletes_stale_section(
    tmp_path: Path,
) -> None:
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
    count = gc_sections_for_file(md_file, corpus_cfg, store)

    assert count == 1
    store.exec_write.assert_called_once()
    call_args = store.exec_write.call_args[0]
    assert "heading-b" in str(call_args)


def testgc_sections_for_file_preserves_current_sections(
    tmp_path: Path,
) -> None:
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
    count = gc_sections_for_file(md_file, corpus_cfg, store)

    assert count == 0
    store.exec_write.assert_not_called()


def testgc_sections_for_file_noop_on_non_md(tmp_path: Path) -> None:
    lua_file = tmp_path / "mod.lua"
    lua_file.write_text("-- code\n")
    store = MagicMock()
    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "test", "root": str(tmp_path)}})
    count = gc_sections_for_file(lua_file, corpus_cfg, store)

    assert count == 0
    store.exec_read.assert_not_called()
    store.exec_write.assert_not_called()


def testderive_file_level_for_path_sets_file_summary(tmp_path: Path) -> None:
    store = MagicMock()
    store.exec_read.return_value = [{"summaries": ["Alpha does X.", "Beta does Y."]}]

    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "test", "root": str(tmp_path)}})
    path = tmp_path / "doc.md"
    derive_file_level_for_path(path, corpus_cfg, store)

    store.exec_write.assert_called_once()
    write_call = store.exec_write.call_args
    assert "SET f.summary" in write_call[0][0]


def testderive_file_level_for_path_scoped_to_target_file(tmp_path: Path) -> None:
    store = MagicMock()
    store.exec_read.return_value = [{"summaries": ["Summary."]}]

    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "test", "root": str(tmp_path)}})
    path = tmp_path / "specific.md"
    derive_file_level_for_path(path, corpus_cfg, store)

    read_call = store.exec_read.call_args
    params = read_call[0][1]
    assert params["path"] == str(path)


def testderive_file_level_for_path_handles_no_sections(tmp_path: Path) -> None:
    store = MagicMock()
    store.exec_read.return_value = []  # no sections

    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "test", "root": str(tmp_path)}})
    derive_file_level_for_path(tmp_path / "empty.md", corpus_cfg, store)

    store.exec_write.assert_not_called()


def test_phase_enumerate_sections_stores_section_hash(tmp_path: Path) -> None:
    import hashlib

    from contextd.indexer.phases import phase_enumerate_sections

    md_file = tmp_path / "doc.md"
    md_file.write_text("## Alpha\n\nBody alpha.\n\n## Beta\n\nBody beta.\n")

    store = MagicMock()
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024, [0.2] * 1024]
    hasher = MagicMock()
    hasher.hash.return_value = "abc123"

    corpus_cfg = CorpusConfig.model_validate(
        {"corpus": {"name": "test", "root": str(tmp_path), "heading_min_level": 2}}
    )
    phase_enumerate_sections([md_file], corpus_cfg, store, embedder, hasher)

    # Collect all dicts passed to upsert_node for Section nodes
    section_upserts = [
        call[0][1] for call in store.upsert_node.call_args_list if call[0][0] == "Section"
    ]
    assert len(section_upserts) == 2

    for upsert in section_upserts:
        assert "hash" in upsert
        h = upsert["hash"]
        assert isinstance(h, str) and len(h) == 32  # MD5 hex digest

    # Re-parse to cross-check each hash against the title+body formula
    from contextd.indexer.heading_parser import HeadingParser

    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md_file.read_text())
    for sec, upsert in zip(sections, section_upserts, strict=True):
        expected = hashlib.md5((sec.title + "\n\n" + sec.body).encode()).hexdigest()
        assert upsert["hash"] == expected
