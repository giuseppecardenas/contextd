"""Unit-extraction seam: extractor selection, ParsedFile lookups, ParseCache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from contextd.corpus_config import CorpusConfig
from contextd.indexer.heading_parser import HeadingParser
from contextd.indexer.units import ParseCache, ParsedFile, extractor_for, own_prose


def _cfg(tmp_path: Path) -> CorpusConfig:
    return CorpusConfig.model_validate(
        {"corpus": {"name": "u", "root": str(tmp_path), "granularity": "section"}}
    )


def test_extractor_for_routes_markdown_only(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    md = extractor_for(cfg, ".md")
    assert isinstance(md, HeadingParser)
    assert extractor_for(cfg, ".lua") is None
    assert extractor_for(cfg, ".json") is None


def test_extractor_respects_heading_bounds(tmp_path: Path) -> None:
    cfg = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "u",
                "root": str(tmp_path),
                "heading_min_level": 3,
                "heading_max_level": 5,
            }
        }
    )
    extractor = extractor_for(cfg, ".md")
    assert extractor is not None
    sections = extractor.parse("## Skipped\n\n### Kept\n\nbody\n")
    # The below-min H2 is not promoted; its text becomes the preamble
    # (borrowing the H2 title). Only "Kept" is a promoted heading section.
    assert [s.title for s in sections] == ["Skipped", "Kept"]
    assert sections[0].ordinal == 0 and sections[0].parent_anchor is None
    assert sections[1].anchor == "kept"


_MD = (
    "## Top\n\ntop prose\n\n"
    "### Mid\n\nmid prose\n\n"
    "#### Leaf\n\nleaf prose\n\n"
    "### Mid Two\n\nmid two prose\n"
)


def _parsed(tmp_path: Path) -> ParsedFile:
    f = tmp_path / "doc.md"
    f.write_text(_MD, encoding="utf-8")
    return ParseCache(_cfg(tmp_path)).get(f)


def test_parent_chain_outermost_first(tmp_path: Path) -> None:
    pf = _parsed(tmp_path)
    assert pf.parent_chain("leaf") == ("Top", "Mid")
    assert pf.parent_chain("mid") == ("Top",)
    assert pf.parent_chain("top") == ()


def test_children_of_document_order(tmp_path: Path) -> None:
    pf = _parsed(tmp_path)
    assert [s.title for s in pf.children_of("top")] == ["Mid", "Mid Two"]
    assert [s.title for s in pf.children_of("mid")] == ["Leaf"]
    assert pf.children_of("leaf") == []


def test_by_anchor_lookup(tmp_path: Path) -> None:
    pf = _parsed(tmp_path)
    sec = pf.by_anchor("mid-two")
    assert sec is not None and sec.title == "Mid Two"
    assert pf.by_anchor("ghost") is None


def test_own_prose_strips_heading_line(tmp_path: Path) -> None:
    pf = _parsed(tmp_path)
    top = pf.by_anchor("top")
    assert top is not None
    assert own_prose(top).strip() == "top prose"


def test_own_prose_empty_for_heading_only_parent() -> None:
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse("## Parent\n### Child\n\nchild body\n")
    parent = sections[0]
    assert own_prose(parent).strip() == ""


def test_cache_parses_each_file_once(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text(_MD, encoding="utf-8")
    cfg = _cfg(tmp_path)
    cache = ParseCache(cfg)
    spy = MagicMock(wraps=HeadingParser(min_level=2, max_level=4))
    # Substitute the extractor the cache would construct.
    cache._files.clear()
    import contextd.indexer.units as units_mod

    original = units_mod.extractor_for
    try:
        units_mod.extractor_for = lambda _cfg, _sfx: spy  # type: ignore[assignment]
        first = cache.get(f)
        second = cache.get(f)
    finally:
        units_mod.extractor_for = original
    assert first is second
    assert spy.parse.call_count == 1


def test_cache_unparseable_suffix_yields_empty_units(tmp_path: Path) -> None:
    f = tmp_path / "mod.lua"
    f.write_text("return {}\n", encoding="utf-8")
    parsed = ParseCache(_cfg(tmp_path)).get(f)
    assert parsed.sections == []
    assert parsed.canonical.endswith("mod.lua")
