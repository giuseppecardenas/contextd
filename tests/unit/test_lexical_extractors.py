"""Deterministic lexical reference extraction — ground-truth edges, no LLM."""

from __future__ import annotations

from unittest.mock import MagicMock

from contextd.corpus_config import LexicalPattern
from contextd.indexer.lexical import (
    LexicalRegistry,
    LuaLexicalExtractor,
    MarkdownLexicalExtractor,
)
from contextd.indexer.phases import RelateDeps, _write_unit_edges
from contextd.inference.context import EmptyRetriever, UnitIdentity
from contextd.inference.relate import InferredRelationship
from contextd.ontology.schema import Ontology


def _identity(**overrides: object) -> UnitIdentity:
    kwargs: dict[str, object] = {
        "corpus": "c",
        "file_path": "C:/x/docs/prd/15b-econ.md",
        "rel_path": "docs/prd/15b-econ.md",
        "suffix": ".md",
        "src_label": "Section",
        "src_id": "C:/x/docs/prd/15b-econ.md#intro",
        "title": "Intro",
        "anchor": "intro",
    }
    kwargs.update(overrides)
    return UnitIdentity(**kwargs)  # type: ignore[arg-type]


# --- markdown extractor -------------------------------------------------------


def test_md_link_resolves_relative_path_against_source_dir() -> None:
    refs = MarkdownLexicalExtractor().extract(
        "see [economy](../03f-economy-feudal.md) for detail", identity=_identity()
    )
    assert any(
        r.target_type == "File" and r.target_name == "C:/x/docs/03f-economy-feudal.md" for r in refs
    )


def test_md_link_with_anchor_targets_section() -> None:
    refs = MarkdownLexicalExtractor().extract(
        "see [decay](03f-economy.md#trade-route-decay)", identity=_identity()
    )
    section_refs = [r for r in refs if r.target_type == "Section"]
    assert section_refs
    assert section_refs[0].target_name == "C:/x/docs/prd/03f-economy.md#trade-route-decay"


def test_md_intra_document_anchor_link() -> None:
    refs = MarkdownLexicalExtractor().extract("see [below](#pricing)", identity=_identity())
    assert refs[0].target_type == "Section"
    assert refs[0].target_name == "C:/x/docs/prd/15b-econ.md#pricing"


def test_md_section_symbol_reference() -> None:
    refs = MarkdownLexicalExtractor().extract(
        "as specified in §12.2.5 and §3.1", identity=_identity()
    )
    names = {r.target_name for r in refs if r.rule == "md-anchor"}
    assert names == {"§12.2.5", "§3.1"}


def test_md_bare_basename_reference_excludes_self_and_http() -> None:
    refs = MarkdownLexicalExtractor().extract(
        "compare 15b-econ.md with textile_chain.lua; see https://x.test/a.md",
        identity=_identity(),
    )
    names = {r.target_name for r in refs if r.rule == "md-basename"}
    assert "textile_chain.lua" in names
    assert "15b-econ.md" not in names  # self
    assert "a.md" not in names  # inside a URL... matched? bare scan is bounded by resolve-or-drop


# --- lua extractor ------------------------------------------------------------


def test_lua_require_maps_module_to_path() -> None:
    refs = LuaLexicalExtractor().extract(
        'local base = require("mods.base.sapient_base")\ndofile("mods/base/util.lua")',
        identity=_identity(suffix=".lua"),
    )
    assert {(r.edge_type, r.target_name) for r in refs} == {
        ("DEPENDS_ON", "mods/base/sapient_base.lua"),
        ("DEPENDS_ON", "mods/base/util.lua"),
    }


# --- registry -----------------------------------------------------------------


def test_registry_routes_by_suffix_and_applies_custom_patterns() -> None:
    ontology = Ontology.load_base().with_aliases({"FRRow": "Ticket"})
    registry = LexicalRegistry(
        ontology,
        [
            LexicalPattern(
                regex=r"\bFR-[A-Z]+-\d+\b",
                edge_type="REFERENCES",
                target_type="FRRow",
            )
        ],
    )
    refs = registry.extract("implements FR-ECO-017 and FR-ECO-017", identity=_identity())
    fr = [r for r in refs if r.target_name == "FR-ECO-017"]
    assert len(fr) == 1  # deduped
    assert fr[0].target_type == "Ticket"  # alias resolved to canon


def test_registry_skips_pattern_for_other_formats() -> None:
    registry = LexicalRegistry(
        Ontology.load_base(),
        [
            LexicalPattern(
                regex=r"\bFR-[A-Z]+-\d+\b",
                edge_type="REFERENCES",
                target_type="Ticket",
                formats=["md"],
            )
        ],
    )
    refs = registry.extract("FR-ECO-017", identity=_identity(suffix=".lua"))
    assert all(r.target_name != "FR-ECO-017" for r in refs)


# --- worker integration: lexical wins over LLM duplicates ---------------------


def test_lexical_edge_written_and_llm_duplicate_skipped() -> None:
    store = MagicMock()
    store.exec_read.return_value = [{"v": "C:/x/docs/03f.md"}]  # File target resolves
    relate = RelateDeps(inferrer=MagicMock(), retriever=EmptyRetriever())
    from contextd.indexer.lexical import LexicalReference

    lex = [
        LexicalReference(
            edge_type="REFERENCES", target_type="File", target_name="03f.md", rule="md-link"
        )
    ]
    llm = [
        InferredRelationship(
            edge_type="REFERENCES",
            target_type="File",
            target_name="03f.md",
            confidence=0.9,
            reason="llm says so",
        )
    ]
    skipped = _write_unit_edges(store, relate, "src#a", "Section", lex, llm, "c")
    assert skipped == 1  # the LLM duplicate
    assert store.upsert_edge.call_count == 1
    props = store.upsert_edge.call_args.kwargs["properties"]
    assert props["method"] == "lexical"
    assert props["confidence"] == 1.0
    assert props["reason"].startswith("lexical:")
