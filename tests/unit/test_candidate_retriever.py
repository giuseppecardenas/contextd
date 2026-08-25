"""GraphCandidateRetriever — per-unit candidate lookup for the relate phase."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

from contextd.indexer.candidates import GraphCandidateRetriever
from contextd.inference.context import UnitIdentity
from contextd.ontology.schema import Ontology

SRC_ID = "C:/x/docs/a.md#intro"


def _identity(**overrides: object) -> UnitIdentity:
    kwargs: dict[str, object] = {
        "corpus": "c",
        "file_path": "C:/x/docs/a.md",
        "rel_path": "docs/a.md",
        "suffix": ".md",
        "src_label": "Section",
        "src_id": SRC_ID,
        "title": "Intro",
        "anchor": "intro",
    }
    kwargs.update(overrides)
    return UnitIdentity(**kwargs)  # type: ignore[arg-type]


def _store(
    entity_rows: list[dict[str, Any]] | None = None,
    same_file_rows: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Store whose exec_read dispatches on the query text."""
    store = MagicMock()

    def _exec_read(query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if "COUNT { (n)--() }" in query:
            return entity_rows or []
        if "ORDER BY s.ordinal" in query:
            return same_file_rows or []
        if "n.embedding AS embedding" in query:
            return []
        return []

    store.exec_read.side_effect = _exec_read
    store.vector_search.return_value = []
    store.full_text_search.return_value = []
    return store


def test_entities_grouped_by_label_and_corpus_scoped() -> None:
    store = _store(entity_rows=[{"name": "spatial hash"}, {"name": "triple-buffer"}])
    retriever = GraphCandidateRetriever(Ontology.load_base())
    bundle = retriever.for_unit(store, identity=_identity())
    # Every mintable label got the same fake rows back; check shape + scoping.
    assert bundle.entities_by_label["Pattern"] == ("spatial hash", "triple-buffer")
    entity_calls = [c for c in store.exec_read.call_args_list if "COUNT { (n)--() }" in c.args[0]]
    assert entity_calls
    assert all(c.args[1]["c"] == "c" for c in entity_calls)


def test_same_file_sections_exclude_self() -> None:
    store = _store(
        same_file_rows=[
            {"id": SRC_ID, "title": "Intro"},
            {"id": "C:/x/docs/a.md#body", "title": "Body"},
        ]
    )
    bundle = GraphCandidateRetriever(Ontology.load_base()).for_unit(store, identity=_identity())
    ids = [s.id for s in bundle.sections]
    assert "C:/x/docs/a.md#body" in ids
    assert SRC_ID not in ids


def test_entity_cache_ttl_prevents_requerying() -> None:
    store = _store(entity_rows=[{"name": "x"}])
    retriever = GraphCandidateRetriever(Ontology.load_base(), cache_ttl_seconds=60.0)
    retriever.for_unit(store, identity=_identity())
    first_count = len(
        [c for c in store.exec_read.call_args_list if "COUNT { (n)--() }" in c.args[0]]
    )
    retriever.for_unit(store, identity=_identity())
    second_count = len(
        [c for c in store.exec_read.call_args_list if "COUNT { (n)--() }" in c.args[0]]
    )
    assert second_count == first_count  # cache hit — no new entity queries


def test_entity_cache_expires_after_ttl() -> None:
    store = _store(entity_rows=[{"name": "x"}])
    retriever = GraphCandidateRetriever(Ontology.load_base(), cache_ttl_seconds=0.01)
    retriever.for_unit(store, identity=_identity())
    before = len([c for c in store.exec_read.call_args_list if "COUNT { (n)--() }" in c.args[0]])
    time.sleep(0.02)
    retriever.for_unit(store, identity=_identity())
    after = len([c for c in store.exec_read.call_args_list if "COUNT { (n)--() }" in c.args[0]])
    assert after == before * 2


def test_vector_similarity_uses_stored_embedding_and_corpus_filter() -> None:
    store = MagicMock()

    def _exec_read(query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if "n.embedding AS embedding" in query:
            return [{"embedding": [0.5] * 4}]
        return []

    store.exec_read.side_effect = _exec_read
    store.vector_search.return_value = [
        {"node": {"id": "C:/x/docs/b.md#s", "title": "S", "corpus": "c"}, "score": 0.9},
    ]
    store.full_text_search.return_value = []
    bundle = GraphCandidateRetriever(Ontology.load_base()).for_unit(store, identity=_identity())
    ids = [s.id for s in bundle.sections]
    assert "C:/x/docs/b.md#s" in ids
    # Stored embedding, not a fresh embed call, fed vector_search.
    assert store.vector_search.call_args.kwargs["query"] == [0.5] * 4
    # Corpus scoping is pushed down to the backend (server-side over-fetch +
    # filter), not applied client-side on the returned rows.
    assert store.vector_search.call_args.kwargs["filters"] == {"corpus": "c"}
    assert store.full_text_search.call_args.kwargs["filters"] == {"corpus": "c"}


def test_missing_embedding_degrades_gracefully() -> None:
    store = _store()  # embedding read returns []
    bundle = GraphCandidateRetriever(Ontology.load_base()).for_unit(store, identity=_identity())
    assert bundle.sections == ()
    store.vector_search.assert_not_called()


def test_fulltext_query_escapes_lucene_specials() -> None:
    from contextd.indexer.candidates import _lucene_escape

    assert _lucene_escape("Save/Load Format") == "Save\\/Load Format"
    assert _lucene_escape("§6.14 Pricing: tiers (v2)") == "§6.14 Pricing\\: tiers \\(v2\\)"

    store = MagicMock()

    def _exec_read(query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if "n.embedding AS embedding" in query:
            return [{"embedding": [0.5] * 4}]
        return []

    store.exec_read.side_effect = _exec_read
    store.vector_search.return_value = []
    store.full_text_search.return_value = []
    GraphCandidateRetriever(Ontology.load_base()).for_unit(
        store, identity=_identity(title="Save/Load Format")
    )
    ft_queries = [c.kwargs["query"] for c in store.full_text_search.call_args_list]
    assert ft_queries
    assert all("/" not in q or "\\/" in q for q in ft_queries)
    assert any(q == "Save\\/Load Format" for q in ft_queries)


def test_source_failures_never_raise() -> None:
    store = MagicMock()
    store.exec_read.side_effect = RuntimeError("db down")
    store.vector_search.side_effect = RuntimeError("db down")
    store.full_text_search.side_effect = RuntimeError("db down")
    bundle = GraphCandidateRetriever(Ontology.load_base()).for_unit(store, identity=_identity())
    assert bundle.entities_by_label == {}
    assert bundle.sections == ()
    assert bundle.files == ()


def test_chunk_leg_adds_sections_reached_through_chunk_vectors() -> None:
    """Body-level similarity (Chunk vectors) surfaces sections whose summaries
    differ but whose text overlaps; they are mapped back to their parent."""
    store = MagicMock()

    def _exec_read(query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if "n.embedding AS embedding" in query:
            return [{"embedding": [0.1, 0.2]}]
        if "ORDER BY s.ordinal" in query:
            return []
        if "MATCH (s:Section) WHERE s.id IN $ids" in query:
            return [{"id": "C:/x/docs/b.md#body", "title": "Body"}]
        return []

    store.exec_read.side_effect = _exec_read

    def _vector_search(
        label: str, property_name: str, query: list[float], k: int, **kw: Any
    ) -> list[dict[str, Any]]:
        if label == "Chunk":
            assert kw["filters"] == {"corpus": "c", "parent_label": "Section"}
            return [
                {
                    "node": {
                        "id": "C:/x/docs/b.md#body~fine~0",
                        "parent_id": "C:/x/docs/b.md#body",
                    },
                    "score": 0.9,
                },
                {
                    "node": {
                        "id": "C:/x/docs/b.md#body~fine~1",
                        "parent_id": "C:/x/docs/b.md#body",
                    },
                    "score": 0.8,
                },
                {"node": {"id": SRC_ID + "~fine~0", "parent_id": SRC_ID}, "score": 0.7},
            ]
        return []

    store.vector_search.side_effect = _vector_search
    store.full_text_search.return_value = []

    retriever = GraphCandidateRetriever(Ontology.load_base())
    sections = retriever._sections(store, _identity())
    assert [s.id for s in sections] == ["C:/x/docs/b.md#body"]
    assert sections[0].title == "Body"


def test_chunk_leg_failure_is_isolated() -> None:
    store = MagicMock()
    store.exec_read.side_effect = lambda q, p: (
        [{"embedding": [0.1]}] if "n.embedding AS embedding" in q else []
    )
    store.vector_search.side_effect = RuntimeError("no chunk index")
    store.full_text_search.return_value = []
    retriever = GraphCandidateRetriever(Ontology.load_base())
    assert retriever._sections(store, _identity()) == ()
