"""Cross-backend vector_search + full_text_search integration tests.

Search paths have the most Cypher-interpolation risk of any backend method
(label/property/threshold go straight into the procedure call). These tests
exercise both backends against a small deterministic corpus so the round-trip
works end-to-end before M5 starts writing real embeddings.
"""

from __future__ import annotations

import pytest

from contextd.storage.base import GraphStore

pytestmark = pytest.mark.integration


_DIM = 1024  # matches both backends' baseline migration


def _basis_vector(first_nonzero: int, second_nonzero: int | None = None) -> list[float]:
    """Build a 1024-dim vector where only index `first_nonzero` (and optionally
    `second_nonzero`, equal weight) is non-zero. Cosine similarity between any
    two such vectors is 0 (orthogonal), 0.7071 (45°), or 1.0 (aligned)."""
    v = [0.0] * _DIM
    v[first_nonzero] = 1.0
    if second_nonzero is not None:
        v[first_nonzero] = 0.7071
        v[second_nonzero] = 0.7071
    return v


_EMBEDDINGS = {
    "a.md": _basis_vector(0),  # aligned with query
    "b.md": _basis_vector(1),  # orthogonal to query
    "c.md": _basis_vector(0, 1),  # 45° from query
}


def _seed_corpus(backend: GraphStore, *, with_embeddings: bool = True) -> None:
    """Insert three File nodes with distinct summaries, optionally with embeddings."""
    corpus = [
        ("a.md", "database migration forward only"),
        ("b.md", "unrelated foo bar text"),
        ("c.md", "migration of schema with indexes"),
    ]
    for path, summary in corpus:
        props: dict[str, object] = {"path": path, "summary": summary, "corpus": "c"}
        if with_embeddings:
            props["embedding"] = _EMBEDDINGS[path]
        backend.upsert_node("File", props)


def _rebuild_fts_index_if_needed(backend: GraphStore) -> None:
    """No-op on the current backend roster.

    Memgraph's TEXT INDEX and Neo4j's FULLTEXT INDEX both pick up late
    writes without an explicit rebuild. The helper is retained for a
    uniform call-site signature and a place to hang future backend-
    specific workarounds.
    """
    return


# ---------- vector_search ----------


@pytest.fixture
def seeded_backend(backend: GraphStore) -> GraphStore:
    _seed_corpus(backend, with_embeddings=True)
    return backend


def test_vector_search_orders_by_similarity(seeded_backend: GraphStore) -> None:
    """Cosine similarity ranks the aligned vector first, then the 45° vector,
    then the orthogonal vector."""
    query = _basis_vector(0)
    results = seeded_backend.vector_search(
        label="File", property_name="embedding", query=query, k=3
    )
    paths = [r["node"]["path"] for r in results]
    assert paths[0] == "a.md"
    # "c.md" is at 45° from the query → closer than orthogonal "b.md".
    assert paths.index("c.md") < paths.index("b.md")


def test_vector_search_rejects_non_finite_threshold(seeded_backend: GraphStore) -> None:
    """nan / inf threshold bypass the parameter bind and land in raw Cypher on
    backends that f-string it. Both backends must refuse at the Python
    boundary with ValueError."""
    import math as _m

    query = _basis_vector(0)
    for bad in (_m.nan, _m.inf, -_m.inf):
        with pytest.raises(ValueError, match="threshold must be finite"):
            seeded_backend.vector_search(
                label="File", property_name="embedding", query=query, k=3, threshold=bad
            )


def test_vector_search_threshold_filters(seeded_backend: GraphStore) -> None:
    """A threshold above the orthogonal-vector score drops it and keeps the
    aligned (cos=1) and 45° (cos≈0.7071) results.

    The ABC normalises the contract as a similarity floor on [0, 1], but the
    scoring origin differs by backend:

    - Memgraph: raw cosine similarity (orthogonal = 0.0).
    - Neo4j: (1 + dot) / 2 normalisation (orthogonal = 0.5, documented in
      ``Neo4jBackend.vector_search``).

    We pick ``threshold=0.6`` so the orthogonal vector is dropped on every
    backend (0.0 < 0.6 on Memgraph; 0.5 < 0.6 on Neo4j) while the
    45° vector (0.7071 raw / ≈0.854 Neo4j) and aligned vector still pass.
    Threshold calibration is inherently backend-aware because of the
    scoring-origin difference; picking a value that works under both
    regimes is the portable shape.
    """
    query = _basis_vector(0)
    results = seeded_backend.vector_search(
        label="File",
        property_name="embedding",
        query=query,
        k=3,
        threshold=0.6,
    )
    paths = {r["node"]["path"] for r in results}
    assert paths == {"a.md", "c.md"}


def test_vector_search_returns_score_on_both_backends(seeded_backend: GraphStore) -> None:
    """ABC contract: returned rows carry a `score` key (cosine similarity in
    [0, 1]) on every backend. No `distance` key is exposed."""
    query = _basis_vector(0)
    results = seeded_backend.vector_search(
        label="File", property_name="embedding", query=query, k=3
    )
    assert len(results) > 0
    for row in results:
        assert "score" in row
        assert "distance" not in row
        assert isinstance(row["score"], int | float)
        assert 0.0 <= row["score"] <= 1.0
    # And the top result for the aligned query vector should be near-1 similarity.
    assert results[0]["score"] > 0.99


# ---------- full_text_search ----------


def test_full_text_search_matches_content_word(backend: GraphStore) -> None:
    """A content word present in two summaries returns both; a word absent
    from every summary returns nothing."""
    _seed_corpus(backend, with_embeddings=False)
    _rebuild_fts_index_if_needed(backend)

    hits = backend.full_text_search(label="File", property_name="summary", query="migration", k=5)
    # Both a.md and c.md mention "migration"; b.md does not.
    hit_paths = {row["node"]["path"] for row in hits}
    assert hit_paths == {"a.md", "c.md"}
    # Every hit must carry a numeric score — ranked output is the point of FTS.
    for row in hits:
        assert isinstance(row["score"], int | float)
        assert row["score"] > 0.0

    empty = backend.full_text_search(label="File", property_name="summary", query="quantum", k=5)
    assert empty == []


def test_full_text_search_section_summary(backend: GraphStore) -> None:
    """Section.summary must be full-text-searchable on every backend.

    Section-granular corpora (spec §5.11) put the LLM's per-section summary on
    ``:Section.summary`` rather than ``:File.summary``. The ``Section_summary_ft``
    index (migration _0003 on both backends) exists so ``search(kind='Section')``
    through the MCP tool stack doesn't crash with a missing-index error.
    """
    sections = [
        ("a.md#intro", "database migration forward only"),
        ("b.md#unrelated", "unrelated foo bar text"),
        ("c.md#schema", "migration of schema with indexes"),
    ]
    for sec_id, summary in sections:
        backend.upsert_node(
            "Section",
            {"id": sec_id, "summary": summary, "corpus": "c"},
        )
    _rebuild_fts_index_if_needed(backend)

    hits = backend.full_text_search(
        label="Section", property_name="summary", query="migration", k=5
    )
    ids = {row["node"]["id"] for row in hits}
    assert ids == {"a.md#intro", "c.md#schema"}
    for row in hits:
        assert isinstance(row["score"], int | float)
        assert row["score"] > 0.0

    empty = backend.full_text_search(label="Section", property_name="summary", query="quantum", k=5)
    assert empty == []
