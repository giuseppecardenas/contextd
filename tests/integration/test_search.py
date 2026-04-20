"""Cross-backend vector_search + full_text_search integration tests.

Search paths have the most Cypher-interpolation risk of any backend method
(label/property/threshold go straight into the procedure call). These tests
exercise both backends against a small deterministic corpus so the round-trip
works end-to-end before M5 starts writing real embeddings.
"""

from __future__ import annotations

from typing import Any

import pytest

from contextd.storage.base import GraphStore

pytestmark = pytest.mark.integration


def _node_prop(node: Any, prop: str) -> Any:
    """Read a property off a `node` result cell.

    The backends return different shapes for node cells: Kuzu yields a dict,
    Memgraph (via gqlalchemy) yields a `Node` model object. This helper
    hides that until/unless the ABC grows a normaliser.
    """
    if isinstance(node, dict):
        return node[prop]
    return getattr(node, prop)


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
    """Memgraph's TEXT INDEX is created in the baseline migration, but empty
    until the first write invalidates the cache. A subsequent search works
    without ceremony — no explicit rebuild is required. Kuzu FTS indexes,
    by contrast, index only rows that were present at CREATE_FTS_INDEX
    time, so populating rows after the migration means the FTS sees no hits.

    This helper drops and recreates the Kuzu FTS index post-seeding. It is
    a no-op on Memgraph.
    """
    if backend.capabilities.name != "kuzu":
        return
    # Drop + recreate so the FTS picks up the just-inserted rows.
    backend.exec_write("CALL DROP_FTS_INDEX('File', 'File_summary_ft')", None)
    backend.exec_write("CALL CREATE_FTS_INDEX('File', 'File_summary_ft', ['summary'])", None)


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
    paths = [_node_prop(r["node"], "path") for r in results]
    assert paths[0] == "a.md"
    # "c.md" is at 45° from the query → closer than orthogonal "b.md".
    assert paths.index("c.md") < paths.index("b.md")


def test_vector_search_threshold_filters(seeded_backend: GraphStore) -> None:
    """threshold=0.5 drops the orthogonal vector (cos=0) and keeps the
    aligned (cos=1) and 45° (cos≈0.7071) results."""
    if seeded_backend.capabilities.name == "kuzu":
        pytest.skip(
            "Kuzu's vector_search uses distance (lower=better); the current "
            "threshold semantics apply only to Memgraph's similarity score."
        )
    query = _basis_vector(0)
    results = seeded_backend.vector_search(
        label="File",
        property_name="embedding",
        query=query,
        k=3,
        threshold=0.5,
    )
    paths = {_node_prop(r["node"], "path") for r in results}
    assert paths == {"a.md", "c.md"}


# ---------- full_text_search ----------


def test_full_text_search_matches_content_word(backend: GraphStore) -> None:
    """A content word present in two summaries returns both; a word absent
    from every summary returns nothing."""
    _seed_corpus(backend, with_embeddings=False)
    _rebuild_fts_index_if_needed(backend)

    hits = backend.full_text_search(label="File", property_name="summary", query="migration", k=5)
    # Both a.md and c.md mention "migration"; b.md does not.
    hit_paths = {_node_prop(row["node"], "path") for row in hits}
    assert hit_paths == {"a.md", "c.md"}

    empty = backend.full_text_search(label="File", property_name="summary", query="quantum", k=5)
    assert empty == []
