"""Unit tests for MCP tool behaviour — Cypher shape, clamps, and descriptors.

Integration coverage (queries actually executing against Neo4j) lives in
tests/integration/test_mcp_tools.py; this file exercises the pure-Python
surface that doesn't need a backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from contextd.mcp import tools


def test_related_clamps_depth_above_max() -> None:
    """Defence in depth: a direct caller passing depth=100 must not
    reach the backend as `[r*1..100]`. Spec-delta #32 clamps at the MCP
    descriptor level; this mirrors the clamp in-function."""
    store = MagicMock()
    store.exec_read.return_value = []
    tools.related(store, "some-id", depth=100)
    cypher = store.exec_read.call_args[0][0]
    assert "[r*1..5]" in cypher
    assert "100" not in cypher


def test_related_clamps_depth_below_min() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    tools.related(store, "some-id", depth=-3)
    cypher = store.exec_read.call_args[0][0]
    assert "[r*1..1]" in cypher


def test_related_default_depth_is_2() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    tools.related(store, "some-id")
    cypher = store.exec_read.call_args[0][0]
    assert "[r*1..2]" in cypher


def test_related_passes_node_id_as_param() -> None:
    """Cypher must not f-string node_id — injection vector."""
    store = MagicMock()
    store.exec_read.return_value = []
    tools.related(store, "file/with/slashes", depth=3)
    cypher, params = store.exec_read.call_args[0]
    assert params == {"id": "file/with/slashes"}
    assert "file/with/slashes" not in cypher


def test_search_strips_embedding_and_flattens_node() -> None:
    """Regression: the raw backend row is ``{node: {..., embedding: [1024 floats]}, score}``.
    That shape was blowing past the MCP client's per-result token ceiling at
    limit>=3 because each row carried ~12KB of embedding noise. The tool must
    (a) drop ``embedding``, (b) flatten the node onto the row so callers see
    ``id``/``summary``/``score`` at the top level.
    """
    store = MagicMock()
    store.full_text_search.return_value = [
        {
            "node": {
                "id": "a.md#intro",
                "path": "a.md",
                "summary": "alpha",
                "key_points": ["k1", "k2"],
                "embedding": [0.1] * 1024,
            },
            "score": 3.14,
        },
        {
            "node": {
                "id": "b.md#main",
                "path": "b.md",
                "summary": "beta",
                "embedding": [0.2] * 1024,
            },
            "score": 2.71,
        },
    ]

    rows = tools.search(store, "alpha", kind="Section", limit=5)

    # No embedder supplied → full-text-only path; the ranker is over-fetched
    # at fetch_k (default 50) and truncated to limit after.
    store.full_text_search.assert_called_once_with("Section", "summary", "alpha", k=50)
    assert len(rows) == 2
    for row in rows:
        assert "embedding" not in row
        assert "node" not in row
        assert "score" in row
    assert rows[0] == {
        "id": "a.md#intro",
        "path": "a.md",
        "summary": "alpha",
        "key_points": ["k1", "k2"],
        "score": 3.14,
    }
    assert rows[1]["id"] == "b.md#main"
    assert rows[1]["score"] == 2.71


def test_search_defaults_to_file_label() -> None:
    store = MagicMock()
    store.full_text_search.return_value = []
    tools.search(store, "query text")
    store.full_text_search.assert_called_once_with("File", "summary", "query text", k=50)


def test_search_handles_rows_without_embedding() -> None:
    """Not every node label carries an embedding (e.g., Pattern). The strip
    filter must be a no-op for rows that never had one."""
    store = MagicMock()
    store.full_text_search.return_value = [
        {"node": {"name": "target1", "summary": "s"}, "score": 1.0},
    ]
    rows = tools.search(store, "q", kind="Pattern")
    assert rows == [{"name": "target1", "summary": "s", "score": 1.0}]


def _fake_embedder() -> MagicMock:
    emb = MagicMock()
    emb.embed.return_value = [[0.1] * 1024]
    return emb


def test_search_hybrid_calls_both_backends_and_embeds_once() -> None:
    store = MagicMock()
    store.full_text_search.return_value = [{"node": {"path": "a.md", "summary": "x"}, "score": 2.0}]
    store.vector_search.return_value = [
        {"node": {"path": "a.md", "summary": "x", "embedding": [0.1] * 1024}, "score": 0.9}
    ]
    emb = _fake_embedder()
    rows = tools.search(store, "q", kind="File", limit=10, embedder=emb)

    emb.embed.assert_called_once_with(["q"])
    store.full_text_search.assert_called_once_with("File", "summary", "q", k=50)
    store.vector_search.assert_called_once()
    vargs, vkwargs = store.vector_search.call_args
    assert vargs[0] == "File" and vargs[1] == "embedding"
    assert vkwargs["k"] == 50
    # Fused output keeps the node flattened with embedding stripped.
    assert rows[0]["path"] == "a.md"
    assert "embedding" not in rows[0]


def test_search_no_embedder_skips_vector_leg() -> None:
    store = MagicMock()
    store.full_text_search.return_value = []
    tools.search(store, "q", kind="File")
    store.vector_search.assert_not_called()


def test_search_noncapable_label_skips_vector_leg() -> None:
    """Pattern has no vector index, so even with an embedder the vector leg is
    not attempted and the query is never embedded."""
    store = MagicMock()
    store.full_text_search.return_value = []
    emb = _fake_embedder()
    tools.search(store, "q", kind="Pattern", embedder=emb)
    store.vector_search.assert_not_called()
    emb.embed.assert_not_called()


def test_search_mode_fulltext_skips_vector_and_embed() -> None:
    store = MagicMock()
    store.full_text_search.return_value = []
    emb = _fake_embedder()
    tools.search(store, "q", kind="File", embedder=emb, mode="fulltext")
    store.vector_search.assert_not_called()
    emb.embed.assert_not_called()


def test_search_mode_vector_on_noncapable_label_returns_empty() -> None:
    """An explicit vector request on a label with no vector index returns []
    (not a silent lexical fallback) so the caller knows it got nothing."""
    store = MagicMock()
    emb = _fake_embedder()
    rows = tools.search(store, "q", kind="Pattern", embedder=emb, mode="vector")
    assert rows == []
    store.full_text_search.assert_not_called()
    store.vector_search.assert_not_called()


def test_search_embed_failure_degrades_to_fulltext() -> None:
    """If embedding raises, search must fall back to full-text, not error."""
    store = MagicMock()
    store.full_text_search.return_value = [{"node": {"path": "a.md", "summary": "x"}, "score": 1.0}]
    emb = MagicMock()
    emb.embed.side_effect = RuntimeError("embedding endpoint unreachable")
    rows = tools.search(store, "q", kind="File", embedder=emb)
    assert rows[0]["path"] == "a.md"
    store.vector_search.assert_not_called()
