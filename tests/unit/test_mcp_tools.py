"""Unit tests for MCP tool behaviour — Cypher shape, clamps, and descriptors.

Integration coverage (queries actually executing against Neo4j) lives in
tests/integration/test_mcp_tools.py; this file exercises the pure-Python
surface that doesn't need a backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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


def test_search_entity_label_uses_mapped_content_property() -> None:
    """Entity kinds are searched against their declared content field, not the
    File/Section `summary` property, so entity content is actually reachable."""
    store = MagicMock()
    store.full_text_search.return_value = []
    tools.search(store, "auth bug", kind="Ticket")
    store.full_text_search.assert_called_once_with("Ticket", "title", "auth bug", k=50)


def test_search_artifact_label_uses_description_property() -> None:
    store = MagicMock()
    store.full_text_search.return_value = []
    tools.search(store, "reusable script", kind="Artifact")
    store.full_text_search.assert_called_once_with(
        "Artifact", "description", "reusable script", k=50
    )


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


# -- get_node -------------------------------------------------------------


def test_get_node_returns_props_without_embedding() -> None:
    store = MagicMock()
    store.exec_read.return_value = [
        {"labels": ["Ticket"], "props": {"id": "INTENG-1", "title": "t", "embedding": [0.1] * 1024}}
    ]
    result = tools.get_node(store, "INTENG-1")
    assert result == {"labels": ["Ticket"], "id": "INTENG-1", "title": "t"}
    cypher, params = store.exec_read.call_args[0]
    assert params == {"id": "INTENG-1"}
    assert "INTENG-1" not in cypher  # bound as param, never f-strung


def test_get_node_returns_none_when_absent() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    assert tools.get_node(store, "nope") is None


# -- explain_relationship -------------------------------------------------


def test_explain_relationship_binds_params_and_returns_provenance() -> None:
    store = MagicMock()
    store.exec_read.return_value = [
        {
            "source": "a.md",
            "target": "INTENG-1",
            "edge_type": "REFERENCES",
            "outbound": True,
            "origin": "inferred",
            "confidence": 0.9,
            "reason": "why",
        }
    ]
    rows = tools.explain_relationship(store, "a.md", "INTENG-1")
    cypher, params = store.exec_read.call_args[0]
    assert params == {"source": "a.md", "target": "INTENG-1"}
    assert "a.md" not in cypher
    assert rows[0]["reason"] == "why"
    assert rows[0]["origin"] == "inferred"


# -- ticket_dossier -------------------------------------------------------


def test_ticket_dossier_groups_neighbors() -> None:
    store = MagicMock()
    store.exec_read.return_value = [
        {
            "ticket": "INTENG-1",
            "ticket_props": {"id": "INTENG-1", "title": "t"},
            "edge_type": "DOCUMENTS",
            "outbound": False,
            "neighbor_labels": ["File"],
            "neighbor": "a.md",
            "neighbor_summary": "sum",
            "neighbor_title": None,
        }
    ]
    result = tools.ticket_dossier(store, "INTENG-1")
    assert result["found"] is True
    assert result["properties"] == {"id": "INTENG-1", "title": "t"}
    assert result["neighbors"] == [
        {
            "edge_type": "DOCUMENTS",
            "direction": "inbound",
            "labels": ["File"],
            "node": "a.md",
            "summary": "sum",
            "title": None,
        }
    ]


def test_ticket_dossier_isolated_ticket_has_empty_neighbors() -> None:
    """OPTIONAL MATCH yields one all-null neighbor row; it must not appear."""
    store = MagicMock()
    store.exec_read.return_value = [
        {
            "ticket": "INTENG-9",
            "ticket_props": {"id": "INTENG-9"},
            "edge_type": None,
            "outbound": None,
            "neighbor_labels": None,
            "neighbor": None,
            "neighbor_summary": None,
            "neighbor_title": None,
        }
    ]
    result = tools.ticket_dossier(store, "INTENG-9")
    assert result["found"] is True
    assert result["neighbors"] == []


def test_ticket_dossier_not_found() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    assert tools.ticket_dossier(store, "NOPE-1") == {
        "ticket": "NOPE-1",
        "found": False,
        "properties": {},
        "neighbors": [],
    }


# -- find_reusable --------------------------------------------------------


def test_find_reusable_keeps_only_reusable_true() -> None:
    store = MagicMock()
    store.full_text_search.return_value = [
        {"node": {"id": "A1", "description": "d", "reusable": True}, "score": 2.0},
        {"node": {"id": "A2", "description": "d2", "reusable": False}, "score": 1.5},
        {"node": {"id": "A3", "description": "d3"}, "score": 1.0},  # reusable unset
    ]
    rows = tools.find_reusable(store, "script")
    store.full_text_search.assert_called_once_with("Artifact", "description", "script", k=50)
    assert [r["id"] for r in rows] == ["A1"]
    assert rows[0]["score"] == 2.0


# -- list_entities --------------------------------------------------------


def test_list_entities_rejects_unknown_kind() -> None:
    store = MagicMock()
    with pytest.raises(ValueError, match="Unknown entity kind"):
        tools.list_entities(store, "Widget")


def test_list_entities_rejects_unknown_property() -> None:
    store = MagicMock()
    with pytest.raises(ValueError, match="Unknown property"):
        tools.list_entities(store, "Ticket", prop="bogus", value="x")


def test_list_entities_interpolates_validated_label_and_binds_values() -> None:
    store = MagicMock()
    store.exec_read.return_value = [
        {"labels": ["Ticket"], "props": {"id": "INTENG-1", "status": "open"}}
    ]
    rows = tools.list_entities(store, "Ticket", prop="status", value="open", corpus="c", limit=10)
    cypher, params = store.exec_read.call_args[0]
    assert "MATCH (n:Ticket)" in cypher
    assert "n.status = $value" in cypher
    assert "n.corpus = $corpus" in cypher
    assert params == {"corpus": "c", "value": "open"}
    assert rows == [{"labels": ["Ticket"], "id": "INTENG-1", "status": "open"}]


# -- check_freshness ------------------------------------------------------


def test_check_freshness_requires_a_scope() -> None:
    store = MagicMock()
    with pytest.raises(ValueError, match="requires node_id or corpus"):
        tools.check_freshness(store)


def test_check_freshness_node_scope_binds_id_and_types() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    tools.check_freshness(store, node_id="a.md")
    cypher, params = store.exec_read.call_args[0]
    assert params == {"types": ["SUPERSEDES", "CONTRADICTS", "NEEDS_UPDATE"], "id": "a.md"}
    assert "type(r) IN $types" in cypher
    assert "a.md" not in cypher  # bound, not f-strung


def test_check_freshness_corpus_scope_binds_corpus() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    tools.check_freshness(store, corpus="notes")
    cypher, params = store.exec_read.call_args[0]
    assert params["corpus"] == "notes"
    assert "a.corpus = $corpus" in cypher


# -- find_contradictions --------------------------------------------------


def test_find_contradictions_no_topic_has_no_where() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    tools.find_contradictions(store)
    cypher, params = store.exec_read.call_args[0]
    assert "CONTRADICTS" in cypher
    assert "WHERE" not in cypher
    assert params == {}


def test_find_contradictions_topic_adds_summary_filter() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    tools.find_contradictions(store, topic="logging")
    cypher, params = store.exec_read.call_args[0]
    assert params == {"topic": "logging"}
    assert "CONTAINS toLower($topic)" in cypher
    assert "logging" not in cypher  # bound, not f-strung


# -- whats_new ------------------------------------------------------------


def test_whats_new_binds_since_and_orders_by_updated() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    tools.whats_new(store, "2026-06-01T00:00:00Z", corpus="notes")
    cypher, params = store.exec_read.call_args[0]
    assert params == {"since": "2026-06-01T00:00:00Z", "corpus": "notes"}
    assert "n.updated >= datetime($since)" in cypher
    assert "ORDER BY n.updated DESC" in cypher
    assert "n.corpus = $corpus" in cypher


# -- timeline -------------------------------------------------------------


def test_timeline_requires_an_anchor() -> None:
    store = MagicMock()
    with pytest.raises(ValueError, match="requires node_id or topic"):
        tools.timeline(store)


def test_timeline_node_anchor_returns_nodes_and_supersedes() -> None:
    store = MagicMock()
    store.exec_read.side_effect = [
        [
            {
                "node": "a.md",
                "labels": ["File"],
                "summary": "s",
                "updated": None,
                "inferred_at": None,
            }
        ],
        [{"newer": "a.md", "older": "old.md", "confidence": 0.9, "reason": "why"}],
    ]
    result = tools.timeline(store, node_id="a.md")
    # Two reads: the ordered nodes, then the SUPERSEDES chains.
    assert store.exec_read.call_count == 2
    nodes_params = store.exec_read.call_args_list[0][0][1]
    assert nodes_params == {"id": "a.md"}
    assert result["nodes"][0]["node"] == "a.md"
    assert result["supersedes"][0] == {
        "newer": "a.md",
        "older": "old.md",
        "confidence": 0.9,
        "reason": "why",
    }


def test_timeline_topic_anchor_binds_topic() -> None:
    store = MagicMock()
    store.exec_read.side_effect = [[], []]
    tools.timeline(store, topic="logging")
    nodes_cypher, nodes_params = store.exec_read.call_args_list[0][0]
    assert nodes_params == {"topic": "logging"}
    assert "CONTAINS toLower($topic)" in nodes_cypher
