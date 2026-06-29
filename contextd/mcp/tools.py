"""MCP tool implementations — each is a thin wrapper over the GraphStore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextd.mcp.readonly_guard import assert_read_only
from contextd.storage.base import GraphStore


@dataclass
class Overview:
    nodes: list[dict[str, Any]]


def describe_project(store: GraphStore, *, corpus: str | None = None, n: int = 40) -> Overview:
    """Top-N File nodes by inbound-citation count with summaries (spec §7.2).

    Narrowed to ``:File`` so the returned rows have a stable shape
    (``path``, ``name``, ``summary``, ``key_points``, ``inbound``).
    Section-level detail is surfaced via ``section_tree(file_path)`` in
    section-mode corpora. In section-mode, ``File.summary`` is populated
    by ``phase_derive_file_level`` as a rollup of child-section summaries
    (spec-delta #39).

    Delta A applied: merged the two WHERE clauses that the plan rendered
    as consecutive WHEREs (a Cypher parse error). Predicates are now joined
    with AND in a single WHERE clause.
    """
    filters = ["n.summary IS NOT NULL"]
    params: dict[str, Any] = {}
    if corpus:
        filters.append("n.corpus = $corpus")
        params["corpus"] = corpus
    where = "WHERE " + " AND ".join(filters)
    cypher = f"""
    MATCH (n:File)
    {where}
    OPTIONAL MATCH ()-[r]->(n)
    WITH n, count(r) AS inbound
    RETURN n.path AS path, n.name AS name,
           n.summary AS summary, n.key_points AS key_points, inbound
    ORDER BY inbound DESC
    LIMIT {n}
    """
    rows = store.exec_read(cypher, params)
    return Overview(nodes=rows)


_SEARCH_STRIP_FIELDS = frozenset({"embedding"})


def search(
    store: GraphStore, query: str, *, kind: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    """Full-text search over node summaries for ``kind`` (defaults to File).

    Vector-similarity fallback is deferred to a future milestone (the plan
    originally framed this as 'hybrid search'; today's implementation is
    full-text only). Callers needing vector-space matches should call
    ``GraphStore.vector_search`` directly for now.

    Result shape: each row is ``{<node_field>: ..., "score": float}``. The
    node's properties are flattened onto the row and the raw ``embedding``
    vector (1024 floats ≈ 12 KB/row on the MCP wire) is dropped — a full
    ``{node, score}`` nested payload from ``full_text_search`` would blow
    past the MCP client's per-result token ceiling at even modest ``limit``
    values.
    """
    label = kind or "File"
    rows = store.full_text_search(label, "summary", query, k=limit)
    return [
        {
            **{k: v for k, v in r["node"].items() if k not in _SEARCH_STRIP_FIELDS},
            "score": r["score"],
        }
        for r in rows
    ]


_RELATED_MAX_DEPTH = 5
_RELATED_MIN_DEPTH = 1


def related(store: GraphStore, node_id: str, *, depth: int = 2) -> list[dict[str, Any]]:
    """Outbound+inbound traversal within N hops (1-5, inclusive).

    Defence in depth: the MCP tool descriptor's JSON schema already clamps
    via ``"minimum": 1, "maximum": 5`` (spec-delta #32), but a direct
    function caller (tests, future CLI wiring) could still pass out-of-range
    ints. We clamp here too so an unbounded variable-length walk is never
    reachable by accident.
    """
    clamped = min(max(depth, _RELATED_MIN_DEPTH), _RELATED_MAX_DEPTH)
    cypher = f"""
    MATCH (a)-[r*1..{clamped}]-(b)
    WHERE (a.path = $id OR a.id = $id OR a.name = $id)
    RETURN DISTINCT b.path AS path, b.id AS id, b.name AS name, b.summary AS summary
    LIMIT 50
    """
    return store.exec_read(cypher, {"id": node_id})


def inbound(store: GraphStore, node_id: str) -> list[dict[str, Any]]:
    cypher = """
    MATCH (a)-[r]->(b)
    WHERE (b.path = $id OR b.id = $id OR b.name = $id)
    RETURN a.path AS path, a.id AS id, a.name AS name, type(r) AS edge_type
    """
    return store.exec_read(cypher, {"id": node_id})


def outbound(store: GraphStore, node_id: str) -> list[dict[str, Any]]:
    cypher = """
    MATCH (a)-[r]->(b)
    WHERE (a.path = $id OR a.id = $id OR a.name = $id)
    RETURN b.path AS path, b.id AS id, b.name AS name, type(r) AS edge_type
    """
    return store.exec_read(cypher, {"id": node_id})


def get_file_summary(store: GraphStore, path: str) -> dict[str, Any] | None:
    rows = store.exec_read(
        "MATCH (n:File {path: $path}) RETURN n.summary AS summary, n.key_points AS key_points",
        {"path": path},
    )
    return rows[0] if rows else None


def query_graph(store: GraphStore, cypher: str) -> list[dict[str, Any]]:
    """Raw Cypher read — guarded against writes."""
    assert_read_only(cypher)
    return store.exec_read(cypher, {})


def section_tree(store: GraphStore, file_path: str) -> list[dict[str, Any]]:
    """Hierarchical outline of a file — section-granular corpora only."""
    cypher = """
    MATCH (f:File {path: $path})-[:CONTAINS]->(s:Section)
    RETURN s.id AS id, s.title AS title, s.level AS level,
           s.ordinal AS ordinal, s.summary AS summary
    ORDER BY s.level, s.ordinal
    """
    return store.exec_read(cypher, {"path": file_path})
