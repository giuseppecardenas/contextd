"""MCP tool implementations — each is a thin wrapper over the GraphStore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

from contextd.mcp.readonly_guard import assert_read_only
from contextd.providers.base import EmbeddingProvider
from contextd.search.fusion import flatten_row, reciprocal_rank_fusion
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


_VECTOR_CAPABLE_LABELS: Final[frozenset[str]] = frozenset({"File", "Section"})
"""Labels that carry BOTH a vector index and a full-text index and can
therefore be searched hybridly. File and Section get both indexes from the
Neo4j baseline + section-fulltext migrations; every other label is full-text
only (Artifact) or neither (Pattern, Risk, …) and degrades to full-text.

This is the third place index coverage is encoded — the migration DDL and
``contextd/storage/_keys.py`` are the others — and must change in lock-step
with any future vector-index migration."""

_DEFAULT_FETCH_K = 50


def search(
    store: GraphStore,
    query: str,
    *,
    kind: str | None = None,
    limit: int = 20,
    embedder: EmbeddingProvider | None = None,
    mode: Literal["hybrid", "fulltext", "vector"] = "hybrid",
    rrf_k: int = 60,
    fetch_k: int | None = None,
    vector_weight: float = 1.0,
    fulltext_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Hybrid search over node summaries, fusing vector + full-text via RRF.

    In ``hybrid`` mode (the default) the query string is embedded once, both
    the vector and full-text rankers are queried at ``fetch_k`` depth, and the
    two result lists are fused by reciprocal rank fusion (see
    :func:`contextd.search.fusion.reciprocal_rank_fusion`). The tool degrades
    to full-text only — never erroring — when any of these hold: no
    ``embedder`` is supplied, the queried ``kind`` is not in
    :data:`_VECTOR_CAPABLE_LABELS`, or embedding/vector-search raises (a flaky
    or unreachable embedding endpoint must not break search). ``mode`` may be
    forced to ``fulltext`` (skip the vector leg, no embed call) or ``vector``
    (vector ranker only; returns an empty list if the vector leg is
    unavailable, so a caller that explicitly asked for vectors learns it got
    nothing rather than silently receiving lexical results).

    Result shape: each row is ``{<node_field>: ..., "score": float}`` with the
    raw ``embedding`` vector stripped (≈12 KB/row would blow past the MCP
    client's per-result token ceiling). ``score`` is an RRF fused score in
    hybrid/vector mode and the backend's raw relevance score in fulltext mode;
    the two are not comparable across modes.

    :param store: the graph store to query.
    :param query: the natural-language / keyword query string.
    :param kind: node label to search; defaults to ``File``.
    :param limit: maximum rows to return after fusion.
    :param embedder: embedding provider for the query vector; ``None`` forces
        full-text only.
    :param mode: ``hybrid`` (default), ``fulltext``, or ``vector``.
    :param rrf_k: RRF damping constant passed through to fusion.
    :param fetch_k: per-ranker candidate depth before fusion; raised to at
        least ``limit``. Defaults to 50 when ``None``.
    :param vector_weight: RRF weight on the vector ranker.
    :param fulltext_weight: RRF weight on the full-text ranker.
    :return: fused result rows, best-first, at most ``limit`` of them.
    """
    label = kind or "File"
    fetch = max(fetch_k or _DEFAULT_FETCH_K, limit)

    want_vector = (
        mode in ("hybrid", "vector") and embedder is not None and label in _VECTOR_CAPABLE_LABELS
    )

    ft_rows: list[dict[str, Any]] = []
    if mode != "vector":
        ft_rows = store.full_text_search(label, "summary", query, k=fetch)

    vec_rows: list[dict[str, Any]] = []
    if want_vector:
        assert embedder is not None  # narrowed by want_vector; restated for mypy
        try:
            query_vec = embedder.embed([query])[0]
            vec_rows = store.vector_search(label, "embedding", query_vec, k=fetch)
        except Exception:
            # The vector leg crosses an external boundary (embedding API +
            # vector-index query). Any failure there must degrade search to
            # full-text rather than erroring the whole tool — broad catch is
            # deliberate isolation of that dependency, not bug-swallowing.
            want_vector = False

    if mode == "fulltext" or (mode == "hybrid" and not want_vector):
        return [flatten_row(r["node"], r["score"]) for r in ft_rows[:limit]]
    if mode == "vector" and not want_vector:
        return []
    return reciprocal_rank_fusion(
        vec_rows,
        ft_rows,
        label=label,
        limit=limit,
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        fulltext_weight=fulltext_weight,
    )


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
