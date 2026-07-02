"""MCP tool implementations — each is a thin wrapper over the GraphStore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

from contextd.mcp.readonly_guard import assert_read_only
from contextd.ontology.schema import Ontology
from contextd.providers.base import EmbeddingProvider
from contextd.search.fusion import flatten_row, reciprocal_rank_fusion
from contextd.storage.base import GraphStore

# Base-ontology node types, loaded once. Used to validate the ``kind`` label and
# any property name that ``list_entities`` interpolates into Cypher, so a
# client-supplied label/property can never inject — only declared identifiers
# reach the query text; all values are bound as parameters.
_ONTOLOGY: Final[Ontology] = Ontology.load_base()
_ALLOWED_LABELS: Final[frozenset[str]] = frozenset(_ONTOLOGY.node_types)


@dataclass
class Overview:
    nodes: list[dict[str, Any]]


def _node_without_embedding(props: dict[str, Any]) -> dict[str, Any]:
    """Drop the embedding vector from a node property map.

    The 1024-float vector is ~12 KB and would blow past an MCP client's
    per-result token budget; every tool that returns full node properties
    routes them through here (the search path uses ``flatten_row`` for the
    same reason).
    """
    return {k: v for k, v in props.items() if k != "embedding"}


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
only (Artifact/Ticket/Pattern/Risk via the entity-content migrations) or
neither (Technology, Client, …) and degrades to full-text.

This is the third place index coverage is encoded — the migration DDL and
``contextd/storage/_keys.py`` are the others — and must change in lock-step
with any future vector-index migration."""

_SEARCH_PROPERTY_BY_LABEL: Final[dict[str, str]] = {
    "File": "summary",
    "Section": "summary",
    "Artifact": "description",
    "Ticket": "title",
    "Pattern": "description",
    "Risk": "description",
}
"""Full-text property searched per label. File/Section carry AI summaries;
entity types carry their declared content field (populated by the indexer and
indexed by the baseline/_0006 full-text migrations). Labels absent here fall
back to ``summary`` — which only matches if such an index exists for them."""


def _search_property(label: str) -> str:
    """Return the full-text property to query for ``label`` (default summary)."""
    return _SEARCH_PROPERTY_BY_LABEL.get(label, "summary")


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
        ft_rows = store.full_text_search(label, _search_property(label), query, k=fetch)

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


def get_node(store: GraphStore, node_id: str) -> dict[str, Any] | None:
    """Return a single node's labels and properties, matched by path/id/name.

    The generic, entity-aware counterpart to :func:`get_file_summary` (which
    reads only ``:File``). Resolves ``node_id`` against the same identity
    predicates the traversal tools use, so it works for the newly-populated
    entity nodes (Ticket, Artifact, Pattern, …) that ``get_file_summary``
    cannot read. The embedding vector is stripped. Returns ``None`` when no
    node matches.
    """
    rows = store.exec_read(
        """
        MATCH (n)
        WHERE n.path = $id OR n.id = $id OR n.name = $id
        RETURN labels(n) AS labels, properties(n) AS props
        LIMIT 1
        """,
        {"id": node_id},
    )
    if not rows:
        return None
    return {"labels": rows[0]["labels"], **_node_without_embedding(rows[0]["props"])}


def explain_relationship(store: GraphStore, source: str, target: str) -> list[dict[str, Any]]:
    """Explain the direct edges between two nodes: type, provenance, reason.

    Matches ``source`` and ``target`` by path/id/name and returns every direct
    edge in either direction. Each row carries ``edge_type``, an ``outbound``
    flag (True when the edge runs source→target), the edge ``origin``
    (inferred / structural / manual), its ``confidence`` (0.0-1.0 on inferred
    edges), and the inferrer's ``reason`` string — the populated fields that
    justify why the link exists. Returns ``[]`` when the two nodes share no
    direct edge.
    """
    cypher = """
    MATCH (a)-[r]-(b)
    WHERE (a.path = $source OR a.id = $source OR a.name = $source)
      AND (b.path = $target OR b.id = $target OR b.name = $target)
    RETURN
      coalesce(a.path, a.id, a.name) AS source,
      coalesce(b.path, b.id, b.name) AS target,
      type(r) AS edge_type,
      startNode(r) = a AS outbound,
      r.origin AS origin,
      r.confidence AS confidence,
      r.reason AS reason
    """
    return store.exec_read(cypher, {"source": source, "target": target})


def ticket_dossier(store: GraphStore, ticket_id: str) -> dict[str, Any]:
    """Assemble a ticket's whole neighborhood in one call.

    Collapses the multi-call manual traversal the start-of-task workflow
    otherwise needs. Returns the ticket's own properties plus every directly
    connected node, each annotated with the connecting ``edge_type``, the
    ``direction`` relative to the ticket, and the neighbor's ``summary`` /
    ``title`` — the File and Section summaries are the real content, since the
    Ticket node itself is typically a thin identifier. ``found`` is ``False``
    (and ``neighbors`` empty) when no such ticket exists.
    """
    rows = store.exec_read(
        """
        MATCH (t:Ticket)
        WHERE t.id = $id OR t.name = $id
        WITH t LIMIT 1
        OPTIONAL MATCH (t)-[r]-(n)
        RETURN
          coalesce(t.id, t.name) AS ticket,
          properties(t) AS ticket_props,
          type(r) AS edge_type,
          startNode(r) = t AS outbound,
          labels(n) AS neighbor_labels,
          coalesce(n.path, n.id, n.name) AS neighbor,
          n.summary AS neighbor_summary,
          n.title AS neighbor_title
        """,
        {"id": ticket_id},
    )
    if not rows:
        return {"ticket": ticket_id, "found": False, "properties": {}, "neighbors": []}
    neighbors: list[dict[str, Any]] = []
    for row in rows:
        # OPTIONAL MATCH yields one all-null neighbor row for a ticket with no
        # edges; skip it so an isolated ticket returns an empty neighbor list.
        if row["edge_type"] is None:
            continue
        neighbors.append(
            {
                "edge_type": row["edge_type"],
                "direction": "outbound" if row["outbound"] else "inbound",
                "labels": row["neighbor_labels"],
                "node": row["neighbor"],
                "summary": row["neighbor_summary"],
                "title": row["neighbor_title"],
            }
        )
    return {
        "ticket": rows[0]["ticket"],
        "found": True,
        "properties": _node_without_embedding(rows[0]["ticket_props"] or {}),
        "neighbors": neighbors,
    }


def find_reusable(store: GraphStore, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """Return reusable Artifact nodes ranked by full-text relevance to ``query``.

    Serves the reuse discipline (check for an existing artifact before creating
    a new one). Full-text-searches ``Artifact.description`` and keeps only the
    artifacts flagged ``reusable = true``. Requires entity content extraction to
    have populated ``Artifact.description`` / ``reusable``; a graph indexed
    before that returns ``[]``.
    """
    rows = store.full_text_search("Artifact", "description", query, k=max(limit, _DEFAULT_FETCH_K))
    results: list[dict[str, Any]] = []
    for row in rows:
        node = row["node"]
        if node.get("reusable") is True:
            results.append(flatten_row(node, row["score"]))
        if len(results) >= limit:
            break
    return results


def list_entities(
    store: GraphStore,
    kind: str,
    *,
    prop: str | None = None,
    value: str | None = None,
    corpus: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List nodes of an entity ``kind`` with their properties (embedding stripped).

    Exploits the typed entity layer — e.g. all ``Integration`` nodes, or
    ``Ticket`` nodes filtered by an equality predicate. ``kind`` must be a
    declared ontology node type, and ``prop`` (when given) must be a declared
    property of that type; both are validated against the ontology before being
    interpolated into the query, so neither can inject. ``value`` and ``corpus``
    are bound as parameters. Raises ``ValueError`` on an unknown kind/property.
    """
    if kind not in _ALLOWED_LABELS:
        raise ValueError(f"Unknown entity kind: {kind!r}")
    filters: list[str] = []
    params: dict[str, Any] = {}
    if corpus is not None:
        filters.append("n.corpus = $corpus")
        params["corpus"] = corpus
    if prop is not None:
        if prop not in _ONTOLOGY.node_types.get(kind, ()):
            raise ValueError(f"Unknown property {prop!r} for kind {kind!r}")
        filters.append(f"n.{prop} = $value")
        params["value"] = value
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    cypher = f"""
    MATCH (n:{kind})
    {where}
    RETURN labels(n) AS labels, properties(n) AS props
    LIMIT {int(limit)}
    """
    rows = store.exec_read(cypher, params)
    return [{"labels": row["labels"], **_node_without_embedding(row["props"])} for row in rows]


# Edge types that signal a node may be stale or in conflict. Read directly off
# the graph; the values are unvalidated inferences, so callers should treat a
# hit as a prompt to check rather than proof.
_FRESHNESS_EDGE_TYPES: Final[list[str]] = ["SUPERSEDES", "CONTRADICTS", "NEEDS_UPDATE"]


def check_freshness(
    store: GraphStore,
    *,
    node_id: str | None = None,
    corpus: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return freshness-signalling edges (SUPERSEDES / CONTRADICTS / NEEDS_UPDATE).

    Scope by ``node_id`` (edges incident to one node, either direction) or by
    ``corpus`` (every such edge with an endpoint in the corpus). Each row gives
    ``source``/``target`` identities, the ``edge_type``, and the edge
    ``origin``/``confidence``/``reason`` so a caller can judge whether a hit is
    still true before acting on it. Raises ``ValueError`` if neither scope is
    given. The underlying edges are unvalidated inferences and are typically
    few; an empty result means none were inferred, not that the node is
    definitively current.
    """
    if not node_id and not corpus:
        raise ValueError("check_freshness requires node_id or corpus")
    params: dict[str, Any] = {"types": _FRESHNESS_EDGE_TYPES}
    if node_id:
        scope = (
            "(a.path = $id OR a.id = $id OR a.name = $id "
            "OR b.path = $id OR b.id = $id OR b.name = $id)"
        )
        params["id"] = node_id
    else:
        scope = "(a.corpus = $corpus OR b.corpus = $corpus)"
        params["corpus"] = corpus
    cypher = f"""
    MATCH (a)-[r]->(b)
    WHERE type(r) IN $types AND {scope}
    RETURN coalesce(a.path, a.id, a.name) AS source,
           coalesce(b.path, b.id, b.name) AS target,
           type(r) AS edge_type,
           r.origin AS origin,
           r.confidence AS confidence,
           r.reason AS reason
    LIMIT {int(limit)}
    """
    return store.exec_read(cypher, params)


def find_contradictions(
    store: GraphStore, topic: str | None = None, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Return CONTRADICTS edge pairs, optionally narrowed by ``topic``.

    Surfaces conflicting guidance so a caller can reconcile it rather than
    following the first match. When ``topic`` is given, only pairs where either
    endpoint's summary contains the topic (case-insensitive) are returned. Each
    row carries both endpoints' identities and summaries plus the edge
    ``confidence``/``reason``. This edge type is sparse in practice, so an empty
    result is common and expected.
    """
    params: dict[str, Any] = {}
    where = ""
    if topic:
        params["topic"] = topic
        where = (
            "WHERE toLower(coalesce(a.summary, '')) CONTAINS toLower($topic) "
            "OR toLower(coalesce(b.summary, '')) CONTAINS toLower($topic)"
        )
    cypher = f"""
    MATCH (a)-[r:CONTRADICTS]->(b)
    {where}
    RETURN coalesce(a.path, a.id, a.name) AS source,
           coalesce(b.path, b.id, b.name) AS target,
           a.summary AS source_summary,
           b.summary AS target_summary,
           r.confidence AS confidence,
           r.reason AS reason
    LIMIT {int(limit)}
    """
    return store.exec_read(cypher, params)
