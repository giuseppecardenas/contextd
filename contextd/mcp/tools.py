"""MCP tool implementations — each is a thin wrapper over the GraphStore."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from contextd.mcp.readonly_guard import assert_read_only
from contextd.ontology.schema import Ontology
from contextd.providers.base import EmbeddingProvider
from contextd.search.collapse import ReturnUnit, collapse
from contextd.search.expand import expand_chunk as _expand_chunk
from contextd.search.fusion import flatten_row, fuse_rankers
from contextd.search.retrieve import ProfileSpec, run_rankers
from contextd.storage.base import GraphStore

if TYPE_CHECKING:
    from contextd.inference.translate import QueryTranslator

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


_VECTOR_CAPABLE_LABELS: Final[frozenset[str]] = frozenset({"File", "Section", "Chunk", "Topic"})
"""Labels that carry BOTH a vector index and a full-text index and can
therefore be searched hybridly. File and Section get both indexes from the
Neo4j baseline + section-fulltext migrations, Chunk and Topic from _0008;
every other label is full-text only (Artifact/Ticket/Pattern/Risk via the
entity-content migrations) or neither (Technology, Client, …) and degrades
to full-text.

This is the third place index coverage is encoded — the migration DDL and
``contextd/storage/_keys.py`` are the others — and must change in lock-step
with any future vector-index migration."""

_SEARCH_PROPERTY_BY_LABEL: Final[dict[str, str]] = {
    "File": "summary",
    "Section": "summary",
    "Chunk": "text",
    "Topic": "summary",
    "Artifact": "description",
    "Ticket": "title",
    "Pattern": "description",
    "Risk": "description",
}
"""Full-text property searched per label. File/Section carry AI summaries;
Chunk is searched on its raw ``text`` (the ``Chunk_text_ft`` index also
covers ``prefix`` and ``keywords``); entity types carry their declared
content field (populated by the indexer and indexed by the baseline/_0006
full-text migrations). Labels absent here fall back to ``summary`` — which
only matches if such an index exists for them."""


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
    corpus: str | None = None,
    profiles: list[str] | None = None,
    profile_weights: dict[str, float] | None = None,
    return_unit: ReturnUnit = "auto",
    auto_merge_threshold: float = 0.5,
    window: int = 1,
    max_evidence_chars: int = 1200,
) -> list[dict[str, Any]]:
    """Hybrid search, fusing vector + full-text rankers via RRF.

    The default ``kind`` is ``Chunk``: retrieval chunks are ranked (one
    vector + one full-text ranker per requested profile, weighted by the
    profile's configured weight) and the fused hits are collapsed
    small-to-big to the best enclosing unit per ``return_unit`` (see
    :func:`contextd.search.collapse.collapse`), each row carrying an
    ``evidence`` block with the best chunk's text, line range and neighbour
    context. Any other ``kind`` keeps the flat node-row shape.

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
    :param corpus: restrict every ranker to one corpus.
    :param profiles: chunk profiles to query (``Chunk`` only); ``None`` queries
        every profile present through one unfiltered ranker pair.
    :param profile_weights: per-profile RRF scale (from the corpus config).
    :param return_unit: ``chunk`` | ``section`` | ``file`` | ``auto``.
    :param auto_merge_threshold: hit ratio at which ``auto`` returns the parent.
    :param window: neighbour chunks attached as evidence context per side.
    :param max_evidence_chars: evidence text truncation.
    :return: result rows, best-first, at most ``limit`` of them.
    """
    label = kind or "Chunk"
    fetch = max(fetch_k or _DEFAULT_FETCH_K, limit)
    filters: dict[str, Any] = {"corpus": corpus} if corpus else {}
    specs: list[ProfileSpec] | None = None
    if label == "Chunk" and profiles:
        weights = profile_weights or {}
        specs = [ProfileSpec(name, weights.get(name, 1.0)) for name in profiles]

    run = run_rankers(
        store,
        query,
        label=label,
        search_prop=_search_property(label),
        mode=mode,
        fetch_k=fetch,
        embedder=embedder,
        vector_capable=label in _VECTOR_CAPABLE_LABELS,
        filters=filters,
        profiles=specs,
        vector_weight=vector_weight,
        fulltext_weight=fulltext_weight,
    )
    if mode == "vector" and not run.used_vector:
        # The caller explicitly asked for vectors and got none: say so rather
        # than silently handing back lexical results.
        return []
    if not run.rankers:
        return []
    # The fused depth for chunks is the fetch depth: collapse decides the
    # final ``limit`` after grouping hits by parent.
    depth = fetch if label == "Chunk" else limit
    if len(run.rankers) == 1 and not run.used_vector:
        # Single full-text ranker: keep the backend's raw relevance score, as
        # the fulltext-mode contract documents.
        rows_only, _ = run.rankers[0]
        fused = [flatten_row(r["node"], r["score"]) for r in rows_only[:depth]]
    else:
        fused = fuse_rankers(run.rankers, label=label, limit=depth, rrf_k=rrf_k)
    if label != "Chunk":
        return fused[:limit]
    return collapse(
        store,
        fused,
        return_unit=return_unit,
        auto_merge_threshold=auto_merge_threshold,
        limit=limit,
        max_evidence_chars=max_evidence_chars,
        window=window,
    )


def expand_chunk(store: GraphStore, chunk_id: str, *, window: int = 2) -> dict[str, Any] | None:
    """A chunk with its ``window`` neighbours on each side and the parent's summary."""
    return _expand_chunk(store, chunk_id, window=max(0, min(int(window), 10)))


def topics(
    store: GraphStore,
    *,
    corpus: str | None = None,
    query: str | None = None,
    layer: int | None = None,
    limit: int = 20,
    embedder: EmbeddingProvider | None = None,
    mode: Literal["hybrid", "fulltext", "vector"] = "hybrid",
    rrf_k: int = 60,
    fetch_k: int | None = None,
    members_per_topic: int = 25,
) -> list[dict[str, Any]]:
    """List cross-document topics (RAPTOR-style cluster summaries) with members.

    With ``query`` the topics are ranked by hybrid search over their
    summaries; without it they are listed by layer then member count.
    """
    filters: dict[str, Any] = {}
    if corpus:
        filters["corpus"] = corpus
    if layer is not None:
        filters["layer"] = int(layer)
    if query:
        run = run_rankers(
            store,
            query,
            label="Topic",
            search_prop="summary",
            mode=mode,
            fetch_k=max(fetch_k or _DEFAULT_FETCH_K, limit),
            embedder=embedder,
            vector_capable=True,
            filters=filters,
        )
        rows = fuse_rankers(run.rankers, label="Topic", limit=limit, rrf_k=rrf_k)
    else:
        conditions = [f"t.{k} = ${k}" for k in filters]
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = store.exec_read(
            f"MATCH (t:Topic) {where} "
            "RETURN t.id AS id, t.corpus AS corpus, t.layer AS layer, t.title AS title, "
            "t.summary AS summary, t.member_count AS member_count "
            f"ORDER BY t.layer, t.member_count DESC LIMIT {int(limit)}",
            filters,
        )
    for row in rows:
        members = store.exec_read(
            "MATCH (m)-[r:BELONGS_TO]->(t:Topic {id: $id}) "
            "RETURN coalesce(m.id, m.path) AS id, labels(m) AS labels, "
            "coalesce(m.title, m.name) AS title, r.probability AS probability "
            f"ORDER BY r.probability DESC LIMIT {int(members_per_topic)}",
            {"id": row.get("id")},
        )
        row["members"] = members
    return rows


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
    OPTIONAL MATCH (s)-[:CONTAINS]->(c:Chunk)
    RETURN s.id AS id, s.title AS title, s.level AS level,
           s.ordinal AS ordinal, s.summary AS summary, count(c) AS chunk_count
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


def whats_new(
    store: GraphStore, since: str, *, corpus: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Return nodes changed at or after ``since``, newest first.

    ``since`` is an ISO-8601 timestamp; it is parsed by Neo4j's ``datetime()``
    and compared against the node ``updated`` stamp the indexer writes at
    enumerate time. Only File and Section nodes carry ``updated`` (they are the
    on-disk-backed, content-bearing nodes), so this reports changed source
    documents — which is what a caller catching up on an evolving corpus wants.
    Optionally scoped to a ``corpus``. Returns ``[]`` on a graph indexed before
    the ``updated`` stamp existed (re-bootstrap to backfill).
    """
    filters = ["n.updated IS NOT NULL", "n.updated >= datetime($since)"]
    params: dict[str, Any] = {"since": since}
    if corpus is not None:
        filters.append("n.corpus = $corpus")
        params["corpus"] = corpus
    where = "WHERE " + " AND ".join(filters)
    cypher = f"""
    MATCH (n)
    {where}
    RETURN coalesce(n.path, n.id, n.name) AS node,
           labels(n) AS labels,
           n.summary AS summary,
           n.updated AS updated
    ORDER BY n.updated DESC
    LIMIT {int(limit)}
    """
    return store.exec_read(cypher, params)


def timeline(
    store: GraphStore,
    node_id: str | None = None,
    topic: str | None = None,
    *,
    limit: int = 50,
) -> dict[str, Any]:
    """Chronological view of nodes relevant to an anchor, with SUPERSEDES chains.

    Anchor by ``node_id`` (the node and its direct neighbors) or by ``topic``
    (nodes whose summary contains the topic, case-insensitive). Returns
    ``nodes`` ordered newest-first by ``updated`` (falling back to
    ``inferred_at``), plus ``supersedes`` — the SUPERSEDES edges in scope,
    each as a newer→older pair — so the caller sees how a decision evolved
    rather than a flat neighbor set. Raises ``ValueError`` if neither anchor is
    given.
    """
    if not node_id and not topic:
        raise ValueError("timeline requires node_id or topic")
    params: dict[str, Any] = {}
    if node_id:
        params["id"] = node_id
        node_match = (
            "MATCH (anchor) WHERE anchor.path = $id OR anchor.id = $id OR anchor.name = $id "
            "WITH anchor LIMIT 1 "
            "MATCH (anchor)-[]-(n) WITH DISTINCT n"
        )
        sup_scope = (
            "(a.path = $id OR a.id = $id OR a.name = $id "
            "OR b.path = $id OR b.id = $id OR b.name = $id)"
        )
    else:
        params["topic"] = topic
        node_match = (
            "MATCH (n) WHERE toLower(coalesce(n.summary, '')) CONTAINS toLower($topic) "
            "WITH DISTINCT n"
        )
        sup_scope = (
            "(toLower(coalesce(a.summary, '')) CONTAINS toLower($topic) "
            "OR toLower(coalesce(b.summary, '')) CONTAINS toLower($topic))"
        )
    nodes_cypher = f"""
    {node_match}
    WHERE n.updated IS NOT NULL OR n.inferred_at IS NOT NULL
    RETURN coalesce(n.path, n.id, n.name) AS node,
           labels(n) AS labels,
           n.summary AS summary,
           n.updated AS updated,
           n.inferred_at AS inferred_at
    ORDER BY coalesce(n.updated, n.inferred_at) DESC
    LIMIT {int(limit)}
    """
    supersedes_cypher = f"""
    MATCH (a)-[r:SUPERSEDES]->(b)
    WHERE {sup_scope}
    RETURN coalesce(a.path, a.id, a.name) AS newer,
           coalesce(b.path, b.id, b.name) AS older,
           r.confidence AS confidence,
           r.reason AS reason
    LIMIT {int(limit)}
    """
    return {
        "nodes": store.exec_read(nodes_cypher, params),
        "supersedes": store.exec_read(supersedes_cypher, params),
    }


def ask(
    store: GraphStore,
    translator: QueryTranslator | None,
    question: str,
    *,
    corpus: str | None = None,
) -> dict[str, Any]:
    """Answer a natural-language question by translating it to Cypher and running it.

    Reuses the same NL→Cypher translator as the CLI ``ask`` command. Returns
    both the generated ``cypher`` and the resulting ``rows`` so the caller can
    inspect what ran and always has the node-level path underneath the answer
    (this tool must never be the only route to the underlying nodes). The
    translator applies the read-only guard to its own output before this runs,
    so the query is guaranteed read-only. Raises ``ValueError`` when no
    inference provider is configured (``translator is None``), and surfaces the
    translator's own errors (e.g. a missing ``prompts/translate`` template) to
    the caller.
    """
    if translator is None:
        raise ValueError("ask is unavailable: no inference provider is configured")
    cypher = translator.translate(question, corpus=corpus)
    return {"cypher": cypher, "rows": store.exec_read(cypher, {})}


def grep_corpus(
    home: Path,
    pattern: str,
    *,
    corpus: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Regex search over corpus file *contents* on disk, scoped to a corpus.

    The graph stores summaries and metadata, not file bodies, so exact strings
    a summary paraphrases away (a flag name, an id, a config key) are only
    findable by reading the source files. This walks the corpus's declared
    include/exclude globs (reusing the indexer's ``enumerate_corpus_files``) and
    returns up to ``limit`` line matches as ``{corpus, path, line, text}``.
    ``corpus`` selects one registered corpus; when omitted, every registered
    corpus is searched. Invalid regex raises ``re.error`` to the caller.
    """
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.pipeline import enumerate_corpus_files

    regex = re.compile(pattern)
    corpora_dir = home / "corpora"
    toml_paths = (
        [corpora_dir / f"{corpus}.toml"]
        if corpus is not None
        else sorted(corpora_dir.glob("*.toml"))
    )
    matches: list[dict[str, Any]] = []
    for toml_path in toml_paths:
        if not toml_path.exists():
            continue
        cfg = CorpusConfig.load(toml_path)
        for file_path in enumerate_corpus_files(cfg):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(
                        {
                            "corpus": cfg.corpus.name,
                            "path": str(file_path),
                            "line": line_no,
                            "text": line.strip()[:400],
                        }
                    )
                    if len(matches) >= limit:
                        return matches
    return matches
