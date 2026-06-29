"""Reciprocal Rank Fusion (RRF) for hybrid search.

Hybrid search fuses two ranked result lists — one from vector (embedding)
similarity, one from full-text (BM25) — into a single relevance ordering.
The hard part is that the two rankers' scores are not comparable as
numbers: Neo4j's vector index returns normalised cosine similarity in
``[0, 1]`` while its full-text index returns unbounded Lucene BM25. Any
fusion that combines the raw scores has to invent a fragile normalisation.

RRF sidesteps that entirely by combining *ranks*, not scores. A node's
fused score is the sum over each ranker of ``weight / (rrf_k + rank)``,
where ``rank`` is the node's 1-based position in that ranker's list and a
node absent from a ranker contributes nothing from it. ``rrf_k`` (default
60, the value from the original RRF paper) dampens the weight of the very
top ranks so that agreement across both rankers, rather than a single
ranker's #1, drives the result. Optional per-modality weights let a caller
bias semantic vs lexical without changing the algorithm.

This module is deliberately backend-agnostic: it operates only on the
``[{"node": {...}, "score": float}]`` row shape that both
``GraphStore.vector_search`` and ``GraphStore.full_text_search`` return,
and imports nothing from ``contextd.storage.<backend>``.
"""

from __future__ import annotations

from typing import Any, Final

from contextd.storage._keys import primary_key_for

_STRIP_FIELDS: Final[frozenset[str]] = frozenset({"embedding"})


def flatten_row(node: dict[str, Any], score: float) -> dict[str, Any]:
    """Flatten a node's properties onto a result row, dropping ``embedding``.

    The 1024-float embedding vector is roughly 12 KB per row and would blow
    past an MCP client's per-result token budget, so it is never returned to
    the caller. The supplied ``score`` (an RRF fused score, or a raw
    relevance score on the full-text-only fallback path) is attached as
    ``score``. This is the single place the result wire-shape is built, so
    the fusion path and the fallback path stay identical.

    :param node: the node's property dict as returned by a backend search.
    :param score: the relevance/fused score to attach to the row.
    :return: a flat dict of the node's properties (minus ``embedding``) plus
        a ``score`` key.
    """
    row = {k: v for k, v in node.items() if k not in _STRIP_FIELDS}
    row["score"] = score
    return row


def reciprocal_rank_fusion(
    vector_rows: list[dict[str, Any]],
    fulltext_rows: list[dict[str, Any]],
    *,
    label: str,
    limit: int,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    fulltext_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Fuse vector and full-text result lists into one ranking via RRF.

    Both inputs are the raw ``{"node": {...}, "score": float}`` rows from
    the backend search methods, each already ordered best-first. Nodes are
    identified across the two lists by their label's primary key
    (``primary_key_for(label)`` — ``path`` for File, ``id`` for Section,
    etc.); a node appearing in both lists has its per-ranker contributions
    summed.

    The vector list is folded before the full-text list, and ties in the
    fused score are broken by first-seen insertion order. Both choices make
    the output fully deterministic: the ordering depends only on list
    positions, never on floating-point score values, which is the property
    that makes RRF immune to the cross-method score-incomparability problem
    described in the module docstring.

    A node whose properties omit the label's primary key is skipped rather
    than raising — a defensive guard for malformed rows; it cannot be keyed
    for fusion and would otherwise collide under a ``None`` key.

    :param vector_rows: vector-search result rows (best-first), possibly empty.
    :param fulltext_rows: full-text-search result rows (best-first), possibly empty.
    :param label: node label whose primary key identifies nodes for fusion.
    :param limit: maximum number of fused rows to return.
    :param rrf_k: RRF constant; larger values flatten the top-rank weighting.
    :param vector_weight: multiplier on the vector ranker's contribution.
    :param fulltext_weight: multiplier on the full-text ranker's contribution.
    :return: fused rows in the flattened ``flatten_row`` shape, ordered by
        descending fused score and truncated to ``limit``.
    """
    key_prop = primary_key_for(label)
    scores: dict[Any, float] = {}
    nodes: dict[Any, dict[str, Any]] = {}
    first_seen: dict[Any, int] = {}

    def _fold(rows: list[dict[str, Any]], weight: float) -> None:
        for rank, row in enumerate(rows, start=1):
            node = row.get("node")
            if not isinstance(node, dict):
                continue
            key = node.get(key_prop)
            if key is None:
                continue
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            if key not in nodes:
                nodes[key] = node
                first_seen[key] = len(first_seen)

    _fold(vector_rows, vector_weight)
    _fold(fulltext_rows, fulltext_weight)

    ranked = sorted(scores, key=lambda k: (-scores[k], first_seen[k]))
    return [flatten_row(nodes[k], scores[k]) for k in ranked[:limit]]
