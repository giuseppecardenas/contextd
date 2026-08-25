"""Ranker execution for hybrid search.

One query may fan out to several rankers: a vector leg and a full-text leg
per chunk profile (``fine``, ``coarse``, ...), each weighted by the
profile's configured ``weight`` times the global modality weights. This
module runs them and hands the ``(rows, weight)`` pairs to
:func:`contextd.search.fusion.fuse_rankers`; it never ranks by itself.

The embedding call happens at most once per query. Any failure on the
vector side (embedding API, vector index) degrades that leg to nothing and
lets the full-text legs stand — search must never error because an
embedder is flaky.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from contextd.providers.base import EmbeddingProvider
from contextd.storage.base import GraphStore

_log = logging.getLogger(__name__)

SearchMode = Literal["hybrid", "fulltext", "vector"]


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    weight: float = 1.0


@dataclass
class RankerRun:
    rankers: list[tuple[list[dict[str, Any]], float]]
    """``(rows, weight)`` per ranker, best-first rows."""
    used_vector: bool
    """Whether at least one vector leg produced rows (or ran without error)."""


def _search_kwargs(filters: dict[str, Any]) -> dict[str, Any]:
    # Only pass ``filters`` when there is something to filter on so callers
    # (and existing tests) that never filter see the plain positional call.
    return {"filters": filters} if filters else {}


def run_rankers(
    store: GraphStore,
    query: str,
    *,
    label: str,
    search_prop: str,
    mode: SearchMode,
    fetch_k: int,
    embedder: EmbeddingProvider | None,
    vector_capable: bool,
    filters: dict[str, Any] | None = None,
    profiles: list[ProfileSpec] | None = None,
    vector_weight: float = 1.0,
    fulltext_weight: float = 1.0,
) -> RankerRun:
    """Run the vector / full-text rankers for ``label``.

    ``profiles`` (Chunk searches only) adds a ``profile`` equality filter per
    entry and scales that entry's rankers by ``ProfileSpec.weight``; ``None``
    runs one unfiltered pair, which covers every profile present.
    """
    base_filters = dict(filters or {})
    want_vector = mode in ("hybrid", "vector") and embedder is not None and vector_capable
    query_vec: list[float] | None = None
    if want_vector:
        assert embedder is not None
        try:
            query_vec = embedder.embed([query])[0]
        except Exception as exc:
            _log.warning(
                "search: query embedding failed (%s: %s); full-text only", type(exc).__name__, exc
            )
            want_vector = False

    specs: list[ProfileSpec | None] = list(profiles) if profiles else [None]
    rankers: list[tuple[list[dict[str, Any]], float]] = []
    used_vector = False
    for spec in specs:
        f = dict(base_filters)
        scale = 1.0
        if spec is not None:
            f["profile"] = spec.name
            scale = spec.weight
        if want_vector and query_vec is not None:
            try:
                rows = store.vector_search(
                    label, "embedding", query_vec, k=fetch_k, **_search_kwargs(f)
                )
                rankers.append((rows, vector_weight * scale))
                used_vector = True
            except Exception as exc:
                _log.warning(
                    "search: vector leg failed for %s (%s: %s)", label, type(exc).__name__, exc
                )
        if mode != "vector":
            try:
                rows = store.full_text_search(
                    label, search_prop, query, k=fetch_k, **_search_kwargs(f)
                )
                rankers.append((rows, fulltext_weight * scale))
            except Exception as exc:
                _log.warning(
                    "search: full-text leg failed for %s (%s: %s)", label, type(exc).__name__, exc
                )
    return RankerRun(rankers=rankers, used_vector=used_vector)
