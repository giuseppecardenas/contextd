"""Drive the ``search`` tool over a bench spec and score every query.

One :func:`run_bench` call is one *configuration* (profile set, return
unit, k); the CLI runs it once per ``--profiles`` value and tabulates the
reports side by side. Each result row is reduced to a
:class:`~contextd.bench.metrics.Target` — path, section anchor, line range
— so the pure metrics can compare it against the spec's expectations.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from contextd.bench.metrics import (
    QueryScore,
    Target,
    line_iou,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise,
)
from contextd.bench.spec import BenchSpec
from contextd.config import SearchConfig
from contextd.mcp.tools import search
from contextd.providers.base import EmbeddingProvider
from contextd.search.collapse import ReturnUnit
from contextd.storage.base import GraphStore


@dataclass
class BenchReport:
    """One configuration's scores, JSON-serialisable via :meth:`to_dict`."""

    config: dict[str, Any]
    scores: list[QueryScore]
    summary: dict[str, float | None]
    latencies_ms: list[float] = field(default_factory=list)
    """Wall-clock per query, index-aligned with ``scores``."""

    @property
    def label(self) -> str:
        """Short human name for table rows: ``fine,coarse/auto@5``."""
        profiles = self.config.get("profiles")
        names = ",".join(profiles) if profiles else "all"
        return f"{names}/{self.config.get('return_unit', '?')}@{self.config.get('k', '?')}"

    def to_dict(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for i, s in enumerate(self.scores):
            row = asdict(s)
            if i < len(self.latencies_ms):
                row["latency_ms"] = self.latencies_ms[i]
            rows.append(row)
        return {"config": dict(self.config), "scores": rows, "summary": dict(self.summary)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchReport:
        scores: list[QueryScore] = []
        latencies: list[float] = []
        for row in data.get("scores", []):
            scores.append(
                QueryScore(
                    query=str(row["query"]),
                    recall=float(row["recall"]),
                    precision=float(row["precision"]),
                    mrr=float(row["mrr"]),
                    iou=None if row.get("iou") is None else float(row["iou"]),
                )
            )
            latencies.append(float(row.get("latency_ms", 0.0)))
        return cls(
            config=dict(data.get("config", {})),
            scores=scores,
            summary=dict(data.get("summary", {})),
            latencies_ms=latencies,
        )


def _anchor_from_id(node_id: Any) -> str | None:
    """The part after ``#`` in a Section id (``path#anchor``); ``None`` otherwise."""
    if not isinstance(node_id, str) or "#" not in node_id:
        return None
    return node_id.split("#", 1)[1] or None


def row_to_target(row: dict[str, Any]) -> Target:
    """Reduce one ``search`` result row to the metrics' ``Target`` shape.

    * ``section`` rows carry their anchor in ``id`` (``path#anchor``).
    * ``chunk`` rows inherit it from ``parent_id`` when the parent is a
      Section; a File-parented chunk has no anchor.
    * ``file`` rows (and any flat node row) have no anchor.

    Lines come from ``evidence.start_line`` / ``end_line`` when both are
    present — for a collapsed parent that is the best chunk's span, which is
    the evidence an assistant would actually read.
    """
    unit = row.get("unit")
    anchor: str | None = None
    if unit == "section":
        anchor = _anchor_from_id(row.get("id"))
    elif unit == "chunk":
        anchor = _anchor_from_id(row.get("parent_id"))

    lines: tuple[int, int] | None = None
    evidence = row.get("evidence")
    if isinstance(evidence, dict):
        start, end = evidence.get("start_line"), evidence.get("end_line")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            lines = (start, end)

    return Target(path=str(row.get("path") or ""), anchor=anchor, lines=lines)


def run_bench(
    store: GraphStore,
    spec: BenchSpec,
    *,
    embedder: EmbeddingProvider | None,
    search_cfg: SearchConfig,
    corpus: str | None,
    profiles: list[str] | None,
    return_unit: ReturnUnit,
    k: int,
    profile_weights: dict[str, float] | None = None,
) -> BenchReport:
    """Score every query in ``spec`` for one configuration.

    Calls :func:`contextd.mcp.tools.search` per query with ``limit`` set to
    the query's own ``k`` when it overrides the run's. Ranking knobs
    (``mode``, ``rrf_k``, ``fetch_k``, weights, ``auto_merge_threshold``)
    come from ``search_cfg`` exactly as the MCP server passes them, so a
    bench run measures what an assistant would get. Neighbour context
    (``window``) is forced to 0: it never affects a score and would only
    add per-row queries to the measured latency.
    """
    scores: list[QueryScore] = []
    latencies: list[float] = []
    for query in spec.queries:
        top_k = query.k or k
        started = time.perf_counter()
        rows = search(
            store,
            query.q,
            limit=top_k,
            embedder=embedder,
            mode=search_cfg.mode,
            rrf_k=search_cfg.rrf_k,
            fetch_k=search_cfg.fetch_k,
            vector_weight=search_cfg.vector_weight,
            fulltext_weight=search_cfg.fulltext_weight,
            corpus=corpus,
            profiles=profiles,
            profile_weights=profile_weights,
            return_unit=return_unit,
            auto_merge_threshold=search_cfg.auto_merge_threshold,
            window=0,
            max_evidence_chars=search_cfg.max_evidence_chars,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        hits = [row_to_target(r) for r in rows]
        scores.append(
            QueryScore(
                query=query.q,
                recall=recall_at_k(hits, query.expect, top_k),
                precision=precision_at_k(hits, query.expect, top_k),
                mrr=reciprocal_rank(hits, query.expect),
                iou=line_iou(hits, query.expect, top_k),
            )
        )

    summary: dict[str, float | None] = dict(summarise(scores))
    summary["latency_ms"] = (sum(latencies) / len(latencies)) if latencies else 0.0
    config: dict[str, Any] = {
        "corpus": corpus,
        "profiles": list(profiles) if profiles else None,
        "return_unit": return_unit,
        "k": k,
        "mode": search_cfg.mode,
        "embedder": embedder is not None,
    }
    return BenchReport(config=config, scores=scores, summary=summary, latencies_ms=latencies)
