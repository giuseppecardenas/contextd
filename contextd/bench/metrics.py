"""Pure retrieval metrics.

An *expectation* names a file path and optionally a section anchor and a
line range; a *hit* is one result row reduced to the same shape. A hit
satisfies an expectation when the paths match and, where the expectation
narrows further, the anchor matches or the line ranges overlap.

* ``recall_at_k`` — fraction of expectations satisfied by the top-k hits.
* ``precision_at_k`` — fraction of the top-k hits that satisfy some
  expectation.
* ``reciprocal_rank`` — 1 / rank of the first satisfying hit (0 if none).
* ``line_iou`` — Chroma-style token IoU approximated at line granularity:
  |expected AND returned| / |expected OR returned| over the line sets of hits
  and expectations that carry line ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextd.indexer.heading_parser import anchor_key


@dataclass(frozen=True)
class Target:
    path: str
    anchor: str | None = None
    lines: tuple[int, int] | None = None
    """``(start, end)`` inclusive-exclusive, 0-based, like ``ChunkSpan``."""

    def line_set(self) -> set[int]:
        if self.lines is None:
            return set()
        return set(range(self.lines[0], self.lines[1]))


def _same_path(a: str, b: str) -> bool:
    a2, b2 = a.replace("\\", "/"), b.replace("\\", "/")
    return a2 == b2 or a2.endswith("/" + b2) or b2.endswith("/" + a2)


def satisfies(hit: Target, expected: Target) -> bool:
    if not _same_path(hit.path, expected.path):
        return False
    if (
        expected.anchor is not None
        and hit.anchor is not None
        and anchor_key(hit.anchor) != anchor_key(expected.anchor)
    ):
        # Modulo hyphen runs: a spec written against pre-fix ``lod-1-lod-2``
        # ids keeps scoring after the corpus re-indexes to GitHub's
        # ``lod-1--lod-2`` form, and vice versa.
        return False
    if expected.lines is not None and hit.lines is not None:
        return bool(hit.line_set() & expected.line_set())
    return True


def recall_at_k(hits: list[Target], expected: list[Target], k: int) -> float:
    if not expected:
        return 1.0
    top = hits[:k]
    found = sum(1 for e in expected if any(satisfies(h, e) for h in top))
    return found / len(expected)


def precision_at_k(hits: list[Target], expected: list[Target], k: int) -> float:
    top = hits[:k]
    if not top:
        return 0.0
    good = sum(1 for h in top if any(satisfies(h, e) for e in expected))
    return good / len(top)


def reciprocal_rank(hits: list[Target], expected: list[Target]) -> float:
    for rank, h in enumerate(hits, start=1):
        if any(satisfies(h, e) for e in expected):
            return 1.0 / rank
    return 0.0


def line_iou(hits: list[Target], expected: list[Target], k: int) -> float | None:
    """``None`` when neither side carries line ranges (metric not applicable)."""
    exp_lines = {(e.path, ln) for e in expected if e.lines for ln in e.line_set()}
    hit_lines = {(h.path, ln) for h in hits[:k] if h.lines for ln in h.line_set()}
    if not exp_lines and not hit_lines:
        return None
    # Paths may be relative on one side; normalise via the hit that matched.
    norm_exp = {(_key(p), ln) for p, ln in exp_lines}
    norm_hit = {(_key(p), ln) for p, ln in hit_lines}
    union = norm_exp | norm_hit
    return len(norm_exp & norm_hit) / len(union) if union else None


def _key(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


@dataclass
class QueryScore:
    query: str
    recall: float
    precision: float
    mrr: float
    iou: float | None


def summarise(scores: list[QueryScore]) -> dict[str, float | None]:
    if not scores:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "iou": None, "queries": 0}
    ious = [s.iou for s in scores if s.iou is not None]
    return {
        "recall": sum(s.recall for s in scores) / len(scores),
        "precision": sum(s.precision for s in scores) / len(scores),
        "mrr": sum(s.mrr for s in scores) / len(scores),
        "iou": (sum(ious) / len(ious)) if ious else None,
        "queries": len(scores),
    }
