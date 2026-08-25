"""``semantic`` — embedding-breakpoint splitting.

Sentences are embedded in buffered groups (each sentence with ``buffer_size``
neighbours on either side, the LlamaIndex / LangChain scheme); the cosine
distance between consecutive groups is the boundary signal. A breakpoint is
placed wherever the distance exceeds a threshold chosen by
``threshold_type``:

* ``percentile`` — the p-th percentile of all distances (default 95);
* ``stddev`` — mean + k·std (default k = 3);
* ``iqr`` — Q3 + k·IQR (default k = 1.5);
* ``gradient`` — the p-th percentile of the distance *gradient* (default 95),
  which favours sharp topic changes over gradual drift.

Groups over ``max_tokens`` are hard-split by the recursive cascade; groups
under ``min_tokens`` are merged into a neighbour. Non-prose blocks (fences,
tables, code, html) are atomic and never embedded for boundary detection.
Costs one embedding per sentence at index time.
"""

from __future__ import annotations

import math
from itertools import pairwise

from contextd.chunking.blocks import segment
from contextd.chunking.model import Chunk, ChunkRequest, LineIndex
from contextd.chunking.sentences import sentence_spans
from contextd.chunking.strategies.base import Piece, pack, verbatim_piece
from contextd.chunking.strategies.recursive import PLAIN_SEPARATORS, split_recursive
from contextd.chunking.tokenizer import Tokenizer
from contextd.corpus_config import ThresholdType
from contextd.providers.base import EmbeddingProvider

_ATOMIC = frozenset({"fence", "code", "table", "html"})

DEFAULT_THRESHOLDS: dict[ThresholdType, float] = {
    "percentile": 95.0,
    "stddev": 3.0,
    "iqr": 1.5,
    "gradient": 95.0,
}


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / (na * nb)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def breakpoints(
    distances: list[float], threshold_type: ThresholdType, threshold: float | None
) -> list[int]:
    """Indices ``i`` such that a boundary falls after sentence ``i``."""
    if len(distances) < 2:
        return []
    amount = DEFAULT_THRESHOLDS[threshold_type] if threshold is None else threshold
    if threshold_type == "percentile":
        cut = _percentile(distances, amount)
        return [i for i, d in enumerate(distances) if d > cut]
    if threshold_type == "stddev":
        mean = sum(distances) / len(distances)
        var = sum((d - mean) ** 2 for d in distances) / len(distances)
        cut = mean + amount * math.sqrt(var)
        return [i for i, d in enumerate(distances) if d > cut]
    if threshold_type == "iqr":
        q1, q3 = _percentile(distances, 25.0), _percentile(distances, 75.0)
        cut = q3 + amount * (q3 - q1)
        return [i for i, d in enumerate(distances) if d > cut]
    # gradient: rate of change of the distance curve.
    grad = [b - a for a, b in pairwise(distances)]
    if not grad:
        return []
    cut = _percentile(grad, amount)
    return [i for i, g in enumerate(grad) if g > cut]


class SemanticStrategy:
    name = "semantic"

    def __init__(self, tokenizer: Tokenizer, embedder: EmbeddingProvider) -> None:
        self._tok = tokenizer
        self._embedder = embedder

    def _prose_groups(self, req: ChunkRequest, start: int, end: int) -> list[tuple[int, int]]:
        """Sentence spans of ``req.text[start:end]`` grouped by embedding breakpoints."""
        sentences = [(start + s, start + e) for s, e in sentence_spans(req.text[start:end])]
        if len(sentences) <= 2:
            return [(sentences[0][0], sentences[-1][1])] if sentences else []
        buffer = req.profile.buffer_size
        combined = [
            " ".join(
                req.text[s:e]
                for s, e in sentences[max(0, i - buffer) : min(len(sentences), i + buffer + 1)]
            )
            for i in range(len(sentences))
        ]
        vectors = self._embedder.embed(combined)
        distances = [_cosine_distance(a, b) for a, b in pairwise(vectors)]
        cuts = set(breakpoints(distances, req.profile.threshold_type, req.profile.threshold))
        groups: list[tuple[int, int]] = []
        gs = sentences[0][0]
        for i, (_, e) in enumerate(sentences):
            if i in cuts or i == len(sentences) - 1:
                groups.append((gs, e))
                if i + 1 < len(sentences):
                    gs = sentences[i + 1][0]
        return groups

    def pieces(self, req: ChunkRequest) -> list[Piece]:
        index = LineIndex(req.text)
        out: list[Piece] = []
        for block in segment(req.text, req.suffix):
            start = index.offset_of_line(block.start_line)
            end = index.offset_of_line(block.end_line)
            if block.kind in _ATOMIC:
                kind = "code" if block.kind in ("fence", "code") else block.kind
                out.append(verbatim_piece(req.text, start, end, kind, self._tok))
                continue
            for gs, ge in self._prose_groups(req, start, end):
                out.extend(
                    split_recursive(
                        req.text,
                        gs,
                        ge,
                        PLAIN_SEPARATORS,
                        self._tok,
                        max_tokens=req.profile.max_tokens,
                        kind="prose",
                    )
                )
        return out

    def chunk(self, req: ChunkRequest) -> list[Chunk]:
        # Semantic groups are the chunks: no greedy packing across
        # breakpoints, only the min_tokens merge and the oversize split.
        return pack(req, self.pieces(req), self._tok, merge_adjacent=False)
