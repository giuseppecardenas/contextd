"""Strategy protocol and the shared piece packer.

Every size-bounded strategy reduces to the same last mile: it produces an
ordered list of :class:`Piece` (character spans of the parent text, or
synthetic text such as a table slice with its header repeated), and
:func:`pack` groups those pieces into chunks that respect ``max_tokens``,
forward-merges anything under ``min_tokens``, and optionally carries an
overlap tail from the previous chunk. Keeping that in one place is what makes
the boundary invariants testable once rather than per strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from contextd.chunking.model import Chunk, ChunkRequest, LineIndex
from contextd.chunking.sentences import sentence_spans
from contextd.chunking.tokenizer import Tokenizer


class ChunkStrategy(Protocol):
    name: str

    def chunk(self, req: ChunkRequest) -> list[Chunk]: ...


@dataclass
class Piece:
    """A candidate chunk fragment.

    ``verbatim`` pieces satisfy ``text == source[start:end]``; a group of
    consecutive verbatim pieces is emitted as the contiguous source slice so
    original spacing survives. Synthetic pieces (table slices, re-fenced code)
    are joined with blank lines instead.
    """

    start: int
    end: int
    text: str
    kind: str
    tokens: int
    verbatim: bool = True
    part: int | None = None


def verbatim_piece(source: str, start: int, end: int, kind: str, tokenizer: Tokenizer) -> Piece:
    text = source[start:end]
    return Piece(start=start, end=end, text=text, kind=kind, tokens=tokenizer.count(text))


def _group_tokens(group: list[Piece]) -> int:
    return sum(p.tokens for p in group)


def _greedy(pieces: list[Piece], max_tokens: int) -> list[list[Piece]]:
    groups: list[list[Piece]] = []
    cur: list[Piece] = []
    cur_tokens = 0
    for p in pieces:
        if cur and cur_tokens + p.tokens > max_tokens:
            groups.append(cur)
            cur, cur_tokens = [], 0
        cur.append(p)
        cur_tokens += p.tokens
    if cur:
        groups.append(cur)
    return groups


def _merge_small(groups: list[list[Piece]], max_tokens: int, min_tokens: int) -> list[list[Piece]]:
    """Fold groups under ``min_tokens`` into a neighbour when the result stays
    under ``max_tokens`` — forward first (RAGFlow / Unstructured convention),
    backward otherwise, kept as-is when neither fits."""
    if min_tokens <= 0:
        return groups
    out: list[list[Piece]] = []
    i = 0
    while i < len(groups):
        g = groups[i]
        if _group_tokens(g) < min_tokens:
            if (
                i + 1 < len(groups)
                and _group_tokens(g) + _group_tokens(groups[i + 1]) <= max_tokens
            ):
                groups[i + 1] = g + groups[i + 1]
                i += 1
                continue
            if out and _group_tokens(out[-1]) + _group_tokens(g) <= max_tokens:
                out[-1] = out[-1] + g
                i += 1
                continue
        out.append(g)
        i += 1
    return out


def _render(source: str, group: list[Piece]) -> str:
    if all(p.verbatim for p in group):
        return source[group[0].start : group[-1].end]
    return "\n\n".join(p.text.rstrip("\n") for p in group) + "\n"


def _kind(group: list[Piece]) -> str:
    kinds = {p.kind for p in group}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _overlap_tail(prev_text: str, overlap_tokens: int, tokenizer: Tokenizer) -> str:
    """Trailing sentences of ``prev_text`` whose token total fits ``overlap_tokens``."""
    if overlap_tokens <= 0:
        return ""
    spans = sentence_spans(prev_text)
    tail_start: int | None = None
    used = 0
    for s, e in reversed(spans):
        n = tokenizer.count(prev_text[s:e])
        if used + n > overlap_tokens:
            break
        used += n
        tail_start = s
    if tail_start is None:
        return ""
    return prev_text[tail_start:].rstrip("\n")


def pack(
    req: ChunkRequest,
    pieces: list[Piece],
    tokenizer: Tokenizer,
    *,
    overlap_tokens: int | None = None,
    merge_adjacent: bool = True,
) -> list[Chunk]:
    """Group pieces into chunks honouring the profile's size knobs.

    ``merge_adjacent=False`` keeps every piece its own chunk unless it is
    under ``min_tokens`` — for strategies whose pieces already *are* the
    intended chunks (semantic groups) and must not be greedily re-packed.
    """
    if not pieces:
        return []
    profile = req.profile
    initial = _greedy(pieces, profile.max_tokens) if merge_adjacent else [[p] for p in pieces]
    groups = _merge_small(initial, profile.max_tokens, profile.min_tokens)
    index = LineIndex(req.text)
    overlap = profile.overlap_tokens if overlap_tokens is None else overlap_tokens
    chunks: list[Chunk] = []
    prev_text = ""
    prev_start = 0
    for ordinal, group in enumerate(groups):
        text = _render(req.text, group)
        start, end = group[0].start, group[-1].end
        if ordinal > 0 and overlap > 0:
            tail = _overlap_tail(prev_text, overlap, tokenizer)
            if tail:
                text = f"{tail}\n{text}"
                # The tail begins somewhere inside the previous chunk; widen
                # the span to the line the tail starts on.
                start = prev_start + max(0, len(prev_text) - len(tail) - 1)
                start = min(start, group[0].start)
        part = group[0].part if len(group) == 1 else None
        chunks.append(
            Chunk(
                ordinal=ordinal,
                text=text,
                span=index.span(start, end, base_line=req.base_line),
                token_count=tokenizer.count(text),
                kind=_kind(group),
                part=part,
            )
        )
        prev_text = _render(req.text, group)
        prev_start = group[0].start
    return chunks
