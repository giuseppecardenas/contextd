"""``late`` — structural boundaries, vectors pooled from one whole-parent pass.

Late chunking (Jina) reverses the usual order: instead of embedding each
chunk's text on its own, the *parent* text goes through the embedding model
once and every chunk's vector is the mean of the token vectors inside the
chunk's character span. Chunks therefore carry context from the whole parent
(pronouns, abbreviations, the topic set up two paragraphs earlier) without
any extra prompt or LLM call.

Boundaries are exactly :class:`StructuralStrategy`'s, so every block rule
(fences, tables, lists, size caps, overlap) applies unchanged; this strategy
only adds one :meth:`TokenEmbedder.embed_spans` call per parent and fills
``Chunk.embedding``. The chunk phase must skip re-embedding chunks that
already carry a vector.

**Prefix semantics.** ``Chunk.prefix`` / ``Chunk.embed_text`` (breadcrumb,
section summary, LLM context) are filled in later by the phase and are *not*
part of the pooled span: late chunking pools the document's own tokens, so a
prefix could only be included by re-running the parent through the model
with the prefix prepended — which would defeat the single-pass design and
shift every offset. The prefix therefore conditions the vector only for
non-late strategies; for ``late`` chunks it still feeds the full-text index.

**Span recovery.** ``Chunk`` records a line span, not a character span, and
a chunk's rendered ``text`` is not always a verbatim slice of the parent
(re-fenced code slices, table slices with the header repeated, overlap
tails). The character span is recovered in three steps, best first:

1. verbatim chunks — ``text.find(chunk.text)`` searched forward from the
   previous chunk;
2. synthetic chunks — the chunk's lines are located in order inside the
   block's line range (the re-fence frame is ignored for code), and the span
   is clipped to start after the previous chunk so consecutive parts of one
   block pool disjoint token ranges instead of all pooling the whole block;
3. otherwise the char range of the chunk's span lines via
   :meth:`LineIndex.offset_of_line`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextd.chunking.model import Chunk, ChunkRequest, LineIndex
from contextd.chunking.strategies.base import ChunkStrategy
from contextd.chunking.strategies.structural import StructuralStrategy
from contextd.chunking.tokenizer import Tokenizer
from contextd.corpus_config import ChunkProfile
from contextd.providers.base import TokenEmbedder

if TYPE_CHECKING:
    from contextd.chunking.strategies import StrategyDeps


class LateStrategy:
    name = "late"

    def __init__(self, tokenizer: Tokenizer, embedder: TokenEmbedder) -> None:
        self._inner = StructuralStrategy(tokenizer)
        self._embedder = embedder

    def chunk(self, req: ChunkRequest) -> list[Chunk]:
        chunks = self._inner.chunk(req)
        if not chunks:
            return chunks
        spans = char_spans(req, chunks)
        vectors = self._embedder.embed_spans(req.text, spans)
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        return chunks


def char_spans(req: ChunkRequest, chunks: list[Chunk]) -> list[tuple[int, int]]:
    """Character span of every chunk within ``req.text`` (see module docs)."""
    text = req.text
    index = LineIndex(text)
    spans: list[tuple[int, int]] = []
    prev_start = 0
    prev_end = 0
    for chunk in chunks:
        line_start = index.offset_of_line(chunk.span.start_line - req.base_line)
        line_end = index.offset_of_line(chunk.span.end_line - req.base_line)
        pos = text.find(chunk.text, prev_end)
        if pos < 0:
            # Overlap tails start inside the previous chunk.
            pos = text.find(chunk.text, prev_start)
        if pos >= 0:
            start, end = pos, pos + len(chunk.text)
        else:
            start, end = _locate_synthetic(text, chunk, line_start, line_end)
            if end > prev_end:
                start = max(start, prev_end)
        if end <= start:  # defensive: never hand the embedder an empty span
            start, end = line_start, max(line_end, line_start + 1)
        spans.append((start, min(end, len(text))))
        prev_start, prev_end = start, end
    return spans


def _locate_synthetic(text: str, chunk: Chunk, lo: int, hi: int) -> tuple[int, int]:
    """Narrow a synthetic chunk to the source region its lines came from.

    Lines are matched in order within ``[lo, hi)`` (the block's line range),
    each search resuming after the previous hit. The first and last line of a
    re-fenced code slice are the fence frame and are skipped; a table slice's
    repeated header simply matches at the block start.
    """
    lines = chunk.text.splitlines(keepends=True)
    if chunk.kind == "code" and len(lines) >= 3:
        lines = lines[1:-1]
    first: int | None = None
    last = lo
    cursor = lo
    for line in lines:
        if not line.strip():
            continue
        pos = text.find(line, cursor, hi)
        if pos < 0:
            pos = text.find(line.rstrip("\n"), cursor, hi)
            if pos < 0:
                continue
            end = pos + len(line.rstrip("\n"))
        else:
            end = pos + len(line)
        if first is None:
            first = pos
        last = end
        cursor = end
    if first is None:
        return lo, hi
    return first, last


def make_late(profile: ChunkProfile, tokenizer: Tokenizer, deps: StrategyDeps) -> ChunkStrategy:
    """Registry factory: ``"late": make_late``.

    Accepts ``deps.token_embedder`` or, when that is unset, a
    ``deps.embedder`` that happens to be a :class:`TokenEmbedder`; raises
    :class:`ChunkingConfigError` otherwise so a misconfigured profile fails at
    pipeline construction rather than mid-bootstrap.
    """
    from contextd.chunking.strategies import ChunkingConfigError

    embedder: object | None = deps.token_embedder
    if embedder is None:
        embedder = deps.embedder
    if not isinstance(embedder, TokenEmbedder):
        got = "none" if embedder is None else type(embedder).__name__
        raise ChunkingConfigError(
            f"profile {profile.name!r}: strategy 'late' requires a token-level embedder "
            f"(providers.embedding = 'local_hf', optional extra contextd[late]); got {got}"
        )
    return LateStrategy(tokenizer, embedder)


__all__ = ["LateStrategy", "char_spans", "make_late"]
