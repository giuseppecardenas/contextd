"""``window`` — fixed-token sliding window (GraphRAG / LightRAG baseline)."""

from __future__ import annotations

from contextd.chunking.model import Chunk, ChunkRequest, LineIndex
from contextd.chunking.tokenizer import Tokenizer


def _snap(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen ``[start, end)`` to whitespace so subword tokens never cut a word."""
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return start, end


def window_spans(
    text: str, tokenizer: Tokenizer, *, max_tokens: int, overlap_tokens: int
) -> list[tuple[int, int]]:
    offsets = tokenizer.offsets(text)
    if not offsets:
        return []
    if len(offsets) <= max_tokens:
        return [_snap(text, offsets[0][0], offsets[-1][1])]
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(offsets):
        j = min(i + max_tokens, len(offsets))
        s, e = _snap(text, offsets[i][0], offsets[j - 1][1])
        # ``offsets`` units and ``count`` units need not agree (the word
        # tokenizer counts 1.3 per word; subword tokenizers re-merge across
        # snapped boundaries), so shrink until the window recounts under the
        # cap. Each step drops ~10 % of the window, so this converges fast.
        while j - i > 1 and tokenizer.count(text[s:e]) > max_tokens:
            j -= max(1, (j - i) // 10)
            s, e = _snap(text, offsets[i][0], offsets[j - 1][1])
        spans.append((s, e))
        if j == len(offsets):
            break
        step = max(1, (j - i) - overlap_tokens)
        i += step
    return spans


class WindowStrategy:
    name = "window"

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    def chunk(self, req: ChunkRequest) -> list[Chunk]:
        index = LineIndex(req.text)
        chunks: list[Chunk] = []
        for ordinal, (s, e) in enumerate(
            window_spans(
                req.text,
                self._tok,
                max_tokens=req.profile.max_tokens,
                overlap_tokens=req.profile.overlap_tokens,
            )
        ):
            text = req.text[s:e]
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    text=text,
                    span=index.span(s, e, base_line=req.base_line),
                    token_count=self._tok.count(text),
                    kind="prose",
                )
            )
        return chunks
