"""``sentence_window`` — one sentence per chunk (LlamaIndex SentenceWindowNodeParser).

The ±``window`` neighbours are *not* stored on the chunk; they are reachable
through ``NEXT_SIBLING`` edges and attached by the query-side expander, so
the graph holds each sentence once. Non-prose blocks (fences, indented code,
tables, html) are emitted whole — shredding a code block by sentence
punctuation produces garbage.
"""

from __future__ import annotations

from contextd.chunking.blocks import segment
from contextd.chunking.model import Chunk, ChunkRequest, LineIndex
from contextd.chunking.sentences import sentence_spans
from contextd.chunking.tokenizer import Tokenizer

_WHOLE_KINDS = frozenset({"fence", "code", "table", "html"})


class SentenceWindowStrategy:
    name = "sentence_window"

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    def chunk(self, req: ChunkRequest) -> list[Chunk]:
        index = LineIndex(req.text)
        chunks: list[Chunk] = []
        for block in segment(req.text, req.suffix):
            start = index.offset_of_line(block.start_line)
            end = index.offset_of_line(block.end_line)
            if block.kind in _WHOLE_KINDS:
                spans = [(start, end)]
                kind = "code" if block.kind in ("fence", "code") else block.kind
            else:
                spans = [(start + s, start + e) for s, e in sentence_spans(req.text[start:end])]
                kind = "sentence"
            for s, e in spans:
                text = req.text[s:e]
                if not text.strip():
                    continue
                chunks.append(
                    Chunk(
                        ordinal=len(chunks),
                        text=text,
                        span=index.span(s, e, base_line=req.base_line),
                        token_count=self._tok.count(text),
                        kind=kind,
                    )
                )
        return chunks
