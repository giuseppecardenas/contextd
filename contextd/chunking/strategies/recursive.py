"""``recursive`` — separator cascade (LangChain ``RecursiveCharacterTextSplitter``).

The cascade splits on the first separator present, recurses into any piece
still over ``max_tokens`` with the remaining separators, and finally packs
the leaves. ``"sentence"`` is a pseudo-separator backed by
:func:`sentence_spans`; ``""`` splits by token offsets as the last resort.
"""

from __future__ import annotations

import re
from itertools import pairwise

from contextd.chunking.model import Chunk, ChunkRequest
from contextd.chunking.sentences import sentence_spans
from contextd.chunking.strategies.base import Piece, pack, verbatim_piece
from contextd.chunking.strategies.window import window_spans
from contextd.chunking.tokenizer import Tokenizer

MARKDOWN_SEPARATORS: list[str] = [
    r"\n#{1,6} ",
    "\n```\n",
    r"\n-{3,}\n",
    "\n\n",
    "\n",
    "sentence",
    " ",
    "",
]
"""LangChain's ``Language.MARKDOWN`` cascade plus the sentence pseudo-separator."""

PLAIN_SEPARATORS: list[str] = ["\n\n", "\n", "sentence", " ", ""]


def _split_on(text: str, start: int, end: int, sep: str) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` on ``sep`` keeping the separator with the
    following piece (``keep_separator="start"``) so headings stay attached to
    their body."""
    if sep == "sentence":
        return [(start + s, start + e) for s, e in sentence_spans(text[start:end])]
    pattern = re.compile(sep) if sep.startswith("\\") else re.compile(re.escape(sep))
    bounds = [start]
    for m in pattern.finditer(text, start, end):
        if m.start() > start:
            bounds.append(m.start())
    bounds.append(end)
    spans: list[tuple[int, int]] = []
    for a, b in pairwise(bounds):
        # Trim the whitespace the separator contributed so pieces (and the
        # chunks rendered from them) never start or end on a blank run.
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            spans.append((a, b))
    return spans


def split_recursive(
    text: str,
    start: int,
    end: int,
    separators: list[str],
    tokenizer: Tokenizer,
    *,
    max_tokens: int,
    kind: str = "prose",
) -> list[Piece]:
    piece = verbatim_piece(text, start, end, kind, tokenizer)
    if piece.tokens <= max_tokens or not separators:
        return [piece]
    sep, rest = separators[0], separators[1:]
    if sep == "":
        return [
            verbatim_piece(text, start + s, start + e, kind, tokenizer)
            for s, e in window_spans(
                text[start:end], tokenizer, max_tokens=max_tokens, overlap_tokens=0
            )
        ]
    parts = _split_on(text, start, end, sep)
    if len(parts) <= 1:
        return split_recursive(text, start, end, rest, tokenizer, max_tokens=max_tokens, kind=kind)
    out: list[Piece] = []
    for s, e in parts:
        out.extend(split_recursive(text, s, e, rest, tokenizer, max_tokens=max_tokens, kind=kind))
    return out


class RecursiveStrategy:
    name = "recursive"

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    def separators(self, req: ChunkRequest) -> list[str]:
        if req.profile.separators:
            return list(req.profile.separators)
        return (
            MARKDOWN_SEPARATORS
            if req.suffix.lower() in (".md", ".markdown", ".mdx")
            else (PLAIN_SEPARATORS)
        )

    def chunk(self, req: ChunkRequest) -> list[Chunk]:
        if not req.text.strip():
            return []
        pieces = split_recursive(
            req.text,
            0,
            len(req.text),
            self.separators(req),
            self._tok,
            max_tokens=req.profile.max_tokens,
        )
        return pack(req, pieces, self._tok)
