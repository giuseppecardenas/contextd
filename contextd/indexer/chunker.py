"""Simple token-boundary chunker with overlap and paragraph awareness.

This is a word-count approximation of tokens — sufficient for the
8k-token default, not intended as a tokenizer-accurate split. Voyage's
and Gemini's token budgets are word-count-plus roughly 30 %; staying
under the configured max_tokens guarantees under-budget calls.
"""

from __future__ import annotations


class TokenChunker:
    def __init__(self, max_tokens: int, overlap_tokens: int) -> None:
        self._max = max_tokens
        self._overlap = overlap_tokens

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self._max:
            return [text]
        chunks: list[str] = []
        step = self._max - self._overlap
        start = 0
        while start < len(words):
            end = min(start + self._max, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += step
        return chunks
