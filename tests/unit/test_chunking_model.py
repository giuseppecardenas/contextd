from __future__ import annotations

import pytest

from contextd.chunking.model import Chunk, ChunkSpan, LineIndex


def test_line_index_basics() -> None:
    li = LineIndex("ab\ncd\nef")
    assert li.line_count == 3
    assert [li.line_of(i) for i in range(9)] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert li.span(0, 5) == ChunkSpan(0, 2)
    assert li.span(4, 8, base_line=10) == ChunkSpan(11, 13)


def test_line_index_trailing_newline_and_empty() -> None:
    assert LineIndex("ab\n").line_count == 1
    assert LineIndex("").line_count == 0
    assert LineIndex("").span(0, 0) == ChunkSpan(0, 0)


def test_empty_span_collapses_to_a_line() -> None:
    li = LineIndex("ab\ncd")
    assert li.span(3, 3) == ChunkSpan(1, 1)


def test_chunk_span_validation() -> None:
    with pytest.raises(ValueError):
        ChunkSpan(2, 1)
    with pytest.raises(ValueError):
        ChunkSpan(-1, 0)


def test_chunk_embed_text_prefix() -> None:
    c = Chunk(ordinal=0, text="body", span=ChunkSpan(0, 1), token_count=1)
    assert c.embed_text == "body"
    c.prefix = "Doc > Sec"
    assert c.embed_text == "Doc > Sec\n\nbody"
