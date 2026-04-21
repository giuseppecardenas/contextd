import pytest

from contextd.indexer.chunker import TokenChunker


def test_small_text_passes_through() -> None:
    chunker = TokenChunker(max_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk("hello world")
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_large_text_is_split_with_overlap() -> None:
    chunker = TokenChunker(max_tokens=5, overlap_tokens=2)
    text = "one two three four five six seven eight nine ten"
    chunks = chunker.chunk(text)
    # With 5-token max and 2-token overlap on a 10-token input,
    # expect at least 3 chunks; adjacent chunks share 2 tokens.
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 5 for c in chunks)


def test_preserves_paragraph_boundaries_when_possible() -> None:
    chunker = TokenChunker(max_tokens=10, overlap_tokens=0)
    text = "para one here.\n\npara two here."
    chunks = chunker.chunk(text)
    assert "para one here." in chunks[0]


def test_init_rejects_overlap_ge_max() -> None:
    # overlap_tokens >= max_tokens would make step <= 0 → infinite loop
    # in chunk(). Validate at construction instead.
    with pytest.raises(ValueError, match="must be <"):
        TokenChunker(max_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError, match="must be <"):
        TokenChunker(max_tokens=50, overlap_tokens=99)
