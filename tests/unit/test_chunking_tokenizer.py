from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from contextd.chunking.tokenizer import (
    TiktokenTokenizer,
    VoyageTokenizer,
    WordTokenizer,
    resolve_tokenizer,
)


def test_word_tokenizer_count_and_offsets() -> None:
    t = WordTokenizer()
    assert t.id == "words"
    assert t.count("") == 0
    assert t.count("one two three four") == 5  # 4 words x 1.3, rounded
    assert t.offsets("one  two\nthree") == [(0, 3), (5, 8), (9, 14)]
    assert t.offsets("") == []


class _FakeEncoding:
    def __init__(self, ids: list[int], offsets: list[tuple[int, int]]) -> None:
        self.ids = ids
        self.offsets = offsets


def test_voyage_tokenizer_uses_hf_offsets() -> None:
    hf = MagicMock()
    hf.encode.return_value = _FakeEncoding([1, 2, 3], [(0, 2), (2, 5), (6, 9)])
    t = VoyageTokenizer("voyage-4-large", hf_tokenizer=hf)
    assert t.id == "voyage:voyage-4-large"
    assert t.count("ab cd ef") == 3
    assert t.offsets("ab cd ef") == [(0, 2), (2, 5), (6, 9)]
    hf.encode.assert_called_with("ab cd ef", add_special_tokens=False)
    assert t.count("") == 0 and t.offsets("") == []


def test_tiktoken_tokenizer_offsets_from_starts() -> None:
    enc = MagicMock()
    enc.encode.return_value = [10, 11, 12]
    enc.decode_with_offsets.return_value = ("abcdef", [0, 2, 4])
    t = TiktokenTokenizer(enc=enc)
    assert t.id == "tiktoken:o200k_base"
    assert t.count("abcdef") == 3
    assert t.offsets("abcdef") == [(0, 2), (2, 4), (4, 6)]


def test_resolve_auto_prefers_voyage_for_voyage_embedder() -> None:
    with patch("contextd.chunking.tokenizer.VoyageTokenizer") as vt:
        vt.return_value = MagicMock(id="voyage:m")
        tok = resolve_tokenizer("auto", embedding_provider="voyage", voyage_model="m")
    assert tok.id == "voyage:m"
    vt.assert_called_once_with("m")


def test_resolve_voyage_degrades_to_words_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch("contextd.chunking.tokenizer._load_hf", side_effect=OSError("offline")),
        caplog.at_level(logging.WARNING),
    ):
        tok = resolve_tokenizer("voyage", embedding_provider="voyage", voyage_model="m")
    assert isinstance(tok, WordTokenizer)
    assert "Voyage tokenizer" in caplog.text


def test_resolve_words_explicit() -> None:
    assert isinstance(
        resolve_tokenizer("words", embedding_provider="voyage", voyage_model="m"), WordTokenizer
    )


def test_resolve_auto_without_tiktoken_falls_back_to_words() -> None:
    with patch.dict("sys.modules", {"tiktoken": None}):
        tok = resolve_tokenizer("auto", embedding_provider="openai_compat", voyage_model="m")
    assert isinstance(tok, WordTokenizer)


def test_resolve_tiktoken_degrades_when_missing(caplog: pytest.LogCaptureFixture) -> None:
    with patch.dict("sys.modules", {"tiktoken": None}), caplog.at_level(logging.WARNING):
        tok = resolve_tokenizer("tiktoken", embedding_provider="openai_compat", voyage_model="m")
    assert isinstance(tok, WordTokenizer)
    assert "tiktoken unavailable" in caplog.text
