"""Pooling math of ``LocalHFEmbedder`` with a fake tokenizer / model.

No model is downloaded: the fake tokenizer is whitespace words with HF-style
``offset_mapping`` (special tokens at both ends carry ``(0, 0)``) and the fake
model returns, per token, a vector encoding the word's number, so a pooled
span must equal the mean of the numbers of the words it covers.
"""

from __future__ import annotations

import math
import re
import sys
from typing import Any

import pytest

from contextd.config import LocalHFConfig
from contextd.providers.base import EmbeddingProvider, TokenEmbedder
from contextd.providers.local_hf import LocalHFEmbedder

_WORD = re.compile(r"\S+")
DIM = 3


class FakeTokenizer:
    """Whitespace words; mimics ``PreTrainedTokenizerFast.__call__``."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        assert return_offsets_mapping
        offsets = [(m.start(), m.end()) for m in _WORD.finditer(text)]
        if add_special_tokens:
            offsets = [(0, 0), *offsets, (0, 0)]
        if truncation and max_length is not None:
            offsets = offsets[:max_length]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

    def num_special_tokens_to_add(self) -> int:
        return 2


def _word_value(word: str) -> float:
    digits = "".join(ch for ch in word if ch.isdigit())
    return float(digits) if digits else -1.0


class FakeModel:
    """Returns ``[word number, 1, 0]`` per token; special tokens get a poison
    value so any leak into pooling is visible."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.max_seq_length = 10_000
        self.calls: list[list[str]] = []
        self.tokenizer = FakeTokenizer()

    def encode(self, texts: list[str], **kwargs: Any) -> list[Any]:
        self.calls.append(list(texts))
        if kwargs.get("output_value") == "token_embeddings":
            out = []
            for text in texts:
                enc = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                rows = []
                for s, e in enc["offset_mapping"]:
                    if e > s:
                        rows.append(self._vec(_word_value(text[s:e])))
                    else:
                        rows.append(self._vec(9999.0))
                out.append(rows)
            return out
        return [self._vec(float(len(t))) for t in texts]

    def _vec(self, value: float) -> list[float]:
        return [value, 1.0, 0.0][: self.dim] + [0.0] * max(0, self.dim - 3)


def _embedder(
    *, dims: int = DIM, max_ctx: int = 8192, normalize: bool = False, model_dim: int = DIM
) -> tuple[LocalHFEmbedder, FakeModel]:
    cfg = LocalHFConfig(
        model="fake/model", dimensions=dims, max_context_tokens=max_ctx, normalize=normalize
    )
    model = FakeModel(model_dim)
    return LocalHFEmbedder(cfg, _model=model, _tokenizer=model.tokenizer), model


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def _span_of(text: str, first: int, last: int) -> tuple[int, int]:
    """Char span from word ``first`` to word ``last`` inclusive."""
    ms = list(_WORD.finditer(text))
    return ms[first].start(), ms[last].end()


def test_is_a_token_embedder_and_an_embedding_provider() -> None:
    emb, _ = _embedder()
    assert isinstance(emb, TokenEmbedder)
    assert isinstance(emb, EmbeddingProvider)
    assert emb.dimensions == DIM
    assert emb.max_context_tokens == 8192


def test_span_pooling_means_the_tokens_inside_each_span() -> None:
    text = _words(10)
    emb, model = _embedder()
    spans = [_span_of(text, 0, 3), _span_of(text, 4, 4), _span_of(text, 5, 9)]
    vectors = emb.embed_spans(text, spans)
    assert [v[0] for v in vectors] == pytest.approx([1.5, 4.0, 7.0])
    assert all(v[1] == 1.0 and v[2] == 0.0 for v in vectors)
    # One forward pass over the whole text.
    assert model.calls == [[text]]


def test_partial_token_overlap_counts_the_token() -> None:
    text = _words(6)
    emb, _ = _embedder()
    s, e = _span_of(text, 2, 3)
    # Cut into the middle of w2 and w3: both still intersect the span.
    (vec,) = emb.embed_spans(text, [(s + 1, e - 1)])
    assert vec[0] == pytest.approx(2.5)


def test_span_with_no_token_borrows_the_nearest_token_vector() -> None:
    text = "w0   w1 w2"
    emb, _ = _embedder()
    gap = (text.index("w0") + 2, text.index("w1"))  # whitespace only
    assert text[gap[0] : gap[1]].strip() == ""
    empty = (text.index("w2"), text.index("w2"))  # zero-width at w2
    vectors = emb.embed_spans(text, [gap, empty])
    assert vectors[0][0] in (0.0, 1.0)  # nearest neighbour, never a zero vector
    assert vectors[0][1] == 1.0
    assert vectors[1] == [2.0, 1.0, 0.0]
    assert all(any(x != 0.0 for x in v) for v in vectors)


def test_long_text_is_windowed_and_spans_pool_across_windows() -> None:
    text = _words(30)
    # 14 body tokens + 2 specials per window; windows overlap by 3 tokens.
    emb, model = _embedder(max_ctx=16)
    spans = [
        _span_of(text, 0, 4),
        _span_of(text, 5, 9),
        _span_of(text, 20, 24),
        _span_of(text, 0, 29),
    ]
    vectors = emb.embed_spans(text, spans)
    assert len(model.calls) == 1
    windows = model.calls[0]
    assert len(windows) > 1
    for w in windows:
        assert len(_WORD.findall(w)) <= 14
    # Every word is covered by at least one window.
    assert set(_WORD.findall(text)) == {w for win in windows for w in _WORD.findall(win)}
    # Spans inside one window, straddling a window edge, and covering all
    # windows all pool to the plain mean of the words they cover: a word seen
    # by two windows contributes each view at half weight, never twice.
    assert [v[0] for v in vectors] == pytest.approx([2.0, 7.0, 22.0, 14.5])
    usage = emb.last_usage()
    assert usage is not None
    assert usage.provider == "local_hf" and usage.call_site == "embedding"
    assert usage.output_tokens == 0
    assert usage.input_tokens == sum(len(_WORD.findall(w)) + 2 for w in windows)


def test_normalize_yields_unit_vectors() -> None:
    text = _words(4)
    emb, _ = _embedder(normalize=True)
    (vec,) = emb.embed_spans(text, [_span_of(text, 0, 3)])
    assert math.sqrt(sum(x * x for x in vec)) == pytest.approx(1.0)


def test_embed_each_text_independently() -> None:
    emb, model = _embedder()
    vectors = emb.embed(["ab", "abcd", ""])
    assert [v[0] for v in vectors] == [2.0, 4.0, 1.0]  # "" is sent as " "
    assert model.calls == [["ab", "abcd", " "]]
    usage = emb.last_usage()
    assert usage is not None and usage.provider == "local_hf"
    assert emb.embed([]) == []


def test_empty_text_falls_back_to_embedding_span_texts() -> None:
    emb, model = _embedder()
    vectors = emb.embed_spans("   ", [(0, 3)])
    assert len(vectors) == 1 and vectors[0][1] == 1.0
    assert model.calls == [[" "]]  # whitespace-only input is sent as a single space


def test_dimension_mismatch_raises_clear_error() -> None:
    emb, _ = _embedder(dims=4, model_dim=3)
    with pytest.raises(ValueError, match=r"3-dimension vector.*is 4"):
        emb.embed(["x"])
    with pytest.raises(ValueError, match=r"3-dimension vector.*is 4"):
        emb.embed_spans("w0 w1", [(0, 2)])


def test_missing_extra_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    emb = LocalHFEmbedder(LocalHFConfig())  # constructing never imports the extra
    with pytest.raises(RuntimeError, match=r"contextd\[late\]"):
        emb.embed(["x"])


def test_config_defaults_match_the_vector_index() -> None:
    cfg = LocalHFConfig()
    assert cfg.dimensions == 1024
    assert cfg.max_context_tokens == 8192
    assert cfg.device == "cpu" and cfg.normalize is True and cfg.batch_size == 8
    with pytest.raises(ValueError):
        LocalHFConfig(dimensions=0)
