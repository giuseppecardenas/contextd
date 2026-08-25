from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextd.chunking.model import ChunkRequest
from contextd.chunking.strategies import ChunkingConfigError, StrategyDeps, build_chunker
from contextd.chunking.strategies.semantic import breakpoints
from contextd.chunking.tokenizer import WordTokenizer
from contextd.corpus_config import BlockRules, ChunkProfile
from contextd.inference.prompts import PromptRenderer
from contextd.providers.base import EmbeddingProvider, InferenceProvider, PromptRequest, UsageRecord

TOK = WordTokenizer()


class KeywordEmbedder(EmbeddingProvider):
    """Deterministic 3-dim vectors keyed on topic words."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(t.count("apple")), float(t.count("rocket")), 1.0] for t in texts]

    def last_usage(self) -> UsageRecord | None:
        return None

    @property
    def dimensions(self) -> int:
        return 3


class FakeInference(InferenceProvider):
    def __init__(self, response: str | Exception) -> None:
        self._r = response

    def generate(self, request: PromptRequest) -> str:
        if isinstance(self._r, Exception):
            raise self._r
        return self._r

    def last_usage(self) -> UsageRecord | None:
        return None


_APPLES = " ".join(f"An apple sentence number {i}." for i in range(6))
_ROCKETS = " ".join(f"A rocket sentence number {i}." for i in range(6))
_DOC = f"{_APPLES} {_ROCKETS}\n\n```\ncode\n```\n"


def _semantic_profile(**kw: object) -> ChunkProfile:
    base: dict[str, object] = {
        "name": "s",
        "strategy": "semantic",
        "max_tokens": 200,
        "min_tokens": 0,
        "buffer_size": 0,
        "threshold": 50.0,
    }
    base.update(kw)
    return ChunkProfile.model_validate(base)


def test_semantic_splits_at_topic_boundary_and_keeps_code_atomic() -> None:
    emb = KeywordEmbedder()
    p = _semantic_profile()
    chunks = build_chunker(p, TOK, StrategyDeps(embedder=emb)).chunk(
        ChunkRequest(text=_DOC, profile=p, blocks=BlockRules())
    )
    assert [c.kind for c in chunks] == ["prose", "prose", "code"]
    assert chunks[0].text.strip() == _APPLES
    assert chunks[1].text.strip() == _ROCKETS
    # One embedding call per prose block, one text per sentence.
    assert len(emb.calls) == 1 and len(emb.calls[0]) == 12


def test_semantic_buffer_widens_embedded_groups() -> None:
    emb = KeywordEmbedder()
    p = _semantic_profile(buffer_size=1)
    build_chunker(p, TOK, StrategyDeps(embedder=emb)).chunk(
        ChunkRequest(text=_DOC, profile=p, blocks=BlockRules())
    )
    assert emb.calls[0][1].count("sentence") == 3  # sentence i with both neighbours


def test_semantic_oversize_group_is_split_and_small_groups_merge() -> None:
    p = _semantic_profile(max_tokens=20, min_tokens=0)
    chunks = build_chunker(p, TOK, StrategyDeps(embedder=KeywordEmbedder())).chunk(
        ChunkRequest(text=_DOC, profile=p, blocks=BlockRules())
    )
    prose = [c for c in chunks if c.kind == "prose"]
    assert len(prose) > 2 and all(c.token_count <= 20 for c in prose)
    p2 = _semantic_profile(max_tokens=200, min_tokens=100)
    merged = build_chunker(p2, TOK, StrategyDeps(embedder=KeywordEmbedder())).chunk(
        ChunkRequest(text=_DOC, profile=p2, blocks=BlockRules())
    )
    # Both ~40-token groups are under min_tokens, so they fold together (and
    # absorb the tiny code block) instead of standing as separate chunks.
    assert len(merged) == 1
    assert _APPLES in merged[0].text and _ROCKETS in merged[0].text


def test_semantic_requires_embedder() -> None:
    with pytest.raises(ChunkingConfigError, match="embedding provider"):
        build_chunker(_semantic_profile(), TOK)


@pytest.mark.parametrize(
    ("kind", "amount", "expected"),
    [
        ("percentile", 50.0, [2]),
        ("stddev", 1.0, [2]),
        ("iqr", 1.5, [2]),
        ("gradient", 90.0, [1]),
    ],
)
def test_breakpoint_rules(kind: str, amount: float, expected: list[int]) -> None:
    assert breakpoints([0.1, 0.1, 0.9, 0.1, 0.1], kind, amount) == expected  # type: ignore[arg-type]


def test_breakpoint_defaults_and_short_inputs() -> None:
    assert breakpoints([0.5], "percentile", None) == []
    assert breakpoints([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9], "percentile", None) == [
        9
    ]


def test_propositions_strategy_emits_one_chunk_per_statement(tmp_path: Path) -> None:
    p = ChunkProfile(name="p", strategy="propositions", max_propositions=3)
    provider = FakeInference(json.dumps({"propositions": ["Fact one.", "Fact two.", "x", "y"]}))
    chunks = build_chunker(
        p, TOK, StrategyDeps(inference=provider, renderer=PromptRenderer(tmp_path))
    ).chunk(ChunkRequest(text="Some body.\nMore.\n", profile=p, blocks=BlockRules(), base_line=7))
    assert [c.text for c in chunks] == ["Fact one.", "Fact two.", "x"]
    assert all(c.kind == "proposition" for c in chunks)
    assert [c.part for c in chunks] == [1, 2, 3]
    assert all((c.span.start_line, c.span.end_line) == (7, 9) for c in chunks)


def test_propositions_strategy_falls_back_to_structural(tmp_path: Path) -> None:
    p = ChunkProfile(name="p", strategy="propositions")
    chunks = build_chunker(
        p,
        TOK,
        StrategyDeps(inference=FakeInference(RuntimeError()), renderer=PromptRenderer(tmp_path)),
    ).chunk(ChunkRequest(text="Some body here.\n", profile=p, blocks=BlockRules()))
    assert len(chunks) == 1 and chunks[0].kind == "paragraph"


def test_propositions_requires_inference_and_renderer(tmp_path: Path) -> None:
    p = ChunkProfile(name="p", strategy="propositions")
    with pytest.raises(ChunkingConfigError, match="inference provider"):
        build_chunker(p, TOK, StrategyDeps(renderer=PromptRenderer(tmp_path)))
    with pytest.raises(ChunkingConfigError, match="prompt renderer"):
        build_chunker(p, TOK, StrategyDeps(inference=FakeInference("{}")))
