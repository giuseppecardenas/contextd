"""``late`` strategy: structural boundaries + one ``embed_spans`` call."""

from __future__ import annotations

import pytest

from contextd.chunking.model import Chunk, ChunkRequest
from contextd.chunking.strategies import (
    STRATEGY_REGISTRY,
    ChunkingConfigError,
    StrategyDeps,
    build_chunker,
)
from contextd.chunking.strategies.late import LateStrategy, char_spans, make_late
from contextd.chunking.strategies.structural import StructuralStrategy
from contextd.chunking.tokenizer import WordTokenizer
from contextd.corpus_config import BlockRules, ChunkProfile
from contextd.providers.base import EmbeddingProvider, TokenEmbedder, UsageRecord

TOK = WordTokenizer()

_LONG_PARA = " ".join(f"word{i}" for i in range(60)) + "."
_DOC = (
    "# Title\n\nIntro sentence one. Intro sentence two.\n\n"
    + _LONG_PARA
    + "\n\n- item one with text\n- item two with text\n- item three\n\n```py\n"
    + "\n".join(f"x{i} = {i}" for i in range(30))
    + "\n\nmore = 1\n```\n\n| h1 | h2 |\n|---|---|\n"
    + "\n".join(f"| r{i} | v{i} |" for i in range(40))
    + "\n\nClosing para. Another sentence here.\n"
)


class FakeTokenEmbedder(TokenEmbedder):
    """Encodes each span as ``[midpoint, start, end]`` so alignment is checkable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[tuple[int, int]]]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 0.0] for t in texts]

    def last_usage(self) -> UsageRecord | None:
        return None

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def max_context_tokens(self) -> int:
        return 8192

    def embed_spans(self, text: str, spans: list[tuple[int, int]]) -> list[list[float]]:
        self.calls.append((text, list(spans)))
        return [[(s + e) / 2.0, float(s), float(e)] for s, e in spans]


class PlainEmbedder(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def last_usage(self) -> UsageRecord | None:
        return None

    @property
    def dimensions(self) -> int:
        return 1


def _req(
    text: str = _DOC,
    *,
    max_tokens: int = 40,
    min_tokens: int = 8,
    overlap: float = 0.0,
    blocks: BlockRules | None = None,
    base_line: int = 0,
) -> ChunkRequest:
    profile = ChunkProfile(
        name="t", strategy="late", max_tokens=max_tokens, min_tokens=min_tokens, overlap=overlap
    )
    return ChunkRequest(
        text=text, profile=profile, blocks=blocks or BlockRules(), base_line=base_line
    )


def _span(chunk: Chunk) -> tuple[int, int]:
    assert chunk.embedding is not None
    return int(chunk.embedding[1]), int(chunk.embedding[2])


def _run(req: ChunkRequest) -> tuple[list[Chunk], FakeTokenEmbedder]:
    emb = FakeTokenEmbedder()
    return LateStrategy(TOK, emb).chunk(req), emb


def test_name_and_structural_boundaries() -> None:
    req = _req()
    chunks, _ = _run(req)
    assert LateStrategy(TOK, FakeTokenEmbedder()).name == "late"
    structural = StructuralStrategy(TOK).chunk(req)
    assert [(c.text, c.span, c.kind, c.part) for c in chunks] == [
        (c.text, c.span, c.kind, c.part) for c in structural
    ]


def test_every_chunk_gets_an_embedding_from_one_call() -> None:
    req = _req(base_line=7)
    chunks, emb = _run(req)
    assert chunks
    assert all(c.embedding is not None and len(c.embedding) == 3 for c in chunks)
    assert len(emb.calls) == 1
    text, spans = emb.calls[0]
    assert text == req.text
    assert len(spans) == len(chunks)
    assert all(0 <= s < e <= len(req.text) for s, e in spans)


def test_verbatim_chunks_are_aligned_to_their_text() -> None:
    req = _req()
    chunks, _ = _run(req)
    verbatim = [c for c in chunks if c.part is None]
    assert verbatim
    for c in verbatim:
        s, e = _span(c)
        assert req.text[s:e] == c.text
        assert c.embedding is not None and c.embedding[0] == pytest.approx((s + e) / 2)


def test_spans_are_monotonic_and_within_chunk_lines() -> None:
    req = _req(base_line=3)
    chunks, _ = _run(req)
    prev_start = -1
    for c in chunks:
        s, e = _span(c)
        assert s >= prev_start
        prev_start = s
        line_of_start = req.text.count("\n", 0, s) + req.base_line
        line_of_last = req.text.count("\n", 0, max(s, e - 1)) + req.base_line
        assert c.span.start_line <= line_of_start
        assert line_of_last < c.span.end_line


def test_synthetic_code_slices_get_distinct_spans_inside_the_fence() -> None:
    req = _req()
    chunks, _ = _run(req)
    code = [c for c in chunks if c.kind == "code" and c.part is not None]
    assert len(code) >= 2
    fence_start = req.text.index("```py\n")
    fence_end = req.text.index("```\n", fence_start + 1) + 4
    prev_end = fence_start
    for c in code:
        s, e = _span(c)
        assert fence_start <= s < e <= fence_end
        assert s >= prev_end  # consecutive parts pool disjoint token ranges
        prev_end = e
        body = c.text.splitlines()[1:-1]
        first = next(ln for ln in body if ln.strip())
        last = next(ln for ln in reversed(body) if ln.strip())
        assert first in req.text[s:e] and last in req.text[s:e]


def test_synthetic_table_slices_get_distinct_spans_inside_the_table() -> None:
    req = _req()
    chunks, _ = _run(req)
    table = [c for c in chunks if c.kind == "table" and c.part is not None]
    assert len(table) >= 2
    table_start = req.text.index("| h1 | h2 |")
    table_end = req.text.index("| r39 | v39 |") + len("| r39 | v39 |\n")
    prev_end = table_start
    for c in table:
        s, e = _span(c)
        assert table_start <= s < e <= table_end
        assert s >= prev_end
        prev_end = e
        last_row = c.text.rstrip("\n").splitlines()[-1]
        assert last_row in req.text[s:e]
    # The first slice starts at the header; later ones start after the
    # previous slice's rows rather than re-pooling the whole table.
    assert _span(table[0])[0] == table_start
    assert _span(table[1])[0] > table_start


def test_overlap_tail_chunks_still_get_spans_covering_their_lines() -> None:
    req = _req(max_tokens=30, min_tokens=4, overlap=0.2)
    chunks, _ = _run(req)
    assert all(c.embedding is not None for c in chunks)
    for c in chunks:
        s, e = _span(c)
        assert 0 <= s < e <= len(req.text)


def test_empty_text_yields_no_chunks_and_no_embed_call() -> None:
    chunks, emb = _run(_req(""))
    assert chunks == [] and emb.calls == []


def test_char_spans_of_plain_verbatim_chunks() -> None:
    req = _req("Alpha beta.\n\nGamma delta.\n")
    chunks = StructuralStrategy(TOK).chunk(req)
    spans = char_spans(req, chunks)
    assert [req.text[s:e] for s, e in spans] == [c.text for c in chunks]


# --- factory ---------------------------------------------------------------


def _profile() -> ChunkProfile:
    return ChunkProfile(name="late_p", strategy="late")


def test_make_late_requires_a_token_embedder() -> None:
    with pytest.raises(ChunkingConfigError, match=r"late_p.*token-level embedder.*got none"):
        make_late(_profile(), TOK, StrategyDeps())
    with pytest.raises(ChunkingConfigError, match=r"got PlainEmbedder"):
        make_late(_profile(), TOK, StrategyDeps(embedder=PlainEmbedder()))
    with pytest.raises(ChunkingConfigError, match=r"got str"):
        make_late(_profile(), TOK, StrategyDeps(token_embedder="nope"))


def test_make_late_builds_with_token_embedder_or_capable_embedder() -> None:
    emb = FakeTokenEmbedder()
    built = make_late(_profile(), TOK, StrategyDeps(token_embedder=emb))
    assert isinstance(built, LateStrategy) and built.name == "late"
    # A TokenEmbedder passed as the general embedder is accepted too.
    built = make_late(_profile(), TOK, StrategyDeps(embedder=emb))
    assert isinstance(built, LateStrategy)


def test_registry_wiring_builds_late_chunker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(STRATEGY_REGISTRY, "late", make_late)
    with pytest.raises(ChunkingConfigError):
        build_chunker(_profile(), TOK)
    built = build_chunker(_profile(), TOK, StrategyDeps(token_embedder=FakeTokenEmbedder()))
    assert isinstance(built, LateStrategy)
    chunks = built.chunk(_req())
    assert chunks and all(c.embedding is not None for c in chunks)
