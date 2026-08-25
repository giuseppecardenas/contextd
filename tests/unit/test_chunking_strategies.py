"""Boundary invariants for the base strategies and the shared packer."""

from __future__ import annotations

from itertools import pairwise

import pytest

from contextd.chunking.model import Chunk, ChunkRequest
from contextd.chunking.strategies import (
    STRATEGY_REGISTRY,
    ChunkingConfigError,
    build_chunker,
)
from contextd.chunking.strategies.base import Piece, pack
from contextd.chunking.tokenizer import WordTokenizer
from contextd.corpus_config import BlockRules, ChunkProfile

TOK = WordTokenizer()

_LONG_PARA = " ".join(f"word{i}" for i in range(60)) + "."
_DOC = (
    "Intro sentence one. Intro sentence two.\n\n"
    + _LONG_PARA
    + "\n\n- item one with text\n- item two with text\n- item three\n\n```py\n"
    + "\n".join(f"x{i} = {i}" for i in range(30))
    + "\n\nmore = 1\n```\n\n| h1 | h2 |\n|---|---|\n"
    + "\n".join(f"| r{i} | v{i} |" for i in range(40))
    + "\n\nClosing para. Another sentence here.\n"
)


def _req(
    text: str = _DOC,
    *,
    strategy: str = "structural",
    max_tokens: int = 40,
    min_tokens: int = 8,
    overlap: float = 0.0,
    blocks: BlockRules | None = None,
    suffix: str = ".md",
    base_line: int = 0,
) -> ChunkRequest:
    profile = ChunkProfile(
        name="t",
        strategy=strategy,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap=overlap,
    )
    return ChunkRequest(
        text=text,
        profile=profile,
        blocks=blocks or BlockRules(),
        suffix=suffix,
        base_line=base_line,
    )


def _run(req: ChunkRequest) -> list[Chunk]:
    return build_chunker(req.profile, TOK).chunk(req)


@pytest.mark.parametrize("strategy", ["structural", "window", "recursive"])
def test_size_bounded_strategies_respect_max_tokens(strategy: str) -> None:
    req = _req(strategy=strategy)
    chunks = _run(req)
    assert chunks
    assert all(c.token_count <= req.profile.max_tokens for c in chunks)
    assert all(c.token_count == TOK.count(c.text) for c in chunks)


@pytest.mark.parametrize("strategy", ["structural", "window", "recursive", "sentence_window"])
def test_ordinals_and_spans_are_monotonic(strategy: str) -> None:
    chunks = _run(_req(strategy=strategy, base_line=100))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.span.start_line >= 100 for c in chunks)
    for a, b in pairwise(chunks):
        assert b.span.start_line >= a.span.start_line
        assert b.span.end_line >= a.span.end_line


@pytest.mark.parametrize("strategy", ["structural", "window", "recursive", "sentence_window"])
def test_every_word_of_the_source_appears_in_some_chunk(strategy: str) -> None:
    chunks = _run(_req(strategy=strategy))
    joined = "\n".join(c.text for c in chunks)
    for word in _DOC.split():
        assert word in joined, word


@pytest.mark.parametrize("strategy", ["structural", "window", "recursive", "sentence_window"])
def test_chunks_never_start_or_end_with_blank_runs(strategy: str) -> None:
    for c in _run(_req(strategy=strategy)):
        assert c.text == c.text.strip("\n") or c.text.endswith("\n")
        assert not c.text.startswith((" ", "\n"))


@pytest.mark.parametrize("strategy", ["structural", "window", "recursive", "sentence_window"])
def test_empty_text_yields_no_chunks(strategy: str) -> None:
    assert _run(_req("", strategy=strategy)) == []
    assert _run(_req("\n\n  \n", strategy=strategy)) == []


def test_small_text_is_a_single_chunk() -> None:
    for strategy in ("structural", "window", "recursive"):
        chunks = _run(_req("Just a short paragraph.", strategy=strategy))
        assert len(chunks) == 1 and chunks[0].text.strip() == "Just a short paragraph."


# --- structural ------------------------------------------------------------


def test_structural_fence_is_never_split_mid_fence() -> None:
    chunks = _run(_req())
    code = [c for c in chunks if c.kind == "code"]
    assert len(code) >= 2
    for c in code:
        assert c.text.startswith("```py\n") and c.text.rstrip("\n").endswith("```")
    assert [c.part for c in code] == list(range(1, len(code) + 1))


def test_structural_small_fence_stays_whole() -> None:
    doc = "para\n\n```sh\necho hi\n```\n\nafter\n"
    chunks = _run(_req(doc, max_tokens=100, min_tokens=0))
    assert len(chunks) == 1 and "```sh\necho hi\n```" in chunks[0].text


def test_structural_unprotected_fence_is_plain_prose() -> None:
    chunks = _run(_req(blocks=BlockRules(protect_code_fences=False)))
    assert all(c.part is None or c.kind == "table" for c in chunks)


def test_structural_table_rows_repeat_header() -> None:
    chunks = _run(_req())
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) > 1
    for c in tables:
        assert c.text.startswith("| h1 | h2 |\n|---|---|\n")
        assert c.token_count <= 40
    rows = [ln for c in tables for ln in c.text.splitlines() if ln.startswith("| r")]
    assert len(rows) == 40 and len(set(rows)) == 40


def test_structural_table_whole_mode_keeps_one_oversize_chunk() -> None:
    chunks = _run(_req(blocks=BlockRules(table_mode="whole")))
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1 and tables[0].token_count > 40


def test_structural_table_prose_mode_falls_back_to_cascade() -> None:
    chunks = _run(_req(blocks=BlockRules(table_mode="prose")))
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) > 1
    assert all(c.token_count <= 40 for c in tables)
    assert not all(c.text.startswith("| h1 |") for c in tables)


def test_structural_list_splits_between_items() -> None:
    doc = "\n".join(f"- item {i} " + " ".join(["w"] * 10) for i in range(12)) + "\n"
    chunks = _run(_req(doc, max_tokens=40, min_tokens=0))
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("- item")
        assert c.token_count <= 40


def test_structural_forward_merges_small_blocks() -> None:
    doc = "Tiny.\n\nAlso tiny.\n\n" + _LONG_PARA + "\n"
    chunks = _run(_req(doc, max_tokens=60, min_tokens=10))
    assert chunks[0].text.startswith("Tiny.\n\nAlso tiny.")


def test_structural_window_fallback_for_oversize_paragraph() -> None:
    chunks = _run(_req(_LONG_PARA, blocks=BlockRules(sentence_fallback="window")))
    assert len(chunks) > 1 and all(c.token_count <= 40 for c in chunks)


def test_structural_overlap_prepends_previous_tail() -> None:
    doc = "\n\n".join(f"Sentence number {i} is here." for i in range(20)) + "\n"
    with_overlap = _run(_req(doc, max_tokens=30, min_tokens=0, overlap=0.3))
    without = _run(_req(doc, max_tokens=30, min_tokens=0))
    assert len(with_overlap) == len(without)
    for a, b in zip(without[1:], with_overlap[1:], strict=True):
        assert b.text.endswith(a.text)
        assert len(b.text) > len(a.text)
        assert b.span.start_line <= a.span.start_line
        assert b.token_count <= 30 + 9  # base + overlap budget


def test_structural_plain_suffix_uses_paragraph_blocks() -> None:
    doc = "para one line\nsame para\n\npara two\n"
    chunks = _run(_req(doc, suffix=".txt", max_tokens=100, min_tokens=0))
    assert len(chunks) == 1 and chunks[0].kind == "paragraph"


# --- window ------------------------------------------------------------------


def test_window_overlap_shares_text_between_neighbours() -> None:
    chunks = _run(_req(_LONG_PARA, strategy="window", max_tokens=20, overlap=0.25))
    assert len(chunks) > 2
    for a, b in pairwise(chunks):
        tail_word = a.text.split()[-1]
        assert tail_word in b.text.split()


def test_window_no_overlap_partitions_words() -> None:
    chunks = _run(_req(_LONG_PARA, strategy="window", max_tokens=20))
    words = [w for c in chunks for w in c.text.split()]
    assert words == _LONG_PARA.split()


# --- recursive ---------------------------------------------------------------


def test_recursive_keeps_heading_with_its_body() -> None:
    doc = "## A\n\n" + _LONG_PARA + "\n\n## B\n\nshort b\n"
    chunks = _run(_req(doc, strategy="recursive", max_tokens=40, min_tokens=0))
    assert len(chunks) > 1
    assert any("## B\n\nshort b" in c.text for c in chunks)
    assert chunks[0].text.startswith("## A")


def test_recursive_custom_separators() -> None:
    profile = ChunkProfile(
        name="t", strategy="recursive", max_tokens=16, min_tokens=0, separators=["|", " "]
    )
    req = ChunkRequest(
        text="a b c|d e f|" + " ".join(["g"] * 30), profile=profile, blocks=BlockRules()
    )
    chunks = _run(req)
    assert chunks[0].text.startswith("a b c|d e f")
    assert all(c.token_count <= 16 for c in chunks)
    # Pieces are cut on the custom separator, so no chunk starts mid-field.
    assert all(not c.text.startswith("|") for c in chunks)


# --- sentence_window ---------------------------------------------------------


def test_sentence_window_one_sentence_per_chunk_and_whole_code_blocks() -> None:
    chunks = _run(_req(strategy="sentence_window"))
    sentences = [c for c in chunks if c.kind == "sentence"]
    assert "Intro sentence one." in [c.text for c in sentences]
    code = [c for c in chunks if c.kind == "code"]
    assert len(code) == 1 and code[0].text.startswith("```py")
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1


# --- packer ------------------------------------------------------------------


def _piece(text: str, start: int = 0, kind: str = "prose") -> Piece:
    return Piece(start=start, end=start + len(text), text=text, kind=kind, tokens=TOK.count(text))


def test_pack_marks_mixed_kinds_and_keeps_part_only_for_single_piece_groups() -> None:
    text = "alpha beta\n\n- one\n"
    pieces = [
        Piece(0, 10, "alpha beta", "paragraph", TOK.count("alpha beta")),
        Piece(12, 18, "- one", "list", TOK.count("- one"), part=3),
    ]
    req = _req(text, max_tokens=100, min_tokens=0)
    chunks = pack(req, pieces, TOK)
    assert len(chunks) == 1 and chunks[0].kind == "mixed" and chunks[0].part is None
    solo = pack(req, [pieces[1]], TOK)
    assert solo[0].part == 3 and solo[0].kind == "list"


def test_pack_backward_merges_trailing_small_group() -> None:
    text = " ".join(["w"] * 30) + " tail"
    big = Piece(0, len(text) - 5, text[:-5], "prose", TOK.count(text[:-5]))
    small = Piece(len(text) - 4, len(text), "tail", "prose", 2)
    req = _req(text, max_tokens=60, min_tokens=5)
    assert len(pack(req, [big, small], TOK)) == 1


def test_pack_synthetic_pieces_join_with_blank_lines() -> None:
    req = _req("irrelevant", max_tokens=100, min_tokens=0)
    a = Piece(0, 5, "row A", "table", 2, verbatim=False)
    b = Piece(0, 5, "row B", "table", 2, verbatim=False)
    assert pack(req, [a, b], TOK)[0].text == "row A\n\nrow B\n"


# --- registry ----------------------------------------------------------------


def test_registry_lists_base_strategies() -> None:
    assert {"structural", "window", "recursive", "sentence_window"} <= set(STRATEGY_REGISTRY)


def test_unknown_strategy_is_a_config_error() -> None:
    profile = ChunkProfile(name="x").model_copy(update={"strategy": "nope"})
    with pytest.raises(ChunkingConfigError, match="not available"):
        build_chunker(profile, TOK)
