"""Boundary invariants for the tree-sitter ``code`` strategy.

The parsing tests need the ``contextd[code]`` extra and skip without it; the
missing-dependency and fallback tests run everywhere.
"""

from __future__ import annotations

import sys
from itertools import pairwise

import pytest

from contextd.chunking.model import Chunk, ChunkRequest, LineIndex
from contextd.chunking.strategies import ChunkingConfigError, StrategyDeps
from contextd.chunking.strategies.code import CodeStrategy, make_code
from contextd.chunking.strategies.recursive import RecursiveStrategy
from contextd.chunking.tokenizer import WordTokenizer
from contextd.corpus_config import BlockRules, ChunkProfile

TOK = WordTokenizer()

_PY_MODULE = '''"""Module docstring."""

import os

# helper one
def alpha(x, y):
    total = x + y
    return total * 2


def beta(items):
    out = []
    for item in items:
        out.append(item.strip())
    return out


@decorated
def gamma(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class Widget:
    """A widget."""

    def __init__(self, name):
        self.name = name

    def render(self):
        return f"<{self.name}>"
'''

_RS_MODULE = """//! Crate docs.

use std::collections::HashMap;

/// Adds two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub struct Counter {
    counts: HashMap<String, usize>,
}

impl Counter {
    pub fn new() -> Self {
        Counter { counts: HashMap::new() }
    }

    pub fn bump(&mut self, key: &str) {
        *self.counts.entry(key.to_string()).or_insert(0) += 1;
    }
}
"""

_TS_MODULE = """import { readFile } from "fs/promises";

// A typed helper.
export interface Config {
  name: string;
  retries: number;
}

export async function load(path: string): Promise<Config> {
  const raw = await readFile(path, "utf-8");
  return JSON.parse(raw) as Config;
}

export class Loader {
  constructor(private readonly path: string) {}

  async run(): Promise<Config> {
    return load(this.path);
  }
}
"""

_UNICODE_MODULE = '''# Größe und Länge — Kommentar
def größe(wert):
    """Berechnet die Größe: 日本語のドキュメント."""
    return f"größe={wert} — 世界 🌍"


def länge(ñandú):
    return len(ñandú) + len("héllo wörld")


class Ünïcode:
    emoji = "🌍🚀"

    def naïve(self):
        return "café"
'''


def _profile(
    max_tokens: int = 40, min_tokens: int = 8, chunk_lines_overlap: int = 2
) -> ChunkProfile:
    return ChunkProfile(
        name="t",
        strategy="code",
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        chunk_lines_overlap=chunk_lines_overlap,
    )


def _req(text: str, suffix: str, profile: ChunkProfile | None = None) -> ChunkRequest:
    return ChunkRequest(
        text=text, profile=profile or _profile(), blocks=BlockRules(), suffix=suffix
    )


@pytest.fixture
def strategy() -> CodeStrategy:
    pytest.importorskip("tree_sitter_language_pack")
    chunker = make_code(_profile(), TOK, StrategyDeps())
    assert isinstance(chunker, CodeStrategy)
    return chunker


def _assert_invariants(req: ChunkRequest, chunks: list[Chunk]) -> None:
    assert chunks
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.token_count <= req.profile.max_tokens for c in chunks)
    assert all(c.token_count == TOK.count(c.text) for c in chunks)
    for a, b in pairwise(chunks):
        assert b.span.start_line >= a.span.start_line
    joined = "\n".join(c.text for c in chunks)
    for line in req.text.splitlines():
        if line.strip():
            assert line.strip() in joined, f"line lost: {line!r}"


def test_python_chunks_align_to_definitions(strategy: CodeStrategy) -> None:
    req = _req(_PY_MODULE, ".py")
    chunks = strategy.chunk(req)
    _assert_invariants(req, chunks)
    assert len(chunks) > 1
    for c in chunks:
        first = c.text.splitlines()[0]
        assert first.startswith(("def ", "class ", "#", "@", "import ", '"""')), first
        assert not first.startswith(" ")
        # A definition-aligned chunk never ends mid-statement.
        assert not c.text.rstrip().endswith((",", "(", "+", "="))
    # Every top-level definition opens a chunk or sits whole inside one.
    for name in ("def alpha", "def beta", "def gamma", "class Widget"):
        assert any(name in c.text for c in chunks)


def test_oversize_leaf_is_line_split_with_overlap(strategy: CodeStrategy) -> None:
    body = "\n".join(f"line {i} of the very long docstring text here" for i in range(30))
    text = f'def doc():\n    """\n{body}\n    """\n    return 1\n'
    # Each docstring line is 11 word-tokens, so five lines fill a part and a
    # two-line tail (22) stays under the half-budget cap on repeated lines.
    req = _req(text, ".py", _profile(max_tokens=60, min_tokens=8, chunk_lines_overlap=2))
    chunks = strategy.chunk(req)
    _assert_invariants(req, chunks)
    parts = [c for c in chunks if c.part is not None]
    assert [c.part for c in parts] == list(range(1, len(parts) + 1))
    assert len(parts) >= 3
    for prev, nxt in pairwise(parts):
        prev_lines = prev.text.rstrip("\n").splitlines()
        nxt_lines = nxt.text.splitlines()
        # The next part opens with the previous part's last two lines.
        assert nxt_lines[:2] == prev_lines[-2:]
    assert all(c.kind == "code" for c in chunks)


def test_zero_line_overlap_repeats_nothing(strategy: CodeStrategy) -> None:
    body = "\n".join(f"line {i} of the very long docstring text here" for i in range(30))
    text = f'def doc():\n    """\n{body}\n    """\n'
    req = _req(text, ".py", _profile(max_tokens=40, min_tokens=8, chunk_lines_overlap=0))
    chunks = strategy.chunk(req)
    parts = [c for c in chunks if c.part is not None]
    assert len(parts) >= 2
    seen: set[str] = set()
    for c in parts:
        for line in c.text.splitlines():
            if line.startswith("line "):
                assert line not in seen
                seen.add(line)


@pytest.mark.parametrize(
    ("text", "suffix", "opener"),
    [
        (_RS_MODULE, ".rs", ("pub ", "impl ", "use ", "//")),
        (_TS_MODULE, ".ts", ("export ", "import ", "//")),
    ],
)
def test_other_languages_parse(
    strategy: CodeStrategy, text: str, suffix: str, opener: tuple[str, ...]
) -> None:
    req = _req(text, suffix, _profile(max_tokens=30, min_tokens=4))
    chunks = strategy.chunk(req)
    _assert_invariants(req, chunks)
    assert len(chunks) > 1
    assert chunks[0].text.splitlines()[0].startswith(opener)


def test_non_ascii_offsets_stay_aligned(strategy: CodeStrategy) -> None:
    req = _req(_UNICODE_MODULE, ".py", _profile(max_tokens=24, min_tokens=4))
    chunks = strategy.chunk(req)
    _assert_invariants(req, chunks)
    assert len(chunks) > 1
    index = LineIndex(req.text)
    for c in chunks:
        # The chunk is an exact slice of the source, and the slice starts on
        # the line the span reports.
        assert c.text in req.text
        assert req.text.index(c.text) == index.offset_of_line(c.span.start_line)
    assert any("🌍🚀" in c.text for c in chunks)
    assert any("def größe" in c.text for c in chunks)


def test_max_tokens_respected_on_dense_source(strategy: CodeStrategy) -> None:
    text = "\n".join(f"value_{i} = compute({i}, {i + 1}, {i + 2})" for i in range(200)) + "\n"
    req = _req(text, ".py", _profile(max_tokens=16, min_tokens=2))
    chunks = strategy.chunk(req)
    _assert_invariants(req, chunks)
    assert len(chunks) > 10


def test_pieces_cover_all_non_blank_source(strategy: CodeStrategy) -> None:
    req = _req(_PY_MODULE, ".py")
    pieces = strategy.pieces(req)
    assert pieces is not None
    covered = [False] * len(req.text)
    for p in pieces:
        for i in range(p.start, p.end):
            covered[i] = True
    for i, ch in enumerate(req.text):
        if not ch.isspace():
            assert covered[i], f"offset {i} ({ch!r}) not covered"


def test_unknown_suffix_falls_back_to_recursive(strategy: CodeStrategy) -> None:
    text = "Plain prose paragraph one.\n\n" + " ".join(f"w{i}" for i in range(120)) + "\n"
    req = _req(text, ".txt")
    got = strategy.chunk(req)
    want = RecursiveStrategy(TOK).chunk(req)
    assert [c.text for c in got] == [c.text for c in want]
    assert [c.span for c in got] == [c.span for c in want]
    assert strategy.pieces(req) is None


def test_unknown_suffix_fallback_needs_no_grammar() -> None:
    def no_parser(_name: str) -> object:
        raise AssertionError("parser must not be requested for an unknown suffix")

    strategy = CodeStrategy(TOK, no_parser)  # type: ignore[arg-type]
    chunks = strategy.chunk(_req("some plain text here\n", ".txt"))
    assert chunks and chunks[0].text.startswith("some plain")


def test_empty_text_yields_no_chunks() -> None:
    def no_parser(_name: str) -> object:
        raise AssertionError("never called")

    strategy = CodeStrategy(TOK, no_parser)  # type: ignore[arg-type]
    assert strategy.chunk(_req("  \n\n", ".py")) == []


def test_make_code_raises_when_extra_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    with pytest.raises(ChunkingConfigError, match=r"pip install 'contextd\[code\]'"):
        make_code(_profile(), TOK, StrategyDeps())
