"""Core chunk types shared by every strategy and the indexer phases."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from contextd.corpus_config import BlockRules, ChunkProfile


@dataclass(frozen=True)
class ChunkSpan:
    """Line range of a chunk within its *file*: ``start_line`` inclusive,
    ``end_line`` exclusive, both 0-based."""

    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 0 or self.end_line < self.start_line:
            raise ValueError(f"invalid span {self.start_line}..{self.end_line}")


@dataclass
class Chunk:
    """One retrieval chunk as produced by a strategy (pre-embedding).

    ``text`` is the raw body stored on the node and returned as evidence.
    ``prefix`` (breadcrumb / summary / LLM context) is prepended for embedding
    and full-text indexing but never mixed into ``text``. ``keywords`` is
    full-text only. ``embedding`` is set only by strategies that produce their
    own vectors (late chunking); the phase embeds every other chunk.
    """

    ordinal: int
    text: str
    span: ChunkSpan
    token_count: int
    kind: str = "prose"
    prefix: str = ""
    keywords: list[str] = field(default_factory=list)
    part: int | None = None
    embedding: list[float] | None = None

    @property
    def embed_text(self) -> str:
        return f"{self.prefix}\n\n{self.text}" if self.prefix else self.text


@dataclass(frozen=True)
class ChunkRequest:
    """Everything a strategy needs to chunk one parent unit.

    ``text`` is the parent's body (a Section body or a whole file) and
    ``base_line`` is the line of the file where ``text`` starts, so spans the
    strategy computes relative to ``text`` are translated into file lines.
    """

    text: str
    profile: ChunkProfile
    blocks: BlockRules
    base_line: int = 0
    breadcrumb: tuple[str, ...] = ()
    suffix: str = ".md"


class LineIndex:
    """Map character offsets of a text to 0-based line numbers.

    Built once per parent text; ``line_of`` is a bisect. A trailing partial
    line (no final newline) counts as a line.
    """

    def __init__(self, text: str) -> None:
        starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                starts.append(i + 1)
        self._starts = starts
        self._length = len(text)

    @property
    def line_count(self) -> int:
        if self._length == 0:
            return 0
        # A text ending in "\n" registers a phantom empty final line.
        return len(self._starts) - (1 if self._starts[-1] == self._length else 0)

    def line_of(self, offset: int) -> int:
        offset = max(0, min(offset, self._length))
        return bisect.bisect_right(self._starts, offset) - 1

    def span(self, start: int, end: int, *, base_line: int = 0) -> ChunkSpan:
        """Line span covering ``text[start:end]`` (end exclusive)."""
        if end <= start:
            line = self.line_of(start)
            return ChunkSpan(base_line + line, base_line + line)
        first = self.line_of(start)
        last = self.line_of(end - 1)
        return ChunkSpan(base_line + first, base_line + last + 1)
