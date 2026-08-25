"""``code`` — syntax-tree chunking for source files (LlamaIndex ``CodeSplitter``
/ Chonkie ``CodeChunker`` algorithm) on top of tree-sitter.

The parent text is parsed with the grammar selected by the request suffix
(:data:`LANGUAGE_BY_SUFFIX`, resolved through ``tree-sitter-language-pack``).
The root node's children are walked in order: a node whose text fits
``max_tokens`` becomes one :class:`Piece`; an oversize node recurses into its
children; an oversize leaf (a long string literal, a minified line) is
hard-split into line groups that fit, each group repeating the previous
group's last ``chunk_lines_overlap`` lines (``verbatim=False``, ``part=n``).
Source between sibling nodes that the grammar did not claim (blank runs are
dropped, everything else is kept) is emitted as its own piece, so the union of
pieces covers every non-blank character of the source. The pieces are then
packed by :func:`contextd.chunking.strategies.base.pack`, so small definitions
merge up to ``max_tokens`` / ``min_tokens`` like every other strategy.

tree-sitter reports **byte** offsets; they are translated to character
offsets through a per-parse prefix table so non-ASCII sources stay aligned.

Unknown suffixes fall back to the ``recursive`` cascade with the plain
separators. The dependency is the ``contextd[code]`` extra; it is
imported lazily by :func:`make_code` so the module itself is always
importable.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, cast

from contextd.chunking.model import Chunk, ChunkRequest
from contextd.chunking.strategies.base import ChunkStrategy, Piece, pack, verbatim_piece
from contextd.chunking.strategies.recursive import PLAIN_SEPARATORS, split_recursive
from contextd.chunking.strategies.window import window_spans
from contextd.chunking.tokenizer import Tokenizer

if TYPE_CHECKING:
    from contextd.chunking.strategies import StrategyDeps
    from contextd.corpus_config import ChunkProfile

_log = logging.getLogger(__name__)

LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "bash",
    ".lua": "lua",
    ".sql": "sql",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}
"""File suffix → ``tree-sitter-language-pack`` grammar name."""

_MISSING_DEPENDENCY = (
    "strategy 'code' requires the optional dependency: pip install 'contextd[code]'"
)


# The strategy only touches this sliver of the tree-sitter API, spelled out as
# protocols so the module type-checks without the optional extra installed.
class SyntaxNode(Protocol):
    @property
    def start_byte(self) -> int: ...

    @property
    def end_byte(self) -> int: ...

    @property
    def children(self) -> Sequence[SyntaxNode]: ...


class SyntaxTree(Protocol):
    @property
    def root_node(self) -> SyntaxNode: ...


class SyntaxParser(Protocol):
    def parse(self, source: bytes, /) -> SyntaxTree: ...


ParserFactory = Callable[[str], SyntaxParser]
"""``get_parser(language_name)``; raises ``LookupError`` for unknown names."""


def _load_parser_factory() -> ParserFactory:
    """Import the optional ``tree-sitter-language-pack`` extra or raise a
    config-time error."""
    try:
        import tree_sitter_language_pack as pack  # type: ignore[import-not-found,unused-ignore]
    except ImportError as exc:
        from contextd.chunking.strategies import ChunkingConfigError

        raise ChunkingConfigError(_MISSING_DEPENDENCY) from exc
    return cast(ParserFactory, pack.get_parser)


class _ByteToChar:
    """Map UTF-8 byte offsets of ``text`` back to character offsets.

    Identity for pure-ASCII text; otherwise a prefix table of the byte offset
    at which every character starts, bisected per lookup. tree-sitter node
    boundaries always fall on character boundaries, so ``bisect_left`` lands
    exactly.
    """

    def __init__(self, text: str, data: bytes) -> None:
        self._starts: list[int] | None = None
        if len(data) != len(text):
            starts = [0] * (len(text) + 1)
            pos = 0
            for i, ch in enumerate(text):
                starts[i] = pos
                pos += len(ch.encode("utf-8"))
            starts[len(text)] = pos
            self._starts = starts

    def __call__(self, byte_offset: int) -> int:
        if self._starts is None:
            return byte_offset
        return bisect.bisect_left(self._starts, byte_offset)


class CodeStrategy:
    name = "code"

    def __init__(self, tokenizer: Tokenizer, parser_factory: ParserFactory) -> None:
        self._tok = tokenizer
        # A tree-sitter parser is not safe to share across threads and the
        # language object behind it is cached by the pack, so one parser is
        # built per ``chunk`` call rather than memoised on the strategy.
        self._get_parser = parser_factory

    # -- fallback ------------------------------------------------------------

    def _recursive(self, req: ChunkRequest) -> list[Chunk]:
        pieces = split_recursive(
            req.text,
            0,
            len(req.text),
            PLAIN_SEPARATORS,
            self._tok,
            max_tokens=req.profile.max_tokens,
            kind="code",
        )
        return pack(req, pieces, self._tok)

    # -- piece builders ------------------------------------------------------

    def _line_units(
        self, text: str, start: int, end: int, max_tokens: int
    ) -> list[tuple[int, int]]:
        """Character spans of the lines in ``text[start:end]``; a line that is
        itself over ``max_tokens`` is replaced by token-window sub-spans."""
        units: list[tuple[int, int]] = []
        pos = start
        while pos < end:
            nl = text.find("\n", pos, end)
            line_end = end if nl == -1 else nl + 1
            if text[pos:line_end].strip():
                if self._tok.count(text[pos:line_end]) <= max_tokens:
                    units.append((pos, line_end))
                else:
                    units.extend(
                        (pos + s, pos + e)
                        for s, e in window_spans(
                            text[pos:line_end],
                            self._tok,
                            max_tokens=max_tokens,
                            overlap_tokens=0,
                        )
                    )
            pos = line_end
        return units

    def _split_lines(self, req: ChunkRequest, start: int, end: int) -> list[Piece]:
        """Hard-split an oversize leaf into line groups under ``max_tokens``,
        repeating the previous group's last ``chunk_lines_overlap`` lines.

        The repeated tail is trimmed from the front until it plus the next
        line fits the budget, and never exceeds half of ``max_tokens`` so
        every part carries at least half a chunk of new material.
        """
        max_tokens = req.profile.max_tokens
        overlap = req.profile.chunk_lines_overlap
        units = self._line_units(req.text, start, end, max_tokens)
        if not units:
            return []
        counts = [self._tok.count(req.text[s:e]) for s, e in units]
        groups: list[list[int]] = []
        cur: list[int] = []
        cur_tokens = 0
        for i, n in enumerate(counts):
            if cur and cur_tokens + n > max_tokens:
                groups.append(cur)
                tail = cur[-overlap:] if overlap > 0 else []
                while tail and (
                    sum(counts[j] for j in tail) + n > max_tokens
                    or sum(counts[j] for j in tail) > max_tokens // 2
                ):
                    tail.pop(0)
                cur = list(tail)
                cur_tokens = sum(counts[j] for j in cur)
            cur.append(i)
            cur_tokens += n
        groups.append(cur)
        if len(groups) == 1:
            s, e = units[groups[0][0]][0], units[groups[0][-1]][1]
            return [verbatim_piece(req.text, s, e, "code", self._tok)]
        pieces: list[Piece] = []
        for part, group in enumerate(groups, start=1):
            s, e = units[group[0]][0], units[group[-1]][1]
            text = req.text[s:e]
            pieces.append(
                Piece(
                    start=s,
                    end=e,
                    text=text,
                    kind="code",
                    tokens=self._tok.count(text),
                    verbatim=False,
                    part=part,
                )
            )
        return pieces

    def _span(self, req: ChunkRequest, start: int, end: int) -> list[Piece]:
        """Pieces for an unclaimed span (gap or leaf), trimmed of blank runs."""
        text = req.text
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            return []
        piece = verbatim_piece(text, start, end, "code", self._tok)
        if piece.tokens <= req.profile.max_tokens:
            return [piece]
        return self._split_lines(req, start, end)

    def _node(self, req: ChunkRequest, node: SyntaxNode, to_char: _ByteToChar) -> list[Piece]:
        start, end = to_char(node.start_byte), to_char(node.end_byte)
        piece = verbatim_piece(req.text, start, end, "code", self._tok)
        if piece.tokens <= req.profile.max_tokens:
            return [piece] if piece.text.strip() else []
        children = node.children
        if not children:
            return self._span(req, start, end)
        return self._walk(req, children, start, end, to_char)

    def _walk(
        self,
        req: ChunkRequest,
        children: Sequence[SyntaxNode],
        start: int,
        end: int,
        to_char: _ByteToChar,
    ) -> list[Piece]:
        """Pieces for ``children`` in order, plus every gap between them so the
        span ``[start, end)`` is fully covered."""
        out: list[Piece] = []
        pos = start
        for child in children:
            c_start = to_char(child.start_byte)
            if c_start > pos:
                out.extend(self._span(req, pos, c_start))
            out.extend(self._node(req, child, to_char))
            pos = max(pos, to_char(child.end_byte))
        if pos < end:
            out.extend(self._span(req, pos, end))
        return out

    # -- entry point ---------------------------------------------------------

    def pieces(self, req: ChunkRequest) -> list[Piece] | None:
        """Syntax-aligned pieces, or ``None`` when no grammar matches the suffix."""
        language = LANGUAGE_BY_SUFFIX.get(req.suffix.lower())
        if language is None:
            return None
        try:
            parser = self._get_parser(language)
        except (LookupError, ValueError) as exc:
            _log.warning("code chunker: no grammar for %s (%s); using recursive", language, exc)
            return None
        data = req.text.encode("utf-8")
        tree = parser.parse(data)
        to_char = _ByteToChar(req.text, data)
        root = tree.root_node
        return self._walk(req, root.children, 0, len(req.text), to_char)

    def chunk(self, req: ChunkRequest) -> list[Chunk]:
        if not req.text.strip():
            return []
        pieces = self.pieces(req)
        if pieces is None:
            return self._recursive(req)
        # Line overlap is the code-specific carry-forward; the sentence tail
        # the packer would otherwise prepend is meaningless for source.
        return pack(req, pieces, self._tok, overlap_tokens=0)


def make_code(_profile: ChunkProfile, tokenizer: Tokenizer, _deps: StrategyDeps) -> ChunkStrategy:
    """Registry factory: build a :class:`CodeStrategy` or raise
    :class:`~contextd.chunking.strategies.ChunkingConfigError` when the
    ``contextd[code]`` extra is not installed."""
    return CodeStrategy(tokenizer, _load_parser_factory())


__all__ = [
    "LANGUAGE_BY_SUFFIX",
    "CodeStrategy",
    "ParserFactory",
    "make_code",
]
