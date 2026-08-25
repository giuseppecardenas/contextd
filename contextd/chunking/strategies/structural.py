"""``structural`` — heading-bounded, size-capped, block-aware (the default).

Blocks come from :mod:`contextd.chunking.blocks`; each is turned into one or
more :class:`Piece` according to the corpus ``[chunking.blocks]`` rules:

* fences are never split mid-fence — an oversize fence is split on blank
  lines and every slice is re-fenced with the original markup and language;
* tables split by row with the header + separator repeated in every slice
  (``rows_with_header``), stay whole (``whole``), or fall through to the
  prose fallback (``prose``);
* lists split between items;
* oversize prose falls back to the recursive cascade or a token window.

Pieces are then packed by :func:`contextd.chunking.strategies.base.pack`.
"""

from __future__ import annotations

from contextd.chunking.blocks import Block, segment
from contextd.chunking.model import Chunk, ChunkRequest, LineIndex
from contextd.chunking.strategies.base import Piece, pack, verbatim_piece
from contextd.chunking.strategies.recursive import PLAIN_SEPARATORS, split_recursive
from contextd.chunking.strategies.window import window_spans
from contextd.chunking.tokenizer import Tokenizer

_FALLBACK_SEPARATORS = ["\n\n", "\n", "sentence", " ", ""]
_ATOMIC_KINDS = frozenset({"heading", "hr", "html", "other"})


class StructuralStrategy:
    name = "structural"

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    # -- per-block piece builders ------------------------------------------

    def _prose(self, req: ChunkRequest, start: int, end: int, kind: str) -> list[Piece]:
        piece = verbatim_piece(req.text, start, end, kind, self._tok)
        if piece.tokens <= req.profile.max_tokens:
            return [piece]
        if req.blocks.sentence_fallback == "window":
            return [
                verbatim_piece(req.text, start + s, start + e, kind, self._tok)
                for s, e in window_spans(
                    req.text[start:end],
                    self._tok,
                    max_tokens=req.profile.max_tokens,
                    overlap_tokens=0,
                )
            ]
        return split_recursive(
            req.text,
            start,
            end,
            _FALLBACK_SEPARATORS if kind != "paragraph" else PLAIN_SEPARATORS,
            self._tok,
            max_tokens=req.profile.max_tokens,
            kind=kind,
        )

    def _fence(self, req: ChunkRequest, block: Block, start: int, end: int) -> list[Piece]:
        cap = req.blocks.max_fence_tokens or req.profile.max_tokens
        piece = verbatim_piece(req.text, start, end, "code", self._tok)
        if not req.blocks.protect_code_fences or piece.tokens <= cap:
            return [piece]
        lines = block.text.splitlines(keepends=True)
        open_line, body, close_line = lines[0], lines[1:-1], lines[-1]
        # Split the body on blank lines, then greedily regroup under the cap.
        paragraphs: list[list[str]] = [[]]
        for ln in body:
            if not ln.strip():
                if paragraphs[-1]:
                    paragraphs.append([])
                continue
            paragraphs[-1].append(ln)
        frame = self._tok.count(open_line + close_line)
        budget = max(1, cap - frame)
        # A paragraph that is itself over budget is re-split into line groups
        # that fit, so no slice ever exceeds the cap by construction.
        units: list[list[str]] = []
        for para in (p for p in paragraphs if p):
            if self._tok.count("".join(para)) <= budget:
                units.append(para)
                continue
            group: list[str] = []
            group_tokens = 0
            for ln in para:
                n = self._tok.count(ln)
                if group and group_tokens + n > budget:
                    units.append(group)
                    group, group_tokens = [], 0
                group.append(ln)
                group_tokens += n
            if group:
                units.append(group)
        pieces: list[Piece] = []
        cur: list[str] = []
        cur_tokens = 0

        def flush() -> None:
            if not cur:
                return
            while cur and not cur[-1].strip():
                cur.pop()
            text = open_line + "".join(cur) + close_line
            pieces.append(
                Piece(
                    start=start,
                    end=end,
                    text=text,
                    kind="code",
                    tokens=self._tok.count(text),
                    verbatim=False,
                    part=len(pieces) + 1,
                )
            )

        for unit in units:
            n = self._tok.count("".join(unit))
            if cur and cur_tokens + n > budget:
                flush()
                cur, cur_tokens = [], 0
            cur.extend([*unit, "\n"])
            cur_tokens += n
        flush()
        return pieces

    def _table(self, req: ChunkRequest, block: Block, start: int, end: int) -> list[Piece]:
        piece = verbatim_piece(req.text, start, end, "table", self._tok)
        mode = req.blocks.table_mode
        if mode == "whole" or piece.tokens <= req.profile.max_tokens:
            return [piece]
        if mode == "prose":
            return self._prose(req, start, end, "table")
        lines = block.text.splitlines(keepends=True)
        raw_header_n = block.meta.get("header_lines")
        header_n = raw_header_n if isinstance(raw_header_n, int) else 2
        header, rows = lines[:header_n], lines[header_n:]
        header_text = "".join(header)
        header_tokens = self._tok.count(header_text)
        budget = max(1, req.profile.max_tokens - header_tokens)
        pieces: list[Piece] = []
        cur: list[str] = []
        cur_tokens = 0
        for row in rows:
            n = self._tok.count(row)
            if cur and cur_tokens + n > budget:
                text = header_text + "".join(cur)
                pieces.append(
                    Piece(start, end, text, "table", self._tok.count(text), False, len(pieces) + 1)
                )
                cur, cur_tokens = [], 0
            cur.append(row)
            cur_tokens += n
        if cur:
            text = header_text + "".join(cur)
            pieces.append(
                Piece(start, end, text, "table", self._tok.count(text), False, len(pieces) + 1)
            )
        return pieces

    def _list(self, req: ChunkRequest, block: Block, index: LineIndex) -> list[Piece]:
        start = index.offset_of_line(block.start_line)
        end = index.offset_of_line(block.end_line)
        piece = verbatim_piece(req.text, start, end, "list", self._tok)
        if piece.tokens <= req.profile.max_tokens or not block.items:
            return (
                [piece]
                if piece.tokens <= req.profile.max_tokens
                else self._prose(req, start, end, "list")
            )
        pieces: list[Piece] = []
        for a, b in block.items:
            s, e = index.offset_of_line(a), index.offset_of_line(b)
            pieces.extend(self._prose(req, s, e, "list"))
        return pieces

    # -- entry point ---------------------------------------------------------

    def pieces(self, req: ChunkRequest) -> list[Piece]:
        index = LineIndex(req.text)
        out: list[Piece] = []
        for block in segment(req.text, req.suffix):
            start = index.offset_of_line(block.start_line)
            end = index.offset_of_line(block.end_line)
            if block.kind == "fence":
                out.extend(self._fence(req, block, start, end))
            elif block.kind == "table":
                out.extend(self._table(req, block, start, end))
            elif block.kind == "list":
                out.extend(self._list(req, block, index))
            elif block.kind in _ATOMIC_KINDS:
                out.append(verbatim_piece(req.text, start, end, block.kind, self._tok))
            elif block.kind == "code":
                out.extend(self._prose(req, start, end, "code"))
            else:
                out.extend(self._prose(req, start, end, block.kind))
        return out

    def chunk(self, req: ChunkRequest) -> list[Chunk]:
        return pack(req, self.pieces(req), self._tok)
