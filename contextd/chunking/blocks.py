"""Markdown block segmentation with line maps.

The ``structural`` strategy packs *blocks*, never raw lines, so a chunk
boundary always falls between two markdown constructs. Segmentation reuses
markdown-it's top-level token stream (``token.map`` gives the line range of
every block) — the same parser ``HeadingParser`` runs, so what counts as a
fence or a table here is what counts upstream.

Block kinds: ``paragraph``, ``list``, ``fence`` (``meta["info"]`` = language),
``code`` (indented), ``table`` (``meta["header_lines"]`` = 2: header row +
separator), ``blockquote``, ``heading``, ``html``, ``hr``, ``other``. Lists
record their item line ranges so an oversize list can be split between
items. Non-markdown text goes through :func:`segment_plain`, which yields
``paragraph`` blocks on blank-line boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from markdown_it import MarkdownIt

_BLOCK_KIND_BY_TOKEN = {
    "paragraph_open": "paragraph",
    "bullet_list_open": "list",
    "ordered_list_open": "list",
    "fence": "fence",
    "code_block": "code",
    "table_open": "table",
    "blockquote_open": "blockquote",
    "heading_open": "heading",
    "html_block": "html",
    "hr": "hr",
}


@dataclass
class Block:
    kind: str
    start_line: int  # inclusive, relative to the segmented text
    end_line: int  # exclusive
    text: str
    meta: dict[str, object] = field(default_factory=dict)
    items: list[tuple[int, int]] = field(default_factory=list)
    """Line ranges of list items (``list`` blocks only)."""

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line


_md = MarkdownIt().enable("table")


def _join(lines: list[str], start: int, end: int) -> str:
    return "".join(lines[start:end])


def segment_markdown(text: str) -> list[Block]:
    """Split markdown into ordered top-level blocks covering every non-blank line."""
    if not text.strip():
        return []
    lines = text.splitlines(keepends=True)
    tokens = _md.parse(text)
    blocks: list[Block] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.level != 0 or tok.map is None:
            i += 1
            continue
        kind = _BLOCK_KIND_BY_TOKEN.get(tok.type, "other")
        start, end = tok.map
        end = min(end, len(lines))
        block = Block(kind=kind, start_line=start, end_line=end, text=_join(lines, start, end))
        if kind == "fence":
            block.meta["info"] = tok.info.strip()
            block.meta["markup"] = tok.markup
        elif kind == "table":
            block.meta["header_lines"] = 2
        elif kind == "list":
            # Item ranges come from the nested list_item_open tokens at level 1.
            j = i + 1
            while j < len(tokens) and not (
                tokens[j].level == 0 and tokens[j].type.endswith("_close")
            ):
                t = tokens[j]
                if t.level == 1 and t.type == "list_item_open" and t.map is not None:
                    block.items.append((t.map[0], min(t.map[1], len(lines))))
                j += 1
        blocks.append(block)
        # Skip to the matching close token for container blocks.
        if tok.nesting == 1:
            depth = 0
            while i < len(tokens):
                if tokens[i].nesting == 1:
                    depth += 1
                elif tokens[i].nesting == -1:
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
        i += 1
    return _coalesce_gaps(blocks, lines)


def _coalesce_gaps(blocks: list[Block], lines: list[str]) -> list[Block]:
    """markdown-it drops nothing that has content, but be defensive: any
    non-blank lines not covered by a block become ``other`` blocks so the
    concatenation of all blocks still covers the text."""
    covered = [False] * len(lines)
    for b in blocks:
        for ln in range(b.start_line, b.end_line):
            covered[ln] = True
    extra: list[Block] = []
    ln = 0
    while ln < len(lines):
        if covered[ln] or not lines[ln].strip():
            ln += 1
            continue
        start = ln
        while ln < len(lines) and not covered[ln] and lines[ln].strip():
            ln += 1
        extra.append(
            Block(kind="other", start_line=start, end_line=ln, text=_join(lines, start, ln))
        )
    if not extra:
        return blocks
    return sorted(blocks + extra, key=lambda b: b.start_line)


def segment_plain(text: str) -> list[Block]:
    """Paragraph blocks on blank-line boundaries for non-markdown text."""
    lines = text.splitlines(keepends=True)
    blocks: list[Block] = []
    ln = 0
    while ln < len(lines):
        if not lines[ln].strip():
            ln += 1
            continue
        start = ln
        while ln < len(lines) and lines[ln].strip():
            ln += 1
        blocks.append(
            Block(kind="paragraph", start_line=start, end_line=ln, text=_join(lines, start, ln))
        )
    return blocks


def segment(text: str, suffix: str) -> list[Block]:
    """Format router: markdown for ``.md`` / ``.markdown``, plain otherwise."""
    if suffix.lower() in (".md", ".markdown", ".mdx"):
        return segment_markdown(text)
    return segment_plain(text)
