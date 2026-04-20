"""Markdown heading extractor powering section-granularity mode (§5.11).

Uses markdown-it-py's AST to identify headings at qualifying levels,
compute GitHub-style anchors, carve body ranges, and emit parent /
sibling ordinal metadata. Downstream consumers turn these into Section
node upserts + CONTAINS / PARENT_OF / NEXT_SIBLING edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from markdown_it import MarkdownIt


@dataclass
class ParsedSection:
    title: str
    level: int
    anchor: str
    body: str
    ordinal: int
    parent_anchor: str | None


_NON_ALNUM = re.compile(r"[^\w\s-]")
_WHITESPACE = re.compile(r"\s+")


def _github_anchor(title: str) -> str:
    lowered = title.lower()
    stripped = _NON_ALNUM.sub("", lowered)
    dashed = _WHITESPACE.sub("-", stripped).strip("-")
    return dashed


class HeadingParser:
    def __init__(self, min_level: int, max_level: int) -> None:
        self._min = min_level
        self._max = max_level
        self._md = MarkdownIt()

    def parse(self, markdown: str) -> list[ParsedSection]:
        tokens = self._md.parse(markdown)
        lines = markdown.splitlines(keepends=True)

        # Collect (level, title, line_index) for qualifying headings.
        heads: list[tuple[int, str, int]] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "heading_open":
                level = int(tok.tag[1])
                if self._min <= level <= self._max:
                    inline = tokens[i + 1]
                    title = inline.content.strip()
                    assert tok.map is not None
                    heads.append((level, title, tok.map[0]))
                i += 3  # heading_open, inline, heading_close
                continue
            i += 1

        sections: list[ParsedSection] = []
        stack: list[ParsedSection] = []  # ancestors
        sibling_ordinals: dict[str | None, int] = {}

        for idx, (level, title, line) in enumerate(heads):
            # Trim stack to ancestors of strictly shallower level.
            while stack and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1] if stack else None
            parent_anchor = parent.anchor if parent else None
            ordinal = sibling_ordinals.get(parent_anchor, 0)
            sibling_ordinals[parent_anchor] = ordinal + 1
            # Compute body extent: lines from this heading until next heading
            # of equal or shallower level (or end of file).
            next_line_bound = len(lines)
            for k in range(idx + 1, len(heads)):
                if heads[k][0] <= level:
                    next_line_bound = heads[k][2]
                    break
            body = "".join(lines[line:next_line_bound])
            anchor = _github_anchor(title)
            section = ParsedSection(
                title=title,
                level=level,
                anchor=anchor,
                body=body,
                ordinal=ordinal,
                parent_anchor=parent_anchor,
            )
            sections.append(section)
            stack.append(section)

        return sections
