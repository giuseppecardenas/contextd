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
from markdown_it.token import Token


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

# Token types whose .content contributes display text.
# "image" tokens carry the alt text in their .content field.
_TEXT_TOKEN_TYPES = {"text", "code_inline", "image"}


def _extract_title(inline: Token) -> str:
    """Return rendered display text from an inline heading token.

    Walks inline.children and collects .content from token types that
    carry display text (``text``, ``code_inline``).  Wrapping tokens
    like ``link_open``/``link_close``, ``em_open``, ``strong_open``,
    etc., are skipped — their enclosed ``text`` children are captured
    naturally by the walk.  Falls back to ``inline.content`` (the raw
    Markdown source) only when ``children`` is None or empty.
    """
    children = inline.children
    if children:
        parts = [tok.content for tok in children if tok.type in _TEXT_TOKEN_TYPES]
        if parts:
            return "".join(parts).strip()
    # Fallback: no children or none contributed text — use raw content.
    return inline.content.strip()


def _github_anchor(title: str) -> str:
    lowered = title.lower()
    stripped = _NON_ALNUM.sub("", lowered)
    dashed = _WHITESPACE.sub("-", stripped).strip("-")
    # Defect fix: punctuation-only headings produce an empty anchor;
    # fall back to "section" so PKs remain non-empty.  Anchor dedup
    # in parse() handles the resulting collision.
    return dashed if dashed else "section"


class HeadingParser:
    def __init__(self, min_level: int, max_level: int) -> None:
        if not (1 <= min_level <= max_level <= 6):
            raise ValueError(
                "min_level and max_level must satisfy 1 <= min_level <= max_level <= 6"
            )
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
                    title = _extract_title(inline)
                    assert tok.map is not None
                    heads.append((level, title, tok.map[0]))
                i += 3  # heading_open, inline, heading_close
                continue
            i += 1

        sections: list[ParsedSection] = []
        stack: list[ParsedSection] = []  # ancestors
        sibling_ordinals: dict[str | None, int] = {}

        # Track seen anchors for GitHub-style dedup (foo, foo-1, foo-2 …).
        seen_anchors: dict[str, int] = {}

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

            # Compute unique anchor — deduplicate GitHub-style.
            # Route around any collision with a manually-authored heading that
            # already claimed the candidate suffix (e.g. ## foo-1 before the
            # dedup'd ## foo would emit foo-1).
            raw_anchor = _github_anchor(title)
            if raw_anchor not in seen_anchors:
                anchor = raw_anchor
                seen_anchors[raw_anchor] = 1
            else:
                count = seen_anchors[raw_anchor]
                while f"{raw_anchor}-{count}" in seen_anchors:
                    count += 1
                anchor = f"{raw_anchor}-{count}"
                seen_anchors[raw_anchor] = count + 1
                seen_anchors[anchor] = 1

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
