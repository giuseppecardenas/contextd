"""Markdown heading extractor powering section-granularity mode (§5.11).

Uses markdown-it-py's AST to identify headings at qualifying levels,
compute GitHub-style anchors, carve body ranges, and emit parent /
sibling ordinal metadata. Downstream consumers turn these into Section
node upserts + CONTAINS / PARENT_OF / NEXT_SIBLING edges.
"""

from __future__ import annotations

import hashlib
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


def section_hash(section: ParsedSection) -> str:
    """The on-graph Section content hash: md5 over title + blank line + body.

    This exact formula is persisted as ``Section.hash`` and compared on every
    incremental pass and daemon sweep — changing it invalidates every stored
    section hash and forces a full re-index of section-granular corpora.
    """
    return hashlib.md5((section.title + "\n\n" + section.body).encode("utf-8")).hexdigest()


_NON_ALNUM = re.compile(r"[^\w\s-]")
_WHITESPACE = re.compile(r"\s+")

# Title for a preamble that contains no heading of its own to borrow a name
# from — a document opening with front matter or bare prose.
_PREAMBLE_TITLE = "Preamble"

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

        # Collect (level, title, line_index) for every heading, then narrow to
        # the qualifying levels. The unfiltered list is kept so the preamble can
        # borrow the document's own title, which is typically an H1 and so sits
        # below min_level.
        all_heads: list[tuple[int, str, int]] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "heading_open":
                level = int(tok.tag[1])
                inline = tokens[i + 1]
                assert tok.map is not None
                all_heads.append((level, _extract_title(inline), tok.map[0]))
                i += 3  # heading_open, inline, heading_close
                continue
            i += 1

        heads = [h for h in all_heads if self._min <= h[0] <= self._max]

        sections: list[ParsedSection] = []
        stack: list[ParsedSection] = []  # ancestors
        sibling_ordinals: dict[str | None, int] = {}

        # Track seen anchors for GitHub-style dedup (foo, foo-1, foo-2 …).
        seen_anchors: dict[str, int] = {}

        # Everything above the first qualifying heading — the document title and
        # its opening prose — belongs to no heading. The File node carries no
        # content of its own in section mode, so without this that text is
        # indexed nowhere: not embedded, not summarised, not searchable. When a
        # document has no qualifying heading at all the whole file is preamble,
        # which also rescues files that would otherwise yield zero sections.
        #
        # The body is bounded explicitly at the first qualifying heading rather
        # than by the equal-or-shallower rule below, because a document whose
        # shallowest heading is deeper than min_level (all ### under min=2) has
        # no such bound and the preamble would swallow the entire file.
        boundary = heads[0][2] if heads else len(lines)
        preamble_body = "".join(lines[:boundary])
        if preamble_body.strip():
            preamble_title = next(
                (t for lvl, t, ln in all_heads if ln < boundary and lvl < self._min),
                _PREAMBLE_TITLE,
            )
            preamble_anchor = _github_anchor(preamble_title)
            seen_anchors[preamble_anchor] = 1
            sections.append(
                ParsedSection(
                    title=preamble_title,
                    level=self._min,
                    anchor=preamble_anchor,
                    body=preamble_body,
                    ordinal=0,
                    parent_anchor=None,
                )
            )
            # Deliberately not pushed onto `stack`: the preamble is a sibling of
            # the top-level sections, not their parent, so no existing
            # PARENT_OF edge changes. It does take ordinal 0, shifting the
            # top-level sibling chain down by one.
            sibling_ordinals[None] = 1

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
