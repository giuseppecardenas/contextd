from __future__ import annotations

from itertools import pairwise

from contextd.chunking.blocks import segment, segment_markdown, segment_plain

_DOC = (
    "Intro para.\n"
    "\n"
    "- a\n"
    "  more a\n"
    "- b\n"
    "\n"
    "```py\n"
    "x = 1\n"
    "```\n"
    "\n"
    "| h1 | h2 |\n"
    "|---|---|\n"
    "| 1 | 2 |\n"
    "\n"
    "> quote\n"
    "\n"
    "---\n"
    "\n"
    "    indented code\n"
    "\n"
    "<div>html</div>\n"
)


def test_kinds_and_line_maps() -> None:
    blocks = segment_markdown(_DOC)
    kinds = [b.kind for b in blocks]
    assert kinds == ["paragraph", "list", "fence", "table", "blockquote", "hr", "code", "html"]
    para = blocks[0]
    assert (para.start_line, para.end_line, para.text) == (0, 1, "Intro para.\n")
    fence = blocks[2]
    assert fence.meta == {"info": "py", "markup": "```"}
    assert fence.text == "```py\nx = 1\n```\n"
    table = blocks[3]
    assert table.meta["header_lines"] == 2
    assert table.text.startswith("| h1 | h2 |\n|---|---|\n")


def test_list_items_recorded() -> None:
    blocks = segment_markdown("- a\n  more a\n- b\n\n1. x\n2. y\n")
    assert [b.kind for b in blocks] == ["list", "list"]
    assert blocks[0].items == [(0, 2), (2, 4)]
    assert blocks[1].items == [(4, 5), (5, 6)]


def test_blocks_are_ordered_and_non_overlapping() -> None:
    blocks = segment_markdown(_DOC)
    for prev, nxt in pairwise(blocks):
        assert prev.end_line <= nxt.start_line


def test_every_non_blank_line_is_covered() -> None:
    lines = _DOC.splitlines()
    covered: set[int] = set()
    for b in segment_markdown(_DOC):
        covered.update(range(b.start_line, b.end_line))
    for i, line in enumerate(lines):
        if line.strip():
            assert i in covered, f"line {i} uncovered: {line!r}"


def test_fence_with_tildes_and_nested_backticks() -> None:
    doc = "~~~text\n```\ninner\n```\n~~~\n\nafter\n"
    blocks = segment_markdown(doc)
    assert [b.kind for b in blocks] == ["fence", "paragraph"]
    assert blocks[0].meta["markup"] == "~~~"
    assert blocks[0].text == "~~~text\n```\ninner\n```\n~~~\n"


def test_heading_inside_body_is_a_block() -> None:
    blocks = segment_markdown("##### deep\n\ntext\n")
    assert [b.kind for b in blocks] == ["heading", "paragraph"]


def test_empty_and_blank_text() -> None:
    assert segment_markdown("") == []
    assert segment_markdown("\n\n  \n") == []


def test_segment_plain_paragraphs() -> None:
    blocks = segment_plain("a\nb\n\n\nc\n")
    assert [(b.start_line, b.end_line, b.text) for b in blocks] == [
        (0, 2, "a\nb\n"),
        (4, 5, "c\n"),
    ]
    assert all(b.kind == "paragraph" for b in blocks)


def test_segment_routes_by_suffix() -> None:
    doc = "- a\n- b\n"
    assert segment(doc, ".md")[0].kind == "list"
    assert segment(doc, ".MD")[0].kind == "list"
    assert segment(doc, ".txt")[0].kind == "paragraph"
