import hashlib

from contextd.indexer.heading_parser import HeadingParser, ParsedSection, anchor_key


def test_extracts_h2_and_h3_within_bounds() -> None:
    md = """# File title

## First section

Body 1

### Subsection 1.1

Body 1.1

## Second section

Body 2
"""
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    # "File title" leads: the H1 is below min_level, so it heads the preamble
    # rather than being dropped. It reports min_level so it sorts as a sibling
    # of the top-level sections.
    assert [s.title for s in sections] == [
        "File title",
        "First section",
        "Subsection 1.1",
        "Second section",
    ]
    assert [s.level for s in sections] == [2, 2, 3, 2]


def test_respects_min_and_max_levels() -> None:
    md = "# H1\n\n## H2\n\n### H3\n\n#### H4\n\n##### H5"
    parser = HeadingParser(min_level=3, max_level=4)
    sections = parser.parse(md)
    # H3/H4 qualify. H1 and H2 sit above the first qualifying heading, so their
    # text is preserved in the preamble (named for the H1) instead of being
    # discarded; H5 is deeper than max_level and folds into H4's body.
    assert [s.title for s in sections] == ["H1", "H3", "H4"]


def test_anchor_matches_github_convention() -> None:
    md = "## §6.14.9 Pricing tiers"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    # GitHub anchor: lowercase, strip punctuation, spaces → dashes.
    assert sections[0].anchor == "6149-pricing-tiers"


def test_body_is_exclusive_of_child_sections() -> None:
    md = "## A\n\nbody a\n\n### A.1\n\nbody a.1\n\n## B\n\nbody b"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    # A's body is exclusive: its own prose only, ending at the next promoted
    # heading of any level. A.1's content lives solely in A.1.
    a = sections[0]
    assert "body a" in a.body
    assert "body a.1" not in a.body
    assert "body b" not in a.body
    a1 = sections[1]
    assert "body a.1" in a1.body


def test_parent_ordinals() -> None:
    md = "## A\n\n### A.1\n\n### A.2\n\n## B"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    # ordinal reflects document-order position among siblings with same parent.
    titles_and_ordinals = [(s.title, s.ordinal) for s in sections]
    assert titles_and_ordinals == [("A", 0), ("A.1", 0), ("A.2", 1), ("B", 1)]


def test_inline_link_in_heading_extracts_rendered_text() -> None:
    md = "## [Config reference](/docs/config.md)"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert sections[0].title == "Config reference"
    assert sections[0].anchor == "config-reference"


def test_inline_code_in_heading_preserves_identifier() -> None:
    md = "## Using `FileHasher` directly"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert sections[0].title == "Using FileHasher directly"
    assert sections[0].anchor == "using-filehasher-directly"


def test_punctuation_only_heading_falls_back_to_section() -> None:
    md = "## ???"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert sections[0].anchor  # non-empty
    assert sections[0].anchor.startswith("section")


def test_anchor_keeps_hyphen_runs_like_github_slugger() -> None:
    """Stripped punctuation between two words leaves *both* spaces behind.

    github-slugger turns each space into a hyphen and never squeezes runs, so
    ``(LOD 1 → LOD 2)`` renders as ``lod-1--lod-2``. The parser used to
    collapse the run, which made every such heading unreachable from a
    GitHub-correct link (322 of 5,217 anchored links in one real corpus).
    """
    md = (
        "## 6.2.1 Aggregation algorithm (LOD 1 → LOD 2)\n\n"
        "## 16.1 Decision: Rust + Bevy\n\n"
        "## 3.2 Tile Scale: 1m × 1m × 3m\n\n"  # noqa: RUF001 -- real corpus heading
        "## A - B\n\n"
        "## ---"
    )
    parser = HeadingParser(min_level=2, max_level=4)
    anchors = [s.anchor for s in parser.parse(md)]
    assert anchors == [
        "621-aggregation-algorithm-lod-1--lod-2",
        "161-decision-rust--bevy",
        "32-tile-scale-1m--1m--3m",
        "a---b",
        "---",
    ]


def test_anchor_key_equates_pre_fix_and_github_forms() -> None:
    assert anchor_key("lod-1--lod-2") == anchor_key("lod-1-lod-2") == "lod-1-lod-2"
    assert anchor_key("a---b") == "a-b"
    assert anchor_key("-foo-") == "foo"
    # Plain anchors are their own key, so the tolerant path is a no-op for them.
    assert anchor_key("6149-pricing-tiers") == "6149-pricing-tiers"


def test_duplicate_titles_dedupe_anchors_like_github() -> None:
    md = "## Notes\n\nbody\n\n## Notes\n\nbody"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    anchors = [s.anchor for s in sections]
    assert anchors == ["notes", "notes-1"]


def test_duplicate_titles_do_not_cross_contaminate_child_ordinals() -> None:
    md = "## Section\n\n### Child A\n\n### Child B\n\n## Section\n\n### Child X\n\n### Child Y"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    titles_and_ordinals = [(s.title, s.ordinal) for s in sections]
    # Each "Section" has two children starting at ordinal 0.
    assert titles_and_ordinals == [
        ("Section", 0),
        ("Child A", 0),
        ("Child B", 1),
        ("Section", 1),  # dedup counter on the Section heading
        ("Child X", 0),  # ordinal restarts under the second (dedup'd) parent
        ("Child Y", 1),
    ]


def test_invalid_level_bounds_raise() -> None:
    import pytest

    with pytest.raises(ValueError, match="1 <= min_level <= max_level <= 6"):
        HeadingParser(min_level=5, max_level=2)
    with pytest.raises(ValueError, match="1 <= min_level <= max_level <= 6"):
        HeadingParser(min_level=0, max_level=4)
    with pytest.raises(ValueError, match="1 <= min_level <= max_level <= 6"):
        HeadingParser(min_level=2, max_level=7)


def test_manual_suffix_collision_routes_around() -> None:
    md = "## foo\n\n## foo-1\n\n## foo"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    anchors = [s.anchor for s in sections]
    # Third `foo` must NOT collide with the manually-authored `foo-1`;
    # it should skip to `foo-2`.
    assert anchors == ["foo", "foo-1", "foo-2"]


def test_image_heading_extracts_alt_text() -> None:
    md = "## ![Company logo](/img/logo.png)"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert sections[0].title == "Company logo"
    assert sections[0].anchor == "company-logo"


def test_image_heading_with_empty_alt_falls_back_to_section() -> None:
    md = "## ![](/img/logo.png)"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    # No alt text → empty title → anchor falls back to "section".
    assert sections[0].anchor == "section"


# ---------------------------------------------------------------------------
# Preamble tests — text above the first qualifying heading
# ---------------------------------------------------------------------------


def test_preamble_captures_title_and_intro_prose() -> None:
    # In section mode the File node holds no content, so without a preamble
    # section this text is indexed nowhere.
    md = "# Economy\n\nThis document covers production chains.\n\n## Catalog\n\nbody"
    parser = HeadingParser(min_level=2, max_level=4)
    preamble = parser.parse(md)[0]
    assert preamble.title == "Economy"
    assert "# Economy" in preamble.body
    assert "This document covers production chains." in preamble.body
    # Bounded at the first qualifying heading.
    assert "body" not in preamble.body


def test_preamble_is_a_sibling_not_a_parent() -> None:
    # The preamble must not reparent top-level sections: they keep the
    # parent_anchor they had before the preamble existed.
    md = "# Title\n\nintro\n\n## A\n\nbody a\n\n## B\n\nbody b"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert [(s.title, s.parent_anchor) for s in sections] == [
        ("Title", None),
        ("A", None),
        ("B", None),
    ]


def test_preamble_takes_ordinal_zero_and_shifts_siblings() -> None:
    md = "# Title\n\nintro\n\n## A\n\n## B"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert [(s.title, s.ordinal) for s in sections] == [("Title", 0), ("A", 1), ("B", 2)]


def test_no_preamble_when_document_opens_on_a_qualifying_heading() -> None:
    md = "## A\n\nbody a\n\n## B\n\nbody b"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert [s.title for s in sections] == ["A", "B"]


def test_blank_lines_before_first_heading_do_not_emit_a_preamble() -> None:
    md = "\n\n   \n\n## A\n\nbody"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert [s.title for s in sections] == ["A"]


def test_document_with_no_qualifying_heading_yields_one_preamble() -> None:
    # Previously this returned zero sections, leaving the file with no content,
    # no embedding and no summary anywhere in the graph.
    md = "# Release notes\n\nEverything here is prose with no subheadings."
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert len(sections) == 1
    assert sections[0].title == "Release notes"
    assert "Everything here is prose" in sections[0].body


def test_preamble_does_not_swallow_file_when_headings_are_deeper_than_min() -> None:
    # The trap the explicit boundary guards: with min_level=2 and only H3s, no
    # later heading is equal-or-shallower, so the equal-or-shallower body rule
    # would run the preamble to end of file and duplicate the whole document.
    md = "intro prose\n\n### Deep A\n\nbody a\n\n### Deep B\n\nbody b"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    preamble = sections[0]
    assert "intro prose" in preamble.body
    assert "body a" not in preamble.body
    assert "body b" not in preamble.body
    assert [s.title for s in sections] == ["Preamble", "Deep A", "Deep B"]


def test_preamble_without_a_heading_falls_back_to_generic_title() -> None:
    md = "Loose front matter.\n\n## A\n\nbody"
    parser = HeadingParser(min_level=2, max_level=4)
    preamble = parser.parse(md)[0]
    assert preamble.title == "Preamble"
    assert preamble.anchor == "preamble"


def test_preamble_anchor_participates_in_dedup() -> None:
    md = "# Notes\n\nintro\n\n## Notes\n\nbody"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert [s.anchor for s in sections] == ["notes", "notes-1"]


# ---------------------------------------------------------------------------
# Section body hash contract tests
# ---------------------------------------------------------------------------


def _section_hash(sec: ParsedSection) -> str:
    # Deliberately a literal copy of heading_parser.section_hash, NOT an
    # import: this block freezes the on-graph hash contract, so it must fail
    # if the production formula ever changes. Importing the helper would make
    # these tests tautological.
    return hashlib.md5((sec.title + "\n\n" + sec.body).encode()).hexdigest()


def test_section_hash_is_stable_across_identical_parses() -> None:
    md = "## Overview\n\nIntro text.\n\n## Details\n\nMore info."
    parser = HeadingParser(min_level=2, max_level=4)
    sections1 = parser.parse(md)
    sections2 = parser.parse(md)
    assert len(sections1) == len(sections2)
    for s1, s2 in zip(sections1, sections2, strict=True):
        assert _section_hash(s1) == _section_hash(s2)


def test_section_hash_changes_on_body_edit() -> None:
    md1 = "## Overview\n\nOriginal text."
    md2 = "## Overview\n\nModified text."
    parser = HeadingParser(min_level=2, max_level=4)
    s1 = parser.parse(md1)[0]
    s2 = parser.parse(md2)[0]
    assert _section_hash(s1) != _section_hash(s2)


def test_section_hash_changes_on_title_change() -> None:
    md1 = "## Overview\n\nSame text."
    md2 = "## Introduction\n\nSame text."
    parser = HeadingParser(min_level=2, max_level=4)
    s1 = parser.parse(md1)[0]
    s2 = parser.parse(md2)[0]
    # Different titles → different hashes (title is in formula AND body contains heading line)
    assert _section_hash(s1) != _section_hash(s2)


def test_section_hash_unchanged_when_sibling_changes() -> None:
    md_before = "## A\n\nBody A.\n\n## B\n\nBody B original."
    md_after = "## A\n\nBody A.\n\n## B\n\nBody B changed."
    parser = HeadingParser(min_level=2, max_level=4)
    a_before = parser.parse(md_before)[0]
    a_after = parser.parse(md_after)[0]
    assert a_before.title == "A"
    assert a_after.title == "A"
    assert _section_hash(a_before) == _section_hash(a_after)


def test_empty_section_body_hash_is_consistent() -> None:
    md = "## Empty Section\n\n## Next Section\n\nSome text."
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    empty = sections[0]
    assert empty.title == "Empty Section"
    h = _section_hash(empty)
    assert isinstance(h, str) and len(h) == 32
    # Stable on re-parse
    empty2 = parser.parse(md)[0]
    assert _section_hash(empty2) == h


def test_parent_body_excludes_children_and_child_edit_keeps_parent_hash() -> None:
    md = "## Parent\n\nParent intro.\n\n### Child\n\nChild content.\n\n## Sibling\n\nSibling text."
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    parent = sections[0]
    assert "Parent intro." in parent.body
    assert "Child content." not in parent.body
    assert "Sibling text." not in parent.body

    # Disjoint-hash guarantee: editing child content must NOT change the
    # parent's hash, so incremental re-index touches only the edited section.
    md2 = (
        "## Parent\n\nParent intro.\n\n### Child\n\nChild MODIFIED.\n\n## Sibling\n\nSibling text."
    )
    parent2 = parser.parse(md2)[0]
    assert _section_hash(parent) == _section_hash(parent2)


def test_sections_carry_file_start_lines() -> None:
    from contextd.indexer.heading_parser import HeadingParser

    md = "# Title\n\nintro\n\n## A\n\nbody a\n\n### A1\n\nbody a1\n\n## B\n\nbody b\n"
    sections = HeadingParser(min_level=2, max_level=4).parse(md)
    by_anchor = {s.anchor: s for s in sections}
    assert by_anchor["title"].start_line == 0  # preamble
    assert by_anchor["a"].start_line == 4
    assert by_anchor["a1"].start_line == 8
    assert by_anchor["b"].start_line == 12
    lines = md.splitlines(keepends=True)
    for s in sections:
        assert md.startswith(s.body, sum(len(ln) for ln in lines[: s.start_line]))
