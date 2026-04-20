from contextd.indexer.heading_parser import HeadingParser


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
    assert [s.title for s in sections] == ["First section", "Subsection 1.1", "Second section"]
    assert [s.level for s in sections] == [2, 3, 2]


def test_respects_min_and_max_levels() -> None:
    md = "# H1\n\n## H2\n\n### H3\n\n#### H4\n\n##### H5"
    parser = HeadingParser(min_level=3, max_level=4)
    sections = parser.parse(md)
    assert [s.title for s in sections] == ["H3", "H4"]


def test_anchor_matches_github_convention() -> None:
    md = "## §6.14.9 Feudal nobility"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    # GitHub anchor: lowercase, strip punctuation, spaces → dashes.
    assert sections[0].anchor == "6149-feudal-nobility"


def test_body_range_includes_content_until_next_equal_or_shallower() -> None:
    md = "## A\n\nbody a\n\n### A.1\n\nbody a.1\n\n## B\n\nbody b"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    # A's body spans everything up to B (inclusive of A.1's content).
    a = sections[0]
    assert "body a" in a.body
    assert "body a.1" in a.body
    assert "body b" not in a.body


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
    md = "## ---"
    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md)
    assert sections[0].anchor  # non-empty
    assert sections[0].anchor.startswith("section")


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
