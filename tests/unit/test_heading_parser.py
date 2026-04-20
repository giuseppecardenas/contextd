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
