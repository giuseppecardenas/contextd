"""``ParsedFile.by_anchor`` tolerates anchors stored before the slug fix.

Every indexer phase that walks a stored Section row back to the parsed file
(summarise, relate, relink-lexical, chunk) looks the section up by the anchor
embedded in the stored id. Those ids were minted by the old slugifier, which
squeezed hyphen runs; the parser now emits GitHub's un-squeezed form, so an
exact lookup would miss every affected section and the phases would silently
skip them until a full re-index.
"""

from __future__ import annotations

from pathlib import Path

from contextd.indexer.heading_parser import ParsedSection
from contextd.indexer.units import ParsedFile


def _sec(anchor: str, title: str = "t") -> ParsedSection:
    return ParsedSection(
        title=title,
        level=2,
        anchor=anchor,
        body="",
        ordinal=0,
        parent_anchor=None,
        start_line=0,
        is_preamble=False,
    )


def _file(*anchors: str) -> ParsedFile:
    return ParsedFile(path=Path("x.md"), canonical="C:/x.md", sections=[_sec(a) for a in anchors])


def test_exact_anchor_wins() -> None:
    pf = _file("lod-1--lod-2", "other")
    assert pf.by_anchor("lod-1--lod-2") is pf.sections[0]


def test_pre_fix_anchor_resolves_to_github_form() -> None:
    pf = _file("621-aggregation-algorithm-lod-1--lod-2", "other")
    hit = pf.by_anchor("621-aggregation-algorithm-lod-1-lod-2")
    assert hit is pf.sections[0]


def test_github_anchor_resolves_against_pre_fix_parse() -> None:
    # The reverse direction: a stored ``--`` id against a single-hyphen parse.
    pf = _file("621-aggregation-algorithm-lod-1-lod-2")
    assert pf.by_anchor("621-aggregation-algorithm-lod-1--lod-2") is pf.sections[0]


def test_ambiguous_loose_match_is_a_miss() -> None:
    # Two sections that only differ by hyphen runs: never guess.
    pf = _file("a--b", "a-b-1")
    pf2 = _file("a--b", "a---b")
    assert pf2.by_anchor("a-b") is None
    assert pf.by_anchor("a-b") is pf.sections[0]


def test_unrelated_anchor_is_a_miss() -> None:
    pf = _file("lod-1--lod-2")
    assert pf.by_anchor("lod-3") is None
