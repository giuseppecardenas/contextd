"""Section-target fallback resolution (`_resolve_section_fallback` ladder).

A Section's PK is the absolute canonical ``path#anchor`` id, which the model
has no way to emit; before this ladder existed, virtually every Section
citation (``§12.2.5``, a title, a relative path) was silently dropped and the
cross-section graph was empty. Each rung is unique-only: ambiguity resolves to
``None`` (drop), never a guess.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from contextd.indexer.phases import _resolve_existing_node

SEC_ID = "C:/Users/x/corpus/docs/03-economy.md#1225-trade-routes"


def test_exact_pk_match_short_circuits() -> None:
    store = MagicMock()
    store.exec_read.return_value = [{"v": SEC_ID}]
    got = _resolve_existing_node(store, "Section", SEC_ID, "c")
    assert got == SEC_ID
    assert store.exec_read.call_count == 1


def test_anchor_fragment_from_relative_path_citation() -> None:
    store = MagicMock()
    # exact PK miss, then anchor-fragment hit
    store.exec_read.side_effect = [[], [{"v": SEC_ID}]]
    got = _resolve_existing_node(store, "Section", "docs/03-economy.md#1225-trade-routes", "c")
    assert got == SEC_ID
    predicate_call = store.exec_read.call_args_list[1]
    assert predicate_call.args[1]["a"] == "1225-trade-routes"
    assert predicate_call.args[1]["c"] == "c"


def test_title_citation_resolves_via_slugified_anchor() -> None:
    store = MagicMock()
    # exact PK miss, slugified-anchor hit
    store.exec_read.side_effect = [[], [{"v": SEC_ID}]]
    got = _resolve_existing_node(store, "Section", "§12.2.5 Trade Routes", "c")
    assert got == SEC_ID
    # "§12.2.5 Trade Routes" slugifies to "1225-trade-routes" (§ stripped first)
    assert store.exec_read.call_args_list[1].args[1]["a"] == "1225-trade-routes"


def test_title_match_is_case_insensitive() -> None:
    store = MagicMock()
    # exact PK miss, slug-anchor miss, title hit
    store.exec_read.side_effect = [[], [], [{"v": SEC_ID}]]
    got = _resolve_existing_node(store, "Section", "Trade Route Decay", "c")
    assert got == SEC_ID
    title_call = store.exec_read.call_args_list[2]
    assert "toLower" in title_call.args[0]


def test_dotted_number_resolves_by_title_prefix(caplog: pytest.LogCaptureFixture) -> None:
    store = MagicMock()
    # exact PK miss, slug-anchor miss, title miss, numbered-prefix hit
    store.exec_read.side_effect = [[], [], [], [{"v": SEC_ID}]]
    with caplog.at_level(logging.INFO, logger="contextd.indexer.phases"):
        got = _resolve_existing_node(store, "Section", "§12.2.5", "c")
    assert got == SEC_ID
    assert "numbered-title prefix" in caplog.text
    prefix_call = store.exec_read.call_args_list[3]
    assert prefix_call.args[1]["t"] == "12.2.5"
    # Space-suffixed prefix so 12.2.5 cannot match a 12.2.50 heading.
    assert "STARTS WITH ($t + ' ')" in prefix_call.args[0]


def test_ambiguous_match_returns_none() -> None:
    store = MagicMock()
    # exact PK miss, then two rows at every fallback rung
    two = [{"v": SEC_ID}, {"v": SEC_ID + "-1"}]
    store.exec_read.side_effect = [[], two, two, two]
    got = _resolve_existing_node(store, "Section", "12.2.5", "c")
    assert got is None


def test_non_numeric_needle_skips_prefix_rung() -> None:
    store = MagicMock()
    store.exec_read.side_effect = [[], [], []]
    got = _resolve_existing_node(store, "Section", "Trade Routes", "c")
    assert got is None
    # exact + slug-anchor + title only; no fourth (prefix) query
    assert store.exec_read.call_count == 3


def test_fallback_queries_exclude_pathless_stubs() -> None:
    store = MagicMock()
    store.exec_read.side_effect = [[], [{"v": SEC_ID}]]
    _resolve_existing_node(store, "Section", "docs/x.md#a", "c")
    assert "n.path IS NOT NULL" in store.exec_read.call_args_list[1].args[0]


def test_github_double_hyphen_link_matches_pre_fix_anchor_within_file() -> None:
    """``path#a--b`` against a Section stored as ``a-b`` by the old slugifier.

    The exact-fragment rung misses; the hyphen-run-tolerant rung compares the
    squeezed forms and is scoped to the cited file so a same-named heading in
    another file cannot win.
    """
    store = MagicMock()
    stored = "C:/Users/x/corpus/docs/03a-economy-core.md#621-aggregation-algorithm-lod-1-lod-2"
    store.exec_read.side_effect = [[], [], [{"v": stored}]]
    got = _resolve_existing_node(
        store,
        "Section",
        "C:/Users/x/corpus/docs/03a-economy-core.md#621-aggregation-algorithm-lod-1--lod-2",
        "c",
    )
    assert got == stored
    tolerant = store.exec_read.call_args_list[2]
    assert "replace(" in tolerant.args[0]
    assert "n.path = $p" in tolerant.args[0]
    assert tolerant.args[1]["a"] == "621-aggregation-algorithm-lod-1-lod-2"
    assert tolerant.args[1]["p"] == "C:/Users/x/corpus/docs/03a-economy-core.md"


def test_bare_fragment_citation_tolerant_rung_is_corpus_wide() -> None:
    store = MagicMock()
    store.exec_read.side_effect = [[], [], [{"v": SEC_ID}]]
    got = _resolve_existing_node(store, "Section", "#1225--trade-routes", "c")
    assert got == SEC_ID
    tolerant = store.exec_read.call_args_list[2]
    assert "n.path = $p" not in tolerant.args[0]
    assert tolerant.args[1]["a"] == "1225-trade-routes"


def test_slugified_title_rung_compares_squeezed_anchors() -> None:
    # The title's GitHub slug is ``lod-1--lod-2``; the rung must still hit a
    # pre-fix ``lod-1-lod-2`` anchor, so it sends the squeezed key and compares
    # against the squeezed stored anchor.
    store = MagicMock()
    store.exec_read.side_effect = [[], [{"v": SEC_ID}]]
    got = _resolve_existing_node(store, "Section", "§6.2.1 Aggregation (LOD 1 → LOD 2)", "c")
    assert got == SEC_ID
    slug_call = store.exec_read.call_args_list[1]
    assert "replace(" in slug_call.args[0]
    assert slug_call.args[1]["a"] == "621-aggregation-lod-1-lod-2"
