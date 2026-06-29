"""Unit tests for reciprocal_rank_fusion — pure, no backend."""

from __future__ import annotations

from typing import Any

from contextd.search.fusion import flatten_row, reciprocal_rank_fusion


def _vrow(path: str, score: float) -> dict[str, Any]:
    return {"node": {"path": path, "summary": f"summary-{path}"}, "score": score}


def test_rrf_fuses_overlapping_nodes_to_the_top() -> None:
    """A node present in both rankers outranks nodes present in only one.

    vector: [a, b]   full-text: [b, c]
    b gets a contribution from both lists; a and c from one each, so b wins.
    """
    vector_rows = [_vrow("a", 0.9), _vrow("b", 0.8)]
    fulltext_rows = [_vrow("b", 5.0), _vrow("c", 2.0)]
    fused = reciprocal_rank_fusion(vector_rows, fulltext_rows, label="File", limit=10)
    paths = [r["path"] for r in fused]
    assert paths == ["b", "a", "c"]


def test_rrf_uses_rank_not_raw_score() -> None:
    """The ``score`` field on input rows is ignored; only rank position counts.

    ``a`` is rank 1 with a raw score of 0.0; ``b`` is rank 2 with a raw score
    of 1.0. If raw scores mattered, ``b`` would win — but RRF ranks ``a``
    first because it sits at rank 1.
    """
    vector_rows = [_vrow("a", 0.0), _vrow("b", 1.0)]
    fused = reciprocal_rank_fusion(vector_rows, [], label="File", limit=10)
    assert [r["path"] for r in fused] == ["a", "b"]


def test_rrf_weights_bias_modality() -> None:
    """A heavier vector weight promotes a vector-only hit over a full-text-only
    hit that shares the same rank."""
    vector_rows = [_vrow("v", 0.5)]
    fulltext_rows = [_vrow("f", 0.5)]
    biased = reciprocal_rank_fusion(
        vector_rows, fulltext_rows, label="File", limit=10, vector_weight=2.0
    )
    assert biased[0]["path"] == "v"
    flipped = reciprocal_rank_fusion(
        vector_rows, fulltext_rows, label="File", limit=10, fulltext_weight=2.0
    )
    assert flipped[0]["path"] == "f"


def test_rrf_tie_break_is_deterministic_and_vector_first() -> None:
    """Equal fused scores break by first-seen order; the vector list is folded
    first, so a tied vector hit precedes a tied full-text hit. Repeated calls
    are byte-identical."""
    vector_rows = [_vrow("v", 0.1)]
    fulltext_rows = [_vrow("f", 0.1)]
    runs = [
        reciprocal_rank_fusion(vector_rows, fulltext_rows, label="File", limit=10) for _ in range(5)
    ]
    assert all(r == runs[0] for r in runs)
    assert [row["path"] for row in runs[0]] == ["v", "f"]


def test_rrf_empty_vector_list_returns_fulltext_only() -> None:
    fused = reciprocal_rank_fusion([], [_vrow("a", 1.0), _vrow("b", 0.5)], label="File", limit=10)
    assert [r["path"] for r in fused] == ["a", "b"]


def test_rrf_empty_both_returns_empty() -> None:
    assert reciprocal_rank_fusion([], [], label="File", limit=10) == []


def test_rrf_truncates_to_limit() -> None:
    vector_rows = [_vrow(p, 1.0) for p in ("a", "b", "c", "d", "e")]
    fused = reciprocal_rank_fusion(vector_rows, [], label="File", limit=3)
    assert len(fused) == 3
    assert [r["path"] for r in fused] == ["a", "b", "c"]


def test_rrf_strips_embedding_and_flattens() -> None:
    vector_rows = [{"node": {"path": "a", "summary": "s", "embedding": [0.1] * 1024}, "score": 0.9}]
    fused = reciprocal_rank_fusion(vector_rows, [], label="File", limit=10)
    assert "embedding" not in fused[0]
    assert "node" not in fused[0]
    assert fused[0]["path"] == "a"
    assert fused[0]["summary"] == "s"
    assert isinstance(fused[0]["score"], float)


def test_rrf_keys_section_by_id() -> None:
    """Section nodes are keyed by ``id`` (not ``path``), so the same section in
    both rankers fuses into a single row."""
    vector_rows = [{"node": {"id": "a.md#intro", "summary": "x"}, "score": 0.9}]
    fulltext_rows = [{"node": {"id": "a.md#intro", "summary": "x"}, "score": 3.0}]
    fused = reciprocal_rank_fusion(vector_rows, fulltext_rows, label="Section", limit=10)
    assert len(fused) == 1
    assert fused[0]["id"] == "a.md#intro"


def test_rrf_skips_node_missing_primary_key() -> None:
    """A row whose node omits the label's PK is skipped, not crashed on."""
    vector_rows = [
        {"node": {"summary": "no path here"}, "score": 0.9},
        _vrow("real", 0.8),
    ]
    fused = reciprocal_rank_fusion(vector_rows, [], label="File", limit=10)
    assert [r["path"] for r in fused] == ["real"]


def test_flatten_row_drops_embedding_and_attaches_score() -> None:
    row = flatten_row({"path": "a", "summary": "s", "embedding": [0.0] * 4}, 0.42)
    assert row == {"path": "a", "summary": "s", "score": 0.42}
