from __future__ import annotations

from contextd.bench.metrics import (
    QueryScore,
    Target,
    line_iou,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    satisfies,
    summarise,
)


def test_satisfies_path_anchor_and_lines() -> None:
    assert satisfies(Target("/abs/notes/a.md"), Target("a.md"))
    assert satisfies(Target("notes/a.md"), Target("/abs/notes/a.md"))
    assert not satisfies(Target("b.md"), Target("a.md"))
    assert satisfies(Target("a.md", anchor="x"), Target("a.md", anchor="x"))
    assert not satisfies(Target("a.md", anchor="y"), Target("a.md", anchor="x"))
    assert satisfies(Target("a.md"), Target("a.md", anchor="x"))  # hit has no anchor → path match
    assert satisfies(Target("a.md", lines=(5, 10)), Target("a.md", lines=(9, 20)))
    assert not satisfies(Target("a.md", lines=(5, 9)), Target("a.md", lines=(9, 20)))


def test_recall_precision_mrr() -> None:
    hits = [Target("x.md"), Target("a.md"), Target("b.md"), Target("a.md")]
    expected = [Target("a.md"), Target("b.md"), Target("c.md")]
    assert recall_at_k(hits, expected, 3) == 2 / 3
    assert recall_at_k(hits, expected, 1) == 0.0
    assert precision_at_k(hits, expected, 4) == 3 / 4
    assert reciprocal_rank(hits, expected) == 0.5
    assert reciprocal_rank([Target("z.md")], expected) == 0.0
    assert recall_at_k(hits, [], 3) == 1.0
    assert precision_at_k([], expected, 3) == 0.0


def test_line_iou() -> None:
    hits = [Target("a.md", lines=(0, 10)), Target("b.md", lines=(0, 5))]
    expected = [Target("a.md", lines=(5, 15))]
    # a.md: expected 5..14, hit 0..9 -> overlap 5..9 (5 lines); union 0..14 (15) + b.md 5 = 20.
    assert line_iou(hits, expected, k=2) == 5 / 20
    assert line_iou([Target("a.md")], [Target("a.md")], k=1) is None


def test_summarise() -> None:
    scores = [QueryScore("q1", 1.0, 0.5, 1.0, 0.5), QueryScore("q2", 0.0, 0.0, 0.0, None)]
    s = summarise(scores)
    assert s == {"recall": 0.5, "precision": 0.25, "mrr": 0.5, "iou": 0.5, "queries": 2}
    assert summarise([])["queries"] == 0
