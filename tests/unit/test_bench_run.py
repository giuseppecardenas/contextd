"""Tests for the bench runner (contextd.bench.run) with a patched ``search``."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from contextd.bench.metrics import Target
from contextd.bench.run import BenchReport, row_to_target, run_bench
from contextd.bench.spec import BenchQuery, BenchSpec
from contextd.config import SearchConfig


def _chunk_row(path: str, parent_id: str, start: int, end: int) -> dict[str, Any]:
    return {
        "unit": "chunk",
        "id": f"{parent_id}~fine~0",
        "path": path,
        "parent_id": parent_id,
        "parent_label": "Section" if "#" in parent_id else "File",
        "profile": "fine",
        "score": 0.5,
        "evidence": {"chunk_id": f"{parent_id}~fine~0", "start_line": start, "end_line": end},
    }


def _section_row(path: str, anchor: str, start: int | None = None) -> dict[str, Any]:
    ev: dict[str, Any] = {"chunk_id": "x"}
    if start is not None:
        ev.update(start_line=start, end_line=start + 3)
    return {"unit": "section", "id": f"{path}#{anchor}", "path": path, "score": 0.4, "evidence": ev}


def _file_row(path: str) -> dict[str, Any]:
    return {"unit": "file", "id": path, "path": path, "score": 0.3, "evidence": {"chunk_id": "y"}}


def test_row_to_target_chunk_section_file() -> None:
    chunk = row_to_target(_chunk_row("/c/a.md", "/c/a.md#intro", 4, 9))
    assert chunk == Target("/c/a.md", anchor="intro", lines=(4, 9))
    # File-parented chunk: no anchor, still carries lines.
    assert row_to_target(_chunk_row("/c/b.md", "/c/b.md", 0, 2)) == Target("/c/b.md", lines=(0, 2))
    assert row_to_target(_section_row("/c/a.md", "usage", 10)) == Target(
        "/c/a.md", anchor="usage", lines=(10, 13)
    )
    assert row_to_target(_section_row("/c/a.md", "usage")) == Target("/c/a.md", anchor="usage")
    assert row_to_target(_file_row("/c/z.md")) == Target("/c/z.md")
    # Flat node row (non-Chunk kind) and degenerate evidence.
    assert row_to_target({"path": "q.md", "score": 1.0}) == Target("q.md")
    assert row_to_target(
        {"unit": "chunk", "path": "q.md", "parent_id": "q.md", "evidence": {"start_line": 3}}
    ) == Target("q.md")


def test_run_bench_scores_and_per_query_k() -> None:
    spec = BenchSpec(
        queries=[
            BenchQuery("hydration", [Target("a.md", anchor="intro", lines=(0, 10))]),
            BenchQuery("cooking", [Target("a.md"), Target("b.md")], k=2),
            BenchQuery("nothing", [Target("zzz.md")]),
        ]
    )
    canned: dict[str, list[dict[str, Any]]] = {
        "hydration": [_chunk_row("/c/a.md", "/c/a.md#intro", 5, 15), _file_row("/c/b.md")],
        "cooking": [_file_row("/c/b.md"), _section_row("/c/a.md", "intro")],
        "nothing": [_file_row("/c/b.md")],
    }
    calls: list[dict[str, Any]] = []

    def fake_search(store: Any, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append({"query": query, **kwargs})
        return canned[query]

    store = MagicMock()
    embedder = MagicMock()
    cfg = SearchConfig(rrf_k=42, fetch_k=77, window=3)
    with patch("contextd.bench.run.search", side_effect=fake_search):
        report = run_bench(
            store,
            spec,
            embedder=embedder,
            search_cfg=cfg,
            corpus="notes",
            profiles=["fine", "coarse"],
            return_unit="auto",
            k=5,
            profile_weights={"fine": 1.0, "coarse": 0.5},
        )

    # search wiring: per-query k override, config knobs, window forced to 0.
    assert [c["limit"] for c in calls] == [5, 2, 5]
    for c in calls:
        assert c["rrf_k"] == 42 and c["fetch_k"] == 77 and c["window"] == 0
        assert c["corpus"] == "notes" and c["profiles"] == ["fine", "coarse"]
        assert c["profile_weights"] == {"fine": 1.0, "coarse": 0.5}
        assert c["return_unit"] == "auto" and c["embedder"] is embedder

    s0, s1, s2 = report.scores
    assert (s0.recall, s0.precision, s0.mrr) == (1.0, 0.5, 1.0)
    # expected lines 0..9, hit 5..14 -> overlap 5, union 15.
    assert s0.iou == 5 / 15
    assert (s1.recall, s1.precision, s1.mrr, s1.iou) == (1.0, 1.0, 1.0, None)
    assert (s2.recall, s2.precision, s2.mrr, s2.iou) == (0.0, 0.0, 0.0, None)

    assert report.summary["queries"] == 3
    assert report.summary["recall"] == 2 / 3
    assert report.summary["iou"] == 5 / 15
    assert len(report.latencies_ms) == 3
    assert report.summary["latency_ms"] is not None
    assert report.summary["latency_ms"] >= 0.0
    assert report.config == {
        "corpus": "notes",
        "profiles": ["fine", "coarse"],
        "return_unit": "auto",
        "k": 5,
        "mode": "hybrid",
        "embedder": True,
        "expand": "none",
        "graph_weight": 2.0,
    }
    assert report.label == "fine,coarse/auto@5"


def test_run_bench_no_profiles_no_embedder() -> None:
    spec = BenchSpec(queries=[BenchQuery("q", [Target("a.md")])])
    with patch("contextd.bench.run.search", return_value=[]) as mock_search:
        report = run_bench(
            MagicMock(),
            spec,
            embedder=None,
            search_cfg=SearchConfig(),
            corpus=None,
            profiles=None,
            return_unit="chunk",
            k=3,
        )
    assert mock_search.call_args.kwargs["profiles"] is None
    assert mock_search.call_args.kwargs["profile_weights"] is None
    assert report.config["profiles"] is None
    assert report.config["embedder"] is False
    assert report.label == "all/chunk@3"
    assert report.scores[0].recall == 0.0


def test_report_to_dict_round_trips_and_is_json() -> None:
    spec = BenchSpec(queries=[BenchQuery("q", [Target("a.md")])])
    with patch("contextd.bench.run.search", return_value=[_file_row("/c/a.md")]):
        report = run_bench(
            MagicMock(),
            spec,
            embedder=None,
            search_cfg=SearchConfig(),
            corpus="c",
            profiles=["fine"],
            return_unit="file",
            k=1,
        )
    data = report.to_dict()
    text = json.dumps(data)  # must be serialisable as-is
    assert set(data) == {"config", "scores", "summary"}
    assert data["scores"][0]["query"] == "q"
    assert data["scores"][0]["latency_ms"] == report.latencies_ms[0]
    back = BenchReport.from_dict(json.loads(text))
    assert back.scores == report.scores
    assert back.config == report.config
    assert back.summary == report.summary
    assert back.latencies_ms == report.latencies_ms


def test_run_bench_threads_expand_and_graph_weight_and_labels_row() -> None:
    spec = BenchSpec(queries=[BenchQuery("q", [Target("a.md")])])
    cfg = SearchConfig(expand="none", expand_seeds=4, graph_weight=1.0)
    with patch("contextd.bench.run.search", return_value=[]) as search:
        report = run_bench(
            MagicMock(),
            spec,
            embedder=None,
            search_cfg=cfg,
            corpus="c",
            profiles=["fine"],
            return_unit="file",
            k=5,
            expand="units",
            graph_weight=2.0,
        )
    kwargs = search.call_args.kwargs
    assert kwargs["expand"] == "units" and kwargs["graph_weight"] == 2.0
    assert kwargs["expand_seeds"] == 4
    assert report.config["expand"] == "units" and report.config["graph_weight"] == 2.0
    assert report.label == "fine/file@5+graph(2.0)"
    # Round-trips through JSON with the new config keys intact.
    again = BenchReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert again.label == report.label

    with patch("contextd.bench.run.search", return_value=[]) as search:
        report = run_bench(
            MagicMock(),
            spec,
            embedder=None,
            search_cfg=cfg,
            corpus="c",
            profiles=None,
            return_unit="auto",
            k=5,
        )
    assert search.call_args.kwargs["expand"] == "none"
    assert report.label == "all/auto@5"
