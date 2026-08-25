"""Tests for bench report rendering and persistence (contextd.bench.report)."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from contextd.bench.metrics import QueryScore
from contextd.bench.report import load_report, render_diff, render_table, save_report
from contextd.bench.run import BenchReport


def _report(
    profiles: list[str] | None, *, recall: float, iou: float | None, latency: float
) -> BenchReport:
    scores = [QueryScore("q1", recall, 0.5, 1.0, iou), QueryScore("q2", recall, 0.25, 0.5, None)]
    return BenchReport(
        config={"corpus": "c", "profiles": profiles, "return_unit": "auto", "k": 5},
        scores=scores,
        summary={
            "recall": recall,
            "precision": 0.375,
            "mrr": 0.75,
            "iou": iou,
            "queries": 2,
            "latency_ms": latency,
        },
        latencies_ms=[latency, latency],
    )


def test_render_table_one_row_per_report() -> None:
    console = Console(record=True, width=160)
    render_table(
        [
            _report(["fine"], recall=0.5, iou=0.25, latency=12.34),
            _report(None, recall=1.0, iou=None, latency=3.0),
        ],
        console,
    )
    text = console.export_text()
    assert "fine/auto@5" in text
    assert "all/auto@5" in text
    assert "0.500" in text and "1.000" in text
    assert "0.250" in text
    assert "12.3" in text
    assert "—" in text  # IoU not applicable on the second row


def test_render_diff_signs() -> None:
    a = _report(["fine"], recall=0.5, iou=0.25, latency=10.0)
    b = _report(["coarse"], recall=0.75, iou=None, latency=8.0)
    console = Console(record=True, width=160)
    render_diff(a, b, console)
    text = console.export_text()
    assert "A: fine/auto@5" in text and "B: coarse/auto@5" in text
    assert "+0.250" in text  # recall went up
    assert "±0.000" in text  # precision / MRR unchanged
    assert "-2.0" in text  # latency went down
    # IoU is not applicable on B: no delta rendered for it.
    iou_line = next(line for line in text.splitlines() if "line IoU" in line)
    assert "—" in iou_line


def test_save_load_round_trip(tmp_path: Path) -> None:
    reports = [
        _report(["fine"], recall=0.5, iou=0.25, latency=1.5),
        _report(["fine", "coarse"], recall=0.9, iou=None, latency=2.5),
    ]
    out = tmp_path / "nested" / "bench.json"
    save_report(reports, out)
    raw = out.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    loaded = load_report(out)
    assert [r.to_dict() for r in loaded] == [r.to_dict() for r in reports]
    assert loaded[1].label == "fine,coarse/auto@5"

    single = tmp_path / "one.json"
    save_report(reports[0], single)
    assert len(load_report(single)) == 1


@pytest.mark.parametrize("body", ["not json", "[]", '{"reports": []}', '{"reports": [{}]}'])
def test_load_report_errors_name_file(tmp_path: Path, body: str) -> None:
    p = tmp_path / "bad.json"
    p.write_text(body, encoding="utf-8")
    if body == '{"reports": [{}]}':
        # An empty report dict is tolerated: no scores, empty config/summary.
        assert load_report(p)[0].scores == []
        return
    with pytest.raises(ValueError, match=r"bad\.json"):
        load_report(p)


def test_load_report_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"missing\.json"):
        load_report(tmp_path / "missing.json")
