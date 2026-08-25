"""Rendering and persistence for bench reports.

``render_table`` puts one configuration per row so profile / unit / k
choices can be read off side by side; ``render_diff`` shows the per-metric
delta between two saved runs (``contextd bench --compare``). JSON files
hold ``{"reports": [...]}`` so one ``--json`` file can carry every
configuration of a run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from contextd.bench.run import BenchReport

_METRICS: tuple[tuple[str, str], ...] = (
    ("recall", "recall@k"),
    ("precision", "precision@k"),
    ("mrr", "MRR"),
    ("iou", "line IoU"),
    ("latency_ms", "latency ms"),
)


def _fmt(value: float | None, *, key: str) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}" if key == "latency_ms" else f"{value:.3f}"


def _fmt_delta(a: float | None, b: float | None, *, key: str) -> str:
    """Signed delta ``b - a``; ``—`` when either side is not applicable."""
    if a is None or b is None:
        return "—"
    delta = b - a
    body = f"{abs(delta):.1f}" if key == "latency_ms" else f"{abs(delta):.3f}"
    if delta > 0:
        return f"+{body}"
    if delta < 0:
        return f"-{body}"
    return f"±{body}"


def _metric(report: BenchReport, key: str) -> float | None:
    value = report.summary.get(key)
    return None if value is None else float(value)


def render_table(reports: Sequence[BenchReport], console: Console) -> None:
    """One row per configuration: label, unit, k, query count, then metrics."""
    table = Table(title="contextd bench")
    table.add_column("configuration", style="bold")
    table.add_column("queries", justify="right")
    for _, heading in _METRICS:
        table.add_column(heading, justify="right")
    for report in reports:
        queries = report.summary.get("queries")
        table.add_row(
            report.label,
            "0" if queries is None else str(int(queries)),
            *(_fmt(_metric(report, key), key=key) for key, _ in _METRICS),
        )
    console.print(table)


def render_diff(a: BenchReport, b: BenchReport, console: Console) -> None:
    """Per-metric ``A`` / ``B`` / signed delta (``B - A``) table."""
    table = Table(title="contextd bench --compare")
    table.add_column("metric", style="bold")
    table.add_column(f"A: {a.label}", justify="right")
    table.add_column(f"B: {b.label}", justify="right")
    table.add_column("delta (B-A)", justify="right")
    for key, heading in _METRICS:
        va, vb = _metric(a, key), _metric(b, key)
        table.add_row(heading, _fmt(va, key=key), _fmt(vb, key=key), _fmt_delta(va, vb, key=key))
    console.print(table)


def save_report(report: BenchReport | Sequence[BenchReport], path: Path) -> None:
    """Write one or more reports as ``{"reports": [...]}`` JSON (UTF-8)."""
    reports = [report] if isinstance(report, BenchReport) else list(report)
    payload: dict[str, Any] = {"reports": [r.to_dict() for r in reports]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_report(path: Path) -> list[BenchReport]:
    """Read a file written by :func:`save_report`.

    Raises ``ValueError`` (naming the file) for unreadable or mis-shaped JSON.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read bench report {path}: {exc}") from exc
    reports_raw = data.get("reports") if isinstance(data, dict) else None
    if not isinstance(reports_raw, list) or not reports_raw:
        raise ValueError(f"bench report {path} has no 'reports' list")
    try:
        return [BenchReport.from_dict(r) for r in reports_raw]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed bench report {path}: {exc}") from exc
