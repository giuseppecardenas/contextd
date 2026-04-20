"""Append-only JSON-lines session log of provider usage.

One file per UTC day under ~/.contextd/state/session-log/YYYY-MM-DD.jsonl.
Consumed by ``contextd costs [--since <date>]`` for reporting.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from contextd.providers.base import UsageRecord


class CostLog:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def append(self, record: UsageRecord) -> None:
        day = record.timestamp[:10]  # YYYY-MM-DD
        path = self._root / f"{day}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def aggregate(self, *, since: str | None = None) -> dict[str, dict[str, int]]:
        """Sum input/output tokens per provider for log entries on or after ``since``."""
        cutoff = datetime.fromisoformat(since) if since else None
        totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0}
        )
        for file in sorted(self._root.glob("*.jsonl")):
            day = datetime.fromisoformat(file.stem)
            if cutoff and day.date() < cutoff.date():
                continue
            for line in file.read_text().splitlines():
                row = json.loads(line)
                totals[row["provider"]]["input_tokens"] += row["input_tokens"]
                totals[row["provider"]]["output_tokens"] += row["output_tokens"]
        return dict(totals)
