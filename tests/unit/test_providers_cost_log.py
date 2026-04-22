import json
from pathlib import Path

from contextd.providers.base import UsageRecord
from contextd.providers.cost_log import CostLog


def test_append_and_read(tmp_path: Path) -> None:
    log = CostLog(tmp_path / "session-log")
    record = UsageRecord(
        provider="gemini",
        model="gemma-4-31b-it",
        call_site="summary",
        input_tokens=100,
        output_tokens=20,
        timestamp="2026-04-20T12:00:00+00:00",
    )
    log.append(record)

    # Single file per UTC day.
    files = list((tmp_path / "session-log").iterdir())
    assert len(files) == 1
    assert files[0].name == "2026-04-20.jsonl"

    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 100


def test_aggregate_since(tmp_path: Path) -> None:
    log = CostLog(tmp_path / "session-log")
    r1 = UsageRecord(
        provider="gemini",
        model="m",
        call_site="summary",
        input_tokens=100,
        output_tokens=20,
        timestamp="2026-04-19T12:00:00+00:00",
    )
    r2 = UsageRecord(
        provider="gemini",
        model="m",
        call_site="summary",
        input_tokens=200,
        output_tokens=40,
        timestamp="2026-04-20T12:00:00+00:00",
    )
    log.append(r1)
    log.append(r2)

    agg = log.aggregate(since="2026-04-20")
    assert agg["gemini"]["input_tokens"] == 200
    assert agg["gemini"]["output_tokens"] == 40
