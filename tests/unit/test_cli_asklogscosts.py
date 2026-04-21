"""Tests for ask, logs, costs CLI commands (spec §8)."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import reload
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import contextd.cli
from contextd.providers.base import UsageRecord
from contextd.providers.cost_log import CostLog


def _setup_home(tmp_path: Path) -> Path:
    home = tmp_path / ".contextd"
    home.mkdir()
    config = f"""
[storage]
backend = "kuzu"

[storage.kuzu]
db_path = "{home}/graph"
"""
    (home / "config.toml").write_text(config)
    (home / "corpora").mkdir()
    (home / "logs").mkdir()
    (home / "state" / "session-log").mkdir(parents=True)
    return home


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


def test_logs_no_file_prints_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    result = CliRunner().invoke(contextd.cli.cli, ["logs"])
    assert result.exit_code == 0
    assert "no log at" in result.output


def test_logs_prints_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path)
    log_path = home / "logs" / "contextd.log"
    log_path.write_text('{"level":"info","msg":"hello"}\n')
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    result = CliRunner().invoke(contextd.cli.cli, ["logs"])
    assert result.exit_code == 0
    assert "hello" in result.output


def test_logs_follow_shells_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path)
    log_path = home / "logs" / "contextd.log"
    log_path.write_text("line\n")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    with patch("subprocess.run") as mock_run:
        result = CliRunner().invoke(contextd.cli.cli, ["logs", "--follow"])
    assert result.exit_code == 0
    assert mock_run.called
    called_args = mock_run.call_args[0][0]
    assert called_args[0] == "tail"
    assert "-f" in called_args
    assert str(log_path) in called_args


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------


def test_costs_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    result = CliRunner().invoke(contextd.cli.cli, ["costs"])
    assert result.exit_code == 0
    assert "no usage recorded yet" in result.output


def test_costs_shows_totals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path)
    session_log_dir = home / "state" / "session-log"
    cost_log = CostLog(session_log_dir)
    record = UsageRecord(
        provider="gemini",
        model="gemini-2.0-flash-exp",
        call_site="summary",
        input_tokens=100,
        output_tokens=50,
        timestamp=datetime.now(UTC).isoformat(),
    )
    cost_log.append(record)
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    result = CliRunner().invoke(contextd.cli.cli, ["costs"])
    assert result.exit_code == 0
    assert "gemini" in result.output
    assert "100" in result.output
    assert "50" in result.output


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_help_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_home(tmp_path)
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    result = CliRunner().invoke(contextd.cli.cli, ["ask", "--help"])
    assert result.exit_code == 0
    assert "QUESTION" in result.output
