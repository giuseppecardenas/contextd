from pathlib import Path

import pytest
from click.testing import CliRunner

import contextd.cli


def test_init_creates_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTD_HOME", str(tmp_path / ".contextd"))
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("VOYAGE_API_KEY", "y")
    runner = CliRunner()
    result = runner.invoke(contextd.cli.cli, ["init", "--yes"])
    assert result.exit_code == 0
    home = tmp_path / ".contextd"
    assert (home / "config.toml").exists()
    assert (home / "corpora").is_dir()
    assert (home / "state").is_dir()
    assert (home / "docker-compose.yml").exists()
    assert (home / "prompts" / "summarise.md").exists()


def test_init_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTD_HOME", str(tmp_path / ".contextd"))
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("VOYAGE_API_KEY", "y")
    runner = CliRunner()

    result1 = runner.invoke(contextd.cli.cli, ["init", "--yes"])
    assert result1.exit_code == 0

    result2 = runner.invoke(contextd.cli.cli, ["init", "--yes"])
    assert result2.exit_code == 0

    home = tmp_path / ".contextd"
    assert (home / "config.toml").exists()
    assert (home / "corpora").is_dir()
    assert (home / "docker-compose.yml").exists()
    assert (home / "prompts" / "summarise.md").exists()
    assert "already present" in result2.output
