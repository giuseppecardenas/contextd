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


def test_init_writes_neo4j_default_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After the M11.8 flip, fresh ``contextd init`` writes neo4j as default backend."""
    monkeypatch.setenv("CONTEXTD_HOME", str(tmp_path / ".contextd"))
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("VOYAGE_API_KEY", "y")
    runner = CliRunner()
    result = runner.invoke(contextd.cli.cli, ["init", "--yes"])
    assert result.exit_code == 0
    config = (tmp_path / ".contextd" / "config.toml").read_text(encoding="utf-8")
    assert 'backend = "neo4j"' in config


def test_init_refresh_prompts_overwrites_stale_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTEXTD_HOME", str(tmp_path / ".contextd"))
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("VOYAGE_API_KEY", "y")
    runner = CliRunner()
    assert runner.invoke(contextd.cli.cli, ["init", "--yes"]).exit_code == 0
    relate = tmp_path / ".contextd" / "prompts" / "relate.md"
    relate.write_text("stale template", encoding="utf-8")

    # Plain init leaves the stale copy alone (copy-if-missing contract).
    assert runner.invoke(contextd.cli.cli, ["init", "--yes"]).exit_code == 0
    assert relate.read_text(encoding="utf-8") == "stale template"

    # --refresh-prompts overwrites it with the packaged text and reports it.
    result = runner.invoke(contextd.cli.cli, ["init", "--yes", "--refresh-prompts"])
    assert result.exit_code == 0
    assert "refreshed" in result.output
    assert "relate.md" in result.output
    assert relate.read_text(encoding="utf-8") != "stale template"


def test_status_reports_prompt_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTD_HOME", str(tmp_path / ".contextd"))
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("VOYAGE_API_KEY", "y")
    runner = CliRunner()
    assert runner.invoke(contextd.cli.cli, ["init", "--yes"]).exit_code == 0
    prompts = tmp_path / ".contextd" / "prompts"
    (prompts / "relate.md").write_text("customised", encoding="utf-8")
    (prompts / "translate.md").unlink()

    result = runner.invoke(contextd.cli.cli, ["status"])

    assert result.exit_code == 0
    assert "prompts:" in result.output
    assert "differs from packaged" in result.output
    assert "MISSING" in result.output
    assert "matches packaged" in result.output  # the untouched templates


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
