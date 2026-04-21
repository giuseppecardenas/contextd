"""Tests for up / down / status CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import contextd.cli


def _setup_contextd_home(tmp_path: Path, backend: str = "memgraph") -> Path:
    home = tmp_path / ".contextd"
    home.mkdir()
    config = f"""
[storage]
backend = "{backend}"

[storage.{backend}]
docker_compose_file = "{home}/docker-compose.yml"
"""
    (home / "config.toml").write_text(config)
    (home / "corpora").mkdir()
    return home


def test_status_lists_corpora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path, backend="memgraph")
    (home / "corpora" / "demo.toml").write_text("""
[corpus]
name = "demo"
root = "/tmp/demo"
""")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    result = CliRunner().invoke(contextd.cli.cli, ["status"])
    assert result.exit_code == 0
    assert "demo" in result.output
    assert "memgraph" in result.output


def test_status_no_corpora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path, backend="memgraph")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    result = CliRunner().invoke(contextd.cli.cli, ["status"])
    assert result.exit_code == 0
    assert "memgraph" in result.output
    assert "0 registered" in result.output


def test_up_memgraph_calls_docker_compose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path, backend="memgraph")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    with (
        patch("subprocess.run") as mock_run,
        patch("contextd.storage.factory.build_graph_store") as mock_build,
    ):
        mock_run.return_value.returncode = 0
        fake_store = mock_build.return_value
        fake_store.connect.return_value = None
        fake_store.apply_migrations.return_value = None
        fake_store.close.return_value = None
        result = CliRunner().invoke(contextd.cli.cli, ["up"])
    assert result.exit_code == 0, result.output
    # Verify docker compose up -d was called with --profile memgraph.
    calls = [c.args[0] for c in mock_run.call_args_list]
    compose_calls = [c for c in calls if "compose" in c]
    assert any("--profile" in c and "memgraph" in c and "up" in c for c in compose_calls)


def test_down_memgraph_calls_docker_compose_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_contextd_home(tmp_path, backend="memgraph")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    with patch("subprocess.run") as mock_run:
        result = CliRunner().invoke(contextd.cli.cli, ["down"])
    assert result.exit_code == 0
    assert any("down" in c.args[0] for c in mock_run.call_args_list)


def test_up_memgraph_without_docker_raises_clickexception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When docker is absent and backend=memgraph, `up` must surface a
    clean ClickException instead of a FileNotFoundError traceback."""
    home = _setup_contextd_home(tmp_path, backend="memgraph")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    with patch("shutil.which", return_value=None):
        result = CliRunner().invoke(contextd.cli.cli, ["up"])
    assert result.exit_code != 0
    assert "docker not on PATH" in result.output
    # ClickException renders via "Error: ..." — not a bare Python traceback.
    assert "Traceback" not in result.output


def test_up_neo4j_calls_compose_with_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".contextd"
    home.mkdir()
    (home / "config.toml").write_text(
        '[storage]\nbackend = "neo4j"\n\n[storage.neo4j]\nhost = "127.0.0.1"\nport = 7687\n'
    )
    # The docker-compose template must exist for `up` to run.
    (home / "docker-compose.yml").write_text("services: {}\n")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))

    calls: list[list[str]] = []

    def fake_run(*args: object, **kwargs: object) -> object:
        calls.append(list(args[0]))  # type: ignore[arg-type]

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/docker")

    # Stub backend connect/migrate path so the test exercises CLI dispatch only.
    with patch("contextd.storage.factory.build_graph_store") as mock_build:
        fake_store = mock_build.return_value
        fake_store.connect.return_value = None
        fake_store.apply_migrations.return_value = None
        fake_store.close.return_value = None
        result = CliRunner().invoke(contextd.cli.cli, ["up"])

    assert result.exit_code == 0, result.output
    # At least one docker compose call with --profile neo4j and `up`.
    compose_calls = [c for c in calls if "compose" in c]
    assert any("--profile" in c and "neo4j" in c and "up" in c for c in compose_calls)
