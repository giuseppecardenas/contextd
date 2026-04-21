"""Tests for up / down / status CLI commands."""

from __future__ import annotations

from importlib import reload
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import contextd.cli


def _setup_contextd_home(tmp_path: Path, backend: str = "kuzu") -> Path:
    home = tmp_path / ".contextd"
    home.mkdir()
    if backend == "kuzu":
        # Use a nested path so the parent dir does NOT pre-exist — lets
        # `test_up_kuzu_creates_db_path` assert that `up` actually created it.
        config = f"""
[storage]
backend = "kuzu"

[storage.kuzu]
db_path = "{home}/nested/graph"
"""
    else:
        config = f"""
[storage]
backend = "memgraph"

[storage.memgraph]
docker_compose_file = "{home}/docker-compose.yml"
"""
    (home / "config.toml").write_text(config)
    (home / "corpora").mkdir()
    return home


def test_status_lists_corpora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path, backend="kuzu")
    (home / "corpora" / "demo.toml").write_text("""
[corpus]
name = "demo"
root = "/tmp/demo"
""")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    result = CliRunner().invoke(contextd.cli.cli, ["status"])
    assert result.exit_code == 0
    assert "demo" in result.output
    assert "kuzu" in result.output


def test_status_no_corpora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path, backend="kuzu")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    result = CliRunner().invoke(contextd.cli.cli, ["status"])
    assert result.exit_code == 0
    assert "kuzu" in result.output
    assert "0 registered" in result.output


def test_up_kuzu_creates_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Kuzu stores its DB in a single file (spec-delta #3), so `up` creates the
    # parent directory; the actual DB file is created by KuzuBackend.connect().
    home = _setup_contextd_home(tmp_path, backend="kuzu")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    result = CliRunner().invoke(contextd.cli.cli, ["up"])
    assert result.exit_code == 0, result.output
    # Parent dir did NOT pre-exist — `up` must have created it. Falsifiable.
    assert (home / "nested").is_dir()
    assert "kuzu database directory" in result.output


def test_up_memgraph_calls_docker_compose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path, backend="memgraph")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    with (
        patch("subprocess.run") as mock_run,
        patch("contextd.storage.factory.build_graph_store") as mock_build,
    ):
        fake_store = mock_build.return_value
        fake_store.connect.return_value = None
        fake_store.apply_migrations.return_value = None
        fake_store.close.return_value = None
        result = CliRunner().invoke(contextd.cli.cli, ["up"])
    assert result.exit_code == 0, result.output
    # Verify docker compose up -d was called.
    assert any("docker" in str(c.args[0]) and "up" in c.args[0] for c in mock_run.call_args_list)


def test_down_kuzu_prints_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path, backend="kuzu")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
    result = CliRunner().invoke(contextd.cli.cli, ["down"])
    assert result.exit_code == 0
    assert "stopped" in result.output


def test_down_memgraph_calls_docker_compose_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_contextd_home(tmp_path, backend="memgraph")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    reload(contextd.cli)
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
    reload(contextd.cli)
    with patch("shutil.which", return_value=None):
        result = CliRunner().invoke(contextd.cli.cli, ["up"])
    assert result.exit_code != 0
    assert "docker not on PATH" in result.output
    # ClickException renders via "Error: ..." — not a bare Python traceback.
    assert "Traceback" not in result.output
