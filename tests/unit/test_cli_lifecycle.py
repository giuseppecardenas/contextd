"""Tests for up / down / status CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import contextd.cli


def _setup_contextd_home(tmp_path: Path, backend: str = "neo4j") -> Path:
    home = tmp_path / ".contextd"
    home.mkdir()
    # ``home.as_posix()`` keeps backslash-free TOML on Windows; pathlib still
    # accepts forward-slash paths there, but ``\U`` in a double-quoted TOML
    # string is parsed as a Unicode escape and fails the test fixture.
    config = f"""
[storage]
backend = "{backend}"

[storage.{backend}]
docker_compose_file = "{home.as_posix()}/docker-compose.yml"
"""
    (home / "config.toml").write_text(config)
    (home / "corpora").mkdir()
    return home


def test_status_lists_corpora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path)
    (home / "corpora" / "demo.toml").write_text("""
[corpus]
name = "demo"
root = "/tmp/demo"
""")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    result = CliRunner().invoke(contextd.cli.cli, ["status"])
    assert result.exit_code == 0
    assert "demo" in result.output
    assert "neo4j" in result.output


def test_status_no_corpora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path)
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    result = CliRunner().invoke(contextd.cli.cli, ["status"])
    assert result.exit_code == 0
    assert "neo4j" in result.output
    assert "0 registered" in result.output


def test_down_calls_docker_compose_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path)
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    with patch("subprocess.run") as mock_run:
        result = CliRunner().invoke(contextd.cli.cli, ["down"])
    assert result.exit_code == 0
    assert any("down" in c.args[0] for c in mock_run.call_args_list)


def test_up_without_docker_raises_clickexception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When docker is absent, `up` must surface a clean ClickException
    instead of a FileNotFoundError traceback."""
    home = _setup_contextd_home(tmp_path)
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
    fake_proc = MagicMock()
    fake_proc.pid = 12345
    with (
        patch("contextd.storage.factory.build_graph_store") as mock_build,
        patch("contextd.cli.infra.subprocess.Popen", return_value=fake_proc),
    ):
        fake_store = mock_build.return_value
        fake_store.connect.return_value = None
        fake_store.apply_migrations.return_value = None
        fake_store.close.return_value = None
        result = CliRunner().invoke(contextd.cli.cli, ["up"])

    assert result.exit_code == 0, result.output
    # At least one docker compose call with --profile neo4j and `up`.
    compose_calls = [c for c in calls if "compose" in c]
    assert any("--profile" in c and "neo4j" in c and "up" in c for c in compose_calls)


# ---------------------------------------------------------------------------
# Task 14.4 — daemon lifecycle helpers
# ---------------------------------------------------------------------------


def test_up_writes_pid_file(tmp_path: Path) -> None:
    from contextd.cli.infra import _write_pid

    pid_file = tmp_path / "indexer.pid"
    with patch("contextd.cli.infra._pid_path", return_value=pid_file):
        _write_pid(pid_file, 77777)
    assert pid_file.read_text().strip() == "77777"


def test_up_skips_daemon_launch_when_already_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _setup_contextd_home(tmp_path)
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    # Pre-existing PID file that points at a live process.
    (home / "state").mkdir(exist_ok=True)
    (home / "state" / "indexer.pid").write_text("42424")
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/docker")
    with (
        patch("subprocess.run") as mock_run,
        patch("contextd.storage.factory.build_graph_store") as mock_build,
        patch("contextd.cli.infra.subprocess.Popen") as mock_popen,
        patch("contextd.cli.infra._daemon_is_running", return_value=True),
    ):
        mock_run.return_value.returncode = 0
        fake_store = mock_build.return_value
        fake_store.connect.return_value = None
        fake_store.apply_migrations.return_value = None
        fake_store.close.return_value = None
        result = CliRunner().invoke(contextd.cli.cli, ["up"])
    assert result.exit_code == 0, result.output
    assert mock_popen.call_count == 0
    assert "already running" in result.output


def test_up_clears_stale_pid_and_launches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _setup_contextd_home(tmp_path)
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    # Pre-existing PID file that points at a dead process.
    (home / "state").mkdir(exist_ok=True)
    pid_file = home / "state" / "indexer.pid"
    pid_file.write_text("99999")
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/docker")
    fake_proc = MagicMock()
    fake_proc.pid = 55555
    with (
        patch("subprocess.run") as mock_run,
        patch("contextd.storage.factory.build_graph_store") as mock_build,
        patch("contextd.cli.infra.subprocess.Popen", return_value=fake_proc) as mock_popen,
        patch("contextd.cli.infra._daemon_is_running", return_value=False),
    ):
        mock_run.return_value.returncode = 0
        fake_store = mock_build.return_value
        fake_store.connect.return_value = None
        fake_store.apply_migrations.return_value = None
        fake_store.close.return_value = None
        result = CliRunner().invoke(contextd.cli.cli, ["up"])
    assert result.exit_code == 0, result.output
    assert mock_popen.call_count == 1
    assert pid_file.read_text().strip() == "55555"


def test_down_prefers_graceful_ipc_stop(tmp_path: Path) -> None:
    """When the IPC stop succeeds and the process exits, no tree-kill happens."""
    from contextd.cli.infra import _stop_daemon

    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text("99999")

    with (
        patch("contextd.cli.infra._pid_path", return_value=pid_file),
        patch("contextd.cli.infra.contextd_home", return_value=tmp_path),
        patch("contextd.cli.infra._request_daemon_stop", return_value=True),
        patch("contextd.cli.infra._wait_for_exit", return_value=True),
        patch("contextd.cli.infra._kill_process_tree") as mock_tree_kill,
    ):
        _stop_daemon()

    mock_tree_kill.assert_not_called()
    assert not pid_file.exists()


def test_down_falls_back_to_tree_kill_on_ipc_failure(tmp_path: Path) -> None:
    """When IPC is unreachable, _stop_daemon must walk the process tree."""
    from contextd.cli.infra import _stop_daemon

    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text("99999")

    with (
        patch("contextd.cli.infra._pid_path", return_value=pid_file),
        patch("contextd.cli.infra.contextd_home", return_value=tmp_path),
        patch("contextd.cli.infra._request_daemon_stop", return_value=False),
        patch("contextd.cli.infra._kill_process_tree") as mock_tree_kill,
    ):
        _stop_daemon()

    mock_tree_kill.assert_called_once_with(99999, timeout=5.0)
    assert not pid_file.exists()


def test_down_falls_back_when_graceful_stop_times_out(tmp_path: Path) -> None:
    """Graceful stop ack'd but process never exits → fallback engages."""
    from contextd.cli.infra import _stop_daemon

    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text("99999")

    with (
        patch("contextd.cli.infra._pid_path", return_value=pid_file),
        patch("contextd.cli.infra.contextd_home", return_value=tmp_path),
        patch("contextd.cli.infra._request_daemon_stop", return_value=True),
        patch("contextd.cli.infra._wait_for_exit", return_value=False),
        patch("contextd.cli.infra._kill_process_tree") as mock_tree_kill,
    ):
        _stop_daemon()

    mock_tree_kill.assert_called_once_with(99999, timeout=5.0)


def test_down_removes_pid_and_socket_files(tmp_path: Path) -> None:
    from contextd._compat import ipc_file_name
    from contextd.cli.infra import _stop_daemon

    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text("99999")
    ipc_file = tmp_path / ipc_file_name()
    ipc_file.write_text("62329\n")

    with (
        patch("contextd.cli.infra._pid_path", return_value=pid_file),
        patch("contextd.cli.infra.contextd_home", return_value=tmp_path),
        patch("contextd.cli.infra._request_daemon_stop", return_value=True),
        patch("contextd.cli.infra._wait_for_exit", return_value=True),
        patch("contextd.cli.infra._kill_process_tree"),
    ):
        _stop_daemon()

    assert not pid_file.exists()
    assert not ipc_file.exists()


def test_down_is_safe_with_no_pid_file(tmp_path: Path) -> None:
    from contextd.cli.infra import _stop_daemon

    with patch("contextd.cli.infra._pid_path", return_value=tmp_path / "missing.pid"):
        _stop_daemon()  # must not raise


def test_daemon_pid_returns_none_on_missing(tmp_path: Path) -> None:
    from contextd.cli.infra import _daemon_pid

    with patch("contextd.cli.infra._pid_path", return_value=tmp_path / "none.pid"):
        assert _daemon_pid() is None


def test_daemon_is_running_uses_psutil() -> None:
    """The liveness check must work cross-platform via psutil.pid_exists."""
    import os

    from contextd.cli.infra import _daemon_is_running

    assert _daemon_is_running(os.getpid()) is True
    # PID 0 is reserved on POSIX (the scheduler) and 'System Idle' on Windows;
    # use a guaranteed-dead high PID instead. 2**31 - 2 is above any real PID.
    assert _daemon_is_running(2**31 - 2) is False


def test_request_daemon_stop_returns_false_when_socket_absent(tmp_path: Path) -> None:
    from contextd.cli.infra import _request_daemon_stop

    assert _request_daemon_stop(tmp_path / "missing.sock") is False


def test_request_daemon_stop_round_trips_through_ipc_server(ipc_path: Path) -> None:
    """End-to-end: spin up an IpcServer and confirm _request_daemon_stop sets the event."""
    import threading
    import time as _time

    from contextd.cli.infra import _request_daemon_stop
    from contextd.daemon_ipc import IpcServer

    stop_event = threading.Event()
    server = IpcServer(
        ipc_path=ipc_path,
        stop_event=stop_event,
        pid=12345,
        corpora=["alpha"],
        start_time=_time.time(),
    )
    server.start()
    try:
        # Wait briefly for the server to bind.
        deadline = _time.monotonic() + 2.0
        while _time.monotonic() < deadline and not ipc_path.exists():
            _time.sleep(0.02)
        assert _request_daemon_stop(ipc_path) is True
        stop_event.wait(timeout=1.0)
        assert stop_event.is_set()
    finally:
        server.stop()
