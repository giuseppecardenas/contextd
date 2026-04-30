from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from contextd._compat import connect_ipc


def _send_recv(ipc_path: Path, payload: str, timeout: float = 2.0) -> str:
    """Send a JSON-lines message and return the raw response string."""
    with connect_ipc(ipc_path) as s:
        s.settimeout(timeout)
        s.sendall(payload.encode())
        return s.recv(4096).decode()


def _wait_for_ipc(ipc_path: Path, timeout: float = 5.0) -> None:
    """Block until the IPC endpoint is connectable or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ipc_path.exists():
            try:
                with connect_ipc(ipc_path) as s:
                    s.settimeout(0.1)
                return
            except OSError:
                pass
        time.sleep(0.05)
    raise TimeoutError(f"IPC endpoint {ipc_path} did not become connectable within {timeout}s")


def test_ipc_server_responds_to_ping(tmp_path: Path) -> None:
    from contextd._compat import ipc_file_name
    from contextd.daemon_ipc import IpcServer

    ipc_path = tmp_path / ipc_file_name()
    stop_event = threading.Event()
    server = IpcServer(
        ipc_path=ipc_path,
        stop_event=stop_event,
        pid=12345,
        corpora=["corp-a"],
        start_time=time.time(),
    )
    server.start()
    try:
        _wait_for_ipc(ipc_path)
        raw = _send_recv(ipc_path, json.dumps({"cmd": "ping"}) + "\n")
        response = json.loads(raw.strip())
        assert response == {"pong": True}
    finally:
        server.stop()


def test_ipc_server_responds_to_status(tmp_path: Path) -> None:
    from contextd._compat import ipc_file_name
    from contextd.daemon_ipc import IpcServer

    ipc_path = tmp_path / ipc_file_name()
    stop_event = threading.Event()
    server = IpcServer(
        ipc_path=ipc_path,
        stop_event=stop_event,
        pid=99,
        corpora=["alpha", "beta"],
        start_time=time.time(),
    )
    server.start()
    try:
        _wait_for_ipc(ipc_path)
        raw = _send_recv(ipc_path, json.dumps({"cmd": "status"}) + "\n")
        response = json.loads(raw.strip())
        assert "corpora" in response
        assert response["corpora"] == ["alpha", "beta"]
        assert response["pid"] == 99
        assert "uptime_seconds" in response
    finally:
        server.stop()


def test_ipc_server_stop_sets_event(tmp_path: Path) -> None:
    from contextd._compat import ipc_file_name
    from contextd.daemon_ipc import IpcServer

    ipc_path = tmp_path / ipc_file_name()
    stop_event = threading.Event()
    server = IpcServer(
        ipc_path=ipc_path,
        stop_event=stop_event,
        pid=1,
        corpora=[],
        start_time=time.time(),
    )
    server.start()
    try:
        _wait_for_ipc(ipc_path)
        raw = _send_recv(ipc_path, json.dumps({"cmd": "stop"}) + "\n")
        response = json.loads(raw.strip())
        assert response == {"ok": True}
        # give the event a moment to be set
        stop_event.wait(timeout=1.0)
        assert stop_event.is_set()
    finally:
        server.stop()


def test_ipc_server_ignores_unknown_command(tmp_path: Path) -> None:
    from contextd._compat import ipc_file_name
    from contextd.daemon_ipc import IpcServer

    ipc_path = tmp_path / ipc_file_name()
    stop_event = threading.Event()
    server = IpcServer(
        ipc_path=ipc_path,
        stop_event=stop_event,
        pid=1,
        corpora=[],
        start_time=time.time(),
    )
    server.start()
    try:
        _wait_for_ipc(ipc_path)
        raw = _send_recv(ipc_path, json.dumps({"cmd": "frobnicate"}) + "\n")
        response = json.loads(raw.strip())
        assert "error" in response
        # Server should still be running — send a ping to confirm
        raw2 = _send_recv(ipc_path, json.dumps({"cmd": "ping"}) + "\n")
        response2 = json.loads(raw2.strip())
        assert response2 == {"pong": True}
    finally:
        server.stop()


def test_ipc_server_handles_corrupt_json(tmp_path: Path) -> None:
    from contextd._compat import ipc_file_name
    from contextd.daemon_ipc import IpcServer

    ipc_path = tmp_path / ipc_file_name()
    stop_event = threading.Event()
    server = IpcServer(
        ipc_path=ipc_path,
        stop_event=stop_event,
        pid=1,
        corpora=[],
        start_time=time.time(),
    )
    server.start()
    try:
        _wait_for_ipc(ipc_path)
        # Send corrupt JSON — server should reply with error and close connection cleanly
        raw = _send_recv(ipc_path, "not-json\n")
        response = json.loads(raw.strip())
        assert "error" in response
        # Server must still be running
        raw2 = _send_recv(ipc_path, json.dumps({"cmd": "ping"}) + "\n")
        response2 = json.loads(raw2.strip())
        assert response2 == {"pong": True}
    finally:
        server.stop()
