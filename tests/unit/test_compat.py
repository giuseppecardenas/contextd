"""Tests for the platform abstraction module."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

from contextd._compat import (
    IS_WINDOWS,
    cleanup_ipc,
    connect_ipc,
    create_ipc_server_socket,
    daemon_popen_kwargs,
    ipc_file_name,
    process_is_alive,
)


def test_ipc_file_name_matches_platform() -> None:
    if IS_WINDOWS:
        assert ipc_file_name() == "ipc.port"
    else:
        assert ipc_file_name() == "ipc.sock"


def test_create_and_connect_roundtrip(ipc_path: Path) -> None:
    server_sock = create_ipc_server_socket(ipc_path)
    try:
        server_sock.listen(1)
        server_sock.settimeout(2.0)

        ready = threading.Event()
        result: dict[str, object] = {}

        def _accept() -> None:
            ready.set()
            conn, _ = server_sock.accept()
            with conn:
                data = conn.recv(4096)
                result["received"] = data.decode()
                conn.sendall(b"pong\n")

        t = threading.Thread(target=_accept, daemon=True)
        t.start()
        ready.wait(timeout=2.0)

        client = connect_ipc(ipc_path)
        with client:
            client.settimeout(2.0)
            client.sendall(b"ping\n")
            reply = client.recv(4096).decode()

        t.join(timeout=2.0)
        assert result["received"] == "ping\n"
        assert reply == "pong\n"
    finally:
        server_sock.close()
        cleanup_ipc(ipc_path)


def test_cleanup_ipc_removes_file(ipc_path: Path) -> None:
    sock = create_ipc_server_socket(ipc_path)
    sock.close()
    assert ipc_path.exists()
    cleanup_ipc(ipc_path)
    assert not ipc_path.exists()


def test_cleanup_ipc_no_error_on_missing(tmp_path: Path) -> None:
    cleanup_ipc(tmp_path / "nonexistent")


def test_process_is_alive_current_process() -> None:
    assert process_is_alive(os.getpid()) is True


def test_process_is_alive_nonexistent_pid() -> None:
    assert process_is_alive(2**31 - 1) is False


def test_daemon_popen_kwargs_has_correct_key() -> None:
    kwargs = daemon_popen_kwargs()
    if sys.platform == "win32":
        assert "creationflags" in kwargs
        assert "start_new_session" not in kwargs
    else:
        assert kwargs == {"start_new_session": True}


def test_ipc_server_socket_creates_endpoint_file(ipc_path: Path) -> None:
    assert not ipc_path.exists()
    sock = create_ipc_server_socket(ipc_path)
    try:
        assert ipc_path.exists()
        if IS_WINDOWS:
            port = int(ipc_path.read_text().strip())
            assert port > 0
    finally:
        sock.close()
        cleanup_ipc(ipc_path)


def test_ipc_roundtrip_with_json_protocol(ipc_path: Path) -> None:
    """Full JSON-lines round-trip matching daemon IPC protocol."""
    server_sock = create_ipc_server_socket(ipc_path)
    try:
        server_sock.listen(1)
        server_sock.settimeout(2.0)

        def _echo_server() -> None:
            conn, _ = server_sock.accept()
            with conn:
                data = conn.recv(4096)
                req = json.loads(data.decode())
                resp = json.dumps({"echo": req["cmd"]}) + "\n"
                conn.sendall(resp.encode())

        t = threading.Thread(target=_echo_server, daemon=True)
        t.start()

        client = connect_ipc(ipc_path)
        with client:
            client.settimeout(2.0)
            client.sendall((json.dumps({"cmd": "ping"}) + "\n").encode())
            raw = client.recv(4096).decode().strip()
        t.join(timeout=2.0)
        assert json.loads(raw) == {"echo": "ping"}
    finally:
        server_sock.close()
        cleanup_ipc(ipc_path)
