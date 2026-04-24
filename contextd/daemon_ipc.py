"""Unix domain socket IPC server for daemon control.

JSON-lines protocol. Runs in a background thread spawned by run_daemon.
Each client connection is handled in its own short-lived daemon thread.

Supported commands:
  {"cmd": "ping"}   → {"pong": true}
  {"cmd": "status"} → {"pid": N, "corpora": ["name", ...], "uptime_seconds": N}
  {"cmd": "stop"}   → {"ok": true}  (sets the shared stop_event)
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time
from pathlib import Path

_log = logging.getLogger(__name__)


class IpcServer:
    """Unix domain socket server for daemon control (JSON-lines protocol).

    Runs in a background thread started by run_daemon. The main loop
    continues unaffected; the server handles each connection in its own
    short-lived thread.

    Supported commands:
      {"cmd": "status"} → {"pid": N, "corpora": ["name", ...], "uptime_seconds": N}
      {"cmd": "stop"}   → {"ok": true}  (sets the shared stop_event)
      {"cmd": "ping"}   → {"pong": true}
    """

    def __init__(
        self,
        socket_path: Path,
        stop_event: threading.Event,
        pid: int,
        corpora: list[str],
        start_time: float,
    ) -> None:
        self._socket_path = socket_path
        self._stop_event = stop_event
        self._pid = pid
        self._corpora = corpora
        self._start_time = start_time
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                data = conn.recv(4096)
                if not data:
                    return
                request = json.loads(data.decode())
                cmd = request.get("cmd")
                if cmd == "ping":
                    response: dict[str, object] = {"pong": True}
                elif cmd == "status":
                    response = {
                        "pid": self._pid,
                        "corpora": self._corpora,
                        "uptime_seconds": int(time.time() - self._start_time),
                    }
                elif cmd == "stop":
                    self._stop_event.set()
                    response = {"ok": True}
                else:
                    response = {"error": f"unknown command: {cmd!r}"}
                conn.sendall((json.dumps(response) + "\n").encode())
            except json.JSONDecodeError:
                conn.sendall((json.dumps({"error": "invalid JSON"}) + "\n").encode())
            except Exception:
                pass

    def _serve(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket = sock
        with contextlib.suppress(FileNotFoundError):
            self._socket_path.unlink()
        sock.bind(str(self._socket_path))
        sock.listen(5)
        sock.settimeout(1.0)
        _log.debug("IPC server listening on %s", self._socket_path)
        while not self._stop_event.is_set():
            try:
                conn, _ = sock.accept()
                threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
            except TimeoutError:
                continue
            except OSError:
                break

    def start(self) -> None:
        """Start the IPC server in a background daemon thread (non-blocking)."""
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Close the server socket and join the background thread."""
        if self._server_socket is not None:
            with contextlib.suppress(OSError):
                self._server_socket.close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with contextlib.suppress(FileNotFoundError):
            self._socket_path.unlink()
