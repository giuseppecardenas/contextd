"""IPC server for daemon control.

JSON-lines protocol over a platform-specific transport (AF_UNIX on
Linux/macOS, localhost TCP on Windows). Runs in a background thread
spawned by run_daemon. Each client connection is handled in its own
short-lived daemon thread.

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

from contextd._compat import cleanup_ipc, create_ipc_server_socket

_log = logging.getLogger(__name__)


class IpcServer:
    """IPC server for daemon control (JSON-lines protocol).

    Transport is platform-specific: AF_UNIX on Linux/macOS, localhost TCP
    on Windows. Runs in a background thread started by run_daemon. The
    main loop continues unaffected; the server handles each connection in
    its own short-lived thread.

    Supported commands:
      {"cmd": "status"} → {"pid": N, "corpora": ["name", ...], "uptime_seconds": N}
      {"cmd": "stop"}   → {"ok": true}  (sets the shared stop_event)
      {"cmd": "ping"}   → {"pong": true}
    """

    def __init__(
        self,
        ipc_path: Path,
        stop_event: threading.Event,
        pid: int,
        corpora: list[str],
        start_time: float,
    ) -> None:
        self._ipc_path = ipc_path
        self._stop_event = stop_event
        self._pid = pid
        self._corpora = corpora
        self._start_time = start_time
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._handler_threads: list[threading.Thread] = []
        self._handler_lock = threading.Lock()

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
                _log.debug("IPC connection error", exc_info=True)

    def _serve(self) -> None:
        try:
            sock = create_ipc_server_socket(self._ipc_path)
        except Exception:
            _log.exception("IPC server failed to start")
            return
        self._server_socket = sock
        try:
            sock.listen(5)
            sock.settimeout(1.0)
            _log.debug("IPC server listening on %s", self._ipc_path)
            while not self._stop_event.is_set():
                try:
                    conn, _ = sock.accept()
                    t = threading.Thread(
                        target=self._handle_connection, args=(conn,), daemon=True
                    )
                    with self._handler_lock:
                        self._handler_threads = [
                            h for h in self._handler_threads if h.is_alive()
                        ]
                        self._handler_threads.append(t)
                    t.start()
                except TimeoutError:
                    continue
                except OSError:
                    break
        finally:
            sock.close()

    def start(self) -> None:
        """Start the IPC server in a background daemon thread (non-blocking)."""
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Close the server socket and join all threads."""
        if self._server_socket is not None:
            with contextlib.suppress(OSError):
                self._server_socket.close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._handler_lock:
            for t in self._handler_threads:
                t.join(timeout=1.0)
            self._handler_threads.clear()
        cleanup_ipc(self._ipc_path)
