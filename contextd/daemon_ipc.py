"""IPC server for daemon control.

Uses a Unix domain socket on POSIX. On Windows (and any other build where
``socket.AF_UNIX`` is unavailable) it falls back to a TCP listener bound to
``127.0.0.1`` on an ephemeral port; the chosen port is written to the same
``socket_path`` as plain text so the client can discover it. Either way the
endpoint is loopback-only — no traffic ever leaves the host.

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

# socket.AF_UNIX is platform-conditional (absent on the Python builds where
# Unix domain sockets are unavailable, e.g. Windows builds without UDS
# support). Resolve it via getattr so mypy doesn't probe for it on Windows.
_AF_UNIX: int | None = getattr(socket, "AF_UNIX", None)


def _open_listener(socket_path: Path) -> socket.socket:
    """Create and bind the IPC listening socket.

    On POSIX, binds an ``AF_UNIX`` socket at ``socket_path``.
    On systems without ``AF_UNIX``, binds an ``AF_INET`` socket to
    ``127.0.0.1:0`` (an ephemeral port) and writes ``"<port>\\n"`` to
    ``socket_path`` so the client can discover the address.
    """
    if _AF_UNIX is not None:
        sock = socket.socket(_AF_UNIX, socket.SOCK_STREAM)
        with contextlib.suppress(FileNotFoundError):
            socket_path.unlink()
        sock.bind(str(socket_path))
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        socket_path.write_text(f"{port}\n")
    sock.listen(5)
    sock.settimeout(1.0)
    return sock


def connect(socket_path: Path, timeout: float = 1.0) -> socket.socket:
    """Open a client connection to the IPC endpoint at ``socket_path``.

    Mirrors the transport choice made by ``_open_listener``. The caller owns
    the returned socket and must close it.
    """
    if _AF_UNIX is not None:
        s = socket.socket(_AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(str(socket_path))
        return s
    port = int(socket_path.read_text().strip())
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(("127.0.0.1", port))
    return s


class IpcServer:
    """IPC server for daemon control (JSON-lines protocol).

    Runs in a background thread started by run_daemon. The main loop
    continues unaffected; the server handles each connection in its own
    short-lived thread. See module docstring for the supported commands and
    the transport-selection rules.
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
                _log.debug("IPC connection error", exc_info=True)

    def _serve(self) -> None:
        try:
            sock = _open_listener(self._socket_path)
        except Exception:
            _log.exception("IPC server failed to start")
            return
        self._server_socket = sock
        try:
            _log.debug("IPC server listening on %s", self._socket_path)
            while not self._stop_event.is_set():
                try:
                    conn, _ = sock.accept()
                    threading.Thread(
                        target=self._handle_connection, args=(conn,), daemon=True
                    ).start()
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
        """Close the server socket and join the background thread."""
        if self._server_socket is not None:
            with contextlib.suppress(OSError):
                self._server_socket.close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with contextlib.suppress(FileNotFoundError):
            self._socket_path.unlink()
