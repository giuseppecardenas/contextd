"""Platform abstraction for daemon lifecycle and IPC.

All OS-specific code lives in this module. Production code outside this
file must not branch on ``sys.platform``, import ``signal`` for handler
registration, or use ``AF_UNIX`` / ``os.kill`` directly.

Unix (Linux / macOS):
  IPC via AF_UNIX domain socket at ``~/.contextd/ipc.sock``.
  Process control via POSIX signals (SIGTERM / SIGKILL).

Windows:
  IPC via localhost TCP (127.0.0.1, ephemeral port written to
  ``~/.contextd/ipc.port``). Process control via kernel32
  TerminateProcess.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

IS_WINDOWS: bool = sys.platform == "win32"

_SignalHandler = "Callable[[int, FrameType | None], object]"


# ---------------------------------------------------------------------------
# IPC helpers
# ---------------------------------------------------------------------------


def ipc_file_name() -> str:
    """Return the IPC endpoint filename for the current platform."""
    return "ipc.port" if IS_WINDOWS else "ipc.sock"


def create_ipc_server_socket(ipc_path: Path) -> socket.socket:
    """Create, bind, and return a server socket ready for ``listen()``.

    On Unix, binds an AF_UNIX socket at *ipc_path* (stale file removed
    first). On Windows, binds a TCP socket to ``127.0.0.1:0`` and writes
    the assigned port number to *ipc_path*.
    """
    if IS_WINDOWS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        ipc_path.write_text(str(port))
    else:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with contextlib.suppress(FileNotFoundError):
            ipc_path.unlink()
        sock.bind(str(ipc_path))
    return sock


def connect_ipc(ipc_path: Path) -> socket.socket:
    """Return a connected client socket to the daemon's IPC endpoint.

    On Unix, connects via AF_UNIX to *ipc_path*. On Windows, reads the
    port number from *ipc_path* and connects via TCP to 127.0.0.1.
    """
    if IS_WINDOWS:
        port = int(ipc_path.read_text().strip())
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
    else:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(ipc_path))
    return sock


def cleanup_ipc(ipc_path: Path) -> None:
    """Remove the IPC endpoint file (socket file or port file)."""
    with contextlib.suppress(FileNotFoundError):
        ipc_path.unlink()


# ---------------------------------------------------------------------------
# Process lifecycle
# ---------------------------------------------------------------------------


def process_is_alive(pid: int) -> bool:
    """Return True if *pid* refers to a running process."""
    if IS_WINDOWS:
        return _win_process_is_alive(pid)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def graceful_terminate(pid: int) -> None:
    """Ask a process to exit. On Unix, sends SIGTERM. On Windows, calls
    TerminateProcess (immediate — Windows has no graceful-signal equivalent).
    """
    if IS_WINDOWS:
        _win_terminate(pid)
    else:
        os.kill(pid, signal.SIGTERM)


def force_kill(pid: int) -> None:
    """Force-kill a process. On Unix, sends SIGKILL. On Windows, calls
    TerminateProcess (same as graceful_terminate — already immediate).
    """
    if IS_WINDOWS:
        _win_terminate(pid)
    else:
        os.kill(pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Signal / stop-handler registration
# ---------------------------------------------------------------------------


def install_stop_handlers(callback: Callable[[int, FrameType | None], object]) -> None:
    """Register *callback* as the handler for platform stop signals.

    On Unix, registers SIGTERM and SIGINT. On Windows, registers SIGINT
    and SIGBREAK (SIGTERM is not catchable on Windows — the runtime
    calls TerminateProcess immediately).
    """
    signal.signal(signal.SIGINT, callback)
    if IS_WINDOWS:
        signal.signal(signal.SIGBREAK, callback)  # type: ignore[attr-defined]
    else:
        signal.signal(signal.SIGTERM, callback)


# ---------------------------------------------------------------------------
# Daemon subprocess launch
# ---------------------------------------------------------------------------


def daemon_popen_kwargs() -> dict[str, object]:
    """Return platform-specific kwargs for ``subprocess.Popen`` to detach
    the daemon from the parent process group.
    """
    if IS_WINDOWS:
        return {
            "creationflags": (
                subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            ),
        }
    return {"start_new_session": True}


# ---------------------------------------------------------------------------
# Windows-only helpers (guarded by IS_WINDOWS at call sites)
# ---------------------------------------------------------------------------


_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_PROCESS_TERMINATE = 0x0001


def _win_process_is_alive(pid: int) -> bool:
    """Check process existence via kernel32 OpenProcess."""
    import ctypes
    import ctypes.wintypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.wintypes.DWORD()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return exit_code.value == _STILL_ACTIVE
        return False
    finally:
        kernel32.CloseHandle(handle)


def _win_terminate(pid: int) -> None:
    """Terminate a process via kernel32 TerminateProcess."""
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(_PROCESS_TERMINATE, False, pid)
    if handle:
        try:
            kernel32.TerminateProcess(handle, 1)
        finally:
            kernel32.CloseHandle(handle)
