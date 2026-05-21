"""Unit-test fixtures."""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from contextd._compat import ipc_file_name


@pytest.fixture
def ipc_path(tmp_path: Path) -> Iterator[Path]:
    """Return an IPC endpoint path short enough for AF_UNIX on macOS.

    macOS limits AF_UNIX paths to 104 bytes.  Pytest's ``tmp_path`` on
    macOS CI produces paths like ``/private/var/folders/.../test_name0/``
    which exceed the limit once ``ipc.sock`` is appended.  On macOS we
    fall back to a directory under ``/tmp`` (always short); on other
    platforms ``tmp_path`` is fine.
    """
    if sys.platform == "darwin":
        with tempfile.TemporaryDirectory(dir="/tmp") as short:
            yield Path(short) / ipc_file_name()
    else:
        yield tmp_path / ipc_file_name()
