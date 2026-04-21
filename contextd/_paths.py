"""Late-bound home-directory accessor.

``contextd_home()`` reads the ``CONTEXTD_HOME`` env var on every call so
tests (and future runtime reconfiguration) can change the value without
reloading modules. Previously a module-level ``CONTEXTD_HOME`` in
``contextd.cli`` captured the value at import time, which forced every
CLI test to call ``importlib.reload(contextd.cli)`` after
``monkeypatch.setenv``. This module replaces that pattern.

Kept free of click/rich so ``contextd.mcp_server`` can import it without
pulling the CLI dependency tree into the MCP process.
"""

from __future__ import annotations

import os
from pathlib import Path


def contextd_home() -> Path:
    """Return the current effective ``~/.contextd`` (or ``$CONTEXTD_HOME``).

    Resolved on every call; no module-level capture. Callers should
    invoke inside function bodies, not at import time.
    """
    return Path(os.environ.get("CONTEXTD_HOME", str(Path.home() / ".contextd")))
