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


def canonical_path(path: Path | str) -> str:
    """Return the canonical string form of a path used as a graph node identity.

    File and Section nodes are keyed by a path string (``File.path`` and the
    ``<path>#<anchor>`` prefix of ``Section.id``). That string MUST be derived
    the same way at every site that creates or matches a node, or re-processing
    a file MERGEs against a different key and creates a duplicate instead of
    updating the existing record.

    The trap on Windows: ``str(WindowsPath(...))`` yields backslashes, but a
    path that ever passed through ``as_posix()`` (older code, or a WSL-era
    index of the same tree) is stored with forward slashes — so the two
    conventions silently diverge and the "previous record" is never updated.
    Pinning one convention (forward slashes, via ``as_posix``) makes the
    identity independent of OS separator and of which code path produced the
    path (bootstrap glob, watchdog event, or debouncer ``resolve()``).

    ``expanduser`` is applied so a ``~``-rooted path canonicalises the same way
    the enumerate phase stores it. No ``resolve()``: inputs are already
    absolute (corpus roots are resolved at ``add-corpus`` time; daemon paths by
    the debouncer), and ``resolve()`` would add a filesystem ``stat`` plus
    surprising symlink rewriting on the deletion/GC paths where the file may no
    longer exist.
    """
    return Path(path).expanduser().as_posix()
