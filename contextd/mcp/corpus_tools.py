"""Per-corpus MCP tool loader and descriptor builder.

At MCP server startup, each corpus TOML in ``~/.contextd/corpora/`` is
scanned for ``[mcp.tools]`` entries.  Each entry maps a tool name to a
Cypher file path.  This module handles:

* Reading and validating Cypher files (read-only guard at load time AND
  at dispatch time — defence in depth).
* Extracting ``$name`` parameter placeholders via regex.
* Building ``mcp.types.Tool`` descriptors with ``<corpus>.<tool>``
  namespacing to avoid collisions.

**Known limitation — dollar-in-string false positives**

``extract_placeholders`` uses a simple regex ``\\$([a-zA-Z_][a-zA-Z0-9_]*)``
against the raw Cypher text.  A literal dollar sign inside a string
value — e.g.::

    WHERE n.description CONTAINS "SHA=pending$"

— will NOT trigger a false-positive because ``$`` followed by a
non-identifier character (``"``) is not matched.  However::

    WHERE n.label CONTAINS "pending$status"

WILL produce a spurious ``status`` placeholder.  In practice Cypher
string literals seldom embed ``$identifier``-shaped tokens, so this is
acceptable for v1.  A proper Cypher tokeniser would be required to
eliminate all false positives.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.types import Tool

from contextd.corpus_config import CorpusConfig, CorpusConfigError
from contextd.mcp.readonly_guard import ReadOnlyGuardError, assert_read_only

_PLACEHOLDER_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")


def extract_placeholders(cypher: str) -> frozenset[str]:
    """Extract parameter names from Cypher ``$name`` placeholders.

    Matches the pattern used by ``exec_read``/``exec_write`` — consistent
    with how Neo4j and Memgraph drivers bind parameters.

    See module docstring for the known ``$identifier``-in-string-literal
    false-positive limitation.
    """
    return frozenset(_PLACEHOLDER_RE.findall(cypher))


@dataclass(frozen=True)
class CorpusTool:
    """Runtime record for a single per-corpus Cypher tool."""

    cypher: str
    placeholders: frozenset[str]
    corpus_name: str


def _build_tool_descriptor(
    corpus_name: str,
    tool_name: str,
    placeholders: frozenset[str],
) -> Tool:
    namespaced = f"{corpus_name}.{tool_name}"
    return Tool(
        name=namespaced,
        description=f"[{corpus_name}] {tool_name} (Cypher tool)",
        inputSchema={
            "type": "object",
            "properties": {k: {"type": "string"} for k in sorted(placeholders)},
            "required": sorted(placeholders),
        },
    )


def build_tool_descriptors(
    home: Path,
) -> tuple[list[Tool], dict[str, CorpusTool]]:
    """Load per-corpus Cypher tools from ``home / "corpora" / "*.toml"``.

    Returns a 2-tuple:
    * ``descriptors`` — ``Tool`` objects suitable for returning from
      ``server.list_tools()``.
    * ``registry`` — ``{namespaced_name: CorpusTool}`` for dispatch.

    Error handling policy (non-fatal — do NOT abort the whole server):
    * Malformed corpus TOML → warning to stderr, skip corpus.
    * Missing Cypher file → warning to stderr, skip tool.
    * Write-containing Cypher → warning to stderr, skip tool.  This is
      logged loudly (corpus name + tool name included) because a
      write-containing Cypher in a corpus TOML is a security concern.
    """
    descriptors: list[Tool] = []
    registry: dict[str, CorpusTool] = {}

    corpora_dir = home / "corpora"
    if not corpora_dir.is_dir():
        return descriptors, registry

    for toml_path in sorted(corpora_dir.glob("*.toml")):
        try:
            corpus_cfg = CorpusConfig.load(toml_path)
        except CorpusConfigError as exc:
            print(
                f"[contextd] WARNING: skipping {toml_path.name} — {exc}",
                file=sys.stderr,
            )
            continue

        corpus_name = corpus_cfg.corpus.name

        for tool_name, cypher_path_str in corpus_cfg.mcp.tools.items():
            cypher_path = Path(cypher_path_str)
            if not cypher_path.is_absolute():
                cypher_path = toml_path.parent / cypher_path

            if not cypher_path.exists():
                print(
                    f"[contextd] WARNING: corpus '{corpus_name}', tool '{tool_name}' "
                    f"— Cypher file not found: {cypher_path}",
                    file=sys.stderr,
                )
                continue

            cypher_text = cypher_path.read_text()

            try:
                assert_read_only(cypher_text)
            except ReadOnlyGuardError as exc:
                print(
                    f"[contextd] WARNING: corpus '{corpus_name}', tool '{tool_name}' "
                    f"— SECURITY: Cypher contains write keyword, tool skipped: {exc}",
                    file=sys.stderr,
                )
                continue

            placeholders = extract_placeholders(cypher_text)
            namespaced = f"{corpus_name}.{tool_name}"

            descriptor = _build_tool_descriptor(corpus_name, tool_name, placeholders)
            descriptors.append(descriptor)
            registry[namespaced] = CorpusTool(
                cypher=cypher_text,
                placeholders=placeholders,
                corpus_name=corpus_name,
            )

    return descriptors, registry


def dispatch_corpus_tool(
    name: str,
    arguments: dict[str, Any],
    registry: dict[str, CorpusTool],
    exec_read: Callable[[str, dict[str, Any]], list[dict[str, Any]]],
) -> list[dict[str, Any]] | dict[str, str]:
    """Dispatch a namespaced corpus tool call.

    Defence in depth: ``assert_read_only`` is called again at dispatch
    time even though the startup check already filtered write-containing
    Cypher.  This guards against any in-memory mutation of the registry
    (unlikely but safe to check).

    Returns either the rows from ``exec_read`` or a ``{"error": ...}``
    dict on missing argument or guard failure.
    """
    if name not in registry:
        raise KeyError(f"Unknown corpus tool: {name!r}")

    tool = registry[name]

    # Defence-in-depth read-only check.
    assert_read_only(tool.cypher)

    # Verify required args are present.
    for placeholder in sorted(tool.placeholders):
        if placeholder not in arguments:
            return {"error": f"missing required argument: {placeholder}"}

    params = {k: arguments[k] for k in tool.placeholders}
    return exec_read(tool.cypher, params)
