"""Rejects Cypher containing write keywords.

Spec §7.4: all MCP tools are read-only. The guard is a thin keyword
check — sufficient because Memgraph and Kuzu both tokenize identically,
and the allow-list (MATCH, WITH, UNWIND, RETURN) is small.
"""

from __future__ import annotations

import re


class ReadOnlyGuardError(ValueError):
    """Raised when Cypher contains a forbidden write keyword."""


_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|DETACH)\b",
    re.IGNORECASE,
)


def assert_read_only(cypher: str) -> None:
    match = _FORBIDDEN.search(cypher)
    if match:
        raise ReadOnlyGuardError(
            f"Cypher contains forbidden write keyword: {match.group(0).upper()}"
        )
