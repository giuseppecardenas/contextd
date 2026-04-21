"""Rejects Cypher containing write keywords.

Spec §7.4: all MCP tools are read-only. The guard is a thin keyword
check — sufficient because Memgraph and Neo4j both tokenize identically,
and the read-only surface (MATCH, WITH, UNWIND, RETURN, and read-only
CALL procedures like db.labels() or text_search.search_all) is small.

The leading negative-lookbehind ``(?<![.\\w])`` ensures we don't
false-positive on dotted property access — ``RETURN n.set AS prop``
used to trip the SET keyword because ``\\b`` fires between ``.`` and
``s``; the lookbehind requires the keyword to NOT be preceded by a
word char or a dot, which matches keyword positions exactly.
"""

from __future__ import annotations

import re


class ReadOnlyGuardError(ValueError):
    """Raised when Cypher contains a forbidden write keyword."""


_FORBIDDEN = re.compile(
    r"(?<![.\w])(CREATE|MERGE|DELETE|SET|REMOVE|DROP|DETACH|FOREACH)\b",
    re.IGNORECASE,
)


def assert_read_only(cypher: str) -> None:
    match = _FORBIDDEN.search(cypher)
    if match:
        raise ReadOnlyGuardError(
            f"Cypher contains forbidden write keyword: {match.group(0).upper()}"
        )
