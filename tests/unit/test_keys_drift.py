"""Drift detection between contextd/storage/_keys.py and the baseline migrations.

Two maps are currently duplicated:
- PRIMARY_KEY_BY_LABEL mirrors the `PRIMARY KEY` declarations in the Kuzu
  baseline DDL (and matches the uniqueness constraints Memgraph's baseline
  declares).
- IMMUTABLE_AFTER_CREATE_BY_LABEL mirrors the labels whose baseline
  migration creates a vector index (Kuzu forbids SET on vector-indexed
  columns after node creation).

If a future migration renames a PK or adds a vector index and _keys.py is
not updated in lock-step, the storage backends silently emit wrong Cypher
(edge MATCHes miss; re-upserts silently drop new embeddings). These tests
parse the Kuzu baseline DDL at import time and assert the maps stay in sync.
"""

from __future__ import annotations

import re

from contextd.migrations.kuzu._0001_baseline import _DDL as KUZU_DDL
from contextd.storage._keys import (
    IMMUTABLE_AFTER_CREATE_BY_LABEL,
    PRIMARY_KEY_BY_LABEL,
)

# Match: CREATE NODE TABLE <Label>(<first-col> <type> PRIMARY KEY, ...)
# `<first-col>` may be backtick-quoted (for reserved words). The capture
# group strips the backticks so the test asserts against the logical name.
_NODE_TABLE_RE = re.compile(
    r"CREATE NODE TABLE (\w+)\(\s*`?(\w+)`?\s+\w+(?:\[[^\]]+\])?\s+PRIMARY KEY",
    re.IGNORECASE,
)
_VECTOR_INDEX_RE = re.compile(
    r"CALL\s+CREATE_VECTOR_INDEX\s*\(\s*'([^']+)'\s*,\s*'[^']+'\s*,\s*'([^']+)'",
    re.IGNORECASE,
)


def _declared_primary_keys() -> dict[str, str]:
    """Extract {label: pk_property} from the Kuzu baseline CREATE NODE TABLE DDL."""
    return {label: pk for ddl in KUZU_DDL for label, pk in _NODE_TABLE_RE.findall(ddl)}


def _declared_vector_index_columns() -> dict[str, set[str]]:
    """Extract {label: {indexed_property, ...}} from the Kuzu baseline
    CALL CREATE_VECTOR_INDEX statements."""
    result: dict[str, set[str]] = {}
    for ddl in KUZU_DDL:
        for label, prop in _VECTOR_INDEX_RE.findall(ddl):
            result.setdefault(label, set()).add(prop)
    return result


def test_primary_key_map_matches_kuzu_ddl() -> None:
    """PRIMARY_KEY_BY_LABEL must contain exactly the PKs declared in the
    Kuzu baseline — same labels, same PK column per label.

    Meta is an exception: it is bootstrapped in KuzuBackend.connect() rather
    than declared in the migration DDL. The map carries it; the DDL does not.
    """
    declared = _declared_primary_keys()
    mapped = {k: v for k, v in PRIMARY_KEY_BY_LABEL.items() if k != "Meta"}
    assert mapped == declared, (
        f"PRIMARY_KEY_BY_LABEL drift vs Kuzu baseline.\n"
        f"  In _keys.py only: {sorted(set(mapped) - set(declared))}\n"
        f"  In DDL only: {sorted(set(declared) - set(mapped))}\n"
        f"  Mismatched PKs: "
        f"{sorted((lbl, mapped[lbl], declared[lbl]) for lbl in set(mapped) & set(declared) if mapped[lbl] != declared[lbl])}"
    )
    assert PRIMARY_KEY_BY_LABEL["Meta"] == "schema_version", (
        "Meta PK is hardcoded in KuzuBackend.connect() as schema_version; _keys.py must match."
    )


def test_immutable_map_covers_every_vector_indexed_column() -> None:
    """Every (label, property) declared in a CALL CREATE_VECTOR_INDEX in the
    Kuzu baseline must appear in IMMUTABLE_AFTER_CREATE_BY_LABEL. Kuzu rejects
    SET on vector-indexed columns after node creation; the map drives the
    dispatch in KuzuBackend.upsert_node."""
    declared = _declared_vector_index_columns()
    for label, props in declared.items():
        mapped = IMMUTABLE_AFTER_CREATE_BY_LABEL.get(label, frozenset())
        missing = props - mapped
        assert not missing, (
            f"Label {label!r} has vector-indexed columns {sorted(missing)} "
            f"that are not in IMMUTABLE_AFTER_CREATE_BY_LABEL. KuzuBackend."
            f"upsert_node would try to SET them on update and raise."
        )
