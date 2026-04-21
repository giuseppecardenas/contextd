"""Forward-only migration runner.

Each backend ships its own migrations (contextd/migrations/memgraph/*.py
and contextd/migrations/neo4j/*.py) because schema DDL differs. The
Meta singleton node records applied IDs; the runner skips any migration
whose ID is already present.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class Migration:
    """One forward-only schema change.

    ``up(store, migration_id)`` must be **idempotent**: if it partially
    succeeds and then fails (e.g., table 2 of 3 raises), the runner halts
    without recording the migration as applied, and the next ``apply()``
    call will re-run ``up`` from the top. Every statement inside ``up``
    must therefore be safe to re-execute — use ``CREATE CONSTRAINT IF
    NOT EXISTS``, ``CREATE INDEX IF NOT EXISTS``, and treat schema
    mutations with care (prefer CREATE-then-REPLACE patterns or split
    into multiple migrations).
    """

    id: int
    name: str
    up: Callable[[Any, int], None]


class MigrationRunner:
    def __init__(self, store: Any, migrations: Sequence[Migration]) -> None:
        self._store = store
        self._migrations = sorted(migrations, key=lambda m: m.id)

    def apply(self) -> None:
        applied = self._current_applied()
        for m in self._migrations:
            if m.id in applied:
                continue
            m.up(self._store, m.id)
            self._record_applied(m.id)

    def _current_applied(self) -> set[int]:
        # The singleton Meta node uses schema_version=0 as a fixed PK.
        rows = self._store.exec_read(
            "MATCH (m:Meta {schema_version: 0}) RETURN m.applied AS applied LIMIT 1",
            None,
        )
        if not rows:
            return set()
        return set(rows[0].get("applied") or [])

    def _record_applied(self, migration_id: int) -> None:
        # migration_id is a developer-controlled int (no injection surface);
        # inlining it avoids any backend-specific type-inference quirks on
        # list-concat against the INT64[] applied column.
        self._store.exec_write(
            f"MERGE (m:Meta {{schema_version: 0}}) "
            f"SET m.applied = coalesce(m.applied, []) + [{int(migration_id)}]",
            None,
        )
