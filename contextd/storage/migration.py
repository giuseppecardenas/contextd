"""Forward-only migration runner.

Each backend ships its own migrations (contextd/migrations/memgraph/*.py
and contextd/migrations/kuzu/*.py) because schema DDL differs. The
Meta singleton node records applied IDs; the runner skips any migration
whose ID is already present.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class Migration:
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
        # The singleton Meta node uses schema_version=0 as a fixed PK. This is
        # redundant on Memgraph (schema-free) but required on Kuzu, which
        # rejects MERGE that omits the primary key.
        rows = self._store.exec_read(
            "MATCH (m:Meta {schema_version: 0}) RETURN m.applied AS applied LIMIT 1",
            None,
        )
        if not rows:
            return set()
        return set(rows[0].get("applied") or [])

    def _record_applied(self, migration_id: int) -> None:
        # migration_id is a developer-controlled int (no injection surface);
        # inlining it sidesteps Kuzu's strict type inference on list-concat,
        # which rejects Python ints as INT8 against an INT64[] column.
        self._store.exec_write(
            f"MERGE (m:Meta {{schema_version: 0}}) "
            f"SET m.applied = coalesce(m.applied, []) + [{int(migration_id)}]",
            None,
        )
