"""Backfill ``inferred_at`` on Section/File nodes that already have inferred edges.

Sibling of ``contextd/migrations/neo4j/_0004_backfill_inferred_at.py`` —
see that file's docstring for the full rationale.

Memgraph and Neo4j both support the ``[{origin: 'inferred'}]`` map-literal
syntax on relationship patterns and the ``coalesce`` expression, so the
DDL is identical across backends.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    "MATCH (s:Section) WHERE (s)-[{origin: 'inferred'}]->() "
    "SET s.inferred_at = coalesce(s.inferred_at, localDateTime())",
    "MATCH (f:File) WHERE (f)-[{origin: 'inferred'}]->() "
    "SET f.inferred_at = coalesce(f.inferred_at, localDateTime())",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=4, name="backfill_inferred_at_memgraph", up=up)
