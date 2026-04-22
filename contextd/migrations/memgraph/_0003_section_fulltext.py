"""Full-text index on :Section for section-granular corpora.

Sibling to Neo4j's _0003_section_fulltext. Memgraph's TEXT INDEX is
label-scoped (no property list); the storage layer's full-text query
resolves ``Section_summary_ft`` by convention on the label+property pair.

Idempotent: Memgraph's ``CREATE TEXT INDEX`` is a no-op on already-present
indexes of the same name.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    "CREATE TEXT INDEX Section_summary_ft ON :Section",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=3, name="section_fulltext_memgraph", up=up)
