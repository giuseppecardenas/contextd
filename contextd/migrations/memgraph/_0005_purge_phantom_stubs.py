r"""Purge phantom File/Section stubs; canonicalize path identity (Memgraph).

Sibling of ``contextd/migrations/neo4j/_0005_purge_phantom_stubs.py`` — see
that file's docstring for the full rationale. Memgraph supports ``DETACH
DELETE``, ``replace()`` and ``CONTAINS`` with the same semantics, so the DDL
is identical across backends.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    "MATCH (s:Section) WHERE s.path IS NULL DETACH DELETE s",
    "MATCH (f:File) WHERE f.hash IS NULL DETACH DELETE f",
    r"MATCH (f:File) WHERE f.path CONTAINS '\\' SET f.path = replace(f.path, '\\', '/')",
    r"MATCH (s:Section) WHERE s.id CONTAINS '\\' "
    r"SET s.id = replace(s.id, '\\', '/'), "
    r"s.path = replace(s.path, '\\', '/'), "
    r"s.file_id = replace(s.file_id, '\\', '/')",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=5, name="purge_phantom_stubs_canonicalize_paths_memgraph", up=up)
