"""Migration _0005 purges phantom File/Section stubs and canonicalizes paths.

Phantom stubs (path-null Sections, hash-null Files) were minted by the old
``phase_relate*`` for unresolvable inferred-edge targets; the indexer no longer
creates them and this migration removes the existing ones (with their dangling
inferred edges). Legacy backslash path identities are rewritten to the
canonical forward-slash form so a re-index updates the existing node instead of
creating a twin.

Parametrized on both backends via the top-level `backend` fixture.
"""

from __future__ import annotations

import pytest

from contextd.storage.base import GraphStore
from contextd.storage.migration import Migration

pytestmark = pytest.mark.integration


def _migration_0005(backend: GraphStore) -> Migration:
    from contextd.migrations.memgraph import ALL_MIGRATIONS as MEMGRAPH_MIGRATIONS
    from contextd.migrations.neo4j import ALL_MIGRATIONS as NEO4J_MIGRATIONS
    from contextd.storage.memgraph import MemgraphBackend

    migrations = MEMGRAPH_MIGRATIONS if isinstance(backend, MemgraphBackend) else NEO4J_MIGRATIONS
    return next(m for m in migrations if m.id == 5)


def _seed(backend: GraphStore) -> None:
    # Real nodes (have hash / path) — must survive.
    backend.upsert_node(
        "File", {"path": "/clean/real.md", "name": "real.md", "hash": "h1", "corpus": "purge_t"}
    )
    backend.upsert_node(
        "Section",
        {"id": "/clean/real.md#sec", "path": "/clean/real.md", "corpus": "purge_t", "summary": "s"},
    )
    # Phantom stubs (no hash / no path) — must be deleted.
    backend.upsert_node("File", {"path": "ghostfile.md", "corpus": "purge_t"})
    backend.upsert_node("Section", {"id": "ghost#anchor", "corpus": "purge_t"})
    # An inferred edge into a phantom stub — DETACH DELETE must drop it too.
    backend.upsert_node("Section", {"id": "/clean/real.md#ref", "path": "/clean/real.md"})
    backend.upsert_edge(
        "/clean/real.md#ref",
        "ghost#anchor",
        "REFERENCES",
        origin="inferred",
        src_label="Section",
        dst_label="Section",
    )
    # Legacy backslash identities — must be canonicalized to forward slashes.
    backend.upsert_node(
        "File", {"path": r"C:\win\file.md", "name": "file.md", "hash": "h2", "corpus": "purge_t"}
    )
    backend.upsert_node(
        "Section",
        {
            "id": r"C:\win\file.md#s",
            "path": r"C:\win\file.md",
            "file_id": r"C:\win\file.md",
            "corpus": "purge_t",
            "summary": "s2",
        },
    )


def test_migration_0005_purges_stubs_and_canonicalizes(backend: GraphStore) -> None:
    _seed(backend)
    _migration_0005(backend).up(backend, 5)

    # Phantom stubs gone.
    assert backend.exec_read("MATCH (f:File {path: 'ghostfile.md'}) RETURN f.path AS p", None) == []
    assert backend.exec_read("MATCH (s:Section {id: 'ghost#anchor'}) RETURN s.id AS i", None) == []
    # The inferred edge that pointed at the deleted stub is gone with it.
    assert backend.exec_read("MATCH ()-[r:REFERENCES]->() RETURN count(r) AS c", None)[0]["c"] == 0

    # Real nodes preserved.
    assert backend.exec_read("MATCH (f:File {path: '/clean/real.md'}) RETURN f.path AS p", None)
    assert backend.exec_read("MATCH (s:Section {id: '/clean/real.md#sec'}) RETURN s.id AS i", None)

    # Backslash identities canonicalized; the backslash form no longer matches.
    assert (
        len(backend.exec_read("MATCH (f:File {path: 'C:/win/file.md'}) RETURN f.path AS p", None))
        == 1
    )
    sec = backend.exec_read(
        "MATCH (s:Section {id: 'C:/win/file.md#s'}) RETURN s.path AS p, s.file_id AS fid",
        None,
    )
    assert len(sec) == 1
    assert sec[0]["p"] == "C:/win/file.md"
    assert sec[0]["fid"] == "C:/win/file.md"
