"""Migration _0004 backfills inferred_at on nodes with outgoing inferred edges.

Verifies the shape of the upgrade path: a graph that was bootstrapped before
the marker-writing code existed should still behave idempotently under the
new resume semantics (phase_relate* filtering on s.inferred_at IS NULL).

Parametrized on both backends via the top-level `backend` fixture.
"""

from __future__ import annotations

import pytest

from contextd.storage.base import GraphStore

pytestmark = pytest.mark.integration


def _seed(backend: GraphStore) -> None:
    # Two sections with outgoing inferred edges (simulating "relate already ran").
    backend.upsert_node("Section", {"id": "a#x", "corpus": "c", "summary": "s1"})
    backend.upsert_node("Section", {"id": "b#y", "corpus": "c", "summary": "s2"})
    # One section that's summarised but relate never ran on it.
    backend.upsert_node("Section", {"id": "c#z", "corpus": "c", "summary": "s3"})

    backend.upsert_node("Pattern", {"name": "target1"})
    backend.upsert_edge(
        "a#x", "target1", "USES", origin="inferred", src_label="Section", dst_label="Pattern"
    )
    backend.upsert_edge(
        "b#y", "target1", "USES", origin="inferred", src_label="Section", dst_label="Pattern"
    )

    # A File with outgoing inferred edge (file-granular case).
    backend.upsert_node("File", {"path": "/tmp/x.md", "corpus": "c", "summary": "fs"})
    backend.upsert_node("Pattern", {"name": "target2"})
    backend.upsert_edge(
        "/tmp/x.md", "target2", "USES", origin="inferred", src_label="File", dst_label="Pattern"
    )


def test_migration_0004_backfills_only_nodes_with_outgoing_inferred(
    backend: GraphStore,
) -> None:
    """The `backend` fixture re-applies migrations on connect. After seeding
    with inferred edges but no markers, running the migrations again should
    mark a#x / b#y / x.md as processed and leave c#z (no edges) unmarked.
    """
    _seed(backend)

    # Run _0004's DDL directly (not via MigrationRunner) because the backend
    # fixture already applied all migrations at connect time — MigrationRunner
    # would skip the already-recorded migration 4. coalesce makes the DDL
    # idempotent regardless.
    from contextd.migrations.neo4j import ALL_MIGRATIONS as NEO4J_MIGRATIONS

    _0004 = next(m for m in NEO4J_MIGRATIONS if m.id == 4)
    _0004.up(backend, _0004.id)

    rows = backend.exec_read(
        "MATCH (s:Section) RETURN s.id AS id, s.inferred_at IS NOT NULL AS marked",
        None,
    )
    marked = {r["id"]: r["marked"] for r in rows}
    assert marked["a#x"] is True
    assert marked["b#y"] is True
    assert marked["c#z"] is False  # zero outgoing inferred edges → stays unmarked

    rows = backend.exec_read(
        "MATCH (f:File) RETURN f.path AS path, f.inferred_at IS NOT NULL AS marked",
        None,
    )
    marked_files = {r["path"]: r["marked"] for r in rows}
    assert marked_files["/tmp/x.md"] is True
