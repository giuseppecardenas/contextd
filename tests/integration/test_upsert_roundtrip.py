from pathlib import Path

import pytest

from contextd.config import KuzuConfig
from contextd.migrations.kuzu import ALL_MIGRATIONS as KUZU_MIGRATIONS
from contextd.migrations.memgraph import ALL_MIGRATIONS as MEMGRAPH_MIGRATIONS
from contextd.storage.base import GraphStore
from contextd.storage.kuzu import KuzuBackend

pytestmark = pytest.mark.integration


def test_upsert_node_roundtrip(backend: GraphStore) -> None:
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    rows = backend.exec_read("MATCH (n:File) WHERE n.path = 'a.md' RETURN n.hash AS hash")
    assert rows[0]["hash"] == "h1"


def test_upsert_node_updates_non_pk_properties(backend: GraphStore) -> None:
    """Re-upsert with same PK but changed non-PK property must update, not duplicate.

    This is the indexer's re-index hot path: when a file's content hashes
    differently, we re-upsert with the same path and a new hash. The MERGE
    clause must match on PK only; matching on all properties creates a second
    node (or errors on PK collision) instead of updating.
    """
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    backend.upsert_node("File", {"path": "a.md", "hash": "h2", "corpus": "c"})
    rows = backend.exec_read("MATCH (n:File) WHERE n.path = 'a.md' RETURN n.hash AS hash")
    assert len(rows) == 1
    assert rows[0]["hash"] == "h2"


def test_upsert_edge_with_origin(backend: GraphStore) -> None:
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    backend.upsert_node("File", {"path": "b.md", "hash": "h2", "corpus": "c"})
    backend.upsert_edge(
        "a.md",
        "b.md",
        "REFERENCES",
        origin="structural",
        src_label="File",
        dst_label="File",
    )
    rows = backend.exec_read(
        "MATCH (a:File)-[r:REFERENCES]->(b:File) "
        "WHERE a.path = 'a.md' AND b.path = 'b.md' "
        "RETURN r.origin AS origin"
    )
    assert rows[0]["origin"] == "structural"


def test_delete_edges_by_origin(backend: GraphStore) -> None:
    # Use REFERENCES (File→File, inferred) + CONTAINS (File→Section, structural)
    # so the scenario works on both Memgraph (schema-free) and Kuzu
    # (schema-first rel tables declared with specific FROM/TO label pairs).
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    backend.upsert_node("File", {"path": "b.md", "hash": "h2", "corpus": "c"})
    backend.upsert_node("Section", {"id": "a.md#s1", "title": "S1", "path": "a.md", "corpus": "c"})
    backend.upsert_edge(
        "a.md",
        "b.md",
        "REFERENCES",
        origin="inferred",
        src_label="File",
        dst_label="File",
    )
    backend.upsert_edge(
        "a.md",
        "a.md#s1",
        "CONTAINS",
        origin="structural",
        src_label="File",
        dst_label="Section",
    )
    # Only the inferred edge should be removed.
    backend.delete_edges("a.md", origin="inferred", src_label="File")

    remaining_refs = backend.exec_read(
        "MATCH (a:File {path: 'a.md'})-[r:REFERENCES]->() RETURN r.origin AS origin"
    )
    remaining_contains = backend.exec_read(
        "MATCH (a:File {path: 'a.md'})-[r:CONTAINS]->() RETURN r.origin AS origin"
    )
    assert remaining_refs == []
    assert len(remaining_contains) == 1
    assert remaining_contains[0]["origin"] == "structural"


def test_upsert_edge_persists_non_origin_properties(backend: GraphStore) -> None:
    """Edge properties beyond `origin` must round-trip on both backends.

    The indexer attaches `confidence` and other scalars to inferred edges;
    silently dropping them would break provenance for the AI-inferred
    relationship review path.
    """
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    backend.upsert_node("File", {"path": "b.md", "hash": "h2", "corpus": "c"})
    backend.upsert_edge(
        "a.md",
        "b.md",
        "REFERENCES",
        origin="inferred",
        properties={"confidence": 0.87},
        src_label="File",
        dst_label="File",
    )
    rows = backend.exec_read(
        "MATCH (a:File {path: 'a.md'})-[r:REFERENCES]->(b:File {path: 'b.md'}) "
        "RETURN r.origin AS origin, r.confidence AS confidence"
    )
    assert len(rows) == 1
    assert rows[0]["origin"] == "inferred"
    assert rows[0]["confidence"] == 0.87


def test_migrations_are_idempotent(backend: GraphStore) -> None:
    """Re-running apply_migrations after a successful first apply must be a
    no-op: Meta.applied stays equal to the set of ids and no migration's
    up() re-runs (DDL re-runs would typically error on 'already exists').

    The backend fixture has already run ALL_MIGRATIONS once; this test calls
    apply_migrations a second time with the same list.
    """
    migrations = MEMGRAPH_MIGRATIONS if backend.capabilities.name == "memgraph" else KUZU_MIGRATIONS

    # Capture the applied set before the second apply.
    before = backend.exec_read("MATCH (m:Meta {schema_version: 0}) RETURN m.applied AS applied")
    applied_before = before[0]["applied"] if before else []

    backend.apply_migrations(migrations)

    after = backend.exec_read("MATCH (m:Meta {schema_version: 0}) RETURN m.applied AS applied")
    applied_after = after[0]["applied"] if after else []

    assert applied_before == applied_after
    # Every declared migration must appear exactly once in the applied list.
    expected_ids = {m.id for m in migrations}
    assert set(applied_after) == expected_ids
    assert len(applied_after) == len(expected_ids)


def test_read_only_kuzu_skips_meta_bootstrap(tmp_path: Path) -> None:
    """A read-only KuzuBackend must not try to create the Meta table. This is
    the MCP-server connection mode: indexer holds the writer, MCP opens
    read-only readers.
    """
    db_path = str(tmp_path / "graph")

    # Prime the DB with a writer first so the file + schema exist.
    writer = KuzuBackend(KuzuConfig(db_path=db_path))
    writer.connect()
    writer.apply_migrations(KUZU_MIGRATIONS)
    writer.close()

    reader = KuzuBackend(KuzuConfig(db_path=db_path), read_only=True)
    reader.connect()
    try:
        rows = reader.exec_read("MATCH (m:Meta {schema_version: 0}) RETURN m.applied AS applied")
        assert rows[0]["applied"] == list({m.id for m in KUZU_MIGRATIONS})
    finally:
        reader.close()


def test_delete_edges_requires_filter(backend: GraphStore) -> None:
    """delete_edges with no origin and no label must refuse — it would wipe
    every outgoing edge including structural and manual ones, violating the
    design invariant that wipe-and-replace operates only on origin='inferred'.
    """
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    with pytest.raises(ValueError, match="origin or label"):
        backend.delete_edges("a.md", src_label="File")
