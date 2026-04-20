import pytest

from contextd.storage.base import GraphStore

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


def test_delete_edges_requires_filter(backend: GraphStore) -> None:
    """delete_edges with no origin and no label must refuse — it would wipe
    every outgoing edge including structural and manual ones, violating the
    design invariant that wipe-and-replace operates only on origin='inferred'.
    """
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    with pytest.raises(ValueError, match="origin or label"):
        backend.delete_edges("a.md", src_label="File")
