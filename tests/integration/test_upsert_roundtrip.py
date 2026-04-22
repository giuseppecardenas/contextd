import pytest

from contextd.migrations.memgraph import ALL_MIGRATIONS as MEMGRAPH_MIGRATIONS
from contextd.migrations.neo4j import ALL_MIGRATIONS as NEO4J_MIGRATIONS
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


def test_risk_merges_on_identical_description(backend: GraphStore) -> None:
    """Risk PK is ``description``; two upserts with identical description
    but different non-PK properties MERGE into one node. This is the
    idempotent re-index invariant documented in contextd/storage/_keys.py:
    the inferrer emits Risks by their description text, so identical text
    means "the same Risk, updated" — not a duplicate."""
    backend.upsert_node(
        "Risk",
        {
            "description": "SHA=pending for FR-001",
            "severity": "low",
            "corpus": "test",
        },
    )
    backend.upsert_node(
        "Risk",
        {
            "description": "SHA=pending for FR-001",
            "severity": "high",
            "corpus": "test",
        },
    )
    rows = backend.exec_read(
        "MATCH (r:Risk {description: $desc}) RETURN r.severity AS severity",
        {"desc": "SHA=pending for FR-001"},
    )
    assert len(rows) == 1, "identical descriptions must MERGE into one node"
    assert rows[0]["severity"] == "high", (
        "the second upsert's non-PK properties must overwrite the first"
    )


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
    # Use REFERENCES (File→File, inferred) + CONTAINS (File→Section, structural).
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
    name = backend.capabilities.name
    if name == "memgraph":
        migrations = MEMGRAPH_MIGRATIONS
    elif name == "neo4j":
        migrations = NEO4J_MIGRATIONS
    else:
        raise AssertionError(f"unexpected backend: {name}")

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


def test_delete_edges_requires_filter(backend: GraphStore) -> None:
    """delete_edges with no origin and no edge_type must refuse — it would
    wipe every outgoing edge including structural and manual ones, violating
    the design invariant that wipe-and-replace operates only on
    origin='inferred'.
    """
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    with pytest.raises(ValueError, match="origin or edge_type"):
        backend.delete_edges("a.md", src_label="File")


def test_upsert_node_missing_pk_raises(backend: GraphStore) -> None:
    """Calling upsert_node with properties that omit the label's declared
    primary key surfaces a clear error naming the missing key — not a
    surprising Cypher binder exception deep in the backend."""
    with pytest.raises(ValueError, match="missing required primary key 'path'"):
        backend.upsert_node("File", {"hash": "h1", "corpus": "c"})


def test_memgraph_upsert_edge_without_labels_works(backend: GraphStore) -> None:
    """Memgraph's label kwargs are advisory; omitting them falls back to
    OR-matching against path/id/name and still round-trips."""
    if backend.capabilities.name != "memgraph":
        pytest.skip("Advisory-endpoint behaviour is Memgraph-only.")
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    backend.upsert_node("File", {"path": "b.md", "hash": "h2", "corpus": "c"})
    backend.upsert_edge("a.md", "b.md", "REFERENCES", origin="structural")
    rows = backend.exec_read(
        "MATCH (a:File {path: 'a.md'})-[r:REFERENCES]->(b:File {path: 'b.md'}) "
        "RETURN r.origin AS origin"
    )
    assert rows[0]["origin"] == "structural"


def test_upsert_node_rejects_unsafe_label(backend: GraphStore) -> None:
    """Cross-backend: label must be an identifier-shaped string. Both
    backends reject at the same boundary before interpolating into Cypher."""
    with pytest.raises(ValueError, match=r"label must match"):
        backend.upsert_node("File); DROP TABLE File; //", {"path": "a.md"})


def test_upsert_node_rejects_unknown_label(backend: GraphStore) -> None:
    """An identifier-shaped but ontology-unknown label must raise via
    primary_key_for — a typo should not silently route to a different PK."""
    with pytest.raises(ValueError, match=r"Unknown node label"):
        backend.upsert_node("Fiel", {"path": "a.md"})


def test_upsert_edge_rejects_unsafe_edge_type(backend: GraphStore) -> None:
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    backend.upsert_node("File", {"path": "b.md", "hash": "h2", "corpus": "c"})
    with pytest.raises(ValueError, match=r"edge_type must match"):
        backend.upsert_edge(
            "a.md",
            "b.md",
            "REF}) DETACH DELETE a, b //",
            origin="structural",
            src_label="File",
            dst_label="File",
        )


def test_delete_edges_rejects_unsafe_edge_type(backend: GraphStore) -> None:
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    with pytest.raises(ValueError, match=r"edge_type must match"):
        backend.delete_edges(
            "a.md",
            edge_type="REF]->() DETACH DELETE a //",
            src_label="File",
        )


def test_vector_search_rejects_unsafe_identifiers(backend: GraphStore) -> None:
    """Label and property_name are f-strung into the procedure call on both
    backends, so they must match an identifier shape."""
    with pytest.raises(ValueError, match=r"label must match"):
        backend.vector_search(
            label="File') YIELD node, score RETURN node //",
            property_name="embedding",
            query=[0.0] * 1024,
            k=3,
        )
    with pytest.raises(ValueError, match=r"property_name must match"):
        backend.vector_search(
            label="File",
            property_name="embedding') //",
            query=[0.0] * 1024,
            k=3,
        )


def test_vector_search_rejects_bad_k(backend: GraphStore) -> None:
    with pytest.raises(ValueError, match=r"non-bool int"):
        backend.vector_search(
            label="File",
            property_name="embedding",
            query=[0.0] * 1024,
            k=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r">= 1"):
        backend.vector_search(label="File", property_name="embedding", query=[0.0] * 1024, k=0)


def test_vector_search_rejects_out_of_range_threshold(backend: GraphStore) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        backend.vector_search(
            label="File",
            property_name="embedding",
            query=[0.0] * 1024,
            k=3,
            threshold=1.5,
        )


def test_full_text_search_rejects_unsafe_identifiers(backend: GraphStore) -> None:
    with pytest.raises(ValueError, match=r"label must match"):
        backend.full_text_search(
            label="File') CALL drop() //",
            property_name="summary",
            query="x",
            k=3,
        )
