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


def test_kuzu_upsert_edge_without_labels_raises(backend: GraphStore) -> None:
    """Kuzu REL tables declare FROM/TO label pairs; the backend must refuse
    a MERGE without both endpoint labels. Memgraph accepts (advisory-
    fallback) but Kuzu raises to keep callers honest."""
    if backend.capabilities.name != "kuzu":
        pytest.skip("Memgraph accepts label-less endpoints; this check is Kuzu-specific.")
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    backend.upsert_node("File", {"path": "b.md", "hash": "h2", "corpus": "c"})
    with pytest.raises(ValueError, match="requires both src_label and dst_label"):
        backend.upsert_edge("a.md", "b.md", "REFERENCES", origin="inferred")


def test_kuzu_delete_edges_without_src_label_raises(backend: GraphStore) -> None:
    if backend.capabilities.name != "kuzu":
        pytest.skip("Memgraph accepts label-less src; this check is Kuzu-specific.")
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    with pytest.raises(ValueError, match="requires src_label"):
        backend.delete_edges("a.md", origin="inferred")


def test_kuzu_upsert_edge_undeclared_property_wrapped(backend: GraphStore) -> None:
    """Passing a property that the REL table does not declare surfaces as a
    ValueError naming the edge type and property set — not Kuzu's bare
    'Cannot find property X for r.' binder exception."""
    if backend.capabilities.name != "kuzu":
        pytest.skip("Memgraph is schema-free on edges; this check is Kuzu-specific.")
    backend.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
    backend.upsert_node("Section", {"id": "a.md#s1", "title": "S1", "path": "a.md", "corpus": "c"})
    # CONTAINS REL table declares only `origin`; `weight` is undeclared.
    with pytest.raises(ValueError, match=r"REL table 'CONTAINS'.*weight"):
        backend.upsert_edge(
            "a.md",
            "a.md#s1",
            "CONTAINS",
            origin="structural",
            properties={"weight": 0.5},
            src_label="File",
            dst_label="Section",
        )


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


def test_kuzu_reupsert_with_changed_embedding_discards_new_vector(backend: GraphStore) -> None:
    """Kuzu rejects SET on vector-indexed columns after node creation. The
    two-phase upsert therefore silently keeps the original embedding on
    re-upsert. This test documents the behaviour so a future "helpful"
    refactor does not change it unnoticed — if the indexer ever needs to
    update an embedding, it must DETACH DELETE and CREATE (spec §5.5 path)."""
    if backend.capabilities.name != "kuzu":
        pytest.skip("Memgraph has no equivalent constraint; this is a Kuzu-only quirk.")
    original = [1.0, 0.0, 0.0] + [0.0] * 1021
    replacement = [0.0, 1.0, 0.0] + [0.0] * 1021
    backend.upsert_node(
        "File", {"path": "a.md", "hash": "h1", "embedding": original, "corpus": "c"}
    )
    backend.upsert_node(
        "File", {"path": "a.md", "hash": "h2", "embedding": replacement, "corpus": "c"}
    )
    rows = backend.exec_read(
        "MATCH (n:File {path: 'a.md'}) RETURN n.hash AS hash, n.embedding AS embedding"
    )
    assert rows[0]["hash"] == "h2"  # hash was updated (mutable property)
    # Embedding retained the original value — SET is not attempted.
    assert rows[0]["embedding"][:3] == original[:3]


def test_upsert_node_rejects_unsafe_label(backend: GraphStore) -> None:
    """Cross-backend: label must be an identifier-shaped string. The defence
    matters on Kuzu where the label is f-strung into CREATE; on Memgraph the
    same string lands in a MERGE pattern. Both reject at the same boundary."""
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


def test_kuzu_upsert_node_rejects_unsafe_property_key(backend: GraphStore) -> None:
    """Kuzu f-strings property keys into CREATE; a key that breaks Cypher
    quoting must fail at the Python boundary, not as a Kuzu binder error.
    Memgraph parameterises via SET n += $props so it doesn't raise here —
    the check is Kuzu-specific."""
    if backend.capabilities.name != "kuzu":
        pytest.skip("Memgraph parameterises property keys; no interpolation surface.")
    with pytest.raises(ValueError, match=r"property key"):
        backend.upsert_node("File", {"path": "a.md", "x'; DROP TABLE File; //": 1})


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
    backends (Kuzu: QUERY_VECTOR_INDEX; Memgraph: vector_search.search)."""
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
