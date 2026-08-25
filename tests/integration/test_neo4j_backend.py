"""Integration test: apply Neo4j baseline migration and verify schema exists."""

from __future__ import annotations

import pytest

try:
    # See the note in tests/conftest.py: testcontainers >= 4.14 moved this module
    # and deprecated the old path, and filterwarnings = ["error", ...] turns the
    # resulting DeprecationWarning into a collection error for the whole file.
    from testcontainers.community.neo4j import Neo4jContainer
except ImportError:  # testcontainers < 4.14
    from testcontainers.neo4j import Neo4jContainer

from contextd.config import Neo4jConfig

pytestmark = pytest.mark.integration


@pytest.fixture
def neo4j_backend():
    from contextd.storage.neo4j import Neo4jBackend

    with Neo4jContainer("neo4j:5.15-community") as container:
        cfg = Neo4jConfig(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(7687)),
            user="neo4j",
            password=container.password,
        )
        backend = Neo4jBackend(cfg)
        backend.connect()
        yield backend
        backend.close()


def test_baseline_migration_creates_indexes(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    # Verify vector index on File.embedding exists.
    rows = neo4j_backend.exec_read("SHOW VECTOR INDEXES YIELD name RETURN collect(name) AS names")
    names = rows[0]["names"]
    assert "File_embedding_idx" in names
    assert "Section_embedding_idx" in names

    # Verify uniqueness constraint on File.path.
    rows = neo4j_backend.exec_read("SHOW CONSTRAINTS YIELD name RETURN collect(name) AS names")
    names = rows[0]["names"]
    assert any("File" in n and "path" in n.lower() for n in names)


def test_upsert_node_roundtrip(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    pk = neo4j_backend.upsert_node(
        "File",
        {"path": "/a.md", "name": "a.md", "corpus": "test", "embedding": [0.1] * 1024},
    )
    assert pk == "/a.md"
    rows = neo4j_backend.exec_read(
        "MATCH (n:File {path: $p}) RETURN n.name AS name",
        {"p": "/a.md"},
    )
    assert rows[0]["name"] == "a.md"

    # Re-upsert updates mutable properties.
    neo4j_backend.upsert_node(
        "File",
        {"path": "/a.md", "name": "renamed.md", "corpus": "test", "embedding": [0.1] * 1024},
    )
    rows = neo4j_backend.exec_read(
        "MATCH (n:File {path: $p}) RETURN n.name AS name",
        {"p": "/a.md"},
    )
    assert rows[0]["name"] == "renamed.md"


def test_upsert_edge_and_delete_edges(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    # Two File nodes.
    neo4j_backend.upsert_node("File", {"path": "/a.md", "corpus": "t"})
    neo4j_backend.upsert_node("File", {"path": "/b.md", "corpus": "t"})
    neo4j_backend.upsert_edge(
        "/a.md",
        "/b.md",
        "REFERENCES",
        origin="inferred",
        properties={"confidence": 0.9},
        src_label="File",
        dst_label="File",
    )
    rows = neo4j_backend.exec_read("MATCH ()-[r:REFERENCES]->() RETURN count(r) AS c")
    assert rows[0]["c"] == 1

    # Delete inferred edges from /a.md.
    neo4j_backend.delete_edges("/a.md", origin="inferred", src_label="File")
    rows = neo4j_backend.exec_read("MATCH ()-[r:REFERENCES]->() RETURN count(r) AS c")
    assert rows[0]["c"] == 0


def test_delete_edges_unfiltered_raises(neo4j_backend) -> None:
    with pytest.raises(ValueError, match="requires at least one of"):
        neo4j_backend.delete_edges("/a.md", src_label="File")


def test_upsert_edge_requires_labels(neo4j_backend) -> None:
    """Both endpoint labels must be supplied on Neo4j — omitting either
    silently binds zero rows (schema-free MATCH) and loses writes."""
    with pytest.raises(ValueError, match="requires both src_label and dst_label"):
        neo4j_backend.upsert_edge("/a.md", "/b.md", "REFERENCES", origin="inferred")
    with pytest.raises(ValueError, match="requires both src_label and dst_label"):
        neo4j_backend.upsert_edge(
            "/a.md", "/b.md", "REFERENCES", origin="inferred", src_label="File"
        )
    with pytest.raises(ValueError, match="requires both src_label and dst_label"):
        neo4j_backend.upsert_edge(
            "/a.md", "/b.md", "REFERENCES", origin="inferred", dst_label="File"
        )


def test_delete_edges_requires_src_label(neo4j_backend) -> None:
    with pytest.raises(ValueError, match="requires src_label"):
        neo4j_backend.delete_edges("/a.md", origin="inferred")


def test_vector_search_roundtrip(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    neo4j_backend.upsert_node(
        "File", {"path": "/a.md", "corpus": "t", "embedding": [1.0] + [0.0] * 1023}
    )
    neo4j_backend.upsert_node(
        "File", {"path": "/b.md", "corpus": "t", "embedding": [0.0, 1.0] + [0.0] * 1022}
    )
    neo4j_backend.upsert_node(
        "File", {"path": "/c.md", "corpus": "t", "embedding": [1.0] + [0.0] * 1023}
    )

    results = neo4j_backend.vector_search("File", "embedding", query=[1.0] + [0.0] * 1023, k=3)
    # /a.md and /c.md are identical to query; /b.md is orthogonal.
    paths = [r["node"]["path"] for r in results]
    scores = [r["score"] for r in results]
    assert paths[0] in {"/a.md", "/c.md"}
    # First two scores should be ~1.0 (identical direction).
    assert scores[0] > 0.99
    # Orthogonal vector should score ~0.5 (cosine 0.0 → similarity 0.5 after
    # Neo4j's [0,1] normalisation).
    assert "/b.md" in paths


def test_vector_search_threshold_filter(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    neo4j_backend.upsert_node(
        "File", {"path": "/a.md", "corpus": "t", "embedding": [1.0] + [0.0] * 1023}
    )
    neo4j_backend.upsert_node(
        "File", {"path": "/b.md", "corpus": "t", "embedding": [0.0, 1.0] + [0.0] * 1022}
    )
    results = neo4j_backend.vector_search(
        "File",
        "embedding",
        query=[1.0] + [0.0] * 1023,
        k=10,
        threshold=0.9,
    )
    paths = [r["node"]["path"] for r in results]
    assert "/a.md" in paths
    assert "/b.md" not in paths


def test_full_text_search_roundtrip(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    neo4j_backend.upsert_node(
        "File", {"path": "/a.md", "corpus": "t", "summary": "alpha beta gamma"}
    )
    neo4j_backend.upsert_node("File", {"path": "/b.md", "corpus": "t", "summary": "delta epsilon"})
    results = neo4j_backend.full_text_search("File", "summary", "alpha", k=5)
    assert len(results) == 1
    assert results[0]["node"]["path"] == "/a.md"
    assert results[0]["score"] > 0


# --- filtered search ----------------------------------------------------------


def _seed_two_corpora(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])
    vec = [1.0] + [0.0] * 1023
    for i in range(3):
        neo4j_backend.upsert_node(
            "File",
            {
                "path": f"/a{i}.md",
                "corpus": "alpha",
                "summary": "shared marker token",
                "embedding": vec,
            },
        )
    neo4j_backend.upsert_node(
        "File",
        {"path": "/b.md", "corpus": "beta", "summary": "shared marker token", "embedding": vec},
    )


def test_vector_search_filters_by_corpus(neo4j_backend) -> None:
    _seed_two_corpora(neo4j_backend)
    vec = [1.0] + [0.0] * 1023
    # Without the filter, k=1 on identical vectors could return any corpus;
    # with it, only the single beta node qualifies even though alpha nodes
    # outnumber it in the index's top-k.
    hits = neo4j_backend.vector_search("File", "embedding", vec, k=1, filters={"corpus": "beta"})
    assert [r["node"]["path"] for r in hits] == ["/b.md"]
    hits = neo4j_backend.vector_search("File", "embedding", vec, k=10, filters={"corpus": "alpha"})
    assert {r["node"]["corpus"] for r in hits} == {"alpha"}
    assert len(hits) == 3
    assert (
        neo4j_backend.vector_search("File", "embedding", vec, k=10, filters={"corpus": "nope"})
        == []
    )


def test_full_text_search_filters_by_corpus(neo4j_backend) -> None:
    _seed_two_corpora(neo4j_backend)
    hits = neo4j_backend.full_text_search(
        "File", "summary", "marker", k=1, filters={"corpus": "beta"}
    )
    assert [r["node"]["path"] for r in hits] == ["/b.md"]
    hits = neo4j_backend.full_text_search(
        "File", "summary", "marker", k=10, filters={"corpus": "alpha"}
    )
    assert len(hits) == 3
    assert {r["node"]["corpus"] for r in hits} == {"alpha"}


def test_filtered_search_limit_is_callers_k(neo4j_backend) -> None:
    _seed_two_corpora(neo4j_backend)
    vec = [1.0] + [0.0] * 1023
    hits = neo4j_backend.vector_search("File", "embedding", vec, k=2, filters={"corpus": "alpha"})
    assert len(hits) == 2
    hits = neo4j_backend.full_text_search(
        "File", "summary", "marker", k=2, filters={"corpus": "alpha"}
    )
    assert len(hits) == 2


# --- upsert_nodes / delete_nodes ----------------------------------------------


def _apply_chunk_schema(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration as m1
    from contextd.migrations.neo4j._0008_chunks_and_topics import migration as m8

    neo4j_backend.apply_migrations([m1, m8])


def test_upsert_nodes_roundtrip_and_update(neo4j_backend) -> None:
    _apply_chunk_schema(neo4j_backend)
    rows = [
        {"id": f"a.md#s~fine~{i}", "corpus": "t", "parent_id": "a.md#s", "ordinal": i, "text": "v1"}
        for i in range(3)
    ]
    assert neo4j_backend.upsert_nodes("Chunk", rows) == 3
    got = neo4j_backend.exec_read(
        "MATCH (c:Chunk {corpus: 't'}) RETURN c.id AS id, c.text AS text ORDER BY c.ordinal"
    )
    assert [r["id"] for r in got] == [r["id"] for r in rows]
    assert {r["text"] for r in got} == {"v1"}

    # Re-upsert MERGEs on id: same node count, properties updated in place.
    for r in rows:
        r["text"] = "v2"
    assert neo4j_backend.upsert_nodes("Chunk", rows) == 3
    got = neo4j_backend.exec_read(
        "MATCH (c:Chunk {corpus: 't'}) RETURN count(c) AS n, collect(c.text) AS t"
    )
    assert got[0]["n"] == 3
    assert set(got[0]["t"]) == {"v2"}


def test_upsert_nodes_batches_beyond_500(neo4j_backend) -> None:
    _apply_chunk_schema(neo4j_backend)
    rows = [{"id": f"big~fine~{i}", "corpus": "big", "ordinal": i} for i in range(1203)]
    assert neo4j_backend.upsert_nodes("Chunk", rows) == 1203
    got = neo4j_backend.exec_read("MATCH (c:Chunk {corpus: 'big'}) RETURN count(c) AS n")
    assert got[0]["n"] == 1203


def test_upsert_nodes_empty_is_noop(neo4j_backend) -> None:
    _apply_chunk_schema(neo4j_backend)
    assert neo4j_backend.upsert_nodes("Chunk", []) == 0


def test_delete_nodes_counts_and_detaches(neo4j_backend) -> None:
    _apply_chunk_schema(neo4j_backend)
    neo4j_backend.upsert_node("File", {"path": "p.md", "corpus": "d"})
    neo4j_backend.upsert_nodes(
        "Chunk",
        [
            {"id": "p.md~fine~0", "corpus": "d", "parent_id": "p.md", "profile": "fine"},
            {"id": "p.md~fine~1", "corpus": "d", "parent_id": "p.md", "profile": "fine"},
            {"id": "p.md~coarse~0", "corpus": "d", "parent_id": "p.md", "profile": "coarse"},
            {"id": "q.md~fine~0", "corpus": "other", "parent_id": "q.md", "profile": "fine"},
        ],
    )
    for cid in ("p.md~fine~0", "p.md~fine~1", "p.md~coarse~0"):
        neo4j_backend.upsert_edge(
            "p.md", cid, "CONTAINS", origin="structural", src_label="File", dst_label="Chunk"
        )
    neo4j_backend.upsert_edge(
        "p.md~fine~0",
        "p.md~fine~1",
        "NEXT_SIBLING",
        origin="structural",
        src_label="Chunk",
        dst_label="Chunk",
    )

    # Two predicates AND together; structural edges go with the nodes.
    assert neo4j_backend.delete_nodes("Chunk", where={"corpus": "d", "profile": "fine"}) == 2
    remaining = neo4j_backend.exec_read("MATCH (c:Chunk) RETURN c.id AS id ORDER BY id")
    assert [r["id"] for r in remaining] == ["p.md~coarse~0", "q.md~fine~0"]
    edges = neo4j_backend.exec_read("MATCH ()-[r]->() RETURN count(r) AS n")
    assert edges[0]["n"] == 1  # only File-CONTAINS->coarse chunk survives

    # No match: zero, not an error.
    assert neo4j_backend.delete_nodes("Chunk", where={"corpus": "missing"}) == 0
    # The parent File is untouched by a Chunk-scoped delete.
    assert neo4j_backend.exec_read("MATCH (f:File) RETURN count(f) AS n")[0]["n"] == 1


def test_delete_nodes_requires_where(neo4j_backend) -> None:
    with pytest.raises(ValueError, match="non-empty where"):
        neo4j_backend.delete_nodes("Chunk", where={})
