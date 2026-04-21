"""Integration test: apply Neo4j baseline migration and verify schema exists."""

from __future__ import annotations

import pytest
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
