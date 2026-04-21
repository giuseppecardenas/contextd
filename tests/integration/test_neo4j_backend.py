"""Integration test: apply Neo4j baseline migration and verify schema exists."""

from __future__ import annotations

import pytest
from testcontainers.neo4j import Neo4jContainer

pytestmark = pytest.mark.integration


@pytest.fixture
def neo4j_backend():
    from contextd.config import Neo4jConfig
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
