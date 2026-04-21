"""Unit tests for Neo4jBackend skeleton (connect / close / capabilities)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# Temporary shim until Task 11.5 adds the real Neo4jConfig.
# Delete this block in Task 11.5.
from pydantic import BaseModel, ConfigDict


class Neo4jConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = 7687
    user: str = "neo4j"
    password: str = "neo4j"
    docker_compose_file: str = "~/.contextd/docker-compose.yml"
    memory_limit_gb: float = 1.0
    cpu_limit: float = 1.0


def test_capabilities_shape() -> None:
    from contextd.storage.neo4j import Neo4jBackend

    cfg = Neo4jConfig()
    backend = Neo4jBackend(cfg)
    caps = backend.capabilities
    assert caps.name == "neo4j"
    assert caps.concurrent_writers == -1  # unlimited
    assert caps.supports_vector_index is True
    assert caps.supports_full_text_index is True
    assert caps.supports_graph_algorithms is True
    assert caps.requires_docker is True
    assert caps.default_connection == "bolt://127.0.0.1:7687"


def test_connect_constructs_driver() -> None:
    from contextd.storage.neo4j import Neo4jBackend

    cfg = Neo4jConfig(host="127.0.0.1", port=7687, user="neo4j", password="test")
    with patch("contextd.storage.neo4j.GraphDatabase") as mock_gd:
        fake_driver = MagicMock()
        mock_gd.driver.return_value = fake_driver
        backend = Neo4jBackend(cfg)
        backend.connect()
        mock_gd.driver.assert_called_once_with("bolt://127.0.0.1:7687", auth=("neo4j", "test"))
        assert backend._driver is fake_driver


def test_close_closes_driver() -> None:
    from contextd.storage.neo4j import Neo4jBackend

    cfg = Neo4jConfig()
    backend = Neo4jBackend(cfg)
    fake = MagicMock()
    backend._driver = fake
    backend.close()
    fake.close.assert_called_once()
    assert backend._driver is None
