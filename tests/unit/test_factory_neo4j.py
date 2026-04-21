"""Factory wiring for Neo4j backend."""

from __future__ import annotations

from contextd.config import Config, Neo4jConfig, StorageConfig


def test_factory_returns_neo4j_when_backend_is_neo4j() -> None:
    from contextd.storage.factory import build_graph_store
    from contextd.storage.neo4j import Neo4jBackend

    cfg = Config(storage=StorageConfig(backend="neo4j", neo4j=Neo4jConfig()))
    store = build_graph_store(cfg)
    assert isinstance(store, Neo4jBackend)


def test_neo4j_config_defaults() -> None:
    cfg = Neo4jConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 7687
    assert cfg.user == "neo4j"
    assert cfg.password == "contextd"
    assert cfg.docker_compose_file == "~/.contextd/docker-compose.yml"
    assert cfg.memory_limit_gb == 1.0
    assert cfg.cpu_limit == 1.0
