"""Parametrize integration and e2e tests across all storage backends.

Every ``backend`` fixture in this tree runs twice: once with Memgraph
via testcontainers, once with Neo4j via testcontainers. Tests that need
one backend only can branch on ``backend.capabilities.name`` and
``pytest.skip(...)`` with justification.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path  # noqa: F401 — retained for existing fixture signature

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy
from testcontainers.neo4j import Neo4jContainer

from contextd.config import MemgraphConfig, Neo4jConfig
from contextd.migrations.memgraph import ALL_MIGRATIONS as MEMGRAPH_MIGRATIONS
from contextd.migrations.neo4j import ALL_MIGRATIONS as NEO4J_MIGRATIONS
from contextd.storage.base import GraphStore
from contextd.storage.memgraph import MemgraphBackend
from contextd.storage.neo4j import Neo4jBackend


@pytest.fixture(params=["memgraph", "neo4j"])
def backend(request: pytest.FixtureRequest) -> Iterator[GraphStore]:
    name = request.param
    if name == "memgraph":
        # testcontainers-python has no built-in MemgraphContainer; use the
        # generic DockerContainer wrapper. memgraph:latest (v3.x) is required
        # because memgraph-platform:latest is pinned at v2.14 (predates
        # vector-index support).
        strategy = LogMessageWaitStrategy("You are running Memgraph").with_startup_timeout(120)
        container = (
            DockerContainer("memgraph/memgraph:latest")
            .with_exposed_ports(7687)
            .waiting_for(strategy)
        )
        container.start()
        try:
            cfg = MemgraphConfig(host="127.0.0.1", port=int(container.get_exposed_port(7687)))
            store: GraphStore = MemgraphBackend(cfg)
            store.connect()
            store.apply_migrations(MEMGRAPH_MIGRATIONS)
            yield store
            store.close()
        finally:
            container.stop()
    else:
        # Neo4j Community via testcontainers. 5.15-community supports vector
        # indexes (added in 5.11). Container exposes 7687 (Bolt) ephemerally.
        with Neo4jContainer("neo4j:5.15-community") as container:
            cfg_n4 = Neo4jConfig(
                host=container.get_container_host_ip(),
                port=int(container.get_exposed_port(7687)),
                user="neo4j",
                password=container.password,
            )
            store = Neo4jBackend(cfg_n4)
            store.connect()
            store.apply_migrations(NEO4J_MIGRATIONS)
            yield store
            store.close()
