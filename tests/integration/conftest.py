"""Parametrize integration tests across both storage backends.

Every ``backend`` fixture in this directory runs twice: once with
Memgraph via testcontainers, once with Kuzu via tmp_path. Tests that
need one backend only can skip via ``pytest.mark.memgraph_only`` or
``pytest.mark.kuzu_only``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from contextd.config import KuzuConfig, MemgraphConfig
from contextd.migrations.kuzu import ALL_MIGRATIONS as KUZU_MIGRATIONS
from contextd.migrations.memgraph import ALL_MIGRATIONS as MEMGRAPH_MIGRATIONS
from contextd.storage.base import GraphStore
from contextd.storage.kuzu import KuzuBackend
from contextd.storage.memgraph import MemgraphBackend


@pytest.fixture(params=["memgraph", "kuzu"])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[GraphStore]:
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
        cfg = KuzuConfig(db_path=str(tmp_path / "graph"))
        store = KuzuBackend(cfg)
        store.connect()
        store.apply_migrations(KUZU_MIGRATIONS)
        yield store
        store.close()
