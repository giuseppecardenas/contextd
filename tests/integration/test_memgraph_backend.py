"""Integration tests for MemgraphBackend.

Uses testcontainers' generic DockerContainer wrapper against the
memgraph/memgraph-platform image. The plan originally referenced
testcontainers.memgraph which does not ship in testcontainers-python 4.x;
we use DockerContainer + a log-based readiness wait instead.
"""

from collections.abc import Iterator

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from contextd.config import MemgraphConfig
from contextd.migrations.memgraph import ALL_MIGRATIONS
from contextd.storage.memgraph import MemgraphBackend
from contextd.storage.migration import MigrationRunner

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def memgraph_port() -> Iterator[int]:
    strategy = LogMessageWaitStrategy("You are running Memgraph").with_startup_timeout(120)
    # memgraph-platform:latest is pinned at v2.14.1, which predates vector-index
    # support (introduced in v2.18). memgraph:latest is v3.x and supports vector
    # indexes + full-text search.
    container = (
        DockerContainer("memgraph/memgraph:latest").with_exposed_ports(7687).waiting_for(strategy)
    )
    container.start()
    try:
        yield int(container.get_exposed_port(7687))
    finally:
        container.stop()


def test_connect_and_run_migrations(memgraph_port: int) -> None:
    cfg = MemgraphConfig(host="127.0.0.1", port=memgraph_port)
    store = MemgraphBackend(cfg)
    store.connect()
    try:
        MigrationRunner(store, ALL_MIGRATIONS).apply()

        meta = store.exec_read("MATCH (m:Meta) RETURN m.applied AS applied")
        # Both migrations (1 = baseline, 2 = corpus_stats) must be recorded.
        assert meta[0]["applied"] == [1, 2]

        # Upsert a node and read it back.
        store.upsert_node("File", {"path": "a.md", "hash": "h1"})
        rows = store.exec_read("MATCH (n:File {path: 'a.md'}) RETURN n.hash AS hash")
        assert rows[0]["hash"] == "h1"
    finally:
        store.close()


def test_capabilities(memgraph_port: int) -> None:
    cfg = MemgraphConfig(host="127.0.0.1", port=memgraph_port)
    store = MemgraphBackend(cfg)
    caps = store.capabilities
    assert caps.name == "memgraph"
    assert caps.requires_docker is True
    assert caps.concurrent_writers == -1
    assert caps.supports_graph_algorithms is True
