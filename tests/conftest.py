"""Provide the Neo4j-backed ``backend`` fixture to integration and e2e tests.

Neo4j Community is the sole storage backend, so the fixture yields a single
backend (no parametrization). Tests in any ``tests/`` subdirectory inherit it
via pytest's upward conftest discovery.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

try:
    # testcontainers >= 4.14 moved the Neo4j module under `community` and
    # deprecated the old path. Since `filterwarnings = ["error", ...]` makes any
    # DeprecationWarning fatal, importing the legacy path on a new version aborts
    # collection for the entire test suite before a single test runs. Try the new
    # location first so the suite works on both, rather than pinning the library.
    from testcontainers.community.neo4j import Neo4jContainer
except ImportError:  # testcontainers < 4.14
    from testcontainers.neo4j import Neo4jContainer

from contextd.config import Neo4jConfig
from contextd.migrations.neo4j import ALL_MIGRATIONS as NEO4J_MIGRATIONS
from contextd.storage.base import GraphStore
from contextd.storage.neo4j import Neo4jBackend


@pytest.fixture
def backend() -> Iterator[GraphStore]:
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
