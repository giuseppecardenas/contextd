from pathlib import Path

import pytest

from contextd.config import KuzuConfig
from contextd.migrations.kuzu import ALL_MIGRATIONS
from contextd.storage.kuzu import KuzuBackend
from contextd.storage.migration import MigrationRunner

pytestmark = pytest.mark.integration


def test_connect_and_run_migrations(tmp_path: Path) -> None:
    cfg = KuzuConfig(db_path=str(tmp_path / "graph"))
    store = KuzuBackend(cfg)
    store.connect()
    try:
        MigrationRunner(store, ALL_MIGRATIONS).apply()

        store.upsert_node("File", {"path": "a.md", "hash": "h1", "corpus": "c"})
        rows = store.exec_read("MATCH (n:File) WHERE n.path = 'a.md' RETURN n.hash AS hash")
        assert rows[0]["hash"] == "h1"
    finally:
        store.close()


def test_capabilities(tmp_path: Path) -> None:
    cfg = KuzuConfig(db_path=str(tmp_path / "graph"))
    store = KuzuBackend(cfg)
    caps = store.capabilities
    assert caps.name == "kuzu"
    assert caps.requires_docker is False
    assert caps.concurrent_writers == 1
    assert caps.supports_graph_algorithms is False
