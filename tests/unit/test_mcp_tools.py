"""Unit tests for MCP tool behaviour — Cypher shape, clamps, and descriptors.

Integration coverage (queries actually executing against Memgraph + Neo4j)
lives in tests/integration/test_mcp_tools.py; this file exercises the
pure-Python surface that doesn't need a backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from contextd.mcp import tools


def test_related_clamps_depth_above_max() -> None:
    """Defence in depth: a direct caller passing depth=100 must not
    reach Memgraph as `[r*1..100]`. Spec-delta #32 clamps at the MCP
    descriptor level; this mirrors the clamp in-function."""
    store = MagicMock()
    store.exec_read.return_value = []
    tools.related(store, "some-id", depth=100)
    cypher = store.exec_read.call_args[0][0]
    assert "[r*1..5]" in cypher
    assert "100" not in cypher


def test_related_clamps_depth_below_min() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    tools.related(store, "some-id", depth=-3)
    cypher = store.exec_read.call_args[0][0]
    assert "[r*1..1]" in cypher


def test_related_default_depth_is_2() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    tools.related(store, "some-id")
    cypher = store.exec_read.call_args[0][0]
    assert "[r*1..2]" in cypher


def test_related_passes_node_id_as_param() -> None:
    """Cypher must not f-string node_id — injection vector."""
    store = MagicMock()
    store.exec_read.return_value = []
    tools.related(store, "file/with/slashes", depth=3)
    cypher, params = store.exec_read.call_args[0]
    assert params == {"id": "file/with/slashes"}
    assert "file/with/slashes" not in cypher
