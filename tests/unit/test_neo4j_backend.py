"""Unit tests for Neo4jBackend: lifecycle, and the Cypher shape of the filtered
search / batch upsert / scoped delete methods against a mocked driver."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from contextd.config import Neo4jConfig
from contextd.storage.neo4j import Neo4jBackend


def test_capabilities_shape() -> None:
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
    cfg = Neo4jConfig(host="127.0.0.1", port=7687, user="neo4j", password="test")
    with patch("contextd.storage.neo4j.GraphDatabase") as mock_gd:
        fake_driver = MagicMock()
        mock_gd.driver.return_value = fake_driver
        backend = Neo4jBackend(cfg)
        backend.connect()
        mock_gd.driver.assert_called_once_with("bolt://127.0.0.1:7687", auth=("neo4j", "test"))
        assert backend._driver is fake_driver


def test_close_closes_driver() -> None:
    cfg = Neo4jConfig()
    backend = Neo4jBackend(cfg)
    fake = MagicMock()
    backend._driver = fake
    backend.close()
    fake.close.assert_called_once()
    assert backend._driver is None


# --- mocked-driver helpers ----------------------------------------------------


def _mocked_backend(**kwargs: Any) -> tuple[Neo4jBackend, MagicMock]:
    """Backend whose driver's ``session()`` context yields ``session``."""
    backend = Neo4jBackend(Neo4jConfig(), **kwargs)
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    # ``session.run(...)`` yields a Result that is both iterable (search) and
    # exposes ``.single()`` (upsert/delete). Empty by default.
    result = MagicMock()
    result.__iter__.side_effect = lambda: iter([])
    session.run.return_value = result
    backend._driver = driver
    return backend, session


def _cypher_and_params(session: MagicMock, call: int = 0) -> tuple[str, dict[str, Any]]:
    args, kwargs = session.run.call_args_list[call]
    cypher = args[0]
    params = dict(args[1]) if len(args) > 1 else dict(kwargs)
    return cypher, params


# --- constructor --------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "4"])
def test_over_fetch_factor_validated(bad: Any) -> None:
    with pytest.raises(ValueError, match="over_fetch_factor"):
        Neo4jBackend(Neo4jConfig(), over_fetch_factor=bad)


# --- vector_search ------------------------------------------------------------


def test_vector_search_unfiltered_shape() -> None:
    backend, session = _mocked_backend()
    backend.vector_search("File", "embedding", [0.1, 0.2], k=7)
    cypher, params = _cypher_and_params(session)
    assert cypher.startswith("CALL db.index.vector.queryNodes($idx, $k, $q) YIELD node, score ")
    assert "WHERE" not in cypher
    assert cypher.endswith("RETURN node, score ORDER BY score DESC LIMIT $limit")
    assert params["idx"] == "File_embedding_idx"
    assert params["k"] == 7  # no over-fetch without filters
    assert params["limit"] == 7
    assert params["q"] == [0.1, 0.2]


def test_vector_search_filters_render_as_where_and_over_fetch() -> None:
    backend, session = _mocked_backend(over_fetch_factor=4)
    backend.vector_search(
        "Chunk", "embedding", [0.1], k=10, filters={"corpus": "docs", "profile": "fine"}
    )
    cypher, params = _cypher_and_params(session)
    assert (
        "YIELD node, score WITH node, score "
        "WHERE node.corpus = $f_corpus AND node.profile = $f_profile "
        "RETURN node, score ORDER BY score DESC LIMIT $limit"
    ) in cypher
    assert params["k"] == 40  # k * over_fetch_factor requested from the index
    assert params["limit"] == 10  # caller's k is the trailing LIMIT
    assert params["f_corpus"] == "docs"
    assert params["f_profile"] == "fine"
    # Values are bound, never interpolated.
    assert "docs" not in cypher and "fine" not in cypher


def test_vector_search_over_fetch_capped_at_1000() -> None:
    backend, session = _mocked_backend(over_fetch_factor=4)
    backend.vector_search("File", "embedding", [0.1], k=600, filters={"corpus": "c"})
    _, params = _cypher_and_params(session)
    assert params["k"] == 1000
    assert params["limit"] == 600


def test_vector_search_empty_filters_is_unfiltered() -> None:
    backend, session = _mocked_backend()
    backend.vector_search("File", "embedding", [0.1], k=3, filters={})
    cypher, params = _cypher_and_params(session)
    assert "WHERE" not in cypher
    assert params["k"] == 3


def test_vector_search_rejects_unsafe_filter_key() -> None:
    backend, session = _mocked_backend()
    with pytest.raises(ValueError, match="property_name"):
        backend.vector_search("File", "embedding", [0.1], k=3, filters={"corpus = 1 OR 1": "x"})
    session.run.assert_not_called()


def test_vector_search_applies_threshold_after_filter() -> None:
    backend, session = _mocked_backend()
    session.run.return_value = [
        {"node": {"path": "/a"}, "score": 0.95},
        {"node": {"path": "/b"}, "score": 0.40},
    ]
    rows = backend.vector_search(
        "File", "embedding", [0.1], k=5, threshold=0.9, filters={"corpus": "c"}
    )
    assert [r["node"]["path"] for r in rows] == ["/a"]


# --- full_text_search ---------------------------------------------------------


def test_full_text_search_unfiltered_shape() -> None:
    backend, session = _mocked_backend()
    backend.full_text_search("Section", "summary", "retry", k=5)
    cypher, params = _cypher_and_params(session)
    assert cypher.startswith(
        "CALL db.index.fulltext.queryNodes($idx, $q, {limit: $k}) YIELD node, score "
    )
    assert "WHERE" not in cypher
    assert cypher.endswith("RETURN node, score ORDER BY score DESC LIMIT $limit")
    assert params["idx"] == "Section_summary_ft"
    assert params["k"] == 5
    assert params["limit"] == 5
    assert params["q"] == "retry"


def test_full_text_search_filters_render_as_where_and_over_fetch() -> None:
    backend, session = _mocked_backend(over_fetch_factor=3)
    backend.full_text_search("Chunk", "text", "retry", k=10, filters={"corpus": "docs"})
    cypher, params = _cypher_and_params(session)
    assert (
        "YIELD node, score WITH node, score WHERE node.corpus = $f_corpus "
        "RETURN node, score ORDER BY score DESC LIMIT $limit"
    ) in cypher
    assert params["idx"] == "Chunk_text_ft"  # multi-property index, lead-property name
    assert params["k"] == 30
    assert params["limit"] == 10
    assert params["f_corpus"] == "docs"


def test_full_text_search_rejects_unsafe_filter_key() -> None:
    backend, session = _mocked_backend()
    with pytest.raises(ValueError, match="property_name"):
        backend.full_text_search("File", "summary", "x", k=3, filters={"a b": "x"})
    session.run.assert_not_called()


# --- upsert_nodes -------------------------------------------------------------


def test_upsert_nodes_empty_returns_zero_without_touching_store() -> None:
    backend, session = _mocked_backend()
    assert backend.upsert_nodes("Chunk", []) == 0
    session.run.assert_not_called()


def test_upsert_nodes_missing_pk_names_key_and_row_index() -> None:
    backend, session = _mocked_backend()
    rows = [{"id": "a~fine~0", "text": "x"}, {"text": "no id"}]
    with pytest.raises(ValueError, match=r"row 1 missing required primary key 'id'"):
        backend.upsert_nodes("Chunk", rows)
    session.run.assert_not_called()  # validated before any write


def test_upsert_nodes_rejects_unknown_label() -> None:
    backend, _ = _mocked_backend()
    with pytest.raises(ValueError, match="Unknown node label"):
        backend.upsert_nodes("Wormhole", [{"id": "x"}])


def test_upsert_nodes_unwind_merge_shape() -> None:
    backend, session = _mocked_backend()
    session.run.return_value.single.return_value = {"c": 2}
    rows = [{"id": "a~fine~0", "text": "x"}, {"id": "a~fine~1", "text": "y"}]
    assert backend.upsert_nodes("Chunk", rows) == 2
    args, kwargs = session.run.call_args
    assert args[0] == (
        "UNWIND $rows AS r MERGE (n:Chunk {id: r.id}) SET n += r RETURN count(n) AS c"
    )
    assert kwargs["rows"] == rows


def test_upsert_nodes_uses_label_primary_key() -> None:
    backend, session = _mocked_backend()
    session.run.return_value.single.return_value = {"c": 1}
    backend.upsert_nodes("File", [{"path": "/a.md"}])
    assert "MERGE (n:File {path: r.path})" in session.run.call_args[0][0]


def test_upsert_nodes_batches_in_chunks_of_500_in_one_session() -> None:
    backend, session = _mocked_backend()
    session.run.return_value.single.side_effect = [{"c": 500}, {"c": 500}, {"c": 200}]
    rows = [{"id": f"p~fine~{i}"} for i in range(1200)]
    assert backend.upsert_nodes("Chunk", rows) == 1200
    assert session.run.call_count == 3
    sizes = [len(c.kwargs["rows"]) for c in session.run.call_args_list]
    assert sizes == [500, 500, 200]
    backend._driver.session.assert_called_once()  # type: ignore[union-attr]


# --- delete_nodes -------------------------------------------------------------


def test_delete_nodes_requires_where() -> None:
    backend, session = _mocked_backend()
    with pytest.raises(ValueError, match="non-empty where"):
        backend.delete_nodes("Chunk", where={})
    session.run.assert_not_called()


def test_delete_nodes_rejects_unsafe_key() -> None:
    backend, session = _mocked_backend()
    with pytest.raises(ValueError, match="property_name"):
        backend.delete_nodes("Chunk", where={"corpus) DETACH DELETE (n": "x"})
    session.run.assert_not_called()


def test_delete_nodes_shape_and_count() -> None:
    backend, session = _mocked_backend()
    session.run.return_value.single.return_value = {"c": 12}
    n = backend.delete_nodes("Chunk", where={"corpus": "docs", "parent_id": "a.md#s"})
    assert n == 12
    cypher, params = _cypher_and_params(session)
    assert cypher == (
        "MATCH (n:Chunk) WHERE n.corpus = $w_corpus AND n.parent_id = $w_parent_id "
        "WITH collect(n) AS ns UNWIND ns AS n DETACH DELETE n RETURN count(*) AS c"
    )
    assert params == {"w_corpus": "docs", "w_parent_id": "a.md#s"}
