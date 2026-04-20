from unittest.mock import MagicMock

from contextd.indexer.entity_resolver import EntityResolver


def test_finds_similar_existing_node_above_threshold() -> None:
    mock_store = MagicMock()
    mock_store.vector_search.return_value = [{"node": {"name": "React"}, "score": 0.95}]
    mock_embed = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    resolver = EntityResolver(store=mock_store, embedder=mock_embed, threshold=0.92)
    resolved = resolver.resolve("Technology", "ReactJS")
    assert resolved == "React"


def test_returns_none_when_no_similar_match() -> None:
    mock_store = MagicMock()
    mock_store.vector_search.return_value = []
    mock_embed = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    resolver = EntityResolver(store=mock_store, embedder=mock_embed, threshold=0.92)
    assert resolver.resolve("Technology", "CompletelyNovelThing") is None


def test_below_threshold_returns_none() -> None:
    mock_store = MagicMock()
    mock_store.vector_search.return_value = [{"node": {"name": "React"}, "score": 0.5}]
    mock_embed = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    resolver = EntityResolver(store=mock_store, embedder=mock_embed, threshold=0.92)
    assert resolver.resolve("Technology", "ReactJS") is None
