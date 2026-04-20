from unittest.mock import MagicMock, patch

from contextd.config import VoyageConfig
from contextd.providers.voyage import VoyageProvider


def test_embed_returns_vectors() -> None:
    cfg = VoyageConfig(model="voyage-3", max_batch_size=128)
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
    mock_result.total_tokens = 50
    mock_client.embed.return_value = mock_result
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key")
        result = provider.embed(["hello", "world"])
    assert len(result) == 2
    assert all(len(v) == 1024 for v in result)
    assert provider.dimensions == 1024


def test_batches_respect_max_batch_size() -> None:
    cfg = VoyageConfig(model="voyage-3", max_batch_size=2)
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
    mock_result.total_tokens = 50
    mock_client.embed.return_value = mock_result
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key")
        provider.embed(["a", "b", "c", "d"])
    # 4 inputs, batch size 2 → 2 calls.
    assert mock_client.embed.call_count == 2


def test_retries_on_rate_limit() -> None:
    cfg = VoyageConfig(model="voyage-3", max_batch_size=128)
    from voyageai.error import RateLimitError

    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_result.total_tokens = 1
    mock_client.embed.side_effect = [RateLimitError("limit"), mock_result]
    with patch("contextd.providers.voyage.voyageai.Client", return_value=mock_client):
        provider = VoyageProvider(cfg, api_key="test-key", backoff_initial=0.01)
        result = provider.embed(["x"])
    assert len(result) == 1
    assert mock_client.embed.call_count == 2
