"""Unit tests for OpenAICompatEmbeddingProvider — local embedding path.

Mirrors test_providers_openai_compat.py: a hand-rolled MagicMock stands in
for the httpx client so tests stay hermetic and never touch a real server.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from contextd.config import OpenAICompatEmbeddingConfig
from contextd.providers.openai_compat_embedding import OpenAICompatEmbeddingProvider


@pytest.fixture
def cfg() -> OpenAICompatEmbeddingConfig:
    return OpenAICompatEmbeddingConfig(
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        model="mxbai-embed-large",
        dimensions=4,
        max_batch_size=2,
        max_retries=3,
        request_timeout_seconds=30.0,
    )


def _vec(fill: float, dims: int = 4) -> list[float]:
    return [fill] * dims


def _embeddings_body(vectors: list[list[float]], *, prompt_tokens: int = 0) -> dict[str, Any]:
    return {
        "data": [
            {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)
        ],
        "model": "mxbai-embed-large",
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


def _mock_client(
    *,
    body: dict[str, Any] | None = None,
    side_effect: list[Any] | None = None,
) -> MagicMock:
    client = MagicMock()
    if side_effect is not None:
        client.post.side_effect = side_effect
        return client
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = body or _embeddings_body([_vec(0.1), _vec(0.2)])
    response.raise_for_status = MagicMock()
    client.post.return_value = response
    return client


def _http_response(status: int, body: dict[str, Any] | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body or _embeddings_body([_vec(0.1)])
    if status >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status = MagicMock()
    return response


def test_embed_posts_to_embeddings_endpoint(cfg: OpenAICompatEmbeddingConfig) -> None:
    client = _mock_client()
    provider = OpenAICompatEmbeddingProvider(cfg, client=client)
    provider.embed(["a", "b"])
    args, kwargs = client.post.call_args
    assert args[0] == "http://localhost:11434/v1/embeddings"
    assert kwargs["json"]["model"] == "mxbai-embed-large"
    assert kwargs["json"]["input"] == ["a", "b"]


def test_embed_returns_one_vector_per_input(cfg: OpenAICompatEmbeddingConfig) -> None:
    client = _mock_client(body=_embeddings_body([_vec(0.1), _vec(0.2)]))
    provider = OpenAICompatEmbeddingProvider(cfg, client=client)
    out = provider.embed(["a", "b"])
    assert out == [_vec(0.1), _vec(0.2)]


def test_embed_batches_by_max_batch_size(cfg: OpenAICompatEmbeddingConfig) -> None:
    client = _mock_client(
        side_effect=[
            _http_response(200, _embeddings_body([_vec(0.1), _vec(0.2)])),
            _http_response(200, _embeddings_body([_vec(0.3)])),
        ]
    )
    provider = OpenAICompatEmbeddingProvider(cfg, client=client)
    out = provider.embed(["a", "b", "c"])
    assert client.post.call_count == 2
    assert out == [_vec(0.1), _vec(0.2), _vec(0.3)]


def test_embed_reorders_by_index_field(cfg: OpenAICompatEmbeddingConfig) -> None:
    out_of_order = {
        "data": [
            {"index": 1, "embedding": _vec(0.2)},
            {"index": 0, "embedding": _vec(0.1)},
        ],
        "usage": {"prompt_tokens": 0},
    }
    client = _mock_client(body=out_of_order)
    provider = OpenAICompatEmbeddingProvider(cfg, client=client)
    out = provider.embed(["a", "b"])
    assert out == [_vec(0.1), _vec(0.2)]


def test_embed_substitutes_blank_input(cfg: OpenAICompatEmbeddingConfig) -> None:
    client = _mock_client(body=_embeddings_body([_vec(0.1)]))
    provider = OpenAICompatEmbeddingProvider(cfg, client=client)
    provider.embed([""])
    assert client.post.call_args.kwargs["json"]["input"] == [" "]


def test_embed_sends_authorization_header_when_api_key_set(
    cfg: OpenAICompatEmbeddingConfig,
) -> None:
    client = _mock_client()
    provider = OpenAICompatEmbeddingProvider(cfg, api_key="sk-abc", client=client)
    provider.embed(["a"])
    assert client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-abc"


def test_embed_omits_authorization_when_api_key_none(
    cfg: OpenAICompatEmbeddingConfig,
) -> None:
    client = _mock_client()
    provider = OpenAICompatEmbeddingProvider(cfg, api_key=None, client=client)
    provider.embed(["a"])
    assert "Authorization" not in client.post.call_args.kwargs["headers"]


def test_embed_retries_on_500_then_succeeds(cfg: OpenAICompatEmbeddingConfig) -> None:
    client = _mock_client(
        side_effect=[
            _http_response(500),
            _http_response(200, _embeddings_body([_vec(0.1), _vec(0.2)])),
        ]
    )
    provider = OpenAICompatEmbeddingProvider(
        cfg, client=client, backoff_initial=0.0, backoff_max=0.0
    )
    out = provider.embed(["a", "b"])
    assert client.post.call_count == 2
    assert out == [_vec(0.1), _vec(0.2)]


def test_embed_gives_up_after_max_retries(cfg: OpenAICompatEmbeddingConfig) -> None:
    client = _mock_client(
        side_effect=[_http_response(500), _http_response(500), _http_response(500)]
    )
    provider = OpenAICompatEmbeddingProvider(
        cfg, client=client, backoff_initial=0.0, backoff_max=0.0
    )
    with pytest.raises(httpx.HTTPStatusError):
        provider.embed(["a"])
    assert client.post.call_count == cfg.max_retries


def test_embed_raises_on_dimension_mismatch(cfg: OpenAICompatEmbeddingConfig) -> None:
    client = _mock_client(body=_embeddings_body([_vec(0.1, dims=3)]))
    provider = OpenAICompatEmbeddingProvider(cfg, client=client)
    with pytest.raises(ValueError, match="dimension"):
        provider.embed(["a"])


def test_dimensions_property_reflects_config(cfg: OpenAICompatEmbeddingConfig) -> None:
    provider = OpenAICompatEmbeddingProvider(cfg, client=_mock_client())
    assert provider.dimensions == 4


def test_last_usage_records_embedding_call_site(cfg: OpenAICompatEmbeddingConfig) -> None:
    client = _mock_client(body=_embeddings_body([_vec(0.1)], prompt_tokens=17))
    provider = OpenAICompatEmbeddingProvider(cfg, client=client)
    provider.embed(["a"])
    usage = provider.last_usage()
    assert usage is not None
    assert usage.provider == "openai_compat"
    assert usage.model == "mxbai-embed-large"
    assert usage.call_site == "embedding"
    assert usage.input_tokens == 17
    assert usage.output_tokens == 0


def test_last_usage_accumulates_tokens_across_batches(
    cfg: OpenAICompatEmbeddingConfig,
) -> None:
    client = _mock_client(
        side_effect=[
            _http_response(200, _embeddings_body([_vec(0.1), _vec(0.2)], prompt_tokens=10)),
            _http_response(200, _embeddings_body([_vec(0.3)], prompt_tokens=5)),
        ]
    )
    provider = OpenAICompatEmbeddingProvider(cfg, client=client)
    provider.embed(["a", "b", "c"])
    usage = provider.last_usage()
    assert usage is not None
    assert usage.input_tokens == 15


def test_embed_empty_list_returns_empty(cfg: OpenAICompatEmbeddingConfig) -> None:
    client = _mock_client()
    provider = OpenAICompatEmbeddingProvider(cfg, client=client)
    assert provider.embed([]) == []
    client.post.assert_not_called()


def test_embed_strips_trailing_slash_from_base_url(cfg: OpenAICompatEmbeddingConfig) -> None:
    cfg = cfg.model_copy(update={"base_url": "http://localhost:11434/v1/"})
    client = _mock_client()
    provider = OpenAICompatEmbeddingProvider(cfg, client=client)
    provider.embed(["a"])
    assert client.post.call_args.args[0] == "http://localhost:11434/v1/embeddings"
