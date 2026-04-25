"""Unit tests for OpenAICompatProvider — local-model inference path.

Mirrors the test_providers_gemini.py pattern: hand-rolled MagicMock for
the HTTP client so tests stay hermetic and don't require a running
local server.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from contextd.config import OpenAICompatConfig
from contextd.providers.base import PromptRequest
from contextd.providers.openai_compat import OpenAICompatProvider


@pytest.fixture
def cfg() -> OpenAICompatConfig:
    return OpenAICompatConfig(
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        model_summary="qwen2.5:7b-instruct",
        model_inference="qwen2.5:14b-instruct",
        model_translation="qwen2.5:14b-instruct",
        max_retries=3,
        request_timeout_seconds=30.0,
        json_mode=True,
    )


def _mock_client(
    body: dict[str, Any] | None = None,
    *,
    status_code: int = 200,
    side_effect: list[Any] | None = None,
) -> MagicMock:
    """Return a MagicMock standing in for httpx.Client.

    When ``side_effect`` is given, ``client.post`` returns each entry in
    order (raises if entry is an exception); otherwise a single
    response with ``status_code`` and ``body`` is returned.
    """
    client = MagicMock()
    if side_effect is not None:
        client.post.side_effect = side_effect
        return client
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body or _default_body()
    response.raise_for_status = MagicMock()
    client.post.return_value = response
    return client


def _default_body() -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }


def _http_response(status: int, body: dict[str, Any] | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body or _default_body()
    if status >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status = MagicMock()
    return response


def test_generate_posts_chat_completions_with_messages(cfg: OpenAICompatConfig) -> None:
    client = _mock_client()
    provider = OpenAICompatProvider(cfg, client=client)
    provider.generate(PromptRequest(system="sys", prompt="user", call_site="summary"))
    args, kwargs = client.post.call_args
    assert args[0] == "http://localhost:11434/v1/chat/completions"
    body = kwargs["json"]
    assert body["model"] == "qwen2.5:7b-instruct"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


def test_generate_uses_call_site_specific_model(cfg: OpenAICompatConfig) -> None:
    client = _mock_client()
    provider = OpenAICompatProvider(cfg, client=client)
    provider.generate(PromptRequest(system="s", prompt="p", call_site="inference"))
    assert client.post.call_args.kwargs["json"]["model"] == "qwen2.5:14b-instruct"


def test_generate_sends_authorization_header_when_api_key_set(cfg: OpenAICompatConfig) -> None:
    client = _mock_client()
    provider = OpenAICompatProvider(cfg, api_key="sk-abc", client=client)
    provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    headers = client.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer sk-abc"


def test_generate_omits_authorization_when_api_key_none(cfg: OpenAICompatConfig) -> None:
    client = _mock_client()
    provider = OpenAICompatProvider(cfg, api_key=None, client=client)
    provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    headers = client.post.call_args.kwargs["headers"]
    assert "Authorization" not in headers


def test_generate_includes_response_format_for_json_call_sites(
    cfg: OpenAICompatConfig,
) -> None:
    client = _mock_client()
    provider = OpenAICompatProvider(cfg, client=client)
    for site in ("summary", "inference"):
        client.post.reset_mock()
        provider.generate(PromptRequest(system="s", prompt="p", call_site=site))  # type: ignore[arg-type]
        assert client.post.call_args.kwargs["json"]["response_format"] == {"type": "json_object"}


def test_generate_omits_response_format_for_translation_call_site(
    cfg: OpenAICompatConfig,
) -> None:
    client = _mock_client()
    provider = OpenAICompatProvider(cfg, client=client)
    provider.generate(PromptRequest(system="s", prompt="p", call_site="translation"))
    assert "response_format" not in client.post.call_args.kwargs["json"]


def test_generate_retries_on_500_then_succeeds(cfg: OpenAICompatConfig) -> None:
    client = _mock_client(
        side_effect=[_http_response(500), _http_response(200)],
    )
    provider = OpenAICompatProvider(cfg, client=client, backoff_initial=0.0, backoff_max=0.0)
    out = provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    assert out == "ok"
    assert client.post.call_count == 2


def test_generate_gives_up_after_max_retries(cfg: OpenAICompatConfig) -> None:
    client = _mock_client(
        side_effect=[_http_response(500), _http_response(500), _http_response(500)],
    )
    provider = OpenAICompatProvider(cfg, client=client, backoff_initial=0.0, backoff_max=0.0)
    with pytest.raises(httpx.HTTPStatusError):
        provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    assert client.post.call_count == cfg.max_retries


def test_generate_records_usage_from_response(cfg: OpenAICompatConfig) -> None:
    client = _mock_client(
        body={
            "choices": [{"message": {"content": "out"}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 9},
        }
    )
    provider = OpenAICompatProvider(cfg, client=client)
    provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    usage = provider.last_usage()
    assert usage is not None
    assert usage.provider == "openai_compat"
    assert usage.input_tokens == 42
    assert usage.output_tokens == 9
    assert usage.call_site == "summary"


def test_generate_records_zero_usage_when_response_omits_it(cfg: OpenAICompatConfig) -> None:
    client = _mock_client(body={"choices": [{"message": {"content": "out"}}]})
    provider = OpenAICompatProvider(cfg, client=client)
    provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    usage = provider.last_usage()
    assert usage is not None
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_generate_strips_trailing_slash_from_base_url(cfg: OpenAICompatConfig) -> None:
    cfg = cfg.model_copy(update={"base_url": "http://localhost:11434/v1/"})
    client = _mock_client()
    provider = OpenAICompatProvider(cfg, client=client)
    provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    assert client.post.call_args.args[0] == "http://localhost:11434/v1/chat/completions"
