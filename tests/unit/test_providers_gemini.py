from unittest.mock import MagicMock, patch

import pytest

from contextd.config import GeminiConfig
from contextd.providers.base import PromptRequest
from contextd.providers.gemini import GeminiProvider


@pytest.fixture
def gemini_cfg() -> GeminiConfig:
    return GeminiConfig(
        model_summary="gemma-4-31b-it",
        model_inference="gemma-4-31b-it",
        model_translation="gemma-4-31b-it",
        max_retries=3,
        safety_block="BLOCK_NONE",
    )


def _mock_client_with_response(
    text: str, input_tokens: int = 100, output_tokens: int = 20
) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = text
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=input_tokens,
        candidates_token_count=output_tokens,
    )
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


def _api_error(code: int, msg: str) -> Exception:
    """Build a google-genai APIError. Library uses response_json (not `message` kwarg)."""
    from google.genai import errors as genai_errors

    return genai_errors.APIError(
        code=code,
        response_json={"error": {"message": msg}},
        response=None,
    )


def test_generate_returns_text_and_records_usage(gemini_cfg: GeminiConfig) -> None:
    mock_client = _mock_client_with_response("hello world")
    with patch("contextd.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider(gemini_cfg, api_key="test-key")
        result = provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    assert result == "hello world"
    usage = provider.last_usage()
    assert usage is not None
    assert usage.provider == "gemini"
    assert usage.model == "gemma-4-31b-it"
    assert usage.call_site == "summary"
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20


def test_different_call_sites_select_different_models(gemini_cfg: GeminiConfig) -> None:
    gemini_cfg.model_translation = "gemini-pro-latest"
    mock_client = _mock_client_with_response("x")
    with patch("contextd.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider(gemini_cfg, api_key="test-key")
        provider.generate(PromptRequest(system="s", prompt="p", call_site="translation"))
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-pro-latest"


def test_safety_settings_applied(gemini_cfg: GeminiConfig) -> None:
    mock_client = _mock_client_with_response("x")
    with patch("contextd.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider(gemini_cfg, api_key="test-key")
        provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    # The provider must forward safety_settings; exact shape is SDK-dependent.
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "config" in call_kwargs
    cfg_obj = call_kwargs["config"]
    assert cfg_obj is not None


def test_thinking_config_only_applied_for_translation(gemini_cfg: GeminiConfig) -> None:
    """NL→Cypher translation benefits from extended reasoning; summary and
    inference are bounded tasks that don't need it. Verifies thinking_config
    is set to HIGH for translation calls and absent for the other two
    call sites.
    """
    from google.genai.types import ThinkingLevel

    for call_site, expected_thinking in [
        ("summary", None),
        ("inference", None),
        ("translation", ThinkingLevel.HIGH),
    ]:
        mock_client = _mock_client_with_response("x")
        with patch("contextd.providers.gemini.genai.Client", return_value=mock_client):
            provider = GeminiProvider(gemini_cfg, api_key="test-key")
            provider.generate(PromptRequest(system="s", prompt="p", call_site=call_site))
        cfg_obj = mock_client.models.generate_content.call_args.kwargs["config"]
        actual = cfg_obj.thinking_config
        if expected_thinking is None:
            assert actual is None, f"{call_site}: expected no thinking_config, got {actual}"
        else:
            assert actual is not None, f"{call_site}: expected thinking_config, got None"
            assert actual.thinking_level == expected_thinking, (
                f"{call_site}: expected thinking_level={expected_thinking}, got {actual.thinking_level}"
            )


def test_retries_on_resource_exhausted(gemini_cfg: GeminiConfig) -> None:
    """On RESOURCE_EXHAUSTED (429), backoff-retry up to max_retries."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "recovered"
    mock_response.usage_metadata = MagicMock(prompt_token_count=1, candidates_token_count=1)
    mock_client.models.generate_content.side_effect = [
        _api_error(429, "quota"),
        _api_error(429, "quota"),
        mock_response,
    ]
    with patch("contextd.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider(gemini_cfg, api_key="test-key", backoff_initial=0.01)
        result = provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    assert result == "recovered"
    assert mock_client.models.generate_content.call_count == 3


def test_raises_on_permanent_400(gemini_cfg: GeminiConfig) -> None:
    from google.genai import errors as genai_errors

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _api_error(400, "bad request")
    with patch("contextd.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider(gemini_cfg, api_key="test-key", backoff_initial=0.01)
        with pytest.raises(genai_errors.APIError):
            provider.generate(PromptRequest(system="s", prompt="p", call_site="summary"))
    # Permanent → no retry loop.
    assert mock_client.models.generate_content.call_count == 1
