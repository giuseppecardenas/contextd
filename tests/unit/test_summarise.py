import json
from unittest.mock import MagicMock

import pytest

from contextd.inference.summarise import FileSummary, Summariser


def test_produces_structured_summary() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "summary": "A file about X.",
            "key_points": ["point 1", "point 2"],
            "entities_mentioned": ["entity1", "entity2"],
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered-prompt"
    summariser = Summariser(provider=mock_provider, renderer=mock_renderer, max_words=100)
    result = summariser.summarise("file content here")
    assert isinstance(result, FileSummary)
    assert result.summary == "A file about X."
    assert result.key_points == ["point 1", "point 2"]
    assert result.entities_mentioned == ["entity1", "entity2"]


def test_forwards_max_words_to_renderer() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "summary": "x",
            "key_points": [],
            "entities_mentioned": [],
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    summariser = Summariser(provider=mock_provider, renderer=mock_renderer, max_words=200)
    summariser.summarise("content")
    mock_renderer.render.assert_called_once()
    call_kwargs = mock_renderer.render.call_args.kwargs
    assert call_kwargs["max_words"] == "200"
    assert call_kwargs["content"] == "content"


def test_handles_code_fences_in_provider_response() -> None:
    """The LLM sometimes wraps JSON in ```json fences — strip them."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = """```json
{"summary": "x", "key_points": [], "entities_mentioned": []}
```"""
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered-prompt"
    summariser = Summariser(provider=mock_provider, renderer=mock_renderer)
    result = summariser.summarise("content")
    assert result.summary == "x"


def test_handles_yaml_language_tagged_fence() -> None:
    """Some LLMs emit non-json language tags; extractor should still work."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = """```yaml
{"summary": "yaml-tagged", "key_points": [], "entities_mentioned": []}
```"""
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered-prompt"
    summariser = Summariser(provider=mock_provider, renderer=mock_renderer)
    result = summariser.summarise("content")
    assert result.summary == "yaml-tagged"


def test_handles_prose_wrapper_around_json() -> None:
    """Provider occasionally prefixes JSON with prose despite instruction."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = (
        "Here is the JSON you requested:\n\n"
        '{"summary": "prose-wrapped", "key_points": [], "entities_mentioned": []}\n\n'
        "I hope this helps!"
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered-prompt"
    summariser = Summariser(provider=mock_provider, renderer=mock_renderer)
    result = summariser.summarise("content")
    assert result.summary == "prose-wrapped"


def test_non_list_optional_field_falls_back_to_empty() -> None:
    """If provider returns a string where a list is expected, drop it
    silently rather than storing chars-as-list under the typed field."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "summary": "x",
            "key_points": "not a list at all",
            "entities_mentioned": None,
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered-prompt"
    summariser = Summariser(provider=mock_provider, renderer=mock_renderer)
    result = summariser.summarise("content")
    assert result.key_points == []
    assert result.entities_mentioned == []


def test_no_json_object_raises_valueerror() -> None:
    """If the response contains no { or }, extractor surfaces a clear error."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "I cannot summarise this file."
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered-prompt"
    summariser = Summariser(provider=mock_provider, renderer=mock_renderer)
    with pytest.raises(ValueError, match="no JSON object"):
        summariser.summarise("content")


def test_missing_summary_key_includes_available_keys() -> None:
    """Actionable error message instead of a bare KeyError — the user wants
    to know what the provider did return, not just what's missing."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps({"text": "some other key", "key_points": []})
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered-prompt"
    summariser = Summariser(provider=mock_provider, renderer=mock_renderer)
    with pytest.raises(KeyError, match="got keys"):
        summariser.summarise("content")


def test_non_string_summary_raises_typeerror() -> None:
    """A provider returning `summary: 42` previously flowed through via cast
    and failed far from the source. Now it raises at the parse site."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {"summary": 42, "key_points": [], "entities_mentioned": []}
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered-prompt"
    summariser = Summariser(provider=mock_provider, renderer=mock_renderer)
    with pytest.raises(TypeError, match="'summary' must be a string"):
        summariser.summarise("content")
