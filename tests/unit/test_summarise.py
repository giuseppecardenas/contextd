import json
from unittest.mock import MagicMock

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
