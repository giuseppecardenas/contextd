import json
from pathlib import Path
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


# ---------------------------------------------------------------------------
# prompt_path override tests
# ---------------------------------------------------------------------------


def test_summariser_uses_default_template_when_no_override() -> None:
    """With prompt_path=None the renderer's .render() method is called (not render_path)."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {"summary": "s", "key_points": [], "entities_mentioned": []}
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "default-prompt"
    summariser = Summariser(provider=mock_provider, renderer=mock_renderer, max_words=50)
    summariser.summarise("body text")
    mock_renderer.render.assert_called_once_with("summarise", content="body text", max_words="50")
    mock_renderer.render_path.assert_not_called()


def test_summariser_uses_override_when_prompt_path_set(tmp_path: Path) -> None:
    """With prompt_path set, render_path() is called instead of render()."""
    override = tmp_path / "override.md"
    override.write_text("Summarise: {{content}}", encoding="utf-8")

    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {"summary": "x", "key_points": [], "entities_mentioned": []}
    )
    mock_renderer = MagicMock()
    mock_renderer.render_path.return_value = "override-prompt"
    summariser = Summariser(
        provider=mock_provider, renderer=mock_renderer, max_words=75, prompt_path=override
    )
    summariser.summarise("some content")
    mock_renderer.render_path.assert_called_once_with(
        override, content="some content", max_words="75"
    )
    mock_renderer.render.assert_not_called()


def test_summariser_override_template_substitutes_content(tmp_path: Path) -> None:
    """Override template with {{content}} receives the actual content value."""
    override = tmp_path / "tpl.md"
    override.write_text("Please summarise: {{content}}", encoding="utf-8")

    from contextd.inference.prompts import PromptRenderer

    real_renderer = PromptRenderer(tmp_path)  # template_dir unused for render_path
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {"summary": "ok", "key_points": [], "entities_mentioned": []}
    )
    summariser = Summariser(
        provider=mock_provider, renderer=real_renderer, max_words=100, prompt_path=override
    )
    summariser.summarise("my body text")
    sent_prompt = mock_provider.generate.call_args[0][0].prompt
    assert "my body text" in sent_prompt


def test_summariser_override_template_substitutes_max_words(tmp_path: Path) -> None:
    """Override template with {{max_words}} receives the max_words value."""
    override = tmp_path / "tpl.md"
    override.write_text("Limit: {{max_words}} words. Body: {{content}}", encoding="utf-8")

    from contextd.inference.prompts import PromptRenderer

    real_renderer = PromptRenderer(tmp_path)
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {"summary": "ok", "key_points": [], "entities_mentioned": []}
    )
    summariser = Summariser(
        provider=mock_provider, renderer=real_renderer, max_words=42, prompt_path=override
    )
    summariser.summarise("content")
    sent_prompt = mock_provider.generate.call_args[0][0].prompt
    assert "42" in sent_prompt
    assert "content" in sent_prompt


def test_summariser_override_template_without_max_words_placeholder_works(tmp_path: Path) -> None:
    """A template that only references {{content}} (no {{max_words}}) is valid.

    Extra kwargs passed to render_path that have no matching placeholder are
    silently ignored — the template contract allows this.
    """
    override = tmp_path / "simple.md"
    override.write_text("Just: {{content}}", encoding="utf-8")

    from contextd.inference.prompts import PromptRenderer

    real_renderer = PromptRenderer(tmp_path)
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {"summary": "done", "key_points": [], "entities_mentioned": []}
    )
    summariser = Summariser(
        provider=mock_provider, renderer=real_renderer, max_words=100, prompt_path=override
    )
    result = summariser.summarise("hello world")
    assert result.summary == "done"
    sent_prompt = mock_provider.generate.call_args[0][0].prompt
    assert "hello world" in sent_prompt
