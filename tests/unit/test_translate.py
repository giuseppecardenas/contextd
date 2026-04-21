from unittest.mock import MagicMock

import pytest

from contextd.inference.translate import QueryTranslator
from contextd.ontology.schema import Ontology


def test_returns_cypher_from_provider() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "MATCH (n:File) RETURN n LIMIT 10"
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    result = translator.translate("show me files")
    assert "MATCH" in result
    assert "LIMIT" in result


def test_strips_fences_and_prose() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = """Sure! Here's the query:
```cypher
MATCH (n:File) RETURN n LIMIT 5
```
That returns five files."""
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    result = translator.translate("files")
    assert result.startswith("MATCH")
    assert "Sure!" not in result


def test_raises_on_empty_or_unrecognised_response() -> None:
    """Empty or prose-only provider output must raise rather than yielding
    an empty Cypher string, which would hit the backend as a syntax error
    with no user-facing context."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "I couldn't translate that question."
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    with pytest.raises(ValueError, match="no Cypher-like content"):
        translator.translate("something unanswerable")


def test_handles_non_cypher_language_tag_fence() -> None:
    """Some LLMs emit ```sql or ```gremlin instead of ```cypher —
    the body is still Cypher; the fence extractor should take it."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = """Here's your query:
```sql
MATCH (n:File) RETURN n LIMIT 7
```
"""
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    result = translator.translate("files")
    assert result.startswith("MATCH")
    assert "LIMIT 7" in result


def test_preserves_multiline_continuation_in_fallback() -> None:
    """When the LLM forgets the fence and spreads a query across lines
    including continuation lines that don't start with a keyword, the
    keyword-line fallback used to drop the continuations. Now it slices
    from the first keyword-line to end-of-text."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = (
        "Sure, here's the query:\nMATCH (n:File)\n-[:REFERENCES]->(m)\nRETURN n, m\nLIMIT 5"
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    result = translator.translate("files and their references")
    assert "MATCH (n:File)" in result
    # The previously-dropped continuation line must survive.
    assert "[:REFERENCES]->(m)" in result
    assert "Sure" not in result
