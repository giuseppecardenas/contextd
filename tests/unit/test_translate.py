import logging
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


def test_injects_corpus_filter_into_labelled_node() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "MATCH (n:File) RETURN n LIMIT 5"
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    result = translator.translate("files", corpus="runeledger-prd")
    assert '{corpus: "runeledger-prd"}' in result
    assert result.startswith("MATCH (n:File {corpus:")


def test_merges_corpus_into_existing_property_map() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "MATCH (n:File {path: 'a.md'}) RETURN n"
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    result = translator.translate("file a", corpus="x")
    assert "path: 'a.md'" in result
    assert 'corpus: "x"' in result
    # Single property map, not two.
    assert result.count("{") == 1
    assert result.count("}") == 1


def test_corpus_none_passes_through_untouched() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "MATCH (n:File) RETURN n"
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    result = translator.translate("files")  # corpus not passed
    assert result == "MATCH (n:File) RETURN n"


def test_corpus_empty_string_passes_through_untouched() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "MATCH (n:File) RETURN n"
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    result = translator.translate("files", corpus="")
    assert result == "MATCH (n:File) RETURN n"


def test_raises_on_unsafe_corpus_name() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "MATCH (n:File) RETURN n"
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    with pytest.raises(ValueError, match="invalid corpus name"):
        translator.translate("files", corpus='x"; MATCH (m) DELETE m //')


def test_unlabelled_first_pattern_skips_injection_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the first MATCH is (n) with no label, corpus filter can't safely
    attach (corpus property isn't universal). Skip injection, log warning,
    pass through the Cypher untouched (caller will see cross-corpus results)."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "MATCH (n) RETURN n LIMIT 5"
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    with caplog.at_level(logging.WARNING):
        result = translator.translate("things", corpus="x")
    assert result == "MATCH (n) RETURN n LIMIT 5"
    assert any("corpus filter" in rec.message for rec in caplog.records)


def test_call_procedure_skips_injection_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CALL db.labels() has no MATCH at all; corpus filtering isn't applicable."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "CALL db.labels()"
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "rendered"
    translator = QueryTranslator(mock_provider, mock_renderer, Ontology.load_base())
    with caplog.at_level(logging.WARNING):
        result = translator.translate("labels", corpus="x")
    assert result == "CALL db.labels()"
    assert any("corpus filter" in rec.message for rec in caplog.records)
