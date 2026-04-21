from unittest.mock import MagicMock

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
