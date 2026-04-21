import json
from unittest.mock import MagicMock

import pytest

from contextd.inference.relate import InferredRelationship, RelationshipInferrer
from contextd.ontology.schema import Ontology


def test_returns_parsed_and_validated_relationships() -> None:
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "relationships": [
                {
                    "type": "REFERENCES",
                    "target_type": "File",
                    "target_name": "other.md",
                    "confidence": 0.95,
                    "reason": "explicit",
                },
                {
                    "type": "UNKNOWN_EDGE",
                    "target_type": "File",
                    "target_name": "x.md",
                    "confidence": 0.9,
                    "reason": "r",
                },
            ]
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    result = inferrer.infer("some content", known_entities=["entity1"])
    # Only the valid one should be kept; UNKNOWN_EDGE is rejected by the ontology.
    assert len(result) == 1
    assert result[0].edge_type == "REFERENCES"
    assert isinstance(result[0], InferredRelationship)


def test_rejects_unknown_target_type() -> None:
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "relationships": [
                {
                    "type": "REFERENCES",
                    "target_type": "Widget",
                    "target_name": "x",
                    "confidence": 0.9,
                    "reason": "r",
                },
            ]
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    result = inferrer.infer("content", known_entities=[])
    assert result == []


def test_handles_yaml_language_tagged_fence() -> None:
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = """```yaml
{"relationships": [{"type": "REFERENCES", "target_type": "File", "target_name": "x.md", "confidence": 0.9, "reason": "r"}]}
```"""
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    result = inferrer.infer("content", known_entities=[])
    assert len(result) == 1
    assert result[0].target_name == "x.md"


def test_handles_prose_wrapper_around_json() -> None:
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = (
        "Here are the relationships I found:\n\n"
        '{"relationships": [{"type": "REFERENCES", "target_type": "File", '
        '"target_name": "y.md", "confidence": 0.9, "reason": "r"}]}\n\n'
        "That's all."
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    result = inferrer.infer("content", known_entities=[])
    assert len(result) == 1
    assert result[0].target_name == "y.md"


def test_non_list_relationships_returns_empty() -> None:
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps({"relationships": "oops"})
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    result = inferrer.infer("content", known_entities=[])
    assert result == []


def test_non_dict_row_is_skipped() -> None:
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "relationships": [
                "bogus",
                {
                    "type": "REFERENCES",
                    "target_type": "File",
                    "target_name": "z.md",
                    "confidence": 0.9,
                    "reason": "r",
                },
            ]
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    result = inferrer.infer("content", known_entities=[])
    assert len(result) == 1
    assert result[0].target_name == "z.md"


def test_no_json_object_raises_valueerror() -> None:
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "I could not infer any relationships."
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    with pytest.raises(ValueError, match="no JSON object"):
        inferrer.infer("content", known_entities=[])
