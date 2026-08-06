import json
from unittest.mock import MagicMock

import pytest

from contextd.inference.relate import InferredRelationship, RelationshipInferrer
from contextd.ontology.schema import STRUCTURAL_EDGE_TYPES, Ontology


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


def test_infer_resolves_edge_aliases() -> None:
    """Aliased edge types are resolved to the canonical name before storage."""
    ontology = Ontology.load_base().with_edge_aliases({"CITES": "REFERENCES"})
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "relationships": [
                {
                    "type": "CITES",
                    "target_type": "File",
                    "target_name": "foo.md",
                    "confidence": 0.9,
                    "reason": "explicit citation",
                }
            ]
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    result = inferrer.infer("content", known_entities=[])
    assert len(result) == 1
    assert result[0].edge_type == "REFERENCES"  # canonical, not "CITES"


def test_extracts_declared_target_properties() -> None:
    """Model-supplied content is kept only for declared, non-system fields;
    undeclared keys and system/PK fields are dropped."""
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "relationships": [
                {
                    "type": "REFERENCES",
                    "target_type": "Ticket",
                    "target_name": "INTENG-1",
                    "confidence": 0.9,
                    "reason": "r",
                    "properties": {
                        "title": "Fix auth",
                        "status": "open",
                        "id": "OVERWRITE-ATTEMPT",  # PK/system field -> dropped
                        "bogus": "nope",  # undeclared -> dropped
                    },
                }
            ]
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    result = inferrer.infer("content", known_entities=[])
    assert result[0].target_properties == {"title": "Fix auth", "status": "open"}


def test_target_properties_default_empty_when_absent() -> None:
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "relationships": [
                {
                    "type": "REFERENCES",
                    "target_type": "Pattern",
                    "target_name": "Singleton",
                    "confidence": 0.9,
                    "reason": "r",
                }
            ]
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    result = inferrer.infer("content", known_entities=[])
    assert result[0].target_properties == {}


def test_target_properties_accepts_lists_and_drops_objects() -> None:
    """String lists are storable; nested objects are rejected."""
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "relationships": [
                {
                    "type": "REFERENCES",
                    "target_type": "Pattern",
                    "target_name": "Repo Pattern",
                    "confidence": 0.9,
                    "reason": "r",
                    "properties": {
                        "description": "desc",
                        "examples": ["a", "b"],
                        "when_to_use": {"nested": "obj"},  # object -> dropped
                    },
                }
            ]
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    result = inferrer.infer("content", known_entities=[])
    assert result[0].target_properties == {"description": "desc", "examples": ["a", "b"]}


def test_prompt_receives_per_type_property_schema() -> None:
    """The relate prompt is given the per-type content schema, excluding the
    enumeration-owned File/Section labels."""
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps({"relationships": []})
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    inferrer.infer("content", known_entities=[])
    schema = mock_renderer.render.call_args.kwargs["target_property_schema"]
    assert "Ticket: status, title" in schema
    assert "File:" not in schema
    assert "Section:" not in schema


def test_missing_or_empty_target_name_is_silently_discarded() -> None:
    """Rows that pass ontology checks but lack target_name are dropped
    silently — consistent with the tolerant-parsing pattern the rest of
    this module uses. Previously a missing target_name raised KeyError
    mid-loop and aborted the whole batch."""
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "relationships": [
                {"type": "REFERENCES", "target_type": "File", "confidence": 0.9, "reason": "r"},
                {
                    "type": "REFERENCES",
                    "target_type": "File",
                    "target_name": "",
                    "confidence": 0.9,
                    "reason": "r",
                },
                {
                    "type": "REFERENCES",
                    "target_type": "File",
                    "target_name": "ok.md",
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
    assert result[0].target_name == "ok.md"


def test_rejects_structural_edge_types() -> None:
    """Structural types are owned by the section enumeration phase.

    Reproduces the real-world failure this guard was added for: the model read
    the phrase "sibling ticket" in prose and emitted a File -NEXT_SIBLING->
    Ticket edge, which passed the old declared-type check because NEXT_SIBLING
    is a legitimate ontology type. It must now be dropped, while a
    non-structural type in the same response is still kept.
    """
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "relationships": [
                {
                    "type": "NEXT_SIBLING",
                    "target_type": "Ticket",
                    "target_name": "INTENG-5299",
                    "confidence": 0.9,
                    "reason": "Explicitly listed as a sibling ticket.",
                },
                {
                    "type": "PARENT_OF",
                    "target_type": "Service",
                    "target_name": "svc",
                    "confidence": 0.9,
                    "reason": "r",
                },
                {
                    "type": "CONTAINS",
                    "target_type": "Pattern",
                    "target_name": "pat",
                    "confidence": 0.9,
                    "reason": "r",
                },
                {
                    "type": "REFERENCES",
                    "target_type": "File",
                    "target_name": "other.md",
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
    assert [r.edge_type for r in result] == ["REFERENCES"]


def test_prompt_allow_list_withholds_structural_edge_types() -> None:
    """The advertised allow-list excludes structural types but keeps the rest."""
    ontology = Ontology.load_base()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps({"relationships": []})
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    inferrer.infer("content", known_entities=[])
    advertised = {
        t.strip() for t in mock_renderer.render.call_args.kwargs["allowed_edge_types"].split(",")
    }
    assert advertised.isdisjoint(STRUCTURAL_EDGE_TYPES)
    assert advertised == set(ontology.edge_types) - STRUCTURAL_EDGE_TYPES


def test_edge_alias_cannot_reintroduce_structural_type() -> None:
    """Rejection applies post-alias-resolution, so an alias cannot smuggle one in."""
    ontology = Ontology.load_base().with_edge_aliases({"HAS_SECTION": "CONTAINS"})
    mock_provider = MagicMock()
    mock_provider.generate.return_value = json.dumps(
        {
            "relationships": [
                {
                    "type": "HAS_SECTION",
                    "target_type": "Pattern",
                    "target_name": "pat",
                    "confidence": 0.9,
                    "reason": "r",
                }
            ]
        }
    )
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(mock_provider, mock_renderer, ontology)
    assert inferrer.infer("content", known_entities=[]) == []
