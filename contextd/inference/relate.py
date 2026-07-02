"""Infers typed relationships from file (or section) content.

Enforces the strict-ontology invariant — any relationship whose edge
type or target type is not declared in the ontology is discarded
silently (spec §3.5). This is the primary defence against hallucinated
edges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

from contextd.inference._json_body import extract_json_body
from contextd.inference.prompts import PromptRenderer
from contextd.ontology.schema import Ontology
from contextd.providers.base import InferenceProvider, PromptRequest


@dataclass
class InferredRelationship:
    edge_type: str
    target_type: str
    target_name: str
    confidence: float
    reason: str
    target_properties: dict[str, Any] = field(default_factory=dict)


# Properties the indexer and pipeline own; the model must never supply them.
# Covers node identity and structural fields, corpus/timestamp bookkeeping, and
# derived summary/embedding data. Inferred entity content is filtered down to
# the declared ontology properties MINUS this set. The primary key is stripped
# separately at the upsert site in the indexer (it equals ``target_name``),
# which also protects ``Risk`` whose primary key is the content field
# ``description``.
_SYSTEM_PROPS: frozenset[str] = frozenset(
    {
        "path",
        "name",
        "id",
        "anchor",
        "level",
        "file_id",
        "ordinal",
        "corpus",
        "updated",
        "created",
        "registered_at",
        "root",
        "content_profile",
        "embedding",
        "summary",
        "key_points",
        "summary_generated_at",
        "summary_confidence",
        "entities_mentioned",
        "hash",
        "size",
        "schema_version",
        "contextd_version",
        "backend_name",
        "initialised_at",
        "start",
        "end",
    }
)

# File/Section are enumeration-owned (inferred references resolve to existing
# nodes and never receive inferred content); Corpus/Meta are not inference
# targets. Every other declared node type is a stub-able entity whose content
# the model may supply.
_NON_CONTENT_LABELS: frozenset[str] = frozenset({"File", "Section", "Corpus", "Meta"})


def _content_property_names(ontology: Ontology, target_type: str) -> frozenset[str]:
    """Return the declared properties of ``target_type`` the model may fill.

    These are the ontology-declared property names for the node type minus the
    indexer-owned :data:`_SYSTEM_PROPS`. Used both to build the per-type schema
    injected into the relate prompt and to filter model-supplied properties at
    parse time so hallucinated keys never reach storage.
    """
    return frozenset(ontology.node_types.get(target_type, ())) - _SYSTEM_PROPS


def _clean_property_value(value: Any) -> str | bool | int | float | list[str] | None:
    """Coerce an LLM-supplied property value into a storable form, or ``None``.

    Accepts booleans, numbers, non-empty strings, and lists of non-empty
    strings; nested objects, nulls, and empty values are rejected (returned as
    ``None`` so the caller drops the key). ``bool`` is checked before
    ``int``/``float`` because ``bool`` is a subclass of ``int``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value or None
    if isinstance(value, list):
        items = [item for item in value if isinstance(item, str) and item]
        return items or None
    return None


class RelationshipInferrer:
    def __init__(
        self,
        provider: InferenceProvider,
        renderer: PromptRenderer,
        ontology: Ontology,
    ) -> None:
        self._provider = provider
        self._renderer = renderer
        self._onto = ontology

    def _target_property_schema(self) -> str:
        """Render the per-type content-property guidance for the relate prompt.

        One line per stub-able node type listing the properties the model may
        populate for that target (declared ontology properties minus the
        indexer-owned system fields). Types with no fillable content are
        omitted so the prompt stays terse.
        """
        lines: list[str] = []
        for label in sorted(self._onto.node_types):
            if label in _NON_CONTENT_LABELS:
                continue
            props = sorted(_content_property_names(self._onto, label))
            if props:
                lines.append(f"- {label}: {', '.join(props)}")
        return "\n".join(lines)

    def infer(self, content: str, known_entities: list[str]) -> list[InferredRelationship]:
        prompt = self._renderer.render(
            "relate",
            content=content,
            known_entities="\n".join(known_entities[:100]),
            allowed_edge_types=", ".join(sorted(self._onto.edge_types)),
            allowed_node_types=", ".join(sorted(self._onto.node_types)),
            target_property_schema=self._target_property_schema(),
        )
        response = self._provider.generate(
            PromptRequest(system="", prompt=prompt, call_site="inference")
        )
        cleaned = extract_json_body(response)
        data = cast(dict[str, Any], json.loads(cleaned))
        valid: list[InferredRelationship] = []
        relationships = data.get("relationships")
        if not isinstance(relationships, list):
            relationships = []
        for row in relationships:
            if not isinstance(row, dict):
                continue
            edge_type = row.get("type")
            target_type = row.get("target_type")
            target_name = row.get("target_name")
            if not isinstance(edge_type, str):
                continue
            resolved_edge_type = self._onto.resolve_edge_alias(edge_type)
            if resolved_edge_type not in self._onto.edge_types:
                continue
            if target_type not in self._onto.node_types:
                continue
            if not isinstance(target_name, str) or not target_name:
                continue
            resolved_target_type = cast(str, target_type)
            valid.append(
                InferredRelationship(
                    edge_type=resolved_edge_type,
                    target_type=resolved_target_type,
                    target_name=target_name,
                    confidence=float(row.get("confidence", 0.0)),
                    reason=cast(str, row.get("reason", "")),
                    target_properties=self._extract_target_properties(
                        resolved_target_type, row.get("properties")
                    ),
                )
            )
        return valid

    def _extract_target_properties(self, target_type: str, raw: Any) -> dict[str, Any]:
        """Filter model-supplied target properties to the allowed, storable set.

        Keeps only keys declared for ``target_type`` in the ontology (minus the
        indexer-owned system fields) whose values coerce to a storable scalar
        or string list. Anything else is dropped, so a hallucinated key or a
        nested-object value never reaches ``upsert_node``. Returns an empty dict
        when ``raw`` is not an object.
        """
        if not isinstance(raw, dict):
            return {}
        allowed = _content_property_names(self._onto, target_type)
        cleaned: dict[str, Any] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or key not in allowed:
                continue
            coerced = _clean_property_value(value)
            if coerced is not None:
                cleaned[key] = coerced
        return cleaned
