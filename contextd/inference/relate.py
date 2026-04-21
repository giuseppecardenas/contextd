"""Infers typed relationships from file (or section) content.

Enforces the strict-ontology invariant — any relationship whose edge
type or target type is not declared in the ontology is discarded
silently (spec §3.5). This is the primary defence against hallucinated
edges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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

    def infer(self, content: str, known_entities: list[str]) -> list[InferredRelationship]:
        prompt = self._renderer.render(
            "relate",
            content=content,
            known_entities="\n".join(known_entities[:100]),
            allowed_edge_types=", ".join(sorted(self._onto.edge_types)),
            allowed_node_types=", ".join(sorted(self._onto.node_types)),
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
            if edge_type not in self._onto.edge_types:
                continue
            if target_type not in self._onto.node_types:
                continue
            if not isinstance(target_name, str) or not target_name:
                continue
            valid.append(
                InferredRelationship(
                    edge_type=cast(str, edge_type),
                    target_type=cast(str, target_type),
                    target_name=target_name,
                    confidence=float(row.get("confidence", 0.0)),
                    reason=cast(str, row.get("reason", "")),
                )
            )
        return valid
