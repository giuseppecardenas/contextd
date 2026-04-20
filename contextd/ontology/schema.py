"""Strict ontology loader + validator.

The base ontology (node types, edge types, valid origin values) ships
in ``contextd/ontology/base.json``. Per-corpus aliases rename base
types without changing semantics; they are applied via
``Ontology.with_aliases()`` which returns a new instance.

AI-inferred relationships are validated against the ontology at write
time; any edge whose type or target type is not declared here is
rejected (spec §3.5). This is the primary defence against hallucinated
relationship types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from importlib import resources
from typing import Any


class OntologyError(ValueError):
    """Raised when an operation targets a type the ontology does not declare."""


@dataclass(frozen=True)
class Ontology:
    node_types: dict[str, list[str]]
    edge_types: set[str]
    edge_origin_values: set[str]
    aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load_base(cls) -> Ontology:
        raw = json.loads(resources.files("contextd.ontology").joinpath("base.json").read_text())
        return cls(
            node_types=dict(raw["node_types"]),
            edge_types=set(raw["edge_types"]),
            edge_origin_values=set(raw["edge_origin_values"]),
        )

    def with_aliases(self, aliases: dict[str, str]) -> Ontology:
        for alias, target in aliases.items():
            if target not in self.node_types:
                raise OntologyError(f"Alias '{alias}' targets unknown node type '{target}'")
        return replace(self, aliases=dict(aliases))

    def resolve_alias(self, name: str) -> str:
        return self.aliases.get(name, name)

    def validate_node(self, node_type: str, properties: dict[str, Any]) -> None:
        resolved = self.resolve_alias(node_type)
        if resolved not in self.node_types:
            raise OntologyError(f"Unknown node type '{node_type}'")

    def validate_edge(self, edge_type: str, *, origin: str) -> None:
        if edge_type not in self.edge_types:
            raise OntologyError(f"Unknown edge type '{edge_type}'")
        if origin not in self.edge_origin_values:
            raise OntologyError(f"Unknown edge origin '{origin}'")
