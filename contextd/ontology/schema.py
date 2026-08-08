"""Strict ontology loader + validator.

The base ontology (node types, edge types, valid origin values) ships
in ``contextd/ontology/base.json``. Per-corpus aliases rename base
types without changing semantics; they are applied via
``Ontology.with_aliases()`` (node-label aliases) or
``Ontology.with_edge_aliases()`` (edge-type aliases), each of which
returns a new instance.

AI-inferred relationships are validated against the ontology at write
time; any edge whose type or target type is not declared here is
rejected (spec §3.5). This is the primary defence against hallucinated
relationship types.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib import resources
from types import MappingProxyType

# Node labels that are never AI-inferred entity targets: File/Section are
# created from on-disk content by the indexer, and Corpus/Meta are indexer
# bookkeeping. Every OTHER declared node type is a "stub-able" entity that the
# relate phase may create on demand as an inference target. Shared by the
# relate phase (which targets receive inferred content) and the
# ``prune-entities`` CLI command (which orphaned nodes are prunable) so the two
# agree on the structural/entity split.
NON_ENTITY_LABELS: frozenset[str] = frozenset({"File", "Section", "Corpus", "Meta"})

# Edge types that describe on-disk document structure and are therefore written
# only by the indexer's section-granular enumeration phase with
# ``origin="structural"``: CONTAINS is File->Section, PARENT_OF is Section->
# Section heading nesting, and NEXT_SIBLING is Section->Section document order.
# The relate phase excludes these from the allow-list it advertises to the model
# and rejects them if the model emits one anyway, because their meaning is
# defined entirely by the heading parser and cannot be recovered from prose. An
# unrestricted allow-list invited name-similarity mistakes such as a File
# -NEXT_SIBLING-> Ticket edge inferred from the phrase "sibling ticket".
STRUCTURAL_EDGE_TYPES: frozenset[str] = frozenset({"CONTAINS", "PARENT_OF", "NEXT_SIBLING"})


class OntologyError(ValueError):
    """Raised when an operation targets a type the ontology does not declare."""


@dataclass(frozen=True)
class Ontology:
    node_types: Mapping[str, tuple[str, ...]]
    edge_types: frozenset[str]
    edge_origin_values: frozenset[str]
    aliases: Mapping[str, str] = field(default_factory=dict)
    edge_aliases: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def load_base(cls) -> Ontology:
        raw = json.loads(
            resources.files("contextd.ontology").joinpath("base.json").read_text(encoding="utf-8")
        )
        node_types: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in raw["node_types"].items()}
        return cls(
            node_types=MappingProxyType(node_types),
            edge_types=frozenset(raw["edge_types"]),
            edge_origin_values=frozenset(raw["edge_origin_values"]),
        )

    def with_aliases(self, aliases: Mapping[str, str]) -> Ontology:
        for alias, target in aliases.items():
            if target not in self.node_types:
                raise OntologyError(f"Alias '{alias}' targets unknown node type '{target}'")
        return replace(self, aliases=MappingProxyType(dict(aliases)))

    def with_edge_aliases(self, edge_aliases: Mapping[str, str]) -> Ontology:
        """Layer domain edge-type aliases onto this ontology.

        Returns a new frozen instance. Validates each target is a canonical
        edge type declared in base.json. Stackable with with_aliases() and
        with itself — call with_edge_aliases again to replace (NOT merge)
        the alias map; callers who want additive semantics merge the dict
        themselves before calling.
        """
        for alias, target in edge_aliases.items():
            if target not in self.edge_types:
                raise OntologyError(f"Edge alias '{alias}' targets unknown edge type '{target}'")
        return replace(self, edge_aliases=MappingProxyType(dict(edge_aliases)))

    def resolve_alias(self, name: str) -> str:
        return self.aliases.get(name, name)

    def resolve_edge_alias(self, name: str) -> str:
        return self.edge_aliases.get(name, name)

    def validate_node(self, node_type: str) -> None:
        resolved = self.resolve_alias(node_type)
        if resolved not in self.node_types:
            raise OntologyError(f"Unknown node type '{node_type}'")

    def validate_edge(self, edge_type: str, *, origin: str) -> None:
        resolved = self.resolve_edge_alias(edge_type)
        if resolved not in self.edge_types:
            raise OntologyError(f"Unknown edge type '{edge_type}'")
        if origin not in self.edge_origin_values:
            raise OntologyError(f"Unknown edge origin '{origin}'")
