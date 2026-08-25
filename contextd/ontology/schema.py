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

# File and Section nodes mirror real on-disk content and are created ONLY by
# the indexer's enumerate phases. Inference may reference them (edges resolve
# to the existing node or are dropped) but must never mint them.
ENUMERATION_OWNED_LABELS: frozenset[str] = frozenset({"File", "Section"})

# Node labels that exist purely for retrieval: Chunk (sub-section retrieval
# units hung off Section/File by the chunking phases, migration _0008) and
# Topic (corpus-level clusters that Section/File join via BELONGS_TO). Neither
# is summarised, related or minted by the LLM — Section and File stay the
# inference units — so the relate phase never advertises them as targets and
# drops any inferred edge that names them.
RETRIEVAL_ONLY_LABELS: frozenset[str] = frozenset({"Chunk", "Topic"})

# Node labels that are never AI-inferred entity targets: the enumeration-owned
# pair above, Corpus/Meta which are indexer bookkeeping (inference had been
# observed minting dozens of junk "Corpus" nodes and nameless "Meta" nodes
# before these were withheld), and the retrieval-only Chunk/Topic pair. Every
# OTHER declared node type is a "mintable" entity that the relate phase may
# create on demand as an inference target — see Ontology.mintable_labels().
# Shared by the relate phase and the ``prune-entities`` CLI command so the two
# agree on the structural/entity split.
NON_ENTITY_LABELS: frozenset[str] = (
    ENUMERATION_OWNED_LABELS | frozenset({"Corpus", "Meta"}) | RETRIEVAL_ONLY_LABELS
)

# Edge types that describe on-disk document structure and are therefore written
# only by the indexer's enumeration and chunking phases with
# ``origin="structural"``: CONTAINS is File->Section and Section/File->Chunk,
# PARENT_OF is Section->Section heading nesting, and NEXT_SIBLING is
# Section->Section / Chunk->Chunk document order.
# The relate phase excludes these from the allow-list it advertises to the model
# and rejects them if the model emits one anyway, because their meaning is
# defined entirely by the heading parser and cannot be recovered from prose. An
# unrestricted allow-list invited name-similarity mistakes such as a File
# -NEXT_SIBLING-> Ticket edge inferred from the phrase "sibling ticket".
STRUCTURAL_EDGE_TYPES: frozenset[str] = frozenset({"CONTAINS", "PARENT_OF", "NEXT_SIBLING"})


class OntologyError(ValueError):
    """Raised when an operation targets a type the ontology does not declare."""


@dataclass(frozen=True)
class EdgeConstraint:
    """Allowed endpoint labels for one edge type; an empty set is a wildcard."""

    src: frozenset[str]
    dst: frozenset[str]


@dataclass(frozen=True)
class Ontology:
    node_types: Mapping[str, tuple[str, ...]]
    edge_types: frozenset[str]
    edge_origin_values: frozenset[str]
    aliases: Mapping[str, str] = field(default_factory=dict)
    edge_aliases: Mapping[str, str] = field(default_factory=dict)
    # Per-edge (src, dst) label constraints. Parallel to ``edge_types`` rather
    # than replacing it so direct constructions (tests, overrides) that omit
    # constraints keep working — an absent entry means unconstrained.
    edge_constraints: Mapping[str, EdgeConstraint] = field(default_factory=dict)

    @classmethod
    def load_base(cls) -> Ontology:
        raw = json.loads(
            resources.files("contextd.ontology").joinpath("base.json").read_text(encoding="utf-8")
        )
        node_types: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in raw["node_types"].items()}
        raw_edges = raw["edge_types"]
        # Two accepted shapes: the legacy flat list (no constraints) and the
        # object form {name: {"src": [...], "dst": [...]}} where an empty
        # array means any inference-legal label.
        constraints: dict[str, EdgeConstraint] = {}
        if isinstance(raw_edges, dict):
            edge_names = frozenset(raw_edges)
            for name, spec in raw_edges.items():
                constraints[name] = EdgeConstraint(
                    src=frozenset(spec.get("src", ())),
                    dst=frozenset(spec.get("dst", ())),
                )
        else:
            edge_names = frozenset(raw_edges)
        return cls(
            node_types=MappingProxyType(node_types),
            edge_types=edge_names,
            edge_origin_values=frozenset(raw["edge_origin_values"]),
            edge_constraints=MappingProxyType(constraints),
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

    def mintable_labels(self) -> frozenset[str]:
        """Node labels inference may create on demand as entity targets.

        Declared node types minus :data:`NON_ENTITY_LABELS`: File/Section are
        enumeration-owned (referenced, never minted), Corpus/Meta are system
        bookkeeping and Chunk/Topic are retrieval-only (none of those four is
        referenced or minted by inference).
        """
        return frozenset(self.node_types) - NON_ENTITY_LABELS

    def inference_target_labels(self) -> frozenset[str]:
        """Node labels the model may name as a relationship target.

        The mintable entity labels plus the enumeration-owned File/Section
        (legal as *reference* targets — resolved to existing nodes, never
        minted). Corpus, Meta, Chunk and Topic are excluded entirely.
        """
        return self.mintable_labels() | ENUMERATION_OWNED_LABELS

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

    def validate_triple(self, src_label: str, edge_type: str, dst_label: str) -> bool:
        """Return whether ``src -[edge]-> dst`` is a permitted combination.

        Edge-type and node-label aliases are resolved first. An unknown edge
        type is invalid; an edge type without a declared constraint (or with
        an empty ``src``/``dst`` set — the wildcard form) permits any label on
        that endpoint. Independent type checks let junk like
        ``Section -DOCUMENTS-> Client`` through; this is the combination-level
        gate.
        """
        resolved_edge = self.resolve_edge_alias(edge_type)
        if resolved_edge not in self.edge_types:
            return False
        constraint = self.edge_constraints.get(resolved_edge)
        if constraint is None:
            return True
        src = self.resolve_alias(src_label)
        dst = self.resolve_alias(dst_label)
        if constraint.src and src not in constraint.src:
            return False
        return not constraint.dst or dst in constraint.dst
