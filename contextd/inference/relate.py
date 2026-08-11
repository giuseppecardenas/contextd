"""Infers typed relationships from file (or section) content.

Enforces the strict-ontology invariant: any relationship whose edge type or
target type is not declared in the ontology is discarded (spec §3.5), with an
INFO log naming the reason so discards are countable from the log.
This is the primary defence against hallucinated edges.

The permitted edge types are narrower than the ontology's declared set. The
structural types in :data:`~contextd.ontology.schema.STRUCTURAL_EDGE_TYPES`
belong to the section-granular enumeration phase, which writes them with
``origin="structural"`` from parsed heading structure, so this module neither
advertises them to the model nor accepts them back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from contextd.inference._json_body import loads_json_body
from contextd.inference.context import CandidateBundle, UnitIdentity, identity_vars
from contextd.inference.prompts import PromptRenderer
from contextd.ontology.schema import NON_ENTITY_LABELS, STRUCTURAL_EDGE_TYPES, Ontology
from contextd.providers.base import InferenceProvider, PromptRequest

_log = logging.getLogger(__name__)

# Word-form confidences some models emit despite the numeric-scale instruction.
_CONFIDENCE_WORDS = {"high": 0.9, "medium": 0.7, "low": 0.5}

# Reasons are stored as an edge property; keep them bounded so a rambling
# completion can't bloat the graph.
_MAX_REASON_CHARS = 500


def _coerce_confidence(value: Any) -> float:
    """Coerce a model-supplied confidence into a float in [0, 1]. Never raises.

    A bare ``float(value)`` on a string like ``"high"`` used to escape
    ``infer()`` and lose the whole unit's edge batch; word forms now map to
    their prompt-documented anchors and anything unparseable becomes 0.0 (which
    the confidence floor then drops) with an INFO log.
    """
    if isinstance(value, bool):
        _log.info("relate drop-signal: boolean confidence %r coerced to 0.0", value)
        return 0.0
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    if isinstance(value, str):
        word = _CONFIDENCE_WORDS.get(value.strip().lower())
        if word is not None:
            return word
        try:
            return min(1.0, max(0.0, float(value)))
        except ValueError:
            pass
    _log.info("relate drop-signal: unparseable confidence %r coerced to 0.0", value)
    return 0.0


def _coerce_reason(value: Any) -> str:
    """Coerce a model-supplied reason into a bounded string. Never raises.

    A non-string reason (dicts have been observed) previously reached the
    store as a map property and aborted the run; anything non-string is
    stringified with an INFO log, and all reasons are truncated.
    """
    if isinstance(value, str):
        text = value
    else:
        _log.info("relate: non-string reason %.80r coerced to str", value)
        text = "" if value is None else str(value)
    return text[:_MAX_REASON_CHARS]


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
# the model may supply. Sourced from the shared ontology constant so the relate
# phase and the prune-entities CLI agree on the structural/entity split.
_NON_CONTENT_LABELS = NON_ENTITY_LABELS


def _emittable_edge_types(ontology: Ontology) -> frozenset[str]:
    """Return the edge types the model is permitted to infer.

    This is the ontology's declared edge types minus
    :data:`~contextd.ontology.schema.STRUCTURAL_EDGE_TYPES`, which are owned by
    the section-granular enumeration phase and carry ``origin="structural"``.
    Used both to build the allow-list injected into the relate prompt and to
    reject structural types at parse time, so the exclusion holds even if the
    prompt template drifts out of sync with this module.
    """
    return ontology.edge_types - STRUCTURAL_EDGE_TYPES


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

    def infer(
        self,
        content: str,
        *,
        identity: UnitIdentity | None = None,
        candidates: CandidateBundle | None = None,
    ) -> list[InferredRelationship]:
        """Infer typed relationships the given content establishes.

        The edge-type allow-list advertised to the model and the allow-list
        enforced at parse time are both :func:`_emittable_edge_types`, which
        excludes the indexer-owned structural types. Rejection is applied to the
        alias-resolved type so a per-corpus edge alias cannot reintroduce a
        structural type under a domain name.

        Side effects: issues one inference-provider call per invocation, billed
        against the ``inference`` call site.

        :param content: File or section body the model reasons over.
        :param identity: Where the content lives (path, section title, parent
            chain) — rendered into the prompt's Source block. ``None`` renders
            empty identity fields; production always supplies it.
        :param candidates: Real graph nodes offered as preferred targets.
            ``None`` behaves as an empty bundle.
        :return: The relationships that passed every validation gate, which may
            be empty. Rows failing any gate are dropped with an INFO log naming
            the reason.
        """
        emittable_edge_types = _emittable_edge_types(self._onto)
        # Advertise the mintable entity labels + File/Section (reference
        # targets) + per-corpus alias names — NOT the full declared set:
        # Corpus/Meta are system labels the model kept minting junk under.
        # Aliases are advertised so a domain corpus can steer the model toward
        # its own vocabulary; they resolve to canonical labels at parse time.
        target_labels = self._onto.inference_target_labels()
        advertised_node_types = target_labels | set(self._onto.aliases)
        bundle = candidates if candidates is not None else CandidateBundle.empty()
        prompt = self._renderer.render(
            "relate",
            content=content,
            candidate_context=bundle.render(),
            allowed_edge_types=", ".join(sorted(emittable_edge_types)),
            allowed_node_types=", ".join(sorted(advertised_node_types)),
            target_property_schema=self._target_property_schema(),
            **identity_vars(identity),
        )
        response = self._provider.generate(
            PromptRequest(system="", prompt=prompt, call_site="inference")
        )
        data = loads_json_body(response)
        valid: list[InferredRelationship] = []
        relationships = data.get("relationships")
        if not isinstance(relationships, list):
            relationships = []
        for row in relationships:
            if not isinstance(row, dict):
                _log.info("relate drop: non-dict row %.120r", row)
                continue
            edge_type = row.get("type")
            target_type = row.get("target_type")
            target_name = row.get("target_name")
            if not isinstance(edge_type, str):
                _log.info("relate drop: non-string edge type in row %.120r", row)
                continue
            resolved_edge_type = self._onto.resolve_edge_alias(edge_type)
            if resolved_edge_type not in emittable_edge_types:
                _log.info(
                    "relate drop: edge type %r (resolved %r) not emittable; target %.80r",
                    edge_type,
                    resolved_edge_type,
                    target_name,
                )
                continue
            if not isinstance(target_type, str):
                _log.info(
                    "relate drop: non-string target type %r; target %.80r",
                    target_type,
                    target_name,
                )
                continue
            # First production use of node-label aliases: a per-corpus alias
            # (e.g. Registry -> Pattern) advertised in the prompt resolves to
            # its canonical label here, so downstream only sees canon.
            resolved_target_type = self._onto.resolve_alias(target_type)
            if resolved_target_type not in target_labels:
                _log.info(
                    "relate drop: target type %r (resolved %r) not an inference target; "
                    "target %.80r",
                    target_type,
                    resolved_target_type,
                    target_name,
                )
                continue
            if not isinstance(target_name, str) or not target_name:
                _log.info(
                    "relate drop: empty or non-string target name for %s -> %s",
                    resolved_edge_type,
                    target_type,
                )
                continue
            valid.append(
                InferredRelationship(
                    edge_type=resolved_edge_type,
                    target_type=resolved_target_type,
                    target_name=target_name,
                    confidence=_coerce_confidence(row.get("confidence", 0.0)),
                    reason=_coerce_reason(row.get("reason", "")),
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
