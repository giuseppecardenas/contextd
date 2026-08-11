"""Entity-resolution cascade: deterministic-first, per-kind normalization.

The audited runeledger graph held ~4,600 entity nodes, 84-91% of them
degree<=1 bare-name stubs, with the same concept minted under case variants
(``Tailor``/``tailor``) and across labels. Every surveyed GraphRAG system
resolves a newly-extracted entity against the live graph deterministically
before minting; this module is contextd's cascade:

  (a) normalize per-kind (casefold only for prose-concept labels — case is
      significant for code-symbol-ish kinds),
  (b) exact normalized-name match against an in-memory per-(corpus, label)
      map, lazily loaded from the graph (backed by the ``name_norm`` btree
      index from migration _0007) and updated on every mint so intra-batch
      duplicates collapse without a second pass,
  (c) fuzzy match (rapidfuzz WRatio, length-gated),
  (d) embedding similarity via ``EntityResolver`` — the name is embedded
      once and the vector is reused at mint time when nothing matches.

Ambiguity (two distinct nodes sharing one normalized name — possible on
legacy data written before ``name_norm``) logs and falls through: matching
would be a guess. LLM adjudication of ambiguous cases is deliberately
deferred; the ``ambiguous`` log lines build the corpus for a later pass.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz, process

from contextd.indexer.entity_resolver import EntityResolver
from contextd.storage._keys import primary_key_for
from contextd.storage.base import GraphStore

_log = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

# Labels whose instances are prose concepts, where case variance is noise.
# Repo/Artifact/Ticket/WorkSession identifiers stay case-sensitive: for code
# symbols and IDs, `Foo` and `foo` may be genuinely different things.
_DEFAULT_CASE_INSENSITIVE: frozenset[str] = frozenset(
    {"Pattern", "Technology", "Client", "Risk", "Service", "Integration"}
)


def normalize_name(name: str, *, casefold: bool) -> str:
    """NFC-normalize, collapse internal whitespace, strip; casefold on request."""
    text = _WHITESPACE.sub(" ", unicodedata.normalize("NFC", name)).strip()
    return text.casefold() if casefold else text


@dataclass(frozen=True)
class ResolutionSettings:
    """Per-corpus knobs for the cascade (``[resolution]`` in the corpus TOML)."""

    case_insensitive_labels: frozenset[str] = _DEFAULT_CASE_INSENSITIVE
    fuzzy_threshold: float = 90.0
    fuzzy_min_length: int = 6
    embedding_threshold: float = 0.92  # normalized-cosine scale; orthogonal = 0.5
    embedding_enabled: bool = True
    confidence_floor: float = 0.5


@dataclass(frozen=True)
class Resolution:
    action: Literal["matched", "minted"]
    pk_value: str
    rule: str  # "exact-norm" | "fuzzy:<score>" | "embedding:<score>" | "minted"
    norm: str  # the normalized form (written as name_norm at mint time)
    vector: list[float] | None = None  # embedding computed during rung (d), reused at mint


class EntityCascadeResolver:
    """Resolve an entity mention to an existing node, or decide to mint.

    One instance per pipeline construction (shared by all relate workers);
    the per-(corpus, label) normalized-name maps are lock-protected and
    updated on every mint, which is what collapses intra-batch duplicates.
    """

    def __init__(
        self,
        store: GraphStore,
        settings: ResolutionSettings,
        embed: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._embed = embed
        self._entity_resolver = (
            EntityResolver(store, embed, threshold=settings.embedding_threshold)
            if embed is not None
            else None
        )
        self._maps: dict[tuple[str, str], dict[str, str | None]] = {}
        self._lock = threading.Lock()

    @property
    def settings(self) -> ResolutionSettings:
        return self._settings

    def normalize(self, label: str, name: str) -> str:
        return normalize_name(name, casefold=label in self._settings.case_insensitive_labels)

    def resolve(self, label: str, name: str, corpus: str) -> Resolution:
        norm = self.normalize(label, name)
        mapping = self._mapping(label, corpus)
        with self._lock:
            known = norm in mapping
            hit = mapping.get(norm)
        if hit is not None:
            if hit != name:
                _log.info(
                    "resolve: %s %.80r matched existing %.80r by exact-norm", label, name, hit
                )
            return Resolution(action="matched", pk_value=hit, rule="exact-norm", norm=norm)
        if known:  # present but None → known-ambiguous normalized name
            _log.info("resolve: %s %.80r ambiguous by exact-norm; minting as-is", label, name)
        fuzzy = self._fuzzy_match(label, name, norm, mapping)
        if fuzzy is not None:
            return fuzzy
        embedded, vec = self._embedding_match(label, name, norm, corpus)
        if embedded is not None:
            return embedded
        with self._lock:
            if not known:
                # Record the mint so intra-batch duplicates collapse; an
                # ambiguous marker is never overwritten.
                mapping[norm] = name
        return Resolution(action="minted", pk_value=name, rule="minted", norm=norm, vector=vec)

    def _fuzzy_match(
        self, label: str, name: str, norm: str, mapping: dict[str, str | None]
    ) -> Resolution | None:
        """Rung (c): rapidfuzz WRatio against the known normalized names.

        Length-gated (short names are collision-prone — the stand-in for
        Graphiti's Shannon-entropy gate). WRatio over token_set_ratio for its
        length-mismatch handling (``runeledger.register_x`` vs
        ``register_x``). Scores in [80, threshold) log as ``ambiguous-fuzzy``
        and fall through to mint — the audit corpus for a future LLM
        adjudication rung.
        """
        s = self._settings
        if len(norm) < s.fuzzy_min_length:
            return None
        with self._lock:
            candidates = {k: v for k, v in mapping.items() if v is not None}
        if not candidates:
            return None
        best = process.extractOne(
            norm, list(candidates.keys()), scorer=fuzz.WRatio, score_cutoff=80.0
        )
        if best is None:
            return None
        matched_norm, score, _ = best
        pk = candidates[matched_norm]
        if score >= s.fuzzy_threshold:
            _log.info(
                "resolve: %s %.80r matched existing %.80r by fuzzy (score %.1f)",
                label,
                name,
                pk,
                score,
            )
            return Resolution(action="matched", pk_value=pk, rule=f"fuzzy:{score:.1f}", norm=norm)
        _log.info(
            "resolve: %s %.80r ambiguous-fuzzy vs %.80r (score %.1f below %.1f); minting",
            label,
            name,
            pk,
            score,
            s.fuzzy_threshold,
        )
        return None

    def _embedding_match(
        self, label: str, name: str, norm: str, corpus: str
    ) -> tuple[Resolution | None, list[float] | None]:
        """Rung (d): embedding similarity, delegated to ``EntityResolver``.

        The name is embedded once; on a miss the vector is returned so the
        mint writes it as the node's ``embedding`` — one embed call serves
        both the check and the mint. Any failure (provider, missing index)
        degrades to a vector-less mint, never blocks the edge.
        """
        s = self._settings
        if not s.embedding_enabled or self._embed is None or self._entity_resolver is None:
            return None, None
        try:
            [vec] = self._embed([name])
        except Exception as exc:
            _log.warning("resolve: embed failed for %s %.80r: %s", label, name, exc)
            return None, None
        try:
            scored = self._entity_resolver.resolve_scored(label, name, vector=vec, corpus=corpus)
        except Exception as exc:
            _log.warning("resolve: embedding search failed for %s %.80r: %s", label, name, exc)
            return None, vec
        if scored is not None:
            pk, score = scored
            _log.info(
                "resolve: %s %.80r matched existing %.80r by embedding (score %.3f)",
                label,
                name,
                pk,
                score,
            )
            return (
                Resolution(action="matched", pk_value=pk, rule=f"embedding:{score:.3f}", norm=norm),
                vec,
            )
        return None, vec

    def _mapping(self, label: str, corpus: str) -> dict[str, str | None]:
        key = (corpus, label)
        with self._lock:
            existing = self._maps.get(key)
            if existing is not None:
                return existing
        loaded = self._load(label, corpus)
        with self._lock:
            return self._maps.setdefault(key, loaded)

    def _load(self, label: str, corpus: str) -> dict[str, str | None]:
        pk = primary_key_for(label)
        try:
            rows = self._store.exec_read(
                # label/pk come from the ontology + _keys map, not user input.
                f"MATCH (n:{label} {{corpus: $c}}) RETURN n.{pk} AS name, n.name_norm AS name_norm",
                {"c": corpus},
            )
        except Exception as exc:
            _log.warning("resolve: cache load failed for %s: %s", label, exc)
            return {}
        mapping: dict[str, str | None] = {}
        for r in rows:
            name = r.get("name")
            if not name:
                continue
            # Legacy nodes predate name_norm; normalize on the fly.
            norm = str(r.get("name_norm") or self.normalize(label, str(name)))
            if norm in mapping and mapping[norm] != name:
                _log.info(
                    "resolve: %s normalized name %.80r is ambiguous in the graph", label, norm
                )
                mapping[norm] = None  # known-ambiguous: never auto-match
            else:
                mapping.setdefault(norm, str(name))
        return mapping
