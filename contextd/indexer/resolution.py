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
  (c) fuzzy match (rapidfuzz) — next commit,
  (d) embedding similarity — later commit.

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
        # Rungs (c) fuzzy and (d) embedding slot in here in later commits.
        with self._lock:
            if not known:
                # Record the mint so intra-batch duplicates collapse; an
                # ambiguous marker is never overwritten.
                mapping[norm] = name
        return Resolution(action="minted", pk_value=name, rule="minted", norm=norm)

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
