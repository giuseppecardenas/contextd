"""Per-label primary-key property shared between storage backends.

Memgraph's baseline migration declares uniqueness constraints keyed by the
property names below (e.g., ``ASSERT n.path IS UNIQUE`` for File, ``n.id``
for Section); Neo4j's baseline declares matching constraints. Sharing the
same map keeps both backends consistent and ensures edge-MATCH queries and
upsert MERGEs use the right key per label.

When migrations add or rename a node-table PK, update this map in lock-step.
"""

from __future__ import annotations

from typing import Final

PRIMARY_KEY_BY_LABEL: Final[dict[str, str]] = {
    "File": "path",
    "Section": "id",
    "Artifact": "id",
    "Ticket": "id",
    "Pattern": "name",
    "Technology": "name",
    "Client": "name",
    "Repo": "name",
    "Service": "name",
    "Integration": "name",
    # Risk is keyed by ``description`` to make re-indexing idempotent: the
    # inferrer emits a Risk node identified solely by its description text,
    # and MERGE semantics collapse identical-text upserts into a single node
    # (the second upsert overwrites properties in-place rather than creating
    # a duplicate).  Two distinct Risks that happen to share identical
    # description phrasing will also merge — in the Runeledger use case this
    # is rare and considered correct; audit-gap entries are nearly always
    # unique prose.  If future content needs co-existing same-phrased Risks,
    # the remedy is to migrate to a content-hash-derived ``id`` field (plus
    # a baseline migration for both Memgraph and Neo4j) — not a field rename.
    # Decision recorded 2026-04-21 (M10.10 follow-up to M10.1 code-review
    # concern that flagged this as potentially fragile).
    "Risk": "description",
    "WorkSession": "id",
    "Corpus": "name",
    "Meta": "schema_version",
}


def primary_key_for(label: str) -> str:
    """Return the primary-key property name for ``label``, or raise."""
    try:
        return PRIMARY_KEY_BY_LABEL[label]
    except KeyError as exc:
        raise ValueError(f"Unknown node label {label!r}; add it to PRIMARY_KEY_BY_LABEL") from exc
