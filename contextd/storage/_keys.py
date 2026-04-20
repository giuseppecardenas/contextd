"""Per-label primary-key property shared between storage backends.

The Kuzu baseline migration declares an explicit PRIMARY KEY for every
node table, so edge-MATCH queries and upsert MERGEs must use the right
key per label. Memgraph's baseline migration declares a matching
uniqueness constraint (e.g., ``ASSERT n.path IS UNIQUE`` for File, ``n.id``
for Section); sharing the same map keeps both backends consistent.

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
