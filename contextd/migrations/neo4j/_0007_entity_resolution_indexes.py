"""Indexes backing the entity-resolution cascade.

Two index families per mintable entity label (the ten node types inference may
create — ``Ontology.mintable_labels()``):

* ``{Label}_name_norm_idx`` — btree on the normalised-name property the
  cascade's exact-normalized rung matches against (written at mint time; the
  index serves cold cache loads).
* ``{Label}_embedding_idx`` — 1024-dim cosine vector index for the cascade's
  embedding rung. ``vector_search`` interpolates the
  ``{label}_{property}_idx`` name (``contextd/storage/neo4j.py``); before this
  migration only ``File``/``Section`` had vector indexes, so
  ``EntityResolver.resolve`` raised on every entity label.

No data backfill: the repair path for pre-existing graphs is a full
wipe + re-bootstrap (remove-corpus + add-corpus + bootstrap), after which
``name_norm``/``embedding`` are written fresh at mint time. Nodes minted
before that carry neither property and simply never match these indexes.

Idempotent: ``IF NOT EXISTS`` guards every statement so replays are safe.
"""

from typing import Any

from contextd.storage.migration import Migration

_ENTITY_LABELS = [
    "Artifact",
    "Ticket",
    "Pattern",
    "Technology",
    "Client",
    "Repo",
    "Service",
    "Integration",
    "Risk",
    "WorkSession",
]

_DDL = [
    *(
        f"CREATE INDEX {label}_name_norm_idx IF NOT EXISTS FOR (n:{label}) ON (n.name_norm)"
        for label in _ENTITY_LABELS
    ),
    *(
        f"CREATE VECTOR INDEX {label}_embedding_idx IF NOT EXISTS "
        f"FOR (n:{label}) ON n.embedding "
        "OPTIONS {indexConfig: {`vector.dimensions`: 1024, "
        "`vector.similarity_function`: 'cosine'}}"
        for label in _ENTITY_LABELS
    ),
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=7, name="entity_resolution_indexes_neo4j", up=up)
