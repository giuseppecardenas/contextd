"""SD #70 — Corpus.node_count + Corpus.edge_count accounting.

No-op on Memgraph: schema-free, so Corpus nodes can accept new properties
at write time without DDL. This migration exists to keep schema_version
parity across backends so cross-backend upgrade paths line up.
"""

from typing import Any

from contextd.storage.migration import Migration


def up(store: Any, version: int) -> None:
    pass


migration = Migration(id=2, name="corpus_stats", up=up)
