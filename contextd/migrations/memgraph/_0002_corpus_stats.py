"""Memgraph sibling of SD #70's Kuzu migration.

No-op: Memgraph is schema-free, so Corpus nodes can accept new
properties at write time without DDL. This migration exists to keep
schema_version parity with Kuzu so cross-backend upgrade paths line up.
"""

from typing import Any

from contextd.storage.migration import Migration


def up(store: Any, version: int) -> None:
    pass


migration = Migration(id=2, name="corpus_stats", up=up)
