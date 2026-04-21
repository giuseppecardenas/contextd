"""Add node_count + edge_count columns to Corpus node table (SD #70).

Enables phase_close to persist corpus-level statistics for the MCP
`describe_project` tool to surface without recounting per request.
Memgraph's sibling migration is a no-op because it's schema-free.
"""

from typing import Any

from contextd.storage.migration import Migration


def up(store: Any, version: int) -> None:
    store.exec_write("ALTER TABLE Corpus ADD node_count INT64", None)
    store.exec_write("ALTER TABLE Corpus ADD edge_count INT64", None)


migration = Migration(id=2, name="corpus_stats", up=up)
