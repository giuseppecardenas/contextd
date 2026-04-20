"""Baseline schema for Memgraph backend.

Creates uniqueness constraints, corpus indexes, vector index, and
full-text indexes per spec §5.9 Step 3.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    # Uniqueness constraints.
    "CREATE CONSTRAINT ON (n:File) ASSERT n.path IS UNIQUE",
    "CREATE CONSTRAINT ON (n:Section) ASSERT n.id IS UNIQUE",
    "CREATE CONSTRAINT ON (n:Artifact) ASSERT n.id IS UNIQUE",
    "CREATE CONSTRAINT ON (n:Ticket) ASSERT n.id IS UNIQUE",
    "CREATE CONSTRAINT ON (n:Pattern) ASSERT n.name IS UNIQUE",
    "CREATE CONSTRAINT ON (n:Technology) ASSERT n.name IS UNIQUE",
    "CREATE CONSTRAINT ON (n:Client) ASSERT n.name IS UNIQUE",
    "CREATE CONSTRAINT ON (n:Repo) ASSERT n.name IS UNIQUE",
    "CREATE CONSTRAINT ON (n:Service) ASSERT n.name IS UNIQUE",
    "CREATE CONSTRAINT ON (n:Integration) ASSERT n.name IS UNIQUE",
    "CREATE CONSTRAINT ON (n:Corpus) ASSERT n.name IS UNIQUE",
    # Corpus-property indexes for multi-corpus filter.
    "CREATE INDEX ON :File(corpus)",
    "CREATE INDEX ON :Section(corpus)",
    "CREATE INDEX ON :Artifact(corpus)",
    "CREATE INDEX ON :Ticket(corpus)",
    "CREATE INDEX ON :Pattern(corpus)",
    "CREATE INDEX ON :Risk(corpus)",
    # Vector index (1024-dim cosine).
    """
    CREATE VECTOR INDEX File_embedding_idx
    ON :File(embedding)
    WITH CONFIG { "dimension": 1024, "metric": "cos", "capacity": 1000000 }
    """,
    """
    CREATE VECTOR INDEX Section_embedding_idx
    ON :Section(embedding)
    WITH CONFIG { "dimension": 1024, "metric": "cos", "capacity": 1000000 }
    """,
    # Full-text index.
    "CREATE INDEX ON :File(summary)",
    "CREATE INDEX ON :Artifact(description)",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=1, name="baseline_memgraph", up=up)
