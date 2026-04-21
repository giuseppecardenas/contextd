"""Corpus-property indexes for multi-corpus filter queries.

Parallel to Memgraph's _0001_baseline corpus indexes (which it bundles
into the baseline). Neo4j 5.x uses ``CREATE INDEX ... FOR (n:Label) ON
(n.corpus)`` — the bracketless ``ON :Label(prop)`` shorthand that
Memgraph accepts is not valid Neo4j syntax.

Split into its own migration rather than folded into _0001_baseline so
the baseline stays a clean mirror of the Memgraph baseline's constraint
+ vector + fulltext scope; corpus-scoped filtering is a separate
concern tied to SD #72's cross-corpus query injection.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    "CREATE INDEX File_corpus_idx IF NOT EXISTS FOR (f:File) ON (f.corpus)",
    "CREATE INDEX Section_corpus_idx IF NOT EXISTS FOR (s:Section) ON (s.corpus)",
    "CREATE INDEX Artifact_corpus_idx IF NOT EXISTS FOR (a:Artifact) ON (a.corpus)",
    "CREATE INDEX Ticket_corpus_idx IF NOT EXISTS FOR (t:Ticket) ON (t.corpus)",
    "CREATE INDEX Pattern_corpus_idx IF NOT EXISTS FOR (p:Pattern) ON (p.corpus)",
    "CREATE INDEX Risk_corpus_idx IF NOT EXISTS FOR (r:Risk) ON (r.corpus)",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=2, name="corpus_indexes_neo4j", up=up)
