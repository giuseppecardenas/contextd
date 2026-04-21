"""Baseline schema for Neo4j backend (reference Cypher implementation).

Neo4j 5.x declares vector and full-text indexes via CREATE ... INDEX DDL
with an OPTIONS map. Uniqueness is a CONSTRAINT, not an index. The schema
is schema-free at the node-table level (unlike Kuzu); nodes gain
properties dynamically.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    # Uniqueness constraints — one per label whose PK we pin.
    "CREATE CONSTRAINT File_path_unique IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE",
    "CREATE CONSTRAINT Section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT Artifact_id_unique IF NOT EXISTS FOR (a:Artifact) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT Ticket_id_unique IF NOT EXISTS FOR (t:Ticket) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT Pattern_name_unique IF NOT EXISTS FOR (p:Pattern) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT Technology_name_unique IF NOT EXISTS "
    "FOR (t:Technology) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT Client_name_unique IF NOT EXISTS FOR (c:Client) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT Repo_name_unique IF NOT EXISTS FOR (r:Repo) REQUIRE r.name IS UNIQUE",
    "CREATE CONSTRAINT Service_name_unique IF NOT EXISTS FOR (s:Service) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT Integration_name_unique IF NOT EXISTS "
    "FOR (i:Integration) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT Risk_desc_unique IF NOT EXISTS FOR (r:Risk) REQUIRE r.description IS UNIQUE",
    "CREATE CONSTRAINT WorkSession_id_unique IF NOT EXISTS "
    "FOR (w:WorkSession) REQUIRE w.id IS UNIQUE",
    "CREATE CONSTRAINT Corpus_name_unique IF NOT EXISTS FOR (c:Corpus) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT Meta_version_unique IF NOT EXISTS "
    "FOR (m:Meta) REQUIRE m.schema_version IS UNIQUE",
    # Vector indexes — Voyage-3 is 1024-dim, cosine similarity.
    "CREATE VECTOR INDEX File_embedding_idx IF NOT EXISTS "
    "FOR (f:File) ON f.embedding "
    "OPTIONS {indexConfig: {"
    "`vector.dimensions`: 1024, "
    "`vector.similarity_function`: 'cosine'}}",
    "CREATE VECTOR INDEX Section_embedding_idx IF NOT EXISTS "
    "FOR (s:Section) ON s.embedding "
    "OPTIONS {indexConfig: {"
    "`vector.dimensions`: 1024, "
    "`vector.similarity_function`: 'cosine'}}",
    # Full-text indexes — Lucene-backed, stemming default (English).
    "CREATE FULLTEXT INDEX File_summary_ft IF NOT EXISTS FOR (f:File) ON EACH [f.summary]",
    "CREATE FULLTEXT INDEX Artifact_description_ft IF NOT EXISTS "
    "FOR (a:Artifact) ON EACH [a.description]",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=1, name="baseline_neo4j", up=up)
