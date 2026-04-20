"""Baseline schema for KùzuDB backend.

Kuzu is schema-first: node tables and rel tables must be declared
before data goes in. Vector and FTS indexes are created via procedure
calls (spec §5.9 Step 3 Kuzu path).
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    # Node tables — properties enumerated per spec §3.1.
    # `embedding` must be a fixed-size ARRAY (DOUBLE[1024]) for Kuzu's vector
    # index; variable-size LIST (DOUBLE[]) is rejected by CREATE_VECTOR_INDEX.
    "CREATE NODE TABLE File(path STRING PRIMARY KEY, name STRING, type STRING, hash STRING, size INT64, updated TIMESTAMP, embedding DOUBLE[1024], summary STRING, key_points STRING[], summary_generated_at TIMESTAMP, summary_confidence DOUBLE, corpus STRING)",
    "CREATE NODE TABLE Section(id STRING PRIMARY KEY, anchor STRING, title STRING, level INT64, path STRING, corpus STRING, file_id STRING, ordinal INT64, embedding DOUBLE[1024], summary STRING, key_points STRING[], entities_mentioned STRING[], summary_generated_at TIMESTAMP, summary_confidence DOUBLE)",
    "CREATE NODE TABLE Artifact(id STRING PRIMARY KEY, title STRING, description STRING, reusable BOOL, created TIMESTAMP, updated TIMESTAMP, corpus STRING)",
    "CREATE NODE TABLE Ticket(id STRING PRIMARY KEY, title STRING, status STRING, created TIMESTAMP, updated TIMESTAMP, corpus STRING)",
    "CREATE NODE TABLE Pattern(name STRING PRIMARY KEY, description STRING, when_to_use STRING, examples STRING[], corpus STRING)",
    "CREATE NODE TABLE Technology(name STRING PRIMARY KEY, version STRING)",
    "CREATE NODE TABLE Client(name STRING PRIMARY KEY)",
    "CREATE NODE TABLE Repo(name STRING PRIMARY KEY, url STRING)",
    "CREATE NODE TABLE Service(name STRING PRIMARY KEY, repo STRING)",
    "CREATE NODE TABLE Integration(name STRING PRIMARY KEY, type STRING)",
    "CREATE NODE TABLE Risk(description STRING PRIMARY KEY, severity STRING, corpus STRING)",
    # `start` and `end` are Kuzu reserved words → backtick-escape.
    "CREATE NODE TABLE WorkSession(id STRING PRIMARY KEY, `start` TIMESTAMP, `end` TIMESTAMP, focus_area STRING)",
    "CREATE NODE TABLE Corpus(name STRING PRIMARY KEY, root STRING, registered_at TIMESTAMP, content_profile STRING)",
    # Meta is bootstrapped by KuzuBackend.connect() so the migration runner can
    # check applied versions before any migration runs.
    # Rel tables per ontology edge types. Kuzu requires explicit src/dst
    # pairing per rel table; MANY_MANY across heterogeneous endpoints.
    "CREATE REL TABLE CONTAINS(FROM File TO Section, origin STRING)",
    "CREATE REL TABLE PARENT_OF(FROM Section TO Section, origin STRING)",
    "CREATE REL TABLE NEXT_SIBLING(FROM Section TO Section, origin STRING)",
    "CREATE REL TABLE REFERENCES(FROM File TO File, FROM Section TO Section, FROM Artifact TO Artifact, origin STRING)",
    "CREATE REL TABLE BELONGS_TO(FROM File TO Ticket, FROM Artifact TO Ticket, origin STRING)",
    # Vector + FTS indexes via procedure calls. Kuzu infers dimension from the
    # column's fixed-size ARRAY type; `dim :=` is not a recognised kwarg.
    "CALL CREATE_VECTOR_INDEX('File', 'File_embedding_idx', 'embedding', metric := 'cosine')",
    "CALL CREATE_VECTOR_INDEX('Section', 'Section_embedding_idx', 'embedding', metric := 'cosine')",
    "CALL CREATE_FTS_INDEX('File', 'File_summary_ft', ['summary'])",
    "CALL CREATE_FTS_INDEX('Artifact', 'Artifact_description_ft', ['description'])",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=1, name="baseline_kuzu", up=up)
