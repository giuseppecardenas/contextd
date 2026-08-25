"""Schema for the retrieval-only ``Chunk`` and ``Topic`` node labels.

Chunks are sub-section retrieval units produced by the chunking phases
(``contextd/chunking/``) and hang off their parent ``Section`` / ``File`` via
``CONTAINS {origin: "structural"}``; Topics are corpus-level clusters that
``Section`` / ``File`` nodes join via ``BELONGS_TO``. Neither label is ever an
LLM inference target (see ``contextd/ontology/schema.py``), so the only
schema they need is what retrieval reads: a uniqueness constraint on ``id``,
a 1024-dimension cosine vector index on ``embedding``, a full-text index, and
btree indexes on the properties the pipeline filters and joins on
(``corpus``, ``parent_id``, ``profile``).

Index names follow the ``{Label}_{property}_idx`` / ``{Label}_{property}_ft``
convention that ``Neo4jBackend.vector_search`` / ``full_text_search``
interpolate. ``Chunk_text_ft`` is a *multi-property* full-text index over
``text``, ``prefix`` and ``keywords`` — one Lucene index so a query term
matches whichever of the three carries it — but it is addressed through the
naming convention by ``property_name="text"`` alone
(``full_text_search("Chunk", "text", ...)``); ``prefix`` and ``keywords``
have no index name of their own. Likewise ``Topic_summary_ft`` also covers
``title`` and is addressed by ``property_name="summary"``.

Idempotent: ``IF NOT EXISTS`` guards every statement so replays are safe.
"""

from typing import Any

from contextd.storage.migration import Migration

_VECTOR_OPTIONS = (
    "OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}"
)

_DDL = [
    # -- Chunk ---------------------------------------------------------------
    "CREATE CONSTRAINT Chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE VECTOR INDEX Chunk_embedding_idx IF NOT EXISTS FOR (c:Chunk) ON c.embedding "
    + _VECTOR_OPTIONS,
    "CREATE FULLTEXT INDEX Chunk_text_ft IF NOT EXISTS "
    "FOR (c:Chunk) ON EACH [c.text, c.prefix, c.keywords]",
    "CREATE INDEX Chunk_corpus_idx IF NOT EXISTS FOR (c:Chunk) ON (c.corpus)",
    "CREATE INDEX Chunk_parent_idx IF NOT EXISTS FOR (c:Chunk) ON (c.parent_id)",
    "CREATE INDEX Chunk_profile_idx IF NOT EXISTS FOR (c:Chunk) ON (c.profile)",
    # -- Topic ---------------------------------------------------------------
    "CREATE CONSTRAINT Topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
    "CREATE VECTOR INDEX Topic_embedding_idx IF NOT EXISTS FOR (t:Topic) ON t.embedding "
    + _VECTOR_OPTIONS,
    "CREATE FULLTEXT INDEX Topic_summary_ft IF NOT EXISTS FOR (t:Topic) ON EACH [t.summary, t.title]",
    "CREATE INDEX Topic_corpus_idx IF NOT EXISTS FOR (t:Topic) ON (t.corpus)",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=8, name="chunks_and_topics_neo4j", up=up)
