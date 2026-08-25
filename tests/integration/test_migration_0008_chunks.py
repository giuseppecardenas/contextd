"""Migration _0008 creates the Chunk / Topic schema: uniqueness constraints on
``id``, 1024-dim cosine vector indexes, full-text indexes and the btree
indexes the chunking pipeline filters on. ``Chunk_text_ft`` is multi-property
(text, prefix, keywords) but reached through ``property_name="text"``.

Exercised against the Neo4j ``backend`` fixture, which applies ALL_MIGRATIONS.
"""

from __future__ import annotations

import pytest

from contextd.storage.base import GraphStore

pytestmark = pytest.mark.integration


def _chunk(corpus: str, ordinal: int, text: str, **extra: object) -> dict[str, object]:
    parent = f"docs/{corpus}.md#s"
    return {
        "id": f"{parent}~fine~{ordinal}",
        "corpus": corpus,
        "path": f"docs/{corpus}.md",
        "parent_id": parent,
        "parent_label": "Section",
        "profile": "fine",
        "strategy": "structural",
        "ordinal": ordinal,
        "kind": "prose",
        "part": 0,
        "text": text,
        "token_count": len(text.split()),
        **extra,
    }


def test_chunk_and_topic_schema_exists(backend: GraphStore) -> None:
    rows = backend.exec_read("SHOW INDEXES YIELD name RETURN name", None)
    names = {r["name"] for r in rows}
    for expected in (
        "Chunk_embedding_idx",
        "Chunk_text_ft",
        "Chunk_corpus_idx",
        "Chunk_parent_idx",
        "Chunk_profile_idx",
        "Topic_embedding_idx",
        "Topic_summary_ft",
        "Topic_corpus_idx",
    ):
        assert expected in names, f"missing index {expected}; have {sorted(names)}"

    rows = backend.exec_read("SHOW CONSTRAINTS YIELD name RETURN name", None)
    constraints = {r["name"] for r in rows}
    assert "Chunk_id_unique" in constraints
    assert "Topic_id_unique" in constraints


def test_vector_indexes_are_1024_cosine(backend: GraphStore) -> None:
    rows = backend.exec_read(
        "SHOW VECTOR INDEXES YIELD name, options "
        "WHERE name IN ['Chunk_embedding_idx', 'Topic_embedding_idx'] "
        "RETURN name, options",
        None,
    )
    assert {r["name"] for r in rows} == {"Chunk_embedding_idx", "Topic_embedding_idx"}
    for r in rows:
        cfg = r["options"]["indexConfig"]
        assert cfg["vector.dimensions"] == 1024
        assert cfg["vector.similarity_function"].lower() == "cosine"


def test_chunk_text_ft_covers_prefix_and_keywords(backend: GraphStore) -> None:
    """The multi-property index is addressed by ``property_name="text"`` and a
    term present only in ``prefix`` or ``keywords`` still matches."""
    backend.upsert_nodes(
        "Chunk",
        [
            _chunk("m8ft", 0, "plain body words", prefix="zebrafish context"),
            _chunk("m8ft", 1, "other body words", keywords="quokka"),
            _chunk("m8ft", 2, "unrelated"),
        ],
    )
    by_prefix = backend.full_text_search("Chunk", "text", "zebrafish", k=5)
    assert [r["node"]["ordinal"] for r in by_prefix] == [0]
    by_keyword = backend.full_text_search("Chunk", "text", "quokka", k=5)
    assert [r["node"]["ordinal"] for r in by_keyword] == [1]


def test_chunk_vector_search_filtered_by_corpus(backend: GraphStore) -> None:
    vec = [1.0] + [0.0] * 1023
    backend.upsert_nodes(
        "Chunk",
        [
            _chunk("m8a", 0, "a", embedding=vec),
            _chunk("m8b", 0, "b", embedding=vec),
        ],
    )
    hits = backend.vector_search("Chunk", "embedding", vec, k=5, filters={"corpus": "m8b"})
    assert [r["node"]["corpus"] for r in hits] == ["m8b"]
    assert hits[0]["score"] > 0.99


def test_topic_upsert_and_search(backend: GraphStore) -> None:
    vec = [0.0, 1.0] + [0.0] * 1022
    backend.upsert_nodes(
        "Topic",
        [
            {
                "id": "m8t/topic/0/0",
                "corpus": "m8t",
                "layer": 0,
                "title": "Retry policy",
                "summary": "Backoff and jitter for transient failures.",
                "embedding": vec,
                "member_count": 3,
            }
        ],
    )
    by_title = backend.full_text_search("Topic", "summary", "retry", k=5, filters={"corpus": "m8t"})
    assert [r["node"]["id"] for r in by_title] == ["m8t/topic/0/0"]
    by_vec = backend.vector_search("Topic", "embedding", vec, k=1, filters={"corpus": "m8t"})
    assert by_vec[0]["node"]["title"] == "Retry policy"


def test_migration_0008_is_idempotent(backend: GraphStore) -> None:
    from contextd.migrations.neo4j import ALL_MIGRATIONS

    m = next(m for m in ALL_MIGRATIONS if m.id == 8)
    assert m.name == "chunks_and_topics_neo4j"
    # The backend fixture already applied it once; a replay must not raise.
    m.up(backend, m.id)
    rows = backend.exec_read(
        "SHOW INDEXES YIELD name WHERE name STARTS WITH 'Chunk_' RETURN name", None
    )
    # Five declared indexes plus the backing index of the uniqueness
    # constraint; a replay must not have created duplicates under other names.
    assert {r["name"] for r in rows} == {
        "Chunk_id_unique",
        "Chunk_embedding_idx",
        "Chunk_text_ft",
        "Chunk_corpus_idx",
        "Chunk_parent_idx",
        "Chunk_profile_idx",
    }
