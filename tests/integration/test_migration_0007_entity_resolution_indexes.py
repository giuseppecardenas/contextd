"""Migration _0007 creates name_norm btree + embedding vector indexes for
every mintable entity label, unblocking the resolution cascade's exact-norm
and embedding rungs (vector_search raised on entity labels before this —
no ``{Label}_embedding_idx`` existed).

Exercised against the Neo4j `backend` fixture, which applies ALL_MIGRATIONS.
"""

from __future__ import annotations

import pytest

from contextd.storage.base import GraphStore

pytestmark = pytest.mark.integration


def test_entity_resolution_indexes_exist(backend: GraphStore) -> None:
    rows = backend.exec_read("SHOW INDEXES YIELD name RETURN name", None)
    names = {r["name"] for r in rows}
    assert "Pattern_name_norm_idx" in names
    assert "Pattern_embedding_idx" in names
    assert "WorkSession_embedding_idx" in names
    assert "Risk_name_norm_idx" in names


def test_vector_search_reaches_entity_label(backend: GraphStore) -> None:
    vec = [0.1] * 1024
    backend.upsert_node(
        "Pattern",
        {
            "name": "spatial hash m7",
            "corpus": "m7_t",
            "name_norm": "spatial hash m7",
            "embedding": vec,
        },
    )
    results = backend.vector_search(label="Pattern", property_name="embedding", query=vec, k=1)
    assert results
    assert results[0]["node"]["name"] == "spatial hash m7"
    # Identical vector → maximum normalized-cosine score.
    assert results[0]["score"] > 0.99


def test_migration_0007_is_idempotent(backend: GraphStore) -> None:
    from contextd.migrations.neo4j import ALL_MIGRATIONS

    m = next(m for m in ALL_MIGRATIONS if m.id == 7)
    # The backend fixture already applied it once; a replay must not raise.
    m.up(backend, m.id)
