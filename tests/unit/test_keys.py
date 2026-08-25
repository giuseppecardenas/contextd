"""Unit tests for contextd.storage._keys."""

from __future__ import annotations

import re

import pytest

from contextd.migrations.neo4j import _0001_baseline, _0008_chunks_and_topics
from contextd.storage._keys import (
    PRIMARY_KEY_BY_LABEL,
    primary_key_for,
)

_UNIQUE_RE = re.compile(r"FOR \(\w+:(\w+)\) REQUIRE \w+\.(\w+) IS UNIQUE")


def test_primary_key_for_known_labels() -> None:
    assert primary_key_for("File") == "path"
    assert primary_key_for("Section") == "id"
    assert primary_key_for("Pattern") == "name"
    assert primary_key_for("Meta") == "schema_version"
    assert primary_key_for("Chunk") == "id"
    assert primary_key_for("Topic") == "id"


def test_primary_key_map_mirrors_uniqueness_constraints() -> None:
    """Every label with a uniqueness constraint in the migrations that declare
    node tables (_0001, _0008) must map to exactly that property, so MERGEs
    and the constraints agree on the key."""
    declared: dict[str, str] = {}
    for stmt in [*_0001_baseline._DDL, *_0008_chunks_and_topics._DDL]:
        m = _UNIQUE_RE.search(stmt)
        if m:
            declared[m.group(1)] = m.group(2)
    assert declared, "regex matched no constraint statements"
    assert {"Chunk", "Topic"} <= set(declared)
    for label, prop in declared.items():
        assert PRIMARY_KEY_BY_LABEL.get(label) == prop, (
            f"{label}: constraint keys on {prop!r} but PRIMARY_KEY_BY_LABEL says "
            f"{PRIMARY_KEY_BY_LABEL.get(label)!r}"
        )


def test_primary_key_for_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="Unknown node label 'DoesNotExist'"):
        primary_key_for("DoesNotExist")


def test_primary_key_map_contains_all_core_labels() -> None:
    # Ontology core types per design §3.1 — ensure none dropped by accident.
    core = {
        "File",
        "Section",
        "Artifact",
        "Ticket",
        "Pattern",
        "Technology",
        "Client",
        "Repo",
        "Service",
        "Integration",
        "Risk",
        "WorkSession",
        "Corpus",
        "Meta",
        # Retrieval-only labels added by migration _0008.
        "Chunk",
        "Topic",
    }
    missing = core - set(PRIMARY_KEY_BY_LABEL)
    assert not missing, f"PRIMARY_KEY_BY_LABEL missing core labels: {missing}"
