"""Unit tests for contextd.storage._keys."""

from __future__ import annotations

import pytest

from contextd.storage._keys import (
    PRIMARY_KEY_BY_LABEL,
    primary_key_for,
)


def test_primary_key_for_known_labels() -> None:
    assert primary_key_for("File") == "path"
    assert primary_key_for("Section") == "id"
    assert primary_key_for("Pattern") == "name"
    assert primary_key_for("Meta") == "schema_version"


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
    }
    missing = core - set(PRIMARY_KEY_BY_LABEL)
    assert not missing, f"PRIMARY_KEY_BY_LABEL missing core labels: {missing}"
