"""Unit tests for contextd.storage._keys."""

from __future__ import annotations

import pytest

from contextd.storage._keys import (
    IMMUTABLE_AFTER_CREATE_BY_LABEL,
    PRIMARY_KEY_BY_LABEL,
    immutable_after_create_for,
    primary_key_for,
)


def test_primary_key_for_known_labels() -> None:
    # Spot-check a handful; the drift test in test_keys_drift.py asserts the
    # full map matches the Kuzu DDL.
    assert primary_key_for("File") == "path"
    assert primary_key_for("Section") == "id"
    assert primary_key_for("Pattern") == "name"
    assert primary_key_for("Meta") == "schema_version"


def test_primary_key_for_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="Unknown node label 'DoesNotExist'"):
        primary_key_for("DoesNotExist")


def test_immutable_after_create_for_known_labels() -> None:
    assert immutable_after_create_for("File") == frozenset({"embedding"})
    assert immutable_after_create_for("Section") == frozenset({"embedding"})


def test_immutable_after_create_for_unknown_label_returns_empty() -> None:
    # Unknown labels don't raise here — the map is additive; absence means "no
    # columns are immutable-after-create for this label". The drift test
    # catches real additions to Kuzu that should land in this map.
    assert immutable_after_create_for("DoesNotExist") == frozenset()


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


def test_immutable_map_values_are_frozensets() -> None:
    # Guards the type — a plain set would allow mutation of the module-level map.
    for label, props in IMMUTABLE_AFTER_CREATE_BY_LABEL.items():
        assert isinstance(props, frozenset), (
            f"IMMUTABLE_AFTER_CREATE_BY_LABEL[{label!r}] is {type(props).__name__}, "
            f"expected frozenset"
        )
