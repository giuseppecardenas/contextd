"""Unit tests for contextd.ontology.overrides.apply_overrides."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextd.ontology.overrides import OntologyOverridesError, apply_overrides
from contextd.ontology.schema import Ontology, OntologyError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base() -> Ontology:
    return Ontology.load_base()


def _write_json(tmp_path: Path, name: str, data: object) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_apply_overrides_adds_edge_aliases(tmp_path: Path) -> None:
    overrides_file = _write_json(
        tmp_path, "overrides.json", {"edge_label_aliases": {"CITES": "REFERENCES"}}
    )
    result = apply_overrides(_base(), overrides_file)
    assert dict(result.edge_aliases) == {"CITES": "REFERENCES"}
    assert result.resolve_edge_alias("CITES") == "REFERENCES"


def test_apply_overrides_missing_edge_label_aliases_is_noop(tmp_path: Path) -> None:
    """Top-level JSON object with no 'edge_label_aliases' key → unchanged Ontology."""
    overrides_file = _write_json(tmp_path, "overrides.json", {})
    base = _base()
    result = apply_overrides(base, overrides_file)
    # identity check: same frozenset, same mapping proxy
    assert result.edge_types == base.edge_types
    assert dict(result.edge_aliases) == {}


def test_apply_overrides_empty_edge_label_aliases_is_noop(tmp_path: Path) -> None:
    """edge_label_aliases present but empty → unchanged Ontology."""
    overrides_file = _write_json(tmp_path, "overrides.json", {"edge_label_aliases": {}})
    base = _base()
    result = apply_overrides(base, overrides_file)
    assert dict(result.edge_aliases) == {}


def test_apply_overrides_unknown_top_level_keys_ignored(tmp_path: Path) -> None:
    """Forward-compat: unknown top-level keys are silently ignored."""
    overrides_file = _write_json(
        tmp_path,
        "overrides.json",
        {"edge_label_aliases": {"CITES": "REFERENCES"}, "future_key": "ignored"},
    )
    result = apply_overrides(_base(), overrides_file)
    assert dict(result.edge_aliases) == {"CITES": "REFERENCES"}


def test_apply_overrides_multiple_aliases(tmp_path: Path) -> None:
    """Multiple aliases from the Acme example are all applied."""
    aliases = {
        "CITES": "REFERENCES",
        "CONSUMES": "USES",
        "SCHEMA_FOR": "DOCUMENTS",
        "REGISTERS": "DOCUMENTED_IN",
        "CLOSES_GAP": "SUPERSEDES",
    }
    overrides_file = _write_json(tmp_path, "overrides.json", {"edge_label_aliases": aliases})
    result = apply_overrides(_base(), overrides_file)
    assert result.resolve_edge_alias("CITES") == "REFERENCES"
    assert result.resolve_edge_alias("REGISTERS") == "DOCUMENTED_IN"
    assert result.resolve_edge_alias("CLOSES_GAP") == "SUPERSEDES"


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


def test_apply_overrides_requires_absolute_path(tmp_path: Path) -> None:
    relative = Path("some/relative/overrides.json")
    with pytest.raises(OntologyOverridesError, match="overrides path must be absolute"):
        apply_overrides(_base(), relative)


def test_apply_overrides_missing_file(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist.json"
    with pytest.raises(OntologyOverridesError, match="overrides file not readable"):
        apply_overrides(_base(), nonexistent)


def test_apply_overrides_invalid_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(OntologyOverridesError, match="overrides file is not valid JSON"):
        apply_overrides(_base(), bad_file)


def test_apply_overrides_non_object_top_level_list(tmp_path: Path) -> None:
    """JSON array at top level → OntologyOverridesError."""
    arr_file = _write_json(tmp_path, "overrides.json", [{"edge_label_aliases": {}}])
    with pytest.raises(
        OntologyOverridesError, match="overrides file top level must be a JSON object"
    ):
        apply_overrides(_base(), arr_file)


def test_apply_overrides_non_object_top_level_string(tmp_path: Path) -> None:
    """JSON string at top level → OntologyOverridesError."""
    str_file = tmp_path / "overrides.json"
    str_file.write_text('"just a string"', encoding="utf-8")
    with pytest.raises(
        OntologyOverridesError, match="overrides file top level must be a JSON object"
    ):
        apply_overrides(_base(), str_file)


def test_apply_overrides_malformed_edge_label_aliases_type(tmp_path: Path) -> None:
    """edge_label_aliases is a string, not a dict → OntologyOverridesError."""
    overrides_file = _write_json(tmp_path, "overrides.json", {"edge_label_aliases": "not a dict"})
    with pytest.raises(OntologyOverridesError, match="'edge_label_aliases' must be an object"):
        apply_overrides(_base(), overrides_file)


def test_apply_overrides_malformed_edge_label_aliases_value_not_string(tmp_path: Path) -> None:
    """edge_label_aliases value is an int → OntologyOverridesError."""
    overrides_file = _write_json(tmp_path, "overrides.json", {"edge_label_aliases": {"CITES": 42}})
    with pytest.raises(
        OntologyOverridesError, match="'edge_label_aliases' keys and values must be strings"
    ):
        apply_overrides(_base(), overrides_file)


def test_apply_overrides_unknown_edge_target_surfaces_ontology_error(tmp_path: Path) -> None:
    """An alias pointing at an unknown canonical edge type raises OntologyError
    (not OntologyOverridesError) — it surfaces from with_edge_aliases as-is."""
    overrides_file = _write_json(
        tmp_path, "overrides.json", {"edge_label_aliases": {"CITES": "NONEXISTENT"}}
    )
    with pytest.raises(OntologyError, match="Edge alias 'CITES' targets unknown edge type"):
        apply_overrides(_base(), overrides_file)


def test_apply_overrides_non_utf8_file(tmp_path: Path) -> None:
    """Non-UTF-8 overrides file raises OntologyOverridesError with UTF-8 naming.

    The error must not be a raw UnicodeDecodeError — CLI wrapping depends
    on our error class. Message names 'UTF-8' so users can disambiguate
    from the 'not readable' (OSError) path.
    """
    path = tmp_path / "overrides.json"
    # bytes that are not valid UTF-8: incomplete multi-byte sequence,
    # then a stray continuation byte.
    path.write_bytes(b'\xff\xfe{"edge_label_aliases": {}}')
    with pytest.raises(OntologyOverridesError, match="UTF-8"):
        apply_overrides(Ontology.load_base(), path)
