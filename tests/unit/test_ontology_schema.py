from pathlib import Path

import pytest

from contextd.ontology.schema import Ontology, OntologyError


def test_loads_base_ontology() -> None:
    onto = Ontology.load_base()
    assert "File" in onto.node_types
    assert "Section" in onto.node_types
    assert "CONTAINS" in onto.edge_types
    assert set(onto.edge_origin_values) == {"inferred", "structural", "manual"}


def test_validates_known_node_type() -> None:
    onto = Ontology.load_base()
    onto.validate_node("File")  # must not raise


def test_rejects_unknown_node_type() -> None:
    onto = Ontology.load_base()
    with pytest.raises(OntologyError, match="Unknown node type 'Widget'"):
        onto.validate_node("Widget")


def test_rejects_unknown_edge_type() -> None:
    onto = Ontology.load_base()
    with pytest.raises(OntologyError, match="Unknown edge type 'USED_WITH'"):
        onto.validate_edge("USED_WITH", origin="inferred")


def test_rejects_unknown_origin() -> None:
    onto = Ontology.load_base()
    with pytest.raises(OntologyError, match="Unknown edge origin 'guessed'"):
        onto.validate_edge("REFERENCES", origin="guessed")  # type: ignore[arg-type]


def test_alias_resolution(tmp_path: Path) -> None:
    onto = Ontology.load_base().with_aliases({"Registry": "Pattern", "FRRow": "Ticket"})
    assert onto.resolve_alias("Registry") == "Pattern"
    assert onto.resolve_alias("FRRow") == "Ticket"
    assert onto.resolve_alias("File") == "File"  # non-alias passes through


# --- immutability tests (SD #63) ---


def test_edge_types_is_frozenset() -> None:
    onto = Ontology.load_base()
    with pytest.raises(AttributeError):
        onto.edge_types.add("FAKE_EDGE")  # type: ignore[attr-defined]


def test_node_types_is_mapping_proxy() -> None:
    onto = Ontology.load_base()
    with pytest.raises(TypeError):
        onto.node_types["File"] = ("new",)  # type: ignore[index]


def test_read_only_access_still_works() -> None:
    onto = Ontology.load_base()
    assert "File" in onto.node_types
    assert "CONTAINS" in onto.edge_types
    assert sorted(onto.edge_types)  # sortable


def test_validate_node_no_properties_arg() -> None:
    onto = Ontology.load_base()
    onto.validate_node("File")  # one-arg form — must not raise


def test_validate_node_unknown_raises() -> None:
    onto = Ontology.load_base()
    with pytest.raises(OntologyError, match="Unknown node type 'Widget'"):
        onto.validate_node("Widget")
