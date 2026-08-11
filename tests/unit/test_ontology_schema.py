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


def test_mintable_labels_excludes_system_and_enumeration_owned() -> None:
    onto = Ontology.load_base()
    mintable = onto.mintable_labels()
    assert mintable == frozenset(
        {
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
        }
    )


def test_inference_target_labels_adds_file_and_section_only() -> None:
    onto = Ontology.load_base()
    targets = onto.inference_target_labels()
    assert targets == onto.mintable_labels() | {"File", "Section"}
    assert "Corpus" not in targets
    assert "Meta" not in targets


def test_load_base_parses_edge_constraints() -> None:
    onto = Ontology.load_base()
    assert "REFERENCES" in onto.edge_types
    c = onto.edge_constraints["IDENTIFIES_RISK"]
    assert c.dst == frozenset({"Risk"})


def test_validate_triple_enforces_constrained_endpoints() -> None:
    onto = Ontology.load_base()
    assert onto.validate_triple("Section", "IDENTIFIES_RISK", "Risk")
    assert not onto.validate_triple("Section", "IDENTIFIES_RISK", "Technology")
    # CREATED_BY dst is Client/WorkSession only — the measured junk class.
    assert not onto.validate_triple("File", "CREATED_BY", "Technology")
    assert onto.validate_triple("File", "CREATED_BY", "WorkSession")


def test_validate_triple_wildcards_and_aliases() -> None:
    onto = Ontology.load_base().with_aliases({"GapEntry": "Risk"})
    # SIMILAR_TO / RELATED_TO are fully wildcarded.
    assert onto.validate_triple("Client", "SIMILAR_TO", "Technology")
    # Node alias on an endpoint resolves before the check.
    assert onto.validate_triple("Section", "IDENTIFIES_RISK", "GapEntry")
    # Unknown edge type is invalid outright.
    assert not onto.validate_triple("File", "HALLUCINATED", "Risk")


def test_validate_triple_unconstrained_when_constraints_absent() -> None:
    # Direct construction without edge_constraints (legacy/test shape) — every
    # declared edge type is unconstrained.
    onto = Ontology.load_base()
    bare = Ontology(
        node_types=onto.node_types,
        edge_types=onto.edge_types,
        edge_origin_values=onto.edge_origin_values,
    )
    assert bare.validate_triple("Client", "IDENTIFIES_RISK", "Technology")


def test_with_aliases_rejects_unknown_target() -> None:
    """An alias whose target isn't a real node type must fail loudly —
    otherwise a typo in a per-corpus config would silently resolve to a
    hallucinated label and bypass validation at write time."""
    onto = Ontology.load_base()
    with pytest.raises(OntologyError, match="unknown node type 'Widgt'"):
        onto.with_aliases({"FileWidget": "Widgt"})


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


# --- edge-alias tests (M10.3) ---


def test_with_edge_aliases_valid() -> None:
    onto = Ontology.load_base().with_edge_aliases({"CITES": "REFERENCES"})
    assert dict(onto.edge_aliases) == {"CITES": "REFERENCES"}
    assert onto.resolve_edge_alias("CITES") == "REFERENCES"
    assert onto.resolve_edge_alias("USES") == "USES"  # non-alias passes through
    onto.validate_edge("CITES", origin="inferred")  # must not raise


def test_with_edge_aliases_rejects_unknown_target() -> None:
    onto = Ontology.load_base()
    with pytest.raises(
        OntologyError, match="Edge alias 'CITES' targets unknown edge type 'NONEXISTENT'"
    ):
        onto.with_edge_aliases({"CITES": "NONEXISTENT"})


def test_with_edge_aliases_is_immutable_on_source() -> None:
    base = Ontology.load_base()
    _derived = base.with_edge_aliases({"CITES": "REFERENCES"})
    # Original instance must be unaffected (frozen dataclass + replace semantics)
    assert dict(base.edge_aliases) == {}


def test_with_aliases_and_with_edge_aliases_stackable() -> None:
    onto = (
        Ontology.load_base()
        .with_aliases({"Registry": "Pattern"})
        .with_edge_aliases({"CITES": "REFERENCES"})
    )
    # Both layers preserved
    assert onto.resolve_alias("Registry") == "Pattern"
    assert onto.resolve_edge_alias("CITES") == "REFERENCES"


def test_validate_edge_through_alias() -> None:
    onto = Ontology.load_base().with_edge_aliases({"CITES": "REFERENCES"})
    onto.validate_edge("CITES", origin="inferred")  # must not raise
    with pytest.raises(OntologyError, match="Unknown edge type 'NONEXISTENT'"):
        onto.validate_edge("NONEXISTENT", origin="inferred")
