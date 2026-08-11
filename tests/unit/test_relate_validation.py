"""Dropped relate rows/edges must be logged and parse defects must not crash.

Before this suite existed, every validation gate in ``RelationshipInferrer``
and ``_apply_inferred_edge`` was a silent ``continue``/``return False`` — the
pipeline was undiagnosable by design. And a string confidence (``"high"``) or
dict reason escaped ``infer()`` entirely: the former lost the whole unit's
edge batch, the latter reached Neo4j as a map property and aborted the run.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from contextd.indexer.phases import _apply_inferred_edge
from contextd.inference.relate import (
    InferredRelationship,
    RelationshipInferrer,
    _coerce_confidence,
    _coerce_reason,
)
from contextd.ontology.schema import Ontology


def _inferrer(rows: list[object]) -> RelationshipInferrer:
    provider = MagicMock()
    provider.generate.return_value = json.dumps({"relationships": rows})
    renderer = MagicMock()
    renderer.render.return_value = "prompt"
    return RelationshipInferrer(provider=provider, renderer=renderer, ontology=Ontology.load_base())


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "type": "REFERENCES",
        "target_type": "Pattern",
        "target_name": "spatial hash",
        "confidence": 0.9,
        "reason": "r",
    }
    row.update(overrides)
    return row


# --- confidence / reason coercion -------------------------------------------


def test_word_confidence_maps_to_documented_anchor() -> None:
    rels = _inferrer([_row(confidence="high")]).infer("content")
    assert len(rels) == 1
    assert rels[0].confidence == 0.9


def test_numeric_string_confidence_parses() -> None:
    rels = _inferrer([_row(confidence="0.75")]).infer("content")
    assert rels[0].confidence == 0.75


def test_garbage_confidence_becomes_zero_not_crash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="contextd.inference.relate"):
        rels = _inferrer([_row(confidence="certainly")]).infer("content")
    assert rels[0].confidence == 0.0
    assert "unparseable confidence" in caplog.text


def test_confidence_clamped_to_unit_interval() -> None:
    assert _coerce_confidence(3.7) == 1.0
    assert _coerce_confidence(-1) == 0.0


def test_dict_reason_coerced_to_string(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="contextd.inference.relate"):
        rels = _inferrer([_row(reason={"because": "x"})]).infer("content")
    assert isinstance(rels[0].reason, str)
    assert "non-string reason" in caplog.text


def test_reason_truncated() -> None:
    assert len(_coerce_reason("x" * 2000)) == 500


# --- drop logging in infer() -------------------------------------------------


@pytest.mark.parametrize(
    ("row", "expected_fragment"),
    [
        ("not a dict", "non-dict row"),
        (_row(type=42), "non-string edge type"),
        (_row(type="CONTAINS"), "not emittable"),
        (_row(type="LOOKS_LIKE"), "not emittable"),
        (_row(target_type="Wormhole"), "not an inference target"),
        (_row(target_type="Corpus"), "not an inference target"),
        (_row(target_type="Meta"), "not an inference target"),
        (_row(target_name=""), "empty or non-string target name"),
    ],
)
def test_each_drop_path_logs_reason(
    row: object, expected_fragment: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="contextd.inference.relate"):
        rels = _inferrer([row]).infer("content")
    assert rels == []
    assert "relate drop" in caplog.text
    assert expected_fragment in caplog.text


# --- gleaning -----------------------------------------------------------------


def _gleaning_inferrer(responses: list[str], rounds: int) -> tuple[RelationshipInferrer, MagicMock]:
    provider = MagicMock()
    provider.generate.side_effect = responses
    renderer = MagicMock()
    renderer.render.return_value = "prompt"
    inferrer = RelationshipInferrer(
        provider=provider,
        renderer=renderer,
        ontology=Ontology.load_base(),
        gleaning_rounds=rounds,
    )
    return inferrer, renderer


def test_gleaning_round_adds_novel_rows_and_dedupes_reemitted() -> None:
    first = json.dumps({"relationships": [_row(target_name="alpha")]})
    glean = json.dumps({"relationships": [_row(target_name="alpha"), _row(target_name="beta")]})
    inferrer, renderer = _gleaning_inferrer([first, glean], rounds=1)

    rels = inferrer.infer("content")

    assert [r.target_name for r in rels] == ["alpha", "beta"]
    glean_call = renderer.render.call_args_list[1]
    assert glean_call.args == ("relate_glean",)
    assert "alpha" in glean_call.kwargs["previous_relationships"]


def test_gleaning_zero_rounds_is_single_call() -> None:
    first = json.dumps({"relationships": [_row()]})
    inferrer, renderer = _gleaning_inferrer([first], rounds=0)
    rels = inferrer.infer("content")
    assert len(rels) == 1
    assert renderer.render.call_count == 1


def test_gleaning_early_exits_when_round_adds_nothing() -> None:
    first = json.dumps({"relationships": [_row(target_name="alpha")]})
    empty_glean = json.dumps({"relationships": [_row(target_name="alpha")]})
    inferrer, renderer = _gleaning_inferrer([first, empty_glean, "UNUSED"], rounds=3)
    rels = inferrer.infer("content")
    assert len(rels) == 1
    assert renderer.render.call_count == 2  # initial + one glean, then early exit


def test_gleaning_failure_keeps_first_pass(caplog: pytest.LogCaptureFixture) -> None:
    first = json.dumps({"relationships": [_row(target_name="alpha")]})
    inferrer, _ = _gleaning_inferrer([first, "not json at all {{{", "also bad"], rounds=2)
    with caplog.at_level(logging.WARNING, logger="contextd.inference.relate"):
        rels = inferrer.infer("content")
    assert [r.target_name for r in rels] == ["alpha"]
    assert "glean round 1 failed" in caplog.text


def test_gleaning_dedupe_is_case_sensitive() -> None:
    first = json.dumps({"relationships": [_row(target_name="Alpha")]})
    glean = json.dumps({"relationships": [_row(target_name="alpha")]})
    inferrer, _ = _gleaning_inferrer([first, glean], rounds=1)
    rels = inferrer.infer("content")
    # Case variants both survive here; casefold merging is the cascade's job.
    assert [r.target_name for r in rels] == ["Alpha", "alpha"]


# --- guards in _apply_inferred_edge ------------------------------------------


def _rel(**overrides: object) -> InferredRelationship:
    kwargs: dict[str, object] = {
        "edge_type": "REFERENCES",
        "target_type": "Pattern",
        "target_name": "spatial hash",
        "confidence": 0.9,
        "reason": "r",
    }
    kwargs.update(overrides)
    return InferredRelationship(**kwargs)  # type: ignore[arg-type]


def test_confidence_below_floor_dropped(caplog: pytest.LogCaptureFixture) -> None:
    store = MagicMock()
    with caplog.at_level(logging.INFO, logger="contextd.indexer.phases"):
        written = _apply_inferred_edge(store, "src.md", "File", _rel(confidence=0.3), "c")
    assert written is False
    assert "below floor" in caplog.text
    store.upsert_edge.assert_not_called()


def test_self_loop_dropped(caplog: pytest.LogCaptureFixture) -> None:
    store = MagicMock()
    # Resolution returns the source itself.
    store.exec_read.return_value = [{"v": "docs/a.md"}]
    rel = _rel(target_type="File", target_name="docs/a.md")
    with caplog.at_level(logging.INFO, logger="contextd.indexer.phases"):
        written = _apply_inferred_edge(store, "docs/a.md", "File", rel, "c")
    assert written is False
    assert "self-loop" in caplog.text
    store.upsert_edge.assert_not_called()


def test_alias_target_type_resolves_to_canonical_label() -> None:
    provider = MagicMock()
    provider.generate.return_value = json.dumps({"relationships": [_row(target_type="Registry")]})
    renderer = MagicMock()
    renderer.render.return_value = "prompt"
    ontology = Ontology.load_base().with_aliases({"Registry": "Pattern"})
    inferrer = RelationshipInferrer(provider=provider, renderer=renderer, ontology=ontology)

    rels = inferrer.infer("content")

    assert len(rels) == 1
    assert rels[0].target_type == "Pattern"


def test_prompt_advertises_aliases_and_withholds_system_labels() -> None:
    provider = MagicMock()
    provider.generate.return_value = json.dumps({"relationships": []})
    renderer = MagicMock()
    renderer.render.return_value = "prompt"
    ontology = Ontology.load_base().with_aliases({"Registry": "Pattern"})
    RelationshipInferrer(provider=provider, renderer=renderer, ontology=ontology).infer("content")

    advertised = renderer.render.call_args.kwargs["allowed_node_types"]
    assert "Registry" in advertised
    assert "Pattern" in advertised
    assert "File" in advertised and "Section" in advertised
    assert "Corpus" not in advertised
    assert "Meta" not in advertised


def test_apply_rejects_system_label_backstop(caplog: pytest.LogCaptureFixture) -> None:
    store = MagicMock()
    with caplog.at_level(logging.INFO, logger="contextd.indexer.phases"):
        written = _apply_inferred_edge(
            store, "src.md", "File", _rel(target_type="Corpus", target_name="junk"), "c"
        )
    assert written is False
    assert "not an inference target" in caplog.text
    store.upsert_node.assert_not_called()


def test_disallowed_triple_dropped(caplog: pytest.LogCaptureFixture) -> None:
    store = MagicMock()
    rel = _rel(edge_type="IDENTIFIES_RISK", target_type="Technology", target_name="Lua")
    with caplog.at_level(logging.INFO, logger="contextd.indexer.phases"):
        written = _apply_inferred_edge(store, "src.md", "File", rel, "c")
    assert written is False
    assert "triple not allowed" in caplog.text
    store.upsert_node.assert_not_called()


def test_unknown_target_label_logged(caplog: pytest.LogCaptureFixture) -> None:
    store = MagicMock()
    with caplog.at_level(logging.INFO, logger="contextd.indexer.phases"):
        written = _apply_inferred_edge(store, "src.md", "File", _rel(target_type="Wormhole"), "c")
    assert written is False
    assert "unknown target label" in caplog.text


def test_unresolved_file_target_logged(caplog: pytest.LogCaptureFixture) -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    rel = _rel(target_type="File", target_name="ghost.md")
    with caplog.at_level(logging.INFO, logger="contextd.indexer.phases"):
        written = _apply_inferred_edge(store, "src.md", "File", rel, "c")
    assert written is False
    assert "did not resolve" in caplog.text
