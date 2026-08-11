"""Merge-summarize entity descriptions: accumulate at match, synthesise at 6+."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from contextd.corpus_config import CorpusConfig
from contextd.indexer.phases import RelateDeps, _apply_inferred_edge, phase_merge_descriptions
from contextd.indexer.resolution import EntityCascadeResolver, ResolutionSettings
from contextd.inference.context import EmptyRetriever
from contextd.inference.merge import DescriptionMerger
from contextd.inference.relate import InferredRelationship


def _rel(**overrides: object) -> InferredRelationship:
    kwargs: dict[str, object] = {
        "edge_type": "REFERENCES",
        "target_type": "Pattern",
        "target_name": "spatial hash",
        "confidence": 0.9,
        "reason": "r",
        "target_properties": {"description": "grid-based neighbour lookup"},
    }
    kwargs.update(overrides)
    return InferredRelationship(**kwargs)  # type: ignore[arg-type]


def test_matched_entity_accumulates_description_fragment() -> None:
    store = MagicMock()
    store.exec_read.return_value = [{"name": "spatial hash", "name_norm": "spatial hash"}]
    resolver = EntityCascadeResolver(store, ResolutionSettings())

    written = _apply_inferred_edge(
        store, "src.md", "File", _rel(), "c", resolver=resolver, settings=resolver.settings
    )

    assert written is True
    # description was diverted to the fragment append, not the direct merge.
    _, props = store.upsert_node.call_args.args
    assert "description" not in props
    frag_calls = [
        c for c in store.exec_write.call_args_list if "description_fragments" in c.args[0]
    ]
    assert len(frag_calls) == 1
    assert frag_calls[0].args[1]["frag"] == "grid-based neighbour lookup"
    # Distinct-append + hard cap encoded in the Cypher.
    assert "IN coalesce(n.description_fragments, [])" in frag_calls[0].args[0]
    assert ">= 12" in frag_calls[0].args[0]


def test_minted_entity_keeps_direct_description() -> None:
    store = MagicMock()
    store.exec_read.return_value = []
    resolver = EntityCascadeResolver(store, ResolutionSettings())

    _apply_inferred_edge(
        store, "src.md", "File", _rel(), "c", resolver=resolver, settings=resolver.settings
    )

    _, props = store.upsert_node.call_args.args
    assert props["description"] == "grid-based neighbour lookup"
    frag_calls = [
        c for c in store.exec_write.call_args_list if "description_fragments" in c.args[0]
    ]
    assert frag_calls == []


def _corpus(tmp_path: Path) -> CorpusConfig:
    return CorpusConfig.model_validate({"corpus": {"name": "c", "root": str(tmp_path)}})


def _deps(merger: DescriptionMerger | None) -> RelateDeps:
    return RelateDeps(inferrer=MagicMock(), retriever=EmptyRetriever(), merger=merger)


def test_phase_merges_entities_over_threshold(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.generate.return_value = json.dumps({"description": "one clean synthesis"})
    merger = DescriptionMerger(provider)
    store = MagicMock()

    def _exec_read(query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if ":Pattern" in query:
            return [{"name": "spatial hash", "fragments": [f"frag {i}" for i in range(6)]}]
        return []

    store.exec_read.side_effect = _exec_read

    result = phase_merge_descriptions(_corpus(tmp_path), _deps(merger), store)

    assert result.processed == 1
    write = next(c for c in store.exec_write.call_args_list if "n.description = $d" in c.args[0])
    assert write.args[1]["d"] == "one clean synthesis"
    # Fragment list resets to the single merged entry — bounded growth.
    assert "n.description_fragments = [$d]" in write.args[0]


def test_phase_skips_risk_label(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.generate.return_value = json.dumps({"description": "x"})
    store = MagicMock()
    store.exec_read.return_value = []

    phase_merge_descriptions(_corpus(tmp_path), _deps(DescriptionMerger(provider)), store)

    queried_labels = {c.args[0] for c in store.exec_read.call_args_list}
    assert not any(":Risk" in q for q in queried_labels)


def test_phase_merge_failure_keeps_fragments(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.generate.side_effect = RuntimeError("provider down")
    store = MagicMock()

    def _exec_read(query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if ":Pattern" in query:
            return [{"name": "x", "fragments": ["a"] * 6}]
        return []

    store.exec_read.side_effect = _exec_read

    result = phase_merge_descriptions(_corpus(tmp_path), _deps(DescriptionMerger(provider)), store)

    assert result.skipped >= 1
    store.exec_write.assert_not_called()


def test_phase_noop_without_merger(tmp_path: Path) -> None:
    store = MagicMock()
    result = phase_merge_descriptions(_corpus(tmp_path), _deps(None), store)
    assert result.processed == 0
    store.exec_read.assert_not_called()
