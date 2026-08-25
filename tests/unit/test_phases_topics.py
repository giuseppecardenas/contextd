from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from contextd.chunking.tokenizer import WordTokenizer
from contextd.corpus_config import CorpusConfig
from contextd.indexer.chunk_deps import ChunkingDeps
from contextd.indexer.phases_topics import input_fingerprint, phase_cluster_topics, topics_dirty
from contextd.inference.prompts import PromptRenderer
from contextd.providers.base import EmbeddingProvider, InferenceProvider, PromptRequest, UsageRecord


class FakeInference(InferenceProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: PromptRequest) -> str:
        self.calls += 1
        return json.dumps({"title": f"Topic {self.calls}", "summary": f"summary {self.calls}"})

    def last_usage(self) -> UsageRecord | None:
        return None


class FakeEmbedder(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), 1.0, 0.5] for t in texts]

    def last_usage(self) -> UsageRecord | None:
        return None

    @property
    def dimensions(self) -> int:
        return 3


def _corpus(tmp_path: Path, **topics: object) -> CorpusConfig:
    return CorpusConfig.model_validate(
        {
            "corpus": {"name": "c", "root": str(tmp_path), "granularity": "section"},
            "chunking": {"tokenizer": "words"},
            "topics": {"enabled": True, "min_members": 3, "max_layers": 2, **topics},
        }
    )


def _deps(tmp_path: Path, inference: InferenceProvider | None) -> ChunkingDeps:
    return ChunkingDeps(
        config=CorpusConfig.model_validate({"corpus": {"name": "c", "root": "."}}).chunking,
        tokenizer=WordTokenizer(),
        embedder=FakeEmbedder(),
        config_fp="fp",
        inference=inference,
        renderer=PromptRenderer(tmp_path),
    )


def _members(n_per_blob: int = 6, dim: int = 8) -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for blob in range(3):
        centre = np.eye(dim)[blob] * 5.0
        for i in range(n_per_blob):
            vec = centre + rng.normal(scale=0.1, size=dim)
            rows.append(
                {
                    "id": f"doc{blob}.md#s{i}",
                    "label": "Section",
                    "summary": f"blob {blob} section {i} summary",
                    "embedding": [float(x) for x in vec],
                }
            )
    return rows


def _store(
    members: list[dict[str, Any]], stored_fp: str | None = None, dirty: bool = False
) -> MagicMock:
    store = MagicMock()

    def _read(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if "MATCH (s:Section" in cypher:
            return members
        if "topic_input_fingerprint" in cypher:
            return [{"fp": stored_fp, "dirty": dirty}]
        return []

    store.exec_read.side_effect = _read
    return store


def test_clusters_write_topics_edges_and_fingerprint(tmp_path: Path) -> None:
    cfg = _corpus(tmp_path)
    inference = FakeInference()
    store = _store(_members())
    result = phase_cluster_topics(cfg, _deps(tmp_path, inference), store)

    assert result.processed >= 3
    store.delete_nodes.assert_called_once_with("Topic", where={"corpus": "c"})
    label, rows = store.upsert_nodes.call_args_list[0].args
    assert label == "Topic"
    assert {r["layer"] for r in rows} == {0}
    assert all(r["id"].startswith("c/topic/0/") and len(r["embedding"]) == 3 for r in rows)
    assert sum(r["member_count"] for r in rows) >= 18
    writes = [c for c in store.exec_write.call_args_list]
    edge_writes = [c for c in writes if "BELONGS_TO" in c.args[0]]
    assert edge_writes and "MATCH (m:Section {id: e.src})" in edge_writes[0].args[0]
    assert all(0.0 <= e["p"] <= 1.0 for e in edge_writes[0].args[1]["edges"])
    final = writes[-1]
    assert "topics_dirty = false" in final.args[0] and final.args[1]["n"] == result.processed
    assert inference.calls == result.processed


def test_skips_when_fingerprint_matches_and_not_dirty(tmp_path: Path) -> None:
    cfg = _corpus(tmp_path)
    members = _members()
    # Compute the fingerprint the phase would derive.
    from contextd.indexer.phases_topics import _load_members

    deps = _deps(tmp_path, FakeInference())
    fp = input_fingerprint(_load_members(_store(members), cfg, deps), cfg)
    store = _store(members, stored_fp=fp)
    result = phase_cluster_topics(cfg, deps, store)
    assert result.processed == 0 and result.skipped == 18
    store.delete_nodes.assert_not_called()

    dirty_store = _store(members, stored_fp=fp, dirty=True)
    assert phase_cluster_topics(cfg, deps, dirty_store).processed >= 3


def test_force_ignores_fingerprint(tmp_path: Path) -> None:
    cfg = _corpus(tmp_path)
    members = _members()
    from contextd.indexer.phases_topics import _load_members

    deps = _deps(tmp_path, FakeInference())
    fp = input_fingerprint(_load_members(_store(members), cfg, deps), cfg)
    assert phase_cluster_topics(cfg, deps, _store(members, stored_fp=fp), force=True).processed >= 3


def test_too_few_members_or_disabled_is_noop(tmp_path: Path) -> None:
    cfg = _corpus(tmp_path)
    store = _store(_members(n_per_blob=1)[:2])
    assert phase_cluster_topics(cfg, _deps(tmp_path, FakeInference()), store).processed == 0
    store.delete_nodes.assert_not_called()
    cfg.topics.enabled = False
    assert (
        phase_cluster_topics(cfg, _deps(tmp_path, FakeInference()), _store(_members())).processed
        == 0
    )


def test_missing_inference_provider_skips(tmp_path: Path) -> None:
    cfg = _corpus(tmp_path)
    result = phase_cluster_topics(cfg, _deps(tmp_path, None), _store(_members()))
    assert result.processed == 0 and result.skipped == 1


def test_fingerprint_changes_with_summaries_and_config(tmp_path: Path) -> None:
    from contextd.indexer.phases_topics import _Member

    members = [_Member("a", "Section", "s1", [1.0], 1), _Member("b", "Section", "s2", [1.0], 1)]
    cfg = _corpus(tmp_path)
    base = input_fingerprint(members, cfg)
    assert base == input_fingerprint(list(members), cfg)
    members[0].summary = "changed"
    assert input_fingerprint(members, cfg) != base
    cfg2 = _corpus(tmp_path, max_layers=1)
    assert input_fingerprint(members, cfg2) != input_fingerprint(members, cfg)


def test_topics_dirty_reads_corpus_flag() -> None:
    store = MagicMock()
    store.exec_read.return_value = [{"fp": "x", "dirty": True}]
    assert topics_dirty(store, "c") is True
    store.exec_read.return_value = []
    assert topics_dirty(store, "c") is False
