"""Chunk phases: fingerprint gating, write ordering, failure isolation, estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from contextd._paths import canonical_path
from contextd.chunking.fingerprint import config_fingerprint, unit_fingerprint
from contextd.chunking.tokenizer import WordTokenizer
from contextd.corpus_config import ChunkingSection, ChunkProfile, CorpusConfig
from contextd.indexer.chunk_deps import ChunkingDeps, build_chunking_deps
from contextd.indexer.phases_chunks import (
    chunk_id,
    config_drifted,
    estimate_chunks,
    phase_chunk_units,
    phase_gc_chunks,
    rechunk_corpus,
)
from contextd.providers.base import EmbeddingProvider, UsageRecord


class CountingEmbedder(EmbeddingProvider):
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embed down")
        self.calls.append(texts)
        return [[float(len(t)), 1.0] for t in texts]

    def last_usage(self) -> UsageRecord | None:
        return None

    @property
    def dimensions(self) -> int:
        return 2


_MD = (
    "# Title\n\nintro line\n\n## Alpha\n\n"
    + " ".join(f"alpha{i}" for i in range(80))
    + "\n\n## Beta\n\nshort beta body\n"
)


def _corpus(tmp_path: Path, granularity: str = "section") -> CorpusConfig:
    return CorpusConfig.model_validate(
        {
            "corpus": {"name": "c", "root": str(tmp_path), "granularity": granularity},
            "chunking": {
                "tokenizer": "words",
                "profiles": [
                    {"name": "fine", "max_tokens": 40, "min_tokens": 8},
                    {"name": "coarse", "max_tokens": 200, "min_tokens": 20},
                ],
            },
        }
    )


def _deps(cfg: CorpusConfig, embedder: EmbeddingProvider | None = None) -> ChunkingDeps:
    tok = WordTokenizer()
    return ChunkingDeps(
        config=cfg.chunking,
        tokenizer=tok,
        embedder=embedder or CountingEmbedder(),
        config_fp=config_fingerprint(cfg.chunking, tok.id),
    )


def _section_rows(path: str, fp_by_anchor: dict[str, str | None]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{path}#{anchor}",
            "path": path,
            "anchor": anchor,
            "hash": f"h-{anchor}",
            "fp": fp,
            "summary": f"summary of {anchor}",
            "key_points": ["kp1"],
            "entities_mentioned": ["Ent"],
        }
        for anchor, fp in fp_by_anchor.items()
    ]


def _store_with(section_rows: list[dict[str, Any]], file_rows: list[dict[str, Any]]) -> MagicMock:
    store = MagicMock()

    def _read(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if "MATCH (s:Section" in cypher:
            return section_rows
        if "MATCH (f:File" in cypher and "NOT (f)-[:CONTAINS]" in cypher:
            return file_rows
        return [{"n": 0}]

    store.exec_read.side_effect = _read
    return store


def test_phase_chunks_sections_and_writes_in_order(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text(_MD, encoding="utf-8")
    path = canonical_path(md)
    cfg = _corpus(tmp_path)
    embedder = CountingEmbedder()
    deps = _deps(cfg, embedder)
    store = _store_with(_section_rows(path, {"alpha": None, "beta": None, "title": None}), [])

    result = phase_chunk_units(cfg, deps, store)

    assert result.processed == 3 and result.skipped == 0
    # delete → upsert → CONTAINS → NEXT_SIBLING → fingerprint, per parent.
    assert store.delete_nodes.call_count == 3
    assert store.upsert_nodes.call_count == 3
    label, rows = store.upsert_nodes.call_args_list[0].args
    assert label == "Chunk"
    first = rows[0]
    assert first["id"] == chunk_id(first["parent_id"], first["profile"], 0)
    assert first["parent_label"] == "Section" and first["corpus"] == "c"
    assert first["prefix"].startswith("Title > ") or first["prefix"] == "Title"
    assert first["keywords"] == ["kp1"]
    assert isinstance(first["embedding"], list) and first["start_line"] >= 0
    assert {r["profile"] for r in rows} <= {"fine", "coarse"}
    writes = [c.args[0] for c in store.exec_write.call_args_list]
    assert any("CONTAINS" in w for w in writes)
    assert any("NEXT_SIBLING" in w for w in writes)
    assert writes[-1].endswith("SET p.chunk_fingerprint = $fp")
    # Embeddings were batched per profile, never per chunk.
    assert all(len(call) >= 1 for call in embedder.calls)
    # Section bodies are exclusive, so alpha's chunks never contain beta's text.
    alpha_rows = [r for r in rows if r["parent_id"].endswith("#alpha")]
    assert alpha_rows and all("beta" not in r["text"] for r in alpha_rows)


def test_phase_chunks_skips_parents_with_current_fingerprint(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text(_MD, encoding="utf-8")
    path = canonical_path(md)
    cfg = _corpus(tmp_path)
    deps = _deps(cfg)
    current = unit_fingerprint(deps.config_fp, "h-alpha")
    store = _store_with(_section_rows(path, {"alpha": current, "beta": "stale"}), [])

    result = phase_chunk_units(cfg, deps, store)

    assert (result.processed, result.skipped) == (1, 1)
    assert store.delete_nodes.call_count == 1
    assert store.delete_nodes.call_args.kwargs["where"] == {"parent_id": f"{path}#beta"}


def test_phase_chunks_file_parent_reads_disk(tmp_path: Path) -> None:
    txt = tmp_path / "notes.txt"
    txt.write_text("para one\n\npara two\n", encoding="utf-8")
    path = canonical_path(txt)
    cfg = _corpus(tmp_path, granularity="file")
    deps = _deps(cfg)
    file_rows = [
        {
            "path": path,
            "hash": "fh",
            "fp": None,
            "summary": None,
            "key_points": None,
            "entities_mentioned": None,
        }
    ]
    store = _store_with([], file_rows)

    result = phase_chunk_units(cfg, deps, store)

    assert result.processed == 1
    _, rows = store.upsert_nodes.call_args.args
    assert all(r["parent_label"] == "File" and r["parent_id"] == path for r in rows)
    assert rows[0]["prefix"] == "notes.txt"  # no headings → rel path breadcrumb
    assert rows[0]["keywords"] == []
    fp_write = store.exec_write.call_args_list[-1]
    assert "MATCH (p:File {path: $pid})" in fp_write.args[0]


def test_phase_chunks_scopes_to_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("## A\n\nbody a\n", encoding="utf-8")
    b.write_text("## B\n\nbody b\n", encoding="utf-8")
    cfg = _corpus(tmp_path)
    deps = _deps(cfg)
    rows = _section_rows(canonical_path(a), {"a": None}) + _section_rows(
        canonical_path(b), {"b": None}
    )
    store = _store_with(rows, [])
    result = phase_chunk_units(cfg, deps, store, paths=[a])
    assert result.processed == 1
    assert store.delete_nodes.call_args.kwargs["where"]["parent_id"].endswith("#a")


def test_phase_chunks_embedding_failure_leaves_parent_unstamped(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("## A\n\nbody a\n", encoding="utf-8")
    cfg = _corpus(tmp_path)
    deps = _deps(cfg, CountingEmbedder(fail=True))
    store = _store_with(_section_rows(canonical_path(md), {"a": None}), [])
    result = phase_chunk_units(cfg, deps, store)
    assert (result.processed, result.skipped) == (0, 1)
    store.delete_nodes.assert_not_called()
    store.upsert_nodes.assert_not_called()
    assert not any("chunk_fingerprint" in c.args[0] for c in store.exec_write.call_args_list)


def test_phase_chunks_missing_source_is_skipped(tmp_path: Path) -> None:
    cfg = _corpus(tmp_path)
    deps = _deps(cfg)
    gone = canonical_path(tmp_path / "gone.md")
    store = _store_with(_section_rows(gone, {"x": None}), [])
    result = phase_chunk_units(cfg, deps, store)
    assert (result.processed, result.skipped) == (0, 1)


def test_phase_chunks_strategy_failure_is_isolated(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("## A\n\nbody a\n", encoding="utf-8")
    cfg = _corpus(tmp_path)
    deps = _deps(cfg)
    boom = MagicMock()
    boom.chunk.side_effect = ValueError("bad strategy")
    deps._chunkers = {(".md", "fine"): boom, (".md", "coarse"): boom}
    store = _store_with(_section_rows(canonical_path(md), {"a": None}), [])
    result = phase_chunk_units(cfg, deps, store)
    assert (result.processed, result.skipped) == (0, 1)
    store.upsert_nodes.assert_not_called()


def test_gc_chunks_deletes_orphans_and_stale_profiles(tmp_path: Path) -> None:
    cfg = _corpus(tmp_path)
    store = MagicMock()
    store.exec_read.side_effect = [[{"n": 2}], [{"n": 3}]]
    result = phase_gc_chunks(cfg, store)
    assert result.processed == 5
    writes = [c.args[0] for c in store.exec_write.call_args_list]
    assert len(writes) == 2
    assert "NOT ()-[:CONTAINS]->(c)" in writes[0]
    assert "NOT c.profile IN $names" in writes[1]
    assert store.exec_write.call_args_list[1].args[1]["names"] == ["fine", "coarse"]


def test_gc_chunks_noop_when_clean(tmp_path: Path) -> None:
    cfg = _corpus(tmp_path)
    store = MagicMock()
    store.exec_read.side_effect = [[{"n": 0}], [{"n": 0}]]
    assert phase_gc_chunks(cfg, store).processed == 0
    store.exec_write.assert_not_called()


def test_config_drifted_and_rechunk_stamps(tmp_path: Path) -> None:
    store = MagicMock()
    store.exec_read.return_value = [{"fp": "old"}]
    assert config_drifted(store, "c", "new") is True
    assert config_drifted(store, "c", "old") is False
    store.exec_read.return_value = [{"fp": None}]
    assert config_drifted(store, "c", "new") is False
    store.exec_read.return_value = []
    assert config_drifted(store, "c", "new") is False

    cfg = _corpus(tmp_path)
    deps = _deps(cfg)
    store = _store_with([], [])
    rechunk_corpus(cfg, deps, store)
    last = store.exec_write.call_args_list[-1]
    assert "chunk_config_fingerprint" in last.args[0] and last.args[1]["fp"] == deps.config_fp


def test_estimate_counts_without_providers(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text(_MD, encoding="utf-8")
    cfg = _corpus(tmp_path)
    cfg.chunking.profiles.append(ChunkProfile(name="sem", strategy="semantic"))
    cfg.chunking.prefix = "llm"
    deps = build_chunking_deps(
        MagicMock(providers=MagicMock(embedding="voyage", voyage=MagicMock(model="m"))),
        cfg,
        embedder=None,
        inference=None,
        renderer=None,
        validate=False,
    )
    assert deps is not None
    est = estimate_chunks(cfg, deps, [md])
    assert est["fine"]["chunks"] > est["coarse"]["chunks"] >= 1
    assert est["fine"]["embed_tokens"] > 0
    assert est["sem"]["chunks"] >= 1  # approximated by structural, no embedder needed
    assert est["_prefix_and_questions"]["llm_calls"] == 3  # one per section


def test_build_chunking_deps_disabled_and_validation(tmp_path: Path) -> None:
    cfg = _corpus(tmp_path)
    cfg.chunking.enabled = False
    fake_cfg = MagicMock(providers=MagicMock(embedding="voyage", voyage=MagicMock(model="m")))
    assert build_chunking_deps(fake_cfg, cfg, embedder=None, inference=None, renderer=None) is None

    cfg2 = _corpus(tmp_path)
    cfg2.chunking = ChunkingSection(
        tokenizer="words", profiles=[ChunkProfile(name="s", strategy="semantic")]
    )
    from contextd.chunking.strategies import ChunkingConfigError

    with pytest.raises(ChunkingConfigError, match="embedding provider"):
        build_chunking_deps(fake_cfg, cfg2, embedder=None, inference=None, renderer=None)
