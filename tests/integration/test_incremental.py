from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contextd.corpus_config import CorpusConfig
from contextd.indexer.hasher import FileHasher
from contextd.indexer.pipeline import run_bootstrap, run_incremental_file
from contextd.inference.summarise import FileSummary
from contextd.storage.base import GraphStore

pytestmark = pytest.mark.integration


def _corpus(tmp_path: Path, granularity: str = "file") -> CorpusConfig:
    return CorpusConfig.model_validate(
        {"corpus": {"name": "inc", "root": str(tmp_path), "granularity": granularity}}
    )


def _fake_summariser(summary: str) -> MagicMock:
    s = MagicMock()
    s.summarise.return_value = FileSummary(summary=summary, key_points=[], entities_mentioned=[])
    return s


def _fake_embedder() -> MagicMock:
    e = MagicMock()
    e.embed.return_value = [[0.1] * 1024]
    return e


def test_incremental_updates_summary_after_file_change(backend: GraphStore, tmp_path: Path) -> None:
    md = tmp_path / "a.md"
    md.write_text("original content")
    corpus = _corpus(tmp_path)
    hasher = FileHasher()

    run_bootstrap(
        corpus,
        backend,
        _fake_embedder(),
        _fake_summariser("original summary"),
        MagicMock(infer=MagicMock(return_value=[])),
        hasher,
        lambda _s: [],
    )

    rows = backend.exec_read("MATCH (f:File {path: $p}) RETURN f.summary AS s", {"p": str(md)})
    assert rows[0]["s"] == "original summary"

    md.write_text("updated content")
    run_incremental_file(
        md,
        corpus,
        backend,
        hasher,
        _fake_embedder(),
        _fake_summariser("updated summary"),
        MagicMock(infer=MagicMock(return_value=[])),
        lambda _s: [],
    )

    rows = backend.exec_read("MATCH (f:File {path: $p}) RETURN f.summary AS s", {"p": str(md)})
    assert rows[0]["s"] == "updated summary"


def test_incremental_deletion_removes_file_node(backend: GraphStore, tmp_path: Path) -> None:
    md = tmp_path / "b.md"
    md.write_text("content")
    corpus = _corpus(tmp_path)
    hasher = FileHasher()

    run_bootstrap(
        corpus,
        backend,
        _fake_embedder(),
        _fake_summariser("s"),
        MagicMock(infer=MagicMock(return_value=[])),
        hasher,
        lambda _s: [],
    )

    md.unlink()
    run_incremental_file(
        md,
        corpus,
        backend,
        hasher,
        _fake_embedder(),
        MagicMock(),
        MagicMock(),
        lambda _s: [],
    )

    rows = backend.exec_read("MATCH (f:File {path: $p}) RETURN f", {"p": str(md)})
    assert rows == []


def test_incremental_clears_inferred_at_before_reindex(backend: GraphStore, tmp_path: Path) -> None:
    md = tmp_path / "c.md"
    md.write_text("content")
    corpus = _corpus(tmp_path)
    hasher = FileHasher()
    inferrer = MagicMock()
    inferrer.infer.return_value = []

    run_bootstrap(
        corpus,
        backend,
        _fake_embedder(),
        _fake_summariser("s"),
        inferrer,
        hasher,
        lambda _s: [],
    )

    md.write_text("updated content")
    inferrer.infer.reset_mock()
    run_incremental_file(
        md,
        corpus,
        backend,
        hasher,
        _fake_embedder(),
        _fake_summariser("updated"),
        inferrer,
        lambda _s: [],
    )

    assert inferrer.infer.called, "infer must be called — inferred_at was cleared"
