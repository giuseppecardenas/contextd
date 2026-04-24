from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


def _make_corpus(tmp_path: Path, granularity: str = "file"):
    from contextd.corpus_config import CorpusConfig

    return CorpusConfig.model_validate(
        {"corpus": {"name": "inc", "root": str(tmp_path), "granularity": granularity}}
    )


def test_clear_file_removes_summary_markers(tmp_path: Path) -> None:
    from contextd.indexer.pipeline import _clear_file_for_reindex

    store = MagicMock()
    path = tmp_path / "a.md"
    _clear_file_for_reindex(path, store)

    cyphers = [c[0][0] for c in store.exec_write.call_args_list]
    assert any("summary" in q for q in cyphers)


def test_clear_file_removes_inferred_at(tmp_path: Path) -> None:
    from contextd.indexer.pipeline import _clear_file_for_reindex

    store = MagicMock()
    _clear_file_for_reindex(tmp_path / "a.md", store)

    cyphers = [c[0][0] for c in store.exec_write.call_args_list]
    assert any("inferred_at" in q for q in cyphers)


def test_clear_file_deletes_inferred_edges(tmp_path: Path) -> None:
    from contextd.indexer.pipeline import _clear_file_for_reindex

    store = MagicMock()
    _clear_file_for_reindex(tmp_path / "a.md", store)

    store.delete_edges.assert_called_once()
    kwargs = store.delete_edges.call_args[1]
    assert kwargs["origin"] == "inferred"


def test_run_incremental_file_deleted_file_detach_deletes_node(
    tmp_path: Path,
) -> None:
    from contextd.indexer.pipeline import run_incremental_file

    store = MagicMock()
    missing = tmp_path / "gone.md"
    corpus = _make_corpus(tmp_path)

    result = run_incremental_file(
        missing,
        corpus,
        store,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        lambda _s: [],
    )

    assert result.action == "deleted"
    cyphers = [c[0][0] for c in store.exec_write.call_args_list]
    assert any("DETACH DELETE" in q for q in cyphers)


def test_run_incremental_file_calls_clear_then_phases(tmp_path: Path) -> None:
    from contextd.indexer.pipeline import run_incremental_file

    md = tmp_path / "a.md"
    md.write_text("# Title\n\nbody")
    corpus = _make_corpus(tmp_path)

    store = MagicMock()
    store.exec_read.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]
    summariser = MagicMock()
    from contextd.inference.summarise import FileSummary

    summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )
    inferrer = MagicMock()
    inferrer.infer.return_value = []

    result = run_incremental_file(
        md,
        corpus,
        store,
        MagicMock(),
        embedder,
        summariser,
        inferrer,
        lambda _s: [],
    )

    assert result.action == "indexed"
    assert store.delete_edges.called
    assert embedder.embed.called


def test_run_incremental_file_section_granular(tmp_path: Path) -> None:
    from contextd.indexer.pipeline import run_incremental_file

    md = tmp_path / "doc.md"
    md.write_text("## Section A\n\nbody\n")
    corpus = _make_corpus(tmp_path, granularity="section")

    store = MagicMock()
    store.exec_read.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]
    summariser = MagicMock()
    from contextd.inference.summarise import FileSummary

    summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )
    inferrer = MagicMock()
    inferrer.infer.return_value = []

    result = run_incremental_file(
        md,
        corpus,
        store,
        MagicMock(),
        embedder,
        summariser,
        inferrer,
        lambda _s: [],
    )

    assert result.action == "indexed"
    assert store.upsert_node.called


def test_run_incremental_file_section_corpus_non_md(tmp_path: Path) -> None:
    from contextd.indexer.pipeline import run_incremental_file

    lua = tmp_path / "mod.lua"
    lua.write_text("-- code\n")
    corpus = _make_corpus(tmp_path, granularity="section")

    store = MagicMock()
    store.exec_read.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]
    summariser = MagicMock()
    from contextd.inference.summarise import FileSummary

    summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )
    inferrer = MagicMock()
    inferrer.infer.return_value = []

    result = run_incremental_file(
        lua,
        corpus,
        store,
        MagicMock(),
        embedder,
        summariser,
        inferrer,
        lambda _s: [],
    )

    assert result.action == "indexed"
    assert embedder.embed.called
