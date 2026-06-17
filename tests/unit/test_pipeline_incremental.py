from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from contextd._paths import canonical_path


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


def test_run_incremental_file_skips_unchanged_file_by_hash(tmp_path: Path) -> None:
    """File-granular: when the File node's stored hash matches current content,
    re-indexing is skipped so no embed/summarise/relate tokens are spent."""
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import run_incremental_file

    md = tmp_path / "a.md"
    md.write_text("# Title\n\nbody")
    corpus = _make_corpus(tmp_path)

    hasher = FileHasher()
    store = MagicMock()
    store.exec_read.return_value = [{"hash": hasher.hash(md)}]
    embedder = MagicMock()
    summariser = MagicMock()
    inferrer = MagicMock()

    result = run_incremental_file(
        md, corpus, store, hasher, embedder, summariser, inferrer, lambda _s: []
    )

    assert result.action == "skipped"
    assert embedder.embed.called is False
    assert summariser.summarise.called is False
    assert store.delete_edges.called is False


def test_run_incremental_file_reindexes_when_hash_differs(tmp_path: Path) -> None:
    """File-granular: a stored hash that differs from current content forces a
    full re-index rather than a skip."""
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import run_incremental_file
    from contextd.inference.summarise import FileSummary

    md = tmp_path / "a.md"
    md.write_text("# Title\n\nbody")
    corpus = _make_corpus(tmp_path)

    # First exec_read is the gate query (stale hash → mismatch); later phase
    # queries return empty so summarise/relate run.
    call_count = [0]

    def _exec_read(query: str, params: object) -> list[object]:
        call_count[0] += 1
        return [{"hash": "stale-does-not-match"}] if call_count[0] == 1 else []

    store = MagicMock()
    store.exec_read.side_effect = _exec_read
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]
    summariser = MagicMock()
    summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )
    inferrer = MagicMock()
    inferrer.infer.return_value = []

    result = run_incremental_file(
        md, corpus, store, FileHasher(), embedder, summariser, inferrer, lambda _s: []
    )

    assert result.action == "indexed"
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


def test_run_incremental_file_returns_skipped_when_no_sections_changed(
    tmp_path: Path,
) -> None:
    """When all Section.hash values match current content, action=='skipped'."""
    import hashlib

    from contextd.indexer.pipeline import run_incremental_file

    md = tmp_path / "doc.md"
    md.write_text("## Alpha\n\nBody alpha.\n")
    corpus = _make_corpus(tmp_path, granularity="section")

    file_path_str = canonical_path(md)
    from contextd.indexer.heading_parser import HeadingParser

    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md.read_text())
    stored_rows = [
        {
            "id": f"{file_path_str}#{sec.anchor}",
            "hash": hashlib.md5((sec.title + "\n\n" + sec.body).encode()).hexdigest(),
        }
        for sec in sections
    ]

    store = MagicMock()
    # First exec_read: differential hash query → stored hashes match current content
    store.exec_read.return_value = stored_rows

    result = run_incremental_file(
        md,
        corpus,
        store,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        lambda _s: [],
    )

    assert result.action == "skipped"


def test_run_incremental_file_clears_only_changed_sections(tmp_path: Path) -> None:
    """Only sections with a stale hash get cleared; unchanged sections are kept."""
    import hashlib
    from unittest.mock import patch

    from contextd.indexer.pipeline import run_incremental_file

    md = tmp_path / "doc.md"
    md.write_text("## Alpha\n\nBody alpha.\n\n## Beta\n\nBody beta.\n")
    corpus = _make_corpus(tmp_path, granularity="section")

    file_path_str = canonical_path(md)
    from contextd.indexer.heading_parser import HeadingParser

    parser = HeadingParser(min_level=2, max_level=4)
    sections = parser.parse(md.read_text())
    alpha = sections[0]

    # Alpha has correct hash; Beta has stale hash
    alpha_hash = hashlib.md5((alpha.title + "\n\n" + alpha.body).encode()).hexdigest()
    stored_rows = [
        {"id": f"{file_path_str}#alpha", "hash": alpha_hash},
        {"id": f"{file_path_str}#beta", "hash": "stale_hash_value"},
    ]

    store = MagicMock()
    # First exec_read: differential hash query. Subsequent calls (phase queries): empty.
    call_count = [0]

    def _exec_read(query: str, params: object) -> list[object]:
        call_count[0] += 1
        return stored_rows if call_count[0] == 1 else []

    store.exec_read.side_effect = _exec_read

    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024, [0.2] * 1024]
    from contextd.inference.summarise import FileSummary

    summariser = MagicMock()
    summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )
    inferrer = MagicMock()
    inferrer.infer.return_value = []

    cleared_ids: list[str] = []

    def spy_clear(section_id: str, corpus_name: str, s: object) -> None:
        cleared_ids.append(section_id)

    with patch("contextd.indexer.pipeline._clear_section_for_reindex", side_effect=spy_clear):
        result = run_incremental_file(
            md, corpus, store, MagicMock(), embedder, summariser, inferrer, lambda _s: []
        )

    assert result.action == "indexed"
    # Only beta's ID should have been cleared
    assert f"{file_path_str}#beta" in cleared_ids
    assert f"{file_path_str}#alpha" not in cleared_ids


def test_run_incremental_file_treats_missing_hash_as_changed(tmp_path: Path) -> None:
    """A Section with stored_hash=None must be treated as changed and re-processed."""
    from unittest.mock import patch

    from contextd.indexer.pipeline import run_incremental_file

    md = tmp_path / "doc.md"
    md.write_text("## Alpha\n\nBody alpha.\n")
    corpus = _make_corpus(tmp_path, granularity="section")

    file_path_str = canonical_path(md)
    # Graph returns section with hash=None (pre-feature node)
    store = MagicMock()
    call_count = [0]

    def _exec_read(query: str, params: object) -> list[object]:
        call_count[0] += 1
        return [{"id": f"{file_path_str}#alpha", "hash": None}] if call_count[0] == 1 else []

    store.exec_read.side_effect = _exec_read

    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]
    from contextd.inference.summarise import FileSummary

    summariser = MagicMock()
    summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )
    inferrer = MagicMock()
    inferrer.infer.return_value = []

    cleared_ids: list[str] = []

    def spy_clear(section_id: str, corpus_name: str, s: object) -> None:
        cleared_ids.append(section_id)

    with patch("contextd.indexer.pipeline._clear_section_for_reindex", side_effect=spy_clear):
        result = run_incremental_file(
            md, corpus, store, MagicMock(), embedder, summariser, inferrer, lambda _s: []
        )

    assert result.action == "indexed"
    assert f"{file_path_str}#alpha" in cleared_ids


def test_run_incremental_file_skips_empty_file(tmp_path: Path) -> None:
    """Zero-byte files (e.g. cargo's .rmeta placeholders) must short-circuit
    to action='skipped' before reaching the embedder — Voyage rejects empty
    input strings."""
    from contextd.indexer.pipeline import run_incremental_file

    empty = tmp_path / "empty.md"
    empty.write_text("")
    corpus = _make_corpus(tmp_path)

    store = MagicMock()
    embedder = MagicMock()
    summariser = MagicMock()
    inferrer = MagicMock()

    result = run_incremental_file(
        empty, corpus, store, MagicMock(), embedder, summariser, inferrer, lambda _s: []
    )

    assert result.action == "skipped"
    assert embedder.embed.called is False


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
