"""Parent roll-up phase: bottom-up synthesis, input-hash gating, skips."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from contextd.corpus_config import CorpusConfig
from contextd.indexer.phases import (
    _rollup_input_hash,
    phase_rollup_sections,
    phase_summarise_sections,
)
from contextd.inference.summarise import FileSummary

_MD = "## Top\n\ntop prose\n\n### Mid\n\nmid prose\n\n#### Leaf\n\nleaf prose\n"


def _corpus(tmp_path: Path) -> CorpusConfig:
    (tmp_path / "doc.md").write_text(_MD, encoding="utf-8")
    return CorpusConfig.model_validate(
        {"corpus": {"name": "c", "root": str(tmp_path), "granularity": "section"}}
    )


def _doc_path(tmp_path: Path) -> str:
    from contextd._paths import canonical_path

    return canonical_path(tmp_path / "doc.md")


def _parent_row(tmp_path: Path, anchor: str, level: int, **extra: Any) -> dict[str, Any]:
    doc = _doc_path(tmp_path)
    row: dict[str, Any] = {
        "id": f"{doc}#{anchor}",
        "path": str(tmp_path / "doc.md"),
        "level": level,
        "summary": None,
        "summary_input_hash": None,
    }
    row.update(extra)
    return row


def test_rollup_processes_deepest_level_first(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    store = MagicMock()
    parents = [
        _parent_row(tmp_path, "top", 2),
        _parent_row(tmp_path, "mid", 3),
    ]

    def _exec_read(query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if "PARENT_OF" in query and "DISTINCT" in query:
            return parents
        if "PARENT_OF" in query:
            return [{"s": "child summary"}]
        return []

    store.exec_read.side_effect = _exec_read
    summariser = MagicMock()
    summariser.roll_up.return_value = "rolled"
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 4]

    result = phase_rollup_sections(corpus, summariser, embedder, store)

    assert result.processed == 2
    # Mid (level 3) rolled before Top (level 2).
    rolled_ids = [
        c.args[1]["id"] for c in store.exec_write.call_args_list if "s.summary =" in c.args[0]
    ]
    assert rolled_ids[0].endswith("#mid")
    assert rolled_ids[1].endswith("#top")
    # Roll-up embedding re-points the parent vector.
    assert "s.embedding = $vec" in store.exec_write.call_args_list[0].args[0]


def test_rollup_receives_own_prose(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    store = MagicMock()

    def _exec_read(query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if "DISTINCT" in query:
            return [_parent_row(tmp_path, "top", 2)]
        if "PARENT_OF" in query:
            return [{"s": "child summary"}]
        return []

    store.exec_read.side_effect = _exec_read
    summariser = MagicMock()
    summariser.roll_up.return_value = "rolled"
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 4]

    phase_rollup_sections(corpus, summariser, embedder, store)

    kw = summariser.roll_up.call_args.kwargs
    assert "top prose" in kw["own_prose"]
    assert kw["child_summaries"] == ["child summary"]
    assert kw["context"].anchor == "top"


def test_rollup_skips_on_matching_input_hash(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    store = MagicMock()
    # Compute the hash the worker will derive for Top's current inputs.
    from contextd.indexer.units import ParseCache, own_prose

    parsed = ParseCache(corpus).get(tmp_path / "doc.md")
    top = parsed.by_anchor("top")
    assert top is not None
    gate = _rollup_input_hash(own_prose(top), ["child summary"])

    def _exec_read(query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if "DISTINCT" in query:
            return [_parent_row(tmp_path, "top", 2, summary="existing", summary_input_hash=gate)]
        if "PARENT_OF" in query:
            return [{"s": "child summary"}]
        return []

    store.exec_read.side_effect = _exec_read
    summariser = MagicMock()

    result = phase_rollup_sections(corpus, summariser, MagicMock(), store)

    assert result.processed == 0
    summariser.roll_up.assert_not_called()


def test_rollup_with_no_inputs_leaves_summary_null(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("## Bare\n### Child\n", encoding="utf-8")
    corpus = CorpusConfig.model_validate(
        {"corpus": {"name": "c", "root": str(tmp_path), "granularity": "section"}}
    )
    store = MagicMock()

    def _exec_read(query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if "DISTINCT" in query:
            return [_parent_row(tmp_path, "bare", 2)]
        if "PARENT_OF" in query:
            return [{"s": None}]  # child never got a summary
        return []

    store.exec_read.side_effect = _exec_read
    summariser = MagicMock()

    result = phase_rollup_sections(corpus, summariser, MagicMock(), store)

    assert result.skipped == 1
    summariser.roll_up.assert_not_called()
    store.exec_write.assert_not_called()


def test_summarise_sections_skips_parents(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    doc = _doc_path(tmp_path)
    store = MagicMock()
    store.exec_read.return_value = [
        {"id": f"{doc}#top", "path": str(tmp_path / "doc.md")},
        {"id": f"{doc}#leaf", "path": str(tmp_path / "doc.md")},
    ]
    summariser = MagicMock()
    summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )

    result = phase_summarise_sections(corpus, summariser, store)

    # Only the leaf is summarised directly; the parent waits for roll-up.
    assert result.processed == 1
    assert summariser.summarise.call_count == 1
    assert summariser.summarise.call_args.kwargs["context"].anchor == "leaf"


def test_relate_sections_marks_proseless_parent_without_llm(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("## Bare\n### Child\n\nchild prose\n", encoding="utf-8")
    corpus = CorpusConfig.model_validate(
        {"corpus": {"name": "c", "root": str(tmp_path), "granularity": "section"}}
    )
    from contextd._paths import canonical_path
    from contextd.indexer.phases import RelateDeps, phase_relate_sections
    from contextd.inference.context import EmptyRetriever

    doc = canonical_path(md)
    store = MagicMock()
    store.exec_read.return_value = [{"id": f"{doc}#bare", "path": str(md)}]
    inferrer = MagicMock()

    phase_relate_sections(corpus, RelateDeps(inferrer=inferrer, retriever=EmptyRetriever()), store)

    inferrer.infer.assert_not_called()
    marks = [c for c in store.exec_write.call_args_list if "inferred_at" in c.args[0]]
    assert len(marks) == 1
