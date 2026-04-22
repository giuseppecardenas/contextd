"""Idempotent-resume + refresh-scope contract for bootstrap phases.

`phase_summarise` / `phase_summarise_sections` skip nodes whose summary is
already set. `phase_relate` / `phase_relate_sections` skip nodes whose
`inferred_at` marker is already set and write the marker after a successful
upsert loop (but not on LLM-error paths).

`_wipe_for_refresh` clears state sized to the named scope; the four scopes
match the dependency-layer stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from contextd.corpus_config import CorpusConfig
from contextd.indexer.phases import (
    phase_relate,
    phase_relate_sections,
    phase_summarise,
    phase_summarise_sections,
)
from contextd.indexer.pipeline import _wipe_for_refresh
from contextd.inference.summarise import FileSummary


def _make_files(tmp_path: Path, n: int, suffix: str = ".txt") -> list[Path]:
    out = []
    for i in range(n):
        f = tmp_path / f"f{i}{suffix}"
        f.write_text(f"content-{i}")
        out.append(f)
    return out


def _make_section_corpus(
    tmp_path: Path, n_files: int, sections_per_file: int
) -> tuple[CorpusConfig, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for i in range(n_files):
        f = tmp_path / f"doc{i}.md"
        body = "\n".join(f"## Heading {j}\n\nbody-{i}-{j}\n" for j in range(sections_per_file))
        f.write_text(body)
        for j in range(sections_per_file):
            rows.append({"id": f"{f}#heading-{j}", "path": str(f)})
    corpus = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "test",
                "root": str(tmp_path),
                "include": ["*.md"],
                "granularity": "section",
                "heading_min_level": 2,
                "heading_max_level": 4,
            }
        }
    )
    return corpus, rows


# --- phase_summarise (file-granular) -----------------------------------------


def test_phase_summarise_skips_already_summarised(tmp_path: Path) -> None:
    files = _make_files(tmp_path, 4)
    summariser = MagicMock()
    summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )
    store = MagicMock()
    # Two of the four files already have a summary on their File node.
    store.exec_read.return_value = [{"path": str(files[0])}, {"path": str(files[2])}]

    result = phase_summarise(files, summariser, store)

    assert result.processed == 2
    assert summariser.summarise.call_count == 2


def test_phase_summarise_all_already_done(tmp_path: Path) -> None:
    files = _make_files(tmp_path, 3)
    summariser = MagicMock()
    store = MagicMock()
    store.exec_read.return_value = [{"path": str(f)} for f in files]

    result = phase_summarise(files, summariser, store)

    assert result.processed == 0
    assert result.skipped == 0
    summariser.summarise.assert_not_called()


def test_phase_summarise_empty_input_short_circuits(tmp_path: Path) -> None:
    # Empty input list must not hit the store with a "IN []" query — just return.
    store = MagicMock()
    result = phase_summarise([], MagicMock(), store)
    assert result.processed == 0
    store.exec_read.assert_not_called()


# --- phase_summarise_sections ------------------------------------------------


def test_phase_summarise_sections_read_query_filters_on_summary(tmp_path: Path) -> None:
    corpus, rows = _make_section_corpus(tmp_path, 1, 2)
    summariser = MagicMock()
    summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )
    store = MagicMock()
    store.exec_read.return_value = rows

    phase_summarise_sections(corpus, summariser, store)

    # The query passed to exec_read must include the IS NULL predicate.
    called_cypher = store.exec_read.call_args.args[0]
    assert "s.summary IS NULL" in called_cypher


# --- phase_relate (file-granular) --------------------------------------------


def test_phase_relate_skips_files_with_inferred_at(tmp_path: Path) -> None:
    files = _make_files(tmp_path, 4)
    inferrer = MagicMock()
    inferrer.infer.return_value = []
    store = MagicMock()
    store.exec_read.return_value = [{"path": str(files[1])}]  # one already done

    result = phase_relate(files, inferrer, store, entity_sampler=lambda _s: [], corpus="c")

    assert result.processed == 3
    assert inferrer.infer.call_count == 3


def test_phase_relate_sets_inferred_at_after_successful_upsert(tmp_path: Path) -> None:
    files = _make_files(tmp_path, 1)
    inferrer = MagicMock()
    inferrer.infer.return_value = []  # zero-edge: must still be marked processed
    store = MagicMock()
    store.exec_read.return_value = []  # none already processed

    phase_relate(files, inferrer, store, entity_sampler=lambda _s: [], corpus="c")

    # Find the exec_write that set inferred_at.
    marker_calls = [c for c in store.exec_write.call_args_list if "SET f.inferred_at" in c.args[0]]
    assert len(marker_calls) == 1
    assert marker_calls[0].args[1]["path"] == str(files[0])


def test_phase_relate_tags_auto_created_targets_with_corpus(tmp_path: Path) -> None:
    """Regression: the inferrer-auto-created destination of each inferred edge
    must carry the ``corpus`` property on MERGE so that it's reachable from
    corpus-scoped queries. Previously the stub MERGE only set the PK, leaving
    the destination untracked with corpus=NULL and invisible to
    ``phase_gc_sections`` (which filters by corpus)."""
    from contextd.inference.relate import InferredRelationship

    files = _make_files(tmp_path, 1)
    inferrer = MagicMock()
    inferrer.infer.return_value = [
        InferredRelationship(
            edge_type="REFERENCES",
            target_type="File",
            target_name="other.md",
            confidence=0.9,
            reason="stub",
        )
    ]
    store = MagicMock()
    store.exec_read.return_value = []

    phase_relate(files, inferrer, store, entity_sampler=lambda _s: [], corpus="runeledger")

    # The upsert_node call for the auto-created target must include corpus.
    target_upserts = [
        c
        for c in store.upsert_node.call_args_list
        if c.args[0] == "File" and c.args[1].get("path") == "other.md"
    ]
    assert len(target_upserts) == 1, "auto-created target was not upserted"
    assert target_upserts[0].args[1].get("corpus") == "runeledger", (
        f"corpus property missing from auto-created stub: {target_upserts[0].args[1]}"
    )


def test_phase_relate_sections_tags_auto_created_targets_with_corpus(tmp_path: Path) -> None:
    """Section-granular counterpart of the auto-created-target corpus-tag
    regression. Same invariant: every inferred-edge target MERGEd here must
    carry the current corpus name so GC and corpus-scoped queries can see it.
    """
    from contextd.inference.relate import InferredRelationship

    corpus, rows = _make_section_corpus(tmp_path, 1, 1)
    inferrer = MagicMock()
    inferrer.infer.return_value = [
        InferredRelationship(
            edge_type="REFERENCES",
            target_type="Pattern",
            target_name="some_target",
            confidence=0.9,
            reason="stub",
        )
    ]
    store = MagicMock()
    store.exec_read.return_value = rows

    phase_relate_sections(corpus, inferrer, store, entity_sampler=lambda _s: [])

    target_upserts = [
        c
        for c in store.upsert_node.call_args_list
        if c.args[0] == "Pattern" and c.args[1].get("name") == "some_target"
    ]
    assert len(target_upserts) == 1
    assert target_upserts[0].args[1].get("corpus") == corpus.corpus.name


def test_phase_relate_does_not_set_marker_on_llm_error(tmp_path: Path) -> None:
    files = _make_files(tmp_path, 1)
    inferrer = MagicMock()
    inferrer.infer.side_effect = RuntimeError("provider failure")
    store = MagicMock()
    store.exec_read.return_value = []

    result = phase_relate(files, inferrer, store, entity_sampler=lambda _s: [], corpus="c")

    assert result.processed == 0
    assert result.skipped == 1
    marker_calls = [c for c in store.exec_write.call_args_list if "SET f.inferred_at" in c.args[0]]
    assert marker_calls == []


# --- phase_relate_sections ---------------------------------------------------


def test_phase_relate_sections_read_query_filters_on_inferred_at(tmp_path: Path) -> None:
    corpus, rows = _make_section_corpus(tmp_path, 1, 2)
    inferrer = MagicMock()
    inferrer.infer.return_value = []
    store = MagicMock()
    store.exec_read.return_value = rows

    phase_relate_sections(corpus, inferrer, store, entity_sampler=lambda _s: [])

    called_cypher = store.exec_read.call_args.args[0]
    assert "s.inferred_at IS NULL" in called_cypher


def test_phase_relate_sections_sets_marker_on_zero_edges(tmp_path: Path) -> None:
    corpus, rows = _make_section_corpus(tmp_path, 1, 1)
    inferrer = MagicMock()
    inferrer.infer.return_value = []  # zero edges but still processed
    store = MagicMock()
    store.exec_read.return_value = rows

    phase_relate_sections(corpus, inferrer, store, entity_sampler=lambda _s: [])

    marker_calls = [c for c in store.exec_write.call_args_list if "SET s.inferred_at" in c.args[0]]
    assert len(marker_calls) == 1
    assert marker_calls[0].args[1]["id"] == rows[0]["id"]


# --- _wipe_for_refresh -------------------------------------------------------


def _wipe_store_calls(store: MagicMock) -> list[str]:
    """Return the list of cypher strings issued via exec_write."""
    return [c.args[0] for c in store.exec_write.call_args_list]


def _wipe_corpus() -> CorpusConfig:
    return CorpusConfig.model_validate(
        {"corpus": {"name": "test", "root": "/tmp", "include": ["**/*"]}}
    )


def test_wipe_inferred_scope() -> None:
    store = MagicMock()
    _wipe_for_refresh(_wipe_corpus(), store, "inferred")
    calls = _wipe_store_calls(store)
    # inferred edges deleted; inferred_at removed from both node labels.
    assert any("r.origin = 'inferred' DELETE r" in c for c in calls)
    assert any("REMOVE n.inferred_at" in c and "Section" in c for c in calls)
    assert any("REMOVE n.inferred_at" in c and "File" in c for c in calls)
    # summaries NOT touched
    assert not any("REMOVE n.summary" in c for c in calls)
    # no DETACH DELETE in this scope
    assert not any("DETACH DELETE" in c for c in calls)


def test_wipe_summaries_scope() -> None:
    store = MagicMock()
    _wipe_for_refresh(_wipe_corpus(), store, "summaries")
    calls = _wipe_store_calls(store)
    assert any("REMOVE n.summary" in c and "Section" in c for c in calls)
    assert any("REMOVE n.summary" in c and "File" in c for c in calls)
    # inferred edges NOT touched
    assert not any("DELETE r" in c for c in calls)
    assert not any("REMOVE n.inferred_at" in c for c in calls)
    assert not any("DETACH DELETE" in c for c in calls)


def test_wipe_llm_scope_is_union() -> None:
    store = MagicMock()
    _wipe_for_refresh(_wipe_corpus(), store, "llm")
    calls = _wipe_store_calls(store)
    assert any("r.origin = 'inferred' DELETE r" in c for c in calls)
    assert any("REMOVE n.inferred_at" in c for c in calls)
    assert any("REMOVE n.summary" in c for c in calls)
    assert not any("DETACH DELETE" in c for c in calls)


def test_wipe_all_scope_detach_deletes_all_three_labels() -> None:
    store = MagicMock()
    _wipe_for_refresh(_wipe_corpus(), store, "all")
    calls = _wipe_store_calls(store)
    assert any("MATCH (n:Section {corpus: $c}) DETACH DELETE n" in c for c in calls)
    assert any("MATCH (n:File {corpus: $c}) DETACH DELETE n" in c for c in calls)
    assert any("MATCH (n:Corpus {name: $c}) DETACH DELETE n" in c for c in calls)
    # No property-level edits when doing a DETACH DELETE wipe.
    assert not any("REMOVE" in c for c in calls)


def test_run_bootstrap_applies_refresh_when_set() -> None:
    # End-to-end at the pipeline level: when run_bootstrap is called with a
    # refresh scope, _wipe_for_refresh runs before any phase code.
    from contextd.indexer.pipeline import run_bootstrap

    corpus = CorpusConfig.model_validate(
        {"corpus": {"name": "x", "root": "/nonexistent", "include": ["**/*"]}}
    )
    store = MagicMock()
    # enumerate_corpus_files over a nonexistent root returns [], so phases
    # run with empty input — we only care that _wipe_for_refresh fired.
    run_bootstrap(
        corpus=corpus,
        store=store,
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        hasher=MagicMock(),
        entity_sampler=lambda _s: [],
        refresh="inferred",
    )
    # At least one exec_write from _wipe_for_refresh must have landed.
    assert any(
        "r.origin = 'inferred' DELETE r" in c.args[0] for c in store.exec_write.call_args_list
    )
