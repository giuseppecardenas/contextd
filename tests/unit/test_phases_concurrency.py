"""Concurrency contract for the four inference-bound phases.

Each phase gained a keyword-only ``concurrency: int = 1`` parameter. When
``concurrency == 1`` the existing serial behaviour is preserved (call
ordering intact for tests). When ``> 1`` workers run in a
``ThreadPoolExecutor`` — these tests prove it by recording the thread id of
each mocked inference call.
"""

from __future__ import annotations

import threading
import time
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
from contextd.inference.relate import InferredRelationship
from contextd.inference.summarise import FileSummary


def _tracking_summariser(delay: float = 0.02) -> tuple[MagicMock, list[int]]:
    """Mock summariser that records the thread id of each call."""
    thread_ids: list[int] = []

    def _summarise(content: str) -> FileSummary:
        thread_ids.append(threading.get_ident())
        time.sleep(delay)  # force overlap windows when workers are parallel
        return FileSummary(summary=f"s:{content[:10]}", key_points=[], entities_mentioned=[])

    m = MagicMock()
    m.summarise.side_effect = _summarise
    return m, thread_ids


def _tracking_inferrer(delay: float = 0.02) -> tuple[MagicMock, list[int]]:
    """Mock inferrer that records the thread id of each call."""
    thread_ids: list[int] = []

    def _infer(content: str, known_entities: list[str]) -> list[InferredRelationship]:
        thread_ids.append(threading.get_ident())
        time.sleep(delay)
        return []

    m = MagicMock()
    m.infer.side_effect = _infer
    return m, thread_ids


def _make_files(tmp_path: Path, n: int, suffix: str = ".txt") -> list[Path]:
    files = []
    for i in range(n):
        f = tmp_path / f"f{i}{suffix}"
        f.write_text(f"content-{i}")
        files.append(f)
    return files


# --- phase_summarise ---------------------------------------------------------


def test_phase_summarise_sequential_preserves_call_order(tmp_path: Path) -> None:
    files = _make_files(tmp_path, 3)
    summariser, thread_ids = _tracking_summariser(delay=0)
    store = MagicMock()

    result = phase_summarise(files, summariser, store, concurrency=1)

    assert result.processed == 3
    assert result.skipped == 0
    assert len(set(thread_ids)) == 1  # serial path stays on the main thread
    # Serial mode → calls happen in file order.
    called_contents = [c.args[0] for c in summariser.summarise.call_args_list]
    assert called_contents == ["content-0", "content-1", "content-2"]


def test_phase_summarise_parallel_uses_multiple_threads(tmp_path: Path) -> None:
    files = _make_files(tmp_path, 8)
    summariser, thread_ids = _tracking_summariser(delay=0.05)
    store = MagicMock()

    result = phase_summarise(files, summariser, store, concurrency=4)

    assert result.processed == 8
    assert result.skipped == 0
    assert len(set(thread_ids)) > 1


def test_phase_summarise_parallel_llm_error_is_skipped(tmp_path: Path) -> None:
    files = _make_files(tmp_path, 4)

    def _maybe_raise(content: str) -> FileSummary:
        if content == "content-2":
            raise RuntimeError("provider failure")
        return FileSummary(summary="ok", key_points=[], entities_mentioned=[])

    summariser = MagicMock()
    summariser.summarise.side_effect = _maybe_raise
    store = MagicMock()

    result = phase_summarise(files, summariser, store, concurrency=4)

    assert result.processed == 3
    assert result.skipped == 1


# --- phase_relate ------------------------------------------------------------


def test_phase_relate_parallel_uses_multiple_threads(tmp_path: Path) -> None:
    files = _make_files(tmp_path, 6)
    inferrer, thread_ids = _tracking_inferrer(delay=0.05)
    store = MagicMock()

    result = phase_relate(files, inferrer, store, entity_sampler=lambda _s: [], concurrency=4)

    assert result.processed == 6
    assert result.skipped == 0
    assert len(set(thread_ids)) > 1


def test_phase_relate_sequential_preserves_order(tmp_path: Path) -> None:
    files = _make_files(tmp_path, 3)
    inferrer, _ = _tracking_inferrer(delay=0)
    store = MagicMock()

    phase_relate(files, inferrer, store, entity_sampler=lambda _s: [], concurrency=1)

    called = [c.args[0] for c in inferrer.infer.call_args_list]
    assert called == ["content-0", "content-1", "content-2"]


# --- section phases ----------------------------------------------------------


def _make_section_corpus(
    tmp_path: Path, n_files: int, sections_per_file: int
) -> tuple[CorpusConfig, list[Path], list[dict[str, Any]]]:
    """Build n .md files with m H2 sections each; return (corpus, files, section rows)."""
    files: list[Path] = []
    rows: list[dict[str, Any]] = []
    for i in range(n_files):
        f = tmp_path / f"doc{i}.md"
        body_lines: list[str] = []
        for j in range(sections_per_file):
            body_lines.append(f"## Heading {j}\n\nbody-{i}-{j}\n")
        f.write_text("\n".join(body_lines))
        files.append(f)
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
    return corpus, files, rows


def test_phase_summarise_sections_parallel_uses_multiple_threads(tmp_path: Path) -> None:
    corpus, _files, rows = _make_section_corpus(tmp_path, n_files=2, sections_per_file=4)
    summariser, thread_ids = _tracking_summariser(delay=0.05)
    store = MagicMock()
    store.exec_read.return_value = rows

    result = phase_summarise_sections(corpus, summariser, store, concurrency=4)

    assert result.processed == len(rows)
    assert result.skipped == 0
    assert len(set(thread_ids)) > 1


def test_phase_summarise_sections_sequential_single_thread(tmp_path: Path) -> None:
    corpus, _files, rows = _make_section_corpus(tmp_path, n_files=2, sections_per_file=2)
    summariser, thread_ids = _tracking_summariser(delay=0)
    store = MagicMock()
    store.exec_read.return_value = rows

    phase_summarise_sections(corpus, summariser, store, concurrency=1)

    assert len(set(thread_ids)) == 1


def test_phase_relate_sections_parallel_uses_multiple_threads(tmp_path: Path) -> None:
    corpus, _files, rows = _make_section_corpus(tmp_path, n_files=2, sections_per_file=4)
    inferrer, thread_ids = _tracking_inferrer(delay=0.05)
    store = MagicMock()
    store.exec_read.return_value = rows

    result = phase_relate_sections(
        corpus, inferrer, store, entity_sampler=lambda _s: [], concurrency=4
    )

    assert result.processed == len(rows)
    assert result.skipped == 0
    assert len(set(thread_ids)) > 1
