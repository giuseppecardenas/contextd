"""A provider failure in an LLM phase must be logged, not silently counted.

``phase_summarise`` / ``phase_relate`` (and their section variants) catch every
exception from the provider and fold it into ``PhaseResult.skipped``, an integer
no caller surfaces per file. Meanwhile ``run_incremental_file`` reports the file
as ``indexed`` regardless, because its node and embedding really were written.
The result was a file with no summary, or no inferred edges, and a log line
claiming success: 92 of 461 files in a real corpus had no summary and 55 no
``inferred_at``, with nothing anywhere recording why.

These tests pin the log line for each of the four failure paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contextd.corpus_config import CorpusConfig
from contextd.indexer.phases import (
    phase_relate,
    phase_relate_sections,
    phase_summarise,
    phase_summarise_sections,
)


def _file(tmp_path: Path, name: str = "a.md") -> Path:
    # A level-2 heading, since the default heading_min_level is 2 and a level-1
    # heading would yield no Section nodes at all.
    f = tmp_path / name
    f.write_text("# Title\n\nintro\n\n## Heading One\n\nbody text\n")
    return f


def _section_corpus(tmp_path: Path) -> CorpusConfig:
    return CorpusConfig.model_validate(
        {"corpus": {"name": "c", "root": str(tmp_path), "granularity": "section"}}
    )


def test_summarise_failure_is_logged_with_path_and_cause(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    f = _file(tmp_path)
    summariser = MagicMock()
    summariser.summarise.side_effect = RuntimeError("429 rate limit")
    store = MagicMock()
    store.exec_read.return_value = []

    with caplog.at_level(logging.WARNING, logger="contextd.indexer.phases"):
        result = phase_summarise([f], summariser, store)

    assert result.skipped == 1
    assert result.processed == 0
    assert str(f) in caplog.text
    assert "RuntimeError" in caplog.text
    assert "429 rate limit" in caplog.text
    assert "without a summary" in caplog.text
    store.exec_write.assert_not_called()


def test_relate_failure_is_logged_and_marker_left_unset(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    f = _file(tmp_path)
    inferrer = MagicMock()
    inferrer.infer.side_effect = ValueError("malformed JSON body")
    store = MagicMock()
    store.exec_read.return_value = []

    with caplog.at_level(logging.WARNING, logger="contextd.indexer.phases"):
        result = phase_relate([f], inferrer, store, entity_sampler=lambda _s: [], corpus="c")

    assert result.skipped == 1
    assert str(f) in caplog.text
    assert "ValueError" in caplog.text
    assert "malformed JSON body" in caplog.text
    assert "retried on the next pass" in caplog.text
    # The inferred_at marker is what makes resume idempotent; it must stay unset.
    assert not any("inferred_at" in str(c) for c in store.exec_write.call_args_list)


def test_summarise_sections_failure_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    f = _file(tmp_path)
    summariser = MagicMock()
    summariser.summarise.side_effect = RuntimeError("safety block")
    store = MagicMock()
    store.exec_read.return_value = [{"id": f"{f}#heading-one", "path": str(f)}]

    with caplog.at_level(logging.WARNING, logger="contextd.indexer.phases"):
        result = phase_summarise_sections(_section_corpus(tmp_path), summariser, store)

    assert result.skipped == 1
    assert "safety block" in caplog.text
    assert "without a summary" in caplog.text


def test_relate_sections_failure_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    f = _file(tmp_path)
    inferrer = MagicMock()
    inferrer.infer.side_effect = RuntimeError("connection reset")
    store = MagicMock()
    store.exec_read.return_value = [{"id": f"{f}#heading-one", "path": str(f)}]

    with caplog.at_level(logging.WARNING, logger="contextd.indexer.phases"):
        result = phase_relate_sections(
            _section_corpus(tmp_path), inferrer, store, entity_sampler=lambda _s: []
        )

    assert result.skipped == 1
    assert "connection reset" in caplog.text
    assert "retried on the next pass" in caplog.text
