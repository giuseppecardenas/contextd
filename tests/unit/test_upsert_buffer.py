from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_append_creates_jsonl_file(tmp_path: Path) -> None:
    from contextd.indexer.upsert_buffer import PendingUpsertBuffer

    buf = PendingUpsertBuffer(tmp_path / "pending-upserts.jsonl")
    buf.append(Path("/some/file.md"), "my-corpus")

    assert buf._buffer_path.exists()
    lines = buf._buffer_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {"path": "/some/file.md", "corpus": "my-corpus"}


def test_append_accumulates_duplicate_paths(tmp_path: Path) -> None:
    from contextd.indexer.upsert_buffer import PendingUpsertBuffer

    buf = PendingUpsertBuffer(tmp_path / "pending-upserts.jsonl")
    buf.append(Path("/some/file.md"), "my-corpus")
    buf.append(Path("/some/file.md"), "my-corpus")

    lines = buf._buffer_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_load_returns_empty_on_missing_file(tmp_path: Path) -> None:
    from contextd.indexer.upsert_buffer import PendingUpsertBuffer

    buf = PendingUpsertBuffer(tmp_path / "pending-upserts.jsonl")
    assert buf.load() == []


def test_load_skips_corrupt_lines(tmp_path: Path) -> None:
    from contextd.indexer.upsert_buffer import PendingUpsertBuffer

    buf_path = tmp_path / "pending-upserts.jsonl"
    buf_path.write_text(
        json.dumps({"path": "/good/file.md", "corpus": "corp"}) + "\n" + "NOT_VALID_JSON\n"
    )
    buf = PendingUpsertBuffer(buf_path)
    records = buf.load()
    assert len(records) == 1
    assert records[0]["path"] == "/good/file.md"


def test_clear_removes_buffer_file(tmp_path: Path) -> None:
    from contextd.indexer.upsert_buffer import PendingUpsertBuffer

    buf = PendingUpsertBuffer(tmp_path / "pending-upserts.jsonl")
    buf.append(Path("/some/file.md"), "corp")
    assert buf._buffer_path.exists()
    buf.clear()
    assert not buf._buffer_path.exists()


def test_replay_calls_run_incremental_for_each_entry(tmp_path: Path) -> None:
    from contextd.indexer.upsert_buffer import PendingUpsertBuffer

    buf = PendingUpsertBuffer(tmp_path / "pending-upserts.jsonl")
    buf.append(Path("/a/file1.md"), "corp")
    buf.append(Path("/a/file2.md"), "corp")

    entry = MagicMock()
    entry.corpus_cfg = MagicMock()
    entry.store = MagicMock()
    entry.hasher = MagicMock()
    entry.embedder = MagicMock()
    entry.summariser = MagicMock()
    entry.inferrer = MagicMock()
    entry.entity_sampler = MagicMock()

    def corpus_lookup(name: str) -> MagicMock:
        return entry

    with patch("contextd.indexer.upsert_buffer.run_incremental_file") as mock_run:
        mock_run.return_value = MagicMock()
        buf.replay(corpus_lookup)

    assert mock_run.call_count == 2


def test_replay_removes_succeeded_entries_from_buffer(tmp_path: Path) -> None:
    from contextd.indexer.upsert_buffer import PendingUpsertBuffer

    buf = PendingUpsertBuffer(tmp_path / "pending-upserts.jsonl")
    buf.append(Path("/a/file1.md"), "corp")
    buf.append(Path("/a/file2.md"), "corp")

    entry = MagicMock()

    call_count = 0

    def fake_run(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("store error")
        return MagicMock()

    def corpus_lookup(name: str) -> MagicMock:
        return entry

    with patch("contextd.indexer.upsert_buffer.run_incremental_file", side_effect=fake_run):
        buf.replay(corpus_lookup)

    remaining = buf.load()
    assert len(remaining) == 1
    assert remaining[0]["path"] == "/a/file2.md"


def test_replay_returns_correct_counts(tmp_path: Path) -> None:
    from contextd.indexer.upsert_buffer import PendingUpsertBuffer

    buf = PendingUpsertBuffer(tmp_path / "pending-upserts.jsonl")
    buf.append(Path("/a/file1.md"), "corp")
    buf.append(Path("/a/file2.md"), "corp")
    buf.append(Path("/a/file3.md"), "corp")

    entry = MagicMock()

    call_count = 0

    def fake_run(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("store error")
        return MagicMock()

    def corpus_lookup(name: str) -> MagicMock:
        return entry

    with patch("contextd.indexer.upsert_buffer.run_incremental_file", side_effect=fake_run):
        succeeded, failed = buf.replay(corpus_lookup)

    assert succeeded == 2
    assert failed == 1


def test_replay_retains_entry_when_corpus_not_found(tmp_path: Path) -> None:
    from contextd.indexer.upsert_buffer import PendingUpsertBuffer

    buf = PendingUpsertBuffer(tmp_path / "buf.jsonl")
    buf.append(Path("/some/file.md"), "missing-corpus")
    succeeded, failed = buf.replay(corpus_lookup=lambda _: None)
    assert succeeded == 0
    assert failed == 1
    records = buf.load()
    assert len(records) == 1
    assert records[0]["corpus"] == "missing-corpus"
