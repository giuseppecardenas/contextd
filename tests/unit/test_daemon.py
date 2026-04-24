from __future__ import annotations

import queue
import time
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_drain_relay_moves_all_queued_paths(tmp_path: Path) -> None:
    from contextd.daemon import DebouncedQueue, _drain_relay_into_debouncer

    relay: queue.Queue[Path] = queue.Queue()
    debouncer = DebouncedQueue(window_seconds=0.05)
    paths = [tmp_path / f"{i}.md" for i in range(3)]
    for p in paths:
        relay.put(p)

    _drain_relay_into_debouncer(relay, debouncer)

    time.sleep(0.1)
    drained = debouncer.drain_if_ready()
    assert {p.resolve() for p in paths} == set(drained)


def test_drain_relay_noop_on_empty_queue() -> None:
    from contextd.daemon import DebouncedQueue, _drain_relay_into_debouncer

    relay: queue.Queue[Path] = queue.Queue()
    debouncer = DebouncedQueue(window_seconds=30.0)
    _drain_relay_into_debouncer(relay, debouncer)
    assert debouncer.drain_if_ready() == []


def test_filter_changed_returns_only_changed_paths(tmp_path: Path) -> None:
    from contextd.daemon import _filter_changed
    from contextd.indexer.hasher import FileHasher

    hasher = FileHasher()
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("content")
    b.write_text("content")

    hasher.mark_seen(a)  # a is already known
    result = _filter_changed([a, b], hasher)
    assert result == [b]


def test_filter_changed_marks_changed_paths_after_check(tmp_path: Path) -> None:
    from contextd.daemon import _filter_changed
    from contextd.indexer.hasher import FileHasher

    hasher = FileHasher()
    f = tmp_path / "f.md"
    f.write_text("content")

    first = _filter_changed([f], hasher)
    assert first == [f]
    second = _filter_changed([f], hasher)
    assert second == []


def test_handle_batch_skips_when_branch_not_allowed(tmp_path: Path) -> None:
    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry, _handle_batch

    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "t", "root": str(tmp_path)}})
    entry = CorpusDaemonEntry(
        corpus_cfg=corpus_cfg,
        store=MagicMock(),
        hasher=MagicMock(),
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=False),
        patch("contextd.daemon.run_incremental_file") as mock_rif,
    ):
        _handle_batch(
            [tmp_path / "a.md"],
            entry,
            inference_concurrency=1,
            incremental_workers=1,
            allowed_branches=[],
        )
    mock_rif.assert_not_called()


def test_handle_batch_skips_when_git_busy(tmp_path: Path) -> None:
    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry, _handle_batch

    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "t", "root": str(tmp_path)}})
    entry = CorpusDaemonEntry(
        corpus_cfg=corpus_cfg,
        store=MagicMock(),
        hasher=MagicMock(),
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=True),
        patch("contextd.daemon.run_incremental_file") as mock_rif,
    ):
        _handle_batch(
            [tmp_path / "a.md"],
            entry,
            inference_concurrency=1,
            incremental_workers=1,
            allowed_branches=[],
        )
    mock_rif.assert_not_called()


def test_handle_batch_calls_run_incremental_for_changed_files(
    tmp_path: Path,
) -> None:
    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry, _handle_batch
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import IncrementalResult

    f = tmp_path / "a.md"
    f.write_text("x")

    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "t", "root": str(tmp_path)}})
    hasher = FileHasher()
    entry = CorpusDaemonEntry(
        corpus_cfg=corpus_cfg,
        store=MagicMock(),
        hasher=hasher,
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=False),
        patch(
            "contextd.daemon.run_incremental_file",
            return_value=IncrementalResult("indexed", str(f)),
        ) as mock_rif,
    ):
        _handle_batch(
            [f],
            entry,
            inference_concurrency=1,
            incremental_workers=1,
            allowed_branches=[],
        )
    mock_rif.assert_called_once()


def test_handle_batch_processes_multiple_files_concurrently(
    tmp_path: Path,
) -> None:
    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry, _handle_batch
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import IncrementalResult

    files = []
    for i in range(4):
        f = tmp_path / f"{i}.md"
        f.write_text("x")
        files.append(f)

    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "t", "root": str(tmp_path)}})
    hasher = FileHasher()
    entry = CorpusDaemonEntry(
        corpus_cfg=corpus_cfg,
        store=MagicMock(),
        hasher=hasher,
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )

    call_times: list[float] = []

    def slow_rif(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        call_times.append(time.monotonic())
        time.sleep(0.05)
        return IncrementalResult("indexed", str(path))

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=False),
        patch("contextd.daemon.run_incremental_file", side_effect=slow_rif),
    ):
        start = time.monotonic()
        _handle_batch(
            files,
            entry,
            inference_concurrency=1,
            incremental_workers=4,
            allowed_branches=[],
        )
        elapsed = time.monotonic() - start

    assert len(call_times) == 4
    assert elapsed < 0.18  # 4 x 0.05s serial = 0.20s; concurrent ~0.05s


def test_handle_batch_logs_and_continues_on_error(tmp_path: Path) -> None:
    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry, _handle_batch
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import IncrementalResult

    files = [tmp_path / f"{i}.md" for i in range(2)]
    for f in files:
        f.write_text("x")

    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "t", "root": str(tmp_path)}})
    hasher = FileHasher()
    entry = CorpusDaemonEntry(
        corpus_cfg=corpus_cfg,
        store=MagicMock(),
        hasher=hasher,
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )

    results = []

    def side_effect(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path).endswith("0.md"):
            raise RuntimeError("store failure")
        results.append(path)
        return IncrementalResult("indexed", str(path))

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=False),
        patch("contextd.daemon.run_incremental_file", side_effect=side_effect),
    ):
        _handle_batch(
            files,
            entry,
            inference_concurrency=1,
            incremental_workers=2,
            allowed_branches=[],
        )

    assert len(results) == 1  # second file still processed


def test_write_and_read_pid(tmp_path: Path) -> None:
    from contextd.daemon import _read_pid, _write_pid

    pid_file = tmp_path / "indexer.pid"
    _write_pid(pid_file, 12345)
    assert _read_pid(pid_file) == 12345


def test_read_pid_returns_none_when_missing(tmp_path: Path) -> None:
    from contextd.daemon import _read_pid

    assert _read_pid(tmp_path / "missing.pid") is None


def test_read_pid_returns_none_on_corrupt_content(tmp_path: Path) -> None:
    from contextd.daemon import _read_pid

    pid_file = tmp_path / "bad.pid"
    pid_file.write_text("not-a-number")
    assert _read_pid(pid_file) is None


def test_daemon_config_default_poll_interval() -> None:
    from contextd.daemon import DaemonConfig

    cfg = DaemonConfig(corpora=[])
    assert cfg.poll_interval_seconds == 1.0


def test_handle_batch_saves_checkpoint_before_processing(tmp_path: Path) -> None:
    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry, _handle_batch
    from contextd.indexer.checkpoint import CheckpointStore
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import IncrementalResult

    f = tmp_path / "a.md"
    f.write_text("x")
    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "t", "root": str(tmp_path)}})
    entry = CorpusDaemonEntry(
        corpus_cfg=corpus_cfg,
        store=MagicMock(),
        hasher=FileHasher(),
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )
    ckpt_store = MagicMock(spec=CheckpointStore)
    call_order: list[str] = []
    ckpt_store.save.side_effect = lambda *a, **kw: call_order.append("save")

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=False),
        patch(
            "contextd.daemon.run_incremental_file",
            side_effect=lambda *a, **kw: (
                call_order.append("rif"),
                IncrementalResult("indexed", str(f)),
            )[1],
        ),
    ):
        _handle_batch(
            [f],
            entry,
            inference_concurrency=1,
            incremental_workers=1,
            allowed_branches=[],
            checkpoint_store=ckpt_store,
        )

    assert call_order == ["save", "rif", "save"]


def test_handle_batch_clears_checkpoint_after_batch_completes(tmp_path: Path) -> None:
    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry, _handle_batch
    from contextd.indexer.checkpoint import CheckpointStore
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import IncrementalResult

    f = tmp_path / "a.md"
    f.write_text("x")
    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "t", "root": str(tmp_path)}})
    entry = CorpusDaemonEntry(
        corpus_cfg=corpus_cfg,
        store=MagicMock(),
        hasher=FileHasher(),
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )
    ckpt_store = MagicMock(spec=CheckpointStore)

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=False),
        patch(
            "contextd.daemon.run_incremental_file",
            return_value=IncrementalResult("indexed", str(f)),
        ),
    ):
        _handle_batch(
            [f],
            entry,
            inference_concurrency=1,
            incremental_workers=1,
            allowed_branches=[],
            checkpoint_store=ckpt_store,
        )

    ckpt_store.clear.assert_called_once_with("t")


def test_handle_batch_does_not_clear_checkpoint_on_error(tmp_path: Path) -> None:
    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry, _handle_batch
    from contextd.indexer.checkpoint import CheckpointStore
    from contextd.indexer.hasher import FileHasher

    f = tmp_path / "a.md"
    f.write_text("x")
    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "t", "root": str(tmp_path)}})
    entry = CorpusDaemonEntry(
        corpus_cfg=corpus_cfg,
        store=MagicMock(),
        hasher=FileHasher(),
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )
    ckpt_store = MagicMock(spec=CheckpointStore)

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=False),
        patch("contextd.daemon.run_incremental_file", side_effect=RuntimeError("store down")),
    ):
        _handle_batch(
            [f],
            entry,
            inference_concurrency=1,
            incremental_workers=1,
            allowed_branches=[],
            checkpoint_store=ckpt_store,
        )

    ckpt_store.clear.assert_not_called()
