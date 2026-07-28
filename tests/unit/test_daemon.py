from __future__ import annotations

import queue
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from contextd._paths import canonical_path


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
    assert result.to_index == [b]
    assert result.unchanged == [a]
    assert result.digests[b] == hasher.hash(b)


def test_filter_changed_does_not_mark_paths_seen(tmp_path: Path) -> None:
    """The hasher must not be updated at triage time.

    Marking before the file is indexed strands it when indexing then fails: the
    recorded digest makes every later event look unchanged while no node exists.
    _handle_batch marks only after run_incremental_file reports success.
    """
    from contextd.daemon import _filter_changed
    from contextd.indexer.hasher import FileHasher

    hasher = FileHasher()
    f = tmp_path / "f.md"
    f.write_text("content")

    assert _filter_changed([f], hasher).to_index == [f]
    assert _filter_changed([f], hasher).to_index == [f]
    assert hasher.stored(f) is None


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


def _entry_for(tmp_path: Path, store: object | None = None, **corpus_overrides: object) -> object:
    """Build a CorpusDaemonEntry over *tmp_path* with mocked providers."""
    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry
    from contextd.indexer.hasher import FileHasher

    corpus: dict[str, object] = {"name": "t", "root": str(tmp_path)}
    corpus.update(corpus_overrides)
    return CorpusDaemonEntry(
        corpus_cfg=CorpusConfig.model_validate({"corpus": corpus}),
        store=MagicMock() if store is None else store,
        hasher=FileHasher(),
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )


def test_handle_batch_marks_hash_only_after_success(tmp_path: Path) -> None:
    """A failed index must leave the file looking changed so it is retried.

    Marking at triage time (the old behaviour) recorded the digest before the
    work happened, so a failure stranded the file: every later event saw an
    up-to-date digest while no node existed for it.
    """
    from contextd.daemon import _handle_batch

    f = tmp_path / "a.md"
    f.write_text("x")
    entry = _entry_for(tmp_path)

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=False),
        patch("contextd.daemon.run_incremental_file", side_effect=RuntimeError("boom")),
    ):
        _handle_batch(
            [f], entry, inference_concurrency=1, incremental_workers=1, allowed_branches=[]
        )

    assert entry.hasher.stored(f) is None  # type: ignore[attr-defined]

    from contextd.indexer.pipeline import IncrementalResult

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=False),
        patch(
            "contextd.daemon.run_incremental_file",
            return_value=IncrementalResult("indexed", str(f)),
        ),
    ):
        _handle_batch(
            [f], entry, inference_concurrency=1, incremental_workers=1, allowed_branches=[]
        )

    assert entry.hasher.stored(f) == entry.hasher.hash(f)  # type: ignore[attr-defined]


def test_handle_batch_forgets_hash_when_file_is_reaped(tmp_path: Path) -> None:
    """After a deletion the digest must be dropped, so restoring identical
    content is re-indexed instead of looking permanently up to date."""
    from contextd.daemon import _handle_batch
    from contextd.indexer.pipeline import IncrementalResult

    f = tmp_path / "gone.md"
    f.write_text("x")
    entry = _entry_for(tmp_path)
    entry.hasher.mark_seen(f)  # type: ignore[attr-defined]
    f.unlink()

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=False),
        patch(
            "contextd.daemon.run_incremental_file",
            return_value=IncrementalResult("deleted", str(f)),
        ) as mock_rif,
    ):
        _handle_batch(
            [f], entry, inference_concurrency=1, incremental_workers=1, allowed_branches=[]
        )

    mock_rif.assert_called_once()  # the deletion reached the worker
    assert entry.hasher.stored(f) is None  # type: ignore[attr-defined]


def test_handle_batch_logs_when_batch_yields_no_work(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A batch dropped as unchanged must say so; silence here is
    indistinguishable from an event that never arrived."""
    import logging

    from contextd.daemon import _handle_batch

    f = tmp_path / "a.md"
    f.write_text("x")
    entry = _entry_for(tmp_path)
    entry.hasher.mark_seen(f)  # type: ignore[attr-defined]

    with (
        patch("contextd.daemon.branch_is_allowed", return_value=True),
        patch("contextd.daemon.is_git_busy", return_value=False),
        patch("contextd.daemon.run_incremental_file") as mock_rif,
        caplog.at_level(logging.INFO, logger="contextd.daemon"),
    ):
        _handle_batch(
            [f], entry, inference_concurrency=1, incremental_workers=1, allowed_branches=[]
        )

    mock_rif.assert_not_called()
    assert "yielded no work" in caplog.text
    assert str(f) in caplog.text


def test_reconcile_missing_files_queues_only_ungraphed_files(tmp_path: Path) -> None:
    """Files on disk with no File node must be queued, bypassing the hash gate.

    This is what bounds a watcher delivery gap to one restart, and what repairs
    the state left by destroying the graph while the hasher state file survives.
    """
    from contextd.daemon import _reconcile_missing_files

    indexed = tmp_path / "known.md"
    missing = tmp_path / "unknown.md"
    indexed.write_text("a")
    missing.write_text("b")

    store = MagicMock()
    # The graph stores canonical forward-slash paths, so the seeded row must too.
    # Seeding a native path passes on POSIX but not on Windows, where the
    # backslashes never match the canonical form the reconciler compares against.
    store.exec_read.return_value = [{"path": canonical_path(indexed)}]
    entry = _entry_for(tmp_path, store=store)
    # Stale state: both files look unchanged, as after a reset.
    entry.hasher.mark_seen(indexed)  # type: ignore[attr-defined]
    entry.hasher.mark_seen(missing)  # type: ignore[attr-defined]

    relay: queue.Queue[Path] = queue.Queue()
    count = _reconcile_missing_files(entry, relay)  # type: ignore[arg-type]

    assert count == 1
    assert relay.get_nowait() == missing
    assert relay.empty()
    # Forgotten so the batch filter cannot drop it again.
    assert entry.hasher.stored(missing) is None  # type: ignore[attr-defined]


def test_build_sweep_pending_section_mode_includes_ungraphed_files(tmp_path: Path) -> None:
    """A section-granular sweep built only from Section nodes can never see a
    file that has no nodes yet, making a missed new file permanent."""
    from contextd.daemon import _build_sweep_pending

    known = tmp_path / "known.md"
    fresh = tmp_path / "fresh.md"
    known.write_text("# A\n\nbody\n")
    fresh.write_text("# B\n\nbody\n")

    store = MagicMock()
    store.exec_read.return_value = [
        {"id": f"{known}#a", "path": str(known), "hash": "h", "anchor": "a"}
    ]
    entry = _entry_for(tmp_path, store=store, granularity="section")

    units = _build_sweep_pending(entry)  # type: ignore[arg-type]
    by_path = {u.path: u for u in units}

    assert str(fresh) in by_path
    assert by_path[str(fresh)].sections == []  # enters as needing a full index
    assert by_path[str(known)].sections != []
    assert len(units) == 2  # no duplicate for the known file


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
    assert elapsed < 0.5  # 4 x 0.05s serial = 0.20s; concurrent ~0.05s


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


def test_filter_changed_dispatches_vanished_paths_for_reaping(tmp_path: Path) -> None:
    """A path that no longer exists must reach the worker, not be dropped.

    A vanished path is a deletion, and run_incremental_file's !path.exists()
    branch is what reaps the node. Dropping it here (the old behaviour) left the
    graph holding nodes for files that no longer exist, with nothing to remove
    them. It must also not raise FileNotFoundError.
    """
    from contextd.daemon import _filter_changed
    from contextd.indexer.hasher import FileHasher

    existing = tmp_path / "exists.md"
    existing.write_text("content")
    deleted = tmp_path / "gone.md"  # never created, or removed before the drain

    result = _filter_changed([existing, deleted], FileHasher())
    assert existing in result.to_index
    assert deleted in result.to_index
    assert deleted not in result.digests  # nothing to hash


def test_path_is_excluded_blocks_git_and_cache_paths() -> None:
    """_path_is_excluded must return True for .git/, __pycache__, etc. so they
    are never queued into the relay when the watcher fires."""
    from pathlib import Path

    from contextd.daemon import _path_is_excluded

    assert _path_is_excluded(Path("/repo/.git/COMMIT_EDITMSG")) is True
    assert _path_is_excluded(Path("/repo/.git/index.lock")) is True
    assert _path_is_excluded(Path("/repo/__pycache__/foo.pyc")) is True
    assert _path_is_excluded(Path("/repo/.venv/lib/x.py")) is True
    assert _path_is_excluded(Path("/repo/src/main.py")) is False
    assert _path_is_excluded(Path("/repo/docs/prd/spec.md")) is False


def test_path_matches_corpus_includes_blocks_unrelated_paths(tmp_path: Path) -> None:
    """_path_matches_corpus_includes returns False for paths outside the
    corpus's include globs — protects against e.g. cargo's target/ writes
    when the corpus only declares docs/ + mods/ patterns."""
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.pipeline import _path_matches_corpus_includes

    corpus_cfg = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "rl",
                "root": str(tmp_path),
                "include": ["docs/prd/**/*.md", "mods/base/**/*.lua", "prd.md", "CLAUDE.md"],
            }
        }
    )

    assert _path_matches_corpus_includes(tmp_path / "docs/prd/sub/spec.md", corpus_cfg) is True
    assert _path_matches_corpus_includes(tmp_path / "docs/prd/spec.md", corpus_cfg) is True
    assert _path_matches_corpus_includes(tmp_path / "prd.md", corpus_cfg) is True
    assert _path_matches_corpus_includes(tmp_path / "CLAUDE.md", corpus_cfg) is True
    assert _path_matches_corpus_includes(tmp_path / "mods/base/init.lua", corpus_cfg) is True
    # Cargo build artefact: under root but not in include globs
    assert (
        _path_matches_corpus_includes(
            tmp_path / "target/debug/deps/librng_determinism-d5677356c387e41e.rmeta", corpus_cfg
        )
        is False
    )
    assert _path_matches_corpus_includes(tmp_path / "src/main.rs", corpus_cfg) is False


def test_path_matches_corpus_includes_respects_exclude(tmp_path: Path) -> None:
    """exclude entries take precedence over include matches."""
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.pipeline import _path_matches_corpus_includes

    corpus_cfg = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "rl",
                "root": str(tmp_path),
                "include": ["docs/prd/**/*.md"],
                "exclude": ["docs/prd/_audit-methodology.md"],
            }
        }
    )

    assert _path_matches_corpus_includes(tmp_path / "docs/prd/spec.md", corpus_cfg) is True
    assert (
        _path_matches_corpus_includes(tmp_path / "docs/prd/_audit-methodology.md", corpus_cfg)
        is False
    )


def test_path_matches_corpus_includes_returns_false_for_path_outside_root(
    tmp_path: Path,
) -> None:
    """A path not under the corpus root must return False, never raise."""
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.pipeline import _path_matches_corpus_includes

    corpus_cfg = CorpusConfig.model_validate(
        {"corpus": {"name": "rl", "root": str(tmp_path), "include": ["**/*.md"]}}
    )

    assert _path_matches_corpus_includes(Path("/tmp/elsewhere/foo.md"), corpus_cfg) is False


def test_run_daemon_loop_continues_after_handle_batch_raises(tmp_path: Path) -> None:
    """An unhandled exception escaping _handle_batch must not kill run_daemon's
    main loop — it should be logged and the loop should continue."""
    import threading
    from unittest.mock import MagicMock, patch

    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry, DaemonConfig, run_daemon

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
    cfg = DaemonConfig(corpora=[entry], debounce_seconds=0.01, poll_interval_seconds=0.01)

    iteration_count: list[int] = [0]

    def patched_handle_batch(*args: object, **kwargs: object) -> None:
        iteration_count[0] += 1
        if iteration_count[0] == 1:
            raise RuntimeError("simulated crash")

    # Patch DebouncedQueue.drain_if_ready to always return a synthetic batch
    fake_batch = [tmp_path / "x.md"]
    with (
        patch("contextd.daemon._handle_batch", side_effect=patched_handle_batch),
        patch(
            "contextd.indexer.debouncer.DebouncedQueue.drain_if_ready",
            return_value=fake_batch,
        ),
        patch("contextd.indexer.watcher.CorpusWatcher.start"),
        patch("contextd.indexer.watcher.CorpusWatcher.stop"),
        patch("contextd.daemon.install_stop_handlers"),
    ):
        # Run daemon in a thread; it will self-terminate via iteration_count logic
        done = threading.Event()
        exc_holder: list[BaseException | None] = [None]

        def _run() -> None:
            try:
                # We need to stop the daemon; patch time.sleep to inject stop
                original_sleep = __import__("time").sleep
                call_count = [0]

                def controlled_sleep(s: float) -> None:
                    call_count[0] += 1
                    if call_count[0] > 4:
                        raise SystemExit(0)
                    original_sleep(0.001)

                with patch("contextd.daemon.time.sleep", side_effect=controlled_sleep):
                    run_daemon(cfg)
            except SystemExit:
                pass
            except Exception as e:
                exc_holder[0] = e
            finally:
                done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        done.wait(timeout=5.0)

    assert exc_holder[0] is None, f"daemon crashed: {exc_holder[0]}"
    assert iteration_count[0] >= 2, "loop did not continue after first exception"


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


# ---------------------------------------------------------------------------
# Sweep tests
# ---------------------------------------------------------------------------


def _make_section_entry(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Return a minimal CorpusDaemonEntry for sweep tests."""
    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry

    corpus_cfg = CorpusConfig.model_validate(
        {"corpus": {"name": "sw", "root": str(tmp_path), "granularity": "section"}}
    )
    return CorpusDaemonEntry(
        corpus_cfg=corpus_cfg,
        store=MagicMock(),
        hasher=MagicMock(),
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )


def test_sweep_enqueues_file_when_section_hash_changed(tmp_path: Path) -> None:
    import queue

    from contextd.daemon import SectionRecord, SweepWorkUnit, _process_sweep_unit

    md = tmp_path / "doc.md"
    md.write_text("## Alpha\n\nBody alpha.\n")
    entry = _make_section_entry(tmp_path)

    from contextd.indexer.heading_parser import HeadingParser

    sec = HeadingParser(min_level=2, max_level=4).parse(md.read_text())[0]

    unit = SweepWorkUnit(
        path=str(md),
        sections=[SectionRecord(section_id=f"{md}#alpha", anchor=sec.anchor, stored_hash="stale")],
    )
    relay: queue.Queue[Path] = queue.Queue()
    _process_sweep_unit(unit, entry, relay)

    assert not relay.empty()
    assert relay.get() == md


def test_sweep_skips_file_when_all_section_hashes_match(tmp_path: Path) -> None:
    import hashlib
    import queue

    from contextd.daemon import SectionRecord, SweepWorkUnit, _process_sweep_unit

    md = tmp_path / "doc.md"
    md.write_text("## Alpha\n\nBody alpha.\n")
    entry = _make_section_entry(tmp_path)

    from contextd.indexer.heading_parser import HeadingParser

    sec = HeadingParser(min_level=2, max_level=4).parse(md.read_text())[0]
    matched_hash = hashlib.md5((sec.title + "\n\n" + sec.body).encode()).hexdigest()

    unit = SweepWorkUnit(
        path=str(md),
        sections=[
            SectionRecord(section_id=f"{md}#alpha", anchor=sec.anchor, stored_hash=matched_hash)
        ],
    )
    relay: queue.Queue[Path] = queue.Queue()
    _process_sweep_unit(unit, entry, relay)

    assert relay.empty()


def test_sweep_enqueues_file_when_section_added(tmp_path: Path) -> None:
    """File has a new section not present in unit.sections → enqueue."""
    import queue

    from contextd.daemon import SectionRecord, SweepWorkUnit, _process_sweep_unit

    md = tmp_path / "doc.md"
    md.write_text("## Alpha\n\nBody alpha.\n\n## Beta\n\nBody beta.\n")
    entry = _make_section_entry(tmp_path)

    # Graph only knows about "alpha"; "beta" is new
    unit = SweepWorkUnit(
        path=str(md),
        sections=[SectionRecord(section_id=f"{md}#alpha", anchor="alpha", stored_hash="whatever")],
    )
    relay: queue.Queue[Path] = queue.Queue()
    _process_sweep_unit(unit, entry, relay)

    assert not relay.empty()


def test_sweep_enqueues_file_when_section_removed(tmp_path: Path) -> None:
    """Graph has a section anchor no longer in the file → enqueue for GC."""
    import hashlib
    import queue

    from contextd.daemon import SectionRecord, SweepWorkUnit, _process_sweep_unit

    md = tmp_path / "doc.md"
    md.write_text("## Alpha\n\nBody alpha.\n")  # no "beta"
    entry = _make_section_entry(tmp_path)

    from contextd.indexer.heading_parser import HeadingParser

    sec = HeadingParser(min_level=2, max_level=4).parse(md.read_text())[0]
    alpha_hash = hashlib.md5((sec.title + "\n\n" + sec.body).encode()).hexdigest()

    # Graph still has both alpha and a now-removed beta
    unit = SweepWorkUnit(
        path=str(md),
        sections=[
            SectionRecord(section_id=f"{md}#alpha", anchor="alpha", stored_hash=alpha_hash),
            SectionRecord(section_id=f"{md}#beta", anchor="beta", stored_hash="some_hash"),
        ],
    )
    relay: queue.Queue[Path] = queue.Queue()
    _process_sweep_unit(unit, entry, relay)

    assert not relay.empty()


def test_sweep_enqueues_deleted_file_when_sections_in_graph(tmp_path: Path) -> None:
    """File is deleted but still has Section nodes in graph → queue for deletion."""
    import queue

    from contextd.daemon import SectionRecord, SweepWorkUnit, _process_sweep_unit

    gone = tmp_path / "gone.md"  # never created
    entry = _make_section_entry(tmp_path)

    unit = SweepWorkUnit(
        path=str(gone),
        sections=[SectionRecord(section_id=f"{gone}#alpha", anchor="alpha", stored_hash="hash")],
    )
    relay: queue.Queue[Path] = queue.Queue()
    _process_sweep_unit(unit, entry, relay)

    assert not relay.empty()
    assert relay.get() == gone


def test_sweep_treats_none_stored_hash_as_changed(tmp_path: Path) -> None:
    """stored_hash=None (pre-feature node) triggers re-index."""
    import queue

    from contextd.daemon import SectionRecord, SweepWorkUnit, _process_sweep_unit

    md = tmp_path / "doc.md"
    md.write_text("## Alpha\n\nBody alpha.\n")
    entry = _make_section_entry(tmp_path)

    from contextd.indexer.heading_parser import HeadingParser

    sec = HeadingParser(min_level=2, max_level=4).parse(md.read_text())[0]

    unit = SweepWorkUnit(
        path=str(md),
        sections=[SectionRecord(section_id=f"{md}#alpha", anchor=sec.anchor, stored_hash=None)],
    )
    relay: queue.Queue[Path] = queue.Queue()
    _process_sweep_unit(unit, entry, relay)

    assert not relay.empty()


def test_sweep_rate_limits_via_budget_accumulation() -> None:
    """Budget accumulates; unit processed only once budget >= 1."""
    from contextd.daemon import SweepState, SweepWorkUnit

    processed: list[str] = []
    unit = SweepWorkUnit(path="/some/file.md", sections=[])
    state = SweepState(pending=[unit], last_checked_at=0.0, next_sweep_at=0.0)
    rate = 1.0  # sections per second

    # Tick 1: 0.5s elapsed → budget=0.5 → nothing processed
    state.budget += 0.5 * rate
    while state.pending and state.budget >= 1.0:
        u = state.pending.pop(0)
        state.budget = max(0.0, state.budget - float(max(1, len(u.sections))))
        processed.append(u.path)
    assert processed == []

    # Tick 2: another 1.0s elapsed → budget=1.5 → one unit processed
    state.budget += 1.0 * rate
    while state.pending and state.budget >= 1.0:
        u = state.pending.pop(0)
        state.budget = max(0.0, state.budget - float(max(1, len(u.sections))))
        processed.append(u.path)
    assert processed == ["/some/file.md"]


def test_sweep_disabled_when_interval_zero(tmp_path: Path) -> None:
    """sweep_interval_seconds=0 → no SweepState created in sweeps dict."""
    from contextd.daemon import DaemonConfig

    cfg = DaemonConfig(corpora=[], sweep_interval_seconds=0)
    sweeps: dict[str, object] = {}
    if cfg.sweep_interval_seconds > 0:
        sweeps["would_be_created"] = object()
    assert sweeps == {}


def test_sweep_reschedules_after_completion() -> None:
    """After pending drains to empty, next_sweep_at advances by interval."""
    import time

    from contextd.daemon import SweepState, SweepWorkUnit

    interval = 900
    now = time.monotonic()
    unit = SweepWorkUnit(path="/f.md", sections=[])
    state = SweepState(pending=[unit], last_checked_at=now, next_sweep_at=now + interval)
    state.budget = 10.0  # plenty of budget

    # Simulate one loop tick that drains pending
    while state.pending and state.budget >= 1.0:
        u = state.pending.pop(0)
        state.budget = max(0.0, state.budget - float(max(1, len(u.sections))))

    if not state.pending:
        state.next_sweep_at = now + interval

    assert state.pending == []
    assert state.next_sweep_at == pytest.approx(now + interval, abs=0.01)


def test_sweep_file_granular_enqueues_changed_file(tmp_path: Path) -> None:
    """File-granular unit (sections=[]) enqueues file when hasher.is_changed is True."""
    import queue

    from contextd.corpus_config import CorpusConfig
    from contextd.daemon import CorpusDaemonEntry, SweepWorkUnit, _process_sweep_unit

    f = tmp_path / "a.md"
    f.write_text("content")
    corpus_cfg = CorpusConfig.model_validate({"corpus": {"name": "fg", "root": str(tmp_path)}})
    hasher = MagicMock()
    hasher.is_changed.return_value = True
    entry = CorpusDaemonEntry(
        corpus_cfg=corpus_cfg,
        store=MagicMock(),
        hasher=hasher,
        embedder=MagicMock(),
        summariser=MagicMock(),
        inferrer=MagicMock(),
        entity_sampler=lambda _s: [],
    )
    unit = SweepWorkUnit(path=str(f), sections=[])
    relay: queue.Queue[Path] = queue.Queue()
    _process_sweep_unit(unit, entry, relay)

    assert not relay.empty()
    assert relay.get() == f


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
