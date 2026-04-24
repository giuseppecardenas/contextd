"""Incremental indexer daemon — watches corpus roots and re-indexes changed files.

Thread model:
  watchdog dispatch thread → on_change(path) → relay: queue.Queue[Path]
  main loop: polls relay every poll_interval_seconds, drains into DebouncedQueue,
             dispatches per-corpus batches to _handle_batch.
  _handle_batch: runs branch/git-lock gates, MD5 filter, then dispatches
                 file-level work to a ThreadPoolExecutor(max_workers=incremental_workers).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import queue
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextd.indexer.checkpoint import Checkpoint, CheckpointStore
from contextd.indexer.debouncer import DebouncedQueue
from contextd.indexer.git_lock import branch_is_allowed, is_git_busy
from contextd.indexer.hasher import FileHasher
from contextd.indexer.pipeline import (
    IncrementalResult,
    enumerate_corpus_files,
    run_incremental_file,
)
from contextd.indexer.upsert_buffer import PendingUpsertBuffer
from contextd.indexer.watcher import CorpusWatcher

_log = logging.getLogger(__name__)


@dataclass
class CorpusDaemonEntry:
    corpus_cfg: Any
    store: Any
    hasher: FileHasher
    embedder: Any
    summariser: Any
    inferrer: Any
    entity_sampler: Callable[[Any], list[str]]
    watcher: CorpusWatcher | None = field(default=None, init=False)


@dataclass
class DaemonConfig:
    corpora: list[CorpusDaemonEntry]
    debounce_seconds: float = 30.0
    poll_interval_seconds: float = 1.0
    inference_concurrency: int = 1
    incremental_workers: int = 4
    allowed_branches: list[str] = field(default_factory=list)


def _drain_relay_into_debouncer(
    relay: queue.Queue[Path],
    debouncer: DebouncedQueue,
) -> None:
    try:
        while True:
            debouncer.add(relay.get_nowait())
    except queue.Empty:
        pass


def _filter_changed(paths: list[Path], hasher: FileHasher) -> list[Path]:
    changed = [p for p in paths if hasher.is_changed(p)]
    for p in changed:
        hasher.mark_seen(p)
    return changed


def _handle_batch(
    batch: list[Path],
    corpus_entry: CorpusDaemonEntry,
    *,
    inference_concurrency: int,
    incremental_workers: int,
    allowed_branches: list[str],
    checkpoint_store: CheckpointStore | None = None,
    upsert_buffer: PendingUpsertBuffer | None = None,
) -> None:
    corpus_root = Path(corpus_entry.corpus_cfg.corpus.root)
    corpus_name = corpus_entry.corpus_cfg.corpus.name

    if not branch_is_allowed(corpus_root, allowed_branches):
        _log.warning("corpus %s: branch not in allowed_branches; skipping batch", corpus_name)
        return
    if is_git_busy(corpus_root):
        _log.warning("corpus %s: git lock detected; skipping batch", corpus_name)
        return

    changed = _filter_changed(batch, corpus_entry.hasher)
    if not changed:
        return

    # Save initial checkpoint before dispatching — lets a crashed daemon know
    # which files were in-flight on next startup.
    if checkpoint_store is not None:
        checkpoint_store.save(
            corpus_name,
            Checkpoint(
                phase="incremental",
                last_committed_batch=0,
                last_committed_file=str(changed[0]),
            ),
        )

    error_event = threading.Event()
    ckpt_lock = threading.Lock()

    def _process(path: Path) -> IncrementalResult | Exception:
        try:
            result = run_incremental_file(
                path,
                corpus_entry.corpus_cfg,
                corpus_entry.store,
                corpus_entry.hasher,
                corpus_entry.embedder,
                corpus_entry.summariser,
                corpus_entry.inferrer,
                corpus_entry.entity_sampler,
                inference_concurrency=inference_concurrency,
            )
            if checkpoint_store is not None:
                with ckpt_lock:
                    checkpoint_store.save(
                        corpus_name,
                        Checkpoint(
                            phase="incremental",
                            last_committed_batch=0,
                            last_committed_file=str(path),
                        ),
                    )
            return result
        except Exception as exc:
            _log.error("corpus %s: failed to index %s: %s", corpus_name, path, exc)
            error_event.set()
            if upsert_buffer is not None:
                upsert_buffer.append(path, corpus_name)
            return exc

    with ThreadPoolExecutor(max_workers=incremental_workers) as executor:
        futures = {executor.submit(_process, p): p for p in changed}
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, IncrementalResult):
                _log.info("corpus %s: %s %s", corpus_name, result.action, result.path)

    if not error_event.is_set() and checkpoint_store is not None:
        checkpoint_store.clear(corpus_name)


def _write_pid(pid_path: Path, pid: int) -> None:
    pid_path.write_text(str(pid))


def _read_pid(pid_path: Path) -> int | None:
    try:
        return int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return None


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def run_daemon(
    config: DaemonConfig,
    *,
    checkpoint_store: CheckpointStore | None = None,
    upsert_buffer: PendingUpsertBuffer | None = None,
) -> None:
    relays: dict[str, queue.Queue[Path]] = {}
    stop_requested = False

    def _on_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)

    # Phase 1: create relays and debouncers (before crash-recovery and watcher setup)
    debouncers: dict[str, DebouncedQueue] = {}
    for entry in config.corpora:
        name = entry.corpus_cfg.corpus.name
        relays[name] = queue.Queue()
        debouncers[name] = DebouncedQueue(window_seconds=config.debounce_seconds)

    # Phase 2: crash-recovery — re-queue files that were in-flight on last shutdown
    if checkpoint_store is not None:
        for entry in config.corpora:
            name = entry.corpus_cfg.corpus.name
            ckpt = checkpoint_store.load(name)
            if ckpt is not None and ckpt.last_committed_file is not None:
                _log.info(
                    "corpus %s: crash recovery — replaying files since %s",
                    name,
                    ckpt.last_committed_file,
                )
                last_file = ckpt.last_committed_file
                for f in enumerate_corpus_files(entry.corpus_cfg):
                    try:
                        if os.path.getmtime(f) >= os.path.getmtime(last_file):
                            relays[name].put(f)
                    except OSError:
                        relays[name].put(f)

    # Phase 3: start watchers
    for entry in config.corpora:
        name = entry.corpus_cfg.corpus.name
        root = Path(entry.corpus_cfg.corpus.root)

        def _make_callback(
            e: CorpusDaemonEntry = entry,
            relay: queue.Queue[Path] = relays[name],
        ) -> Callable[[Path], None]:
            def _cb(path: Path) -> None:
                if _path_under(path, Path(e.corpus_cfg.corpus.root)):
                    relay.put(path)

            return _cb

        entry.watcher = CorpusWatcher(root, _make_callback())
        entry.watcher.start()
        _log.info("watching corpus %s at %s", name, root)

    try:
        while not stop_requested:
            for entry in config.corpora:
                name = entry.corpus_cfg.corpus.name
                _drain_relay_into_debouncer(relays[name], debouncers[name])
                batch = debouncers[name].drain_if_ready()
                if batch:
                    _handle_batch(
                        batch,
                        entry,
                        inference_concurrency=config.inference_concurrency,
                        incremental_workers=config.incremental_workers,
                        allowed_branches=config.allowed_branches,
                        checkpoint_store=checkpoint_store,
                        upsert_buffer=upsert_buffer,
                    )
            time.sleep(config.poll_interval_seconds)
    finally:
        for entry in config.corpora:
            if entry.watcher is not None:
                entry.watcher.stop()
        _log.info("daemon stopped")


def main() -> None:
    from contextd._paths import contextd_home
    from contextd.cli._shared import _load_cfg
    from contextd.cli.corpora import _build_pipeline_deps
    from contextd.corpus_config import CorpusConfig

    cfg = _load_cfg()

    log_path = Path(cfg.logging.path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=cfg.logging.max_log_bytes,
        backupCount=cfg.logging.log_backup_count,
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.basicConfig(level=cfg.logging.level.upper(), handlers=[handler])

    corpora_dir = contextd_home() / "corpora"
    entries: list[CorpusDaemonEntry] = []
    for toml_path in sorted(corpora_dir.glob("*.toml")):
        try:
            corpus_cfg = CorpusConfig.load(toml_path)
        except Exception as exc:
            _log.warning("skipping corpus %s: %s", toml_path.stem, exc)
            continue
        deps = _build_pipeline_deps(cfg, corpus_cfg, toml_path.stem, toml_path)
        entries.append(
            CorpusDaemonEntry(
                corpus_cfg=corpus_cfg,
                store=deps.store,
                hasher=deps.hasher,
                embedder=deps.embedder,
                summariser=deps.summariser,
                inferrer=deps.inferrer,
                entity_sampler=lambda _s: [],
            )
        )

    state_dir = contextd_home() / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    pid_path = state_dir / "indexer.pid"
    _write_pid(pid_path, os.getpid())
    _log.info("daemon started (pid=%d, corpora=%d)", os.getpid(), len(entries))

    checkpoint_store = CheckpointStore(contextd_home() / "state" / "checkpoints")

    upsert_buffer = PendingUpsertBuffer(state_dir / "pending-upserts.jsonl")
    corpus_lookup = {e.corpus_cfg.corpus.name: e for e in entries}
    succeeded, failed = upsert_buffer.replay(corpus_lookup.get)
    if succeeded or failed:
        _log.info("upsert buffer replay: %d succeeded, %d failed", succeeded, failed)

    daemon_cfg = DaemonConfig(
        corpora=entries,
        debounce_seconds=float(cfg.indexer.debounce_seconds),
        inference_concurrency=cfg.indexer.inference_concurrency,
        incremental_workers=cfg.indexer.incremental_workers,
        allowed_branches=cfg.indexer.allowed_branches,
    )

    try:
        run_daemon(daemon_cfg, checkpoint_store=checkpoint_store, upsert_buffer=upsert_buffer)
    finally:
        pid_path.unlink(missing_ok=True)
