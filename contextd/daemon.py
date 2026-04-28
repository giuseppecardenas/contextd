"""Incremental indexer daemon — watches corpus roots and re-indexes changed files.

Thread model:
  watchdog dispatch thread → on_change(path) → relay: queue.Queue[Path]
  main loop: polls relay every poll_interval_seconds, drains into DebouncedQueue,
             dispatches per-corpus batches to _handle_batch.
  _handle_batch: runs branch/git-lock gates, MD5 filter, then dispatches
                 file-level work to a ThreadPoolExecutor(max_workers=incremental_workers).
"""

from __future__ import annotations

import contextlib
import hashlib
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
from contextd.indexer.heading_parser import HeadingParser
from contextd.indexer.pipeline import (
    _DEFAULT_EXCLUDE_DIRS,
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
    sweep_interval_seconds: int = 900
    sweep_rate_sections_per_second: float = 0.017


@dataclass
class SectionRecord:
    """A Section node's identity + stored hash, as retrieved from the graph."""

    section_id: str
    anchor: str
    stored_hash: str | None  # None if Section was indexed before this feature


@dataclass
class SweepWorkUnit:
    """One file's worth of sweep work.

    ``sections`` is non-empty for section-granular corpora.
    Empty list signals file-granular mode.
    """

    path: str
    sections: list[SectionRecord]


@dataclass
class SweepState:
    pending: list[SweepWorkUnit]
    last_checked_at: float
    next_sweep_at: float
    budget: float = 0.0


def _path_is_excluded(path: Path) -> bool:
    """Return True if *path* contains a default-excluded directory component.

    Mirrors the exclude logic in enumerate_corpus_files so that watchdog events
    for .git temp files, __pycache__, etc. are dropped before entering the relay
    rather than crashing later when the debounce batch drains.
    """
    return any(part in _DEFAULT_EXCLUDE_DIRS for part in path.parts)


def _build_sweep_pending(entry: CorpusDaemonEntry) -> list[SweepWorkUnit]:
    """Build the pending work list for a new sweep pass.

    Section-granular: queries Section nodes from graph grouped by file path.
    File-granular: enumerates corpus files from disk.
    """
    corpus_name = entry.corpus_cfg.corpus.name

    if entry.corpus_cfg.corpus.granularity == "section":
        rows = entry.store.exec_read(
            "MATCH (s:Section {corpus: $corpus}) "
            "WHERE s.path IS NOT NULL "
            "RETURN s.id AS id, s.path AS path, s.hash AS hash, s.anchor AS anchor",
            {"corpus": corpus_name},
        )
        by_file: dict[str, list[SectionRecord]] = {}
        for row in rows:
            by_file.setdefault(row["path"], []).append(
                SectionRecord(
                    section_id=row["id"],
                    anchor=row.get("anchor") or "",
                    stored_hash=row.get("hash"),
                )
            )
        return [SweepWorkUnit(path=fp, sections=secs) for fp, secs in by_file.items()]

    return [
        SweepWorkUnit(path=str(p), sections=[]) for p in enumerate_corpus_files(entry.corpus_cfg)
    ]


def _process_sweep_unit(
    unit: SweepWorkUnit,
    entry: CorpusDaemonEntry,
    relay: queue.Queue[Path],
) -> None:
    """Check one sweep work unit; enqueue path in relay if re-indexing is needed."""
    path = Path(unit.path)

    if unit.sections:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            # File deleted but still has Section nodes in graph.
            # Queue so run_incremental_file's !path.exists() branch fires.
            relay.put(path)
            return

        parser = HeadingParser(
            min_level=entry.corpus_cfg.corpus.heading_min_level,
            max_level=entry.corpus_cfg.corpus.heading_max_level,
        )
        current_sections = parser.parse(text)
        current_hashes: dict[str, str] = {
            sec.anchor: hashlib.md5((sec.title + "\n\n" + sec.body).encode()).hexdigest()
            for sec in current_sections
        }

        stored_anchors = {rec.anchor for rec in unit.sections}
        current_anchors = set(current_hashes.keys())

        changed = (
            bool(stored_anchors - current_anchors)
            or bool(current_anchors - stored_anchors)
            or any(
                current_hashes.get(rec.anchor) != rec.stored_hash
                for rec in unit.sections
                if rec.anchor in current_anchors
            )
        )
        if changed:
            relay.put(path)
    else:
        try:
            if entry.hasher.is_changed(path):
                relay.put(path)
        except OSError:
            pass


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
    changed: list[Path] = []
    for p in paths:
        try:
            if hasher.is_changed(p):
                hasher.mark_seen(p)
                changed.append(p)
        except OSError:
            # File was deleted between the watchdog event and the debounce drain
            # (common for .git temp files). Skip rather than crash; the deletion
            # case for corpus files is handled separately via on_deleted (TODO).
            pass
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
    ipc_socket_path: Path | None = None,
) -> None:
    relays: dict[str, queue.Queue[Path]] = {}
    stop_event = threading.Event()

    def _on_stop(signum: int, frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)

    # Phase 1: create relays and debouncers (before crash-recovery and watcher setup)
    debouncers: dict[str, DebouncedQueue] = {}
    for entry in config.corpora:
        name = entry.corpus_cfg.corpus.name
        relays[name] = queue.Queue()
        debouncers[name] = DebouncedQueue(window_seconds=config.debounce_seconds)

    # Phase 1b: initialise sweep states (one per corpus, if sweep is enabled)
    sweeps: dict[str, SweepState] = {}
    if config.sweep_interval_seconds > 0:
        _sweep_start = time.monotonic()
        for entry in config.corpora:
            sweeps[entry.corpus_cfg.corpus.name] = SweepState(
                pending=[],
                last_checked_at=_sweep_start,
                next_sweep_at=_sweep_start + config.sweep_interval_seconds,
            )

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
                if _path_under(path, Path(e.corpus_cfg.corpus.root)) and not _path_is_excluded(
                    path
                ):
                    relay.put(path)

            return _cb

        entry.watcher = CorpusWatcher(root, _make_callback())
        entry.watcher.start()
        _log.info("watching corpus %s at %s", name, root)

    # Phase 4: start IPC server (if a socket path was provided)
    ipc_server = None
    if ipc_socket_path is not None:
        from contextd.daemon_ipc import IpcServer

        corpus_names = [e.corpus_cfg.corpus.name for e in config.corpora]
        ipc_server = IpcServer(
            socket_path=ipc_socket_path,
            stop_event=stop_event,
            pid=os.getpid(),
            corpora=corpus_names,
            start_time=time.time(),
        )
        ipc_server.start()

    try:
        while not stop_event.is_set():
            for entry in config.corpora:
                name = entry.corpus_cfg.corpus.name
                try:
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

                    if config.sweep_interval_seconds > 0 and name in sweeps:
                        sweep = sweeps[name]
                        now = time.monotonic()
                        elapsed = now - sweep.last_checked_at
                        sweep.last_checked_at = now

                        if not sweep.pending and now >= sweep.next_sweep_at:
                            sweep.pending = _build_sweep_pending(entry)
                            sweep.budget = 0.0
                            _log.info(
                                "corpus %s: sweep started (%d files, %d sections)",
                                name,
                                len(sweep.pending),
                                sum(len(u.sections) for u in sweep.pending),
                            )
                        elif sweep.pending:
                            sweep.budget += elapsed * config.sweep_rate_sections_per_second
                            while sweep.pending and sweep.budget >= 1.0:
                                unit = sweep.pending.pop(0)
                                cost = float(max(1, len(unit.sections)))
                                sweep.budget = max(0.0, sweep.budget - cost)
                                try:
                                    _process_sweep_unit(unit, entry, relays[name])
                                except Exception:
                                    _log.exception(
                                        "corpus %s: sweep error processing %s",
                                        name,
                                        unit.path,
                                    )
                            if not sweep.pending:
                                sweep.next_sweep_at = now + config.sweep_interval_seconds
                                _log.info(
                                    "corpus %s: sweep complete, next in %ds",
                                    name,
                                    config.sweep_interval_seconds,
                                )
                except Exception:
                    _log.exception("corpus %s: unhandled error in main loop; skipping batch", name)
            time.sleep(config.poll_interval_seconds)
    finally:
        if ipc_server is not None:
            ipc_server.stop()
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

    # Quiet noisy third-party loggers. Neo4j's server-side "Cartesian product"
    # performance notifications and httpx/google_genai per-call traces drown
    # the daemon's own INFO lines; keep warnings/errors from them.
    for noisy in ("neo4j.notifications", "httpx", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

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

    for entry in entries:
        entry.store.connect()

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
        sweep_interval_seconds=cfg.indexer.sweep_interval_seconds,
        sweep_rate_sections_per_second=cfg.indexer.sweep_rate_sections_per_second,
    )

    ipc_socket_path = contextd_home() / "ipc.sock"

    try:
        run_daemon(
            daemon_cfg,
            checkpoint_store=checkpoint_store,
            upsert_buffer=upsert_buffer,
            ipc_socket_path=ipc_socket_path,
        )
    finally:
        for entry in entries:
            with contextlib.suppress(Exception):
                entry.store.close()
        pid_path.unlink(missing_ok=True)
