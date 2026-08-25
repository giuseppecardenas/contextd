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
import logging
import logging.handlers
import os
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextd._compat import install_stop_handlers, ipc_file_name
from contextd._paths import canonical_path
from contextd.indexer.checkpoint import Checkpoint, CheckpointStore
from contextd.indexer.debouncer import DebouncedQueue
from contextd.indexer.git_lock import branch_is_allowed, is_git_busy
from contextd.indexer.hasher import FileHasher
from contextd.indexer.heading_parser import section_hash
from contextd.indexer.pipeline import (
    _DEFAULT_EXCLUDE_DIRS,
    IncrementalResult,
    _path_matches_corpus_includes,
    enumerate_corpus_files,
    run_incremental_file,
)
from contextd.indexer.units import extractor_for
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
    relate: Any  # indexer.phases.RelateDeps; Any matches the sibling fields
    chunking: Any = None  # indexer.chunk_deps.ChunkingDeps | None
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


@dataclass
class BatchTriage:
    """How a debounced batch splits into work and no-ops.

    ``to_index`` holds paths to dispatch, including vanished paths whose nodes
    need reaping. ``digests`` carries the content digest observed for each
    existing path in ``to_index``, so the hasher can be updated post-success
    without re-reading the file. ``unchanged`` holds paths whose digest already
    matched, retained purely so a no-op batch can say why in the log.
    """

    to_index: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    digests: dict[Path, str] = field(default_factory=dict)


def _path_is_excluded(path: Path) -> bool:
    """Return True if *path* contains a default-excluded directory component.

    Mirrors the exclude logic in enumerate_corpus_files so that watchdog events
    for .git temp files, __pycache__, etc. are dropped before entering the relay
    rather than crashing later when the debounce batch drains.
    """
    return any(part in _DEFAULT_EXCLUDE_DIRS for part in path.parts)


def _build_sweep_pending(entry: CorpusDaemonEntry) -> list[SweepWorkUnit]:
    """Build the pending work list for a new sweep pass.

    Section-granular: Section nodes from the graph grouped by file path, unioned
    with a disk enumeration. The union matters: a graph-only list can never
    contain a file that has no Section nodes yet, so any new file the watcher
    failed to deliver would be invisible to every subsequent sweep and therefore
    missed permanently rather than merely late. Files present on disk but absent
    from the graph enter as section-less units, which ``_process_sweep_unit``
    treats as needing a full index.

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
        units = [SweepWorkUnit(path=fp, sections=secs) for fp, secs in by_file.items()]
        known = {canonical_path(fp) for fp in by_file}
        units.extend(
            SweepWorkUnit(path=str(p), sections=[])
            for p in enumerate_corpus_files(entry.corpus_cfg)
            if canonical_path(p) not in known
        )
        return units

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
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # File deleted but still has Section nodes in graph.
            # Queue so run_incremental_file's !path.exists() branch fires.
            relay.put(path)
            return

        extractor = extractor_for(entry.corpus_cfg, Path(unit.path).suffix)
        if extractor is None:
            # Section records exist for a file no unit parser covers — treat
            # as changed so the incremental path can reconcile.
            relay.put(path)
            return
        current_sections = extractor.parse(text)
        current_hashes: dict[str, str] = {sec.anchor: section_hash(sec) for sec in current_sections}

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


def _fmt_paths(paths: list[Path], limit: int = 5) -> str:
    """Render a path list for a log line, truncating past *limit* entries."""
    shown = ", ".join(str(p) for p in paths[:limit])
    if len(paths) > limit:
        return f"{shown} (+{len(paths) - limit} more)"
    return shown


def _filter_changed(paths: list[Path], hasher: FileHasher) -> BatchTriage:
    """Split a debounced batch into work to dispatch and paths to ignore.

    A path is work when its content digest differs from the hasher's record, or
    when it has vanished: a vanished path is a deletion, which
    ``run_incremental_file`` turns into a node reap, so it must reach the worker
    rather than being dropped here. Paths whose digest matches are returned
    separately so the caller can log why a batch produced nothing, which is
    otherwise indistinguishable from an event that never arrived.

    The hasher is deliberately NOT updated here. Recording a digest before the
    file is successfully indexed strands the file if indexing then fails: it
    looks up to date to every later event while no node exists for it. Callers
    mark the digests carried in ``digests`` only after ``run_incremental_file``
    reports success.
    """
    triage = BatchTriage()
    for p in paths:
        try:
            if not p.exists():
                triage.to_index.append(p)
                continue
            digest = hasher.hash(p)
        except OSError:
            # Unreadable or vanished mid-check. Dispatch it so the failure is
            # logged and buffered by the worker instead of silently swallowed.
            triage.to_index.append(p)
            continue
        if digest != hasher.stored(p):
            triage.to_index.append(p)
            triage.digests[p] = digest
        else:
            triage.unchanged.append(p)
    return triage


def _make_relay_callback(
    entry: CorpusDaemonEntry, relay: queue.Queue[Path]
) -> Callable[[Path], None]:
    """Build the watchdog callback that filters events into *relay*.

    Applies the same under-root, exclude and include contract that
    ``enumerate_corpus_files`` enforces, so build artefacts and out-of-scope
    paths are rejected before reaching the pipeline. Rejections are logged at
    debug level: an event silently vanishing here is indistinguishable from one
    the OS never delivered, which makes delivery bugs very hard to isolate.
    """

    def _cb(path: Path) -> None:
        if (
            _path_under(path, Path(entry.corpus_cfg.corpus.root))
            and not _path_is_excluded(path)
            and _path_matches_corpus_includes(path, entry.corpus_cfg)
        ):
            relay.put(path)
        else:
            _log.debug(
                "corpus %s: event for %s dropped (outside corpus scope)",
                entry.corpus_cfg.corpus.name,
                path,
            )

    return _cb


def _start_watcher(entry: CorpusDaemonEntry, relay: queue.Queue[Path]) -> None:
    """Attach and start a fresh watcher for *entry*, replacing any prior one."""
    entry.watcher = CorpusWatcher(
        Path(entry.corpus_cfg.corpus.root), _make_relay_callback(entry, relay)
    )
    entry.watcher.start()


def _reconcile_missing_files(entry: CorpusDaemonEntry, relay: queue.Queue[Path]) -> int:
    """Enqueue corpus files that exist on disk but have no node in the graph.

    The watcher is the only low-latency path into the index, and anything it
    fails to deliver (a bug, a dead observer thread, a change made while the
    daemon was down) is otherwise left to the periodic sweep, which is rate
    limited to roughly one file per minute and has taken days to complete a pass
    on a large corpus. This pass makes the graph authoritative at startup: any
    file with no node is queued immediately, so a delivery gap costs one restart
    rather than being permanent.

    It also repairs the state divergence left by destroying the graph (for
    example ``contextd reset``) while the hasher's state file survives. Every
    file would then hash as unchanged and be filtered out of both the watcher and
    sweep paths, indexing nothing at all with no error anywhere. Queued paths are
    forgotten by the hasher first so the batch filter cannot drop them again.

    Returns the number of files enqueued. Section-granular corpora are covered
    too, since a file with no ``File`` node has no ``Section`` nodes either.
    """
    corpus_name = entry.corpus_cfg.corpus.name
    rows = entry.store.exec_read(
        "MATCH (n:File {corpus: $c}) WHERE n.path IS NOT NULL RETURN n.path AS path",
        {"c": corpus_name},
    )
    indexed = {row["path"] for row in rows}

    missing = [
        p for p in enumerate_corpus_files(entry.corpus_cfg) if canonical_path(p) not in indexed
    ]
    entry.hasher.forget_many(missing)
    for path in missing:
        relay.put(path)
    return len(missing)


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

    triage = _filter_changed(batch, corpus_entry.hasher)
    changed = triage.to_index
    if not changed:
        _log.info(
            "corpus %s: batch of %d path(s) yielded no work; all unchanged: %s",
            corpus_name,
            len(batch),
            _fmt_paths(triage.unchanged),
        )
        return
    if triage.unchanged:
        _log.debug(
            "corpus %s: %d of %d batched path(s) unchanged: %s",
            corpus_name,
            len(triage.unchanged),
            len(batch),
            _fmt_paths(triage.unchanged),
        )

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
                corpus_entry.relate,
                inference_concurrency=inference_concurrency,
                chunking=corpus_entry.chunking,
            )
            if result.action == "deleted":
                corpus_entry.hasher.forget(path)
            else:
                corpus_entry.hasher.mark_seen(path, triage.digests.get(path))
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


def _rechunk_if_config_drifted(entry: CorpusDaemonEntry, *, incremental_workers: int) -> None:
    """Re-chunk a corpus whose stored chunking-config fingerprint is stale."""
    from contextd.indexer.phases_chunks import config_drifted, rechunk_corpus

    if entry.chunking is None:
        return
    name = entry.corpus_cfg.corpus.name
    try:
        if not config_drifted(entry.store, name, entry.chunking.config_fp):
            return
        _log.warning("corpus %s: chunking config changed, re-chunking", name)
        result = rechunk_corpus(
            entry.corpus_cfg, entry.chunking, entry.store, concurrency=incremental_workers
        )
        _log.info(
            "corpus %s: re-chunk complete (processed=%d skipped=%d)",
            name,
            result.processed,
            result.skipped,
        )
    except Exception as exc:
        _log.error("corpus %s: re-chunk after config drift failed: %s", name, exc)


def _write_pid(pid_path: Path, pid: int) -> None:
    pid_path.write_text(str(pid), encoding="utf-8")


def _read_pid(pid_path: Path) -> int | None:
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
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

    install_stop_handlers(_on_stop)

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

    # Phase 1c: chunking-config drift. A changed [chunking] block invalidates
    # every parent's fingerprint; re-chunk now (fingerprint-gated, embedding
    # cost only) rather than waiting for the sweep, which keys on content.
    for entry in config.corpora:
        _rechunk_if_config_drifted(entry, incremental_workers=config.incremental_workers)

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

    # Phase 2b: reconcile disk against the graph — catches anything the watcher
    # never delivered, plus drift accumulated while the daemon was down.
    for entry in config.corpora:
        name = entry.corpus_cfg.corpus.name
        try:
            reconciled = _reconcile_missing_files(entry, relays[name])
        except Exception:
            _log.exception("corpus %s: startup reconciliation failed; continuing", name)
            continue
        if reconciled:
            _log.info(
                "corpus %s: startup reconciliation queued %d file(s) missing from the graph",
                name,
                reconciled,
            )
        else:
            _log.info("corpus %s: startup reconciliation found no missing files", name)

    # Phase 3: start watchers
    for entry in config.corpora:
        name = entry.corpus_cfg.corpus.name
        _start_watcher(entry, relays[name])
        _log.info("watching corpus %s at %s", name, entry.corpus_cfg.corpus.root)

    # Phase 4: start IPC server (if a socket path was provided)
    ipc_server = None
    if ipc_socket_path is not None:
        from contextd.daemon_ipc import IpcServer

        corpus_names = [e.corpus_cfg.corpus.name for e in config.corpora]
        ipc_server = IpcServer(
            ipc_path=ipc_socket_path,
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
                    if entry.watcher is not None and not entry.watcher.is_alive():
                        _log.warning(
                            "corpus %s: watcher thread is dead; restarting it and "
                            "reconciling against the graph to recover missed events",
                            name,
                        )
                        with contextlib.suppress(Exception):
                            entry.watcher.stop()
                        _start_watcher(entry, relays[name])
                        with contextlib.suppress(Exception):
                            recovered = _reconcile_missing_files(entry, relays[name])
                            _log.info(
                                "corpus %s: watcher restarted; queued %d missing file(s)",
                                name,
                                recovered,
                            )

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
    except BaseException as exc:
        # Catch everything (incl. SystemExit, KeyboardInterrupt, signals, thread
        # crashes propagated as BaseException) so a silent death always leaves a
        # message + traceback in the log. Re-raised so exit semantics are preserved.
        _log.exception("daemon terminating: %s: %s", type(exc).__name__, exc)
        raise
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
                relate=deps.relate,
                chunking=deps.chunking,
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

    ipc_socket_path = contextd_home() / ipc_file_name()

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
