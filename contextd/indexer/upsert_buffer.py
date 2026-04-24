"""JSONL-backed buffer of file paths that failed incremental indexing."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from contextd.indexer.pipeline import run_incremental_file

if TYPE_CHECKING:
    from contextd.daemon import CorpusDaemonEntry

_log = logging.getLogger(__name__)


class PendingUpsertBuffer:
    """JSONL-backed buffer of file paths that failed incremental indexing.

    Location: ~/.contextd/state/pending-upserts.jsonl
    Format: one JSON object per line: {"path": "/abs/path/to/file.md", "corpus": "name"}

    On daemon startup, call replay() to re-attempt all buffered paths before
    starting the watchers. Successfully replayed entries are removed from the
    buffer. Entries that fail replay remain for the next startup.
    """

    def __init__(self, buffer_path: Path) -> None:
        self._buffer_path = buffer_path

    def append(self, path: Path, corpus_name: str) -> None:
        """Append a failed-path record atomically (atomic via tmp + os.replace)."""
        records = self.load()
        records.append({"path": str(path), "corpus": corpus_name})
        tmp = self._buffer_path.with_suffix(".tmp")
        self._buffer_path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        os.replace(tmp, self._buffer_path)

    def load(self) -> list[dict[str, str]]:
        """Read all buffered records; return [] if file absent or corrupt line."""
        if not self._buffer_path.exists():
            return []
        records: list[dict[str, str]] = []
        for line in self._buffer_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                _log.warning("upsert buffer: skipping corrupt line: %r", line)
        return records

    def clear(self) -> None:
        """Delete the buffer file (called after successful full replay)."""
        self._buffer_path.unlink(missing_ok=True)

    def replay(
        self,
        corpus_lookup: Callable[[str], CorpusDaemonEntry | None],
        *,
        inference_concurrency: int = 1,
    ) -> tuple[int, int]:
        """Attempt run_incremental_file for each buffered path.

        Returns (succeeded, failed). Removes succeeded entries; failed entries
        remain in the buffer for the next restart.
        """
        records = self.load()
        if not records:
            return (0, 0)
        succeeded = 0
        failed_records: list[dict[str, str]] = []
        for rec in records:
            entry = corpus_lookup(rec["corpus"])
            if entry is None:
                _log.warning(
                    "upsert buffer: corpus %r not found; deferring %s",
                    rec["corpus"],
                    rec["path"],
                )
                failed_records.append(rec)
                continue
            try:
                run_incremental_file(
                    Path(rec["path"]),
                    entry.corpus_cfg,
                    entry.store,
                    entry.hasher,
                    entry.embedder,
                    entry.summariser,
                    entry.inferrer,
                    entry.entity_sampler,
                    inference_concurrency=inference_concurrency,
                )
                succeeded += 1
            except Exception as exc:
                _log.warning("upsert buffer: replay failed for %s: %s", rec["path"], exc)
                failed_records.append(rec)
        if failed_records:
            tmp = self._buffer_path.with_suffix(".tmp")
            tmp.write_text("\n".join(json.dumps(r) for r in failed_records) + "\n")
            os.replace(tmp, self._buffer_path)
        else:
            self.clear()
        return (succeeded, len(failed_records))
