"""Per-corpus checkpoint file for bootstrap resumption (spec §5.9).

Stores the last-committed phase + batch index so a resumed bootstrap
picks up at the next batch, not from phase 5a.

Single-writer: ``CheckpointStore`` assumes one indexer process per corpus
at a time. Concurrent ``save()`` calls on the same corpus are
undefined — add an external lock before running two indexers against
the same ``root``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


_INVALID_CORPUS_CHARS = frozenset({"/", "\\", "\0"})


def _validate_corpus_name(name: str) -> None:
    """Reject names that would escape the root or hit reserved filesystem forms."""
    if not name:
        raise ValueError("corpus name must not be empty")
    if name in {".", ".."}:
        raise ValueError(f"corpus name {name!r} is reserved (filesystem navigation)")
    if name.startswith("."):
        raise ValueError(f"corpus name {name!r} must not start with '.'")
    if any(c in _INVALID_CORPUS_CHARS for c in name):
        raise ValueError(
            f"corpus name {name!r} contains an invalid character; "
            "path separators and NUL are not allowed"
        )


@dataclass
class Checkpoint:
    phase: str
    last_committed_batch: int
    last_committed_file: str | None


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, corpus: str, checkpoint: Checkpoint) -> None:
        _validate_corpus_name(corpus)
        # Atomic write: the whole point of this module is crash-safe resume.
        # Truncate-then-write (plain write_text) leaves a partial JSON on
        # SIGKILL / power-loss, which defeats the recovery purpose.
        target = self._root / f"{corpus}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(checkpoint)))
        os.replace(tmp, target)

    def load(self, corpus: str) -> Checkpoint | None:
        _validate_corpus_name(corpus)
        path = self._root / f"{corpus}.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(
                f"Checkpoint file {path} must be a JSON object; got {type(raw).__name__}"
            )
        phase = raw.get("phase")
        batch = raw.get("last_committed_batch")
        last_file = raw.get("last_committed_file")
        if not isinstance(phase, str):
            raise ValueError(f"Checkpoint {path}: 'phase' must be a string")
        if not isinstance(batch, int) or isinstance(batch, bool):
            raise ValueError(f"Checkpoint {path}: 'last_committed_batch' must be an int")
        if last_file is not None and not isinstance(last_file, str):
            raise ValueError(f"Checkpoint {path}: 'last_committed_file' must be a string or null")
        return Checkpoint(phase=phase, last_committed_batch=batch, last_committed_file=last_file)

    def clear(self, corpus: str) -> None:
        _validate_corpus_name(corpus)
        path = self._root / f"{corpus}.json"
        path.unlink(missing_ok=True)
