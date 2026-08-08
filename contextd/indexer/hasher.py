"""MD5-based file hasher with persistent state.

Thread-safety: ``mark_seen`` and ``forget`` are called from the daemon's
incremental worker threads, so state mutation and the persist that follows it
are serialised under a lock, and the file is replaced atomically. Without both,
concurrent completions interleave a read-modify-write and lose entries, and a
crash mid-write leaves truncated JSON that ``_load_state`` rejects, which stops
the daemon from starting at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable
from pathlib import Path


class FileHasher:
    def __init__(self, state_path: Path | None = None) -> None:
        self._state_path = state_path
        self._lock = threading.Lock()
        self._state: dict[str, str] = self._load_state()

    def hash(self, path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    def is_changed(self, path: Path) -> bool:
        current = self.hash(path)
        previous = self._state.get(str(path))
        return current != previous

    def stored(self, path: Path) -> str | None:
        """Return the recorded digest for *path*, or None if it is unknown.

        Lets a caller that has already hashed the file compare without a second
        read, and lets a caller distinguish "known and identical" from "never
        seen", which ``is_changed`` collapses into a single boolean.
        """
        return self._state.get(str(path))

    def mark_seen(self, path: Path, digest: str | None = None) -> None:
        """Record *path* as indexed at *digest*, hashing the file if not supplied.

        Callers that hashed the file earlier in the pipeline should pass that
        digest rather than letting this method re-read the file. Re-reading here
        would record whatever is on disk at completion time, which for a file
        edited during indexing means recording the newer content while the graph
        holds the older, causing the newer edit to be filtered out as unchanged
        and lost. Passing the digest observed before indexing errs the other way:
        a concurrent edit is re-indexed redundantly rather than dropped.
        """
        resolved = digest if digest is not None else self.hash(path)
        with self._lock:
            self._state[str(path)] = resolved
            self._persist()

    def forget(self, path: Path) -> None:
        """Drop any recorded digest for *path*; a no-op if it is not recorded.

        Called after a file's node is reaped so that restoring the file later,
        even with byte-identical content, is seen as changed and re-indexed. A
        retained digest would make the restored file look up to date forever
        while no node exists for it in the graph.
        """
        self.forget_many([path])

    def forget_many(self, paths: Iterable[Path]) -> int:
        """Drop recorded digests for *paths* in a single persist.

        Reconciliation can forget an entire corpus at once (for example after the
        graph was destroyed), and persisting per path would rewrite the whole
        state file once per entry. Returns the number of entries removed.
        """
        with self._lock:
            removed = sum(1 for p in paths if self._state.pop(str(p), None) is not None)
            if removed:
                self._persist()
        return removed

    def _load_state(self) -> dict[str, str]:
        if not (self._state_path and self._state_path.exists()):
            return {}
        raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(
                f"Hasher state file {self._state_path} must be a JSON object; got {type(raw).__name__}"
            )
        out: dict[str, str] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(
                    f"Hasher state file {self._state_path} entries must be str→str; "
                    f"got {type(k).__name__}→{type(v).__name__}"
                )
            out[k] = v
        return out

    def _persist(self) -> None:
        """Write state atomically; callers must already hold ``self._lock``."""
        if self._state_path:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
            os.replace(tmp, self._state_path)
