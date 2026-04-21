"""MD5-based file hasher with persistent state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class FileHasher:
    def __init__(self, state_path: Path | None = None) -> None:
        self._state_path = state_path
        self._state: dict[str, str] = self._load_state()

    def hash(self, path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    def is_changed(self, path: Path) -> bool:
        current = self.hash(path)
        previous = self._state.get(str(path))
        return current != previous

    def mark_seen(self, path: Path) -> None:
        self._state[str(path)] = self.hash(path)
        self._persist()

    def _load_state(self) -> dict[str, str]:
        if not (self._state_path and self._state_path.exists()):
            return {}
        raw = json.loads(self._state_path.read_text())
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
        if self._state_path:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self._state, indent=2))
