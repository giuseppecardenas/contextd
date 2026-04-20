"""MD5-based file hasher with persistent state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast


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
        if self._state_path and self._state_path.exists():
            return cast(dict[str, str], json.loads(self._state_path.read_text()))
        return {}

    def _persist(self) -> None:
        if self._state_path:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self._state, indent=2))
