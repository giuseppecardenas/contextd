"""Per-corpus checkpoint file for bootstrap resumption (spec §5.9).

Stores the last-committed phase + batch index so a resumed bootstrap
picks up at the next batch, not from phase 5a.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


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
        (self._root / f"{corpus}.json").write_text(json.dumps(asdict(checkpoint)))

    def load(self, corpus: str) -> Checkpoint | None:
        path = self._root / f"{corpus}.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        return Checkpoint(**raw)

    def clear(self, corpus: str) -> None:
        path = self._root / f"{corpus}.json"
        path.unlink(missing_ok=True)
