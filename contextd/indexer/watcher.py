"""Cross-platform filesystem watcher using `watchdog`.

watchdog resolves to inotify on Linux, FSEvents on macOS, and
ReadDirectoryChangesW on Windows. From WSL2 Ubuntu, inotify is used
transparently on the WSL filesystem (spec §12.6.3 Path A).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver


class CorpusWatcher:
    def __init__(self, root: Path, on_change: Callable[[Path], None]) -> None:
        self._root = root
        self._on_change = on_change
        self._observer: BaseObserver | None = None

    def start(self) -> None:
        class _H(FileSystemEventHandler):
            def on_modified(inner, event: FileSystemEvent) -> None:  # noqa: N805
                if not event.is_directory:
                    self._on_change(Path(os.fsdecode(event.src_path)))

            def on_created(inner, event: FileSystemEvent) -> None:  # noqa: N805
                if not event.is_directory:
                    self._on_change(Path(os.fsdecode(event.src_path)))

        self._observer = Observer()
        self._observer.schedule(_H(), str(self._root), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
