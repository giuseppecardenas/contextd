"""Cross-platform filesystem watcher using `watchdog`.

watchdog resolves to inotify on Linux, FSEvents on macOS, and
ReadDirectoryChangesW on Windows. From WSL2 Ubuntu, inotify is used
transparently on the WSL filesystem (spec §12.6.3 Path A).

Thread-safety: the ``on_change`` callback fires on watchdog's dispatch
thread, NOT the caller's thread. Downstream consumers (e.g. the M5
``DebouncedQueue``) are not thread-safe; wire through a
``queue.Queue`` or add a lock before the callback mutates shared state.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

_log = logging.getLogger(__name__)


class CorpusWatcher:
    def __init__(self, root: Path, on_change: Callable[[Path], None]) -> None:
        self._root = root
        self._on_change = on_change
        self._observer: BaseObserver | None = None

    def start(self) -> None:
        if self._observer is not None:
            raise RuntimeError(
                "CorpusWatcher already started; call stop() before starting again. "
                "Double-start would orphan the first observer thread (still running, "
                "still holding inotify watches, still firing callbacks)."
            )

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
            if self._observer.is_alive():
                _log.warning(
                    "CorpusWatcher observer thread did not join within 5s; "
                    "orphaning the thread. Process exit will reap it."
                )
            self._observer = None
