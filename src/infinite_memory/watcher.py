from __future__ import annotations

from pathlib import Path
import threading
import time

from .indexer import MemoryIndexer


class MarkdownChangeHandler:
    def __init__(self, indexer: MemoryIndexer, debounce_seconds: float = 0.5):
        self.indexer = indexer
        self.debounce_seconds = debounce_seconds
        self._pending: dict[Path, float] = {}
        self._lock = threading.Lock()

    def mark(self, path: str | Path) -> None:
        p = Path(path)
        if p.suffix.lower() != ".md":
            return
        with self._lock:
            self._pending[p.resolve()] = time.time()

    def drain_ready(self) -> list[Path]:
        now = time.time()
        ready: list[Path] = []
        with self._lock:
            for path, ts in list(self._pending.items()):
                if now - ts >= self.debounce_seconds:
                    ready.append(path)
                    del self._pending[path]
        return ready

    def process_ready(self) -> None:
        for path in self.drain_ready():
            if path.exists():
                self.indexer.index_file(path, force=False)
                print(f"indexed {path}", flush=True)
            else:
                self.indexer.remove_file(path)
                print(f"removed {path}", flush=True)


def watch(indexer: MemoryIndexer) -> None:
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except Exception:
        _poll_watch(indexer)
        return

    handler = MarkdownChangeHandler(indexer)

    class _Handler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                handler.mark(event.src_path)

        def on_modified(self, event):
            if not event.is_directory:
                handler.mark(event.src_path)

        def on_moved(self, event):
            if not event.is_directory:
                handler.mark(event.src_path)
                handler.mark(event.dest_path)

        def on_deleted(self, event):
            if not event.is_directory:
                handler.mark(event.src_path)

    observer = Observer()
    for root in indexer.config.watch.paths:
        if root.exists():
            observer.schedule(_Handler(), str(root), recursive=True)
    observer.start()
    print("watching markdown changes; press Ctrl+C to stop", flush=True)
    try:
        while True:
            handler.process_ready()
            time.sleep(0.2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def _poll_watch(indexer: MemoryIndexer) -> None:
    print("watchdog unavailable; falling back to polling", flush=True)
    indexer.index_all(force=False)
    known = {str(path): path.stat().st_mtime_ns for path in indexer.iter_markdown_files() if path.exists()}
    try:
        while True:
            time.sleep(indexer.config.watch.poll_interval_seconds)
            current_paths = indexer.iter_markdown_files()
            current = {str(path): path.stat().st_mtime_ns for path in current_paths if path.exists()}
            for path_str, mtime in current.items():
                if known.get(path_str) != mtime:
                    indexer.index_file(Path(path_str), force=False)
                    print(f"indexed {path_str}", flush=True)
            for path_str in set(known) - set(current):
                indexer.remove_file(Path(path_str))
                print(f"removed {path_str}", flush=True)
            known = current
    except KeyboardInterrupt:
        return
