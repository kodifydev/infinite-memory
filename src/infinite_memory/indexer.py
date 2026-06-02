from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from .chunking import chunk_markdown
from .config import Config
from .db import MemoryDB, SearchHit
from .embeddings import EmbeddingProvider


@dataclass(frozen=True)
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0


class MemoryIndexer:
    def __init__(self, config: Config, db: MemoryDB, embedder: EmbeddingProvider):
        self.config = config
        self.db = db
        self.embedder = embedder

    def iter_markdown_files(self) -> list[Path]:
        files: set[Path] = set()
        excludes = set(self.config.watch.exclude_dirs)
        for root in self.config.watch.paths:
            if root.is_file() and root.suffix.lower() == ".md":
                files.add(root.resolve())
                continue
            if not root.exists():
                continue
            for glob in self.config.watch.include_globs:
                for path in root.glob(glob):
                    if not path.is_file() or path.suffix.lower() != ".md":
                        continue
                    if any(part in excludes for part in path.parts):
                        continue
                    files.add(path.resolve())
        return sorted(files)

    def index_all(self, *, force: bool = False) -> IndexStats:
        scanned = indexed = skipped = chunks = 0
        paths = self.iter_markdown_files()
        for path in paths:
            scanned += 1
            did_index, chunk_count = self.index_file(path, force=force)
            if did_index:
                indexed += 1
                chunks += chunk_count
            else:
                skipped += 1
        removed = self.db.prune_missing({str(p) for p in paths})
        return IndexStats(scanned=scanned, indexed=indexed, skipped=skipped, removed=removed, chunks=chunks)

    def index_file(self, path: Path, *, force: bool = False) -> tuple[bool, int]:
        path = path.resolve()
        if path.suffix.lower() != ".md" or not path.exists():
            self.db.remove_path(path)
            return True, 0
        stat = path.stat()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        current = self.db.indexed_file(path)
        if (
            not force
            and current is not None
            and current["content_hash"] == digest
            and int(current["mtime_ns"]) == stat.st_mtime_ns
        ):
            return False, 0

        chunks = chunk_markdown(path, self.config.chunking)
        texts = [chunk.text for chunk in chunks]
        embeddings: list[list[float]] = []
        batch_size = max(1, self.config.embedding.batch_size)
        for start in range(0, len(texts), batch_size):
            embeddings.extend(self.embedder.embed(texts[start : start + batch_size]))

        rows = [
            (chunk.index, chunk.start_line, chunk.end_line, chunk.text, embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.db.upsert_file(
            path=path,
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            content_hash=digest,
            chunks=rows,
        )
        return True, len(rows)

    def remove_file(self, path: Path) -> None:
        self.db.remove_path(path.resolve())

    def search(self, query: str, *, max_results: int = 5, min_score: float | None = None) -> list[SearchHit]:
        query_embedding = self.embedder.embed([query])[0]
        return self.db.search(
            query_embedding,
            query,
            max_results=max(1, max_results),
            vector_weight=self.config.search.vector_weight,
            lexical_weight=self.config.search.lexical_weight,
            min_score=self.config.search.min_score if min_score is None else min_score,
        )
