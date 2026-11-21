from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from .chunking import chunk_markdown
from .config import Config
from .db import MemoryDB, SearchHit
from .embeddings import EmbeddingProvider
from .sources import SourceSyncResult, sync_source


@dataclass(frozen=True)
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0
    # Chunks whose embedding had to be computed; the rest of `chunks` were reused.
    embedded: int = 0
    sources: tuple[SourceSyncResult, ...] = ()


class MemoryIndexer:
    INDEX_SIGNATURE_KEY = "index_signature_v1"

    def __init__(self, config: Config, db: MemoryDB, embedder: EmbeddingProvider):
        self.config = config
        self.db = db
        self.embedder = embedder
        self.embedded_chunks = 0

    def index_signature(self) -> str:
        chunking: dict[str, int | str] = {
            "tokens": self.config.chunking.tokens,
            "overlap": self.config.chunking.overlap,
        }
        # "window" is how every index was built before strategies existed: naming it
        # would make those indexes look stale and embed everything again for nothing.
        if self.config.chunking.strategy != "window":
            chunking["strategy"] = self.config.chunking.strategy
        return json.dumps(
            {
                "chunking": chunking,
                "embedding": {
                    "provider": self.config.embedding.provider,
                    "model": self.config.embedding.model,
                    "dimension": self.config.embedding.dimension,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def iter_markdown_files(self) -> list[Path]:
        files: set[Path] = set()
        excludes = set(self.config.watch.exclude_dirs)
        roots = [*self.config.watch.paths, *(source.mirror_path for source in self.config.sources)]
        for root in roots:
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

    def sync_sources(self) -> tuple[SourceSyncResult, ...]:
        """Bring every remote source's mirror folder up to date; a failing one is reported."""
        return tuple(sync_source(source, self.db) for source in self.config.sources)

    def index_all(self, *, force: bool = False) -> IndexStats:
        scanned = indexed = skipped = chunks = 0
        embedded_before = self.embedded_chunks
        # First, so that what changed in a remote source is indexed in this same pass. A
        # source that cannot be reached keeps its mirrored files, so nothing is pruned.
        sources = self.sync_sources()
        signature = self.index_signature()
        effective_force = force or self.db.get_meta(self.INDEX_SIGNATURE_KEY) != signature
        paths = self.iter_markdown_files()
        for path in paths:
            scanned += 1
            did_index, chunk_count = self.index_file(path, force=effective_force)
            if did_index:
                indexed += 1
                chunks += chunk_count
            else:
                skipped += 1
        removed = self.db.prune_missing({str(p) for p in paths})
        self.db.set_meta(self.INDEX_SIGNATURE_KEY, signature)
        return IndexStats(
            scanned=scanned, indexed=indexed, skipped=skipped, removed=removed, chunks=chunks,
            embedded=self.embedded_chunks - embedded_before, sources=sources,
        )

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
        # A file that grew or changed in one place keeps most of its chunks word for word:
        # those keep their embedding. A forced index (new model or chunking) computes all.
        known = {} if force else self.db.chunk_embeddings(path)
        embeddings: list[list[float] | None] = [
            json.loads(known[text]) if text in known else None for text in texts
        ]
        missing = [index for index, embedding in enumerate(embeddings) if embedding is None]
        batch_size = max(1, self.config.embedding.batch_size)
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            computed = self.embedder.embed([texts[index] for index in batch])
            for index, embedding in zip(batch, computed, strict=True):
                embeddings[index] = embedding
        self.embedded_chunks += len(missing)

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
            candidate_multiplier=self.config.search.candidate_multiplier,
        )
