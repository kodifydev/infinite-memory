from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import sqlite3
import time


@dataclass(frozen=True)
class SearchHit:
    path: str
    startLine: int
    endLine: int
    score: float
    snippet: str
    source: str = "memory"


class MemoryDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def close(self) -> None:
        self.conn.close()

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(path, chunk_index),
                FOREIGN KEY(path) REFERENCES files(path) ON DELETE CASCADE
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                path UNINDEXED
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
            """
        )
        self.conn.commit()

    def indexed_file(self, path: Path) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM files WHERE path = ?", (str(path),)).fetchone()

    def upsert_file(
        self,
        *,
        path: Path,
        mtime_ns: int,
        size: int,
        content_hash: str,
        chunks: list[tuple[int, int, int, str, list[float]]],
    ) -> None:
        now = time.time()
        path_str = str(path)
        with self.conn:
            self.remove_path(path)
            self.conn.execute(
                """
                INSERT INTO files(path, mtime_ns, size, content_hash, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                  mtime_ns=excluded.mtime_ns,
                  size=excluded.size,
                  content_hash=excluded.content_hash,
                  indexed_at=excluded.indexed_at
                """,
                (path_str, mtime_ns, size, content_hash, now),
            )
            for chunk_index, start_line, end_line, text, embedding in chunks:
                cur = self.conn.execute(
                    """
                    INSERT INTO chunks(
                      path, chunk_index, start_line, end_line, text,
                      embedding, embedding_dim, content_hash, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        path_str,
                        chunk_index,
                        start_line,
                        end_line,
                        text,
                        json.dumps(embedding, separators=(",", ":")),
                        len(embedding),
                        content_hash,
                        now,
                    ),
                )
                rowid = int(cur.lastrowid)
                self.conn.execute(
                    "INSERT INTO chunks_fts(rowid, text, path) VALUES (?, ?, ?)",
                    (rowid, text, path_str),
                )

    def remove_path(self, path: Path | str) -> None:
        path_str = str(path)
        ids = [row[0] for row in self.conn.execute("SELECT id FROM chunks WHERE path = ?", (path_str,))]
        for rowid in ids:
            self.conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
        self.conn.execute("DELETE FROM chunks WHERE path = ?", (path_str,))
        self.conn.execute("DELETE FROM files WHERE path = ?", (path_str,))

    def prune_missing(self, existing_paths: set[str]) -> int:
        rows = self.conn.execute("SELECT path FROM files").fetchall()
        removed = 0
        with self.conn:
            for row in rows:
                if row["path"] not in existing_paths:
                    self.remove_path(row["path"])
                    removed += 1
        return removed

    def count_files(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])

    def count_chunks(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        *,
        max_results: int,
        vector_weight: float,
        lexical_weight: float,
        min_score: float,
    ) -> list[SearchHit]:
        rows = self.conn.execute(
            "SELECT id, path, start_line, end_line, text, embedding FROM chunks"
        ).fetchall()
        lexical_scores = self._lexical_scores(query_text)
        best_lexical = max(lexical_scores.values(), default=0.0) or 1.0

        hits: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            embedding = json.loads(row["embedding"])
            vector_score = _cosine(query_embedding, embedding)
            lexical_score = lexical_scores.get(int(row["id"]), 0.0) / best_lexical
            score = vector_weight * vector_score + lexical_weight * lexical_score
            if score >= min_score:
                hits.append((score, row))

        hits.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                path=row["path"],
                startLine=int(row["start_line"]),
                endLine=int(row["end_line"]),
                score=round(float(score), 6),
                snippet=row["text"],
            )
            for score, row in hits[:max_results]
        ]

    def _lexical_scores(self, query_text: str) -> dict[int, float]:
        terms = [t for t in query_text.replace('"', " ").split() if t.strip()]
        if not terms:
            return {}
        fts_query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms[:12])
        try:
            rows = self.conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts WHERE chunks_fts MATCH ?",
                (fts_query,),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        scores: dict[int, float] = {}
        for row in rows:
            scores[int(row["rowid"])] = 1.0 / (1.0 + abs(float(row["rank"])))
        return scores


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    an = math.sqrt(sum(x * x for x in a))
    bn = math.sqrt(sum(y * y for y in b))
    if not an or not bn:
        return 0.0
    return max(0.0, min(1.0, dot / (an * bn)))
