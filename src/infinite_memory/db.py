from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import sqlite3
import time
from typing import Any

FTS_QUERY_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
VECTOR_CANDIDATE_LIMIT = 200
VECTOR_DISTANCE_METRIC = "cosine"


@dataclass(frozen=True)
class SearchHit:
    path: str
    startLine: int
    endLine: int
    score: float
    snippet: str
    source: str = "memory"


@dataclass
class _Candidate:
    row: sqlite3.Row
    vector_score: float = 0.0
    lexical_score: float = 0.0


class MemoryDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._sqlite_vec: Any | None = None
        self._vector_dim: int | None = None
        self._ensure_schema()
        self._load_sqlite_vec()

    def close(self) -> None:
        self.conn.close()

    @property
    def vector_backend(self) -> str:
        return "sqlite-vec" if self._sqlite_vec is not None else "python"

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
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

    def _load_sqlite_vec(self) -> None:
        try:
            import sqlite_vec

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self._sqlite_vec = sqlite_vec
            stored_dim = self.conn.execute(
                "SELECT value FROM meta WHERE key = 'vector_dim'"
            ).fetchone()
            if stored_dim:
                self._vector_dim = int(stored_dim["value"])
                with self.conn:
                    self._ensure_vector_table(self._vector_dim)
        except Exception:
            try:
                self.conn.enable_load_extension(False)
            except Exception:
                pass
            self._sqlite_vec = None
            self._vector_dim = None

    def _ensure_vector_table(self, dimension: int) -> bool:
        if self._sqlite_vec is None or dimension <= 0:
            return False
        schema_row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'chunks_vec'"
        ).fetchone()
        schema_sql = str(schema_row["sql"] or "").lower().replace(" ", "") if schema_row else ""
        uses_cosine = f"distance_metric={VECTOR_DISTANCE_METRIC}" in schema_sql
        needs_rebuild = self._vector_dim != dimension or not uses_cosine
        if needs_rebuild:
            self.conn.execute("DROP TABLE IF EXISTS chunks_vec")
            self.conn.execute("DELETE FROM meta WHERE key = 'vector_dim'")
            self._vector_dim = None
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec "
            f"USING vec0(embedding float[{dimension}] distance_metric={VECTOR_DISTANCE_METRIC})"
        )
        if self._vector_dim != dimension:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES ('vector_dim', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(dimension),),
            )
            self._vector_dim = dimension
        if needs_rebuild:
            rows = self.conn.execute(
                "SELECT id, embedding FROM chunks WHERE embedding_dim = ?", (dimension,)
            ).fetchall()
            for row in rows:
                embedding = json.loads(row["embedding"])
                self.conn.execute(
                    "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                    (int(row["id"]), self._sqlite_vec.serialize_float32(embedding)),
                )
        return True

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
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
        vector_ready = self._ensure_vector_table(len(chunks[0][4])) if chunks and chunks[0][4] else False
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
                if vector_ready and embedding:
                    self.conn.execute(
                        "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                        (rowid, self._sqlite_vec.serialize_float32(embedding)),
                    )

    def remove_path(self, path: Path | str) -> None:
        path_str = str(path)
        ids = [row[0] for row in self.conn.execute("SELECT id FROM chunks WHERE path = ?", (path_str,))]
        for rowid in ids:
            self.conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
            if self._sqlite_vec is not None:
                try:
                    self.conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
                except sqlite3.OperationalError:
                    pass
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

    def count_vector_rows(self) -> int:
        if self._sqlite_vec is None:
            return 0
        try:
            return int(self.conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0])
        except sqlite3.OperationalError:
            return 0

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        *,
        max_results: int,
        vector_weight: float,
        lexical_weight: float,
        min_score: float,
        candidate_multiplier: int,
    ) -> list[SearchHit]:
        candidate_limit = min(
            VECTOR_CANDIDATE_LIMIT,
            max(1, math.floor(max_results * max(1, candidate_multiplier))),
        )
        candidates: dict[int, _Candidate] = {}

        for row, score in self._vector_candidates(query_embedding, candidate_limit):
            rowid = int(row["id"])
            candidates[rowid] = _Candidate(row=row, vector_score=score)

        keyword_ids: set[int] = set()
        for row, score in self._keyword_candidates(query_text, candidate_limit):
            rowid = int(row["id"])
            keyword_ids.add(rowid)
            candidate = candidates.get(rowid)
            if candidate:
                candidate.lexical_score = score
                candidate.row = row
            else:
                candidates[rowid] = _Candidate(row=row, lexical_score=score)

        ranked: list[tuple[float, int, sqlite3.Row]] = []
        for rowid, candidate in candidates.items():
            score = vector_weight * candidate.vector_score + lexical_weight * candidate.lexical_score
            ranked.append((score, rowid, candidate.row))
        ranked.sort(key=lambda item: item[0], reverse=True)

        strict = [entry for entry in ranked if entry[0] >= min_score]
        if strict or not keyword_ids:
            selected = strict[:max_results]
        else:
            relaxed_min_score = min(min_score, lexical_weight)
            selected = [
                entry for entry in ranked if entry[1] in keyword_ids and entry[0] >= relaxed_min_score
            ][:max_results]

        return [
            SearchHit(
                path=row["path"],
                startLine=int(row["start_line"]),
                endLine=int(row["end_line"]),
                score=round(float(score), 6),
                snippet=row["text"],
            )
            for score, _rowid, row in selected
        ]

    def _vector_candidates(
        self, query_embedding: list[float], limit: int
    ) -> list[tuple[sqlite3.Row, float]]:
        if not query_embedding or limit <= 0:
            return []
        if self._sqlite_vec is not None and self._vector_dim == len(query_embedding):
            try:
                rows = self.conn.execute(
                    """
                    WITH nearest AS (
                        SELECT rowid, distance
                          FROM chunks_vec
                         WHERE embedding MATCH ? AND k = ?
                         ORDER BY distance ASC
                    )
                    SELECT c.id, c.path, c.start_line, c.end_line, c.text,
                           nearest.distance AS dist
                      FROM nearest
                      JOIN chunks c ON c.id = nearest.rowid
                     ORDER BY nearest.distance ASC
                    """,
                    (self._sqlite_vec.serialize_float32(query_embedding), limit),
                ).fetchall()
                return [(row, max(0.0, min(1.0, 1.0 - float(row["dist"])))) for row in rows]
            except sqlite3.Error:
                pass

        rows = self.conn.execute(
            "SELECT id, path, start_line, end_line, text, embedding FROM chunks"
        ).fetchall()
        scored: list[tuple[sqlite3.Row, float]] = []
        for row in rows:
            embedding = json.loads(row["embedding"])
            score = _cosine(query_embedding, embedding)
            if _number_is_finite(score):
                scored.append((row, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _keyword_candidates(self, query_text: str, limit: int) -> list[tuple[sqlite3.Row, float]]:
        fts_query = _build_fts_query(query_text)
        if not fts_query or limit <= 0:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT c.id, c.path, c.start_line, c.end_line, c.text,
                       bm25(chunks_fts) AS rank
                  FROM chunks_fts
                  JOIN chunks c ON c.id = chunks_fts.rowid
                 WHERE chunks_fts MATCH ?
                 ORDER BY rank ASC
                 LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(row, _bm25_rank_to_score(float(row["rank"]))) for row in rows]


def _build_fts_query(raw: str) -> str | None:
    tokens = [token.strip() for token in FTS_QUERY_TOKEN_RE.findall(raw) if token.strip()]
    if not tokens:
        return None
    return " AND ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def _bm25_rank_to_score(rank: float) -> float:
    if not math.isfinite(rank):
        return 1 / 1000
    if rank < 0:
        relevance = -rank
        return relevance / (1 + relevance)
    return 1 / (1 + rank)


def _number_is_finite(value: float) -> bool:
    return math.isfinite(value)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    an = math.sqrt(sum(x * x for x in a))
    bn = math.sqrt(sum(y * y for y in b))
    if not an or not bn:
        return 0.0
    return max(0.0, min(1.0, dot / (an * bn)))
