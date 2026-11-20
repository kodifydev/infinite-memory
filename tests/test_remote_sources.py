from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import stat
import threading
from urllib.parse import parse_qs, urlsplit

import pytest

from infinite_memory.cli import main
from infinite_memory.config import load_config
from infinite_memory.db import MemoryDB
from infinite_memory.embeddings import build_embedding_provider
from infinite_memory.indexer import MemoryIndexer
from infinite_memory.sources import mirror_relative_path

TOKEN = "local-test-token"


class FakeSource:
    """A real loopback HTTP server speaking the changes protocol over an in-memory store."""

    def __init__(self) -> None:
        self.sequence = 0
        # id -> (sequence, path, content, deleted)
        self.documents: dict[str, tuple[int, str, str, bool]] = {}
        self.requests: list[dict[str, str]] = []
        self.status = 200
        self.forget_cursors_before = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urlsplit(self.path)
                query = {key: values[0] for key, values in parse_qs(url.query).items()}
                owner.requests.append(query)
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return self._reply(401, {"detail": "unauthorized"})
                if url.path != "/memory/changes":
                    return self._reply(404, {"detail": "not found"})
                if owner.status != 200:
                    return self._reply(owner.status, {"detail": "boom"})
                self._reply(*owner.changes(query.get("cursor"), int(query["limit"])))

            def _reply(self, status, payload):
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/memory"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def put(self, doc_id: str, path: str, content: str) -> None:
        self.sequence += 1
        self.documents[doc_id] = (self.sequence, path, content, False)

    def delete(self, doc_id: str) -> None:
        self.sequence += 1
        _, path, _, _ = self.documents[doc_id]
        self.documents[doc_id] = (self.sequence, path, "", True)

    def changes(self, cursor: str | None, limit: int):
        since = int(cursor) if cursor is not None else 0
        if cursor is not None and since < self.forget_cursors_before:
            return 410, {"detail": "cursor expired"}
        changed = sorted(
            (seq, doc_id, path, content, deleted)
            for doc_id, (seq, path, content, deleted) in self.documents.items()
            # A first listing has nothing to say about documents already gone.
            if seq > since and not (cursor is None and deleted)
        )
        page = changed[:limit]
        next_cursor = str(page[-1][0] if page else max(since, self.sequence))
        return 200, {
            "protocol": 1,
            "documents": [
                {"id": doc_id, "deleted": True} if deleted
                else {"id": doc_id, "path": path, "content": content, "deleted": False}
                for _seq, doc_id, path, content, deleted in page
            ],
            "next_cursor": next_cursor,
            "has_more": len(changed) > limit,
        }


@pytest.fixture
def source(monkeypatch):
    fake = FakeSource()
    monkeypatch.setenv("TEST_SOURCE_TOKEN", TOKEN)
    yield fake
    fake.close()


def write_config(tmp_path: Path, source: FakeSource | None = None, *, notes: Path | None = None,
                 page_size: int = 100, min_interval: float = 0) -> Path:
    lines = [
        "[watch]",
        f"paths = [{json.dumps(str(notes))}]" if notes else "paths = []",
        f"db_path = {json.dumps(str(tmp_path / 'index.sqlite'))}",
        "",
        "[chunking]",
        "tokens = 40",
        "overlap = 8",
        "",
        "[embedding]",
        'provider = "hash"',
        "dimension = 64",
        "batch_size = 4",
        "",
        "[search]",
        "min_score = 0.0",
    ]
    if source is not None:
        lines += [
            "",
            "[[sources]]",
            'name = "wiki"',
            f"url = {json.dumps(source.url)}",
            f"mirror_path = {json.dumps(str(tmp_path / 'mirror'))}",
            'token_env = "TEST_SOURCE_TOKEN"',
            f"page_size = {page_size}",
            f"min_interval_seconds = {min_interval}",
        ]
    config = tmp_path / "memory.toml"
    config.write_text("\n".join(lines), encoding="utf-8")
    return config


class CountingEmbedder:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return self.inner.embed(texts)


def build(config_path: Path):
    config = load_config(config_path)
    db = MemoryDB(config.watch.db_path)
    embedder = CountingEmbedder(build_embedding_provider(config.embedding))
    return MemoryIndexer(config, db, embedder), db, embedder


# --- Only what changed is embedded again ------------------------------------------------


def test_file_that_grows_only_embeds_its_new_chunks(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    journal = notes / "journal.md"
    paragraphs = [f"Entry {number}: " + f"word{number} " * 60 for number in range(6)]
    journal.write_text("\n\n".join(paragraphs), encoding="utf-8")
    indexer, db, embedder = build(write_config(tmp_path, notes=notes))
    try:
        first = indexer.index_all()
        embedder.texts.clear()
        journal.write_text("\n\n".join([*paragraphs, "Entry 6: " + "fresh " * 60]), encoding="utf-8")

        second = indexer.index_all()

        assert first.embedded == first.chunks > 3
        assert second.indexed == 1 and second.chunks > first.chunks
        # The file was re-chunked whole, but only its tail was sent to the embedding model.
        assert 0 < second.embedded == len(embedder.texts) < second.chunks
        assert all("fresh" in text or "word5" in text for text in embedder.texts)
        assert indexer.search("fresh", max_results=1)[0].path == str(journal.resolve())
        assert indexer.search("word2", max_results=1)[0].path == str(journal.resolve())
    finally:
        db.close()


def test_forced_index_embeds_everything_again(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "note.md").write_text("alpha beta gamma delta " * 80, encoding="utf-8")
    indexer, db, embedder = build(write_config(tmp_path, notes=notes))
    try:
        first = indexer.index_all()
        again = indexer.index_all(force=True)
        untouched = indexer.index_all()

        assert again.embedded == again.chunks == first.chunks
        assert (untouched.indexed, untouched.embedded) == (0, 0)
    finally:
        db.close()


# --- Remote sources ---------------------------------------------------------------------


def test_remote_documents_are_mirrored_indexed_and_searchable(tmp_path: Path, source: FakeSource) -> None:
    source.put("d1", "logistics/shipping.md", "# Envíos\n\nSEUR recoge los pedidos a las 17:00.\n")
    source.put("d2", "finance/margins", "# Márgenes\n\nLa caja de temporada deja un 32 % de margen.\n")
    indexer, db, _ = build(write_config(tmp_path, source))
    try:
        stats = indexer.index_all()

        mirror = tmp_path / "mirror"
        shipping = mirror / "logistics" / "shipping.md"
        assert shipping.read_text(encoding="utf-8").startswith("# Envíos")
        # A path without extension still becomes a Markdown file, or it would not be indexed.
        assert (mirror / "finance" / "margins.md").is_file()
        assert stat.S_IMODE(shipping.stat().st_mode) == 0o444
        assert [(r.name, r.ok, r.written, r.removed) for r in stats.sources] == [("wiki", True, 2, 0)]
        assert stats.indexed == 2
        hit = indexer.search("margen caja de temporada", max_results=1)[0]
        assert hit.path == str((mirror / "finance" / "margins.md").resolve())
        assert "32 %" in hit.snippet
    finally:
        db.close()


def test_folder_filled_by_hand_becomes_a_read_only_mirror(tmp_path: Path, source: FakeSource) -> None:
    mirror = tmp_path / "mirror"
    (mirror / "area").mkdir(parents=True)
    existing = mirror / "area" / "note.md"
    existing.write_text("ya estaba aquí", encoding="utf-8")
    source.put("d1", "area/note.md", "ya estaba aquí")
    indexer, db, _ = build(write_config(tmp_path, source))
    try:
        stats = indexer.index_all()

        # Nothing to write, but the file can no longer be edited in place.
        assert stats.sources[0].written == 0
        assert stat.S_IMODE(existing.stat().st_mode) == 0o444
    finally:
        db.close()


def test_nothing_changed_is_one_empty_answer_and_no_work(tmp_path: Path, source: FakeSource) -> None:
    source.put("d1", "a.md", "uno dos tres")
    indexer, db, embedder = build(write_config(tmp_path, source))
    try:
        indexer.index_all()
        source.requests.clear()
        embedder.texts.clear()

        stats = indexer.index_all()

        assert source.requests == [{"limit": "100", "cursor": "1"}]
        assert (stats.sources[0].written, stats.indexed, embedder.texts) == (0, 0, [])
    finally:
        db.close()


def test_edit_rename_and_delete_follow_the_source(tmp_path: Path, source: FakeSource) -> None:
    source.put("d1", "area/one.md", "primer texto sobre naranjas")
    source.put("d2", "area/two.md", "segundo texto sobre limones")
    indexer, db, _ = build(write_config(tmp_path, source))
    mirror = tmp_path / "mirror"
    try:
        indexer.index_all()
        source.put("d1", "other/renamed.md", "primer texto sobre mandarinas")
        source.delete("d2")

        stats = indexer.index_all()

        assert sorted(p.relative_to(mirror).as_posix() for p in mirror.rglob("*.md")) == ["other/renamed.md"]
        # The folder that became empty goes too.
        assert not (mirror / "area").exists()
        assert (stats.sources[0].written, stats.sources[0].removed, stats.removed) == (1, 1, 2)
        assert indexer.search("mandarinas", max_results=1)[0].path.endswith("other/renamed.md")
        assert db.count_files() == 1
        assert all("limones" not in hit.snippet for hit in indexer.search("limones", max_results=5))
    finally:
        db.close()


def test_changes_are_read_page_by_page(tmp_path: Path, source: FakeSource) -> None:
    for number in range(5):
        source.put(f"d{number}", f"doc-{number}.md", f"documento número {number}")
    indexer, db, _ = build(write_config(tmp_path, source, page_size=2))
    try:
        stats = indexer.index_all()

        assert stats.sources[0].written == 5
        assert [request.get("cursor") for request in source.requests] == [None, "2", "4"]
        assert len(list((tmp_path / "mirror").glob("*.md"))) == 5
    finally:
        db.close()


def test_unreachable_source_keeps_what_was_mirrored(tmp_path: Path, source: FakeSource) -> None:
    source.put("d1", "a.md", "texto que ya estaba sincronizado")
    indexer, db, _ = build(write_config(tmp_path, source))
    try:
        indexer.index_all()
        source.status = 503

        stats = indexer.index_all()

        assert (stats.sources[0].ok, stats.sources[0].error) == (False, "HTTP 503")
        assert (stats.removed, db.count_files()) == (0, 1)
        assert indexer.search("sincronizado", max_results=1)[0].path.endswith("a.md")
    finally:
        db.close()


def test_expired_cursor_lists_everything_again_and_drops_what_is_gone(tmp_path: Path, source: FakeSource) -> None:
    source.put("d1", "keep.md", "se queda")
    source.put("d2", "gone.md", "se fue mientras no mirábamos")
    indexer, db, _ = build(write_config(tmp_path, source))
    mirror = tmp_path / "mirror"
    try:
        indexer.index_all()
        # The source dropped its history, tombstones included.
        del source.documents["d2"]
        source.put("d3", "new.md", "nuevo")
        source.forget_cursors_before = source.sequence

        stats = indexer.index_all()

        assert stats.sources[0].ok
        assert sorted(p.name for p in mirror.glob("*.md")) == ["keep.md", "new.md"]
        assert stats.sources[0].removed == 1
    finally:
        db.close()


def test_min_interval_skips_asking_too_often(tmp_path: Path, source: FakeSource) -> None:
    source.put("d1", "a.md", "uno")
    indexer, db, _ = build(write_config(tmp_path, source, min_interval=3600))
    try:
        indexer.index_all()
        source.requests.clear()
        source.put("d2", "b.md", "dos")

        stats = indexer.index_all()

        assert source.requests == [] and stats.sources[0].skipped
        assert not (tmp_path / "mirror" / "b.md").exists()
    finally:
        db.close()


def test_missing_token_is_reported_not_raised(tmp_path: Path, source: FakeSource, monkeypatch) -> None:
    monkeypatch.delenv("TEST_SOURCE_TOKEN")
    indexer, db, _ = build(write_config(tmp_path, source))
    try:
        stats = indexer.index_all()

        assert not stats.sources[0].ok and "TEST_SOURCE_TOKEN" in stats.sources[0].error
        assert source.requests == []
    finally:
        db.close()


@pytest.mark.parametrize("path", ["../outside.md", "/etc/passwd", "a/../../b.md", ".hidden/x.md", "a\\b.md", ""])
def test_document_paths_never_leave_the_mirror(path: str) -> None:
    with pytest.raises(ValueError):
        mirror_relative_path(path)


def test_a_bad_path_from_the_source_writes_nothing_outside(tmp_path: Path, source: FakeSource) -> None:
    source.put("d1", "../escaped.md", "no debería salir")
    indexer, db, _ = build(write_config(tmp_path, source))
    try:
        stats = indexer.index_all()

        assert not stats.sources[0].ok and "invalid document path" in stats.sources[0].error
        assert not (tmp_path / "escaped.md").exists()
    finally:
        db.close()


def test_mirror_inside_a_watched_folder_is_indexed_once(tmp_path: Path, source: FakeSource) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "local.md").write_text("nota local", encoding="utf-8")
    source.put("d1", "remote.md", "nota remota")
    config = write_config(tmp_path, source, notes=notes)
    config.write_text(
        config.read_text(encoding="utf-8").replace(str(tmp_path / "mirror"), str(notes / "wiki")),
        encoding="utf-8",
    )
    indexer, db, _ = build(config)
    try:
        stats = indexer.index_all()

        assert (stats.scanned, stats.indexed, db.count_files()) == (2, 2, 2)
    finally:
        db.close()


def test_cli_index_and_status_report_the_sources(tmp_path: Path, source: FakeSource, capsys) -> None:
    source.put("d1", "a.md", "uno dos tres")
    config = write_config(tmp_path, source)

    assert main(["--config", str(config), "index"]) == 0
    indexed = json.loads(capsys.readouterr().out)
    assert main(["--config", str(config), "status"]) == 0
    status = json.loads(capsys.readouterr().out)

    assert indexed["sources"] == [{"name": "wiki", "ok": True, "written": 1, "removed": 0}]
    assert indexed["embedded"] == indexed["chunks"] == 1
    assert status["sources"][0]["name"] == "wiki" and status["sources"][0]["documents"] == 1
    assert status["sources"][0]["synced_at"]


def test_config_refuses_sources_it_cannot_use(tmp_path: Path) -> None:
    config = tmp_path / "memory.toml"
    config.write_text('[watch]\npaths = []\n\n[[sources]]\nname = "Bad Name"\nurl = "http://x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="name"):
        load_config(config)
    config.write_text('[watch]\npaths = []\n\n[[sources]]\nname = "ok"\nurl = "ftp://x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="http"):
        load_config(config)
    config.write_text("[watch]\npaths = []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sources"):
        load_config(config)
