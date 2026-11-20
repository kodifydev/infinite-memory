"""Remote memory sources: documents kept behind an HTTP API, mirrored here as files.

A source answers one question, "what changed since this cursor?":

    GET <url>/changes?limit=100&cursor=<opaque>
    {"protocol": 1,
     "documents": [{"id": "...", "path": "area/topic.md", "content": "...", "deleted": false}],
     "next_cursor": "...", "has_more": false}

Without a cursor the source lists every live document. Each document is written to the
source's mirror folder, which is indexed like any watched folder, so search results and
whatever reads the files afterwards keep working unchanged. The mirror is a copy: edits
belong in the source, and the next change there overwrites the file.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import SourceConfig
from .db import MemoryDB

PROTOCOL_VERSION = 1
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_PATH_CHARS = 512
# A source that keeps saying "more" without moving its cursor would never end.
MAX_PAGES = 10_000


class SourceError(Exception):
    """The source could not be read; what is already mirrored stays as it is."""


class CursorExpired(SourceError):
    """The source no longer knows this cursor (HTTP 410): everything is listed again."""


@dataclass(frozen=True)
class RemoteDocument:
    id: str
    path: str
    content: str
    deleted: bool


@dataclass(frozen=True)
class ChangesPage:
    documents: list[RemoteDocument]
    next_cursor: str
    has_more: bool


@dataclass(frozen=True)
class SourceSyncResult:
    name: str
    ok: bool
    written: int = 0
    removed: int = 0
    # Asked too recently (min_interval_seconds): the source was not contacted.
    skipped: bool = False
    error: str | None = None


def cursor_key(name: str) -> str:
    return f"source_cursor:{name}"


def synced_at_key(name: str) -> str:
    return f"source_synced_at:{name}"


def fetch_changes(source: SourceConfig, cursor: str | None) -> ChangesPage:
    query = {"limit": str(source.page_size)}
    if cursor is not None:
        query["cursor"] = cursor
    request = urllib.request.Request(
        f"{source.url}/changes?{urllib.parse.urlencode(query)}",
        headers={"Accept": "application/json"},
    )
    if source.token_env:
        token = os.getenv(source.token_env)
        if not token:
            raise SourceError(f"environment variable {source.token_env} is not set")
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=source.timeout_seconds) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 410:
            raise CursorExpired("the source no longer knows the saved cursor") from None
        raise SourceError(f"HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SourceError(str(getattr(exc, "reason", exc))) from None
    if len(body) > MAX_RESPONSE_BYTES:
        raise SourceError("response too large; lower page_size")
    try:
        return _parse_page(json.loads(body))
    except (ValueError, KeyError, TypeError) as exc:
        raise SourceError(f"invalid response: {exc}") from None


def _parse_page(data: dict) -> ChangesPage:
    if data.get("protocol", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol {data.get('protocol')}")
    documents: list[RemoteDocument] = []
    for raw in data["documents"]:
        deleted = bool(raw.get("deleted", False))
        doc_id = raw["id"]
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("document id must be a non-empty string")
        path, content = raw.get("path", ""), raw.get("content", "")
        if not deleted and not (isinstance(path, str) and isinstance(content, str)):
            raise ValueError(f"document {doc_id} needs a path and a content")
        documents.append(
            RemoteDocument(id=doc_id, path=str(path or ""), content=str(content or ""), deleted=deleted)
        )
    next_cursor = data["next_cursor"]
    if not isinstance(next_cursor, str) or not next_cursor:
        raise ValueError("next_cursor must be a non-empty string")
    return ChangesPage(documents=documents, next_cursor=next_cursor, has_more=bool(data.get("has_more", False)))


def mirror_relative_path(path: str) -> str:
    """Where a document goes inside the mirror folder; never outside it."""
    if not path or len(path) > MAX_PATH_CHARS or "\\" in path or any(ord(c) < 32 for c in path):
        raise ValueError(f"invalid document path: {path!r}")
    parts = PurePosixPath(path).parts
    if path.startswith("/") or any(part in ("", ".", "..") or part.startswith(".") for part in parts):
        raise ValueError(f"invalid document path: {path!r}")
    relative = "/".join(parts)
    return relative if relative.lower().endswith(".md") else relative + ".md"


def _target(root: Path, relative: str) -> Path:
    target = root / relative
    # A symlink planted inside the mirror must not take a write somewhere else.
    if not target.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"document path leaves the mirror folder: {relative!r}")
    return target


def _write(root: Path, relative: str, content: str) -> bool:
    target = _target(root, relative)
    data = content.encode("utf-8")
    try:
        if target.read_bytes() == data:
            return False
    except OSError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    # Read-only says what it is: a copy. The source is where it gets edited.
    temporary.chmod(0o444)
    os.replace(temporary, target)
    return True


def _remove(root: Path, relative: str) -> None:
    target = root / relative
    target.unlink(missing_ok=True)
    folder = target.parent
    while folder != root and root in folder.parents:
        try:
            folder.rmdir()
        except OSError:
            break
        folder = folder.parent


def sync_source(source: SourceConfig, db: MemoryDB) -> SourceSyncResult:
    """Bring the mirror folder up to date with the source. Never raises."""
    last = db.get_meta(synced_at_key(source.name))
    if source.min_interval_seconds and last and time.time() - float(last) < source.min_interval_seconds:
        return SourceSyncResult(source.name, ok=True, skipped=True)

    root = source.mirror_path
    written = removed = 0
    try:
        root.mkdir(parents=True, exist_ok=True)
        cursor = db.get_meta(cursor_key(source.name))
        full = cursor is None
        known = db.source_documents(source.name)
        seen: set[str] = set()
        for _ in range(MAX_PAGES):
            try:
                page = fetch_changes(source, cursor)
            except CursorExpired:
                if full:
                    raise
                cursor, full = None, True
                seen.clear()
                continue
            for document in page.documents:
                previous = known.get(document.id)
                if document.deleted:
                    if previous is not None:
                        _remove(root, previous)
                        db.remove_source_document(source.name, document.id)
                        del known[document.id]
                        removed += 1
                    continue
                relative = mirror_relative_path(document.path)
                if previous is not None and previous != relative:
                    _remove(root, previous)
                if _write(root, relative, document.content):
                    written += 1
                if previous != relative:
                    db.set_source_document(source.name, document.id, relative)
                    known[document.id] = relative
                seen.add(document.id)
            if page.has_more and page.next_cursor == cursor:
                raise SourceError("the source says there is more but did not move its cursor")
            cursor = page.next_cursor
            if not full:
                # Each page is already applied, so a later failure resumes from here.
                db.set_meta(cursor_key(source.name), cursor)
            if not page.has_more:
                break
        else:
            raise SourceError("the source never stopped saying there is more")
        if full:
            # A full listing is the whole truth: whatever it did not name is gone.
            for doc_id, relative in list(known.items()):
                if doc_id not in seen:
                    _remove(root, relative)
                    db.remove_source_document(source.name, doc_id)
                    removed += 1
            db.set_meta(cursor_key(source.name), cursor)
        db.set_meta(synced_at_key(source.name), str(time.time()))
    except (SourceError, ValueError, OSError) as exc:
        return SourceSyncResult(source.name, ok=False, written=written, removed=removed, error=str(exc))
    return SourceSyncResult(source.name, ok=True, written=written, removed=removed)
