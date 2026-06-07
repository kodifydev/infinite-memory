from __future__ import annotations

import argparse
import json
import sys

from .config import default_config_text, load_config, resolve_config_path
from .db import MemoryDB
from .embeddings import build_embedding_provider
from .indexer import MemoryIndexer
from .watcher import watch


def _build_indexer(config_path: str | None) -> tuple[MemoryIndexer, MemoryDB]:
    config = load_config(config_path)
    db = MemoryDB(config.watch.db_path)
    embedder = build_embedding_provider(config.embedding)
    return MemoryIndexer(config, db, embedder), db


def cmd_init(args: argparse.Namespace) -> int:
    config_path = resolve_config_path(args.config)
    if config_path.exists() and not args.force:
        print(f"Config already exists: {config_path}. Use --force to overwrite.", file=sys.stderr)
        return 2
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        default_config_text(
            args.path,
            db_path=args.db_path,
            provider=args.provider,
            model=args.model,
            deployment=args.deployment,
            api_version=args.api_version,
        ),
        encoding="utf-8",
    )
    print(f"wrote {config_path}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    indexer, db = _build_indexer(args.config)
    try:
        stats = indexer.index_all(force=args.force)
        print(
            json.dumps(
                {
                    "success": True,
                    "scanned": stats.scanned,
                    "indexed": stats.indexed,
                    "skipped": stats.skipped,
                    "removed": stats.removed,
                    "chunks": stats.chunks,
                    "db_path": str(indexer.config.watch.db_path),
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        db.close()


def cmd_search(args: argparse.Namespace) -> int:
    indexer, db = _build_indexer(args.config)
    try:
        hits = indexer.search(args.query, max_results=args.max_results, min_score=args.min_score)
        payload = {
            "success": True,
            "query": args.query,
            "results": [hit.__dict__ for hit in hits],
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for hit in hits:
                print(f"{hit.score:.4f} {hit.path}:{hit.startLine}-{hit.endLine}")
                print(hit.snippet)
                print("---")
        return 0
    finally:
        db.close()


def cmd_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db = MemoryDB(config.watch.db_path)
    try:
        print(
            json.dumps(
                {
                    "success": True,
                    "db_path": str(config.watch.db_path),
                    "files": db.count_files(),
                    "chunks": db.count_chunks(),
                    "paths": [str(p) for p in config.watch.paths],
                    "embedding_provider": config.embedding.provider,
                    "embedding_model": config.embedding.model,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        db.close()


def cmd_watch(args: argparse.Namespace) -> int:
    indexer, db = _build_indexer(args.config)
    try:
        stats = indexer.index_all(force=False)
        print(f"initial index: scanned={stats.scanned} indexed={stats.indexed} skipped={stats.skipped}")
        watch(indexer)
        return 0
    finally:
        db.close()


def _add_config_option(parser: argparse.ArgumentParser, *, dest: str = "config") -> None:
    parser.add_argument(
        "--config",
        dest=dest,
        help=(
            "Config path. Defaults to $INFINITE_MEMORY_CONFIG, then "
            "$XDG_CONFIG_HOME/infinite-memory/config.toml, then "
            "~/.config/infinite-memory/config.toml."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infinite-memory")
    _add_config_option(parser, dest="global_config")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Write a starter config")
    _add_config_option(p_init)
    p_init.add_argument("--path", default="./memory", help="Markdown folder to index")
    p_init.add_argument("--db-path", help="SQLite index path to write into the config")
    p_init.add_argument("--provider", default="azure_openai", help="Embedding provider")
    p_init.add_argument("--model", default="text-embedding-3-large", help="Embedding model")
    p_init.add_argument("--deployment", help="Azure OpenAI embedding deployment name")
    p_init.add_argument("--api-version", help="Azure OpenAI API version")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config")
    p_init.set_defaults(func=cmd_init)

    p_index = sub.add_parser("index", help="Index configured Markdown folders")
    _add_config_option(p_index)
    p_index.add_argument("--force", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Search indexed Markdown chunks")
    _add_config_option(p_search)
    p_search.add_argument("query")
    p_search.add_argument("--max-results", type=int, default=5)
    p_search.add_argument("--min-score", type=float)
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_status = sub.add_parser("status", help="Print index status")
    _add_config_option(p_status)
    p_status.set_defaults(func=cmd_status)

    p_watch = sub.add_parser("watch", help="Watch Markdown changes and update the index")
    _add_config_option(p_watch)
    p_watch.set_defaults(func=cmd_watch)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "config", None) is None:
        args.config = getattr(args, "global_config", None)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
