from __future__ import annotations

import json
from pathlib import Path

from infinite_memory.config import default_config_path, load_config, resolve_config_path
from infinite_memory.db import MemoryDB
from infinite_memory.embeddings import build_embedding_provider
from infinite_memory.indexer import MemoryIndexer
from infinite_memory.cli import main


def write_config(tmp_path: Path, notes: Path) -> Path:
    cfg = tmp_path / "memory.toml"
    cfg.write_text(
        "\n".join([
            "[watch]",
            f"paths = [\"{notes}\"]",
            f"db_path = \"{tmp_path / 'index.sqlite'}\"",
            "",
            "[chunking]",
            "max_lines = 4",
            "overlap_lines = 1",
            "min_chars = 5",
            "",
            "[embedding]",
            "provider = \"hash\"",
            "dimension = 128",
            "batch_size = 2",
            "",
            "[search]",
            "vector_weight = 0.8",
            "lexical_weight = 0.2",
        ]),
        encoding="utf-8",
    )
    return cfg


def test_indexes_markdown_and_searches_with_file_and_lines(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    incident = notes / "incident.md"
    incident.write_text(
        "# Shipping labels\n\nSEUR pickup labels failed for mixed frozen boxes.\nThe workaround was to reprint warehouse stickers.\n",
        encoding="utf-8",
    )
    other = notes / "recipe.md"
    other.write_text("# Cake\n\nChocolate sponge and orange zest.\n", encoding="utf-8")
    cfg = write_config(tmp_path, notes)

    config = load_config(cfg)
    db = MemoryDB(config.watch.db_path)
    try:
        indexer = MemoryIndexer(config, db, build_embedding_provider(config.embedding))
        stats = indexer.index_all(force=True)
        assert stats.indexed == 2
        assert stats.chunks >= 2

        hits = indexer.search("SEUR frozen warehouse labels", max_results=2)
        assert hits
        assert hits[0].path.endswith("incident.md")
        assert hits[0].startLine == 1
        assert hits[0].endLine >= 3
        assert "SEUR pickup labels" in hits[0].snippet
        assert 0 <= hits[0].score <= 1
    finally:
        db.close()


def test_reindexes_changed_markdown(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    note = notes / "state.md"
    note.write_text("# First\n\nOld apple content.\n", encoding="utf-8")
    cfg = write_config(tmp_path, notes)
    config = load_config(cfg)
    db = MemoryDB(config.watch.db_path)
    try:
        indexer = MemoryIndexer(config, db, build_embedding_provider(config.embedding))
        indexer.index_all(force=True)
        note.write_text("# Second\n\nNew banana content.\n", encoding="utf-8")
        did_index, _ = indexer.index_file(note, force=False)
        assert did_index
        hits = indexer.search("banana", max_results=1)
        assert hits[0].path.endswith("state.md")
        assert "banana" in hits[0].snippet
    finally:
        db.close()


def test_cli_index_and_search_json(tmp_path: Path, capsys) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "memory.md").write_text("# Billing\n\nHolded invoice directions need review.\n", encoding="utf-8")
    cfg = write_config(tmp_path, notes)

    assert main(["index", "--config", str(cfg), "--force"]) == 0
    assert main(["search", "--config", str(cfg), "Holded directions", "--json"]) == 0
    output = capsys.readouterr().out.strip().splitlines()
    payload = json.loads("\n".join(output[1:]))
    assert payload["success"] is True
    assert payload["results"][0]["path"].endswith("memory.md")
    assert "startLine" in payload["results"][0]


def test_cli_uses_global_config_option_before_subcommand(tmp_path: Path, capsys) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "memory.md").write_text("# Search\n\nGlobal config option works.\n", encoding="utf-8")
    cfg = write_config(tmp_path, notes)

    assert main(["--config", str(cfg), "index", "--force"]) == 0
    assert main(["--config", str(cfg), "search", "Global config", "--json"]) == 0
    output = capsys.readouterr().out.strip().splitlines()
    payload = json.loads("\n".join(output[1:]))
    assert payload["success"] is True
    assert payload["results"][0]["path"].endswith("memory.md")


def test_config_resolution_prefers_env_then_xdg(tmp_path: Path, monkeypatch) -> None:
    explicit_cfg = tmp_path / "explicit.toml"
    env_cfg = tmp_path / "env.toml"
    xdg_home = tmp_path / "xdg-config"
    monkeypatch.setenv("INFINITE_MEMORY_CONFIG", str(env_cfg))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))

    assert resolve_config_path(explicit_cfg) == explicit_cfg.resolve()
    assert resolve_config_path() == env_cfg.resolve()

    monkeypatch.delenv("INFINITE_MEMORY_CONFIG")
    assert default_config_path() == xdg_home / "infinite-memory" / "config.toml"
    assert resolve_config_path() == xdg_home / "infinite-memory" / "config.toml"


def test_cli_init_writes_default_xdg_config(tmp_path: Path, monkeypatch, capsys) -> None:
    xdg_config = tmp_path / "config-home"
    xdg_data = tmp_path / "data-home"
    notes = tmp_path / "notes"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))
    monkeypatch.delenv("INFINITE_MEMORY_CONFIG", raising=False)

    assert main(["init", "--path", str(notes), "--provider", "hash", "--model", "hash"]) == 0
    capsys.readouterr()
    cfg = xdg_config / "infinite-memory" / "config.toml"
    assert cfg.exists()
    config = load_config()
    assert config.watch.paths == [notes.resolve()]
    assert config.watch.db_path == xdg_data / "infinite-memory" / "index.sqlite"
    assert config.embedding.provider == "hash"
    assert config.embedding.model == "hash"


def test_lexical_bm25_prefers_stronger_exact_matches(tmp_path: Path) -> None:
    db = MemoryDB(tmp_path / "index.sqlite")
    try:
        exact = tmp_path / "exact.md"
        broad = tmp_path / "broad.md"
        db.upsert_file(
            path=exact,
            mtime_ns=1,
            size=1,
            content_hash="exact",
            chunks=[
                (
                    0,
                    1,
                    1,
                    "Appstle always_invoice_customers fiscal_id obligatorio",
                    [1.0],
                )
            ],
        )
        db.upsert_file(
            path=broad,
            mtime_ns=1,
            size=1,
            content_hash="broad",
            chunks=[(0, 1, 1, "obligatorio memory_search config", [1.0])],
        )

        hits = db.search(
            [],
            "Appstle always_invoice_customers fiscal_id obligatorio",
            max_results=2,
            vector_weight=0.0,
            lexical_weight=1.0,
            min_score=0.0,
        )
        assert hits[0].path == str(exact)
        assert hits[0].score > hits[1].score
    finally:
        db.close()
