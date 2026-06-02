from __future__ import annotations

import json
from pathlib import Path

from infinite_memory.config import load_config
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
