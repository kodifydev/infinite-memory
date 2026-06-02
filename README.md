# infinite-memory

`infinite-memory` indexes Markdown folders into a local SQLite database and lets you search them semantically, returning the text snippet plus the source file and line range — the same shape OpenClaw's `memory_search` exposes.

It currently supports:

- Linux-first file watching for `*.md` changes (`watchdog`/inotify, polling fallback).
- OpenAI embeddings.
- Azure OpenAI embeddings.
- Deterministic local hash embeddings for tests/offline smoke checks.
- SQLite storage with vector cosine ranking + FTS5 lexical fallback/boost.

## Install locally

```bash
uv sync --dev
uv run infinite-memory --help
```

Or, from another project:

```bash
pip install -e /path/to/infinite-memory
```

## Quick start

Create a config:

```bash
uv run infinite-memory init --config ./memory.toml --path ~/my-md-notes
```

Edit the embedding provider in `memory.toml`.

For Azure OpenAI:

```toml
[embedding]
provider = "azure_openai"
model = "text-embedding-3-large"
# Optional; if omitted the CLI reads the environment variables below.
deployment = "text-embedding-3-large"
api_version = "2024-02-01"
```

Required env vars by default:

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
# Optional:
export AZURE_OPENAI_API_VERSION="2024-02-01"
```

For OpenAI:

```toml
[embedding]
provider = "openai"
model = "text-embedding-3-large"
```

Required env var:

```bash
export OPENAI_API_KEY="..."
```

Index and search:

```bash
uv run infinite-memory index --config ./memory.toml --force
uv run infinite-memory search --config ./memory.toml "shipping label bug" --max-results 5 --json
```

Watch for Markdown changes and auto-index them:

```bash
uv run infinite-memory watch --config ./memory.toml
```

## Result shape

`search --json` returns:

```json
{
  "success": true,
  "query": "shipping label bug",
  "results": [
    {
      "path": "/notes/incidents.md",
      "startLine": 12,
      "endLine": 31,
      "score": 0.82,
      "snippet": "...",
      "source": "memory"
    }
  ]
}
```

## Notes

- The database defaults to `~/.local/share/infinite-memory/index.sqlite`.
- Secrets are intentionally read only from environment variables, never from config files.
- If embeddings are unavailable, use `provider = "hash"` for offline dev/testing; production should use `openai` or `azure_openai`.
