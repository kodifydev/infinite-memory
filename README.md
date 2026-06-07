# infinite-memory

`infinite-memory` indexes Markdown folders into a local SQLite database and lets you search them semantically, returning the text snippet plus the source file and line range — the same shape OpenClaw's `memory_search` exposes.

It supports:

- Linux-first file watching for `*.md` changes (`watchdog`/inotify, polling fallback).
- OpenAI embeddings.
- Azure OpenAI embeddings.
- Deterministic local hash embeddings for tests/offline smoke checks.
- SQLite storage with vector cosine ranking + FTS5 lexical fallback/boost.
- XDG config/data paths, so day-to-day use does not need `uv run` or `--config`.

## Install on Linux

Recommended user-facing install:

```bash
curl -fsSL https://raw.githubusercontent.com/kodifydev/infinite-memory/main/install.sh | bash
```

The installer:

- requires Python 3.11+
- creates a dedicated virtualenv at `~/.local/share/infinite-memory/venv`
- installs the package from GitHub
- symlinks the command to `~/.local/bin/infinite-memory`

If `~/.local/bin` is not on your `PATH`, add this to your shell profile:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Advanced alternatives:

```bash
pipx install git+https://github.com/kodifydev/infinite-memory.git
# or, for developers with uv:
uv tool install git+https://github.com/kodifydev/infinite-memory.git
```

## Quick start

Create the default config:

```bash
infinite-memory init --path ~/memories
```

That writes:

```text
~/.config/infinite-memory/config.toml
```

or, if set:

```text
$XDG_CONFIG_HOME/infinite-memory/config.toml
```

Index and search:

```bash
infinite-memory index --force
infinite-memory search "shipping label bug" --max-results 5 --json
```

Watch for Markdown changes and auto-index them:

```bash
infinite-memory watch
```

## Config resolution

For `index`, `search`, `status`, and `watch`, `--config` is optional.

Resolution order:

1. explicit CLI path:

   ```bash
   infinite-memory --config ./memory.toml search "shipping label bug"
   # also supported for compatibility:
   infinite-memory search --config ./memory.toml "shipping label bug"
   ```

2. environment variable:

   ```bash
   export INFINITE_MEMORY_CONFIG=./memory.toml
   infinite-memory search "shipping label bug"
   ```

3. default XDG path:

   ```text
   $XDG_CONFIG_HOME/infinite-memory/config.toml
   ~/.config/infinite-memory/config.toml
   ```

The database defaults to:

```text
$XDG_DATA_HOME/infinite-memory/index.sqlite
~/.local/share/infinite-memory/index.sqlite
```

## Configure embeddings

`init` can write the common embedding settings directly:

```bash
infinite-memory init \
  --path ~/memories \
  --provider azure_openai \
  --model text-embedding-3-small \
  --deployment text-embedding-3-small
```

Secrets are intentionally read only from environment variables, never from config files.

For Azure OpenAI:

```toml
[embedding]
provider = "azure_openai"
model = "text-embedding-3-large"
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

For offline/dev smoke checks:

```bash
infinite-memory init --path ./notes --provider hash --model hash --force
```

## Local development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run infinite-memory --help
```

Editable install from a checkout:

```bash
pip install -e /path/to/infinite-memory
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

## Uninstall

If installed with `install.sh`:

```bash
curl -fsSL https://raw.githubusercontent.com/kodifydev/infinite-memory/main/uninstall.sh | bash
```

To also remove config and index data:

```bash
curl -fsSL https://raw.githubusercontent.com/kodifydev/infinite-memory/main/uninstall.sh | bash -s -- --purge
```
