# infinite-memory

`infinite-memory` indexes Markdown folders into a local SQLite database and lets you search them semantically, returning the text snippet plus the source file and line range — the same shape OpenClaw's `memory_search` exposes.

It supports:

- Linux-first file watching for `*.md` changes (`watchdog`/inotify, polling fallback).
- Remote sources: documents behind an HTTP API, mirrored as files and kept up to date by polling.
- OpenAI embeddings.
- Azure OpenAI embeddings.
- Deterministic local hash embeddings for tests/offline smoke checks.
- SQLite storage with OpenClaw-aligned hybrid vector + FTS5 ranking.
- Optional `sqlite-vec` vector index, with Python cosine fallback when the extension is unavailable.
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

Search defaults are intentionally aligned with OpenClaw's built-in memory search:

```toml
[chunking]
tokens = 400
overlap = 80

[search]
vector_weight = 0.7
lexical_weight = 0.3
min_score = 0.35
candidate_multiplier = 4
```

Lexical search uses FTS5 with `AND` between query terms, then merges keyword and vector candidates before applying the hybrid score.

## Remote sources

Besides local folders, `infinite-memory` can index documents that live somewhere else:
a database behind an API, a wiki, another service. Each source is mirrored into a local
folder of Markdown files and that folder is indexed like any other, so search results
keep pointing at a file and a line range.

```toml
[[sources]]
name = "wiki"
url = "https://example.com/api/memory"      # changes are read from <url>/changes
token_env = "INFINITE_MEMORY_WIKI_TOKEN"     # bearer token, read from the environment
# token_command = "/path/to/print-token"     # or a command that prints it (short-lived tokens)
mirror_path = "~/memories/wiki"              # default: <index folder>/sources/<name>
# page_size = 100
# min_interval_seconds = 0                   # do not ask again sooner than this
# poll_interval_seconds = 30                 # how often `watch` asks
```

`[watch].paths` may be empty when there is at least one source.

Sources are asked on every `infinite-memory index` and periodically by `watch`. Nothing
has to reach this machine: it only makes outgoing requests, so no port or webhook is
needed. When nothing changed the answer is one empty page. A source that cannot be
reached is reported in the `sources` list of the output and what was already mirrored
stays searchable.

The mirror is a copy, written read-only. Edit documents in the source; the next change
there overwrites the file.

### The protocol a source implements

One endpoint:

```text
GET <url>/changes?limit=100[&cursor=<opaque>]
Authorization: Bearer <token>
```

```json
{
  "protocol": 1,
  "documents": [
    { "id": "8f2c", "path": "logistics/shipping.md", "content": "# Shipping\n...", "deleted": false },
    { "id": "77a1", "deleted": true }
  ],
  "next_cursor": "1042",
  "has_more": false
}
```

- **Without `cursor`** the source lists every live document, in pages. Whatever a full
  listing does not name is removed from the mirror.
- **With `cursor`** it returns what changed after it, oldest first: new and edited
  documents with their whole content, and `"deleted": true` for the ones that are gone.
  A renamed document keeps its `id` and comes with its new `path`.
- **`next_cursor`** is opaque to the client and is sent back as it came. It must move
  forward whenever `has_more` is true.
- **`410 Gone`** means the source no longer knows the cursor; the client lists everything
  again.
- **`id`** is stable for the life of a document. **`path`** is relative, uses `/`, and
  has no `..` or hidden parts; `.md` is added when missing.

Anything else in the response is ignored, so a source can add fields of its own.

## Only what changed is embedded

A changed file is split into chunks again, but a chunk whose text is the same as before
keeps its embedding. Appending to a long file or editing one paragraph costs those
chunks, not the whole file. `index --force`, or a change of embedding model or chunking,
computes everything again. The `embedded` number in the output of `index` says how many
chunks were sent to the embedding model.

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
