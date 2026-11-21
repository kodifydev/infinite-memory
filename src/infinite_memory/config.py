from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Any

APP_NAME = "infinite-memory"
CONFIG_ENV_VAR = "INFINITE_MEMORY_CONFIG"
DEFAULT_CONFIG_PATH = Path("~/.config/infinite-memory/config.toml")
DEFAULT_DB_PATH = Path("~/.local/share/infinite-memory/index.sqlite")


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def _xdg_path(env_var: str, fallback_home_relative: str, *parts: str) -> Path:
    root = os.getenv(env_var)
    if root:
        return expand_path(Path(root).joinpath(*parts))
    return expand_path(Path(fallback_home_relative).joinpath(*parts))


def default_config_path() -> Path:
    return _xdg_path("XDG_CONFIG_HOME", "~/.config", APP_NAME, "config.toml")


def default_db_path() -> Path:
    return _xdg_path("XDG_DATA_HOME", "~/.local/share", APP_NAME, "index.sqlite")


def resolve_config_path(path: str | Path | None = None) -> Path:
    if path:
        return expand_path(path)
    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        return expand_path(env_path)
    return default_config_path()


@dataclass(frozen=True)
class WatchConfig:
    paths: list[Path] = field(default_factory=list)
    include_globs: list[str] = field(default_factory=lambda: ["**/*.md"])
    exclude_dirs: list[str] = field(default_factory=lambda: [".git", "node_modules", ".venv", "venv"])
    db_path: Path = field(default_factory=default_db_path)
    poll_interval_seconds: float = 2.0


CHUNKING_STRATEGIES = {"window", "markdown_blocks"}


@dataclass(frozen=True)
class ChunkingConfig:
    tokens: int = 400
    overlap: int = 80
    # "window": fixed-size windows (OpenClaw-aligned). "markdown_blocks": one chunk per paragraph or
    # list item, prefixed with "<title> — <section>" so each fact embeds on its own with its context.
    strategy: str = "window"


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "azure_openai"
    model: str = "text-embedding-3-large"
    batch_size: int = 32
    dimension: int = 384
    timeout_seconds: float = 60.0
    base_url: str | None = None
    deployment: str | None = None
    api_version: str | None = None
    openai_api_key_env: str = "OPENAI_API_KEY"
    azure_endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    azure_api_key_env: str = "AZURE_OPENAI_API_KEY"
    azure_deployment_env: str = "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    azure_api_version_env: str = "AZURE_OPENAI_API_VERSION"


@dataclass(frozen=True)
class SearchConfig:
    vector_weight: float = 0.7
    lexical_weight: float = 0.3
    min_score: float = 0.35
    candidate_multiplier: int = 4


SOURCE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class SourceConfig:
    """Documents that live behind an HTTP API and are mirrored here as Markdown files."""

    name: str
    # Base address; the changes are read from "<url>/changes".
    url: str
    # Folder the documents are written to; it is indexed like any watched folder.
    mirror_path: Path
    # Environment variable holding the bearer token, if the source needs one.
    token_env: str | None = None
    # Or a command that prints the token, for sources whose tokens are short-lived.
    token_command: str | None = None
    timeout_seconds: float = 20.0
    page_size: int = 100
    # Do not ask the source again when it was asked less than this ago (0 = always ask).
    min_interval_seconds: float = 0.0
    # How often `watch` asks the source.
    poll_interval_seconds: float = 30.0


@dataclass(frozen=True)
class Config:
    watch: WatchConfig
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    sources: list[SourceConfig] = field(default_factory=list)


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def load_config(path: str | Path | None = None) -> Config:
    config_path = resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            "Config file not found: "
            f"{config_path}\n"
            "Run:\n"
            "  infinite-memory init --path ~/your-markdown-folder\n"
            "Or pass one explicitly:\n"
            "  infinite-memory --config ./memory.toml search \"your query\""
        )

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    watch_raw = raw.get("watch", {})
    chunk_raw = raw.get("chunking", {})
    embed_raw = raw.get("embedding", {})
    search_raw = raw.get("search", {})

    paths = [expand_path(p) for p in _as_list(watch_raw.get("paths"), [])]
    sources_raw = raw.get("sources", [])
    if not paths and not sources_raw:
        raise ValueError("Config needs a folder in [watch].paths or a remote [[sources]] entry")

    watch = WatchConfig(
        paths=paths,
        include_globs=_as_list(watch_raw.get("include_globs"), ["**/*.md"]),
        exclude_dirs=_as_list(watch_raw.get("exclude_dirs"), [".git", "node_modules", ".venv", "venv"]),
        db_path=expand_path(watch_raw.get("db_path", default_db_path())),
        poll_interval_seconds=float(watch_raw.get("poll_interval_seconds", 2.0)),
    )
    chunking = ChunkingConfig(
        tokens=max(1, int(chunk_raw.get("tokens", 400))),
        overlap=max(0, int(chunk_raw.get("overlap", 80))),
        strategy=str(chunk_raw.get("strategy", "window")),
    )
    if chunking.strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"chunking.strategy must be one of {sorted(CHUNKING_STRATEGIES)}")
    if chunking.overlap >= chunking.tokens:
        raise ValueError("chunking.overlap must be lower than chunking.tokens")

    embedding = EmbeddingConfig(
        provider=str(embed_raw.get("provider", "azure_openai")),
        model=str(embed_raw.get("model", "text-embedding-3-large")),
        batch_size=int(embed_raw.get("batch_size", 32)),
        dimension=int(embed_raw.get("dimension", 384)),
        timeout_seconds=float(embed_raw.get("timeout_seconds", 60.0)),
        base_url=embed_raw.get("base_url"),
        deployment=embed_raw.get("deployment"),
        api_version=embed_raw.get("api_version"),
        openai_api_key_env=str(embed_raw.get("openai_api_key_env", "OPENAI_API_KEY")),
        azure_endpoint_env=str(embed_raw.get("azure_endpoint_env", "AZURE_OPENAI_ENDPOINT")),
        azure_api_key_env=str(embed_raw.get("azure_api_key_env", "AZURE_OPENAI_API_KEY")),
        azure_deployment_env=str(
            embed_raw.get("azure_deployment_env", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        ),
        azure_api_version_env=str(embed_raw.get("azure_api_version_env", "AZURE_OPENAI_API_VERSION")),
    )
    search = SearchConfig(
        vector_weight=float(search_raw.get("vector_weight", 0.7)),
        lexical_weight=float(search_raw.get("lexical_weight", 0.3)),
        min_score=float(search_raw.get("min_score", 0.35)),
        candidate_multiplier=max(1, int(search_raw.get("candidate_multiplier", 4))),
    )
    sources = _load_sources(sources_raw, watch.db_path)
    return Config(watch=watch, chunking=chunking, embedding=embedding, search=search, sources=sources)


def _load_sources(raw: Any, db_path: Path) -> list[SourceConfig]:
    if not isinstance(raw, list):
        raise ValueError("Config sources must be written as [[sources]] tables")
    sources: list[SourceConfig] = []
    for entry in raw:
        name = str(entry.get("name", ""))
        if not SOURCE_NAME_RE.match(name):
            raise ValueError(
                "Each [[sources]] entry needs a name of lowercase letters, numbers, - or _"
            )
        if any(source.name == name for source in sources):
            raise ValueError(f"Source name used twice: {name}")
        url = str(entry.get("url", "")).rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Source {name} needs an http(s) url")
        if entry.get("token_env") and entry.get("token_command"):
            raise ValueError(f"Source {name}: set token_env or token_command, not both")
        mirror = entry.get("mirror_path")
        sources.append(
            SourceConfig(
                name=name,
                url=url,
                mirror_path=expand_path(mirror) if mirror else db_path.parent / "sources" / name,
                token_env=entry.get("token_env") or None,
                token_command=entry.get("token_command") or None,
                timeout_seconds=float(entry.get("timeout_seconds", 20.0)),
                page_size=max(1, int(entry.get("page_size", 100))),
                min_interval_seconds=max(0.0, float(entry.get("min_interval_seconds", 0.0))),
                poll_interval_seconds=max(1.0, float(entry.get("poll_interval_seconds", 30.0))),
            )
        )
    return sources


def _toml_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def default_config_text(
    path: str | Path,
    *,
    db_path: str | Path | None = None,
    provider: str = "azure_openai",
    model: str = "text-embedding-3-large",
    deployment: str | None = None,
    api_version: str | None = None,
) -> str:
    note_path = Path(path).expanduser()
    database_path = db_path or default_db_path()
    lines = [
        "[watch]",
        f"paths = [{_toml_string(note_path)}]",
        f"db_path = {_toml_string(database_path)}",
        'include_globs = ["**/*.md"]',
        'exclude_dirs = [".git", "node_modules", ".venv", "venv"]',
        "poll_interval_seconds = 2",
        "",
        "[chunking]",
        "# OpenClaw-aligned approximate-token chunking: chars ~= tokens * 4.",
        "tokens = 400",
        "overlap = 80",
        "",
        "[embedding]",
        '# Use "azure_openai", "openai", or "hash" for offline tests.',
        f"provider = {_toml_string(provider)}",
        f"model = {_toml_string(model)}",
        "batch_size = 32",
    ]
    if deployment:
        lines.append(f"deployment = {_toml_string(deployment)}")
    else:
        lines.append(f"# deployment = {_toml_string(model)}")
    if api_version:
        lines.append(f"api_version = {_toml_string(api_version)}")
    else:
        lines.append('# api_version = "2024-02-01"')
    lines.extend(
        [
            "",
            "[search]",
            "vector_weight = 0.7",
            "lexical_weight = 0.3",
            "min_score = 0.35",
            "candidate_multiplier = 4",
            "",
        ]
    )
    return "\n".join(lines)
