from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
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


@dataclass(frozen=True)
class ChunkingConfig:
    tokens: int = 400
    overlap: int = 80


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


@dataclass(frozen=True)
class Config:
    watch: WatchConfig
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


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
    if not paths:
        raise ValueError("Config [watch].paths must include at least one folder")

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
    )
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
    return Config(watch=watch, chunking=chunking, embedding=embedding, search=search)


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
