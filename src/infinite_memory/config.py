from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib
from typing import Any

DEFAULT_CONFIG_PATH = Path("~/.config/infinite-memory/config.toml")
DEFAULT_DB_PATH = Path("~/.local/share/infinite-memory/index.sqlite")


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


@dataclass(frozen=True)
class WatchConfig:
    paths: list[Path] = field(default_factory=list)
    include_globs: list[str] = field(default_factory=lambda: ["**/*.md"])
    exclude_dirs: list[str] = field(default_factory=lambda: [".git", "node_modules", ".venv", "venv"])
    db_path: Path = field(default_factory=lambda: expand_path(DEFAULT_DB_PATH))
    poll_interval_seconds: float = 2.0


@dataclass(frozen=True)
class ChunkingConfig:
    max_lines: int = 24
    overlap_lines: int = 4
    min_chars: int = 20


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
    vector_weight: float = 0.75
    lexical_weight: float = 0.25
    min_score: float = 0.0


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


def load_config(path: str | Path | None) -> Config:
    config_path = expand_path(path or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

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
        db_path=expand_path(watch_raw.get("db_path", DEFAULT_DB_PATH)),
        poll_interval_seconds=float(watch_raw.get("poll_interval_seconds", 2.0)),
    )
    chunking = ChunkingConfig(
        max_lines=int(chunk_raw.get("max_lines", 24)),
        overlap_lines=int(chunk_raw.get("overlap_lines", 4)),
        min_chars=int(chunk_raw.get("min_chars", 20)),
    )
    if chunking.overlap_lines >= chunking.max_lines:
        raise ValueError("chunking.overlap_lines must be lower than chunking.max_lines")

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
        azure_deployment_env=str(embed_raw.get("azure_deployment_env", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")),
        azure_api_version_env=str(embed_raw.get("azure_api_version_env", "AZURE_OPENAI_API_VERSION")),
    )
    search = SearchConfig(
        vector_weight=float(search_raw.get("vector_weight", 0.75)),
        lexical_weight=float(search_raw.get("lexical_weight", 0.25)),
        min_score=float(search_raw.get("min_score", 0.0)),
    )
    return Config(watch=watch, chunking=chunking, embedding=embedding, search=search)


def default_config_text(path: str | Path) -> str:
    note_path = str(Path(path).expanduser())
    return "".join([
        "[watch]\n",
        f"paths = [\"{note_path}\"]\n",
        "db_path = \"~/.local/share/infinite-memory/index.sqlite\"\n",
        "include_globs = [\"**/*.md\"]\n",
        "exclude_dirs = [\".git\", \"node_modules\", \".venv\", \"venv\"]\n",
        "poll_interval_seconds = 2\n\n",
        "[chunking]\n",
        "max_lines = 24\n",
        "overlap_lines = 4\n",
        "min_chars = 20\n\n",
        "[embedding]\n",
        "# Use \"azure_openai\", \"openai\", or \"hash\" for offline tests.\n",
        "provider = \"azure_openai\"\n",
        "model = \"text-embedding-3-large\"\n",
        "batch_size = 32\n",
        "# deployment = \"text-embedding-3-large\"\n",
        "# api_version = \"2024-02-01\"\n\n",
        "[search]\n",
        "vector_weight = 0.75\n",
        "lexical_weight = 0.25\n",
        "min_score = 0.0\n",
    ])
