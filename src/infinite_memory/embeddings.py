from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .config import EmbeddingConfig

TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingConfigError(RuntimeError):
    pass


@dataclass
class HashEmbeddingProvider:
    dimension: int = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


@dataclass
class OpenAIEmbeddingProvider:
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 60.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        url = self.base_url.rstrip("/") + "/embeddings"
        payload = {"model": self.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = _post_json(url, payload, headers, self.timeout_seconds)
        return _extract_embeddings(data, len(texts))


@dataclass
class AzureOpenAIEmbeddingProvider:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2024-02-01"
    timeout_seconds: float = 60.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        endpoint = self.endpoint.rstrip("/")
        deployment = urllib.parse.quote(self.deployment, safe="")
        api_version = urllib.parse.quote(self.api_version, safe="")
        url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
        payload = {"input": texts}
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        data = _post_json(url, payload, headers, self.timeout_seconds)
        return _extract_embeddings(data, len(texts))


BEDROCK_COHERE_MAX_TEXTS = 96


@dataclass
class BedrockCohereEmbeddingProvider:
    """Cohere Embed (v3 / v4) on Amazon Bedrock, authenticated with a Bedrock API key (bearer token).

    Cohere embeds documents and queries asymmetrically: indexed chunks use input_type
    "search_document" and searches use "search_query", which noticeably improves retrieval.
    """

    model: str
    api_key: str
    region: str
    output_dimension: int | None = None
    timeout_seconds: float = 60.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "search_document")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "search_query")

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        model = urllib.parse.quote(self.model, safe="")
        url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{model}/invoke"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BEDROCK_COHERE_MAX_TEXTS):
            batch = texts[start : start + BEDROCK_COHERE_MAX_TEXTS]
            payload: dict = {
                "texts": batch,
                "input_type": input_type,
                "embedding_types": ["float"],
                "truncate": "END",
            }
            if self.output_dimension:
                payload["output_dimension"] = self.output_dimension
            data = _post_json(url, payload, headers, self.timeout_seconds)
            vectors.extend(_extract_cohere_embeddings(data, len(batch)))
        return vectors


def _extract_cohere_embeddings(data: dict, expected_count: int) -> list[list[float]]:
    embeddings = data.get("embeddings")
    # embedding_types=["float"] returns {"float": [...]}; older v3 responses return the list directly.
    vectors = embeddings.get("float") if isinstance(embeddings, dict) else embeddings
    if not isinstance(vectors, list):
        raise RuntimeError(f"Embedding response missing embeddings: {list(data.keys())}")
    if len(vectors) != expected_count or any(not isinstance(v, list) for v in vectors):
        raise RuntimeError("Embedding response count mismatch")
    return [[float(x) for x in vector] for vector in vectors]


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout_seconds: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Embedding HTTP {exc.code}: {body_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Embedding request failed: {exc.reason}") from exc


def _extract_embeddings(data: dict, expected_count: int) -> list[list[float]]:
    rows = data.get("data")
    if not isinstance(rows, list):
        raise RuntimeError(f"Embedding response missing data array: {data.keys()}")
    rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
    vectors = [row.get("embedding") for row in rows]
    if len(vectors) != expected_count or any(not isinstance(v, list) for v in vectors):
        raise RuntimeError("Embedding response count mismatch")
    return [[float(x) for x in vector] for vector in vectors]


def build_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    provider = config.provider.lower().strip()
    if provider in {"hash", "local_hash", "test"}:
        return HashEmbeddingProvider(dimension=config.dimension)

    if provider == "openai":
        api_key = os.getenv(config.openai_api_key_env)
        if not api_key:
            raise EmbeddingConfigError(f"Missing env var {config.openai_api_key_env}")
        return OpenAIEmbeddingProvider(
            model=config.model,
            api_key=api_key,
            base_url=config.base_url or "https://api.openai.com/v1",
            timeout_seconds=config.timeout_seconds,
        )

    if provider in {"azure", "azure_openai"}:
        endpoint = os.getenv(config.azure_endpoint_env)
        api_key = os.getenv(config.azure_api_key_env)
        deployment = (
            config.deployment
            or os.getenv(config.azure_deployment_env)
            or os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )
        api_version = config.api_version or os.getenv(config.azure_api_version_env) or "2024-02-01"
        missing = []
        if not endpoint:
            missing.append(config.azure_endpoint_env)
        if not api_key:
            missing.append(config.azure_api_key_env)
        if not deployment:
            missing.append(config.azure_deployment_env)
        if missing:
            raise EmbeddingConfigError("Missing env vars: " + ", ".join(missing))
        return AzureOpenAIEmbeddingProvider(
            endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            api_version=api_version,
            timeout_seconds=config.timeout_seconds,
        )

    if provider in {"bedrock", "bedrock_cohere"}:
        api_key = os.getenv(config.bedrock_api_key_env)
        region = config.bedrock_region or os.getenv(config.bedrock_region_env)
        missing = []
        if not api_key:
            missing.append(config.bedrock_api_key_env)
        if not region:
            missing.append(config.bedrock_region_env)
        if missing:
            raise EmbeddingConfigError("Missing env vars: " + ", ".join(missing))
        # Model IDs, cross-region profiles (eu.cohere.embed-v4:0) and ARNs all contain "cohere.".
        if "cohere." not in config.model:
            raise EmbeddingConfigError(f"Bedrock provider only supports Cohere Embed models, got {config.model}")
        return BedrockCohereEmbeddingProvider(
            model=config.model,
            api_key=api_key,
            region=region,
            output_dimension=config.output_dimension,
            timeout_seconds=config.timeout_seconds,
        )

    raise EmbeddingConfigError(f"Unsupported embedding provider: {config.provider}")


def embed_queries(embedder: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    """Embed search queries, using the provider's query mode when it has one (e.g. Cohere)."""
    query_embed = getattr(embedder, "embed_queries", None)
    return query_embed(texts) if callable(query_embed) else embedder.embed(texts)
