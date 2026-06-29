"""Global configuration loader for ~/.contextd/config.toml.

Schema-validated via pydantic. The default config (shipped in the
package at contextd/default_config.toml) fills in any fields the user
omits, so minimal user configs work correctly.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BackendName = Literal["neo4j"]
SafetyBlock = Literal[
    "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"
]


class ConfigError(ValueError):
    """Raised when a config file is malformed or contains invalid values."""


class GeminiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_summary: str = "gemma-4-31b-it"
    model_inference: str = "gemma-4-31b-it"
    model_translation: str = "gemma-4-31b-it"
    max_retries: int = 5
    safety_block: SafetyBlock = "BLOCK_NONE"
    daily_budget: str | int = "unlimited"


class VoyageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "voyage-4-large"
    max_batch_size: int = 128


InferenceProviderName = Literal["gemini", "openai_compat"]
EmbeddingProviderName = Literal["voyage", "openai_compat"]


class OpenAICompatEmbeddingConfig(BaseModel):
    """Config for embeddings served by a local OpenAI-compatible server.

    Targets the OpenAI ``/embeddings`` endpoint shape exposed by llama.cpp's
    server, Ollama (``/v1/`` mode), LM Studio, vLLM, and LocalAI. Selecting
    ``providers.embedding = "openai_compat"`` together with an
    ``providers.openai_compat`` inference backend lets the entire indexing
    pipeline run offline with no cloud API calls.

    ``dimensions`` MUST match the vector-index dimension declared in the
    baseline migrations (1024). The default model ``mxbai-embed-large`` emits
    1024-dim vectors and so drops into the existing index unchanged; choosing
    a model with a different output width (e.g. the 768-dim
    ``nomic-embed-text``) requires editing the migration DDL on both backends.
    The provider validates returned vector length against ``dimensions`` and
    raises rather than writing mismatched vectors into the index.
    """

    model_config = ConfigDict(extra="forbid")
    base_url: str = "http://localhost:11434/v1"
    api_key_env: str | None = None
    model: str = "mxbai-embed-large"
    dimensions: int = Field(default=1024, gt=0)
    max_batch_size: int = Field(default=64, ge=1)
    max_retries: int = Field(default=5, ge=0)
    request_timeout_seconds: float = Field(default=120.0, gt=0)


class OpenAICompatConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str = "http://localhost:11434/v1"
    api_key_env: str | None = None
    model_summary: str = "qwen2.5:7b-instruct"
    model_inference: str = "qwen2.5:14b-instruct"
    model_translation: str = "qwen2.5:14b-instruct"
    max_retries: int = Field(default=5, ge=0)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    json_mode: bool = True
    max_output_tokens: int | None = Field(default=None, ge=1)


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: InferenceProviderName = "gemini"
    inference: InferenceProviderName = "gemini"
    translation: InferenceProviderName = "gemini"
    embedding: EmbeddingProviderName = "voyage"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    openai_compat_embedding: OpenAICompatEmbeddingConfig = Field(
        default_factory=OpenAICompatEmbeddingConfig
    )
    voyage: VoyageConfig = Field(default_factory=VoyageConfig)


class Neo4jConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = 7687
    user: str = "neo4j"
    # Must match NEO4J_AUTH in `contextd/docker_compose.yml` (neo4j/contextd).
    # Neo4j's image rejects the default `neo4j/neo4j` credential — it forces
    # a password change on first login — so we ship a non-default here.
    password: str = "contextd"
    docker_compose_file: str = "~/.contextd/docker-compose.yml"
    memory_limit_gb: float = 1.0
    cpu_limit: float = 1.0


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: BackendName = "neo4j"
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    # Neo4j is the sole storage backend. ``BackendName`` is kept as a Literal
    # (rather than inlined) so that adding a second backend later requires only
    # widening the Literal, adding a factory branch, and a migrations dir —
    # the GraphStore ABC seam and the abstraction-invariant CI grep already
    # keep consumers decoupled from the concrete backend.


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary_max_words: int = 100


class IndexerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    debounce_seconds: int = 30
    git_lock_check: bool = True
    parallel_embedding_batches: int = 4
    inference_concurrency: int = Field(default=1, ge=1)
    allowed_branches: list[str] = Field(default_factory=list)
    incremental_workers: int = Field(default=4, ge=1)
    sweep_interval_seconds: int = Field(default=900, ge=0)
    sweep_rate_sections_per_second: float = Field(default=0.017, ge=0.001)


class McpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transport: Literal["stdio", "http-sse"] = "stdio"
    http_port: int | None = None


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str = "info"
    format: Literal["json", "text"] = "json"
    path: str = "~/.contextd/logs/contextd.log"
    max_log_bytes: int = Field(default=10_485_760, ge=0)
    log_backup_count: int = Field(default=5, ge=0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load_default(cls) -> Config:
        raw = tomllib.loads(resources.files("contextd").joinpath("default_config.toml").read_text())
        return cls.model_validate(raw)

    @classmethod
    def load(cls, path: Path) -> Config:
        default_raw = tomllib.loads(
            resources.files("contextd").joinpath("default_config.toml").read_text()
        )
        user_raw = tomllib.loads(path.read_text())
        merged = _deep_merge(default_raw, user_raw)
        try:
            return cls.model_validate(merged)
        except Exception as exc:
            raise ConfigError(str(exc)) from exc


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore
        else:
            out[k] = v
    return out
