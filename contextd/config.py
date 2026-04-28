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

BackendName = Literal["memgraph", "neo4j"]
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


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: InferenceProviderName = "gemini"
    inference: InferenceProviderName = "gemini"
    translation: InferenceProviderName = "gemini"
    embedding: Literal["voyage"] = "voyage"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    voyage: VoyageConfig = Field(default_factory=VoyageConfig)


class MemgraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = 7687
    docker_compose_file: str = "~/.contextd/docker-compose.yml"
    memory_limit_gb: float = 1.0
    cpu_limit: float = 1.0


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
    backend: BackendName = (
        "neo4j"  # was "memgraph"; flipped in M11.8 for reference-Cypher reliability
    )
    memgraph: MemgraphConfig = Field(default_factory=MemgraphConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    # The `Literal["memgraph", "neo4j"]` type constraint on BackendName
    # above is enforced by pydantic v2 before any @field_validator runs — a
    # manual validator was redundant and has been removed. Adding a new backend
    # requires updating BackendName + the factory + the migrations dirs.


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
