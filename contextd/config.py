"""Global configuration loader for ~/.contextd/config.toml.

Schema-validated via pydantic. The default config (shipped in the
package at contextd/default_config.toml) fills in any fields the user
omits, so minimal user configs work correctly.
"""

from __future__ import annotations

import re
import tomllib
from importlib import resources
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

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
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    """Sampling temperature for summary/inference calls (translation keeps
    the provider default). Unset = provider default; ~0.2 recommended for
    JSON extraction."""


class VoyageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "voyage-4-large"
    max_batch_size: int = 128


EmbeddingProviderName = Literal["voyage", "openai_compat", "local_hf"]

_OPENAI_COMPAT_PREFIX = "openai_compat:"
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_MIGRATION_HINT = (
    "openai_compat backends are now named profiles: define a "
    "[providers.openai_compat.<profile>] table and reference it as "
    '"openai_compat:<profile>" (e.g. summary = "openai_compat:local"). '
    "To migrate a pre-profiles config, rename [providers.openai_compat] to "
    "[providers.openai_compat.local] and change each call-site set to "
    '"openai_compat" to "openai_compat:local".'
)


def openai_compat_profile(ref: str) -> str | None:
    """Return the profile name for an ``openai_compat:<profile>`` ref, else None."""
    if ref.startswith(_OPENAI_COMPAT_PREFIX):
        return ref[len(_OPENAI_COMPAT_PREFIX) :]
    return None


def _validate_inference_ref(ref: str) -> str:
    if ref == "gemini":
        return ref
    if ref == "openai_compat":
        raise ValueError(f'"openai_compat" without a profile is no longer valid. {_MIGRATION_HINT}')
    profile = openai_compat_profile(ref)
    if profile is not None:
        if not _PROFILE_NAME_RE.match(profile):
            raise ValueError(f"invalid openai_compat profile name {profile!r} in {ref!r}")
        return ref
    raise ValueError(
        f"unknown inference provider {ref!r}: expected 'gemini' or 'openai_compat:<profile>'"
    )


InferenceProviderRef = Annotated[str, AfterValidator(_validate_inference_ref)]


class OpenAICompatEmbeddingConfig(BaseModel):
    """Config for embeddings served by a local OpenAI-compatible server.

    Targets the OpenAI ``/embeddings`` endpoint shape exposed by llama.cpp's
    server, Ollama (``/v1/`` mode), LM Studio, vLLM, and LocalAI. Selecting
    ``providers.embedding = "openai_compat"`` together with
    ``providers.openai_compat.<profile>`` inference profiles lets the entire
    indexing pipeline run offline with no cloud API calls.

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


class LocalHFConfig(BaseModel):
    """Config for in-process embeddings via ``sentence-transformers``
    (``providers.embedding = "local_hf"``, optional extra ``contextd[late]``).

    This is the only provider that implements ``TokenEmbedder`` and therefore
    the only one the ``late`` chunking strategy accepts: one forward pass per
    parent text, per-chunk vectors mean-pooled from the token vectors.

    ``dimensions`` MUST match the vector-index width in the baseline
    migrations (1024) and is validated against the model's output at first
    use. The default ``BAAI/bge-m3`` emits 1024-dim vectors over an 8192-token
    context; ``nomic-ai/modernbert-embed-base`` (768-dim) or other widths
    require editing the migration DDL. ``max_context_tokens`` caps one forward
    pass and must not exceed the model's positional limit; longer texts are
    embedded in overlapping windows.
    """

    model_config = ConfigDict(extra="forbid")
    model: str = "BAAI/bge-m3"
    dimensions: int = Field(default=1024, gt=0)
    device: str = "cpu"
    max_context_tokens: int = Field(default=8192, ge=16)
    normalize: bool = True
    batch_size: int = Field(default=8, ge=1)


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
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    """Sampling temperature for summary/inference calls (translation keeps
    the server default). Unset = server default; ~0.2 recommended for JSON
    extraction."""


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: InferenceProviderRef = "gemini"
    inference: InferenceProviderRef = "gemini"
    translation: InferenceProviderRef = "gemini"
    embedding: EmbeddingProviderName = "voyage"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    openai_compat: dict[str, OpenAICompatConfig] = Field(default_factory=dict)
    openai_compat_embedding: OpenAICompatEmbeddingConfig = Field(
        default_factory=OpenAICompatEmbeddingConfig
    )
    voyage: VoyageConfig = Field(default_factory=VoyageConfig)
    local_hf: LocalHFConfig = Field(default_factory=LocalHFConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_openai_compat_shape(cls, data: object) -> object:
        # A pre-profiles [providers.openai_compat] deep-merged over the packaged
        # default leaves scalar keys (base_url, model_summary, ...) beside the
        # profile tables; surface a migration hint instead of pydantic's opaque
        # per-key "not a valid dictionary" error.
        if isinstance(data, dict):
            oc = data.get("openai_compat")
            if isinstance(oc, dict):
                bad = sorted(
                    k for k, v in oc.items() if not isinstance(v, dict | OpenAICompatConfig)
                )
                if bad:
                    raise ValueError(
                        f"[providers.openai_compat] contains non-profile keys {bad}. "
                        f"{_MIGRATION_HINT}"
                    )
        return data

    @model_validator(mode="after")
    def _check_profile_refs(self) -> ProvidersConfig:
        for site, ref in (
            ("summary", self.summary),
            ("inference", self.inference),
            ("translation", self.translation),
        ):
            profile = openai_compat_profile(ref)
            if profile is not None and profile not in self.openai_compat:
                defined = ", ".join(sorted(self.openai_compat)) or "(none defined)"
                raise ValueError(
                    f"providers.{site} = {ref!r} references profile {profile!r}, but no "
                    f"[providers.openai_compat.{profile}] table exists. "
                    f"Defined profiles: {defined}"
                )
        return self


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
    relate_gleaning_rounds: int = Field(default=1, ge=0, le=5)
    """Extra "what did you miss?" relate passes per unit (Microsoft-GraphRAG
    gleaning). Each round re-sends the unit's content, roughly doubling relate
    input spend at 1; set 0 to opt out."""


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


SearchMode = Literal["hybrid", "fulltext", "vector"]
ReturnUnit = Literal["chunk", "section", "file", "auto"]


class SearchConfig(BaseModel):
    """Hybrid-search ranking knobs consumed by the ``search`` MCP tool.

    ``mode`` selects the retrieval strategy: ``hybrid`` (default) fuses
    vector-similarity and full-text results via reciprocal rank fusion (see
    :mod:`contextd.search.fusion`), degrading to full-text when no embedder
    is available or the queried label has no vector index; ``fulltext`` and
    ``vector`` force a single ranker. ``rrf_k`` is the RRF damping constant
    (larger flattens the top-rank weighting). ``fetch_k`` is the per-ranker
    candidate depth pulled before fusion; the tool raises it to at least the
    caller's ``limit``. ``vector_weight`` / ``fulltext_weight`` bias the two
    rankers and must not both be zero, which would zero every fused score.
    """

    model_config = ConfigDict(extra="forbid")
    mode: SearchMode = "hybrid"
    rrf_k: int = Field(default=60, ge=1)
    fetch_k: int = Field(default=50, ge=1)
    vector_weight: float = Field(default=1.0, ge=0.0)
    fulltext_weight: float = Field(default=1.0, ge=0.0)
    chunk_profiles: list[str] | None = None
    """Chunk profiles the ``search`` tool queries by default; ``None`` → every
    profile that exists in the graph."""
    return_unit: ReturnUnit = "auto"
    """Unit the ``search`` tool collapses chunk hits to. ``auto`` → the
    enclosing Section when one exists, else the File."""
    auto_merge_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    """Fraction of a parent's chunks that must hit before the parent replaces
    its chunks in the result list (LlamaIndex / Haystack convention: 0.5)."""
    window: int = Field(default=1, ge=0)
    """Neighbouring chunks (each side) attached as context to a chunk hit."""
    max_evidence_chars: int = Field(default=1200, ge=100)
    over_fetch_factor: int = Field(default=4, ge=1, le=20)
    """Multiplier on ``k`` when a filtered vector/full-text search must
    post-filter results (the backend procedures cannot pre-filter)."""

    @model_validator(mode="after")
    def _weights_not_both_zero(self) -> SearchConfig:
        if self.vector_weight == 0.0 and self.fulltext_weight == 0.0:
            raise ValueError(
                "search.vector_weight and search.fulltext_weight must not both be zero"
            )
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    @classmethod
    def load_default(cls) -> Config:
        raw = tomllib.loads(
            resources.files("contextd").joinpath("default_config.toml").read_text(encoding="utf-8")
        )
        return cls.model_validate(raw)

    @classmethod
    def load(cls, path: Path) -> Config:
        default_raw = tomllib.loads(
            resources.files("contextd").joinpath("default_config.toml").read_text(encoding="utf-8")
        )
        user_raw = tomllib.loads(path.read_text(encoding="utf-8"))
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
