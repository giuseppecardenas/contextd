"""Per-corpus configuration loader.

Corpus TOML files live at ~/.contextd/corpora/<name>.toml and hold
per-corpus overrides — granularity choice, heading-level bounds,
ontology aliases, include/exclude globs, and per-corpus MCP tools.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Granularity = Literal["file", "section"]


class CorpusConfigError(ValueError):
    """Raised when a corpus config is malformed."""


class CorpusSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    root: str
    include: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude: list[str] = Field(default_factory=list)
    granularity: Granularity = "file"
    heading_min_level: int = 2
    heading_max_level: int = 4
    content_profile: str | None = None


class EmbeddingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "voyage-4-large"


Strategy = Literal[
    "structural",
    "window",
    "recursive",
    "sentence_window",
    "semantic",
    "late",
    "propositions",
    "code",
]
PrefixMode = Literal["none", "breadcrumb", "section_summary", "llm"]
ThresholdType = Literal["percentile", "stddev", "iqr", "gradient"]
TokenizerName = Literal["auto", "voyage", "tiktoken", "words"]
AugmentField = Literal["key_points", "entities_mentioned", "questions"]

_PROFILE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

# Strategies that split the parent into size-bounded pieces and therefore
# honour ``max_tokens`` / ``min_tokens`` / ``overlap``. ``sentence_window``
# emits one sentence per chunk and ``propositions`` one statement per chunk,
# so the size knobs are ignored for them (accepted, not rejected, so a profile
# can flip strategy without editing every field).
_SIZE_BOUNDED: frozenset[str] = frozenset(
    {"structural", "window", "recursive", "semantic", "late", "code"}
)


class ChunkProfile(BaseModel):
    """One retrieval-chunk size/strategy configuration (``[[chunking.profiles]]``).

    Every profile produces its own ``Chunk`` nodes under each Section/File;
    the ``search`` tool queries the profiles it is asked for and fuses them.
    ``weight`` scales the profile's rankers inside reciprocal rank fusion.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    strategy: Strategy = "structural"
    max_tokens: int = Field(default=256, ge=16)
    min_tokens: int = Field(default=48, ge=0)
    overlap: float = Field(default=0.0, ge=0.0, lt=0.5)
    """Fraction of ``max_tokens`` carried forward from the previous chunk."""
    weight: float = Field(default=1.0, ge=0.0)
    # sentence_window
    window: int = Field(default=1, ge=0)
    """Neighbouring sentences (each side) the query-side expander attaches."""
    # semantic
    buffer_size: int = Field(default=1, ge=0)
    threshold_type: ThresholdType = "percentile"
    threshold: float | None = None
    # recursive
    separators: list[str] | None = None
    # propositions
    max_propositions: int = Field(default=40, ge=1)
    # code
    chunk_lines_overlap: int = Field(default=10, ge=0)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _PROFILE_NAME.match(v):
            raise ValueError(
                f"profile name {v!r} must match [a-z][a-z0-9_]* "
                "(it is embedded in Chunk ids and index filters)"
            )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> ChunkProfile:
        if self.strategy in _SIZE_BOUNDED and self.min_tokens >= self.max_tokens:
            raise ValueError(
                f"profile {self.name!r}: min_tokens ({self.min_tokens}) must be < "
                f"max_tokens ({self.max_tokens})"
            )
        if self.threshold is not None:
            if self.threshold_type in ("percentile", "gradient") and not (
                0.0 < self.threshold <= 100.0
            ):
                raise ValueError(
                    f"profile {self.name!r}: {self.threshold_type} threshold must be in (0, 100]"
                )
            if self.threshold_type in ("stddev", "iqr") and self.threshold <= 0.0:
                raise ValueError(
                    f"profile {self.name!r}: {self.threshold_type} threshold must be > 0"
                )
        if self.separators is not None and not self.separators:
            raise ValueError(f"profile {self.name!r}: separators must be a non-empty list")
        return self

    @property
    def overlap_tokens(self) -> int:
        return int(self.max_tokens * self.overlap)


class ChunkProfileOverride(BaseModel):
    """Partial profile applied to every profile for one file suffix."""

    model_config = ConfigDict(extra="forbid")
    strategy: Strategy | None = None
    max_tokens: int | None = Field(default=None, ge=16)
    min_tokens: int | None = Field(default=None, ge=0)
    overlap: float | None = Field(default=None, ge=0.0, lt=0.5)

    def apply(self, profile: ChunkProfile) -> ChunkProfile:
        patch = {k: v for k, v in self.model_dump().items() if v is not None}
        return profile.model_copy(update=patch) if patch else profile


class BlockRules(BaseModel):
    """How the ``structural`` strategy treats markdown blocks."""

    model_config = ConfigDict(extra="forbid")
    protect_code_fences: bool = True
    table_mode: Literal["rows_with_header", "whole", "prose"] = "rows_with_header"
    sentence_fallback: Literal["recursive", "window"] = "recursive"
    max_fence_tokens: int | None = Field(default=None, ge=16)
    """Cap for a single fence before it is split on blank lines; ``None`` →
    the profile's ``max_tokens``."""


def _default_augment() -> list[AugmentField]:
    return ["key_points"]


def _default_profiles() -> list[ChunkProfile]:
    return [
        ChunkProfile(name="fine", max_tokens=256, min_tokens=48),
        ChunkProfile(name="coarse", max_tokens=1024, min_tokens=200, overlap=0.1),
    ]


class ChunkingSection(BaseModel):
    """Retrieval-chunk configuration (``[chunking]``).

    Chunks are retrieval-only leaves beneath Section/File nodes: they are
    embedded and full-text indexed but never summarised or related by the
    LLM. ``prefix`` selects the context prepended to each chunk's embedded
    text; ``augment_fulltext`` copies parent-summary fields (or LLM-generated
    questions) into the chunk's full-text-only ``keywords`` field.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    tokenizer: TokenizerName = "auto"
    prefix: PrefixMode = "breadcrumb"
    augment_fulltext: list[AugmentField] = Field(default_factory=_default_augment)
    profiles: list[ChunkProfile] = Field(default_factory=_default_profiles)
    blocks: BlockRules = Field(default_factory=BlockRules)
    suffix_overrides: dict[str, ChunkProfileOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _profiles_consistent(self) -> ChunkingSection:
        if self.enabled and not self.profiles:
            raise ValueError("[chunking] enabled but no profiles declared")
        names = [p.name for p in self.profiles]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"[chunking] duplicate profile names: {dupes}")
        for suffix in self.suffix_overrides:
            if not suffix.startswith(".") or len(suffix) < 2:
                raise ValueError(
                    f"[chunking.suffix_overrides] key {suffix!r} must be a file suffix "
                    "starting with '.'"
                )
        return self

    def profile(self, name: str) -> ChunkProfile:
        for p in self.profiles:
            if p.name == name:
                return p
        raise KeyError(name)

    def profiles_for(self, suffix: str) -> list[ChunkProfile]:
        """Profiles with any suffix override applied. The one routing point."""
        override = self.suffix_overrides.get(suffix)
        if override is None:
            return list(self.profiles)
        return [override.apply(p) for p in self.profiles]


class TopicsSection(BaseModel):
    """Cross-document cluster summaries (``[topics]``); RAPTOR-style, off by default."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    source: Literal["section", "file"] = "section"
    max_layers: int = Field(default=3, ge=1, le=5)
    min_members: int = Field(default=3, ge=2)
    soft_threshold: float = Field(default=0.1, gt=0.0, lt=1.0)
    max_cluster_tokens: int = Field(default=3500, ge=200)
    pca_dims: int = Field(default=32, ge=2)
    seed: int = 0


class OntologySection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base: str = "default"
    overrides: str | None = None
    aliases: dict[str, str] = Field(default_factory=dict)


class McpSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tools: dict[str, str] = Field(default_factory=dict)
    """Map MCP-tool-name → path-to-cypher-file (relative to corpus root)."""


class SummarizationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_override: str | None = None
    overrides: dict[str, str] = Field(default_factory=dict)
    """Per-suffix prompt routing: glob (against corpus-root-relative posix
    path, first match in declaration order wins) → template path relative to
    the corpus TOML, or ``builtin:<name>`` for a packaged template copied to
    ``~/.contextd/prompts/<name>.md``. Falls back to ``prompt_override``,
    then the packaged default."""
    max_words: int | None = None
    """Per-corpus override of the global [inference] summary_max_words.

    Leave unset (None) to inherit the global default. Set to a positive
    integer to cap this corpus's per-file (or per-section) summaries at
    that word count. Short-note corpora may lower to 50; dense manuscript
    corpora may raise to 200.
    """


class LexicalPattern(BaseModel):
    """One per-corpus deterministic reference pattern (``[[lexical.patterns]]``).

    ``target_type`` may be an ontology alias (e.g. ``FRRow``); it resolves to
    canon at extraction time. ``formats`` limits the pattern to file suffixes
    (without the dot); empty means all formats. ``capture`` selects the regex
    group used as the target name (0 = whole match).
    """

    model_config = ConfigDict(extra="forbid")
    regex: str
    edge_type: str
    target_type: str
    formats: list[str] = Field(default_factory=list)
    capture: int = 0

    @field_validator("regex")
    @classmethod
    def _compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"invalid regex {v!r}: {exc}") from exc
        return v


class LexicalSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patterns: list[LexicalPattern] = Field(default_factory=list)


class ResolutionSection(BaseModel):
    """Per-corpus knobs for the entity-resolution cascade.

    ``case_insensitive_labels`` names the entity kinds whose instances are
    prose concepts (case variance is noise); every other kind stays
    case-sensitive, because code symbols and IDs may differ only by case.
    All fields default to the shipped cascade settings.
    """

    model_config = ConfigDict(extra="forbid")
    case_insensitive_labels: list[str] = Field(
        default_factory=lambda: [
            "Pattern",
            "Technology",
            "Client",
            "Risk",
            "Service",
            "Integration",
        ]
    )
    fuzzy_threshold: float = 90.0
    fuzzy_min_length: int = 6
    embedding_threshold: float = 0.92
    embedding_enabled: bool = True
    confidence_floor: float = 0.5
    exact_only_pattern: str = r"\d"
    """Regex (``re.search``) for identifier-like names that must never be
    fuzzy- or embedding-matched — a near-miss on an id is a different id.
    Default: any name containing a digit. Empty string disables the guard."""


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus: CorpusSection
    embedding: EmbeddingSection = Field(default_factory=EmbeddingSection)
    ontology: OntologySection = Field(default_factory=OntologySection)
    mcp: McpSection = Field(default_factory=McpSection)
    summarization: SummarizationSection = Field(default_factory=SummarizationSection)
    resolution: ResolutionSection = Field(default_factory=ResolutionSection)
    lexical: LexicalSection = Field(default_factory=LexicalSection)
    chunking: ChunkingSection = Field(default_factory=ChunkingSection)
    topics: TopicsSection = Field(default_factory=TopicsSection)

    @classmethod
    def load(cls, path: Path) -> CorpusConfig:
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise CorpusConfigError(f"invalid TOML syntax in {path}: {exc}") from exc
        granularity = raw.get("corpus", {}).get("granularity")
        if granularity == "auto":
            raise CorpusConfigError(
                "'auto' is reserved for a future version; use 'file' or 'section'"
            )
        removed = {"chunk_tokens", "chunk_overlap"} & set(raw.get("embedding", {}))
        if removed:
            raise CorpusConfigError(
                f"[embedding] {', '.join(sorted(removed))} were removed: embedding-side "
                "chunking is replaced by retrieval chunks. Delete the key(s) and configure "
                "[chunking] profiles instead ([[chunking.profiles]] name/strategy/max_tokens) "
                "— see docs/cli.md."
            )
        try:
            return cls.model_validate(raw)
        except Exception as exc:
            raise CorpusConfigError(str(exc)) from exc
