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

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    chunk_tokens: int = 32000
    chunk_overlap: int = 200


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


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus: CorpusSection
    embedding: EmbeddingSection = Field(default_factory=EmbeddingSection)
    ontology: OntologySection = Field(default_factory=OntologySection)
    mcp: McpSection = Field(default_factory=McpSection)
    summarization: SummarizationSection = Field(default_factory=SummarizationSection)
    resolution: ResolutionSection = Field(default_factory=ResolutionSection)
    lexical: LexicalSection = Field(default_factory=LexicalSection)

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
        try:
            return cls.model_validate(raw)
        except Exception as exc:
            raise CorpusConfigError(str(exc)) from exc
