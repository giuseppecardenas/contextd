"""Per-corpus configuration loader.

Corpus TOML files live at ~/.contextd/corpora/<name>.toml and hold
per-corpus overrides — granularity choice, heading-level bounds,
ontology aliases, include/exclude globs, and per-corpus MCP tools.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    model: str = "voyage-3"
    chunk_tokens: int = 8000
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
    max_words: int | None = None
    """Per-corpus override of the global [inference] summary_max_words.

    Leave unset (None) to inherit the global default. Set to a positive
    integer to cap this corpus's per-file (or per-section) summaries at
    that word count. Short-note corpora may lower to 50; dense manuscript
    corpora may raise to 200.
    """


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus: CorpusSection
    embedding: EmbeddingSection = Field(default_factory=EmbeddingSection)
    ontology: OntologySection = Field(default_factory=OntologySection)
    mcp: McpSection = Field(default_factory=McpSection)
    summarization: SummarizationSection = Field(default_factory=SummarizationSection)

    @classmethod
    def load(cls, path: Path) -> CorpusConfig:
        try:
            raw = tomllib.loads(path.read_text())
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
