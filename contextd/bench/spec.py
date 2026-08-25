"""Bench spec: the labelled query file ``contextd bench`` scores against.

A spec is a list of queries, each with the targets a good retriever should
surface. TOML is the supported format (parsed with the stdlib ``tomllib``);
YAML is accepted only when ``pyyaml`` happens to be importable — it is not a
dependency and is never installed on the user's behalf.

.. code-block:: toml

    [[queries]]
    q = "what do the notes say about sourdough hydration"
    expect = [
        { path = "note-3.md", anchor = "hydration", lines = [12, 30] },
        { path = "note-7.md" },
    ]
    k = 5   # optional per-query override of the run's --k

``lines`` is ``[start, end)`` 0-based, like ``ChunkSpan``; ``anchor`` is the
Section's GitHub-style heading slug (the part after ``#`` in a Section id).
Both are optional: a bare ``path`` is satisfied by any hit in that file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from contextd.bench.metrics import Target

_YAML_SUFFIXES: frozenset[str] = frozenset({".yaml", ".yml"})


class BenchSpecError(ValueError):
    """Raised when a bench file is missing, unreadable, or malformed."""


class _TargetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    anchor: str | None = Field(default=None, min_length=1)
    lines: tuple[int, int] | None = None

    @model_validator(mode="after")
    def _lines_well_formed(self) -> _TargetModel:
        if self.lines is not None:
            start, end = self.lines
            if start < 0 or end <= start:
                raise ValueError(
                    f"lines must be [start, end) with 0 <= start < end, got {list(self.lines)}"
                )
        return self

    def to_target(self) -> Target:
        return Target(path=self.path, anchor=self.anchor, lines=self.lines)


class _QueryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    q: str = Field(min_length=1)
    expect: list[_TargetModel] = Field(min_length=1)
    k: int | None = Field(default=None, ge=1)


class _SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queries: list[_QueryModel] = Field(min_length=1)


@dataclass
class BenchQuery:
    q: str
    expect: list[Target]
    k: int | None = None
    """Per-query override of the run's top-k; ``None`` uses the run's ``k``."""


@dataclass
class BenchSpec:
    queries: list[BenchQuery] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, source: str = "<mapping>") -> BenchSpec:
        """Validate an already-parsed mapping; ``source`` names it in errors."""
        try:
            model = _SpecModel.model_validate(raw)
        except ValidationError as exc:
            raise BenchSpecError(f"invalid bench spec {source}: {exc}") from exc
        return cls(
            queries=[
                BenchQuery(q=q.q, expect=[t.to_target() for t in q.expect], k=q.k)
                for q in model.queries
            ]
        )

    @classmethod
    def load(cls, path: Path) -> BenchSpec:
        return load_spec(path)


def _parse_yaml(text: str, path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError as exc:
        raise BenchSpecError(
            f"{path}: YAML bench files need the optional 'pyyaml' package, which is not "
            "installed; write the spec as .toml (the supported format) instead"
        ) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise BenchSpecError(f"invalid YAML syntax in {path}: {exc}") from exc


def load_spec(path: Path) -> BenchSpec:
    """Parse and validate the bench file at ``path``.

    Raises :class:`BenchSpecError` (a ``ValueError``) naming the file for a
    missing file, an unsupported suffix, a syntax error, or a schema error.
    """
    if not path.is_file():
        raise BenchSpecError(f"bench spec not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchSpecError(f"cannot read bench spec {path}: {exc}") from exc

    suffix = path.suffix.lower()
    raw: Any
    if suffix == ".toml":
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise BenchSpecError(f"invalid TOML syntax in {path}: {exc}") from exc
    elif suffix in _YAML_SUFFIXES:
        raw = _parse_yaml(text, path)
    else:
        raise BenchSpecError(
            f"{path}: unsupported bench spec format {suffix or '(no suffix)'!r}; "
            "use .toml (or .yaml when pyyaml is installed)"
        )
    if not isinstance(raw, dict):
        raise BenchSpecError(f"invalid bench spec {path}: top level must be a table")
    return BenchSpec.from_mapping(raw, source=str(path))
