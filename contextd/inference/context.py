"""Shared per-unit context contract for summarise/relate inference calls.

Two halves of the pipeline compose through this module:

* the extraction side renders :func:`identity_vars` into prompt templates so
  the model knows *what* it is reading (previously it received an anonymous
  text blob — no path, no title, no corpus);
* the resolution side consumes :class:`UnitIdentity` (``src_label``/``src_id``)
  for retrieval and edge writing, and supplies a :class:`CandidateBundle` of
  real graph nodes the model can cite by exact id instead of inventing names.

``CandidateRetriever`` is the seam between the two: the relate phases call
``for_unit`` once per section/file, inside the worker, so candidates are
retrieved per call — not sampled once per phase.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from contextd.storage.base import GraphStore


@dataclass(frozen=True)
class UnitIdentity:
    """Identity of the text unit being summarised or related."""

    corpus: str
    file_path: str  # canonical forward-slash path == File.path
    rel_path: str  # posix path relative to corpus root (prompt + routing key)
    suffix: str  # ".md", ".lua", ...
    src_label: str  # "File" | "Section"
    src_id: str  # File.path or Section.id ("<path>#<anchor>")
    title: str | None = None
    anchor: str | None = None
    parent_titles: tuple[str, ...] = ()


def identity_vars(identity: UnitIdentity | None) -> dict[str, str]:
    """Template variables for the prompt's Source block.

    Every key is present on every call — ``PromptRenderer._substitute`` raises
    on a template placeholder missing from kwargs but silently ignores extra
    kwargs, so returning the full set keeps both new packaged templates and
    stale user copies (which lack the placeholders) working.
    """
    if identity is None:
        return {
            "source_path": "",
            "section_title": "",
            "section_anchor": "",
            "parent_chain": "",
            "corpus_name": "",
        }
    return {
        "source_path": identity.rel_path,
        "section_title": identity.title or "",
        "section_anchor": identity.anchor or "",
        "parent_chain": " > ".join(identity.parent_titles),
        "corpus_name": identity.corpus,
    }


@dataclass(frozen=True)
class SectionCandidate:
    id: str  # Section.id — the only directly-resolvable Section target form
    title: str


@dataclass(frozen=True)
class FileCandidate:
    path: str
    name: str


@dataclass(frozen=True)
class CandidateBundle:
    """Structured per-unit candidate context offered to the relate model."""

    entities_by_label: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    sections: tuple[SectionCandidate, ...] = ()
    files: tuple[FileCandidate, ...] = ()

    @classmethod
    def empty(cls) -> CandidateBundle:
        return cls()

    def render(self, *, per_group_cap: int = 15) -> str:
        """Deterministic prompt block: labels sorted, per-group caps applied."""
        lines: list[str] = []
        for label in sorted(self.entities_by_label):
            names = self.entities_by_label[label][:per_group_cap]
            if names:
                lines.append(f"{label}: " + "; ".join(names))
        if self.sections:
            lines.append("Sections (cite by the exact id shown):")
            lines.extend(f"  {c.id} — {c.title}" for c in self.sections[:per_group_cap])
        if self.files:
            lines.append("Files (cite by the exact path or bare filename shown):")
            lines.extend(f"  {c.path}" for c in self.files[:per_group_cap])
        return "\n".join(lines) if lines else "(none known yet)"


class CandidateRetriever(Protocol):
    """Per-unit candidate lookup, called inside the relate worker."""

    def for_unit(self, store: GraphStore, *, identity: UnitIdentity) -> CandidateBundle: ...


class EmptyRetriever:
    """No-op retriever: behavior-identical to the pre-candidate pipeline."""

    def for_unit(self, store: GraphStore, *, identity: UnitIdentity) -> CandidateBundle:
        return CandidateBundle.empty()
