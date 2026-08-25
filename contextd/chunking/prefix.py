"""Chunk prefixes: the context prepended to a chunk before embedding.

``breadcrumb`` is the zero-cost form of contextual retrieval (Docling
``contextualize``, GraphRAG ``prepend_metadata``): the heading path tells the
embedder *where* a 200-token chunk sits. ``section_summary`` substitutes the
parent's LLM summary when the graph has one. ``llm`` is applied by the
indexer phase (it needs a provider); this module only knows how to fall back
when that call fails.
"""

from __future__ import annotations

from contextd.chunking.model import Chunk
from contextd.corpus_config import PrefixMode


def breadcrumb_text(breadcrumb: tuple[str, ...], rel_path: str) -> str:
    """``Doc title > Section > Subsection`` or the file path when no headings apply."""
    parts = [p.strip() for p in breadcrumb if p and p.strip()]
    return " > ".join(parts) if parts else rel_path


def static_prefix(
    mode: PrefixMode,
    *,
    breadcrumb: tuple[str, ...],
    rel_path: str,
    parent_summary: str | None,
) -> str:
    """Prefix for the modes that need no LLM call (``llm`` falls back to breadcrumb)."""
    if mode == "none":
        return ""
    crumb = breadcrumb_text(breadcrumb, rel_path)
    if mode == "section_summary" and parent_summary and parent_summary.strip():
        return f"{crumb}\n{parent_summary.strip()}"
    return crumb


def apply_prefix(chunks: list[Chunk], prefix: str) -> list[Chunk]:
    for c in chunks:
        c.prefix = prefix
    return chunks
