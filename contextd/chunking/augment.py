"""Full-text augmentation of chunks (RAGFlow auto-keyword / auto-question).

``keywords`` is indexed by the ``Chunk_text_ft`` full-text index but never
embedded, so it strengthens the lexical leg without perturbing vectors.
Static fields are copied from the parent's summary output at zero cost; the
``questions`` field costs one LLM call per parent (see :mod:`~.llm`).
"""

from __future__ import annotations

from contextd.chunking.model import Chunk
from contextd.corpus_config import AugmentField


def static_keywords(
    fields: list[AugmentField],
    *,
    key_points: list[str] | None,
    entities_mentioned: list[str] | None,
) -> list[str]:
    out: list[str] = []
    if "key_points" in fields:
        out.extend(k.strip() for k in key_points or [] if k and k.strip())
    if "entities_mentioned" in fields:
        out.extend(e.strip() for e in entities_mentioned or [] if e and e.strip())
    # Order-preserving dedup.
    seen: set[str] = set()
    unique: list[str] = []
    for k in out:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def apply_keywords(
    chunks: list[Chunk], shared: list[str], per_chunk: list[list[str]] | None
) -> None:
    for i, c in enumerate(chunks):
        own = per_chunk[i] if per_chunk is not None and i < len(per_chunk) else []
        c.keywords = [*shared, *own]
