"""Vector-similarity entity resolver (spec §5.6).

Before creating a new node of a given label, check for a semantically
similar existing node. The default threshold is 0.92 cosine similarity
(on the backend's normalized ``(1+cos)/2`` scale — orthogonal is 0.5)
for strong match; corpus config can tune per-corpus.

Wired as rung (d) of ``EntityCascadeResolver``: the cascade embeds the
candidate name once, delegates the search here with ``vector=``, and
reuses the same vector at mint time when nothing matches — one embed
call serves both the check and the mint. ``corpus=`` post-filters
results because ``GraphStore.vector_search`` has no corpus parameter.
"""

from __future__ import annotations

from collections.abc import Callable

from contextd.storage._keys import primary_key_for
from contextd.storage.base import GraphStore

Embedder = Callable[[list[str]], list[list[float]]]


class EntityResolver:
    def __init__(self, store: GraphStore, embedder: Embedder, *, threshold: float = 0.92) -> None:
        self._store = store
        self._embed = embedder
        self._threshold = threshold

    def resolve_scored(
        self,
        label: str,
        name: str,
        *,
        vector: list[float] | None = None,
        corpus: str | None = None,
    ) -> tuple[str, float] | None:
        """Return ``(canonical_pk, score)`` of a matching node, or ``None``.

        ``vector`` skips the embed call (the caller already computed it);
        ``corpus`` post-filters hits to one corpus. ``k=5`` leaves headroom
        for the post-filter to discard other-corpus neighbours.
        """
        vec = vector if vector is not None else self._embed([name])[0]
        results = self._store.vector_search(
            label=label,
            property_name="embedding",
            query=vec,
            k=5,
            threshold=self._threshold,
        )
        key = primary_key_for(label)
        for top in results:
            score = float(top.get("score", 0.0))
            if score < self._threshold:
                continue
            node = top["node"]
            if corpus is not None and node.get("corpus") != corpus:
                continue
            if key in node:
                return str(node[key]), score
        return None

    def resolve(
        self,
        label: str,
        name: str,
        *,
        vector: list[float] | None = None,
        corpus: str | None = None,
    ) -> str | None:
        """Return the canonical id of a matching existing node, or None."""
        scored = self.resolve_scored(label, name, vector=vector, corpus=corpus)
        return scored[0] if scored is not None else None
