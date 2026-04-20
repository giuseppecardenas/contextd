"""Vector-similarity entity resolver (spec §5.6).

Before creating a new node of a given label, check for a semantically
similar existing node. The default threshold is 0.92 cosine similarity
for strong match; corpus config can tune per-corpus.
"""

from __future__ import annotations

from collections.abc import Callable

from contextd.storage.base import GraphStore

Embedder = Callable[[list[str]], list[list[float]]]


class EntityResolver:
    def __init__(self, store: GraphStore, embedder: Embedder, *, threshold: float = 0.92) -> None:
        self._store = store
        self._embed = embedder
        self._threshold = threshold

    def resolve(self, label: str, name: str) -> str | None:
        """Return the canonical id of a matching existing node, or None."""
        [vector] = self._embed([name])
        results = self._store.vector_search(
            label=label,
            property_name="embedding",
            query=vector,
            k=1,
            threshold=self._threshold,
        )
        if not results:
            return None
        top = results[0]
        if float(top.get("score", 0.0)) < self._threshold:
            return None
        node = top["node"]
        # Return path, id, or name — whichever is the canonical key for this label.
        for key in ("path", "id", "name"):
            if key in node:
                return str(node[key])
        return None
