"""Factory that constructs the configured GraphStore backend."""

from __future__ import annotations

from contextd.config import Config
from contextd.storage.base import GraphStore


class StorageFactoryError(RuntimeError):
    """Raised when the configured backend cannot be constructed."""


def build_graph_store(cfg: Config) -> GraphStore:
    # Deferred imports keep backend SDKs out of the import path when the other
    # backend is selected (preserves the abstraction-invariant grep in CI).
    # The type: ignores come off in tasks 3.2 / 3.3 when the backend modules land.
    backend = cfg.storage.backend
    if backend == "memgraph":
        from contextd.storage.memgraph import MemgraphBackend  # type: ignore[import-untyped]

        return MemgraphBackend(cfg.storage.memgraph)  # type: ignore[no-any-return]
    if backend == "kuzu":
        from contextd.storage.kuzu import KuzuBackend  # type: ignore[import-untyped]

        return KuzuBackend(cfg.storage.kuzu)  # type: ignore[no-any-return]
    raise StorageFactoryError(f"Unknown backend: {backend!r}")
