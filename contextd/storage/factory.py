"""Factory that constructs the configured GraphStore backend."""

from __future__ import annotations

from contextd.config import Config
from contextd.storage.base import GraphStore


class StorageFactoryError(RuntimeError):
    """Raised when the configured backend cannot be constructed."""


def build_graph_store(cfg: Config) -> GraphStore:
    # The deferred import keeps the Neo4j SDK confined to the storage layer,
    # preserving the abstraction-invariant grep in CI and the single
    # instantiation seam should a second backend ever be added.
    backend = cfg.storage.backend
    if backend == "neo4j":
        from contextd.storage.neo4j import Neo4jBackend

        return Neo4jBackend(cfg.storage.neo4j)
    raise StorageFactoryError(f"Unknown backend: {backend!r}")
