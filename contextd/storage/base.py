"""Abstract base for storage backends plus capability introspection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

BackendName = Literal["memgraph", "kuzu"]
Origin = Literal["inferred", "structural", "manual"]


@dataclass(frozen=True)
class BackendCapabilities:
    """Static declaration of what a storage backend supports.

    Callers adapt behaviour via these flags rather than trying an
    operation and reacting to failures — e.g., the MCP ``query_graph``
    tool rejects Cypher that calls procedures the backend lacks, with
    a clear error citing the capability gap.
    """

    name: BackendName
    concurrent_writers: int
    """-1 means unlimited; integers >= 1 are the maximum concurrent writer count."""
    supports_vector_index: bool
    supports_full_text_index: bool
    supports_graph_algorithms: bool
    requires_docker: bool
    default_connection: str

    @property
    def unlimited_writers(self) -> bool:
        return self.concurrent_writers == -1


class GraphStore(ABC):
    """Common interface for pluggable graph stores (Memgraph, KùzuDB).

    All higher layers (indexer, MCP server, CLI) depend on this ABC.
    Backend-specific imports are confined to ``contextd/storage/memgraph.py``
    and ``contextd/storage/kuzu.py``; a CI grep step (see .github/workflows/
    ci.yml) enforces the separation.
    """

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def apply_migrations(self, migrations: Sequence[Any]) -> None: ...

    @abstractmethod
    def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
        """Insert or update a node; return its canonical id."""

    @abstractmethod
    def upsert_edge(
        self,
        src_id: str,
        dst_id: str,
        label: str,
        origin: Origin,
        properties: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    def delete_edges(
        self,
        src_id: str,
        *,
        origin: Origin | None = None,
        edge_type: str | None = None,
    ) -> None: ...

    @abstractmethod
    def exec_read(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def exec_write(self, cypher: str, params: dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def vector_search(
        self,
        label: str,
        property_name: str,
        query: list[float],
        k: int,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def full_text_search(
        self,
        label: str,
        property_name: str,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]: ...

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...
