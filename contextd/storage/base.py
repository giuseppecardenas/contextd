"""Abstract base for storage backends plus capability introspection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

BackendName = Literal["neo4j"]
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
    """Common interface for the graph + vector store (Neo4j).

    All higher layers (indexer, MCP server, CLI) depend on this ABC rather
    than on the concrete backend. Backend-specific imports are confined to
    ``contextd/storage/neo4j.py``; a CI grep step (see .github/workflows/
    ci.yml) enforces the separation, keeping the seam open for a future
    second backend without coupling consumers to today's single one.
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
        edge_type: str,
        origin: Origin,
        properties: dict[str, Any] | None = None,
        *,
        src_label: str | None = None,
        dst_label: str | None = None,
    ) -> None:
        """Create or update an edge.

        ``edge_type`` is the relationship type (REFERENCES, CONTAINS, …).
        ``src_label`` / ``dst_label`` are the endpoint *node* labels; they
        are required on Neo4j because a MERGE without the endpoint label
        match silently binds zero rows on schema-free Neo4j, which fails to
        create the edge with no visible error.
        """

    @abstractmethod
    def delete_edges(
        self,
        src_id: str,
        *,
        origin: Origin | None = None,
        edge_type: str | None = None,
        src_label: str | None = None,
    ) -> None:
        """Delete outgoing edges from ``src_id``, filtered by origin and/or type.

        Implementations MUST raise ``ValueError`` when both ``origin`` and
        ``edge_type`` are None — a caller that omits both would wipe every
        outgoing edge regardless of provenance, which violates the design
        invariant that wipe-and-replace on re-index operates only on
        ``origin="inferred"``. Callers must opt in explicitly.

        ``src_label`` narrows the MATCH to one node label; required on
        Neo4j so the key-property lookup (path/id/name) is unambiguous
        when the endpoint is not a File (Section/Artifact/Pattern/etc.
        have non-"path" PKs).
        """

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
    ) -> list[dict[str, Any]]:
        """Return nearest neighbours of ``query`` by cosine similarity.

        ``threshold`` is a cosine-similarity floor in ``[0.0, 1.0]`` (higher is
        more similar). Implementations MUST raise ``ValueError`` on non-finite
        inputs. The returned dicts have ``node`` and ``score`` keys.
        ``score`` is cosine similarity in ``[0, 1]`` (higher is more
        similar). Neo4j normalises via ``(1 + dot) / 2`` so orthogonal vectors
        score 0.5 (not 0.0) and identical-direction vectors score 1.0;
        callers that pick thresholds must account for this normalisation.
        """

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
