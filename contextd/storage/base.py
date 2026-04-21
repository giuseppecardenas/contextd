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
        are required on schema-first backends (Kuzu) and advisory on
        schema-free backends (Memgraph, which uses them to narrow the
        MATCH). When omitted on Kuzu, the backend raises because REL tables
        declare fixed FROM/TO label pairs and a lookup without labels is
        ambiguous.
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
        schema-first backends (Kuzu) so the key-property lookup
        (path/id/name) is unambiguous against the node table schema.
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
        inputs. The returned dicts have ``node`` and ``score`` keys on both
        backends. ``score`` is cosine similarity in ``[0, 1]`` (higher is more
        similar). Kuzu's index procedure natively returns distance; the backend
        converts to similarity internally. The conversion assumes the index was
        declared with ``metric := 'cosine'`` and that stored + query vectors are
        unit-normalised — Voyage-3 satisfies both.
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
