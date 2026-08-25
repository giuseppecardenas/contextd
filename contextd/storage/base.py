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
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest neighbours of ``query`` by cosine similarity.

        ``threshold`` is a cosine-similarity floor in ``[0.0, 1.0]`` (higher is
        more similar). Implementations MUST raise ``ValueError`` on non-finite
        inputs. The returned dicts have ``node`` and ``score`` keys.
        ``score`` is cosine similarity in ``[0, 1]`` (higher is more
        similar). Neo4j normalises via ``(1 + dot) / 2`` so orthogonal vectors
        score 0.5 (not 0.0) and identical-direction vectors score 1.0;
        callers that pick thresholds must account for this normalisation.

        ``filters`` is an equality map ``{property: value}``; a returned node
        must satisfy every entry (``node.corpus = "docs"`` and so on). The
        native index procedures cannot pre-filter, so implementations apply
        the predicates *after* the index lookup and over-fetch from the index
        (``k`` times a backend-specific factor, capped) so that a filtered
        result set can still reach ``k`` rows. A result shorter than ``k`` is
        therefore possible when the matching nodes are rare among the index's
        nearest neighbours. Filter keys are validated as safe identifiers
        (they are interpolated into the query); values are always bound as
        parameters. ``None`` or ``{}`` applies no filter.
        """

    @abstractmethod
    def full_text_search(
        self,
        label: str,
        property_name: str,
        query: str,
        k: int,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-``k`` lexical matches for ``query`` on one index.

        ``property_name`` selects the index by the ``{Label}_{property}_ft``
        naming convention; a multi-property index (``Chunk_text_ft`` covers
        ``text``, ``prefix`` and ``keywords``) is addressed by its lead
        property. ``score`` is the backend's raw relevance score (Lucene BM25
        on Neo4j — unbounded, not comparable with ``vector_search`` scores).

        ``filters`` has the same contract as on :meth:`vector_search`: an
        equality map applied after the index procedure, with the index
        over-fetched so the filtered set can still fill ``k`` rows.
        """

    @abstractmethod
    def upsert_nodes(self, label: str, rows: list[dict[str, Any]]) -> int:
        """Batch insert-or-update nodes of one label; return the rows written.

        Each row is MERGEd on the label's primary key (``PRIMARY_KEY_BY_LABEL``)
        and the remaining properties are set additively, exactly as
        :meth:`upsert_node` does for a single node — but in bulk, so callers
        that write thousands of nodes (chunk generation writes roughly ten
        chunks per section) avoid one round trip per node. Implementations
        may split ``rows`` into transaction-sized batches.

        Every row MUST carry the primary key: a row without it raises
        ``ValueError`` naming the missing key and the row's index, before
        anything is written. An empty ``rows`` list returns ``0`` without
        touching the store.
        """

    @abstractmethod
    def delete_nodes(self, label: str, *, where: dict[str, Any]) -> int:
        """Delete every ``label`` node matching all of ``where``; return the count.

        ``where`` is an equality map ``{property: value}`` and MUST be
        non-empty — implementations raise ``ValueError`` otherwise, because
        an unfiltered delete of a whole label is never a legitimate
        operation (corpus deletion and refresh both scope by ``corpus`` at
        minimum). Deletion detaches the nodes first, so every edge touching
        them — whatever its ``origin`` — goes with them; callers rely on this
        to drop a parent's structural ``CONTAINS`` / ``NEXT_SIBLING`` edges
        together with its chunks. Keys are validated as safe identifiers;
        values are bound as parameters.
        """

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...
