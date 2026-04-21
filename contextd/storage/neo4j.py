"""Neo4j Community backend using the Bolt protocol via the official neo4j driver.

Neo4j is the reference Cypher implementation — LLM-emitted Cypher (from
the translator) executes most reliably against it. The backend manages a
single driver instance; individual operations open short-lived sessions
per call, which is the idiomatic pattern for the neo4j-python-driver.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from neo4j import GraphDatabase

from contextd.storage.base import BackendCapabilities, GraphStore, Origin
from contextd.storage.migration import Migration, MigrationRunner

if TYPE_CHECKING:
    # Neo4jConfig lands in contextd.config in Task 11.5; until then the
    # annotation is type-only and the tests pass a pydantic shim with the
    # same shape. Once 11.5 lands, move this to an unconditional import
    # and drop the type: ignore directives on __init__.
    from contextd.config import Neo4jConfig  # type: ignore[attr-defined]

_CAPABILITIES = BackendCapabilities(
    name="neo4j",
    concurrent_writers=-1,
    supports_vector_index=True,
    supports_full_text_index=True,
    supports_graph_algorithms=True,
    requires_docker=True,
    default_connection="bolt://127.0.0.1:7687",
)


class Neo4jBackend(GraphStore):
    def __init__(self, config: Neo4jConfig) -> None:  # type: ignore[no-any-unimported]
        self._cfg = config
        self._driver: Any | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return _CAPABILITIES

    def connect(self) -> None:
        uri = f"bolt://{self._cfg.host}:{self._cfg.port}"
        self._driver = GraphDatabase.driver(uri, auth=(self._cfg.user, self._cfg.password))

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def apply_migrations(self, migrations: Sequence[Any]) -> None:
        typed: list[Migration] = list(migrations)
        MigrationRunner(self, typed).apply()

    # Remaining methods (upsert_node, upsert_edge, delete_edges, exec_read,
    # exec_write, vector_search, full_text_search) implemented in Tasks 11.3-4.
    # Raise NotImplementedError explicitly so the ABC's abstractmethod contract
    # doesn't silently pass at instantiation time.
    def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
        raise NotImplementedError("Task 11.3")

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
        raise NotImplementedError("Task 11.3")

    def delete_edges(
        self,
        src_id: str,
        *,
        origin: Origin | None = None,
        edge_type: str | None = None,
        src_label: str | None = None,
    ) -> None:
        raise NotImplementedError("Task 11.3")

    def exec_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError("Task 11.3")

    def exec_write(self, cypher: str, params: dict[str, Any] | None = None) -> None:
        raise NotImplementedError("Task 11.3")

    def vector_search(
        self,
        label: str,
        property_name: str,
        query: list[float],
        k: int,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Task 11.4")

    def full_text_search(
        self,
        label: str,
        property_name: str,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Task 11.4")
