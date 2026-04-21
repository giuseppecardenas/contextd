"""Neo4j Community backend using the Bolt protocol via the official neo4j driver.

Neo4j is the reference Cypher implementation — LLM-emitted Cypher (from
the translator) executes most reliably against it. The backend manages a
single driver instance; individual operations open short-lived sessions
per call, which is the idiomatic pattern for the neo4j-python-driver.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from neo4j import Driver, GraphDatabase

from contextd.storage._identifiers import validate_identifier
from contextd.storage._keys import primary_key_for
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
        self._driver: Driver | None = None

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

    def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
        assert self._driver is not None
        validate_identifier(label, kind="label")
        key = primary_key_for(label)
        if key not in properties:
            raise ValueError(
                f"upsert_node({label!r}, ...) missing required primary key "
                f"{key!r}; properties were {sorted(properties)}"
            )
        cypher = f"MERGE (n:{label} {{{key}: $key_value}}) SET n += $props RETURN n.{key} AS id"
        with self._driver.session() as session:
            result = session.run(cypher, key_value=properties[key], props=properties)
            row = result.single()
            assert row is not None
            return str(row["id"])

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
        # Both labels required: a MERGE without the endpoint label match silently
        # binds zero rows on schema-free Neo4j (unlike Memgraph's OR-across-PKs
        # fallback), which would fail to create the edge with no visible error.
        # Matches KuzuBackend's strict-labels contract.
        if src_label is None or dst_label is None:
            raise ValueError(
                "Neo4jBackend.upsert_edge requires both src_label and dst_label; "
                f"got src_label={src_label!r}, dst_label={dst_label!r}"
            )
        assert self._driver is not None
        validate_identifier(src_label, kind="src_label")
        validate_identifier(dst_label, kind="dst_label")
        validate_identifier(edge_type, kind="edge_type")
        props = {**(properties or {}), "origin": origin}
        src_key = primary_key_for(src_label)
        dst_key = primary_key_for(dst_label)
        cypher = (
            f"MATCH (a:{src_label}), (b:{dst_label}) "
            f"WHERE a.{src_key} = $src AND b.{dst_key} = $dst "
            f"MERGE (a)-[r:{edge_type}]->(b) "
            f"SET r += $props"
        )
        with self._driver.session() as session:
            session.run(cypher, src=src_id, dst=dst_id, props=props)

    def delete_edges(
        self,
        src_id: str,
        *,
        origin: Origin | None = None,
        edge_type: str | None = None,
        src_label: str | None = None,
    ) -> None:
        if origin is None and edge_type is None:
            raise ValueError(
                "delete_edges requires at least one of origin or edge_type — "
                "an unfiltered delete would wipe structural and manual edges."
            )
        # src_label required: without it the MATCH would silently bind zero rows
        # on schema-free Neo4j when the endpoint is not a File (Section/Artifact/
        # Pattern/etc. have non-"path" PKs). Matches KuzuBackend's strict-labels
        # contract.
        if src_label is None:
            raise ValueError(
                "Neo4jBackend.delete_edges requires src_label; node tables "
                "do not share a common set of key properties."
            )
        assert self._driver is not None
        validate_identifier(src_label, kind="src_label")
        if edge_type is not None:
            validate_identifier(edge_type, kind="edge_type")
        src_key = primary_key_for(src_label)
        conditions: list[str] = [f"a.{src_key} = $src"]
        params: dict[str, Any] = {"src": src_id}
        if origin is not None:
            conditions.append("r.origin = $origin")
            params["origin"] = origin
        edge_fragment = f":{edge_type}" if edge_type else ""
        cypher = (
            f"MATCH (a:{src_label})-[r{edge_fragment}]->() "
            f"WHERE {' AND '.join(conditions)} "
            f"DELETE r"
        )
        with self._driver.session() as session:
            session.run(cypher, **params)

    def exec_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assert self._driver is not None
        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def exec_write(self, cypher: str, params: dict[str, Any] | None = None) -> None:
        assert self._driver is not None
        with self._driver.session() as session:
            session.run(cypher, params or {})

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
