"""Memgraph backend using the Bolt protocol via gqlalchemy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gqlalchemy import Memgraph

from contextd.config import MemgraphConfig
from contextd.storage.base import BackendCapabilities, GraphStore, Origin
from contextd.storage.migration import Migration, MigrationRunner

_CAPABILITIES = BackendCapabilities(
    name="memgraph",
    concurrent_writers=-1,
    supports_vector_index=True,
    supports_full_text_index=True,
    supports_graph_algorithms=True,
    requires_docker=True,
    default_connection="bolt://127.0.0.1:7687",
)


class MemgraphBackend(GraphStore):
    def __init__(self, config: MemgraphConfig) -> None:
        self._cfg = config
        # gqlalchemy is untyped; the client is effectively Any at the type level.
        self._client: Any | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return _CAPABILITIES

    def connect(self) -> None:
        self._client = Memgraph(host=self._cfg.host, port=self._cfg.port)

    def close(self) -> None:
        # gqlalchemy's Memgraph class manages connection pooling internally and
        # does not expose an explicit close. Releasing the reference is sufficient;
        # the cached Bolt connection is torn down on GC.
        self._client = None

    def apply_migrations(self, migrations: Sequence[Any]) -> None:
        typed: list[Migration] = list(migrations)
        MigrationRunner(self, typed).apply()

    def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
        assert self._client is not None
        key = _primary_key_for(label, properties)
        cypher = f"MERGE (n:{label} {{{key}: $key_value}}) SET n += $props RETURN n.{key} AS id"
        rows = list(
            self._client.execute_and_fetch(
                cypher, {"key_value": properties[key], "props": properties}
            )
        )
        return str(rows[0]["id"])

    def upsert_edge(
        self,
        src_id: str,
        dst_id: str,
        label: str,
        origin: Origin,
        properties: dict[str, Any] | None = None,
        *,
        src_label: str | None = None,
        dst_label: str | None = None,
    ) -> None:
        # Memgraph is schema-free; src_label/dst_label are advisory and used
        # only to narrow the MATCH for efficiency when provided.
        assert self._client is not None
        props = {**(properties or {}), "origin": origin}
        src_frag = f":{src_label}" if src_label else ""
        dst_frag = f":{dst_label}" if dst_label else ""
        cypher = (
            f"MATCH (a{src_frag}), (b{dst_frag}) "
            "WHERE (a.path = $src OR a.id = $src OR a.name = $src) "
            "AND (b.path = $dst OR b.id = $dst OR b.name = $dst) "
            f"MERGE (a)-[r:{label}]->(b) "
            "SET r += $props"
        )
        self._client.execute(cypher, {"src": src_id, "dst": dst_id, "props": props})

    def delete_edges(
        self,
        src_id: str,
        *,
        origin: Origin | None = None,
        label: str | None = None,
        src_label: str | None = None,
    ) -> None:
        assert self._client is not None
        conditions = ["(a.path = $src OR a.id = $src OR a.name = $src)"]
        params: dict[str, Any] = {"src": src_id}
        if origin is not None:
            conditions.append("r.origin = $origin")
            params["origin"] = origin
        src_frag = f":{src_label}" if src_label else ""
        label_fragment = f":{label}" if label else ""
        cypher = (
            f"MATCH (a{src_frag})-[r{label_fragment}]->() WHERE {' AND '.join(conditions)} DELETE r"
        )
        self._client.execute(cypher, params)

    def exec_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assert self._client is not None
        return list(self._client.execute_and_fetch(cypher, params or {}))

    def exec_write(self, cypher: str, params: dict[str, Any] | None = None) -> None:
        assert self._client is not None
        self._client.execute(cypher, params or {})

    def vector_search(
        self,
        label: str,
        property_name: str,
        query: list[float],
        k: int,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        filter_clause = f"WHERE score >= {threshold}" if threshold is not None else ""
        cypher = (
            f"CALL vector_search.search('{label}_{property_name}_idx', $k, $q) "
            "YIELD node, similarity AS score "
            f"{filter_clause} "
            "RETURN node, score "
            "ORDER BY score DESC"
        )
        return self.exec_read(cypher, {"k": k, "q": query})

    def full_text_search(
        self,
        label: str,
        property_name: str,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        cypher = (
            f"CALL text_search.search('{label}_{property_name}_ft', $q) "
            "YIELD node, score "
            "RETURN node, score "
            "ORDER BY score DESC "
            f"LIMIT {k}"
        )
        return self.exec_read(cypher, {"q": query})


def _primary_key_for(label: str, props: dict[str, Any]) -> str:
    for candidate in ("path", "id", "name"):
        if candidate in props:
            return candidate
    raise ValueError(f"No primary-key property found for {label!r}: need one of path/id/name")
