"""Memgraph backend using the Bolt protocol via gqlalchemy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from gqlalchemy import Memgraph, Node, Relationship

from contextd.config import MemgraphConfig
from contextd.storage._keys import PRIMARY_KEY_BY_LABEL, primary_key_for
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
        key = primary_key_for(label)
        if key not in properties:
            raise ValueError(
                f"upsert_node({label!r}, ...) missing required primary key "
                f"{key!r}; properties were {sorted(properties)}"
            )
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
        # Memgraph is schema-free. When the caller provides a node label, we
        # use the declared PK property (File → path, Section → id, …) for an
        # unambiguous match. Without a label we fall back to OR-matching
        # against path/id/name — works but risks mis-binding when different
        # labels share a key value (File{path:"X"} vs Pattern{name:"X"}).
        assert self._client is not None
        props = {**(properties or {}), "origin": origin}
        src_pat, src_where, src_params = _endpoint_match("a", "src", src_id, src_label)
        dst_pat, dst_where, dst_params = _endpoint_match("b", "dst", dst_id, dst_label)
        where_parts = [w for w in (src_where, dst_where) if w]
        where_clause = f"WHERE {' AND '.join(where_parts)} " if where_parts else ""
        cypher = (
            f"MATCH {src_pat}, {dst_pat} {where_clause}MERGE (a)-[r:{label}]->(b) SET r += $props"
        )
        self._client.execute(cypher, {**src_params, **dst_params, "props": props})

    def delete_edges(
        self,
        src_id: str,
        *,
        origin: Origin | None = None,
        label: str | None = None,
        src_label: str | None = None,
    ) -> None:
        if origin is None and label is None:
            raise ValueError(
                "delete_edges requires at least one of origin or label — "
                "an unfiltered delete would wipe structural and manual edges."
            )
        assert self._client is not None
        src_pat, src_where, src_params = _endpoint_match("a", "src", src_id, src_label)
        params: dict[str, Any] = dict(src_params)
        where_parts: list[str] = []
        if src_where:
            where_parts.append(src_where)
        if origin is not None:
            where_parts.append("r.origin = $origin")
            params["origin"] = origin
        label_fragment = f":{label}" if label else ""
        where_clause = f"WHERE {' AND '.join(where_parts)} " if where_parts else ""
        cypher = f"MATCH {src_pat}-[r{label_fragment}]->() {where_clause}DELETE r"
        self._client.execute(cypher, params)

    def exec_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assert self._client is not None
        return [
            {k: _normalise_cell(v) for k, v in row.items()}
            for row in self._client.execute_and_fetch(cypher, params or {})
        ]

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
        # Memgraph requires the filter to follow a WITH re-projection rather
        # than dangling after YIELD — a bare `YIELD ... WHERE` parses as
        # `YIELD` in the preceding CALL and tokenizes WHERE as the start of
        # the next clause (which must be WITH/MATCH/...).
        params: dict[str, Any] = {"k": k, "q": query}
        if threshold is None:
            filter_clause = ""
        else:
            if not math.isfinite(threshold):
                raise ValueError(f"threshold must be finite; got {threshold!r}")
            filter_clause = "WITH node, score WHERE score >= $threshold "
            params["threshold"] = threshold
        cypher = (
            f"CALL vector_search.search('{label}_{property_name}_idx', $k, $q) "
            "YIELD node, similarity AS score "
            f"{filter_clause}"
            "RETURN node, score "
            "ORDER BY score DESC"
        )
        return self.exec_read(cypher, params)

    def full_text_search(
        self,
        label: str,
        property_name: str,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        # Memgraph exposes two search procedures: `text_search.search` takes a
        # Lucene expression (e.g. `data.summary:migration`) and `search_all`
        # scans every indexed property. search_all is the right fit here — the
        # caller provides a plain keyword and expects a BM25-style match across
        # the configured index. Both procedures YIELD node + score.
        cypher = (
            f"CALL text_search.search_all('{label}_{property_name}_ft', $q) "
            "YIELD node, score "
            "RETURN node, score "
            "ORDER BY score DESC "
            f"LIMIT {k}"
        )
        return self.exec_read(cypher, {"q": query})


def _normalise_cell(value: Any) -> Any:
    """Convert gqlalchemy Node / Relationship instances to plain dicts.

    Kuzu's exec_read returns plain dicts for node/relationship cells already;
    Memgraph's underlying gqlalchemy ORM yields ``Node`` / ``Relationship``
    pydantic models. Callers writing ``row["node"]["path"]`` should get the
    same shape on both backends. We return ``dict(value)`` which exposes the
    node's properties plus `_labels` / `_id` metadata (shape is not fully
    unified with Kuzu — `_id` is an int on Memgraph vs. a dict on Kuzu —
    but `node["path"]` works everywhere).
    """
    if isinstance(value, Node | Relationship):
        return dict(value)
    return value


def _endpoint_match(
    var: str, param_base: str, value: Any, label: str | None
) -> tuple[str, str, dict[str, Any]]:
    """Build a MATCH-pattern fragment + optional WHERE clause + params for an
    edge endpoint. Returned as ``(pattern, where, params)``.

    When ``label`` is provided and known to the PK map, inline the declared
    PK in the pattern — unambiguous on the Memgraph side (constraints make
    the PK unique per label) and efficient (uses the PK index). When the
    label is unknown or absent, fall back to OR-matching against
    ``path``/``id``/``name``: a wider net, but it risks mis-binding when
    different labels share a key value (e.g. ``File{path:"X"}`` and
    ``Pattern{name:"X"}``), so callers should pass ``*_label`` kwargs when
    they know the endpoint's type.
    """
    if label is not None:
        pk = PRIMARY_KEY_BY_LABEL.get(label)
        if pk is not None:
            pk_param = f"{param_base}_{pk}"
            return f"({var}:{label} {{{pk}: ${pk_param}}})", "", {pk_param: value}
    prefix = f"({var}:{label})" if label else f"({var})"
    where = (
        f"({var}.path = ${param_base} OR {var}.id = ${param_base} OR {var}.name = ${param_base})"
    )
    return prefix, where, {param_base: value}
