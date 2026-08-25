"""Neo4j Community backend using the Bolt protocol via the official neo4j driver.

Neo4j is the reference Cypher implementation — LLM-emitted Cypher (from
the translator) executes most reliably against it. The backend manages a
single driver instance; individual operations open short-lived sessions
per call, which is the idiomatic pattern for the neo4j-python-driver.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from neo4j import Driver, GraphDatabase

from contextd.config import Neo4jConfig
from contextd.storage._identifiers import (
    validate_identifier,
    validate_search_k,
    validate_threshold,
)
from contextd.storage._keys import primary_key_for
from contextd.storage.base import BackendCapabilities, GraphStore, Origin
from contextd.storage.migration import Migration, MigrationRunner

_CAPABILITIES = BackendCapabilities(
    name="neo4j",
    concurrent_writers=-1,
    supports_vector_index=True,
    supports_full_text_index=True,
    supports_graph_algorithms=True,
    requires_docker=True,
    default_connection="bolt://127.0.0.1:7687",
)


# Neo4j's vector/full-text procedures cannot pre-filter, so a filtered search
# over-fetches ``k * over_fetch_factor`` rows from the index before the WHERE
# clause; the cap bounds the procedure's work on very large ``k``.
_MAX_PROCEDURE_K = 1000
# UNWIND batch size for ``upsert_nodes``: bounds the transaction (and the
# parameter payload — chunk rows carry a 1024-float embedding each).
_UPSERT_BATCH_SIZE = 500


class Neo4jBackend(GraphStore):
    def __init__(self, config: Neo4jConfig, *, over_fetch_factor: int = 4) -> None:
        if isinstance(over_fetch_factor, bool) or not isinstance(over_fetch_factor, int):
            raise ValueError(f"over_fetch_factor must be a non-bool int; got {over_fetch_factor!r}")
        if over_fetch_factor < 1:
            raise ValueError(f"over_fetch_factor must be >= 1; got {over_fetch_factor!r}")
        self._cfg = config
        self._over_fetch_factor = over_fetch_factor
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
        # binds zero rows on schema-free Neo4j, which would fail to create the
        # edge with no visible error.
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
        # Pattern/etc. have non-"path" PKs).
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

    @staticmethod
    def _equality_predicates(
        values: dict[str, Any], *, alias: str, prefix: str, kind: str
    ) -> tuple[str, dict[str, Any]]:
        """Render ``{k: v}`` as ``alias.k = $prefix_k AND ...`` plus its params.

        Keys are interpolated (Neo4j cannot parameterise property names) and
        therefore go through ``validate_identifier``; values are always bound
        as ``$prefix_key`` parameters so they can carry any type or content.
        """
        clauses: list[str] = []
        params: dict[str, Any] = {}
        for key, value in values.items():
            validate_identifier(key, kind=kind)
            clauses.append(f"{alias}.{key} = ${prefix}_{key}")
            params[f"{prefix}_{key}"] = value
        return " AND ".join(clauses), params

    def _procedure_k(self, k: int, *, filtered: bool) -> int:
        """Rows to request from the index procedure before post-filtering."""
        if not filtered:
            return k
        return min(k * self._over_fetch_factor, _MAX_PROCEDURE_K)

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
        """Call Neo4j's db.index.vector.queryNodes procedure.

        Neo4j returns (node, score) where score is cosine similarity in [0, 1]
        (higher is more similar). Matches the ABC's contract directly — no
        distance-to-similarity conversion needed.

        Note: Neo4j normalises similarity via ``(1 + dot) / 2``, so
        orthogonal vectors score 0.5 (not 0.0); identical direction scores
        1.0; anti-parallel scores 0.0. Threshold filtering is applied
        client-side.

        ``filters`` render as ``WITH node, score WHERE node.k = $f_k AND …``
        after the CALL; because the procedure has no pre-filter the index is
        asked for ``min(k * over_fetch_factor, 1000)`` rows and the caller's
        ``k`` becomes the trailing ``LIMIT``.
        """
        assert self._driver is not None
        validate_identifier(label, kind="label")
        validate_identifier(property_name, kind="property_name")
        validate_search_k(k)
        validated_threshold = validate_threshold(threshold)
        index_name = f"{label}_{property_name}_idx"
        where, filter_params = self._equality_predicates(
            filters or {}, alias="node", prefix="f", kind="property_name"
        )
        where_clause = f"WITH node, score WHERE {where} " if where else ""
        cypher = (
            "CALL db.index.vector.queryNodes($idx, $k, $q) "
            "YIELD node, score "
            f"{where_clause}"
            "RETURN node, score "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )
        params: dict[str, Any] = {
            "idx": index_name,
            "k": self._procedure_k(k, filtered=bool(where)),
            "q": query,
            "limit": k,
            **filter_params,
        }
        with self._driver.session() as session:
            result = session.run(cypher, params)
            rows: list[dict[str, Any]] = [
                {"node": dict(r["node"]), "score": float(r["score"])} for r in result
            ]
        if validated_threshold is not None:
            rows = [r for r in rows if r["score"] >= validated_threshold]
        return rows

    def full_text_search(
        self,
        label: str,
        property_name: str,
        query: str,
        k: int,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Call Neo4j's db.index.fulltext.queryNodes procedure.

        ``score`` is a Lucene BM25 relevance score (unbounded positive float;
        higher is more relevant), NOT normalised like vector_search's
        similarity. Do not compare scores directly across the two search
        types.

        ``filters`` follow the same shape as ``vector_search``. The full-text
        procedure takes no positional ``k``, so the over-fetch is passed as
        its ``{limit: $k}`` option and the caller's ``k`` becomes the trailing
        ``LIMIT``.
        """
        assert self._driver is not None
        validate_identifier(label, kind="label")
        validate_identifier(property_name, kind="property_name")
        validate_search_k(k)
        index_name = f"{label}_{property_name}_ft"
        where, filter_params = self._equality_predicates(
            filters or {}, alias="node", prefix="f", kind="property_name"
        )
        where_clause = f"WITH node, score WHERE {where} " if where else ""
        cypher = (
            "CALL db.index.fulltext.queryNodes($idx, $q, {limit: $k}) "
            "YIELD node, score "
            f"{where_clause}"
            "RETURN node, score "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )
        params: dict[str, Any] = {
            "idx": index_name,
            "q": query,
            "k": self._procedure_k(k, filtered=bool(where)),
            "limit": k,
            **filter_params,
        }
        with self._driver.session() as session:
            result = session.run(cypher, params)
            return [{"node": dict(r["node"]), "score": float(r["score"])} for r in result]

    def upsert_nodes(self, label: str, rows: list[dict[str, Any]]) -> int:
        """Batch MERGE ``rows`` on the label's primary key via UNWIND.

        Rows are written in batches of ``_UPSERT_BATCH_SIZE`` inside one
        session so a large chunk set neither becomes one giant transaction
        nor one round trip per node. Every row is validated before the first
        write so a malformed row cannot leave a partial batch behind.
        """
        assert self._driver is not None
        validate_identifier(label, kind="label")
        key = primary_key_for(label)
        for index, row in enumerate(rows):
            if key not in row:
                raise ValueError(
                    f"upsert_nodes({label!r}, ...) row {index} missing required primary key "
                    f"{key!r}; properties were {sorted(row)}"
                )
        if not rows:
            return 0
        cypher = (
            f"UNWIND $rows AS r MERGE (n:{label} {{{key}: r.{key}}}) SET n += r "
            "RETURN count(n) AS c"
        )
        written = 0
        with self._driver.session() as session:
            for offset in range(0, len(rows), _UPSERT_BATCH_SIZE):
                batch = rows[offset : offset + _UPSERT_BATCH_SIZE]
                record = session.run(cypher, rows=batch).single()
                assert record is not None
                written += int(record["c"])
        return written

    def delete_nodes(self, label: str, *, where: dict[str, Any]) -> int:
        """DETACH DELETE the ``label`` nodes matching every ``where`` predicate."""
        if not where:
            raise ValueError(
                f"delete_nodes({label!r}) requires a non-empty where map — "
                "an unfiltered delete would wipe every node of the label."
            )
        assert self._driver is not None
        validate_identifier(label, kind="label")
        predicates, params = self._equality_predicates(
            where, alias="n", prefix="w", kind="property_name"
        )
        cypher = (
            f"MATCH (n:{label}) WHERE {predicates} "
            "WITH collect(n) AS ns "
            "UNWIND ns AS n "
            "DETACH DELETE n "
            "RETURN count(*) AS c"
        )
        with self._driver.session() as session:
            record = session.run(cypher, params).single()
            # ``count(*)`` over zero rows still yields one row carrying 0.
            assert record is not None
            return int(record["c"])
