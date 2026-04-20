"""KùzuDB embedded-backend implementation.

Kuzu is single-writer multi-reader. The indexer owns the writer handle;
the MCP server and CLI open read-only handles. The connection model
matches Kuzu's Python SDK surface (Database + Connection objects).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import kuzu

from contextd.config import KuzuConfig
from contextd.storage._identifiers import (
    validate_identifier,
    validate_property_keys,
    validate_search_k,
    validate_threshold,
)
from contextd.storage._keys import (
    immutable_after_create_for,
    primary_key_for,
)
from contextd.storage.base import BackendCapabilities, GraphStore, Origin
from contextd.storage.migration import Migration, MigrationRunner

_CAPABILITIES = BackendCapabilities(
    name="kuzu",
    concurrent_writers=1,
    supports_vector_index=True,
    supports_full_text_index=True,
    supports_graph_algorithms=False,
    requires_docker=False,
    default_connection="~/.contextd/graph/",
)


class KuzuBackend(GraphStore):
    def __init__(self, config: KuzuConfig, *, read_only: bool = False) -> None:
        self._cfg = config
        self._db: Any | None = None
        self._conn: Any | None = None
        self._read_only = read_only

    @property
    def capabilities(self) -> BackendCapabilities:
        return _CAPABILITIES

    def connect(self) -> None:
        # Kuzu ≥ 0.10 stores the database in a single file; the path must not
        # be an existing directory. We ensure the parent dir exists and pass
        # the file path through.
        db_path = Path(self._cfg.db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(
            str(db_path),
            buffer_pool_size=self._cfg.buffer_pool_size_mb * 1024 * 1024,
            max_num_threads=self._cfg.max_threads,
            read_only=self._read_only,
        )
        self._conn = kuzu.Connection(self._db)
        if not self._read_only:
            # Kuzu is schema-first: the MigrationRunner's MATCH (m:Meta) query
            # errors before any migration runs unless the table exists. Bootstrap
            # Meta here so the runner's forward-only logic is backend-portable.
            #
            # The column shape MUST stay a subset of (or equal to) whatever a
            # future Meta-altering migration produces. The runner only reads
            # `schema_version` and `applied`; those two are load-bearing. The
            # other columns exist for parity with the Memgraph-derived design
            # and future migrations. If a migration ever changes Meta's PK,
            # update this bootstrap in lock-step.
            self._conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Meta("
                "schema_version INT64 PRIMARY KEY, "
                "contextd_version STRING, "
                "backend_name STRING, "
                "initialised_at TIMESTAMP, "
                "applied INT64[])"
            )

    def close(self) -> None:
        # Kuzu objects close on GC.
        self._conn = None
        self._db = None

    def apply_migrations(self, migrations: Sequence[Any]) -> None:
        typed: list[Migration] = list(migrations)
        MigrationRunner(self, typed).apply()

    def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
        # Two-phase upsert. Kuzu has two constraints the ABC contract must
        # reconcile:
        #
        # 1. MERGE with all properties in the match pattern fails on re-upsert
        #    with a changed non-PK value (trips the PK uniqueness constraint).
        # 2. Vector-indexed columns (e.g. File.embedding) cannot be assigned
        #    via SET after node creation — the error is "Cannot set property
        #    vec in table embeddings..." and applies even inside MERGE...
        #    ON CREATE SET. They can only be set at CREATE time.
        #
        # Resolution: if the node doesn't yet exist, CREATE with all props
        # inline; if it does, SET only the mutable (non-PK, non-vector)
        # properties. The immutable set is label-specific (see _keys.py).
        # Kuzu does not support `SET n += $props`; individual assignments are
        # required.
        #
        # Concurrency: the check-then-write pattern is TOCTOU-safe here only
        # because Kuzu is single-writer (see capabilities.concurrent_writers
        # == 1 — declared in _CAPABILITIES above). A hypothetical multi-writer
        # backend reusing this path would race between the existence check
        # and the CREATE; explicit locking would be required there. The
        # assertion below is the runtime guardrail for that assumption.
        assert self._conn is not None
        assert self.capabilities.concurrent_writers == 1, (
            "KuzuBackend.upsert_node two-phase check-then-CREATE is only safe "
            "on single-writer backends; capabilities.concurrent_writers must be 1."
        )
        validate_identifier(label, kind="label")
        key = primary_key_for(label)
        if key not in properties:
            raise ValueError(
                f"upsert_node({label!r}, ...) missing required primary key "
                f"{key!r}; properties were {sorted(properties)}"
            )
        # Kuzu f-strings property names into CREATE/SET assignments (no native
        # `SET n += $props` support), so every caller-supplied key must be a
        # safe Cypher identifier. Values still bind via $params.
        validate_property_keys(properties, context=f"upsert_node(label={label!r})")
        pk_value = properties[key]
        existing = self.exec_read(
            f"MATCH (n:{label} {{{key}: ${key}}}) RETURN n.{key} AS pk LIMIT 1",
            {key: pk_value},
        )
        if not existing:
            prop_list = ", ".join(f"{k}: ${k}" for k in properties)
            self._conn.execute(f"CREATE (n:{label} {{{prop_list}}})", properties)
            return str(pk_value)

        immutable = immutable_after_create_for(label)
        mutable = {k: v for k, v in properties.items() if k != key and k not in immutable}
        if mutable:
            assignments = ", ".join(f"n.{k} = ${k}" for k in mutable)
            params = {key: pk_value, **mutable}
            self._conn.execute(f"MATCH (n:{label} {{{key}: ${key}}}) SET {assignments}", params)
        return str(pk_value)

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
        # Kuzu REL tables are declared with FROM/TO label pairs; a MERGE that
        # omits the endpoint labels is ambiguous ("Create rel r bound by
        # multiple node labels is not supported"). Both labels are required.
        if src_label is None or dst_label is None:
            raise ValueError(
                "KuzuBackend.upsert_edge requires both src_label and dst_label; "
                f"got src_label={src_label!r}, dst_label={dst_label!r}"
            )
        assert self._conn is not None
        validate_identifier(src_label, kind="src_label")
        validate_identifier(dst_label, kind="dst_label")
        validate_identifier(edge_type, kind="edge_type")
        props = {**(properties or {}), "origin": origin}
        validate_property_keys(props, context=f"upsert_edge(edge_type={edge_type!r})")
        # Kuzu rejects WHERE clauses that reference properties the label does
        # not declare, so select the one primary-key property per endpoint
        # label rather than OR-ing over path/id/name. primary_key_for raises
        # on unknown labels — silent fallback to "id" would mis-match nodes.
        src_key = primary_key_for(src_label)
        dst_key = primary_key_for(dst_label)
        # Kuzu has no `SET r += $props`; enumerate one assignment per property
        # so non-origin properties (e.g. confidence) round-trip. The REL table
        # must declare every property column — undeclared columns surface as
        # a Kuzu binder exception with a terse "Cannot find property X for r"
        # message; wrap it with context pointing at the edge type so a caller
        # knows which REL table needs a migration.
        assignments = ", ".join(f"r.{k} = ${k}" for k in props)
        cypher = (
            f"MATCH (a:{src_label} {{{src_key}: $src}}), "
            f"(b:{dst_label} {{{dst_key}: $dst}}) "
            f"MERGE (a)-[r:{edge_type}]->(b) "
            f"SET {assignments}"
        )
        try:
            self._conn.execute(cypher, {"src": src_id, "dst": dst_id, **props})
        except RuntimeError as exc:
            # Kuzu's Python SDK raises a generic RuntimeError with a string
            # message rather than typed error codes, so the substring-match is
            # the best available signal. This is brittle across Kuzu versions:
            # if the message wording changes, the fallback `raise` path below
            # will surface the bare binder error instead of the wrapped
            # ValueError. Update the substrings in lock-step with SDK upgrades.
            msg = str(exc)
            if "Cannot find property" in msg and "for r" in msg:
                raise ValueError(
                    f"KuzuBackend.upsert_edge: REL table {edge_type!r} does not declare "
                    f"every property in {sorted(props)}. Add the missing column via a "
                    f"migration (CREATE REL TABLE ... or ALTER TABLE). "
                    f"Kuzu message: {msg}"
                ) from exc
            raise

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
        if src_label is None:
            raise ValueError(
                "KuzuBackend.delete_edges requires src_label; node tables "
                "do not share a common set of key properties."
            )
        assert self._conn is not None
        validate_identifier(src_label, kind="src_label")
        if edge_type is not None:
            validate_identifier(edge_type, kind="edge_type")
        key = primary_key_for(src_label)
        conditions: list[str] = []
        params: dict[str, Any] = {"src": src_id}
        if origin is not None:
            conditions.append("r.origin = $origin")
            params["origin"] = origin
        where_clause = f"WHERE {' AND '.join(conditions)} " if conditions else ""
        edge_fragment = f":{edge_type}" if edge_type else ""
        cypher = (
            f"MATCH (a:{src_label} {{{key}: $src}})-[r{edge_fragment}]->() {where_clause}DELETE r"
        )
        self._conn.execute(cypher, params)

    def exec_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assert self._conn is not None
        result = self._conn.execute(cypher, params or {})
        rows: list[dict[str, Any]] = []
        while result.has_next():
            rows.append(dict(zip(result.get_column_names(), result.get_next(), strict=False)))
        return rows

    def exec_write(self, cypher: str, params: dict[str, Any] | None = None) -> None:
        assert self._conn is not None
        self._conn.execute(cypher, params or {})

    def vector_search(
        self,
        label: str,
        property_name: str,
        query: list[float],
        k: int,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        # Kuzu's QUERY_VECTOR_INDEX takes (table, index, query, k) and returns
        # (node, distance) rather than Memgraph's (node, similarity). `k` is a
        # literal because Kuzu infers Python ints as INT8 and rejects the bind
        # against the expected INT64 parameter type.
        #
        # The ABC's `threshold` is expressed as cosine similarity (0..1, higher
        # is better) to match Memgraph's contract. On Kuzu this is converted to
        # a distance cap. The conversion is only correct when:
        #   - the index was declared with `metric := 'cosine'` (baseline is), AND
        #   - query and indexed vectors are unit-normalised (Voyage-3 output is).
        # Under both conditions, cosine_distance = 1 - cosine_similarity, so the
        # similarity-threshold >= t filter becomes a distance <= (1 - t) filter.
        # A callers that passes arbitrary-norm vectors through a Kuzu backend
        # gets unexpected ranking; the invariant is documented but not enforced.
        validate_identifier(label, kind="label")
        validate_identifier(property_name, kind="property_name")
        validate_search_k(k)
        validated_threshold = validate_threshold(threshold)
        cypher = (
            f"CALL QUERY_VECTOR_INDEX('{label}', '{label}_{property_name}_idx', "
            f"$q, {k}) "
            "RETURN node, distance "
            "ORDER BY distance ASC"
        )
        rows = self.exec_read(cypher, {"q": query})
        if validated_threshold is not None:
            distance_cap = 1.0 - validated_threshold
            rows = [r for r in rows if r["distance"] <= distance_cap]
        return rows

    def full_text_search(
        self,
        label: str,
        property_name: str,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        validate_identifier(label, kind="label")
        validate_identifier(property_name, kind="property_name")
        validate_search_k(k)
        cypher = (
            f"CALL QUERY_FTS_INDEX('{label}', '{label}_{property_name}_ft', $q) "
            "RETURN node, score "
            "ORDER BY score DESC "
            f"LIMIT {k}"
        )
        return self.exec_read(cypher, {"q": query})
