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
        assert self._conn is not None
        key = _primary_key_for(properties)
        prop_list = ", ".join(f"{k}: ${k}" for k in properties)
        cypher = f"MERGE (n:{label} {{{prop_list}}})"
        self._conn.execute(cypher, properties)
        return str(properties[key])

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
        # Kuzu REL tables are declared with FROM/TO label pairs; a MERGE that
        # omits the endpoint labels is ambiguous ("Create rel r bound by
        # multiple node labels is not supported"). Both labels are required.
        if src_label is None or dst_label is None:
            raise ValueError(
                "KuzuBackend.upsert_edge requires both src_label and dst_label; "
                f"got src_label={src_label!r}, dst_label={dst_label!r}"
            )
        assert self._conn is not None
        props = {**(properties or {}), "origin": origin}
        # Kuzu rejects WHERE clauses that reference properties the label does
        # not declare, so select the one primary-key property per endpoint
        # label rather than OR-ing over path/id/name.
        src_key = _PRIMARY_KEY_BY_LABEL.get(src_label, "id")
        dst_key = _PRIMARY_KEY_BY_LABEL.get(dst_label, "id")
        cypher = (
            f"MATCH (a:{src_label} {{{src_key}: $src}}), "
            f"(b:{dst_label} {{{dst_key}: $dst}}) "
            f"MERGE (a)-[r:{label}]->(b) "
            "SET r.origin = $origin"
        )
        self._conn.execute(cypher, {"src": src_id, "dst": dst_id, **props})

    def delete_edges(
        self,
        src_id: str,
        *,
        origin: Origin | None = None,
        label: str | None = None,
        src_label: str | None = None,
    ) -> None:
        if src_label is None:
            raise ValueError(
                "KuzuBackend.delete_edges requires src_label; node tables "
                "do not share a common set of key properties."
            )
        assert self._conn is not None
        key = _PRIMARY_KEY_BY_LABEL.get(src_label, "id")
        conditions: list[str] = []
        params: dict[str, Any] = {"src": src_id}
        if origin is not None:
            conditions.append("r.origin = $origin")
            params["origin"] = origin
        where_clause = f"WHERE {' AND '.join(conditions)} " if conditions else ""
        label_fragment = f":{label}" if label else ""
        cypher = (
            f"MATCH (a:{src_label} {{{key}: $src}})-[r{label_fragment}]->() {where_clause}DELETE r"
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
        cypher = (
            f"CALL QUERY_VECTOR_INDEX('{label}_{property_name}_idx', $q, $k) "
            "RETURN node, distance "
            "ORDER BY distance ASC"
        )
        rows = self.exec_read(cypher, {"k": k, "q": query})
        if threshold is not None:
            rows = [r for r in rows if r["distance"] <= (1 - threshold)]
        return rows

    def full_text_search(
        self,
        label: str,
        property_name: str,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        cypher = (
            f"CALL QUERY_FTS_INDEX('{label}', '{label}_{property_name}_ft', $q) "
            "RETURN node, score "
            "ORDER BY score DESC "
            f"LIMIT {k}"
        )
        return self.exec_read(cypher, {"q": query})


def _primary_key_for(props: dict[str, Any]) -> str:
    for candidate in ("path", "id", "name"):
        if candidate in props:
            return candidate
    raise ValueError("No primary-key property: need one of path/id/name")


# Per-label primary-key property for Kuzu edge-MATCH queries (mirrors the PK
# declarations in migrations/kuzu/_0001_baseline.py).
_PRIMARY_KEY_BY_LABEL: dict[str, str] = {
    "File": "path",
    "Section": "id",
    "Artifact": "id",
    "Ticket": "id",
    "Pattern": "name",
    "Technology": "name",
    "Client": "name",
    "Repo": "name",
    "Service": "name",
    "Integration": "name",
    "Risk": "description",
    "WorkSession": "id",
    "Corpus": "name",
    "Meta": "schema_version",
}
