# M11: Neo4j Backend Stand-up + Kuzu Excision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the archived Kuzu embedded backend with Neo4j Community (in Docker), make Neo4j the new default, and remove every Kuzu-specific line of code + migration + test from the tree.

**Architecture:** Contextd's `GraphStore` ABC already accommodates multiple backends. Swap one backend out (Kuzu) and one in (Neo4j) behind the same abstraction. Both remaining backends (Memgraph, Neo4j) speak Cypher over Bolt, so the dialect layer simplifies substantially; the IMMUTABLE_AFTER_CREATE workaround table becomes empty and can be deleted. Neo4j becomes the default because it is the reference Cypher implementation — LLM-generated Cypher (from the translator) executes most reliably against it.

**Tech Stack:** `neo4j` Python driver (Bolt, official), Neo4j 5.15-community Docker image, testcontainers-python's `Neo4jContainer`, existing `gqlalchemy`/Memgraph infrastructure unchanged.

**Baseline:** Commit `5a278b0` (post-SD #80). 250 unit + 65 integration tests passing. CI triad + abstraction-invariant grep green.

**Final state:** 255+ unit + 65 integration tests (one backend parametrisation swap, no net change in count). `contextd init` sets up Neo4j by default. No `kuzu` / `contextd.storage.kuzu` / `contextd/migrations/kuzu/` anywhere in the tree. `pyproject.toml` no longer depends on the `kuzu` package.

---

## File Structure (end state)

```
contextd/
  storage/
    __init__.py
    base.py              # GraphStore ABC — Kuzu-specific docstrings pruned
    factory.py           # memgraph | neo4j (kuzu branch removed)
    memgraph.py          # unchanged
    neo4j.py             # NEW — Neo4jBackend(GraphStore)
    _keys.py             # PRIMARY_KEY_BY_LABEL kept; IMMUTABLE_AFTER_CREATE_BY_LABEL removed
    _identifiers.py      # unchanged
    migration.py         # unchanged
  migrations/
    __init__.py
    memgraph/            # unchanged
    neo4j/               # NEW
      __init__.py
      _0001_baseline.py
  config.py              # KuzuConfig → Neo4jConfig; BackendName Literal updated
  docker_compose.yml     # Memgraph service kept; Neo4j service added as default
  cli/
    infra.py             # up/down probing neo4j or memgraph based on config

tests/
  integration/
    conftest.py          # backend fixture parametrised over ["memgraph", "neo4j"]
    test_neo4j_backend.py    # NEW — replaces test_kuzu_backend.py
    test_memgraph_backend.py # unchanged
    test_upsert_roundtrip.py # unchanged (parametrised via fixture)
    test_pipeline.py         # unchanged (parametrised via fixture)
    test_mcp_tools.py        # unchanged
    test_search.py           # unchanged

pyproject.toml           # kuzu removed, neo4j>=5.15 added

docs/
  design.md              # §2.5 / §12.6 / §13.4 rewrites
  implementation-plan.md # M11 section added; M3 Kuzu tasks archived

CLAUDE.md                # narrative reflecting memgraph+neo4j; SD #71 removed from Permanent limitations
README.md                # quickstart + backend choice narrative
```

---

## Task 11.1: Add Neo4j driver dependency + Neo4jBackend skeleton

**Files:**
- Modify: `pyproject.toml` (add `neo4j>=5.15`)
- Create: `contextd/storage/neo4j.py`
- Create: `tests/unit/test_neo4j_backend.py`

- [ ] **Step 1: Add driver dependency**

Edit `pyproject.toml`, `[project].dependencies`. Add line (alphabetical within the section):

```toml
"neo4j>=5.15",
```

Install in venv:

```bash
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Expected output: `Installed neo4j-5.x.x` plus existing packages unchanged.

- [ ] **Step 2: Write the failing unit test**

Create `tests/unit/test_neo4j_backend.py`:

```python
"""Unit tests for Neo4jBackend skeleton (connect / close / capabilities)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from contextd.config import Neo4jConfig


def test_capabilities_shape() -> None:
    from contextd.storage.neo4j import Neo4jBackend

    cfg = Neo4jConfig()
    backend = Neo4jBackend(cfg)
    caps = backend.capabilities
    assert caps.name == "neo4j"
    assert caps.concurrent_writers == -1  # unlimited
    assert caps.supports_vector_index is True
    assert caps.supports_full_text_index is True
    assert caps.supports_graph_algorithms is True
    assert caps.requires_docker is True
    assert caps.default_connection == "bolt://127.0.0.1:7687"


def test_connect_constructs_driver() -> None:
    from contextd.storage.neo4j import Neo4jBackend

    cfg = Neo4jConfig(host="127.0.0.1", port=7687, user="neo4j", password="test")
    with patch("contextd.storage.neo4j.GraphDatabase") as mock_gd:
        fake_driver = MagicMock()
        mock_gd.driver.return_value = fake_driver
        backend = Neo4jBackend(cfg)
        backend.connect()
        mock_gd.driver.assert_called_once_with(
            "bolt://127.0.0.1:7687", auth=("neo4j", "test")
        )
        assert backend._driver is fake_driver


def test_close_closes_driver() -> None:
    from contextd.storage.neo4j import Neo4jBackend

    cfg = Neo4jConfig()
    backend = Neo4jBackend(cfg)
    fake = MagicMock()
    backend._driver = fake
    backend.close()
    fake.close.assert_called_once()
    assert backend._driver is None
```

Note: this test imports `Neo4jConfig` from `contextd.config` — Task 11.5 adds this type. For now the test imports will fail; this is fine. Task 11.1 focuses on the Backend class; the Config integration lands in 11.5.

**Workaround to get this task unit-testable before 11.5:** create a local stub in the test file:

```python
# Temporary shim until Task 11.5 adds the real Neo4jConfig.
# Delete this block in Task 11.5.
from pydantic import BaseModel, ConfigDict


class Neo4jConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = 7687
    user: str = "neo4j"
    password: str = "neo4j"
    docker_compose_file: str = "~/.contextd/docker-compose.yml"
    memory_limit_gb: float = 1.0
    cpu_limit: float = 1.0
```

Replace the `from contextd.config import Neo4jConfig` line with the block above. Task 11.5 will delete the shim and restore the real import.

- [ ] **Step 3: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/unit/test_neo4j_backend.py -v
```

Expected: all three tests FAIL with `ModuleNotFoundError: No module named 'contextd.storage.neo4j'`.

- [ ] **Step 4: Implement the skeleton**

Create `contextd/storage/neo4j.py`:

```python
"""Neo4j Community backend using the Bolt protocol via the official neo4j driver.

Neo4j is the reference Cypher implementation — LLM-emitted Cypher (from
the translator) executes most reliably against it. The backend manages a
single driver instance; individual operations open short-lived sessions
per call, which is the idiomatic pattern for the neo4j-python-driver.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from neo4j import GraphDatabase

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


class Neo4jBackend(GraphStore):
    def __init__(self, config: Neo4jConfig) -> None:
        self._cfg = config
        self._driver: Any | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return _CAPABILITIES

    def connect(self) -> None:
        uri = f"bolt://{self._cfg.host}:{self._cfg.port}"
        self._driver = GraphDatabase.driver(
            uri, auth=(self._cfg.user, self._cfg.password)
        )

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

    def exec_read(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
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
```

Also add `"neo4j"` to the `BackendName` Literal in `contextd/storage/base.py:10`:

```python
BackendName = Literal["memgraph", "kuzu", "neo4j"]
```

(Kuzu stays in the Literal until Task 11.9 removes it.)

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_neo4j_backend.py -v
```

Expected: 3 passed.

- [ ] **Step 6: CI triad**

```bash
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -v
```

Expected: all four exit 0. Unit suite count = 253 (+3).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml contextd/storage/neo4j.py contextd/storage/base.py \
  tests/unit/test_neo4j_backend.py
git commit -m "$(cat <<'EOF'
feat(storage): Neo4jBackend skeleton — connect/close/capabilities (M11.1)

Add the neo4j Python driver dependency and scaffold Neo4jBackend with
lifecycle methods + capabilities declaration. Remaining ABC methods
raise NotImplementedError pending M11.2 (migration) and M11.3 (upserts).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11.2: Neo4j baseline migration + integration test

**Files:**
- Create: `contextd/migrations/neo4j/__init__.py` (empty)
- Create: `contextd/migrations/neo4j/_0001_baseline.py`
- Create: `tests/integration/test_neo4j_backend.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_neo4j_backend.py`:

```python
"""Integration test: apply Neo4j baseline migration and verify schema exists."""
from __future__ import annotations

import pytest
from testcontainers.neo4j import Neo4jContainer

pytestmark = pytest.mark.integration


@pytest.fixture
def neo4j_backend():
    from contextd.config import Neo4jConfig
    from contextd.storage.neo4j import Neo4jBackend

    with Neo4jContainer("neo4j:5.15-community") as container:
        cfg = Neo4jConfig(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(7687)),
            user="neo4j",
            password=container.password,
        )
        backend = Neo4jBackend(cfg)
        backend.connect()
        yield backend
        backend.close()


def test_baseline_migration_creates_indexes(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    # Verify vector index on File.embedding exists.
    rows = neo4j_backend.exec_read(
        "SHOW VECTOR INDEXES YIELD name RETURN collect(name) AS names"
    )
    names = rows[0]["names"]
    assert "File_embedding_idx" in names
    assert "Section_embedding_idx" in names

    # Verify uniqueness constraint on File.path.
    rows = neo4j_backend.exec_read(
        "SHOW CONSTRAINTS YIELD name RETURN collect(name) AS names"
    )
    names = rows[0]["names"]
    assert any("File" in n and "path" in n.lower() for n in names)
```

Note: `Neo4jConfig` uses the Task 11.1 shim for now; real config lands in 11.5. The test also relies on `exec_read` from Task 11.3, so this test will fail until 11.3 lands. Mark it with `pytest.mark.xfail(reason="blocked on M11.3")` temporarily OR write the migration now and unblock the test in 11.3. **Choose:** write migration now (this task), test runs green in 11.3.

- [ ] **Step 2: Write the migration**

Create `contextd/migrations/neo4j/__init__.py` (empty file).

Create `contextd/migrations/neo4j/_0001_baseline.py`:

```python
"""Baseline schema for Neo4j backend (reference Cypher implementation).

Neo4j 5.x declares vector and full-text indexes via CREATE ... INDEX DDL
with an OPTIONS map. Uniqueness is a CONSTRAINT, not an index. The schema
is schema-free at the node-table level (unlike Kuzu); nodes gain
properties dynamically.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    # Uniqueness constraints — one per label whose PK we pin.
    "CREATE CONSTRAINT File_path_unique IF NOT EXISTS "
    "FOR (f:File) REQUIRE f.path IS UNIQUE",
    "CREATE CONSTRAINT Section_id_unique IF NOT EXISTS "
    "FOR (s:Section) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT Artifact_id_unique IF NOT EXISTS "
    "FOR (a:Artifact) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT Ticket_id_unique IF NOT EXISTS "
    "FOR (t:Ticket) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT Pattern_name_unique IF NOT EXISTS "
    "FOR (p:Pattern) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT Technology_name_unique IF NOT EXISTS "
    "FOR (t:Technology) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT Client_name_unique IF NOT EXISTS "
    "FOR (c:Client) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT Repo_name_unique IF NOT EXISTS "
    "FOR (r:Repo) REQUIRE r.name IS UNIQUE",
    "CREATE CONSTRAINT Service_name_unique IF NOT EXISTS "
    "FOR (s:Service) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT Integration_name_unique IF NOT EXISTS "
    "FOR (i:Integration) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT Risk_desc_unique IF NOT EXISTS "
    "FOR (r:Risk) REQUIRE r.description IS UNIQUE",
    "CREATE CONSTRAINT WorkSession_id_unique IF NOT EXISTS "
    "FOR (w:WorkSession) REQUIRE w.id IS UNIQUE",
    "CREATE CONSTRAINT Corpus_name_unique IF NOT EXISTS "
    "FOR (c:Corpus) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT Meta_version_unique IF NOT EXISTS "
    "FOR (m:Meta) REQUIRE m.schema_version IS UNIQUE",
    # Vector indexes — Voyage-3 is 1024-dim, cosine similarity.
    "CREATE VECTOR INDEX File_embedding_idx IF NOT EXISTS "
    "FOR (f:File) ON f.embedding "
    "OPTIONS {indexConfig: {"
    "`vector.dimensions`: 1024, "
    "`vector.similarity_function`: 'cosine'}}",
    "CREATE VECTOR INDEX Section_embedding_idx IF NOT EXISTS "
    "FOR (s:Section) ON s.embedding "
    "OPTIONS {indexConfig: {"
    "`vector.dimensions`: 1024, "
    "`vector.similarity_function`: 'cosine'}}",
    # Full-text indexes — Lucene-backed, stemming default (English).
    "CREATE FULLTEXT INDEX File_summary_ft IF NOT EXISTS "
    "FOR (f:File) ON EACH [f.summary]",
    "CREATE FULLTEXT INDEX Artifact_description_ft IF NOT EXISTS "
    "FOR (a:Artifact) ON EACH [a.description]",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=1, name="baseline_neo4j", up=up)
```

- [ ] **Step 3: Commit (migration only; test exercises it in 11.3)**

```bash
git add contextd/migrations/neo4j/__init__.py \
  contextd/migrations/neo4j/_0001_baseline.py \
  tests/integration/test_neo4j_backend.py
git commit -m "$(cat <<'EOF'
feat(storage): Neo4j baseline migration + integration-test scaffold (M11.2)

Uniqueness constraints for every indexed node label, 1024-dim cosine
vector indexes on File.embedding and Section.embedding, full-text
indexes on File.summary and Artifact.description. Test is scaffolded;
exec_read integration unblocks it in M11.3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Note: skip `pytest tests/integration/test_neo4j_backend.py` at this point — the test relies on exec_read which is still NotImplementedError. Unit suite runs fine.

---

## Task 11.3: Implement upsert_node / upsert_edge / delete_edges / exec_read / exec_write

**Files:**
- Modify: `contextd/storage/neo4j.py` (add method bodies)
- Modify: `tests/integration/test_neo4j_backend.py` (add round-trip tests)

- [ ] **Step 1: Write failing integration tests**

Append to `tests/integration/test_neo4j_backend.py`:

```python
def test_upsert_node_roundtrip(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    pk = neo4j_backend.upsert_node(
        "File",
        {"path": "/a.md", "name": "a.md", "corpus": "test", "embedding": [0.1] * 1024},
    )
    assert pk == "/a.md"
    rows = neo4j_backend.exec_read(
        "MATCH (n:File {path: $p}) RETURN n.name AS name",
        {"p": "/a.md"},
    )
    assert rows[0]["name"] == "a.md"

    # Re-upsert updates mutable properties.
    neo4j_backend.upsert_node(
        "File",
        {"path": "/a.md", "name": "renamed.md", "corpus": "test", "embedding": [0.1] * 1024},
    )
    rows = neo4j_backend.exec_read(
        "MATCH (n:File {path: $p}) RETURN n.name AS name",
        {"p": "/a.md"},
    )
    assert rows[0]["name"] == "renamed.md"


def test_upsert_edge_and_delete_edges(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    # Two File nodes.
    neo4j_backend.upsert_node("File", {"path": "/a.md", "corpus": "t"})
    neo4j_backend.upsert_node("File", {"path": "/b.md", "corpus": "t"})
    neo4j_backend.upsert_edge(
        "/a.md", "/b.md", "REFERENCES", origin="inferred",
        properties={"confidence": 0.9},
        src_label="File", dst_label="File",
    )
    rows = neo4j_backend.exec_read(
        "MATCH ()-[r:REFERENCES]->() RETURN count(r) AS c"
    )
    assert rows[0]["c"] == 1

    # Delete inferred edges from /a.md.
    neo4j_backend.delete_edges("/a.md", origin="inferred", src_label="File")
    rows = neo4j_backend.exec_read(
        "MATCH ()-[r:REFERENCES]->() RETURN count(r) AS c"
    )
    assert rows[0]["c"] == 0


def test_delete_edges_unfiltered_raises(neo4j_backend) -> None:
    import pytest
    with pytest.raises(ValueError, match="requires at least one of"):
        neo4j_backend.delete_edges("/a.md", src_label="File")
```

- [ ] **Step 2: Verify the tests fail**

```bash
source .venv/bin/activate
pytest tests/integration/test_neo4j_backend.py -v
```

Expected: 4 tests fail (3 new + 1 existing baseline test) — all with `NotImplementedError: Task 11.3`.

- [ ] **Step 3: Implement the methods**

Replace the NotImplementedError bodies in `contextd/storage/neo4j.py`:

```python
def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
    assert self._driver is not None
    validate_identifier(label, kind="label")
    key = primary_key_for(label)
    if key not in properties:
        raise ValueError(
            f"upsert_node({label!r}, ...) missing required primary key "
            f"{key!r}; properties were {sorted(properties)}"
        )
    cypher = (
        f"MERGE (n:{label} {{{key}: $key_value}}) "
        f"SET n += $props "
        f"RETURN n.{key} AS id"
    )
    with self._driver.session() as session:
        result = session.run(
            cypher, key_value=properties[key], props=properties
        )
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
    assert self._driver is not None
    validate_identifier(edge_type, kind="edge_type")
    props = {**(properties or {}), "origin": origin}

    # src_label / dst_label are advisory on Neo4j (schema-free). When provided
    # they narrow the MATCH. When omitted, we MATCH any label with the
    # corresponding PK — slower but correct. Primary key lookup uses the
    # canonical label→PK map.
    src_key = primary_key_for(src_label) if src_label else "path"
    dst_key = primary_key_for(dst_label) if dst_label else "path"
    src_pattern = f"(a:{src_label})" if src_label else "(a)"
    dst_pattern = f"(b:{dst_label})" if dst_label else "(b)"

    cypher = (
        f"MATCH {src_pattern} WHERE a.{src_key} = $src "
        f"MATCH {dst_pattern} WHERE b.{dst_key} = $dst "
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
    assert self._driver is not None
    if edge_type is not None:
        validate_identifier(edge_type, kind="edge_type")
    src_key = primary_key_for(src_label) if src_label else "path"
    src_pattern = f"(a:{src_label})" if src_label else "(a)"
    conditions: list[str] = []
    params: dict[str, Any] = {"src": src_id}
    if origin is not None:
        conditions.append("r.origin = $origin")
        params["origin"] = origin
    where_clause = f"WHERE {' AND '.join(conditions)} " if conditions else ""
    edge_fragment = f":{edge_type}" if edge_type else ""
    cypher = (
        f"MATCH {src_pattern} WHERE a.{src_key} = $src "
        f"WITH a MATCH (a)-[r{edge_fragment}]->() {where_clause}DELETE r"
    )
    with self._driver.session() as session:
        session.run(cypher, **params)


def exec_read(
    self, cypher: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    assert self._driver is not None
    with self._driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(record) for record in result]


def exec_write(self, cypher: str, params: dict[str, Any] | None = None) -> None:
    assert self._driver is not None
    with self._driver.session() as session:
        session.run(cypher, params or {})
```

- [ ] **Step 4: Run integration tests**

```bash
pytest tests/integration/test_neo4j_backend.py -v
```

Expected: 4 passed.

- [ ] **Step 5: CI triad (unit + types + format)**

```bash
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -v
```

Expected: all green. Unit suite still 253 (no new unit tests added in this task).

- [ ] **Step 6: Commit**

```bash
git add contextd/storage/neo4j.py tests/integration/test_neo4j_backend.py
git commit -m "$(cat <<'EOF'
feat(storage): Neo4jBackend upserts + delete_edges + exec (M11.3)

MERGE-based upsert_node / upsert_edge; scoped delete_edges that rejects
unfiltered deletion to preserve the wipe-and-replace invariant;
session-per-call exec_read and exec_write. 4 integration tests pass
against neo4j:5.15-community via testcontainers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11.4: Implement vector_search + full_text_search

**Files:**
- Modify: `contextd/storage/neo4j.py`
- Modify: `tests/integration/test_neo4j_backend.py`

- [ ] **Step 1: Write failing integration tests**

Append to `tests/integration/test_neo4j_backend.py`:

```python
def test_vector_search_roundtrip(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    neo4j_backend.upsert_node(
        "File", {"path": "/a.md", "corpus": "t", "embedding": [1.0] + [0.0] * 1023}
    )
    neo4j_backend.upsert_node(
        "File", {"path": "/b.md", "corpus": "t", "embedding": [0.0, 1.0] + [0.0] * 1022}
    )
    neo4j_backend.upsert_node(
        "File", {"path": "/c.md", "corpus": "t", "embedding": [1.0] + [0.0] * 1023}
    )

    results = neo4j_backend.vector_search(
        "File", "embedding", query=[1.0] + [0.0] * 1023, k=3
    )
    # /a.md and /c.md are identical to query; /b.md is orthogonal.
    paths = [r["node"]["path"] for r in results]
    scores = [r["score"] for r in results]
    assert paths[0] in {"/a.md", "/c.md"}
    # First two scores should be ~1.0 (identical direction).
    assert scores[0] > 0.99
    # Orthogonal vector should score ~0.5 (cosine 0.0 → similarity 0.5 after
    # Neo4j's [0,1] normalisation).
    assert "/b.md" in paths


def test_vector_search_threshold_filter(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    neo4j_backend.upsert_node(
        "File", {"path": "/a.md", "corpus": "t", "embedding": [1.0] + [0.0] * 1023}
    )
    neo4j_backend.upsert_node(
        "File", {"path": "/b.md", "corpus": "t", "embedding": [0.0, 1.0] + [0.0] * 1022}
    )
    results = neo4j_backend.vector_search(
        "File", "embedding",
        query=[1.0] + [0.0] * 1023,
        k=10,
        threshold=0.9,
    )
    paths = [r["node"]["path"] for r in results]
    assert "/a.md" in paths
    assert "/b.md" not in paths


def test_full_text_search_roundtrip(neo4j_backend) -> None:
    from contextd.migrations.neo4j._0001_baseline import migration

    neo4j_backend.apply_migrations([migration])

    neo4j_backend.upsert_node(
        "File", {"path": "/a.md", "corpus": "t", "summary": "alpha beta gamma"}
    )
    neo4j_backend.upsert_node(
        "File", {"path": "/b.md", "corpus": "t", "summary": "delta epsilon"}
    )
    results = neo4j_backend.full_text_search("File", "summary", "alpha", k=5)
    assert len(results) == 1
    assert results[0]["node"]["path"] == "/a.md"
    assert results[0]["score"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/integration/test_neo4j_backend.py -v
```

Expected: 3 new tests fail with `NotImplementedError: Task 11.4`. Previous 4 still pass.

- [ ] **Step 3: Implement vector_search + full_text_search**

Replace the NotImplementedError bodies in `contextd/storage/neo4j.py`:

```python
def vector_search(
    self,
    label: str,
    property_name: str,
    query: list[float],
    k: int,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Call Neo4j's db.index.vector.queryNodes procedure.

    Neo4j returns (node, score) where score is cosine similarity in [0, 1]
    (higher is more similar). Matches the ABC's contract directly — no
    distance-to-similarity conversion needed (unlike Kuzu did).
    """
    assert self._driver is not None
    validate_identifier(label, kind="label")
    validate_identifier(property_name, kind="property_name")
    validate_search_k(k)
    validated_threshold = validate_threshold(threshold)
    index_name = f"{label}_{property_name}_idx"
    cypher = (
        f"CALL db.index.vector.queryNodes($idx, $k, $q) "
        f"YIELD node, score "
        f"RETURN node, score "
        f"ORDER BY score DESC"
    )
    with self._driver.session() as session:
        result = session.run(cypher, idx=index_name, k=k, q=query)
        rows = [
            {"node": dict(r["node"]), "score": float(r["score"])}
            for r in result
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
) -> list[dict[str, Any]]:
    assert self._driver is not None
    validate_identifier(label, kind="label")
    validate_identifier(property_name, kind="property_name")
    validate_search_k(k)
    index_name = f"{label}_{property_name}_ft"
    cypher = (
        f"CALL db.index.fulltext.queryNodes($idx, $q) "
        f"YIELD node, score "
        f"RETURN node, score "
        f"ORDER BY score DESC "
        f"LIMIT {k}"
    )
    with self._driver.session() as session:
        result = session.run(cypher, idx=index_name, q=query)
        return [
            {"node": dict(r["node"]), "score": float(r["score"])}
            for r in result
        ]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/integration/test_neo4j_backend.py -v
```

Expected: 7 passed.

- [ ] **Step 5: CI triad**

```bash
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -v
```

All green.

- [ ] **Step 6: Commit**

```bash
git add contextd/storage/neo4j.py tests/integration/test_neo4j_backend.py
git commit -m "$(cat <<'EOF'
feat(storage): Neo4jBackend vector + full-text search (M11.4)

db.index.vector.queryNodes + db.index.fulltext.queryNodes, both
returning {node, score} per the ABC contract. Neo4j natively returns
cosine similarity — no distance conversion (simpler than the Kuzu
implementation it replaces). Threshold filter applied client-side.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11.5: Add Neo4jConfig + wire factory

**Files:**
- Modify: `contextd/config.py` (add `Neo4jConfig`; update `StorageConfig`)
- Modify: `contextd/storage/factory.py` (add `"neo4j"` branch)
- Modify: `tests/unit/test_neo4j_backend.py` (remove shim)
- Create: `tests/unit/test_factory_neo4j.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_factory_neo4j.py`:

```python
"""Factory wiring for Neo4j backend."""
from __future__ import annotations

from contextd.config import Config, Neo4jConfig, StorageConfig


def test_factory_returns_neo4j_when_backend_is_neo4j() -> None:
    from contextd.storage.factory import build_graph_store
    from contextd.storage.neo4j import Neo4jBackend

    cfg = Config(storage=StorageConfig(backend="neo4j", neo4j=Neo4jConfig()))
    store = build_graph_store(cfg)
    assert isinstance(store, Neo4jBackend)


def test_neo4j_config_defaults() -> None:
    cfg = Neo4jConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 7687
    assert cfg.user == "neo4j"
    assert cfg.password == "neo4j"
```

Also update `tests/unit/test_neo4j_backend.py` — remove the local `Neo4jConfig` shim block and restore:

```python
from contextd.config import Neo4jConfig
```

- [ ] **Step 2: Verify the tests fail**

```bash
pytest tests/unit/test_factory_neo4j.py tests/unit/test_neo4j_backend.py -v
```

Expected: `ImportError: cannot import name 'Neo4jConfig' from 'contextd.config'`.

- [ ] **Step 3: Add Neo4jConfig + wire factory**

Edit `contextd/config.py`. Add after the `KuzuConfig` class (around line 64):

```python
class Neo4jConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = 7687
    user: str = "neo4j"
    password: str = "neo4j"
    docker_compose_file: str = "~/.contextd/docker-compose.yml"
    memory_limit_gb: float = 1.0
    cpu_limit: float = 1.0
```

Update `StorageConfig` (around line 67-71) to include `neo4j`:

```python
class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: BackendName = "memgraph"
    memgraph: MemgraphConfig = Field(default_factory=MemgraphConfig)
    kuzu: KuzuConfig = Field(default_factory=KuzuConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
```

(BackendName already includes `"neo4j"` from Task 11.1.)

Edit `contextd/storage/factory.py` and add the neo4j branch. Locate the existing `if cfg.storage.backend == "memgraph":` block and add:

```python
def build_graph_store(cfg: Config) -> GraphStore:
    backend = cfg.storage.backend
    if backend == "memgraph":
        from contextd.storage.memgraph import MemgraphBackend
        return MemgraphBackend(cfg.storage.memgraph)
    if backend == "kuzu":
        from contextd.storage.kuzu import KuzuBackend
        return KuzuBackend(cfg.storage.kuzu)
    if backend == "neo4j":
        from contextd.storage.neo4j import Neo4jBackend
        return Neo4jBackend(cfg.storage.neo4j)
    raise ValueError(f"unknown backend {backend!r}")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_factory_neo4j.py tests/unit/test_neo4j_backend.py -v
```

Expected: 5 passed (3 existing in test_neo4j_backend + 2 new in test_factory_neo4j).

- [ ] **Step 5: CI triad**

```bash
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -v
```

Unit suite: 255 (+2 from the factory tests).

- [ ] **Step 6: Commit**

```bash
git add contextd/config.py contextd/storage/factory.py \
  tests/unit/test_factory_neo4j.py tests/unit/test_neo4j_backend.py
git commit -m "$(cat <<'EOF'
feat(storage): wire Neo4jConfig + factory branch (M11.5)

Add Neo4jConfig (host/port/user/password/docker settings) to the config
model. Wire the "neo4j" factory branch behind a deferred import, matching
the memgraph/kuzu pattern. Factory still supports all three backends
during the transition; kuzu removal lands in M11.9.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11.6: Docker-compose template + CLI up/down/status support

**Files:**
- Modify: `contextd/docker_compose.yml`
- Modify: `contextd/cli/infra.py`
- Modify: `tests/unit/test_cli_lifecycle.py`

- [ ] **Step 1: Inspect the current docker_compose.yml**

```bash
cat contextd/docker_compose.yml
```

Current shape expected (from SD #25/#26 context):

```yaml
services:
  memgraph:
    image: memgraph/memgraph:latest
    ports:
      - "7687:7687"
    volumes:
      - contextd_memgraph_data:/var/lib/memgraph
    restart: unless-stopped
volumes:
  contextd_memgraph_data:
```

- [ ] **Step 2: Add the neo4j service**

Replace `contextd/docker_compose.yml` with:

```yaml
# Docker Compose template deployed by `contextd init` to ~/.contextd/docker-compose.yml.
# Both services declared; `contextd up` starts only the service matching
# [storage] backend in config.toml via `docker compose up -d <service>`.
services:
  memgraph:
    image: memgraph/memgraph:latest
    profiles: ["memgraph"]
    ports:
      - "7687:7687"
    volumes:
      - contextd_memgraph_data:/var/lib/memgraph
    restart: unless-stopped

  neo4j:
    image: neo4j:5.15-community
    profiles: ["neo4j"]
    ports:
      - "7687:7687"
      - "7474:7474"
    environment:
      NEO4J_AUTH: "neo4j/contextd"
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - contextd_neo4j_data:/data
      - contextd_neo4j_logs:/logs
    restart: unless-stopped

volumes:
  contextd_memgraph_data:
  contextd_neo4j_data:
  contextd_neo4j_logs:
```

Key points:
- Both services bound to port 7687 — only one can run at a time. User choice enforced by compose profile + `contextd up` dispatching to the right service.
- Neo4j auth is hardcoded to `neo4j/contextd` in the template; the config file's `Neo4jConfig.password` defaults match. Users who change one must change the other.
- `apoc` plugin included for utility procedures (not load-bearing today, standard kit).

- [ ] **Step 3: Write failing CLI lifecycle test**

Add to `tests/unit/test_cli_lifecycle.py`:

```python
def test_up_neo4j_calls_compose_with_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextd.cli
    from click.testing import CliRunner

    home = tmp_path / ".contextd"
    home.mkdir()
    (home / "config.toml").write_text(
        "[storage]\nbackend = \"neo4j\"\n\n"
        "[storage.neo4j]\nhost = \"127.0.0.1\"\nport = 7687\n"
    )
    # The docker-compose template must exist for `up` to run.
    (home / "docker-compose.yml").write_text("services: {}\n")
    monkeypatch.setenv("CONTEXTD_HOME", str(home))

    calls: list[list[str]] = []
    def fake_run(*args, **kwargs):
        calls.append(list(args[0]))
        class R:
            returncode = 0
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")

    result = CliRunner().invoke(contextd.cli.cli, ["up"])
    assert result.exit_code == 0
    # At least one docker compose call with --profile neo4j.
    compose_calls = [c for c in calls if "compose" in c]
    assert any("--profile" in c and "neo4j" in c for c in compose_calls)
```

- [ ] **Step 4: Run the test to confirm it fails**

```bash
pytest tests/unit/test_cli_lifecycle.py::test_up_neo4j_calls_compose_with_profile -v
```

Expected: FAIL (current `up` doesn't dispatch on profile).

- [ ] **Step 5: Update `contextd/cli/infra.py`**

Find the `up` command. Current shape (simplified):

```python
@cli.command()
def up() -> None:
    cfg = _load_cfg()
    if cfg.storage.backend == "memgraph":
        subprocess.run(["docker", "compose", "-f", str(compose_file), "up", "-d"], check=False)
    elif cfg.storage.backend == "kuzu":
        # kuzu path — no container
        ...
```

Update to pass the `--profile <backend>` flag to compose so only the selected service starts:

```python
@cli.command()
def up() -> None:
    """Start the configured backend."""
    cfg = _load_cfg()
    backend = cfg.storage.backend
    if backend == "kuzu":
        # Embedded backend — no container to start.
        console.print("[green]✓[/] kuzu backend is embedded; no container needed")
        return
    if shutil.which("docker") is None:
        raise click.ClickException(
            "docker not found on PATH; install Docker Desktop or docker-ce"
        )
    compose_file = contextd_home() / "docker-compose.yml"
    if not compose_file.exists():
        raise click.ClickException(
            f"{compose_file} missing — run `contextd init`"
        )
    cmd = [
        "docker", "compose",
        "-f", str(compose_file),
        "--profile", backend,
        "up", "-d",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise click.ClickException(f"docker compose up failed (exit {result.returncode})")
    console.print(f"[green]✓[/] {backend} started")
```

Apply the same `--profile <backend>` pattern to `down` and `status`.

- [ ] **Step 6: Run tests**

```bash
pytest tests/unit/test_cli_lifecycle.py -v
```

Expected: all lifecycle tests pass (existing + new neo4j one).

- [ ] **Step 7: CI triad**

```bash
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -v
```

All green.

- [ ] **Step 8: Commit**

```bash
git add contextd/docker_compose.yml contextd/cli/infra.py tests/unit/test_cli_lifecycle.py
git commit -m "$(cat <<'EOF'
feat(cli): docker-compose profiles + neo4j up/down/status (M11.6)

Docker Compose template gains profiled memgraph + neo4j services; only
the profile matching [storage] backend starts. CLI `up`/`down`/`status`
pass `--profile <backend>` so the right container is targeted. Kuzu
(embedded, no container) handled with an explicit no-op message.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11.7: Parametrise integration tests on Neo4j

**Files:**
- Modify: `tests/integration/conftest.py` (add neo4j to the `backend` parametrisation)
- Verify: all existing integration tests (`test_upsert_roundtrip.py`, `test_pipeline.py`, `test_mcp_tools.py`, `test_search.py`) pass on neo4j

- [ ] **Step 1: Read the current conftest**

```bash
cat tests/integration/conftest.py
```

Current shape expected:

```python
import pytest

@pytest.fixture(params=["memgraph", "kuzu"])
def backend(request, ...):
    # yields a connected GraphStore
    ...
```

- [ ] **Step 2: Add neo4j to the parametrisation**

Update `tests/integration/conftest.py` to add a `"neo4j"` branch that spins up a Neo4jContainer:

```python
import pytest
from testcontainers.memgraph import MemgraphContainer
from testcontainers.neo4j import Neo4jContainer


@pytest.fixture(params=["memgraph", "kuzu", "neo4j"])
def backend(request, tmp_path):
    if request.param == "memgraph":
        from contextd.config import MemgraphConfig
        from contextd.storage.memgraph import MemgraphBackend
        from contextd.migrations.memgraph._0001_baseline import migration as baseline
        from contextd.migrations.memgraph._0002_corpus_stats import migration as stats

        with MemgraphContainer("memgraph/memgraph:latest") as container:
            cfg = MemgraphConfig(
                host=container.get_container_host_ip(),
                port=int(container.get_exposed_port(7687)),
            )
            backend = MemgraphBackend(cfg)
            backend.connect()
            backend.apply_migrations([baseline, stats])
            yield backend
            backend.close()

    elif request.param == "kuzu":
        from contextd.config import KuzuConfig
        from contextd.storage.kuzu import KuzuBackend
        from contextd.migrations.kuzu._0001_baseline import migration as baseline
        from contextd.migrations.kuzu._0002_corpus_stats import migration as stats

        cfg = KuzuConfig(db_path=str(tmp_path / "kuzu.db"))
        backend = KuzuBackend(cfg)
        backend.connect()
        backend.apply_migrations([baseline, stats])
        yield backend
        backend.close()

    elif request.param == "neo4j":
        from contextd.config import Neo4jConfig
        from contextd.storage.neo4j import Neo4jBackend
        from contextd.migrations.neo4j._0001_baseline import migration as baseline

        with Neo4jContainer("neo4j:5.15-community") as container:
            cfg = Neo4jConfig(
                host=container.get_container_host_ip(),
                port=int(container.get_exposed_port(7687)),
                user="neo4j",
                password=container.password,
            )
            backend = Neo4jBackend(cfg)
            backend.connect()
            backend.apply_migrations([baseline])
            yield backend
            backend.close()
```

- [ ] **Step 3: Run the full integration suite against all three backends**

```bash
pytest tests/integration -v
```

Expected (triple parametrisation):
- 65 existing tests × 3 backends = ~195 test runs
- Some may skip (e.g., Kuzu-specific guards that were in place)
- Everything else passes

**If a Neo4j-specific failure surfaces**, the most likely causes:
- **`upsert_edge` with missing src_label on Neo4j** — the Neo4j backend's `upsert_edge` handles `src_label=None` gracefully (advisory); verify.
- **Case sensitivity in Cypher keywords** — Neo4j is stricter about reserved words than Memgraph in certain contexts. If a test's raw Cypher trips on this, either update the Cypher or skip the test with a `@pytest.mark.skipif(backend == "neo4j", reason="...")`.
- **Index-name collisions** — Memgraph and Neo4j share port 7687 in the compose template; testcontainers avoids this by binding to ephemeral ports, so no issue in CI.

Fix each Neo4j failure by either:
1. Updating the test's Cypher to portable syntax (preferred)
2. Marking it as Memgraph-only with a justification comment
3. Updating the Neo4jBackend if the fix is implementation-side

Escalate with `BLOCKED` status if a test failure points to a design-level mismatch (e.g., Neo4j can't express something the ABC contract requires).

- [ ] **Step 4: CI triad**

```bash
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -v
```

All green.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/conftest.py tests/integration/test_*.py
git commit -m "$(cat <<'EOF'
test: parametrise integration suite on neo4j + memgraph + kuzu (M11.7)

Triple parametrisation proves the Neo4j backend implements the full
ABC contract. Kuzu leg still runs — removal lands in M11.9. Any
test-specific skips or adjustments noted inline; the shared
expectations stay in the parametrised body.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11.8: Flip default backend to Neo4j

**Files:**
- Modify: `contextd/config.py` (`StorageConfig.backend` default)
- Modify: `contextd/_defaults/config.toml` (if such a template file exists — check via `contextd/cli/__init__.py::init`)
- Modify: `tests/unit/test_config.py` (update default-backend assertion if present)
- Modify: `tests/unit/test_cli_init.py` (verify `contextd init` writes neo4j as default)

- [ ] **Step 1: Locate the default-backend source of truth**

Run:

```bash
grep -rn 'backend.*=.*"memgraph"' contextd/
```

Two places expected:
1. `contextd/config.py` in `StorageConfig`: `backend: BackendName = "memgraph"`
2. The packaged template written by `contextd init` (could be inlined in `cli/__init__.py`'s `init` function, or a separate `.toml` resource file — verify).

- [ ] **Step 2: Write failing test**

Update `tests/unit/test_cli_init.py::test_init_creates_layout` or add a new test:

```python
def test_init_writes_neo4j_default_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextd.cli
    from click.testing import CliRunner

    home = tmp_path / ".contextd"
    monkeypatch.setenv("CONTEXTD_HOME", str(home))
    result = CliRunner().invoke(contextd.cli.cli, ["init"])
    assert result.exit_code == 0
    config = (home / "config.toml").read_text()
    assert 'backend = "neo4j"' in config
```

Also update `tests/unit/test_config.py` to assert `StorageConfig().backend == "neo4j"`.

- [ ] **Step 3: Flip the defaults**

In `contextd/config.py`:

```python
class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: BackendName = "neo4j"  # was "memgraph"
    memgraph: MemgraphConfig = Field(default_factory=MemgraphConfig)
    kuzu: KuzuConfig = Field(default_factory=KuzuConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
```

In the `init` command (likely `contextd/cli/__init__.py`), wherever the default `config.toml` is written, ensure the default backend is `neo4j`. If the template is inlined, update the string; if it's a resource file, update the file.

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_cli_init.py tests/unit/test_config.py -v
```

Expected: pass.

- [ ] **Step 5: CI triad**

```bash
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -v
```

All green.

- [ ] **Step 6: Commit**

```bash
git add contextd/config.py contextd/cli tests/unit/test_cli_init.py \
  tests/unit/test_config.py
git commit -m "$(cat <<'EOF'
feat(config): Neo4j becomes the default backend (M11.8)

[storage] backend defaults to "neo4j" in fresh installs. Existing
installs are unaffected — their config.toml already pins the chosen
backend. Memgraph remains a first-class alternative.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11.9: Excise Kuzu

**Files (deletions):**
- `contextd/storage/kuzu.py`
- `contextd/migrations/kuzu/` (directory)
- `tests/integration/test_kuzu_backend.py`

**Files (modifications):**
- `contextd/config.py` — remove `KuzuConfig`, remove `kuzu` field on `StorageConfig`, update `BackendName` Literal.
- `contextd/storage/factory.py` — remove `"kuzu"` branch.
- `contextd/storage/base.py` — prune Kuzu-specific docstrings; update `BackendName` Literal.
- `contextd/storage/_keys.py` — delete `IMMUTABLE_AFTER_CREATE_BY_LABEL` map + `immutable_after_create_for` function.
- `contextd/storage/memgraph.py` — remove any `immutable_after_create_for` import if present.
- `tests/integration/conftest.py` — remove `"kuzu"` from the `backend` fixture parametrisation.
- `pyproject.toml` — remove `kuzu` from `[project].dependencies`.
- `contextd/docker_compose.yml` — no change needed (no kuzu service existed).
- `contextd/cli/infra.py` — remove the `if backend == "kuzu":` early-return branch.

- [ ] **Step 1: Grep for every kuzu reference to make the excision scope concrete**

```bash
grep -rn -i "kuzu" contextd/ tests/ pyproject.toml | grep -v __pycache__
```

Expected hits: the files listed above. Note every line.

- [ ] **Step 2: Delete the Kuzu-specific files**

```bash
rm contextd/storage/kuzu.py
rm -r contextd/migrations/kuzu/
rm tests/integration/test_kuzu_backend.py
```

- [ ] **Step 3: Strip Kuzu from config.py**

Edit `contextd/config.py`:
- Delete the `class KuzuConfig(BaseModel)` block (~7 lines).
- Remove `kuzu: KuzuConfig = Field(default_factory=KuzuConfig)` from `StorageConfig`.
- Update `BackendName` Literal in `contextd/storage/base.py:10`: `Literal["memgraph", "neo4j"]`.

- [ ] **Step 4: Strip Kuzu from factory.py**

Remove the `if backend == "kuzu":` block. The factory now has only memgraph + neo4j branches + the `ValueError` fallback.

- [ ] **Step 5: Strip Kuzu-specific infrastructure from `_keys.py`**

`contextd/storage/_keys.py` — delete:

```python
IMMUTABLE_AFTER_CREATE_BY_LABEL: Final[dict[str, frozenset[str]]] = { ... }

def immutable_after_create_for(label: str) -> frozenset[str]:
    return IMMUTABLE_AFTER_CREATE_BY_LABEL.get(label, frozenset())
```

Keep `PRIMARY_KEY_BY_LABEL` + `primary_key_for` — both still used by Memgraph and Neo4j.

- [ ] **Step 6: Strip Kuzu-specific docstrings from base.py**

`contextd/storage/base.py` — `upsert_edge` docstring currently mentions "required on schema-first backends (Kuzu — REL tables declare fixed FROM/TO label pairs...)". Rewrite:

```
``src_label`` / ``dst_label`` are the endpoint *node* labels; advisory
on schema-free backends (Memgraph, Neo4j — used to narrow the MATCH).
Retained on the ABC for future schema-first backends that would
require them.
```

Same adjustment for `delete_edges` docstring.

- [ ] **Step 7: Strip Kuzu from conftest.py**

`tests/integration/conftest.py` — remove the `"kuzu"` param entry and its entire `elif request.param == "kuzu":` block from the `backend` fixture.

- [ ] **Step 8: Strip Kuzu from phases.py**

`contextd/indexer/phases.py` — search for references to `immutable_after_create` or `IMMUTABLE_AFTER_CREATE`. None should remain in production code (the helper was deleted in Step 5, so any remaining import would be a red flag).

Run:

```bash
grep -rn "IMMUTABLE_AFTER_CREATE\|immutable_after_create" contextd/ tests/
```

Expected: zero hits. If any, remove them.

- [ ] **Step 9: Strip Kuzu from pyproject.toml**

Remove the `"kuzu"` (or `"kuzu>=0.10"`) line from `[project].dependencies`. Run:

```bash
uv pip sync pyproject.toml  # or uv pip install -e ".[dev]"
```

Confirm the `kuzu` package is uninstalled:

```bash
pip show kuzu 2>&1 | head -1
```

Expected: `WARNING: Package(s) not found: kuzu`.

- [ ] **Step 10: Verify nothing imports the deleted symbols**

```bash
grep -rn "from contextd.storage.kuzu\|import kuzu\|KuzuBackend\|KuzuConfig" contextd/ tests/
```

Expected: zero hits.

- [ ] **Step 11: Run the CI triad**

```bash
source .venv/bin/activate
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -v
```

Expected: all green. Unit count = 252 or similar (lost 3 Kuzu-specific unit tests if any existed; check).

Also run abstraction-invariant grep:

```bash
# grep command from .github/workflows/ci.yml's abstraction-invariant job
```

Expected: zero hits. With Kuzu gone, the check becomes simpler.

Run the integration suite to confirm the double parametrisation works:

```bash
pytest tests/integration -v
```

Expected: ~130 test runs (65 × 2 backends). All pass.

- [ ] **Step 12: Commit the excision**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(storage): excise Kuzu — archived upstream, superseded by Neo4j (M11.9)

Delete contextd/storage/kuzu.py, contextd/migrations/kuzu/, KuzuConfig,
IMMUTABLE_AFTER_CREATE_BY_LABEL, the kuzu factory branch, the kuzu
integration-test parametrisation, and the kuzu Python-package dependency.
BackendName Literal is now {memgraph, neo4j}.

The Kuzu project was archived upstream in 2026. SD #71 documented the
permanent vector-index mutation bug that motivated the replacement;
Neo4j (M11.1-8) carries the embedded-backend slot's responsibilities.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11.10: Documentation propagation

**Files:**
- Modify: `README.md`
- Modify: `docs/design.md` (§2.5 + §12.6 + §13.4)
- Modify: `docs/implementation-plan.md` (archive M3 Kuzu tasks; add M11 section pointer)
- Modify: `CLAUDE.md` (multiple sections)

- [ ] **Step 1: Update CLAUDE.md**

Search-and-replace targets:

- Any narrative line citing "Memgraph or KùzuDB" → "Memgraph or Neo4j".
- Any list of "backends: memgraph, kuzu" → "backends: memgraph, neo4j".
- **Remove** the SD #71 entry from "Permanent limitations (upstream, no fix available)" — Kuzu is gone, so the limitation no longer applies. Replace that subsection's intro with a note that the section is currently empty:

```markdown
### Permanent limitations (upstream, no fix available)

_No permanent upstream limitations at this time._ (SD #71 — Kuzu
embedding mutability — was resolved by the M11 Kuzu-to-Neo4j migration
in 2026-04-21; both Memgraph and Neo4j support mutable embedding
columns.)
```

- Update the "As of YYYY-MM-DD" narrative at the top of CLAUDE.md to reflect the new milestone state: M11 complete (or close), current backends are memgraph + neo4j.
- Update the "Test suite" tally if counts changed.
- Update the "Memgraph / Docker" section to also cover Neo4j image + port notes.
- Update the "Tech Stack" section: replace `kuzu` with `neo4j` in the storage line.

- [ ] **Step 2: Update docs/design.md**

§2.5 "Storage-backend abstraction" — rewrite. Key replacements:
- "Memgraph (default) or KùzuDB (embedded)" → "Memgraph or Neo4j (both run in Docker; either can be default)"
- Remove §2.5.3 dialect discussion of Kuzu-specific UPSERT quirks and IMMUTABLE_AFTER_CREATE — this is no longer needed. Replace with a note that both backends support standard MERGE semantics.
- Update §2.5.4 "Choosing a backend": Neo4j is the reference Cypher implementation (broader LLM-generated-query compatibility, deeper ecosystem); Memgraph is the performance-on-RAM alternative.

§12.6 "Deployment topologies" — both backends now require Docker. The "Path A — KùzuDB backend, no Docker at all" option is gone. Rewrite the WSL2 sub-section to list Memgraph + Neo4j options via Docker Desktop.

§13.4 "Recommended backend for the Runeledger dev environment" — currently recommends Kuzu. Rewrite to recommend Neo4j (or note either is fine; Neo4j preferred for the Cypher-compatibility reasons above).

- [ ] **Step 3: Update docs/implementation-plan.md**

Add an "M11" section at the top referencing this plan file. Add an archival note to M3 (the original Kuzu-specific tasks):

```markdown
**M3 historical note:** Tasks 3.1–3.4 originally implemented the Kuzu
backend. Kuzu was excised in M11 (2026-04-21) after the project was
archived upstream. The Kuzu-specific Tasks are preserved here for git
history lookup but are no longer part of the live architecture. See
`docs/superpowers/plans/2026-04-21-M11-neo4j-backend-kuzu-excision.md`.
```

- [ ] **Step 4: Update README.md**

- Quickstart: change any `backend = "kuzu"` examples to `backend = "neo4j"`.
- Install section: Docker is required for both backends; reword accordingly.
- Architecture diagram (if any): replace Kuzu with Neo4j.

- [ ] **Step 5: Verify no stale Kuzu references remain in docs**

```bash
grep -rni "kuzu\|kùzu" README.md docs/ CLAUDE.md
```

Expected: only occurrences in **historical context** (e.g., "originally implemented Kuzu", the removed-SD-#71 note). No live-architecture references. If any current-state references remain, fix them.

- [ ] **Step 6: CI triad (docs don't affect tests, but CI lint catches trailing-whitespace etc.)**

```bash
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -v
```

All green.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/design.md docs/implementation-plan.md CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: propagate M11 — Neo4j replaces Kuzu throughout the narrative

Update README, design.md (§2.5, §12.6, §13.4), implementation-plan.md
(M11 pointer + M3 archival note), and CLAUDE.md. Remove SD #71 from
Permanent limitations (resolved by M11). Runeledger recommended backend
is now Neo4j.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance Criteria (M11 as a whole)

Before marking M11 complete:

1. `git grep -i 'kuzu\|kùzu'` returns only **historical-context hits** (git log commit messages, the M3 archival note, the SD #71 resolved note). Zero live-architecture references in `contextd/`, `tests/`, `pyproject.toml`, `README.md`, `docs/design.md`.
2. Full CI green on the double parametrisation: `pytest tests/integration -v` passes on memgraph + neo4j (~130 test runs).
3. `contextd init` on a fresh home writes `backend = "neo4j"`.
4. `contextd up` starts the neo4j container; `contextd down` stops it; `contextd status` reports reachable.
5. `contextd add-corpus <path> && contextd index <name> --bootstrap` completes on Neo4j against a fixture corpus.
6. `contextd ask "list all files" --corpus <name>` returns results (exercises the SD #72 corpus-filter injection against the new backend).
7. `pyproject.toml` no longer depends on `kuzu`; `pip show kuzu` returns "not found".
8. CLAUDE.md's `Permanent limitations` subsection no longer lists SD #71.

## Risk Register

- **Neo4j Cypher dialect edge-cases.** Likely discovery during Task 11.7 parametrisation. Mitigation: keep tests portable where possible; document backend-specific skips with justifications.
- **testcontainers-python `Neo4jContainer` stability in CI.** First-time use in the project. If flaky, fallback is `services:` in `.github/workflows/ci.yml` (GitHub Actions-native Neo4j service container).
- **Default-auth password hardcoding.** `neo4j/contextd` in the template is fine for local-only single-user. Document that users must change both the compose env var AND `Neo4jConfig.password` in `config.toml` in lock-step.
- **Port collision.** Both backends bind 7687 locally. Compose profiles ensure only one runs at a time; document this.
- **Integration-test Docker load.** Running Neo4jContainer + MemgraphContainer in parametrisation may be slow. Acceptable tradeoff; CI fixture ordering reuses containers across tests within a parametrisation.

## Out of Scope (explicitly)

- Adding HTTP/SSE transport to the MCP server (spec §7.1 deferred).
- Implementing `contextd export <corpus> --format cypher-dump` for backend migration (spec §2.5.4). A separate later milestone.
- Neo4j multi-database support (enterprise-only; Community is single-db; contextd stores all corpora in one db with `corpus` property, no change).
- Runtime backend swap without restart (not a goal; `contextd down && up` cycle remains the documented procedure).
