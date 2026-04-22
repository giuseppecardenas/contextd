# Architecture

Contextd decomposes into three cooperating layers: **indexing pipeline**, **storage**, and **inference/MCP surface**. Each runs in its own process and interacts with the others through defined interfaces.

---

## Three-layer decomposition

### Layer 1 — Indexing pipeline

The indexer walks a corpus's file tree, hashes each file, generates embeddings and summaries via external AI providers, infers typed relationships between nodes, and upserts everything into the graph store.

Entry point: `contextd/indexer/pipeline.py`. The public functions are:

- `enumerate_corpus_files(corpus)` — expands include globs, drops `.git`/`.venv`/`__pycache__`/`node_modules` components and symlinks, returns a `list[Path]`.
- `run_bootstrap(corpus, store, embedder, summariser, inferrer, hasher, entity_sampler)` — runs the full multi-phase bootstrap sequence.

**File-granular bootstrap phases** (when `corpus.granularity = "file"`):

1. `phase_enumerate` — create `File` nodes, compute embeddings, record MD5 hashes.
2. `phase_embed` — accounting stub (embedding done at CREATE in phase 1).
3. `phase_summarise` — summarise each file; write `summary`, `key_points`, `confidence`.
4. `phase_relate` — infer typed edges between `File` nodes; wipe-and-replace `origin="inferred"` edges only.
5. `phase_close` — register the corpus as a `Corpus` node; persist stats.

**Section-granular bootstrap phases** (when `corpus.granularity = "section"`):

1. `phase_enumerate_sections` — parse headings; create `File` + `Section` nodes with `CONTAINS`/`PARENT_OF`/`NEXT_SIBLING` structural edges; compute section embeddings at CREATE time.
2. `phase_embed_sections` — accounting stub.
3. `phase_summarise_sections` — summarise each section; write `summary`, `key_points`.
4. `phase_relate_sections` — infer typed edges from `Section` nodes; wipe-and-replace `inferred` edges only.
5. `phase_gc_sections` — DETACH-DELETE stale `Section` nodes absent from the current parse output.
6. `phase_derive_file_level` — derive `File.summary` by rolling up child-section summaries.
7. `phase_close` — register the corpus; persist stats.

Markdown files are routed through the section-level pipeline; non-markdown files (e.g. `.lua`) are routed through the file-granular pipeline within the same bootstrap run.

**Incremental re-indexing** is not yet implemented. The `--incremental` flag is wired in the CLI but reports "not yet implemented in this build."

**Indexer daemon** — `contextd up` currently starts the storage container and applies migrations. A long-running file-watcher daemon (`CorpusWatcher` + `DebouncedQueue`, built in M5) is implemented in `contextd/indexer/watcher.py` and `contextd/indexer/debounced_queue.py` but is not yet wired into `contextd up`. The bootstrap is invoked manually via `contextd index <name> --bootstrap`.

---

### Layer 2 — Storage

The storage layer is a pluggable graph + vector store selected by `[storage] backend` in `~/.contextd/config.toml`. Two backends ship: **Neo4j Community** (default) and **Memgraph** — both run in Docker, both bind port 7687 over the Bolt protocol, and only one runs at a time via docker-compose profiles.

#### GraphStore ABC

`contextd/storage/base.py` defines the `GraphStore` abstract base class. All higher layers (indexer, MCP server, CLI) depend exclusively on this ABC. Backend-specific modules (`memgraph.py`, `neo4j.py`) are confined to `contextd/storage/`; a CI grep step in `.github/workflows/ci.yml` enforces the separation.

The ABC surface:

| Method | Purpose |
|---|---|
| `connect() / close()` | Lifecycle management |
| `apply_migrations(migrations)` | Forward-only schema migrations |
| `upsert_node(label, props) → id` | Insert or update a node |
| `upsert_edge(src, dst, edge_type, origin, props, *, src_label, dst_label)` | Insert or update an edge |
| `delete_edges(src, *, origin, edge_type, src_label)` | Scoped delete; raises `ValueError` when both `origin` and `edge_type` are `None` |
| `exec_read(cypher, params)` | Read-only Cypher |
| `exec_write(cypher, params)` | Write Cypher |
| `vector_search(label, prop, query, k, threshold)` | Cosine-similarity nearest-neighbour lookup |
| `full_text_search(label, prop, query, k)` | Full-text index query |
| `capabilities` | `BackendCapabilities` frozen dataclass |

`src_label` and `dst_label` are advisory on both backends but are required for correct edge MERGE semantics on Neo4j.

#### BackendCapabilities

`BackendCapabilities` is a frozen dataclass returned by `GraphStore.capabilities`. Callers adapt behaviour via these flags rather than attempting an operation and reacting to failures.

| Field | Type | Purpose |
|---|---|---|
| `name` | `BackendName` | `"memgraph"` or `"neo4j"` — identifies the active backend |
| `concurrent_writers` | `int` | Maximum concurrent writers; `-1` means unlimited |
| `supports_vector_index` | `bool` | Whether the backend exposes a native cosine-similarity vector index |
| `supports_full_text_index` | `bool` | Whether the backend exposes a native full-text search index |
| `supports_graph_algorithms` | `bool` | Whether built-in graph algorithm procedures (e.g. PageRank) are available |
| `requires_docker` | `bool` | `True` for both current backends — they run as Docker containers |
| `default_connection` | `str` | Default Bolt URI used by the factory when no config override is present |

The `unlimited_writers` property returns `True` when `concurrent_writers == -1`.

#### Primary-key map

`contextd/storage/_keys.py` contains `PRIMARY_KEY_BY_LABEL` — the canonical label-to-PK-property mapping that mirrors the migration DDL for both backends. Both `upsert_node` and `delete_edges` implementations delegate PK lookups here. When a migration adds a new node label with a PK, this map must be updated in lock-step.

```python
# abbreviated extract
PRIMARY_KEY_BY_LABEL = {
    "File":    "path",
    "Section": "id",
    "Pattern": "name",
    "Ticket":  "id",
    "Risk":    "description",
    ...
}
```

#### Factory and backend selection

`contextd/storage/factory.py::build_graph_store(cfg)` reads `cfg.storage.backend` and returns the matching concrete instance via deferred import. Deferred import keeps each backend SDK out of the import path when the other backend is active — this is what the abstraction-invariant grep checks.

#### Backend comparison

| Property | Neo4j Community (default) | Memgraph |
|---|---|---|
| Bolt port | 7687 | 7687 |
| Vector index | Native (no plugin) | Native |
| Full-text index | Native | Native |
| Graph algorithms | Limited (no GDS on Community) | Built-in (`pagerank`, `community_detection.louvain`) |
| Docker image | `neo4j:5` | `memgraph:latest` (v3.x — do NOT use `memgraph-platform:latest`, pinned at v2.14) |
| Compose profile | `--profile neo4j` | `--profile memgraph` |

#### Migrations

`contextd/storage/migration.py` implements a `MigrationRunner` that applies migrations in order and records the schema version in a `Meta` node. Migration files are split by backend:

- `contextd/migrations/memgraph/` — Memgraph-specific DDL (vector index syntax, uniqueness constraints)
- `contextd/migrations/neo4j/` — Neo4j-specific DDL (vector index options, constraint syntax)

Migrations are forward-only; no rollback support.

---

### Layer 3 — Inference and MCP surface

**Inference providers** (both via HTTPS):

- `GeminiProvider` — Gemini Flash by default; used for file/section summarisation, relationship inference, and natural-language to Cypher translation. Configured via `GEMINI_API_KEY`.
- `VoyageProvider` — Voyage AI `voyage-4-large` model (1024-dim, 32k-token context per input; `voyage-3`, `voyage-3-large`, `voyage-code-3` are also registered for users who want to override via `[providers.voyage] model`). Used for document and section embeddings. Configured via `VOYAGE_API_KEY`.

Both implement an ABC (`InferenceProvider` / `EmbeddingProvider`) defined in `contextd/providers/base.py`. Usage is logged to `~/.contextd/state/session-log/` via `CostLog`.

**MCP server** — `contextd/mcp_server.py` implements a stdio MCP server registered as the `contextd-mcp` console script. It exposes 8 generic tools plus per-corpus Cypher tools (see [mcp.md](mcp.md)). The server connects to the storage backend over Bolt at startup and holds the connection for the session lifetime.

**CLI** — `contextd` is a short-lived Click process. Each invocation connects to the backend, does its work, and disconnects. See [cli.md](cli.md).

---

## Process model

| Process | Lifetime | Managed by |
|---|---|---|
| Storage backend (Docker) | Long-lived | `contextd up` / `contextd down` |
| `contextd-mcp` | Long-lived (session) | MCP client (Claude Desktop / Cursor) spawns on stdio |
| `contextd` CLI | Short-lived | User invocation |
| Indexer daemon | Not yet wired | Planned; `CorpusWatcher` exists in M5 |

The storage backend and MCP server are the only persistent processes at runtime. All data reads from the CLI go directly to the backend via Bolt; there is no intermediate daemon for data-plane queries.

---

## Architecture invariants (CI-enforced)

1. **No backend imports outside `contextd/storage/`.** Enforced by a `grep` job in `.github/workflows/ci.yml`. The factory's deferred imports are the only place backend module names appear outside the storage package.
2. **Every edge carries `origin ∈ {inferred, structural, manual}`.** Wipe-and-replace on re-index operates only on `origin="inferred"`.
3. **AI-inferred edges are ontology-validated at write time.** `Ontology.validate_edge()` rejects types not in `contextd/ontology/base.json`.
4. **Section-granular mode is opt-in per corpus.** File-granular is the default.
