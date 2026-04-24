# Contextd

**Files are the source of truth; AI is the semantic layer.**

Contextd is a locally-hosted knowledge layer for your project files. It indexes markdown, code, and structured data into a hybrid graph + vector store (Neo4j Community or Memgraph, your choice), generates AI-inferred relationships and per-file summaries, and exposes the result to Claude Desktop, Cursor, and any MCP-speaking client through an MCP server. Cold-start any AI session with a compact, semantically-organised overview of your entire corpus.

- **Storage:** Neo4j Community 5.x (default) or Memgraph 3.x — both run in Docker, both bind port 7687.
- **Inference:** Google Gemma (`gemma-4-31b-it` default) via the Gemini API for summarisation, relationship inference, and NL→Cypher translation; Voyage AI `voyage-4-large` (1024-dim, 32k-token context) for vector embeddings.
- **Interface:** stdio MCP server (`contextd-mcp`) and CLI (`contextd`).
- **Privacy:** all state lives under `~/.contextd/`; no data is stored outside your machine beyond the per-file API calls.

> **Status: alpha.** v0.1.0 is pre-PyPI. Use the dev install below. The repo is private while the final documentation milestone lands.

---

## Quickstart

```bash
# 1. Install (dev path — see Install section for details)
git clone git@github.com:giuseppecardenas/contextd.git
cd contextd
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Set API keys
export GEMINI_API_KEY=<your-key>
export VOYAGE_API_KEY=<your-key>

# 3. First-run wizard — creates ~/.contextd/ layout
contextd init

# 4. Start the storage backend + indexer daemon
contextd up

# 5. Register a corpus and index it
contextd add-corpus examples/minimal-notes --name notes
contextd index notes --bootstrap

# 6. Query the graph
contextd ask "which files reference note-1?"
```

After `contextd up` and a successful bootstrap, open Claude Desktop and call the `describe_project` MCP tool — you get a JSON primer of the top-cited files with summaries, ready for an AI assistant to anchor its session.

---

## Concepts

### File nodes vs Section nodes

By default, each file in a corpus is indexed as a single `File` node with an AI-generated summary and a `voyage-4-large` embedding (1024-dim). This is **file-granular** mode.

For heavily structured markdown corpora (long PRDs, specs, design docs), **section-granular** mode promotes each heading to a first-class `Section` node. Structural edges — `CONTAINS`, `PARENT_OF`, `NEXT_SIBLING` — model the document tree; AI infers semantic edges (`REFERENCES`, `DOCUMENTS`, `SUPERSEDES`, etc.) across section boundaries. Enable it with `--granularity section` in `add-corpus`.

### Edge origins

Every edge carries one of three `origin` values:

| Origin | Meaning |
|---|---|
| `structural` | Document structure (CONTAINS / PARENT\_OF / NEXT\_SIBLING). Never overwritten on re-index. |
| `inferred` | AI-inferred semantic relationship. Wipe-and-replace on re-index. |
| `manual` | Hand-authored. Never overwritten on re-index. |

### The GraphStore abstraction

All higher layers (indexer, MCP server, CLI) talk to the backend through the `GraphStore` ABC. The concrete backend — `Neo4jBackend` or `MemgraphBackend` — is selected by a single line in `~/.contextd/config.toml`:

```toml
[storage]
backend = "neo4j"   # or "memgraph"
```

Switching backends is a `contextd down && contextd up` cycle; no code change or re-installation needed.

### Ontology

The base ontology (`contextd/ontology/base.json`) defines the node types and edge types the AI is allowed to infer. Unrecognised types are silently discarded at index time. Per-corpus aliases let you map domain-specific vocabulary to canonical types — see [Ontology customisation](#ontology-customisation) below.

Node types: `File`, `Section`, `Artifact`, `Ticket`, `Pattern`, `Technology`, `Client`, `Repo`, `Service`, `Integration`, `Risk`, `WorkSession`, `Corpus`, `Meta`.

Edge types: `CONTAINS`, `PARENT_OF`, `NEXT_SIBLING`, `BELONGS_TO`, `CREATED_BY`, `DOCUMENTS`, `DOCUMENTED_IN`, `APPLIES_TO`, `PART_OF`, `SIMILAR_TO`, `RELATED_TO`, `REFERENCES`, `SUPERSEDES`, `CONTRADICTS`, `USES`, `MODIFIES`, `DEPENDS_ON`, `IDENTIFIES_RISK`, `RECOMMENDS`, `EDITED_DURING`, `VERIFIED_ON`, `NEEDS_UPDATE`.

---

## Install

### Pre-PyPI (current)

The package is not yet on PyPI. Clone the repo and install in editable mode:

```bash
git clone git@github.com:giuseppecardenas/contextd.git
cd contextd
uv venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"
```

Requirements: Python 3.11+, Docker (Docker Desktop or any Docker engine with Compose v2).

### After v0.1.0 on PyPI (future)

```bash
pipx install contextd
```

The `pipx` path is the intended production install once the package is published. For now, use the dev install above.

---

## Usage walkthrough

This walkthrough uses `examples/minimal-notes` — a personal-notes fixture (10 note files + a README) included in the repo. The same steps apply to any corpus.

### Step 1 — First-run setup

```bash
contextd init
```

Creates `~/.contextd/` with the directory layout, copies `config.toml` from the package default, and checks for required env vars (`GEMINI_API_KEY`, `VOYAGE_API_KEY`) and Docker. Set both keys in your shell before proceeding:

```bash
export GEMINI_API_KEY=<from https://aistudio.google.com/app/apikey>
export VOYAGE_API_KEY=<from https://www.voyageai.com/>
```

### Step 2 — Start the backend

```bash
contextd up
```

Runs `docker compose --profile neo4j up -d` (using `~/.contextd/docker-compose.yml`), waits for Neo4j to be ready, then applies the schema migrations (constraints, vector index, full-text index). Output:

```
✓ neo4j container up at 127.0.0.1:7687
✓ migrations applied
ready
```

To use Memgraph instead, set `backend = "memgraph"` in `~/.contextd/config.toml` before running `contextd up`.

### Step 3 — Register the corpus

```bash
contextd add-corpus examples/minimal-notes --name notes
```

Writes `~/.contextd/corpora/notes.toml` with root path, glob pattern (`**/*.md`), and `granularity = "file"`. To index at section granularity:

```bash
contextd add-corpus examples/minimal-notes --name notes --granularity section
```

### Step 4 — Bootstrap the index

```bash
contextd index notes --bootstrap
```

Runs the five-phase bootstrap pipeline:

1. **enumerate** — discovers files, computes embeddings at CREATE time.
2. **embed** — accounting stub (embeddings already written in phase 1).
3. **summarise** — calls Gemini Flash once per file to generate `summary` and `key_points`.
4. **relate** — calls Gemini Flash once per file to infer typed edges between nodes.
5. **close** — persists corpus stats and checkpoint state.

Progress is printed per phase:

```
found 11 files in corpus 'notes'
  ✓ enumerate: processed=11 skipped=0
  ✓ embed: processed=11 skipped=0
  ✓ summarise: processed=11 skipped=0
  ✓ relate: processed=11 skipped=0
  ✓ close: processed=1 skipped=0
```

To preview token cost without indexing:

```bash
contextd index notes --estimate-only
```

**Resuming a partial run.** Re-running `contextd index <corpus> --bootstrap` is idempotent: nodes that already have a summary or an `inferred_at` marker (written by a prior successful `relate` pass) are skipped automatically. A killed-mid-run bootstrap only needs a plain `--bootstrap` restart to finish — no extra flag.

**Wiping a layer for a fresh re-inference.** Use `--refresh <scope>`:

| scope | wipes | preserves | re-costs |
|---|---|---|---|
| `inferred` | `origin='inferred'` edges + `inferred_at` markers | summaries, structural, embeddings | Gemini relate only |
| `summaries` | `summary`/`key_points`/`summary_confidence` | inferred edges, structural, embeddings | Gemini summarise only |
| `llm` | both of the above | structural, embeddings | all Gemini work |
| `all` | DETACH DELETE every `Section`/`File`/`Corpus` node for this corpus | nothing (structural + inferred edges cascade-deleted) | Voyage + Gemini from zero |

```bash
contextd index notes --bootstrap --refresh llm       # e.g. after a prompt change
```

**Incremental re-index (one-shot).** To re-index only files that have changed since the last run:

```bash
contextd index notes --incremental
```

Scans every file in the corpus, uses MD5 hashing to skip unchanged files, and re-processes only what changed. Output:

```
  ✓ incremental scan complete: indexed=2 deleted=0 skipped=9
```

For continuous watching, use the daemon — see [Incremental indexer daemon](#incremental-indexer-daemon) below.

### Step 5 — Query

```bash
contextd ask "which notes mention cooking?"
```

Translates the question to Cypher via Gemini, runs the query, prints results as JSON. The Cypher is printed first so you can inspect what was generated.

### Step 6 — Connect Claude Desktop

Add `contextd-mcp` to your Claude Desktop config (see [MCP integration](#mcp-integration)). Once connected, call `describe_project` from the Claude Desktop interface. For the minimal-notes corpus you get something like:

```json
[
  {"path": "examples/minimal-notes/note-1.md", "name": "note-1", "summary": "...", "key_points": [...], "inbound": 3},
  ...
]
```

The tool returns up to 40 nodes ordered by inbound-citation count, each with its AI summary. An AI assistant can use this as a session primer — one tool call replaces reading every file.

---

## MCP integration

Contextd exposes a stdio MCP server. It works with any MCP-speaking client over stdio — Claude Desktop (macOS and Windows), Cursor (cross-platform), Zed, or your own MCP-compatible tooling. The `contextd-mcp` binary reads/writes JSON-RPC on stdin/stdout; consult your client's MCP config docs for how to register the server.

**macOS (Claude Desktop)** — config at `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "contextd": {
      "command": "contextd-mcp",
      "args": []
    }
  }
}
```

**Windows (Claude Desktop)** — config at `%APPDATA%\Claude\claude_desktop_config.json` — same JSON structure as above.

**Cursor / Zed / other clients** — register `contextd-mcp` as a stdio MCP server; consult your client's docs for the exact config format.

If `contextd-mcp` is not on your PATH (e.g., when using a venv), use the absolute path to the binary:

```json
{
  "mcpServers": {
    "contextd": {
      "command": "/home/you/src/contextd/.venv/bin/contextd-mcp",
      "args": []
    }
  }
}
```

### Generic tools (always registered)

| Tool | What it does |
|---|---|
| `describe_project` | Top-N File nodes by inbound-citation count with summaries. Accepts `corpus` and `n` (default 40). |
| `search` | Full-text search over summaries. Accepts `query`, optional `kind` (default `File`; `Section` also works in section-granular corpora), optional `limit` (default 20). |
| `related` | Outbound + inbound traversal within N hops (1–5). Accepts `node_id` and `depth` (default 2). |
| `inbound` | What cites this node? Accepts `node_id`. |
| `outbound` | What does this node cite? Accepts `node_id`. |
| `get_file_summary` | Summary + key points for a single file. Accepts `path`. |
| `section_tree` | Outline of a file (section-granular corpora only). Accepts `file_path`. |
| `query_graph` | Read-only Cypher escape hatch. Accepts `cypher`. Write keywords are rejected by a guard. |

### Per-corpus tools

Corpus adapters can register additional Cypher-backed tools in their corpus TOML under `[mcp.tools]`. These appear in the tool list namespaced as `<corpus>.<tool>` so they never collide with the generic tools. An AI assistant calling a registered tool runs the pre-authored Cypher with `$placeholder` parameters bound, all through the same read-only guard.

Most adapters do fine without this — `query_graph` + the generic tools compose into the same queries and are tunable in-session without a server restart. Reach for `[mcp.tools]` when a query is called often enough that baking it in pays back the loss of tunability.

See [docs/mcp.md](docs/mcp.md) for the full tool reference.

---

## Incremental indexer daemon

`contextd up` starts both the graph backend **and** a background file-watching daemon (`contextd-indexer`). The daemon monitors every registered corpus root for file changes and re-indexes modified files automatically — no manual `contextd index` calls needed after the initial bootstrap.

### Lifecycle

```bash
contextd up        # start graph backend + indexer daemon
contextd status    # show daemon state (pid, uptime, corpora)
contextd down      # stop daemon + graph backend
```

`contextd status` tries the IPC socket first for live runtime info, and falls back to the PID file if the daemon is not reachable:

```
backend:  neo4j running at 127.0.0.1:7687
daemon:   running (pid=12345, uptime=42s, corpora=['notes'])
```

### How it works

1. A `CorpusWatcher` (inotify / FSEvents) fires on every file change under the corpus root.
2. Events are collected by a `DebouncedQueue` with a configurable window (default 30 s) to batch rapid edits into a single pass.
3. After the window closes, the daemon runs MD5 checks to skip files whose content did not actually change.
4. Changed files are re-indexed concurrently (default 4 workers) by calling the same phase pipeline as `--incremental`.
5. **Crash recovery:** before processing each batch, a checkpoint is written. If the daemon is killed mid-batch, the next startup replays any in-flight files. Paths that fail indexing are buffered to `~/.contextd/state/pending-upserts.jsonl` and retried on the next start.

### Branch gate

If you check out a remote branch for comparison, the daemon would re-index every differing file — potentially hundreds of Gemini calls against a scratch branch. The `allowed_branches` gate prevents this:

```toml
# ~/.contextd/config.toml
[indexer]
allowed_branches = ["main", "develop"]
```

When the corpus repo's active branch is not in the list, both the daemon and `contextd index --incremental` skip all work and log a warning. An empty list (the default) allows all branches.

Detached HEAD is always blocked when a whitelist is configured (a comparison checkout puts the repo in detached HEAD state).

### Tuning

```toml
[indexer]
debounce_seconds    = 30   # seconds to wait after the last event before dispatching a batch
incremental_workers = 4    # concurrent file workers per batch (distinct from inference_concurrency)
inference_concurrency = 1  # LLM call parallelism within each file's summarise+relate phases

[logging]
max_log_bytes  = 10485760  # 10 MB per log file; 0 disables rotation
log_backup_count = 5       # number of rotated files to keep
```

The daemon writes only to the configured log file (`~/.contextd/logs/contextd.log` by default) — no terminal output — so these rotation settings matter for long-running installs.

---

## Ontology customisation

The base ontology is intentionally general. Domain-specific corpora map their vocabulary to Contextd's canonical node and edge types through two mechanisms:

1. **Inline aliases** (`[ontology.aliases]` in the corpus TOML) — map domain names to canonical node types. For example, `Registry = "Pattern"` tells the AI that an inferred `Registry` node should be stored as a `Pattern`.

2. **Override file** (`[ontology] overrides = "ontology.json"`) — a JSON file that adds domain-specific edge-type aliases. All aliases are validated against the base ontology at index time; unrecognised types are silently discarded, not stored.

Adapter configs live next to the corpus they describe (by convention, a `.contextd/` directory at the corpus root containing `corpus.toml`, `ontology.json`, and `prompts/`). `contextd add-corpus <path> --from <path>/.contextd/corpus.toml` rewrites relative paths in the template to absolute paths anchored at the template's directory, so the adapter stays portable. See [docs/ontology.md](docs/ontology.md) for the full customisation reference.

---

## Configuration reference

Contextd uses three levels of config, all under `~/.contextd/`:

| Path | Purpose |
|---|---|
| `~/.contextd/config.toml` | Global config — storage backend, provider models, inference settings, logging. Created by `contextd init`. |
| `~/.contextd/corpora/<name>.toml` | Per-corpus config — root path, glob includes, granularity, ontology aliases and overrides, MCP tools, summarisation overrides. Created by `contextd add-corpus`. |
| `~/.contextd/state/` | Runtime state — checkpoint JSON per corpus, cost log, session log. Not hand-edited. |

Key global config fields (full reference in [docs/architecture.md](docs/architecture.md)):

```toml
[storage]
backend = "neo4j"           # "neo4j" (default) or "memgraph"

[providers]
inference = "gemini"
embedding = "voyage"

[inference]
summary_max_words = 100

[indexer]
debounce_seconds       = 30   # seconds to collect FS events before dispatching a batch
parallel_embedding_batches = 4
inference_concurrency  = 1    # LLM call parallelism (summarise+relate); 5 is a good
                              # default for Gemma free-tier (15 RPM quota)
incremental_workers    = 4    # concurrent file workers per incremental batch
allowed_branches       = []   # whitelist; empty = allow all branches

[logging]
level          = "info"
format         = "json"
path           = "~/.contextd/logs/contextd.log"
max_log_bytes  = 10485760   # 10 MB; 0 = no rotation
log_backup_count = 5
```

Full corpus config schema and CLI reference live in [docs/cli.md](docs/cli.md).

---

## Cost analysis

Each file (or section, in section-granular mode) triggers two Gemini API calls (summarise + relate) and contributes to a batched Voyage AI embedding call per bootstrap. At `gemma-4-31b-it` and `voyage-4-large` pricing:

- **Per file:** sub-cent in typical cases (short markdown files); files with dense content may run a few cents each.
- **Typical 100-file corpus:** order of $0.10–$1.00 for a full bootstrap, depending on file sizes and summary length (`[inference] summary_max_words` is the main lever).
- **Incremental re-index:** only changed files are re-processed (MD5 hash gating), so ongoing cost is proportional to the edit rate, not corpus size.

These are order-of-magnitude estimates. Exact spend is logged per session to `~/.contextd/state/session-log/`. Inspect it with:

```bash
contextd costs
contextd costs --since 2026-04-01
```

---

## Security

Contextd is a single-user local tool. Its security posture reflects that:

- **API keys** (`GEMINI_API_KEY`, `VOYAGE_API_KEY`) are read from env vars at process startup. They are never written to disk by Contextd.
- **Graph store** binds to `127.0.0.1:7687` only. Neither Neo4j nor Memgraph is exposed beyond the loopback interface in the default compose config.
- **MCP read-only guard.** The `query_graph` tool and all per-corpus Cypher tools pass through `assert_read_only` before execution. The guard rejects Cypher containing `CREATE`, `MERGE`, `DELETE`, `SET`, `REMOVE`, `DROP`, `DETACH`, `FOREACH`, and `CALL` with side-effecting procedures. This guards against prompt-injection attacks that attempt to write to the graph through the MCP surface.
- **Do not expose Contextd's MCP server over a network.** It is designed for stdio transport to a locally-running MCP client. Running it as a shared network service is out of scope and untested.

---

## Roadmap / Known gaps

Items that are designed and partially built but not yet wired or shipped:

| Gap | Detail |
|---|---|
| **Hybrid search** | `search()` is full-text only. Vector-similarity fallback (hybrid ranking) is deferred; callers needing vector-space matches can call `GraphStore.vector_search` directly for now. |
| **`CONTEXTD_INFERENCE_DAILY_BUDGET`** | The design specifies an env var cap on daily Gemini calls. Not implemented; manual cost monitoring via `contextd costs` is the current guard. |
| **Per-corpus MCP tool `$` false positives** | `extract_placeholders` uses a simple regex and will match `$` inside Cypher string literals as spurious parameters. Proper Cypher tokenisation is deferred. |
| **Stale `CheckpointStore` entries on incremental section refresh** | If an incremental re-index refreshes sections in a section-granular corpus, existing checkpoint entries for the affected file are not invalidated. They will report the old phase state until the next full bootstrap. (`contextd/indexer/phases.py` `TODO(M9-followup)`) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, the architectural invariants that CI enforces, and the commit-message conventions.

---

## Licence

Contextd is released under the MIT Licence. See [LICENSE](LICENSE).
