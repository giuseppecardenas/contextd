# Contextd

**Files are the source of truth; AI is the semantic layer.**

Contextd is a locally-hosted knowledge layer for your project files. It indexes markdown, code, and structured data into a hybrid graph + vector store (Neo4j Community or Memgraph, your choice), generates AI-inferred relationships and per-file summaries, and exposes the result to Claude Desktop, Cursor, and any MCP-speaking client through an MCP server. Cold-start any AI session with a compact, semantically-organised overview of your entire corpus.

- **Storage:** Neo4j Community 5.x (default) or Memgraph 3.x — both run in Docker, both bind port 7687.
- **Inference:** Google Gemini Flash for summarisation and relationship inference; Voyage AI `voyage-3` (1024-dim) for vector embeddings.
- **Interface:** stdio MCP server (`contextd-mcp`), CLI (`contextd`), and an optional background indexer daemon (`contextd-indexer`).
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

# 4. Start the storage backend (Neo4j by default)
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

By default, each file in a corpus is indexed as a single `File` node with an AI-generated summary and a `voyage-3` embedding. This is **file-granular** mode.

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

This walkthrough uses `examples/minimal-notes` — a 10-file personal-notes fixture included in the repo. The same steps apply to any corpus.

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
found 10 files in corpus 'notes'
  ✓ enumerate: processed=10 skipped=0
  ✓ embed: processed=10 skipped=0
  ✓ summarise: processed=10 skipped=0
  ✓ relate: processed=10 skipped=0
  ✓ close: processed=1 skipped=0
```

To preview token cost without indexing:

```bash
contextd index notes --estimate-only
```

### Step 5 — Query

```bash
contextd ask "which notes mention cooking?"
```

Translates the question to Cypher via Gemini, runs the query, prints results as JSON. The Cypher is printed first so you can inspect what was generated.

### Step 6 — Connect Claude Desktop

Add `contextd-mcp` to your Claude Desktop config (see [MCP integration](#mcp-integration)). Once connected, call `describe_project` from the Claude Desktop interface. For a 10-note corpus you get something like:

```json
[
  {"path": "examples/minimal-notes/note-1.md", "name": "note-1", "summary": "...", "key_points": [...], "inbound": 3},
  ...
]
```

The tool returns up to 40 nodes ordered by inbound-citation count, each with its AI summary. An AI assistant can use this as a session primer — one tool call replaces reading every file.

---

## MCP integration

Contextd exposes a stdio MCP server. Add it to your Claude Desktop config at `~/.config/claude-desktop/config.json` (Linux/WSL) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

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

If `contextd-mcp` is not on your PATH (e.g., in a venv), use the absolute path:

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
| `search` | Full-text search over summaries. Accepts `query`. |
| `related` | Outbound + inbound traversal within N hops (1–5). Accepts `node_id` and `depth` (default 2). |
| `inbound` | What cites this node? Accepts `node_id`. |
| `outbound` | What does this node cite? Accepts `node_id`. |
| `get_file_summary` | Summary + key points for a single file. Accepts `path`. |
| `section_tree` | Outline of a file (section-granular corpora only). Accepts `file_path`. |
| `query_graph` | Read-only Cypher escape hatch. Accepts `cypher`. Write keywords are rejected by a guard. |

### Per-corpus tools

Corpus adapters can register additional Cypher-backed tools in their corpus TOML under `[mcp.tools]`. These appear in the tool list namespaced as `<corpus>.<tool>` so they never collide with the generic tools.

For example, the `runeledger-prd` adapter registers `runeledger-prd.four_surface`, `runeledger-prd.find_dangling_registrations`, and `runeledger-prd.audit_stale_shas`. An AI assistant calling `runeledger-prd.four_surface(registry_name="economy")` runs the pre-authored Cypher with `$registry_name` bound, all through the same read-only guard.

See `examples/runeledger-prd/` for a working adapter and [docs/mcp.md](docs/mcp.md) for the full tool reference.

---

## Ontology customisation

The base ontology is intentionally general. Domain-specific corpora map their vocabulary to Contextd's canonical node and edge types through two mechanisms:

1. **Inline aliases** (`[ontology.aliases]` in the corpus TOML) — map domain names to canonical node types. For example, `Registry = "Pattern"` tells the AI that an inferred `Registry` node should be stored as a `Pattern`.

2. **Override file** (`[ontology] overrides = "ontology.json"`) — a JSON file that adds domain-specific edge-type aliases. All aliases are validated against the base ontology at index time; unrecognised types are silently discarded, not stored.

The `examples/runeledger-prd/` adapter is the reference implementation. It maps four domain node types (`Registry`, `FRRow`, `LuaFile`, `GapEntry`) and five domain edge types (`CITES`, `CONSUMES`, `SCHEMA_FOR`, `REGISTERS`, `CLOSES_GAP`) to canonical equivalents, uses a custom summary prompt, and registers three MCP tools. See [docs/ontology.md](docs/ontology.md) for the full customisation reference.

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

[logging]
level = "info"
format = "json"
path = "~/.contextd/logs/contextd.log"
```

Full corpus config schema and CLI reference live in [docs/cli.md](docs/cli.md).

---

## Cost analysis

Each file triggers two Gemini Flash API calls (summarise + relate) and one Voyage AI batch embedding call per bootstrap. At Gemini Flash and Voyage-3 pricing:

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
- **Graph store** binds to `127.0.0.1:7687` only. Neither Neo4j nor Memgraph is exposed beyond the loophole interface in the default compose config.
- **MCP read-only guard.** The `query_graph` tool and all per-corpus Cypher tools pass through `assert_read_only` before execution. The guard rejects Cypher containing `CREATE`, `MERGE`, `DELETE`, `SET`, `REMOVE`, `DROP`, `DETACH`, `FOREACH`, and `CALL` with side-effecting procedures. This guards against prompt-injection attacks that attempt to write to the graph through the MCP surface.
- **Do not expose Contextd's MCP server over a network.** It is designed for stdio transport to a locally-running MCP client. Running it as a shared network service is out of scope and untested.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, the architectural invariants that CI enforces, and the commit-message conventions.

---

## Licence

Contextd is released under the MIT Licence. See [LICENSE](LICENSE).
