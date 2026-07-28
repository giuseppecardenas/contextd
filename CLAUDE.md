# CLAUDE.md

**contextd** is a locally-hosted GraphRAG knowledge layer and MCP server. It
indexes markdown, code, and structured-data corpora into a Neo4j graph + vector
store, generates per-file summaries and AI-inferred typed relationships
(Gemini + Voyage), and serves the result to AI assistants over the Model Context
Protocol.

## Prerequisites

- Python 3.11+ (the code uses 3.11-only idioms).
- `uv` for environment management.
- Docker running Neo4j Community 5.x (`neo4j:5`), Bolt on port 7687. Started for
  you by `contextd up`.
- Environment variables: `GEMINI_API_KEY` (inference/summaries) and
  `VOYAGE_API_KEY` (embeddings).
- All persistent state lives under `~/.contextd/` (config, corpora registry,
  logs, checkpoints).

## Setup

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
```

Activate the venv (`source .venv/bin/activate`) at the start of every session
that runs code.

## Core Workflow

```bash
contextd init                              # first-run wizard: ~/.contextd/ layout + config
contextd up                                # start the Neo4j container, apply migrations
contextd add-corpus <path> --name <name>   # register a corpus (--granularity file|section)
contextd index <name> --bootstrap          # index it (--incremental / --refresh also available)
contextd ask "<query>"                      # natural-language query -> Cypher -> rows
```

Supporting commands: `contextd status`, `contextd down` (stops the backend,
keeps data), `contextd reset` (removes the backend and deletes all indexed
data), `contextd list-corpora`, `contextd remove-corpus <name>` (unregister +
delete one corpus's data), `contextd prune-entities` (reap orphaned entity
nodes), `contextd logs`, `contextd costs`.

### MCP server

```bash
contextd-mcp        # stdio MCP server (for Claude Desktop / Cursor, or manual debugging)
```

`contextd-mcp` speaks JSON-RPC over stdin/stdout. Do not print to stdout from
server code (see Logging below).

## Testing & Quality

```bash
pytest tests/unit -v                 # fast, no Docker required
pytest tests/integration tests/e2e   # require Docker (spins up Neo4j via testcontainers)

ruff check contextd tests            # lint
ruff format --check contextd tests   # format check (drop --check to apply)
mypy --strict contextd               # type check
```

Every commit must leave all four gates green: `ruff check`,
`ruff format --check`, `mypy --strict contextd`, `pytest tests/unit`. Also run
the abstraction-invariant grep defined in `.github/workflows/ci.yml` before
pushing.

## Code Style & Architecture

- **Strict typing.** Every public function and method is fully annotated;
  `mypy --strict` must pass. Use a narrow `# type: ignore[code]` only for genuine
  third-party limitations.
- **Async at I/O boundaries.** Prefer async handlers for network and database
  I/O; keep pure logic synchronous.
- **Error boundaries around externals.** Wrap Neo4j and AI-API calls
  (Gemini, Voyage) in explicit error handling with retry/backoff; never let a
  provider failure crash the indexer or the MCP loop.
- **Logging, never `print`.** Use `structlog`; logs are written to
  `~/.contextd/logs/`. A stray `print()` to stdout corrupts the MCP stdio
  protocol frames and breaks the server.
- **Storage stays behind the ABC.** Do not import `contextd.storage.neo4j`
  outside `contextd/storage/`. Consumers depend on the `GraphStore` interface;
  the factory is the only place the concrete backend is named. (CI-enforced.)
- **Every edge carries `origin` in {inferred, structural, manual}.**
  Re-index wipe-and-replace touches only `origin="inferred"` edges; structural
  and manual edges are preserved.
- **AI-inferred edges are ontology-validated at write time**, which is the
  primary defense against hallucinated relationship types.

## Commits

Conventional commits (`type(scope): summary`), one self-contained change per
commit, CI green before each commit. Never bypass hooks (`--no-verify`), never
amend pushed commits.
