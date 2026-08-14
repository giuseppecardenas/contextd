# CLAUDE.md

**contextd** is a locally-hosted GraphRAG knowledge layer and MCP server. It
indexes markdown, code, and structured-data corpora into a Neo4j graph + vector
store, generates per-file summaries and AI-inferred typed relationships
(Gemini, or any OpenAI-compatible provider profile such as Ollama Cloud or
DeepSeek; embeddings via Voyage), and serves the result to AI assistants over
the Model Context Protocol.

## Prerequisites

- Python 3.11+ (the code uses 3.11-only idioms).
- `uv` for environment management.
- Docker running Neo4j Community 5.x (`neo4j:5`), Bolt on port 7687. Started for
  you by `contextd up`.
- Environment variables: `GEMINI_API_KEY` (inference/summaries) and
  `VOYAGE_API_KEY` (embeddings). Provider choice is per call-site (`summary`,
  `inference`, `translation`, `embedding`), so these can be mixed or replaced.
  A call-site set to `openai_compat:<profile>` reads its key from the variable
  named by `[providers.openai_compat.<profile>] api_key_env` instead — e.g.
  `OLLAMA_API_KEY` with `base_url = "https://ollama.com/v1"` (Ollama Cloud),
  or `DEEPSEEK_API_KEY` with `base_url = "https://api.deepseek.com/v1"`.
  Any number of profiles can coexist, mixed per call-site. Keys are read from
  the environment at startup and never written to disk.
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

ruff check .                         # lint
ruff format --check .                # format check (drop --check to apply)
mypy --strict contextd               # type check
```

Every commit must leave all four gates green: `ruff check`,
`ruff format --check`, `mypy --strict contextd`, `pytest tests/unit`. Also run
the abstraction-invariant grep defined in `.github/workflows/ci.yml` before
pushing.

Run the ruff commands over `.`, not over `contextd tests`. CI checks the whole
repository, so a narrower local scope silently misses files outside those two
directories (docs, examples, scripts) and the break only surfaces after a push.

## Code Style & Architecture

- **Strict typing.** Every public function and method is fully annotated;
  `mypy --strict` must pass. Use a narrow `# type: ignore[code]` only for genuine
  third-party limitations.
- **Async at I/O boundaries.** Prefer async handlers for network and database
  I/O; keep pure logic synchronous.
- **Error boundaries around externals.** Wrap Neo4j and AI-API calls
  (Gemini, Voyage, any `openai_compat` endpoint) in explicit error handling
  with retry/backoff; never let a provider failure crash the indexer or the
  MCP loop.
- **Treat model output as hostile input.** Parse it through
  `contextd.inference._json_body.loads_json_body`, never a bare `json.loads`.
  Enabling a provider's JSON mode does not guarantee well-formed JSON —
  DeepSeek emits trailing commas despite `response_format`, so the parser
  tries a strict parse first and repairs trailing commas only on failure,
  using a string-literal-aware scanner (summary prose legitimately contains
  `, ]`). A 200 carrying an empty completion is a retryable provider miss, not
  an answer; see `OpenAICompatProvider.generate`.
- **Every text-mode file I/O pins `encoding="utf-8"`.** `Path.read_text`,
  `Path.write_text` and `open` otherwise default to
  `locale.getpreferredencoding(False)` — UTF-8 on a Linux CI runner, cp1252 on
  a stock Windows install — so a bare call silently mangles non-ASCII corpus
  content into mojibake that is then embedded, summarised and hashed. Nothing
  raises: cp1252 maps most bytes to wrong-but-valid characters, and
  `errors="replace"` does not help. It also makes section hashes
  platform-dependent, so a daemon run on both Windows and Linux re-indexes
  unchanged files forever. Pin reads and their paired writes together, so
  contextd never writes in one encoding and reads back in another.
  (Enforced by `tests/unit/test_encoding_invariant.py`, which walks the
  package AST — the defect cannot reproduce on a UTF-8 CI runner, so a
  behavioural test would pass on CI while the bug was live on Windows.)
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
