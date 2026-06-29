# CLI Reference

The `contextd` command is a short-lived Click process. Each invocation connects to the configured storage backend, performs its action, and disconnects. All commands read `~/.contextd/config.toml` (falling back to built-in defaults when absent).

---

## `contextd init`

**Synopsis:** `contextd init [--yes]`

| Flag | Default | Description |
|---|---|---|
| `--yes` | off | Accept all defaults non-interactively (reserved; not yet used by any prompt) |

Creates the `~/.contextd/` directory layout, writes a default `config.toml` and `docker-compose.yml` if not already present, copies the three default prompt templates (`summarise.md`, `relate.md`, `translate.md`) into `~/.contextd/prompts/`, and checks for required env vars and Docker on PATH.

```bash
contextd init
```

Run once on a fresh install. Safe to re-run — existing files are not overwritten.

---

## `contextd up`

**Synopsis:** `contextd up`

Starts the storage backend container for the configured backend (default: `neo4j`), then applies pending migrations. Requires Docker on PATH.

```bash
# Start the Neo4j backend container + indexer daemon
contextd up
```

The backend is Neo4j, declared by `[storage] backend = "neo4j"` in `config.toml`. The matching docker-compose profile (`--profile neo4j`) is activated; the container binds port 7687.

---

## `contextd down`

**Synopsis:** `contextd down`

Stops the active storage backend container.

```bash
contextd down
```

---

## `contextd status`

**Synopsis:** `contextd status`

Reports the configured backend name and lists registered corpora from `~/.contextd/corpora/*.toml`.

```bash
contextd status
# backend: neo4j
# corpora: 2 registered
#   - my-notes
#   - project-docs
```

---

## `contextd add-corpus`

**Synopsis:** `contextd add-corpus PATH [--name NAME] [--granularity {file,section}] [--from TEMPLATE]`

| Argument / Flag | Default | Description |
|---|---|---|
| `PATH` | required | Directory to index (must exist) |
| `--name` | directory basename | Corpus identifier used in all subsequent commands |
| `--granularity` | `file` | Index whole files (`file`) or promote headings to Section nodes (`section`) |
| `--from TEMPLATE` | none | Path to a corpus TOML template; only `corpus.root` and `corpus.name` are overridden from `PATH`/`--name` |

Registers a corpus by writing `~/.contextd/corpora/<name>.toml`. Prints a warning and exits early if the corpus is already registered.

**Basic registration:**

```bash
contextd add-corpus ~/notes --name my-notes
contextd add-corpus ~/notes --name my-notes --granularity section
```

**From a template** (copies the full adapter config — ontology aliases, prompt overrides, per-corpus MCP tools):

```bash
contextd add-corpus /path/to/project \
  --name project-docs \
  --from /path/to/project/.contextd/corpus.toml
```

When `--from` is provided, relative paths in the template (`ontology.json`, `prompts/summary.md`, any `[mcp.tools]` entries) are rewritten to absolute paths anchored at the template's directory. The `--granularity` flag is ignored when `--from` is used; the template's `[corpus] granularity` value is used instead.

---

## `contextd list-corpora`

**Synopsis:** `contextd list-corpora`

Lists all registered corpus names and their TOML paths.

```bash
contextd list-corpora
# - my-notes (/home/user/.contextd/corpora/my-notes.toml)
# - project-docs (/home/user/.contextd/corpora/project-docs.toml)
```

---

## `contextd index`

**Synopsis:** `contextd index CORPUS_NAME [--bootstrap] [--incremental] [--estimate-only]`

| Argument / Flag | Default | Description |
|---|---|---|
| `CORPUS_NAME` | required | Name of a registered corpus |
| `--bootstrap` | off | Full re-index from scratch |
| `--incremental` | off | Re-index only changed files (not yet implemented — reports a warning) |
| `--estimate-only` | off | Count files and estimate token spend without calling any provider |

Exactly one of `--bootstrap` or `--incremental` is required (unless `--estimate-only` is passed alone).

**Bootstrap a corpus:**

```bash
contextd index my-notes --bootstrap
```

**Cost preview (no provider calls, no graph writes):**

```bash
contextd index my-notes --estimate-only
# found 342 files in corpus 'my-notes'
# ~85500 input tokens projected (2 call types per file)
```

The token estimate is based on UTF-8 character count ÷ 4 (rough heuristic). The "2 call types per file" are summarisation and relationship inference.

---

## `contextd ask`

**Synopsis:** `contextd ask QUESTION [--corpus CORPUS_NAME]`

| Argument / Flag | Default | Description |
|---|---|---|
| `QUESTION` | required | Natural-language question |
| `--corpus` | none | Restrict the generated Cypher to a single corpus via `WHERE n.corpus = $corpus` |

Translates `QUESTION` to a Cypher query via `QueryTranslator` (uses the Gemini API), prints the generated Cypher, executes it against the backend, and prints the results as JSON.

```bash
contextd ask "which files reference the auth module?"
contextd ask "what are the riskiest gap entries?" --corpus project-docs
```

Translation failures and backend errors are rendered as `Error: ...` messages rather than Python tracebacks.

---

## `contextd logs`

**Synopsis:** `contextd logs [--follow]`

| Flag | Default | Description |
|---|---|---|
| `--follow` | off | Tail the log continuously (Ctrl-C to stop) |

Reads `~/.contextd/logs/contextd.log` (structured JSON). With `--follow`, runs `tail -f` on the file. Ctrl-C exits cleanly.

```bash
contextd logs
contextd logs --follow
```

---

## `contextd costs`

**Synopsis:** `contextd costs [--since DATE]`

| Flag | Default | Description |
|---|---|---|
| `--since` | none | `YYYY-MM-DD` lower bound (inclusive) |

Aggregates token spend from `~/.contextd/state/session-log/` by provider.

```bash
contextd costs
contextd costs --since 2026-04-01
# gemini: input=12540 output=3820
# voyage: input=88400 output=0
```
