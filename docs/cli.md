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

Stops the indexer daemon and the storage backend container (`docker compose stop`) without removing them. The container and its Docker data volumes are preserved, so the indexed knowledge graph survives; a later `contextd up` restarts the same container with its data intact.

```bash
contextd down
```

To permanently delete the indexed data, use `contextd reset`.

---

## `contextd reset`

**Synopsis:** `contextd reset`

Stops the indexer daemon and removes the storage backend container together with its Docker data volumes (`contextd_neo4j_data` / `contextd_neo4j_logs`) via `docker compose down --volumes`. This permanently deletes the entire knowledge graph: every node, edge, summary, and embedding contextd has indexed.

```bash
contextd reset
```

The command prints an explicit data-loss warning and then proceeds without an interactive prompt, so treat it as irreversible. The only way to recover the data is to re-index each corpus from source with `contextd index <corpus> --bootstrap`. Corpus registrations under `~/.contextd/corpora/` and `config.toml` are left untouched, so a subsequent `contextd up` brings up a fresh, empty backend ready for re-indexing. Use `contextd down` instead when you only want to stop the backend and keep the indexed data.

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

**Synopsis:** `contextd index CORPUS_NAME [--bootstrap] [--incremental] [--estimate-only] [--refresh SCOPE]`

| Argument / Flag | Default | Description |
|---|---|---|
| `CORPUS_NAME` | required | Name of a registered corpus |
| `--bootstrap` | off | Full (idempotent, resumable) index |
| `--incremental` | off | Re-index only changed files/sections (hash-gated), then re-chunk their parents |
| `--estimate-only` | off | Count files, estimate token spend, and dry-run the chunkers (per-profile chunk / embedding-token counts) without calling any provider |
| `--refresh SCOPE` | none | Wipe one layer before bootstrap: `inferred`, `lexical` (only the edges written by lexical extraction, rewritten through the current ontology/resolver — no LLM cost; use after a resolver, alias or `[[lexical.patterns]]` change), `summaries`, `llm`, `chunks` (retrieval chunks + fingerprints, embedding cost only), `topics`, or `all` |

The `[chunking]` / `[topics]` corpus-config blocks that drive the chunk and topic phases are documented in [chunking.md](chunking.md).

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

## `contextd remove-corpus`

**Synopsis:** `contextd remove-corpus CORPUS_NAME`

Unregisters a corpus and permanently deletes its indexed data. It performs three steps in order: `DETACH DELETE` of the corpus's `File` / `Section` / `Corpus` nodes from the graph (cascading to their edges), removal of the local per-corpus state files (the hasher index-state and the bootstrap checkpoint under `~/.contextd/state/`), and finally deletion of the corpus registration TOML. The corpus's source files on disk are never touched, so the corpus can be re-registered with `add-corpus` and re-indexed.

The graph delete runs before the registration TOML is removed, so if the backend is unreachable the command fails while the corpus is still registered and can be retried once the backend is up. This command prints a data-loss warning and proceeds without an interactive prompt.

```bash
contextd remove-corpus my-notes
```

Inference-target entity nodes (`Ticket`, `Artifact`, `Pattern`, `Risk`, and the other stub-able types) are not corpus-scoped, so they are deliberately left in place; run `contextd prune-entities` afterwards to reap any that are now orphaned.

---

## `contextd prune-entities`

**Synopsis:** `contextd prune-entities`

Deletes orphaned entity nodes across every corpus. The relate phase creates entity nodes as the targets of relationships from files; when every file that referenced an entity has been re-indexed away or removed (for example via `remove-corpus`), the entity can be left with no relationships at all. This command `DETACH DELETE`s exactly those zero-degree entity nodes.

The prunable labels are derived from the base ontology minus the structural labels `File`, `Section`, `Corpus`, and `Meta` (the shared `NON_ENTITY_LABELS` set), so structural nodes are never pruned even when edgeless: an isolated file is still a real file.

```bash
contextd prune-entities
# ✓ pruned 12 orphaned entities
```

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

---

## `contextd bench`

**Synopsis:** `contextd bench CORPUS [--queries PATH] [--profiles a,b]... [--return-unit UNIT] [--k N] [--expand none|units] [--graph-weight W] [--json PATH]`
**Synopsis:** `contextd bench --compare A.json B.json`

| Argument / Flag | Default | Description |
|---|---|---|
| `CORPUS` | required (unless `--compare`) | Name of a registered corpus |
| `--queries` | `<corpus root>/.contextd/bench.toml` | Bench spec file (TOML; YAML only if `pyyaml` happens to be installed) |
| `--profiles` | every profile in the graph | Comma-separated chunk profiles to query. Repeat the option to bench several configurations in one run — each value becomes one table row |
| `--return-unit` | `[search] return_unit` from `config.toml` | `chunk`, `section`, `file`, or `auto` (small-to-big collapse target) |
| `--k` | `5` | Top-k depth for recall/precision/IoU; a query's own `k` overrides it |
| `--expand` | `[search] expand` from `config.toml` | `none` or `units` — fuse Sections/Files linked to the top hits through shared entities with the direct hits (see `search` in [docs/mcp.md](mcp.md)); the table row is suffixed `+graph(W)` |
| `--graph-weight` | `[search] graph_weight` | RRF weight of the expanded rows relative to the direct hits (only with `--expand units`) |
| `--json` | none | Save every configuration's report (`{"reports": [...]}`) to this file |
| `--compare` | none | Diff two saved reports (signed per-metric delta, `B - A`) and exit without touching the backend |

Runs every query in the spec through the `search` MCP tool — with the same
`[search]` knobs the server uses (`mode`, `rrf_k`, `fetch_k`, weights,
`auto_merge_threshold`) — and scores the returned rows against the spec's
expectations:

| Metric | Meaning |
|---|---|
| recall@k | fraction of expected targets satisfied by the top-k rows |
| precision@k | fraction of the top-k rows that satisfy some expectation |
| MRR | 1 / rank of the first satisfying row, averaged over queries |
| line IoU | Chroma-style token IoU at line granularity, over the line sets of hits and expectations that carry `lines` (`—` when neither side does) |
| latency ms | mean wall-clock per `search` call |

A row satisfies an expectation when the paths match (either side may be
absolute) and, where the expectation narrows further, the section anchor
matches or the line ranges overlap. Neighbour context (`[search] window`) is
forced to 0 — it never changes a score and would only pad the latency.

The embedding provider is built the way `contextd ask` builds its providers;
if that fails (no key, unreachable endpoint) the run warns and continues
full-text only, so the report's `config.embedder` records which retrieval
path was measured.

**Spec format** (`bench.toml`):

```toml
[[queries]]
q = "what do the notes say about sourdough hydration"
expect = [
    { path = "note-3.md", anchor = "hydration", lines = [12, 30] },
    { path = "note-7.md" },
]
k = 5   # optional per-query override of --k

[[queries]]
q = "cooking"
expect = [{ path = "note-3.md" }, { path = "note-7.md" }]
```

`path` is required and compared as a suffix, so corpus-relative paths work.
`anchor` is the Section's GitHub-style heading slug (the part after `#` in a
Section id), compared modulo hyphen runs so `lod-1--lod-2` and `lod-1-lod-2`
name the same heading; `lines` is `[start, end)`, 0-based and end-exclusive like
`ChunkSpan`. Unknown keys, empty `expect` lists, and malformed ranges are
rejected with the file name in the error.

```bash
# One row per --profiles value
contextd bench notes --profiles fine --profiles coarse --profiles fine,coarse

# Measure chunk-level hits at depth 10 and keep the report
contextd bench notes --return-unit chunk --k 10 --json runs/chunk-k10.json

# Diff two saved runs
contextd bench --compare runs/before.json runs/after.json
```

`examples/minimal-notes/.contextd/bench.toml` is a working spec for the
example corpus (register it with `--from examples/minimal-notes/.contextd/corpus.toml`).
