# MCP Reference

The Contextd MCP server exposes the graph store to Claude Desktop, Cursor, and any other MCP-speaking client via stdio JSON-RPC.

---

## Transport

The server speaks the [Model Context Protocol](https://modelcontextprotocol.io/) over stdio. It is registered as the `contextd-mcp` console script and launched by the MCP client — Claude Desktop or Cursor spawns it as a subprocess, connects over stdin/stdout, and keeps it alive for the session. The server connects to the storage backend (Neo4j) over Bolt at startup and holds the connection until the client disconnects.

The server source lives at `contextd/mcp_server.py`. Tool implementations are in `contextd/mcp/tools.py` (generic tools) and `contextd/mcp/corpus_tools.py` (per-corpus tools).

---

## Client registration

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "contextd": {
      "command": "contextd-mcp"
    }
  }
}
```

If `contextd-mcp` is not on the system PATH (e.g. installed in a venv), use the absolute path:

```json
{
  "mcpServers": {
    "contextd": {
      "command": "/path/to/.venv/bin/contextd-mcp"
    }
  }
}
```

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "contextd": {
      "command": "contextd-mcp"
    }
  }
}
```

---

## Read-only safety

All tools — generic and per-corpus — are read-only. The guard is implemented in `contextd/mcp/readonly_guard.py` as `assert_read_only(cypher)`. It raises `ReadOnlyGuardError` on any Cypher containing the keywords:

```
CREATE  MERGE  DELETE  SET  REMOVE  DROP  DETACH  FOREACH
```

A negative lookbehind `(?<![.\w])` prevents false positives on dotted property access (e.g. `RETURN n.set AS prop` does not trigger the `SET` match).

For per-corpus tools, the guard runs **twice**: once at server startup when the Cypher file is loaded, and once again at dispatch time (defence in depth). A write-containing Cypher file causes the tool to be skipped at load time with a warning logged to stderr.

---

## Generic tools (19)

These are always present regardless of which corpora are registered.

### `describe_project`

Top-N `File` nodes by inbound-citation count with their summaries. Useful as a session-start primer.

| Input | Type | Required | Default |
|---|---|---|---|
| `corpus` | string | no | all corpora |
| `n` | integer | no | 40 |

Returns: array of `{path, name, summary, key_points, inbound}` objects ordered by `inbound` descending.

---

### `search`

Hybrid search over node summaries: vector (embedding) similarity and full-text (BM25) results fused by reciprocal rank fusion (RRF). RRF combines the two rankers by rank position rather than by raw score, which sidesteps the fact that cosine similarity and BM25 are not comparable as numbers.

| Input | Type | Required | Default |
|---|---|---|---|
| `query` | string | yes | — |
| `kind` | string | no | `"File"` |
| `limit` | integer | no | 20 |
| `mode` | string | no | server config (`hybrid`) |

`mode` is one of `hybrid`, `fulltext`, or `vector`. The server degrades to full-text automatically when no embedder is configured, the `kind` has no vector index (only `File` and `Section` do), or the embedding call fails; `mode = "vector"` on a non-vector-capable kind returns an empty result rather than a silent lexical fallback. The RRF constant, candidate depth, and per-modality weights are set server-side via the `[search]` config block.

The full-text property searched is resolved per label: `File` and `Section` are searched on `summary`, while the entity kinds `Artifact`, `Pattern`, and `Risk` are searched on `description` and `Ticket` on `title` (the content fields the indexer populates and migrations `_0001` / `_0006` index). Entity kinds have no vector index, so they always run full-text only.

Returns: array of matching node objects (the raw embedding vector is stripped). In `hybrid` / `vector` mode `score` is an RRF fused score; in `fulltext` mode it is the backend's raw relevance score — the two are not comparable across modes.

---

### `related`

Outbound and inbound traversal within N hops from a named node.

| Input | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `node_id` | string | yes | — | matched against `path`, `id`, or `name` |
| `depth` | integer | no | 2 | 1–5 inclusive |

Returns: array of `{path, id, name, summary}` for up to 50 distinct neighbours. Depth is clamped to [1, 5] both in the JSON schema and in the function body.

---

### `inbound`

What nodes cite the target node?

| Input | Type | Required |
|---|---|---|
| `node_id` | string | yes |

Returns: array of `{path, id, name, edge_type}` for all inbound neighbours.

---

### `outbound`

What nodes does the target node cite?

| Input | Type | Required |
|---|---|---|
| `node_id` | string | yes |

Returns: array of `{path, id, name, edge_type}` for all outbound neighbours.

---

### `get_file_summary`

Summary and key points for a single file by its path.

| Input | Type | Required |
|---|---|---|
| `path` | string | yes |

Returns: `{summary, key_points}` or `null` if the file is not indexed.

---

### `get_node`

Full labels and properties for a single node, matched by `path`, `id`, or `name` (the raw embedding vector is stripped). The entity-aware counterpart to `get_file_summary`: it reads any node type, so a `Ticket`, `Artifact`, or `Pattern` surfaced by `search` can be read in full.

| Input | Type | Required |
|---|---|---|
| `node_id` | string | yes |

Returns: `{labels, ...properties}` or `null` if no node matches.

---

### `section_tree`

Hierarchical outline of a file — section-granular corpora only.

| Input | Type | Required |
|---|---|---|
| `file_path` | string | yes |

Returns: array of `{id, title, level, ordinal, summary}` ordered by level then ordinal.

---

### `explain_relationship`

Every direct edge between two nodes, with provenance. Matches both endpoints by `path`/`id`/`name` and returns each edge in either direction.

| Input | Type | Required |
|---|---|---|
| `source` | string | yes |
| `target` | string | yes |

Returns: array of `{source, target, edge_type, outbound, origin, confidence, reason}`. `outbound` is `true` when the edge runs source→target; `origin` is `inferred` / `structural` / `manual`; `confidence` and `reason` are the properties inferred edges carry.

---

### `ticket_dossier`

A ticket's whole neighborhood in one call — collapses the multi-hop manual traversal the start-of-task workflow otherwise needs.

| Input | Type | Required |
|---|---|---|
| `ticket_id` | string | yes |

Returns: `{ticket, found, properties, neighbors}`, where each neighbor is `{edge_type, direction, labels, node, summary, title}`. `found` is `false` (and `neighbors` empty) when no such ticket exists.

---

### `find_reusable`

Reusable `Artifact` nodes ranked by full-text relevance to the query. Serves the "check for an existing artifact before creating a new one" discipline.

| Input | Type | Required | Default |
|---|---|---|---|
| `query` | string | yes | — |
| `limit` | integer | no | 20 |

Returns: array of matching `Artifact` nodes with `reusable = true`, embedding stripped, each with a relevance `score`. Requires entity content extraction to have populated `Artifact.description` / `reusable`.

---

### `list_entities`

Nodes of an entity `kind` with their properties, optionally filtered.

| Input | Type | Required | Default |
|---|---|---|---|
| `kind` | string | yes | — |
| `prop` | string | no | — |
| `value` | string | no | — |
| `corpus` | string | no | all |
| `limit` | integer | no | 50 |

`kind` must be a declared ontology node type and `prop` (when given) a declared property of that type; both are validated before interpolation, and `value`/`corpus` are bound as parameters. Returns: array of `{labels, ...properties}` (embedding stripped).

---

### `check_freshness`

Freshness-signalling edges (`SUPERSEDES` / `CONTRADICTS` / `NEEDS_UPDATE`) so a caller can judge whether a recalled hit is still current.

| Input | Type | Required |
|---|---|---|
| `node_id` | string | one of node_id / corpus |
| `corpus` | string | one of node_id / corpus |
| `limit` | integer | no (default 200) |

Scope by `node_id` (edges incident to one node) or `corpus` (every such edge with an endpoint in the corpus). Returns: array of `{source, target, edge_type, origin, confidence, reason}`. These edges are sparse, unvalidated inferences — an empty result means none were inferred, not that the node is definitively current.

---

### `find_contradictions`

`CONTRADICTS` edge pairs, optionally narrowed by topic.

| Input | Type | Required | Default |
|---|---|---|---|
| `topic` | string | no | — |
| `limit` | integer | no | 50 |

When `topic` is given, only pairs where either endpoint's summary contains it (case-insensitive) are returned. Returns: array of `{source, target, source_summary, target_summary, confidence, reason}`. Sparse in practice; often empty.

---

### `whats_new`

Nodes changed at or after an ISO-8601 timestamp, newest first — the changed source documents for catching up on an evolving corpus.

| Input | Type | Required | Default |
|---|---|---|---|
| `since` | string | yes | — |
| `corpus` | string | no | all |
| `limit` | integer | no | 50 |

Compares against the `updated` stamp the indexer writes on `File` / `Section` nodes. Returns: array of `{node, labels, summary, updated}`. Returns empty on a graph indexed before the `updated` stamp existed (re-bootstrap to backfill).

---

### `timeline`

Chronological view of nodes relevant to an anchor, plus the `SUPERSEDES` chains among them, to show how a decision evolved rather than a flat neighbor set.

| Input | Type | Required |
|---|---|---|
| `node_id` | string | one of node_id / topic |
| `topic` | string | one of node_id / topic |
| `limit` | integer | no (default 50) |

Anchor by `node_id` (the node and its direct neighbors) or `topic` (nodes whose summary contains it). Returns: `{nodes, supersedes}` — `nodes` ordered newest-first by `updated` (falling back to `inferred_at`), and `supersedes` the in-scope `SUPERSEDES` edges as newer→older pairs.

---

### `ask`

Natural-language question answered by translating it to Cypher (reusing the CLI `ask` translator) and running it.

| Input | Type | Required |
|---|---|---|
| `question` | string | yes |
| `corpus` | string | no |

Returns: `{cypher, rows}` — the generated Cypher and its result rows, so the caller sees what ran and always has the node-level path underneath the answer. Requires a configured inference provider (one LLM call per invocation); the translator applies the read-only guard to its own output. Errors (no provider, missing `prompts/translate` template) are surfaced as an `{"error": ...}` payload.

---

### `grep_corpus`

Regex search over corpus file *contents* on disk, for exact strings (a flag name, an id, a config key) that summaries paraphrase away. The graph stores summaries and metadata, not file bodies, so disk is the only source.

| Input | Type | Required | Default |
|---|---|---|---|
| `pattern` | string | yes | — |
| `corpus` | string | no | all |
| `limit` | integer | no | 100 |

Walks the corpus's declared include/exclude globs (reusing the indexer's file enumeration). Returns: array of `{corpus, path, line, text}` line matches, capped at `limit`.

---

### `query_graph`

Raw read-only Cypher escape hatch. The `assert_read_only` guard rejects any write keywords.

| Input | Type | Required |
|---|---|---|
| `cypher` | string | yes |

Returns: array of result rows as JSON objects. All tool call exceptions (guard rejections, backend errors) are returned as `{"error": "ExceptionType: message"}` rather than protocol exceptions.

---

## Per-corpus tools

Per-corpus tools are declared in a corpus TOML's `[mcp.tools]` section as `tool_name = "path/to/query.cypher"`. At server startup, the server scans `~/.contextd/corpora/*.toml` and registers each declared tool under the namespaced name `<corpus-name>.<tool-name>`.

**Tool naming:** the namespace separator is a dot. Generic tools never contain a dot; per-corpus tools always do. The dispatcher routes on this distinction.

**Input schema:** `$name` placeholders in the Cypher file become required string arguments. For example, a Cypher file containing `WHERE r.name = $registry_name` produces a tool with `required: ["registry_name"]`.

```toml
# ~/.contextd/corpora/my-corpus.toml
[mcp.tools]
find_recent = "tools/find_recent.cypher"
```

The Cypher in `tools/find_recent.cypher` can use `$param` placeholders; the tool descriptor will require those as string arguments.

**Error handling at load time (non-fatal):**

- Malformed corpus TOML → warning to stderr; corpus skipped.
- Missing Cypher file → warning to stderr; tool skipped.
- Write-containing Cypher → warning to stderr (logged as `SECURITY:`); tool skipped.

**Known limitation:** `extract_placeholders` uses a simple regex `\$([a-zA-Z_][a-zA-Z0-9_]*)`. String literals containing `$identifier`-shaped tokens (e.g. `WHERE n.label CONTAINS "pending$status"`) will produce a spurious `status` placeholder. In practice Cypher string literals seldom embed dollar-prefixed identifiers.

---

## Tool result shape

All tools return results as MCP `TextContent` with a `"text"` field containing a JSON-serialised payload:

```json
[{"type": "text", "text": "[{\"path\": \"docs/spec.md\", ...}]"}]
```

Non-serialisable values (datetimes, Paths) fall back to `str()`.
