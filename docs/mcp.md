# MCP Reference

The Contextd MCP server exposes the graph store to Claude Desktop, Cursor, and any other MCP-speaking client via stdio JSON-RPC.

---

## Transport

The server speaks the [Model Context Protocol](https://modelcontextprotocol.io/) over stdio. It is registered as the `contextd-mcp` console script and launched by the MCP client — Claude Desktop or Cursor spawns it as a subprocess, connects over stdin/stdout, and keeps it alive for the session. The server connects to the storage backend (Neo4j or Memgraph) over Bolt at startup and holds the connection until the client disconnects.

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

## Generic tools (8)

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

Full-text search over node summaries.

| Input | Type | Required | Default |
|---|---|---|---|
| `query` | string | yes | — |
| `kind` | string | no | `"File"` |
| `limit` | integer | no | 20 |

Returns: array of matching node objects. Vector-similarity fallback is not yet implemented; this is full-text only.

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

### `section_tree`

Hierarchical outline of a file — section-granular corpora only.

| Input | Type | Required |
|---|---|---|
| `file_path` | string | yes |

Returns: array of `{id, title, level, ordinal, summary}` ordered by level then ordinal.

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
