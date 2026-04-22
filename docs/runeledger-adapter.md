# Runeledger-PRD Adapter

The `examples/runeledger-prd/` directory is a Contextd adapter that configures the indexer to treat the Runeledger PRD as a section-granular knowledge corpus. Every `§N.M` subheading becomes a `Section` node; markdown-link anchors resolve to typed `Section → Section` edges at inference time.

For Runeledger-specific architecture and vocabulary, see the Runeledger project's own `CLAUDE.md` at `/home/giuseppe/src/games/runeledger/CLAUDE.md`.

---

## Setup

```bash
# Register the corpus using the adapter as a template
contextd add-corpus /path/to/runeledger \
  --name runeledger-prd \
  --from examples/runeledger-prd/corpus.toml

# Bootstrap the index
contextd index runeledger-prd --bootstrap
```

`--from examples/runeledger-prd/corpus.toml` copies the full adapter config into `~/.contextd/corpora/runeledger-prd.toml`. All relative paths in the template are rewritten to absolute paths anchored at the template directory, so they continue to resolve regardless of where the command is run.

The `examples/runeledger-prd/README.md` contains the same setup steps.

---

## Files in `examples/runeledger-prd/`

### `corpus.toml`

The root adapter config. Key sections:

```toml
[corpus]
name        = "runeledger-prd"
root        = "/home/giuseppe/src/games/runeledger"
include     = ["docs/prd/**/*.md", "mods/base/**/*.lua", "prd.md", "CLAUDE.md"]
exclude     = ["docs/prd/_audit-methodology.md"]
granularity = "section"
heading_min_level = 2
heading_max_level = 4
```

- `granularity = "section"` routes all `.md` files through the section-level pipeline. `.lua` files have no markdown headings and are handled file-granularly within the same bootstrap run.
- `heading_min_level = 2` / `heading_max_level = 4` restricts which headings become `Section` nodes (H2–H4 only).

```toml
[ontology.aliases]
Registry = "Pattern"
FRRow    = "Ticket"
LuaFile  = "File"
GapEntry = "Risk"
```

Maps Runeledger domain names to canonical Contextd node types. The inferrer uses these aliases when writing nodes; Cypher queries against base types (`Pattern`, `Ticket`, etc.) match the aliased nodes transparently.

```toml
[ontology]
overrides = "ontology.json"
```

Points to `examples/runeledger-prd/ontology.json` for edge-label aliases (see below).

```toml
[mcp.tools]
four_surface              = "tools/four_surface.cypher"
find_dangling_registrations = "tools/dangling.cypher"
audit_stale_shas          = "tools/stale_shas.cypher"
```

Registers three per-corpus MCP tools. After `contextd-mcp` starts, these appear as `runeledger-prd.four_surface`, `runeledger-prd.find_dangling_registrations`, and `runeledger-prd.audit_stale_shas`.

```toml
[summarization]
prompt_override = "prompts/summary.md"
```

Overrides the default summarise template with a Runeledger-specific prompt.

---

### `ontology.json`

Edge-label aliases applied on top of the base ontology. Each alias maps a Runeledger edge name to a canonical Contextd edge type:

```json
{
  "edge_label_aliases": {
    "CITES":      "REFERENCES",
    "CONSUMES":   "USES",
    "SCHEMA_FOR": "DOCUMENTS",
    "REGISTERS":  "DOCUMENTS",
    "CLOSES_GAP": "SUPERSEDES"
  }
}
```

Both `SCHEMA_FOR` and `REGISTERS` alias to `DOCUMENTS` — the distinction lives only in the Cypher queries that pattern-match on the relationship direction and endpoint types.

---

### `prompts/summary.md`

A Runeledger-specific summarisation prompt. It instructs the inferrer to:

- Output JSON with `summary`, `key_points`, and `entities_mentioned` fields.
- Emphasise subsystem placement (economy / worldgen / magic / diplomacy), inbound dependencies, and pinned coefficients.
- Surface `§N.M` anchors, `register_*()` identifiers, `FR-XXX-NNN` rows, and commodity-class names in `entities_mentioned`.

The prompt is injected via `[summarization] prompt_override` and overrides the default `~/.contextd/prompts/summarise.md` for this corpus only.

---

### `tools/four_surface.cypher`

Returns the four "surfaces" of a named registry: the consumer prose section, the modding-API schema section, the Lua implementation file, and the FR row.

```cypher
MATCH (r:Pattern {name: $registry_name})
OPTIONAL MATCH (r)<-[:USES]-(consumer:Section)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(schema:Section)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(lua:File)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(fr:Ticket)
RETURN r.name AS registry, consumer.id AS consumer_section,
       schema.id AS schema_section, lua.path AS lua_file,
       fr.id AS fr_row;
```

MCP tool: `runeledger-prd.four_surface(registry_name: string)`

---

### `tools/dangling.cypher`

Finds registries missing any of the four surfaces.

```cypher
MATCH (r:Pattern)
OPTIONAL MATCH (r)<-[:USES]-(consumer:Section)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(schema:Section)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(lua:File)
OPTIONAL MATCH (r)<-[:DOCUMENTS]-(fr:Ticket)
WITH r, consumer, schema, lua, fr
WHERE consumer IS NULL OR schema IS NULL OR lua IS NULL OR fr IS NULL
RETURN r.name AS registry,
       consumer IS NOT NULL AS has_consumer,
       schema IS NOT NULL AS has_schema,
       lua IS NOT NULL AS has_lua,
       fr IS NOT NULL AS has_fr;
```

MCP tool: `runeledger-prd.find_dangling_registrations()` (no required args)

---

### `tools/stale_shas.cypher`

Returns gap-log entries that still carry `SHA=pending` — i.e., gaps not yet closed by a commit.

```cypher
MATCH (g:Risk)
WHERE g.description CONTAINS "SHA=pending"
RETURN g.description AS entry, g.severity AS severity
ORDER BY g.severity;
```

MCP tool: `runeledger-prd.audit_stale_shas()` (no required args)

---

## How the adapter wires

| Config surface | What it does |
|---|---|
| `[corpus] granularity = "section"` | Each `§N.M` heading becomes a `Section` node; markdown anchors resolve to `Section → Section` edges |
| `[ontology.aliases]` | Domain names (`Registry`, `FRRow`, `LuaFile`, `GapEntry`) map to canonical types at write time |
| `[ontology] overrides = "ontology.json"` | Domain edge names (`CITES`, `CONSUMES`, etc.) map to canonical edge types at inference time |
| `[summarization] prompt_override` | Corpus-specific summary instructions replace the default prompt |
| `[mcp.tools]` | Three Cypher files become callable MCP tools namespaced as `runeledger-prd.<tool>` |
