# Runeledger-PRD Contextd adapter

Configures Contextd to index the Runeledger PRD (markdown + Lua + the book-style index in `prd.md`) at **section granularity**. Every `§N.M` subheading becomes a `Section` node; markdown links with anchor fragments (`[§6.14.9](03f-economy-feudal.md#614-...)`) resolve to typed `Section → Section` edges at parse time.

## Setup

```bash
# 1. Clone or copy the contextd repo so you have the adapter files:
#    git clone git@github.com:giuseppecardenas/contextd.git
#    cd contextd

# 2. Register the Runeledger corpus using this adapter's TOML as the template:
contextd add-corpus /path/to/runeledger \
  --name runeledger-prd \
  --from examples/runeledger-prd/corpus.toml

# 3. Bootstrap the index:
contextd index runeledger-prd --bootstrap
```

`--from examples/runeledger-prd/corpus.toml` copies the full adapter config (ontology aliases, custom summary prompt, MCP tools) into `~/.contextd/corpora/runeledger-prd.toml`. All relative paths in the template (e.g. `ontology.json`, `prompts/summary.md`, `tools/*.cypher`) are rewritten to absolute paths anchored at the template's directory, so they continue to resolve regardless of where you run the command from.

## Recommended backend

Neo4j Community — the default `storage.backend = "neo4j"` in `~/.contextd/config.toml`. Memgraph also works (`backend = "memgraph"`). Both require Docker (Docker Desktop with WSL2 integration, or any Docker engine).

## Exposed MCP tools

This adapter registers three corpus-scoped Cypher tools. When Claude Desktop / Cursor connects to the Contextd MCP server, these appear alongside the built-in generic tools:

- `runeledger-prd.four_surface(registry_name)` — consumer prose + modding-API schema + Lua file + FR row for a named registry
- `runeledger-prd.find_dangling_registrations()` — registries missing any of the four surfaces
- `runeledger-prd.audit_stale_shas()` — gap-log entries with `SHA=pending`

The `runeledger-prd.` prefix namespaces them against tools from other corpora. The MCP server dispatches corpus tools exclusively by their fully-qualified `<corpus>.<tool>` name; short-name calls are not supported.

## How the adapter wires

- **Section-granular indexing** (`[corpus] granularity = "section"`) — each `§N.M` subheading in the PRD becomes a `Section` node; markdown-link anchors resolve to typed `Section → Section` edges.
- **Domain → canonical node-type aliases** (`[ontology.aliases]` in `corpus.toml`) — Runeledger's vocabulary (`Registry`, `FRRow`, `LuaFile`, `GapEntry`) maps to Contextd's canonical node types (`Pattern`, `Ticket`, `File`, `Risk`) at inference time.
- **Domain → canonical edge-type aliases** (`[ontology] overrides = "ontology.json"`) — Runeledger's edge vocabulary (`CITES`, `CONSUMES`, `SCHEMA_FOR`, `REGISTERS`, `CLOSES_GAP`) maps to canonical edge types (`REFERENCES`, `USES`, `DOCUMENTS`, `DOCUMENTS`, `SUPERSEDES`). All aliases are validated against the base ontology at index time; unrecognised types are silently discarded.
- **Custom summary prompt** (`[summarization] prompt_override = "prompts/summary.md"`) — forces the summariser to emphasise subsystem placement (economy / worldgen / magic / diplomacy) and pin concrete coefficients.
- **Per-corpus MCP tools** (`[mcp.tools]`) — the three Cypher files in `tools/` become callable MCP tools at server startup, namespaced as `runeledger-prd.<tool>`.
- **Lua files** — `mods/base/**/*.lua` files are treated file-granularly in this otherwise-section-granular corpus (they have no markdown headings); each Lua file gets a single file-level summary and participates in `REGISTERS` → `DOCUMENTS` inferred edges pointing at `Pattern` nodes.
