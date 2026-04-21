# Runeledger-PRD Contextd adapter

Configures Contextd to index the Runeledger PRD (markdown + Lua + the book-style index in `prd.md`) at **section granularity**. Every `§N.M` subheading becomes a `Section` node; markdown links with anchor fragments (`[§6.14.9](03f-economy-feudal.md#614-...)`) resolve to typed `Section → Section` edges at parse time.

## Setup

```bash
# From ~/src/contextd (the Contextd install):
contextd add-corpus /home/giuseppe/src/games/runeledger --name runeledger-prd
contextd index runeledger-prd --bootstrap
```

## Recommended backend

Neo4j Community (`[storage] backend = "neo4j"` in `~/.contextd/config.toml`). Neo4j is the default backend as of M11.8 and the recommended choice for all new corpora. Requires Docker (Docker Desktop with WSL2 integration, or any Docker engine). See Contextd's spec §13.4.

## Exposed MCP tools

- `four_surface(registry_name)` — consumer prose + modding-API schema + Lua file + FR row for a named registry
- `find_dangling_registrations()` — registries missing any of the four surfaces
- `audit_stale_shas()` — gap-log entries with `SHA=pending`
