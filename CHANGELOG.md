# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-04-21

Initial alpha release.

### Added

- Hybrid graph + vector storage with pluggable Neo4j Community / Memgraph backends
  (Bolt driver; vector + full-text indexes; forward-only migration runner)
- Google Gemini inference provider (`gemini-flash-latest` default, configurable)
  with retry, BLOCK_NONE safety settings, and append-only cost log
- Voyage AI embedding provider (`voyage-3`, 1024-dim) with batched embedding and retry
- Five-phase bootstrap pipeline for file-granular corpora (`enumerate` → `embed` →
  `summarise` → `relate` → `close`); seven-phase minimum for section-granular corpora
  (`enumerate_sections` → `gc_sections` → `embed_sections` → `summarise_sections` →
  `relate_sections` → `derive_file_level` → `close`, with additional file-granular
  passes for non-markdown files in the same corpus)
- Interruption recovery via `CheckpointStore` (per-corpus JSON resume state)
- Filesystem watcher (`CorpusWatcher`) with debounced incremental re-index
- MCP server with eight baseline tools over stdio transport (`describe_project`,
  `search`, `related`, `inbound`, `outbound`, `get_file_summary`, `query_graph`,
  `section_tree`), plus per-corpus Cypher tool registration (namespaced
  `<corpus>.<tool>`)
- CLI with commands: `init`, `up`, `down`, `status`, `add-corpus`, `list-corpora`,
  `index`, `ask`, `logs`, `costs`
- `contextd add-corpus --from TEMPLATE` registers a corpus from an authored TOML,
  with template-parent-relative path rewriting
- File-granular and section-granular indexing modes, opt-in per corpus via
  `[corpus] granularity = "section"`
- Per-corpus edge-label aliases via `[ontology] overrides = "<json>"`, allowing
  project-specific relationship vocabulary to map to ontology-canonical types
- Per-corpus prompt override via `[summarization] prompt_override`, replacing the
  default summarise template for a specific corpus
- Ontology-validated AI-inferred edges: unknown node or edge types are silently
  discarded at write time
- Natural-language-to-Cypher query translation (`contextd ask "<question>"`) via
  `QueryTranslator` + read-only guard
- Runeledger-PRD example adapter (`examples/runeledger-prd/`)
- Minimal-notes example adapter (`examples/minimal-notes/`)
- PowerShell wrappers for Windows 11 + WSL2 users

[0.1.0]: https://github.com/giuseppecardenas/contextd/releases/tag/v0.1.0
