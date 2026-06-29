# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed

- Memgraph storage backend and the `gqlalchemy` dependency. Neo4j Community is
  now the sole storage backend. The `GraphStore` ABC, the storage factory, and
  the abstraction-invariant CI grep are retained so a second backend can be
  added later without recoupling consumers.

### Changed

- Default embedding model flipped `voyage-3` → `voyage-4-large` (still 1024-dim,
  cosine, drop-in against the existing vector indexes). Higher Voyage free-tier
  quota and 32k-token context per input (vs `voyage-3`'s 8k).
- `[embedding] chunk_tokens` default raised `8000` → `32000` to match
  `voyage-4-large`'s larger per-input context window — the chunker now only
  splits genuinely very long inputs. Users who explicitly override to
  `voyage-3` / `voyage-3-large` / `voyage-code-3` (still supported) should
  lower `chunk_tokens` back to `8000` in their corpus TOML.

## [0.1.0] — 2026-04-21

Initial alpha release.

### Added

- Hybrid graph + vector storage with pluggable Neo4j Community / Memgraph backends
  (Bolt driver; vector + full-text indexes; forward-only migration runner)
- Google Gemini inference provider (`gemma-4-31b-it` default, configurable via
  `[providers.gemini] model_summary|model_inference|model_translation`).
  Translation calls request `thinkingLevel = HIGH` via the Gemini API's
  `generationConfig.thinkingConfig`; summary and inference calls do not.
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
- Minimal-notes example adapter (`examples/minimal-notes/`)
- PowerShell wrappers for Windows 11 + WSL2 users

[0.1.0]: https://github.com/giuseppecardenas/contextd/releases/tag/v0.1.0
