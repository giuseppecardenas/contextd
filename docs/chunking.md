# Chunking, retrieval units and topics

Contextd indexes a corpus at two granularities that serve two different jobs:

| Band | Unit | Size | Job |
|---|---|---|---|
| Extraction | `Section` (heading-bounded) or `File` | ~500–2000 tokens | LLM summarisation and relationship inference |
| Retrieval | `Chunk` | ~200–1000 tokens (configurable) | Vector + full-text search, evidence for citations |

`Chunk` nodes are **retrieval-only**: they are embedded and full-text indexed but never summarised or related by the LLM, so adding or re-tuning chunk profiles costs embedding calls, not inference calls. Every chunk hangs off its parent through `CONTAINS {origin: "structural"}` and is ordered by `NEXT_SIBLING`.

## Configuration (`[chunking]` in the corpus TOML)

```toml
[chunking]
enabled   = true
tokenizer = "auto"               # "auto" | "voyage" | "tiktoken" | "words"
prefix    = "breadcrumb"         # "none" | "breadcrumb" | "section_summary" | "llm"
augment_fulltext = ["key_points"]  # any of key_points | entities_mentioned | questions

[[chunking.profiles]]
name       = "fine"              # [a-z][a-z0-9_]*; part of every chunk id
strategy   = "structural"        # see the strategy table
max_tokens = 256
min_tokens = 48                  # smaller pieces are forward-merged
overlap    = 0.0                 # fraction of max_tokens carried from the previous chunk
weight     = 1.0                 # RRF scale for this profile's rankers at query time

[[chunking.profiles]]
name       = "coarse"
max_tokens = 1024
min_tokens = 200
overlap    = 0.1

[chunking.blocks]                # structural strategy only
protect_code_fences = true       # never split inside a fence; oversize fences re-fenced per group
table_mode          = "rows_with_header"   # | "whole" | "prose"
sentence_fallback   = "recursive"          # | "window"  (for an oversize paragraph)
max_fence_tokens    = 512        # optional; defaults to the profile's max_tokens

[chunking.suffix_overrides]      # per file type, applied to every profile
".py"  = { strategy = "code" }
".txt" = { strategy = "recursive", max_tokens = 200 }
```

### Tokenizer

`auto` follows the embedder: Voyage's own tokenizer (the HuggingFace model Voyage publishes, fetched once from the Hub) when `providers.embedding = "voyage"`, `tiktoken` (`o200k_base`, extra `contextd[tiktoken]`) when installed for anything else, and the whitespace-word estimate otherwise. Explicit choices degrade to `words` with a warning rather than aborting a bootstrap. The tokenizer id is part of the chunk fingerprint, so switching it re-chunks the corpus.

### Strategies

| `strategy` | What it does | Needs |
|---|---|---|
| `structural` (default) | Packs markdown blocks (paragraphs, lists, fences, tables, quotes) up to `max_tokens`, forward-merges under `min_tokens`. Fences are never split mid-fence; tables split by row with the header repeated; lists split between items; oversize prose falls back to the recursive cascade or a token window. | — |
| `window` | Fixed-token sliding window with whitespace snapping (GraphRAG / LightRAG baseline). | — |
| `recursive` | Separator cascade (`\n# `, "```", `---`, blank line, newline, sentence, space) à la LangChain, then packed. | — |
| `sentence_window` | One sentence per chunk; `window` neighbours are attached at query time through `NEXT_SIBLING`, not stored twice. Code/table blocks stay whole. | — |
| `semantic` | Embeds buffered sentence groups and cuts where consecutive cosine distance exceeds a `percentile` / `stddev` / `iqr` / `gradient` threshold (`threshold_type`, `threshold`, `buffer_size`). One embedding per sentence at index time. | embedding provider |
| `late` | `structural` boundaries, but each chunk's vector is mean-pooled from one forward pass over the whole parent (Jina late chunking). | `providers.embedding = "local_hf"` (extra `contextd[late]`) |
| `propositions` | One LLM call per parent yields self-contained statements (Dense X); each is a chunk. Falls back to `structural` on provider failure. | inference provider |
| `code` | tree-sitter definitions with line-group splitting (`chunk_lines_overlap`) for oversize nodes. | extra `contextd[code]` |

Strategies whose provider or extra is missing fail at **pipeline construction** (`ChunkingConfigError`, surfaced by `contextd index`), never mid-bootstrap.

### Prefix and augmentation

The prefix is prepended to the chunk text for embedding and full-text indexing (never mixed into the stored `text`):

- `breadcrumb` — `Document title > Section > Subsection` (the file path when a file has no headings). Zero cost; the default.
- `section_summary` — breadcrumb plus the parent's LLM summary.
- `llm` — Anthropic-style contextual retrieval: one `summary`-call-site request per parent returns a situating sentence for each chunk (batched, so cost is one call per parent, not per chunk). Falls back to the breadcrumb.

`augment_fulltext` copies the parent's `key_points` / `entities_mentioned` onto the chunk's full-text-only `keywords` field; `questions` adds one LLM call per parent generating RAGFlow-style questions each chunk answers.

## Fingerprints and re-chunking

Chunks are a pure function of (chunking config, tokenizer, parent content). Each Section/File stores `chunk_fingerprint = sha256(config fingerprint : parent hash)`; a parent whose stored value equals the current one is skipped. That one comparison gives resume-after-crash, incremental re-index, the daemon sweep, and "config changed" identical semantics. Per parent the phase deletes old chunks, batch-upserts the new rows, writes the structural edges and stamps the fingerprint last, so a crash leaves the parent un-stamped and it is redone.

The corpus-level fingerprint is stored on the `Corpus` node; the daemon re-chunks a corpus at startup when it differs from the running config. `contextd index <corpus> --refresh chunks` drops every chunk and marker (embedding cost only); `--estimate-only` dry-runs the chunkers and reports chunk and embedding-token counts per profile.

## Query side

`search` (MCP) defaults to `kind = "Chunk"`: one vector + one full-text ranker per requested profile, fused by weighted RRF, then collapsed small-to-big:

- `return_unit = "chunk"` — every hit as-is;
- `"section"` / `"file"` — always the enclosing unit;
- `"auto"` (default) — the parent when at least `auto_merge_threshold` (0.5) of its chunks in the best-covered profile were retrieved, else the chunk (the LlamaIndex / Haystack auto-merging rule).

Every row carries `evidence` (`chunk_id`, `start_line`, `end_line`, `text`, `context_before`, `context_after`). `expand_chunk` returns a chunk with its neighbours and the parent summary. The `[search]` block sets `chunk_profiles`, `return_unit`, `auto_merge_threshold`, `window`, `max_evidence_chars` and `over_fetch_factor`. See [mcp.md](mcp.md).

## Topics (`[topics]`)

RAPTOR-style cross-document clustering: Section (or File) embeddings are L2-normalised, PCA-reduced and soft-clustered with a Gaussian mixture chosen by BIC (numpy only); each cluster becomes a `Topic` node whose title and summary the LLM writes from the members' summaries, with `BELONGS_TO {probability}` membership edges. Higher layers cluster the topic embeddings until `max_layers` or a single cluster.

```toml
[topics]
enabled            = true   # off by default
source             = "section"   # | "file"
max_layers         = 3
min_members        = 3
soft_threshold     = 0.1    # responsibility needed for multi-membership
max_cluster_tokens = 3500   # clusters above this are re-clustered
pca_dims           = 32
seed               = 0
```

Topics rebuild when the member set or their summaries change (`Corpus.topic_input_fingerprint`); incremental passes flag the corpus dirty and the daemon re-clusters on `[indexer] topics_recluster_interval_seconds` (default 3600). `--refresh topics` forces a rebuild. Query them with the `topics` MCP tool or `search(kind="Topic")`.

## Benchmarking

`contextd bench <corpus>` scores a labelled query file (`<root>/.contextd/bench.toml`) against `search` — recall@k, precision@k, MRR, line-level IoU and latency — per profile configuration, and `--compare` diffs two saved runs. See [cli.md](cli.md).
