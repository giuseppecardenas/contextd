# Contextd Feature Expansion: Competitive Analysis and Roadmap

**Date:** 2026-07-07
**Methodology:** Analyzed 34 local GraphRAG / knowledge graph projects on GitHub, spanning academic reference implementations to production-ready tools. Keywords used: knowledge graph, graphrag, context graph, agent memory, local-first RAG, code knowledge graph.

---

## Landscape Analysis: Local GraphRAG / Knowledge Graph Projects

### Projects Examined

| Stars | Project | Key Differentiator |
|---|---|---|
| 37,429 | HKUDS/LightRAG | Simple, fast GraphRAG with community detection, incremental updates, multimodal (RagAnything), reranker support |
| 34,241 | microsoft/graphrag | Community summarization, global/local/drift search modes, hierarchical community reports |
| 27,960 | DeusData/codebase-memory-mcp | Tree-sitter AST-based code knowledge graph, 158 languages, sub-ms queries, single C binary |
| 3,918 | circlemind-ai/fast-graphrag | PageRank-based graph exploration, adaptive retrieval, 6x cheaper than Microsoft GraphRAG |
| 3,817 | gusye1234/nano-graphrag | 1100-line hackable GraphRAG, portable backends (faiss, neo4j, ollama), async + typed |
| 598 | swarmclawai/swarmvault | LLM Wiki pattern, edge provenance tagging (extracted/inferred/ambiguous), contradiction detection, approval workflows |
| 515 | automataIA/graphrag-rs | Rust GraphRAG with WASM deployment, WebGPU acceleration, three deployment architectures |
| 244 | Eshaan-Nair/ArcRift | Browser extension syncing AI chat context to local IDE agents via SQLite knowledge graph |
| 230 | varun29ankuS/shodh-memory | LLM-free memory, Hebbian strengthening, exponential decay, spreading activation, 55ms latency, ROS2/Zenoh |
| 150 | devwhodevs/engraph | 5-lane hybrid search (semantic + BM25 + graph + reranker + temporal), Obsidian-native, section-level editing |
| 137 | Uranid/mnem | Git for AI knowledge, versioned graph (branch/diff/merge/rollback), deterministic ingest, WASM support |
| 66 | jaylfc/taosmd | Append-only provable memory, verifier checks facts against source, 97% Recall@5, runs on 8GB RAM |
| 56 | MihaiBuilds/memory-vault | Postgres + pgvector, hybrid search, knowledge graph, local LLM chat, web dashboard |
| 26 | ADVASYS/ragraph | Self-organizing graph with entity merging, agentic reasoning (LLM decides when to search/expand/summarize) |
| 26 | LeandroPG19/cuba-memorys | Bitemporal facts, Hebbian learning, spreading activation, PageRank, community detection, 25 MCP tools |
| 25 | xfloukiex-lab/magpie-search | Federated search across 5 sources with trust tiers (fact > reference > lead > stale) |
| 21 | mkupermann/throughline | Persistent long-term memory for Claude Code via PostgreSQL + pgvector + Streamlit |
| 18 | Hashevolution/James-RAG-Evol | Replayable RAG with append-only audit log, `reconstruct_graph_at(t)`, bitemporal validity windows |
| 13 | david-franz/ctx-sys | Code knowledge graph via tree-sitter + embeddings, keyword + vector + graph fusion via RRF |
| 11 | neo4j-labs/meta-knowledge-graph | Self-improving memory layer, lifecycle hooks capture sessions, LLM distills durable learnings, system prompt evolution |
| 9 | wzdavid/ThinkWiki | Documents/notes into local Markdown wiki with AI agents |
| 8 | zrg-team/memorall | Browser extension turning web reading into searchable knowledge graph |
| 7 | rupertgermann/open-recall | GraphRAG PKM with spaced repetition flashcards, entity web research, hybrid retrieval |
| 6 | 199-biotechnologies/engram | MCP memory server with BM25 + semantic + knowledge graph, temporal decay |
| 5 | Preciso-GR/preciso-graphrag | Agent-first GraphRAG, drop files and get reusable knowledge graphs |
| 4 | Rayen-Hamza/Klippy | Multimodal RAG (text + image + audio) with knowledge graph reasoning for sub-4B LLMs |
| 3 | chaoscypherinc/chaoscypher | Local-first GraphRAG with inspectable knowledge graph |
| 3 | IASolutionOrg/Cortex | Universal long-term memory for AI agents with vector + graph search |
| 2 | subzone/knowledge-master | Code knowledge graph with blast radius analysis, convention enforcement, MCP server |
| 2 | immutlex/immutlex | Immutable content-addressed docs, wikilink graph, 74 MCP tools, cognitive maintenance |
| 2 | n1x-technologies/n1x-cortex | Markdown vault/codebase into cited AI-queryable knowledge graph |
| 2 | stevepridemore/graph-memory | Personal knowledge graph for Claude with Neo4j + semantic search |
| 2 | bxw91/brainpalace | Vector Graph RAG for code and docs with persistent chat-session memory, AST chunking |
| 707 | ChristopherLyon/graphrag-workbench | Interactive 3D knowledge graph visualization with community detection |

---

## Unique Functionalities Found Across Projects

### 1. Versioned Knowledge (mnem)

**What it does:** Treats the knowledge graph like a git repository. Every write is committed, you can branch, diff, merge, or roll back any fact. Forgetting is first-class: revoke a fact and every retrieval path filters it out, with an audit trail preserved.

**Why it matters for contextd:** Currently, `--refresh` is destructive. There is no way to say "show me what the graph looked like before I re-indexed" or "undo the last inference run."

### 2. LLM-Free Memory with Cognitive Dynamics (shodh-memory)

**What it does:** Zero LLM calls for storage or retrieval. Uses local NER (TinyBERT), local embeddings (MiniLM), typed relation extraction via lexical cues, Hebbian strengthening (memories used frequently become easier to find), exponential-to-power-law decay (irrelevant memories fade), and spreading activation (recalling one thing surfaces related things). 55ms store latency.

**Why it matters for contextd:** contextd requires LLM calls for relationship inference. A deterministic, LLM-free fallback path for entity extraction and relation typing would make incremental updates nearly free and the daemon vastly faster.

### 3. Community Detection and Hierarchical Summarization (Microsoft GraphRAG, fast-graphrag)

**What it does:** Clusters the graph into communities (Leiden algorithm), generates per-community summaries, and answers "global" questions by synthesizing across community reports rather than doing point lookups. Fast-graphrag adds PageRank-based exploration for query routing.

**Why it matters for contextd:** contextd has no notion of communities. For a 500-file corpus, there is no way to ask "what are the major themes?" without reading every summary.

### 4. Append-Only Audit Log with Replayable State (SEKOS/James-RAG-Evol)

**What it does:** Every graph mutation lands in an append-only event log. `reconstruct_graph_at(t)` replays the graph to any historical point byte-identically. Contradictions are flagged deterministically, not probabilistically.

**Why it matters for contextd:** contextd has `SUPERSEDES` and `CONTRADICTS` edge types but no temporal replay. You cannot answer "what did the graph say about X on June 1?"

### 5. Federated Multi-Source Search with Trust Tiers (magpie-search)

**What it does:** Fans a single query across five sources (conversation history, local files, knowledge graph, vector store, live web), fuses results via RRF, and tags each result with a trust tier: `fact > reference > lead > stale`. Deduplicates across sources.

**Why it matters for contextd:** contextd searches only its own graph. There is no way to combine graph results with live file grep, web lookup, or external vector stores in a single ranked response.

### 6. Self-Improving Agent Memory with System Prompt Evolution (neo4j-labs/meta-knowledge-graph)

**What it does:** Lifecycle hooks capture every agent session. An LLM extraction loop distills durable learnings. Accumulated learnings periodically fold into the agent's system prompt, with version history preserved. The graph improves itself over time.

**Why it matters for contextd:** contextd is passive: it indexes what you write. It never learns from how the graph is queried or what answers are useful.

### 7. Code-Level AST Knowledge Graphs (codebase-memory-mcp)

**What it does:** Uses tree-sitter to parse 158 languages into an AST-based knowledge graph (functions, classes, call chains, HTTP routes, cross-service links). Indexes the Linux kernel in 3 minutes. Sub-millisecond queries. Ships as a single static C binary.

**Why it matters for contextd:** contextd is markdown-only. Source code is explicitly unsupported. A tree-sitter integration would let developers index both docs and code in one graph.

### 8. 5-Lane Hybrid Search with Intent Classification (engraph)

**What it does:** Semantic embeddings + BM25 full-text + graph expansion + cross-encoder reranking + temporal scoring, fused via RRF. An LLM orchestrator classifies queries and adapts lane weights per intent (e.g., time-aware queries activate temporal lane).

**Why it matters for contextd:** contextd has 2-lane search (vector + fulltext). No temporal scoring, no reranking, no adaptive lane weighting.

### 9. Multimodal Ingestion (Klippy, LightRAG, open-recall)

**What it does:** Ingests text, images (BLIP captioning + OCR), audio (Whisper transcription), PDFs, and code into a unified vector space. Knowledge graph reasoning works across all modalities.

**Why it matters for contextd:** contextd only handles markdown. PDFs, images, and audio are common in knowledge corpora.

### 10. Immutable Content-Addressed Documents with Wikilink Graphs (immutlex)

**What it does:** Every document revision is content-addressed (like git blobs). Wikilinks (`[[...]]`) become first-class graph edges. Forward links from the body, backlinks auto-generated, semantic edges inferred. 74 MCP tools. Scheduled "cognitive maintenance" runs overnight.

**Why it matters for contextd:** contextd does not understand wikilinks. Many Obsidian users (a huge target audience) rely on `[[wikilinks]]` as their primary linking mechanism.

### 11. Spaced Repetition and Active Recall (open-recall)

**What it does:** Generates flashcards from documents, schedules review via spaced repetition algorithms (Again/Hard/Good/Easy ratings). Treats the knowledge graph as something to actively learn from, not just search.

**Why it matters for contextd:** contextd is a retrieval tool. It does not help users internalize knowledge.

### 12. Contradiction Detection and Approval Workflows (SwarmVault)

**What it does:** Every inferred edge is tagged `extracted`, `inferred`, or `ambiguous`. Contradiction detection flags conflicting claims. New concepts land in `wiki/candidates/` first for review. `compile --approve` stages changes into reviewable approval bundles.

**Why it matters for contextd:** contextd creates inferred edges silently. There is no review step, no contradiction flagging beyond the `CONTRADICTS` edge type, and no approval workflow.

### 13. Blast Radius Analysis (knowledge-master)

**What it does:** "What depends on this service/file/technology?" Traverses the dependency graph to show what would break if something changes.

**Why it matters for contextd:** contextd has `DEPENDS_ON` edges but no dedicated blast-radius tool that computes transitive impact.

### 14. Bitemporal Facts and Valid-Time Modeling (cuba-memorys)

**What it does:** Every fact has both a system time (when it was recorded) and a valid time (when it was true in the real world). Enables queries like "what did we believe about X at time T?" and "what was actually true about X during period P?"

**Why it matters for contextd:** contextd tracks `updated` timestamps but has no concept of valid-time ranges. Documents describe things that were true at specific times; the graph does not model this.

### 15. WASM Deployment for Browser-Based Graphs (graphrag-rs, mnem)

**What it does:** Compiles the entire engine to WebAssembly. The knowledge graph runs in a browser tab with no server, enabling zero-install demos and embedded documentation explorers.

**Why it matters for contextd:** contextd requires Docker + Neo4j. A WASM fallback for small corpora would dramatically lower the barrier to entry.

---

## Original Ideas That Extend Beyond Existing Projects

These are functionalities not found implemented in any of the 34 projects examined, but which would compose naturally with contextd's architecture:

### A. Differential Corpus Comparison ("Knowledge Diff")

No project offers a way to compare two corpus snapshots and produce a structured diff: "these 5 edges appeared, these 3 disappeared, these 2 nodes changed summary, this new contradiction emerged." Combined with the versioning concept from mnem, this would let you answer "what changed in our architecture docs between sprint 42 and sprint 44?" at the graph level, not just the file level.

### B. Confidence Decay with Citation Freshness

Instead of binary fresh/stale, assign a decay function to every node and edge based on the age of its source material and how many other nodes cite it. Nodes cited by many recent documents stay high-confidence; nodes cited only by a 2-year-old doc decay toward a "needs verification" threshold. The MCP tool `check_freshness` already exists conceptually but today it requires explicit `NEEDS_UPDATE` edges rather than computing decay automatically.

### C. Query-Driven Lazy Indexing

Instead of indexing everything upfront, index file metadata (path, size, mtime) and embeddings only. When a query hits a cluster of files that have never been summarized, trigger on-demand summarization for just those files. This turns the cost model from O(corpus) upfront to O(query-relevant-subset) amortized. No project currently does this.

### D. Cross-Corpus Federation with Schema Alignment

contextd already supports multiple corpora but they are isolated graphs. A federation layer would let you query across corpora with automatic schema alignment: "find all Tickets in my work-knowledge corpus that relate to Patterns in my architecture-docs corpus." This goes beyond magpie-search's multi-source fusion because it preserves typed relationships across corpus boundaries rather than just ranking results.

### E. Provenance-Weighted Retrieval

When the same fact appears in multiple sources, weight it by the provenance chain. A fact stated in an authoritative spec that is cited by 3 other documents outranks the same fact mentioned offhandedly in a meeting note. No project currently factors citation-graph topology into retrieval ranking beyond simple inbound-count ordering.

### F. Interactive Graph Refinement via MCP ("Teach Mode")

Let the user correct the graph through natural language during an AI session: "Node X and Y are not actually related" or "This pattern should be called Observer, not Listener." These corrections become `manual` origin edges that survive re-indexing. The neo4j-labs meta-knowledge-graph does session-capture but not interactive correction of the graph itself.

### G. Corpus Health Score and Maintenance Agent

Compute a single health metric per corpus based on: percentage of nodes with stale summaries, orphan nodes with no edges, contradiction density, community cohesion scores, and average node freshness. Expose it via `contextd status` and optionally trigger a "maintenance pass" that only re-indexes the unhealthiest nodes. engraph has "vault health diagnostics" for broken links, but no project computes a single composite health score with automated remediation.

### H. Embeddings-Free Graph-Only Retrieval Mode

For environments where even local embeddings are too expensive (Raspberry Pi, CI runners), offer a pure-graph retrieval mode that uses only BM25 full-text + typed edge traversal + PageRank for ranking. shodh-memory proves that LLM-free memory is viable; the same principle applies to retrieval where the graph structure itself carries enough signal.

---

## Prioritized Roadmap

| Priority | Feature | Effort | Impact | Notes |
|---|---|---|---|---|
| **High** | Wikilink parsing for Obsidian users | Low | High | Opens the largest single user base for local knowledge tools |
| **High** | Community detection + cluster summaries | Medium | High | Enables "what are the themes?" global queries |
| **High** | LLM-free fallback for entity extraction | Medium | High | Makes the daemon vastly cheaper to run continuously |
| **High** | Query-driven lazy indexing | Medium | High | Drops the upfront cost barrier for large corpora |
| **Medium** | Knowledge versioning (branch/diff/rollback) | High | High | Unique differentiator, no Python project does this well |
| **Medium** | Confidence decay with automatic freshness | Low | Medium | Builds on existing `check_freshness` infrastructure |
| **Medium** | Cross-corpus federation | Medium | Medium | Natural extension of existing multi-corpus support |
| **Medium** | Blast radius / transitive impact tool | Low | Medium | High value for architecture-doc corpora |
| **Medium** | Interactive graph refinement ("Teach Mode") | Medium | Medium | Leverages existing `manual` origin edges |
| **Medium** | Corpus health score | Low | Medium | Good UX signal, motivates maintenance |
| **Medium** | Adaptive search lanes (reranking, temporal) | Medium | Medium | Improves retrieval quality significantly |
| **Low** | WASM deployment for small corpora | High | Medium | Removes Docker dependency for demos |
| **Low** | Multimodal ingestion (PDF, images) | High | Medium | Large scope but large audience |
| **Low** | Spaced repetition / active recall | Medium | Low | Niche but sticky for learning-focused users |
| **Low** | Append-only audit log with replay | High | Low | Enterprise feature, overkill for personal use |
| **Low** | Code-level AST integration (tree-sitter) | High | Medium | Large engineering effort, separate tool may be better |

---

## Quick Wins (implementable in days, not weeks)

1. **Wikilink edge extraction** during the `relate` phase: regex for `[[target]]` patterns, create `REFERENCES` edges to matching File nodes.
2. **Blast radius MCP tool**: given a node, return all transitively reachable nodes via `DEPENDS_ON` edges with hop count.
3. **Corpus health score**: count orphan nodes, stale nodes (updated > N days ago with no inbound edges), contradiction edges. Return a 0-100 score via `contextd status`.
4. **Confidence decay calculation**: on `check_freshness`, compute a decay score based on `updated` timestamp age weighted by inbound citation count, without requiring explicit `NEEDS_UPDATE` edges.
5. **PageRank for `describe_project`**: replace simple inbound-count ordering with PageRank for better "importance" ranking.
