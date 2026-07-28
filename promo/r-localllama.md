# r/LocalLLaMA draft

**Suggested title:**
> contextd: a local graph+vector knowledge layer (GraphRAG) that serves your corpus to any MCP client — Neo4j/Memgraph, MIT

**Note on this sub:** read their self-promotion rules first. They tolerate genuine OSS but downvote anything that smells like an ad. Lead with architecture and the honest "what's local vs. not" breakdown below — do NOT bury it, this crowd will ask immediately.

---

**Body:**

Open-sourced (MIT) a tool I built for myself: **contextd**, a local GraphRAG knowledge layer. It indexes a corpus (markdown / code / structured data) into a hybrid **graph + vector store**, summarises each file or section, and infers typed relationships between nodes, then serves the whole thing to an AI assistant over MCP.

**The problem I was solving:** every new session starts from scratch with an empty context window. The model knows nothing about my corpus until I re-paste the same files, and the quality of any answer is capped by what I bothered to stuff into context that session. There are already a bazillion libraries chasing this — it's the single most ubiquitous LLM problem — so I'm not claiming novelty on the *problem*, just on the shape of my answer: a persistent, queryable graph the assistant pulls from on demand, rather than me front-loading a context window and hoping I picked the right files.

**Being upfront about "local," because this sub will ask:**

- **Everything can be fully local now.** Both inference (summaries, relationship inference, NL→Cypher) and embeddings run through an OpenAI-compatible provider. Point it at **two llama.cpp servers** — one for chat, one for embeddings — (or Ollama / LM Studio / vLLM / LocalAI) and no inference or embedding data leaves your machine. No API key needed for a local server. Gemini and Voyage AI are the defaults but not required.
- **Graph + vector store:** Neo4j Community or Memgraph, both in Docker over Bolt. All query serving, the MCP server, the daemon, your raw files, and the generated graph are local.
- **Zero cloud calls possible.** The entire indexing pipeline (summarise, embed, infer relationships, translate NL→Cypher) can run against local models. The only external dependency is Docker for the graph store itself.

So the practical state: **fully offline capable — inference + embeddings both local via llama.cpp; storage needs Docker** (Neo4j or Memgraph).

**Why graph and not just vectors:**

Pure vector RAG retrieves "things that sound similar." The graph layer adds typed, AI-inferred edges (validated against an ontology at write time, so the model can't invent edge types), which lets the assistant traverse *relationships* — "what cites this," "what does this depend on," "what's the neighbourhood around this node" — instead of only nearest-neighbour chunks. The MCP tools expose both: `search` (full-text over summaries + vectors), `related`/`inbound`/`outbound` (graph traversal), and a read-only `query_graph` Cypher escape hatch.

**Other bits this crowd tends to care about:**

- Incremental indexing daemon (`contextd-indexer`) — file watcher + debounced queue + hash-gating, so re-indexing only touches changed files. Crash-safe resume.
- File-granular (whole files as nodes) or opt-in section-granular (headings promoted to nodes with `CONTAINS`/`PARENT_OF`/`NEXT_SIBLING` edges).
- Cost logging for every external API call, so you can see exactly what indexing spent.
- Python 3.11+, `mypy --strict`, ~440 tests across both backends.

Single-user / local by design — no multi-user concerns anywhere.

Repo: https://github.com/giuseppecardenas/contextd

Local inference + local embeddings via llama.cpp/Ollama both work — the whole pipeline can now run fully offline. I tested it with mxbai-embed-large for embeddings and Qwen2.5-1.5B for inference, both served by llama.cpp servers. Curious to hear what embedding models others have tried and how they compare.
