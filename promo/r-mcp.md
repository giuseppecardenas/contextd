# r/mcp draft

**Suggested title:**
> contextd: an MCP server that gives Claude/Cursor a GraphRAG view of your notes and code (local graph store, MIT)

**Flair:** Server / Show-and-tell (pick whatever the sub's "I built a server" flair is)

---

**Body:**

I built an MCP server I've been wanting for my own workflow and finally cleaned it up to open-source (MIT): **contextd**.

**The problem I started from:** every new session begins with an empty context window. The assistant knows nothing about my project until I re-feed it the same files, and I'm back to explaining the same corpus from scratch every single time. There are already a bazillion libraries and features chasing this exact problem — it's the most ubiquitous pain point of working with LLMs — and this is my take on it, built around MCP rather than a bespoke retrieval layer bolted to one client.

The idea: instead of dumping raw files into context, contextd indexes a corpus (markdown, code, structured data) into a hybrid **graph + vector store**, generates a per-file/per-section summary, and infers *typed* relationships between nodes. Your assistant then queries that knowledge layer over MCP instead of grepping blindly, so a fresh session can orient itself in your corpus in a couple of tool calls instead of a wall of pasted files.

**The 8 generic MCP tools it exposes:**

- `describe_project` — ranked primer of the most-cited files in a corpus
- `search` — full-text search over the generated summaries
- `related` / `inbound` / `outbound` — traverse the inferred relationship graph
- `get_file_summary` — summary + key points for one file
- `section_tree` — outline of a file (section-granular corpora)
- `query_graph` — read-only Cypher escape hatch (write keywords are rejected)

Plus you can register per-corpus Cypher tools in a small TOML adapter when a query runs often enough to be worth pinning.

**How it's wired:**

- Storage: **Neo4j Community (default) or Memgraph**, both in Docker over Bolt. Pluggable behind a single `GraphStore` ABC.
- Inference: Gemini by default, but it speaks the OpenAI-compatible API, so you can route summaries / relationship inference / NL→Cypher to a **local llama.cpp server** (or Ollama / LM Studio / vLLM) and keep inference on-box.
- Embeddings: Voyage AI by default, but also supports a **local embedding server** (llama.cpp, Ollama, LM Studio) via the same OpenAI-compatible provider — so the *entire pipeline* can run fully offline if you want. No cloud calls required.
- The graph itself and all serving are local regardless.
- Edges are ontology-validated at write time, so the model can't invent relationship types — hallucinated edges get rejected before they're written.
- Runs as a stdio MCP server (`contextd-mcp`), connects to the backend at startup and holds it for the session.

It's single-user / local by design — no multi-tenant anything. I use it to give Claude a real map of a 400+ file notes corpus instead of a flat file dump, and the difference in answer quality is the whole reason I built it.

Repo (setup + MCP config for Claude Desktop / Cursor in the README): https://github.com/giuseppecardenas/contextd

Happy to answer anything about the tool design — the `query_graph` + ontology-validation combination was the part I went back and forth on the most.
