# r/selfhosted draft

**Suggested title:**
> contextd: self-hosted GraphRAG knowledge layer for your notes/docs — Docker (Neo4j or Memgraph), serves to AI assistants over MCP, MIT

**Note on this sub:** they're strict that it must genuinely be self-hostable (it is) and want a clear "what does this do for me" framing. Mention the license and Docker up front. Be honest that indexing uses external AI APIs (see below) — the data plane is local but the indexing plane is not, and they'll respect the disclosure.

---

**Body:**

**The problem:** every time you start a fresh session with an AI assistant, the context window is empty and it knows nothing about your stuff. You end up re-pasting the same notes and docs over and over just to get it oriented. There are already a bazillion tools chasing this ubiquitous problem; contextd is my self-hosted take, where your corpus lives in a store *you* run and the assistant pulls from it on demand instead of you hand-feeding it every session.

**What it is:** contextd is a self-hosted knowledge layer that turns a folder of markdown/code/docs into a queryable **graph + vector store**, then serves it to an AI assistant (Claude Desktop, Cursor, anything that speaks MCP). Think "RAG over your own corpus," but with an actual relationship graph instead of just a pile of embeddings. MIT licensed.

**What it does for you:** point it at a notes vault or a repo, and your assistant can ask it "summarise this," "what's related to this," "what cites this file," "search the corpus" — and get answers grounded in *your* content with the relationships between documents made explicit, instead of the assistant guessing from a flat file dump.

**Self-hosting setup:**

- Storage runs in **Docker** — **Neo4j Community (default) or Memgraph**, your choice via a docker-compose profile. Both bind Bolt on 7687.
- `contextd up` starts the container and applies schema migrations; `contextd down` stops it. A `contextd-indexer` daemon watches your corpus and re-indexes incrementally (hash-gated, so only changed files get reprocessed).
- The MCP server (`contextd-mcp`) is a stdio process your AI client spawns; it connects to the local backend over Bolt.
- CLI for everything else: `init`, `add-corpus`, `index`, `status`, `ask`, `logs`, `costs`.
- Python 3.11+. Single-user / local by design.

**How local can it actually run (honest breakdown):** your files, the graph, the vector store, and all query serving always stay on your box. For the *indexing* step:

- **Inference (summaries, relationship inference, NL→Cypher) can be fully local.** It speaks the OpenAI-compatible API, so you point it at a **llama.cpp server** (or Ollama / LM Studio / vLLM) and nothing leaves your machine. No API key needed for a local server. Gemini is the default but not required.
- **Embeddings can also be fully local.** Same OpenAI-compatible provider — point it at a second llama.cpp server (or Ollama/LM Studio) serving an embedding model like mxbai-embed-large, and no file content leaves your machine for the embedding step either. Voyage AI is the default but not required.

So the entire indexing pipeline can now run 100% offline: inference + embeddings both local, your files never leave your box. The only external dependency is Docker for the graph store (Neo4j or Memgraph).

Repo with full setup instructions: https://github.com/giuseppecardenas/contextd

Feedback welcome, especially from anyone running local models end-to-end — curious what embedding models people have tried and what works well.
