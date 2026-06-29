# Contextd (Context Daemon Abbrev.)

[![CI](https://github.com/giuseppecardenas/contextd/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/giuseppecardenas/contextd/actions/workflows/ci.yml)

Contextd is a locally-hosted knowledge layer for your project files. It indexes a corpus of markdown files into a hybrid graph + vector store (Neo4j Community), generates AI-inferred relationships and per-file summaries, and exposes the result to Claude Desktop, Cursor, and any MCP-speaking client through an MCP server. Cold-start any AI session with a compact, semantically-organised overview of your entire corpus.

- **Input:** markdown (`.md`) files only. Markdown is the one format contextd is built and tested for today; the default corpus include pattern is `**/*.md`, the section-granular mode parses markdown headings, and the summarisation prompts assume prose-style documents. Pointing contextd at source code or structured data (JSON, CSV, and similar) is untested and unsupported at present, and although the section-granular pipeline contains a fallback path that routes non-markdown files through file-granular indexing, that path has not been exercised against real non-markdown corpora and should not be relied upon.
- **Storage:** Neo4j Community 5.x — runs in Docker, binds port 7687 over Bolt.
- **Inference:** Google Gemma (`gemma-4-31b-it` default) via the Gemini API for summarisation, relationship inference, and NL→Cypher translation; Voyage AI `voyage-4-large` (1024-dim, 32k-token context) for vector embeddings. Both inference and embeddings can instead run against a local OpenAI-compatible server (llama.cpp, Ollama, vLLM, LM Studio), so the whole pipeline can run fully offline.
- **Interface:** stdio MCP server (`contextd-mcp`) and CLI (`contextd`).
- **Platforms:** runs natively on Linux, macOS, and Windows 11. The daemon, IPC, file-watching, and Docker invocation are platform-agnostic — Windows is not a WSL-only afterthought. WSL2 is still a fine host on Windows if you prefer it, but a native Python install on Windows is equally supported.
- **Privacy:** all state lives under `~/.contextd/` (`%USERPROFILE%\.contextd` on Windows); no data is stored outside your machine beyond the per-file API calls.

> **Status: alpha.** v0.1.0 is pre-PyPI. Use the dev install below.

---

## Quickstart

```bash
# 1. Install (dev path — see Install section for details)
git clone git@github.com:giuseppecardenas/contextd.git
cd contextd
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Set API keys
export GEMINI_API_KEY=<your-key>
export VOYAGE_API_KEY=<your-key>

# 3. First-run wizard — creates ~/.contextd/ layout
contextd init

# 4. Start the storage backend + indexer daemon
contextd up

# 5. Register a corpus and index it
contextd add-corpus examples/minimal-notes --name notes
contextd index notes --bootstrap

# 6. Query the graph
contextd ask "which files reference note-1?"
```

After `contextd up` and a successful bootstrap, open Claude Desktop and call the `describe_project` MCP tool — you get a JSON primer of the top-cited files with summaries, ready for an AI assistant to anchor its session.

---

## Concepts

### File nodes vs Section nodes

By default, each file in a corpus is indexed as a single `File` node with an AI-generated summary and a `voyage-4-large` embedding (1024-dim). This is **file-granular** mode.

For heavily structured markdown corpora (long PRDs, specs, design docs), **section-granular** mode promotes each heading to a first-class `Section` node. Structural edges — `CONTAINS`, `PARENT_OF`, `NEXT_SIBLING` — model the document tree; AI infers semantic edges (`REFERENCES`, `DOCUMENTS`, `SUPERSEDES`, etc.) across section boundaries. Enable it with `--granularity section` in `add-corpus`.

### Edge origins

Every edge carries one of three `origin` values:

| Origin | Meaning |
|---|---|
| `structural` | Document structure (CONTAINS / PARENT\_OF / NEXT\_SIBLING). Never overwritten on re-index. |
| `inferred` | AI-inferred semantic relationship. Wipe-and-replace on re-index. |
| `manual` | Hand-authored. Never overwritten on re-index. |

**Inferred edges attach to real nodes, never placeholders.** When inference references a `File` or `Section` (node types created only by indexing on-disk content), the edge is wired to the existing node — matched by canonical path/id, or for files by unique basename — or dropped if nothing matches. It never mints a stub. Only abstract entity types (`Pattern`, `Risk`, `Ticket`, `Technology`, …) are created on demand as inference targets. Node identity is a canonical forward-slash path, so re-indexing a file updates its node in place instead of creating a separator-variant duplicate.

### The GraphStore abstraction

All higher layers (indexer, MCP server, CLI) talk to the backend through the `GraphStore` ABC, never to a concrete backend directly. Today the only backend is `Neo4jBackend`, constructed by the storage factory from `~/.contextd/config.toml`:

```toml
[storage]
backend = "neo4j"
```

The ABC and the factory seam are deliberately retained so a second backend could be added later (widen `BackendName`, add a factory branch and a migrations dir) without changing any consumer.

### Ontology

The base ontology (`contextd/ontology/base.json`) defines the node types and edge types the AI is allowed to infer. Unrecognised types are silently discarded at index time. Per-corpus aliases let you map domain-specific vocabulary to canonical types — see [Ontology customisation](#ontology-customisation) below.

Node types: `File`, `Section`, `Artifact`, `Ticket`, `Pattern`, `Technology`, `Client`, `Repo`, `Service`, `Integration`, `Risk`, `WorkSession`, `Corpus`, `Meta`.

Edge types: `CONTAINS`, `PARENT_OF`, `NEXT_SIBLING`, `BELONGS_TO`, `CREATED_BY`, `DOCUMENTS`, `DOCUMENTED_IN`, `APPLIES_TO`, `PART_OF`, `SIMILAR_TO`, `RELATED_TO`, `REFERENCES`, `SUPERSEDES`, `CONTRADICTS`, `USES`, `MODIFIES`, `DEPENDS_ON`, `IDENTIFIES_RISK`, `RECOMMENDS`, `EDITED_DURING`, `VERIFIED_ON`, `NEEDS_UPDATE`.

---

## Install

### Pre-PyPI (current)

The package is not yet on PyPI. Clone the repo and install in editable mode.

**Linux / macOS:**

```bash
git clone git@github.com:giuseppecardenas/contextd.git
cd contextd
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
export GEMINI_API_KEY=<your-key>
export VOYAGE_API_KEY=<your-key>
```

**Windows 11 (native PowerShell):**

```powershell
git clone git@github.com:giuseppecardenas/contextd.git
Set-Location contextd
uv venv
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
$env:GEMINI_API_KEY = "<your-key>"
$env:VOYAGE_API_KEY = "<your-key>"
```

Requirements (both platforms): Python 3.11+, Docker (Docker Desktop on Windows/macOS or any Docker engine with Compose v2 on Linux). Docker Desktop on Windows is supported — `contextd up` calls `docker compose` against the same daemon that `docker` does from PowerShell. On Windows, Docker Desktop must be set to Linux containers (the default) for the Neo4j image.

The remaining steps in this README are written with `bash` syntax. The equivalent PowerShell forms differ only in shell-specific surface (`export` ↔ `$env:`, `~/.contextd/` ↔ `$env:USERPROFILE\.contextd\`); the `contextd` CLI invocations themselves are identical on both platforms.

### Global command without venv activation (optional)

After the editable install, the `contextd`, `contextd-mcp`, and `contextd-indexer` entry points live in `.venv/bin/` (Linux/macOS) or `.venv\Scripts\` (Windows) and are only on `PATH` while the venv is activated. To call them from any shell — useful for Claude Desktop's MCP config, scheduled tasks, or one-off invocations — drop file links into a directory that's already on your user `PATH`. The venv stubs are thin launchers that import the editable-installed package, so the link always picks up the latest source on disk; there is no compile step to keep in sync.

**Linux / macOS** (symlinks, no privileges needed):

```bash
# run from the repo root, after creating the venv
mkdir -p ~/.local/bin
ln -s "$PWD/.venv/bin/contextd"         ~/.local/bin/contextd
ln -s "$PWD/.venv/bin/contextd-mcp"     ~/.local/bin/contextd-mcp
ln -s "$PWD/.venv/bin/contextd-indexer" ~/.local/bin/contextd-indexer
```

**Windows 11** (hardlinks, no admin needed — symlinks on Windows require admin elevation or Developer Mode, hardlinks need neither, with the one constraint that source and destination must live on the same volume):

```powershell
# run from the repo root, after creating the venv
$src = "$PWD\.venv\Scripts"
$dst = "$env:USERPROFILE\.local\bin"   # any dir on your user PATH
foreach ($name in "contextd.exe","contextd-mcp.exe","contextd-indexer.exe") {
    fsutil hardlink create "$dst\$name" "$src\$name" | Out-Null
}
```

**Caveat — pip stub regeneration.** The links above point at pip-generated launcher stubs. Editing `.py` files in the source tree never affects them — the editable install loads source directly at import time, regardless of which copy of the stub ran. The one situation that *does* matter is if `[project.scripts]` in `pyproject.toml` changes (entry-point name or module path): pip writes a fresh stub to the venv, and on Windows the hardlink in `~/.local/bin` ends up pointing at the now-orphaned old inode. If you suspect drift, re-run the link-creation command. To verify on Windows:

```powershell
fsutil hardlink list "$env:USERPROFILE\.local\bin\contextd.exe"
```

If the venv path still appears in the output, the hardlinks are healthy; if only the `.local\bin` entry is listed, recreate the links.

### After v0.1.0 on PyPI (future)

```bash
pipx install contextd
```

The `pipx` path is the intended production install once the package is published. For now, use the dev install above.

---

## Usage walkthrough

This walkthrough uses `examples/minimal-notes` — a personal-notes fixture (10 note files + a README) included in the repo. The same steps apply to any corpus.

### Step 1 — First-run setup

```bash
contextd init
```

Creates `~/.contextd/` with the directory layout, copies `config.toml` from the package default, and checks for required env vars (`GEMINI_API_KEY`, `VOYAGE_API_KEY`) and Docker. Set both keys in your shell before proceeding:

```bash
export GEMINI_API_KEY=<from https://aistudio.google.com/app/apikey>
export VOYAGE_API_KEY=<from https://www.voyageai.com/>
```

### Step 2 — Start the backend

```bash
contextd up
```

Runs `docker compose --profile neo4j up -d` (using `~/.contextd/docker-compose.yml`), waits for Neo4j to be ready, then applies the schema migrations (constraints, vector index, full-text index). Output:

```
✓ neo4j container up at 127.0.0.1:7687
✓ migrations applied
ready
```

### Step 3 — Register the corpus

```bash
contextd add-corpus examples/minimal-notes --name notes
```

Writes `~/.contextd/corpora/notes.toml` with root path, glob pattern (`**/*.md`), and `granularity = "file"`. To index at section granularity:

```bash
contextd add-corpus examples/minimal-notes --name notes --granularity section
```

### Step 4 — Bootstrap the index

```bash
contextd index notes --bootstrap
```

Runs the five-phase bootstrap pipeline:

1. **enumerate** — discovers files, computes embeddings at CREATE time.
2. **embed** — accounting stub (embeddings already written in phase 1).
3. **summarise** — calls Gemma once per file to generate `summary` and `key_points`.
4. **relate** — calls Gemma once per file to infer typed edges between nodes.
5. **close** — persists corpus stats and checkpoint state.

Progress is printed per phase:

```
found 11 files in corpus 'notes'
  ✓ enumerate: processed=11 skipped=0
  ✓ embed: processed=11 skipped=0
  ✓ summarise: processed=11 skipped=0
  ✓ relate: processed=11 skipped=0
  ✓ close: processed=1 skipped=0
```

To preview token cost without indexing:

```bash
contextd index notes --estimate-only
```

**Resuming a partial run.** Re-running `contextd index <corpus> --bootstrap` is idempotent: nodes that already have a summary or an `inferred_at` marker (written by a prior successful `relate` pass) are skipped automatically. A killed-mid-run bootstrap only needs a plain `--bootstrap` restart to finish — no extra flag.

**Wiping a layer for a fresh re-inference.** Use `--refresh <scope>`:

| scope | wipes | preserves | re-costs |
|---|---|---|---|
| `inferred` | `origin='inferred'` edges + `inferred_at` markers | summaries, structural, embeddings | Gemini relate only |
| `summaries` | `summary`/`key_points`/`summary_confidence` | inferred edges, structural, embeddings | Gemini summarise only |
| `llm` | both of the above | structural, embeddings | all Gemini work |
| `all` | DETACH DELETE every `Section`/`File`/`Corpus` node for this corpus | nothing (structural + inferred edges cascade-deleted) | Voyage + Gemini from zero |

```bash
contextd index notes --bootstrap --refresh llm       # e.g. after a prompt change
```

**Incremental re-index (one-shot).** To re-index only files that have changed since the last run:

```bash
contextd index notes --incremental
```

Scans every file in the corpus. For file-granular corpora, MD5 hashing skips unchanged files. For section-granular corpora, each section's stored hash is compared to the current content — only sections that changed, were added, or were removed trigger re-inference; unchanged sections are protected by IS-NULL guards and skipped automatically. Output:

```
  ✓ incremental scan complete: indexed=2 deleted=0 skipped=9
```

For continuous watching, use the daemon — see [Incremental indexer daemon](#incremental-indexer-daemon) below.

### Step 5 — Query

```bash
contextd ask "which notes mention cooking?"
```

Translates the question to Cypher via Gemini, runs the query, prints results as JSON. The Cypher is printed first so you can inspect what was generated.

### Step 6 — Connect Claude Desktop

Add `contextd-mcp` to your Claude Desktop config (see [MCP integration](#mcp-integration)). Once connected, call `describe_project` from the Claude Desktop interface. For the minimal-notes corpus you get something like:

```json
[
  {"path": "examples/minimal-notes/note-1.md", "name": "note-1", "summary": "...", "key_points": [...], "inbound": 3},
  ...
]
```

The tool returns up to 40 nodes ordered by inbound-citation count, each with its AI summary. An AI assistant can use this as a session primer — one tool call replaces reading every file.

### Step 7 — Monitor costs and logs

```bash
# Tail the daemon's structured JSON log
contextd logs --follow

# Aggregated provider token spend
contextd costs
contextd costs --since 2026-04-01

# List registered corpora
contextd list-corpora

# Backend + daemon + corpora state at a glance
contextd status
```

Each indexing session logs per-provider input/output token counts to `~/.contextd/state/session-log/`. The `costs` command aggregates these into a per-provider summary. The `status` command queries the daemon's IPC endpoint (Unix socket on Linux/macOS, localhost TCP on Windows) for live runtime info (pid, uptime, watched corpora) and falls back to a PID-file check if the endpoint is unreachable.

---

## CLI reference

| Command | Purpose | Key flags |
|---|---|---|
| `contextd init` | First-run wizard — creates `~/.contextd/` layout | `--yes` (non-interactive) |
| `contextd up` | Start graph backend + indexer daemon | — |
| `contextd down` | Stop daemon + graph backend | — |
| `contextd status` | Report daemon, backend, and corpora state | — |
| `contextd add-corpus PATH` | Register a corpus for indexing | `--name`, `--granularity {file,section}`, `--from TEMPLATE` |
| `contextd list-corpora` | List registered corpora | — |
| `contextd index CORPUS` | Run an indexing pass | `--bootstrap`, `--incremental`, `--estimate-only`, `--refresh {inferred,summaries,llm,all}` |
| `contextd ask "QUESTION"` | NL→Cypher query translation and execution | `--corpus NAME` |
| `contextd logs` | Tail the structured JSON log | `--follow` |
| `contextd costs` | Aggregated provider token spend | `--since YYYY-MM-DD` |

Three console scripts are installed:

| Script | Role |
|---|---|
| `contextd` | CLI (all commands above) |
| `contextd-mcp` | stdio MCP server for Claude Desktop / Cursor / Zed |
| `contextd-indexer` | Background indexer daemon (launched automatically by `contextd up`) |

---

## MCP integration

Contextd exposes a stdio MCP server. It works with any MCP-speaking client over stdio — Claude Desktop (macOS and Windows), Cursor (cross-platform), Zed, or your own MCP-compatible tooling. The `contextd-mcp` binary reads/writes JSON-RPC on stdin/stdout; consult your client's MCP config docs for how to register the server.

**macOS (Claude Desktop)** — config at `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "contextd": {
      "command": "contextd-mcp",
      "args": []
    }
  }
}
```

**Windows (Claude Desktop)** — config at `%APPDATA%\Claude\claude_desktop_config.json` — same JSON structure as above.

**Cursor / Zed / other clients** — register `contextd-mcp` as a stdio MCP server; consult your client's docs for the exact config format.

If `contextd-mcp` is not on your PATH (e.g., when using a venv), use the absolute path to the binary. Replace `/path/to/contextd` below with your actual clone location (run `realpath .venv/bin/contextd-mcp` on Linux/macOS, or `Resolve-Path .venv\Scripts\contextd-mcp.exe` on Windows, from the repo root to print it).

**Linux/macOS:**

```json
{
  "mcpServers": {
    "contextd": {
      "command": "/path/to/contextd/.venv/bin/contextd-mcp",
      "args": []
    }
  }
}
```

**Windows:**

```json
{
  "mcpServers": {
    "contextd": {
      "command": "C:\\path\\to\\contextd\\.venv\\Scripts\\contextd-mcp.exe",
      "args": []
    }
  }
}
```

### Generic tools (always registered)

| Tool | What it does |
|---|---|
| `describe_project` | Top-N File nodes by inbound-citation count with summaries. Accepts `corpus` and `n` (default 40). |
| `search` | Full-text search over summaries. Accepts `query`, optional `kind` (default `File`; `Section` also works in section-granular corpora), optional `limit` (default 20). |
| `related` | Outbound + inbound traversal within N hops (1–5). Accepts `node_id` and `depth` (default 2). |
| `inbound` | What cites this node? Accepts `node_id`. |
| `outbound` | What does this node cite? Accepts `node_id`. |
| `get_file_summary` | Summary + key points for a single file. Accepts `path`. |
| `section_tree` | Outline of a file (section-granular corpora only). Accepts `file_path`. |
| `query_graph` | Read-only Cypher escape hatch. Accepts `cypher`. Write keywords are rejected by a guard. |

### Per-corpus tools

Corpus adapters can register additional Cypher-backed tools in their corpus TOML under `[mcp.tools]`. These appear in the tool list namespaced as `<corpus>.<tool>` so they never collide with the generic tools. An AI assistant calling a registered tool runs the pre-authored Cypher with `$placeholder` parameters bound, all through the same read-only guard.

Most adapters do fine without this — `query_graph` + the generic tools compose into the same queries and are tunable in-session without a server restart. Reach for `[mcp.tools]` when a query is called often enough that baking it in pays back the loss of tunability.

See [docs/mcp.md](docs/mcp.md) for the full tool reference.

---

## Incremental indexer daemon

`contextd up` starts both the graph backend **and** a background file-watching daemon (`contextd-indexer`). The daemon monitors every registered corpus root for file changes and re-indexes modified files automatically — no manual `contextd index` calls needed after the initial bootstrap.

### Lifecycle

```bash
contextd up        # start graph backend + indexer daemon
contextd status    # show daemon state (pid, uptime, corpora)
contextd down      # stop daemon + graph backend
```

`contextd status` tries the IPC endpoint first for live runtime info, and falls back to the PID file if the daemon is not reachable:

```
backend:  neo4j running at 127.0.0.1:7687
daemon:   running (pid=12345, uptime=42s, corpora=['notes'])
```

### How it works

1. A `CorpusWatcher` (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows) fires on every file change under the corpus root.
2. Events are collected by a `DebouncedQueue` with a configurable window (default 30 s) to batch rapid edits into a single pass.
3. After the window closes, the daemon runs hash checks. For file-granular corpora, unchanged files are skipped by MD5. For section-granular corpora, section-level hashes are compared — only changed, added, or removed sections are cleared and re-inferred; unchanged sections are left untouched.
4. Changed files are re-indexed concurrently (default 4 workers) by calling the same phase pipeline as `--incremental`.
5. **Crash recovery:** before processing each batch, a checkpoint is written. If the daemon is killed mid-batch, the next startup replays any in-flight files. Paths that fail indexing are buffered to `~/.contextd/state/pending-upserts.jsonl` and retried on the next start.

### Periodic sweep (WSL2 / Windows-side edits)

On Linux and macOS, `CorpusWatcher` relies on inotify / FSEvents. On WSL2, edits made from the Windows side (e.g., VS Code Windows app writing into the WSL filesystem) silently bypass inotify — the daemon never receives the event and the index goes stale with no error.

The **periodic sweep** is the fix. On a configurable interval (default 900 s), the daemon queries the graph for all Section nodes (section-granular corpora) or scans disk for all corpus files (file-granular corpora) and compares stored hashes to current content. Any file where at least one section changed, was added, or was removed is re-enqueued for indexing — exactly as if inotify had fired.

Rate limiting uses a budget-accumulation model: `sweep_rate_sections_per_second` (default 0.017 ≈ 1 section/minute) accrues per elapsed main-loop tick. The daemon processes one work unit only when the accumulated budget ≥ 1.0, deducting `max(1, len(sections))` per unit. This keeps the daemon idle between sweeps. Set `sweep_interval_seconds = 0` to disable the sweep entirely.

```toml
[indexer]
sweep_interval_seconds         = 900    # 0 disables the sweep
sweep_rate_sections_per_second = 0.017  # budget rate; default ≈ 1 section/minute
```

Log output when the sweep is active:

```
corpus notes: sweep started (12 files, 47 sections)
corpus notes: sweep complete, next in 900s
corpus notes: indexed /path/to/changed-file.md
```

### Branch gate

If you check out a remote branch for comparison, the daemon would re-index every differing file — potentially hundreds of Gemini calls against a scratch branch. The `allowed_branches` gate prevents this:

```toml
# ~/.contextd/config.toml
[indexer]
allowed_branches = ["main", "develop"]
```

When the corpus repo's active branch is not in the list, both the daemon and `contextd index --incremental` skip all work and log a warning. An empty list (the default) allows all branches.

Detached HEAD is always blocked when a whitelist is configured (a comparison checkout puts the repo in detached HEAD state).

### Tuning

```toml
[indexer]
debounce_seconds               = 30    # seconds to wait after the last event before dispatching a batch
incremental_workers            = 4     # concurrent file workers per batch (distinct from inference_concurrency)
inference_concurrency          = 1     # LLM call parallelism within each file's summarise+relate phases
sweep_interval_seconds         = 900   # how often to run the periodic sweep; 0 disables it
sweep_rate_sections_per_second = 0.017 # budget accumulation rate; ≈ 1 section/minute

[logging]
max_log_bytes  = 10485760  # 10 MB per log file; 0 disables rotation
log_backup_count = 5       # number of rotated files to keep
```

The daemon writes only to the configured log file (`~/.contextd/logs/contextd.log` by default) — no terminal output — so these rotation settings matter for long-running installs.

### Running under systemd (optional)

`contextd up` spawns the daemon as a detached child of your shell, so WSL VM shutdown or logout kills it without a clean `SIGTERM` — you'll see unexplained gaps in the log followed by fresh `daemon started` lines. To auto-restart across WSL/machine boots, install the provided user unit:

```bash
# 1. Provider keys — systemd user units do NOT source ~/.bashrc, so put the
#    keys the daemon needs in a dedicated env file with restrictive perms.
mkdir -p ~/.config/contextd
umask 077
cat > ~/.config/contextd/env <<EOF
GEMINI_API_KEY=$GEMINI_API_KEY
VOYAGE_API_KEY=$VOYAGE_API_KEY
EOF
chmod 600 ~/.config/contextd/env

# 2. Install and enable the unit (run from the contextd repo root).
mkdir -p ~/.config/systemd/user
cp scripts/systemd/contextd-indexer.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now contextd-indexer.service

# 3. Allow the unit to run without an active login session (WSL users: required).
loginctl enable-linger $USER
```

Verify:

```bash
systemctl --user status contextd-indexer    # → Active: active (running)
tail -f ~/.contextd/logs/contextd.log       # → "daemon started ... watching corpus ..."
```

If `status` shows `code=exited, status=1/FAILURE`, run `journalctl --user -u contextd-indexer -n 30` — the most common cause is a missing or unreadable `~/.config/contextd/env` (the daemon refuses to start without `GEMINI_API_KEY` / `VOYAGE_API_KEY`).

Then use `systemctl --user {status,stop,restart} contextd-indexer` instead of `contextd up/down` for the daemon portion. `contextd up` still handles the graph backend container; just skip its daemon-launch step by not calling it (or let the PID-file guard skip it for you — `up` detects the running daemon and leaves it alone).

---

## Windows support

Contextd runs natively on Windows (Python 3.11+ installed directly, CLI in cmd.exe or PowerShell, daemon as a native Windows process). No WSL2 required.

The only platform difference is the daemon's IPC transport: on Linux/macOS the daemon uses a Unix domain socket (`~/.contextd/ipc.sock`); on Windows it binds to `127.0.0.1` on an ephemeral TCP port and writes the port number to `~/.contextd/ipc.port`. The JSON-lines wire protocol is identical — this is transparent to the user and to MCP clients.

Process management also adapts: the daemon detaches via `CREATE_NEW_PROCESS_GROUP` instead of `start_new_session`, and `contextd down` uses `TerminateProcess` instead of `SIGTERM`. All platform-specific code is isolated in `contextd/_compat.py`; no other module branches on `sys.platform`.

### Install on Windows

```powershell
git clone https://github.com/giuseppecardenas/contextd.git
cd contextd
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Requirements: Python 3.11+, Docker Desktop (with Linux containers enabled — required for the Neo4j image).

### WSL2 alternative

If you prefer to run inside WSL2, four PowerShell wrappers under `scripts/windows/` forward commands into the WSL distribution:

| Script | Equivalent |
|---|---|
| `contextd-init.ps1` | `wsl -d $Distro -- contextd init` |
| `contextd-up.ps1` | `wsl -d $Distro -- contextd up` |
| `contextd-down.ps1` | `wsl -d $Distro -- contextd down` |
| `contextd-status.ps1` | `wsl -d $Distro -- contextd status` |

Set `$Distro` to your WSL distribution name (e.g., `Ubuntu`).

**WSL2-specific note:** edits made from the Windows side (e.g., VS Code Windows app writing into the WSL filesystem) silently bypass Linux inotify — the daemon never sees the event. The [periodic sweep](#periodic-sweep-wsl2--windows-side-edits) compensates by polling on a configurable interval (default 900 s). If you primarily edit from Windows, consider lowering `sweep_interval_seconds` for faster catch-up.

---

## Ontology customisation

The base ontology is intentionally general. Domain-specific corpora map their vocabulary to Contextd's canonical node and edge types through two mechanisms:

1. **Inline aliases** (`[ontology.aliases]` in the corpus TOML) — map domain names to canonical node types. For example, `Registry = "Pattern"` tells the AI that an inferred `Registry` node should be stored as a `Pattern`.

2. **Override file** (`[ontology] overrides = "ontology.json"`) — a JSON file that adds domain-specific edge-type aliases. All aliases are validated against the base ontology at index time; unrecognised types are silently discarded, not stored.

Adapter configs live next to the corpus they describe (by convention, a `.contextd/` directory at the corpus root containing `corpus.toml`, `ontology.json`, and `prompts/`). `contextd add-corpus <path> --from <path>/.contextd/corpus.toml` rewrites relative paths in the template to absolute paths anchored at the template's directory, so the adapter stays portable. See [docs/ontology.md](docs/ontology.md) for the full customisation reference.

---

## Writing a domain adapter

A domain adapter teaches Contextd the vocabulary and structure of a specific corpus. By convention, adapters live in a `.contextd/` directory colocated with the corpus root:

```
my-project/
├── .contextd/
│   ├── corpus.toml          # corpus config with ontology aliases, globs, prompts
│   ├── ontology.json        # domain-specific edge-type aliases
│   └── prompts/
│       └── summary.md       # custom summarisation prompt template
├── docs/
│   └── spec.md
└── src/
    └── ...
```

Register the adapter:

```bash
contextd add-corpus /path/to/my-project --name my-project --from /path/to/my-project/.contextd/corpus.toml
```

The `--from` flag copies the template's settings (ontology overrides, prompt paths, include/exclude patterns) into the registered corpus TOML at `~/.contextd/corpora/my-project.toml`, rewriting relative paths to absolute paths anchored at the template's directory. The adapter stays portable across machines.

**When to use an adapter vs plain `add-corpus`:**

- **Plain `add-corpus`** — the default ontology and prompts work well for general markdown corpora.
- **Adapter** — your corpus has domain-specific vocabulary (e.g., `Registry` should map to `Pattern`), you want a custom summarisation prompt, or you need specific include/exclude patterns or section-level heading bounds.

---

## Configuration reference

Contextd uses three levels of config, all under `~/.contextd/`:

| Path | Purpose |
|---|---|
| `~/.contextd/config.toml` | Global config — storage backend, provider models, inference settings, logging. Created by `contextd init`. |
| `~/.contextd/corpora/<name>.toml` | Per-corpus config — root path, glob includes, granularity, ontology aliases and overrides, MCP tools, summarisation overrides. Created by `contextd add-corpus`. |
| `~/.contextd/state/` | Runtime state — checkpoint JSON per corpus, cost log, session log. Not hand-edited. |

Key global config fields (full reference in [docs/architecture.md](docs/architecture.md)):

```toml
[storage]
backend = "neo4j"           # Neo4j Community (the only backend)

[providers]
# Inference provider per call-site. Each independently picks "gemini"
# (cloud) or "openai_compat" (any local OpenAI-compatible HTTP server:
# Ollama, LM Studio, vLLM, llama.cpp, LocalAI). The embedding provider
# picks "voyage" (cloud) or "openai_compat" (local /embeddings endpoint);
# set all four to "openai_compat" to run the pipeline fully offline.
summary     = "gemini"
inference   = "gemini"
translation = "gemini"
embedding   = "voyage"

[inference]
summary_max_words = 100

[indexer]
debounce_seconds               = 30   # seconds to collect FS events before dispatching a batch
parallel_embedding_batches     = 4
inference_concurrency          = 1    # LLM call parallelism (summarise+relate); 5 is a good
                                      # default for Gemma free-tier (15 RPM quota)
incremental_workers            = 4    # concurrent file workers per incremental batch
allowed_branches               = []   # whitelist; empty = allow all branches
sweep_interval_seconds         = 900  # 0 disables; how often to check for missed changes
sweep_rate_sections_per_second = 0.017 # budget rate; default ≈ 1 section/minute

[logging]
level          = "info"
format         = "json"
path           = "~/.contextd/logs/contextd.log"
max_log_bytes  = 10485760   # 10 MB; 0 = no rotation
log_backup_count = 5
```

Full corpus config schema and CLI reference live in [docs/cli.md](docs/cli.md).

### Per-corpus configuration

Each corpus is configured via a TOML file at `~/.contextd/corpora/<name>.toml`, created by `contextd add-corpus`. Key fields:

```toml
[corpus]
name = "notes"
root = "/path/to/notes"
include = ["**/*.md"]           # glob patterns; default ["**/*"]
exclude = ["drafts/**"]         # glob patterns to skip; default []
granularity = "file"            # "file" (default) or "section"
heading_min_level = 2           # section mode: shallowest heading to promote (default 2)
heading_max_level = 4           # section mode: deepest heading to promote (default 4)

[embedding]
model = "voyage-4-large"        # override global embedding model
chunk_tokens = 32000            # max tokens per chunk (default 32000)
chunk_overlap = 200             # overlap between chunks (default 200)

[ontology]
base = "default"                # base ontology name
overrides = "ontology.json"     # path to domain-specific edge aliases (relative to TOML)
aliases = { Registry = "Pattern", Procedure = "Artifact" }  # node-label aliases

[mcp.tools]
# Per-corpus Cypher tools; each key becomes <corpus>.<key> in the MCP tool list
# my_query = "queries/my_query.cypher"

[summarization]
prompt_override = "prompts/summary.md"  # custom summarisation prompt
max_words = 150                          # override global summary_max_words
```

The `include` and `exclude` patterns use glob syntax with `**` for recursive matching. Files under `.git`, `.venv`, `__pycache__`, and `node_modules` are always excluded regardless of patterns.

### Using a local model

Each of `summary`, `inference`, and `translation` can independently target a local OpenAI-compatible HTTP server (Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI) instead of Gemini, and embeddings can likewise be served locally. A common split is to push the high-volume summary + relate traffic to a local model and keep translation (Cypher generation for `contextd ask`) on Gemini for higher quality:

```toml
[providers]
summary     = "openai_compat"
inference   = "openai_compat"
translation = "gemini"
embedding   = "voyage"

[providers.openai_compat]
base_url        = "http://localhost:11434/v1"   # Ollama default
# api_key_env   = "OPENAI_API_KEY"               # only for servers that require a token
model_summary   = "qwen2.5:7b-instruct"
model_inference = "qwen2.5:14b-instruct"
model_translation = "qwen2.5:14b-instruct"
max_retries     = 5
request_timeout_seconds = 120.0
json_mode       = true   # sends response_format JSON for summary+inference call-sites
```

Quality floor: the `inference` call-site (relationship edges) emits strict typed-edge JSON and benefits from larger models (14B+). Smaller models are fine for `summary` only. See the [accuracy caveat](#accuracy-caveat) below for a detailed breakdown and recommended hybrid config.

#### Running fully offline

With all four providers set to `openai_compat`, the entire indexing pipeline can run against locally-hosted models — no `GEMINI_API_KEY` or `VOYAGE_API_KEY` is needed. The only remaining dependency is the storage backend, which already runs locally in Docker.

Below is a concrete, tested setup using **llama.cpp**'s built-in OpenAI-compatible server with two models (one for chat, one for embeddings). The same pattern works with Ollama, vLLM, or LM Studio — just adjust `base_url` and model names.

##### 1. Download models

```bash
# Embedding model — bge-m3, 1024-dim, 8192-token context (~670 MB).
# Handles files up to ~24 KB without chunking.
curl -L -o ~/models/bge-m3-q8_0.gguf \
  https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF/resolve/main/bge-m3-q8_0.gguf

# Chat model (~1.6–8 GB depending on choice)
# Option A — small, fast (Qwen 2.5 1.5B, ~1.5 GB):
curl -L -o ~/models/qwen2.5-1.5b-instruct-q8_0.gguf \
  https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q8_0.gguf

# Option B — larger, higher quality (Gemma 4 4B, ~7.5 GB):
curl -L -o ~/models/gemma-4-E4B-it-Q8_0.gguf \
  https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf
```

##### 2. Launch the servers

```bash
# Embedding server on port 8081 (--ctx-size 8192 = bge-m3's native window)
llama-server \
  -m ~/models/bge-m3-q8_0.gguf \
  --host 127.0.0.1 --port 8081 \
  --embeddings --pooling cls --ctx-size 8192 &

# Chat server on port 8080 (--ctx-size 0 = model's maximum)
llama-server \
  -m ~/models/qwen2.5-1.5b-instruct-q8_0.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 0 &

# Wait for both to be ready
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8081/health
```

##### 3. Configure contextd

```toml
# ~/.contextd/config.toml
[providers]
summary     = "openai_compat"
inference   = "openai_compat"
translation = "openai_compat"
embedding   = "openai_compat"

[providers.openai_compat]
base_url        = "http://127.0.0.1:8080/v1"
model_summary   = "qwen2.5-1.5b-instruct"
model_inference = "qwen2.5-1.5b-instruct"
model_translation = "qwen2.5-1.5b-instruct"
max_retries     = 3
request_timeout_seconds = 300.0
json_mode       = false            # disabled — small local models often trip on
max_output_tokens = 256            # grammar-constrained JSON; cap tokens instead

[providers.openai_compat_embedding]
base_url    = "http://127.0.0.1:8081/v1"
model       = "bge-m3"
dimensions  = 1024
max_batch_size = 8
max_retries = 3
request_timeout_seconds = 300.0
```

##### 4. Index and verify

```bash
contextd init
contextd up
contextd add-corpus ~/my-notes --name notes
contextd index notes --bootstrap
```

##### Accuracy caveat

Self-hosted models trade convenience and privacy for accuracy. Cloud models (Gemini, Voyage AI) produce higher-quality summaries, more reliable relationship inference, and better NL→Cypher translation — especially at the `relate` phase, which emits strict typed-edge JSON and benefits from larger models (14B+). When running fully offline:

- **Summaries** are generally solid even on small models (1.5B–4B parameters). The prompt is bounded and the output shape is simple.
- **Relationship inference** degrades noticeably below ~4B parameters. Small models may produce malformed JSON, hallucinate entity types, or return no edges at all. If edge quality matters, keep inference on Gemini (`inference = "gemini"`) and route only summarisation and embeddings locally.
- **Embeddings** (bge-m3) are the one component where local quality is on par — the embedding model's output is deterministic and the 1024-dim vectors are directly comparable to Voyage's. bge-m3's 8192-token context window handles most files (~24 KB) without truncation.
- **NL→Cypher translation** (`contextd ask`) is the hardest task for small models. Expect malformed queries or empty results. For reliable Cypher generation, leave `translation = "gemini"`.

A practical split that preserves privacy for the bulk of the work while keeping the quality-critical parts on cloud models:

```toml
[providers]
summary     = "openai_compat"   # high-volume, local model handles it well
inference   = "gemini"          # quality-critical typed-edge JSON
translation = "gemini"          # NL→Cypher needs a large model
embedding   = "openai_compat"   # bge-m3 is on par with cloud embeddings
```

The vector index is fixed at **1024 dimensions**, so the embedding model must emit 1024-dim vectors. The provider validates each returned vector's length against `dimensions` and raises a clear error rather than writing a mismatched vector into the index, so a wrong model choice fails fast instead of corrupting search.

**Migrating an existing config.** The pre-v0.2 single-line `inference = "gemini"` under `[providers]` is replaced by three lines (`summary`, `inference`, `translation`). Pydantic will reject the old shape with `extra fields not permitted` — rename and the rest of the file stays unchanged.

---

## Cost analysis

Each file (or section, in section-granular mode) triggers two Gemini API calls (summarise + relate) and contributes to a batched Voyage AI embedding call per bootstrap. At `gemma-4-31b-it` and `voyage-4-large` pricing:

- **Per file:** sub-cent in typical cases (short markdown files); files with dense content may run a few cents each.
- **Typical 100-file corpus:** order of $0.10–$1.00 for a full bootstrap, depending on file sizes and summary length (`[inference] summary_max_words` is the main lever).
- **Incremental re-index:** for file-granular corpora, only changed files are re-processed (MD5 hash gating). For section-granular corpora, only changed sections trigger LLM calls — unchanged sections are skipped even within a modified file. Ongoing cost is proportional to the edit rate, not corpus size.

These are order-of-magnitude estimates. Exact spend is logged per session to `~/.contextd/state/session-log/`. Inspect it with:

```bash
contextd costs
contextd costs --since 2026-04-01
```

---

## Security

Contextd is a single-user local tool. Its security posture reflects that:

- **API keys** (`GEMINI_API_KEY`, `VOYAGE_API_KEY`) are read from env vars at process startup. They are never written to disk by Contextd.
- **Graph store** binds to `127.0.0.1:7687` only. Neo4j is not exposed beyond the loopback interface in the default compose config.
- **MCP read-only guard.** The `query_graph` tool and all per-corpus Cypher tools pass through `assert_read_only` before execution. The guard rejects Cypher containing `CREATE`, `MERGE`, `DELETE`, `SET`, `REMOVE`, `DROP`, `DETACH`, `FOREACH`, and `CALL` with side-effecting procedures. This guards against prompt-injection attacks that attempt to write to the graph through the MCP surface.
- **Do not expose Contextd's MCP server over a network.** It is designed for stdio transport to a locally-running MCP client. Running it as a shared network service is out of scope and untested.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `contextd up` fails with "Docker not found" | Docker engine not running or not installed | Start Docker Desktop or install Docker Engine; verify with `docker ps` |
| `contextd index` hangs or is very slow | Gemini API rate limit (15 RPM on free tier) | Reduce `inference_concurrency`; or point `summary`/`inference` at a [local model](#using-a-local-model) to avoid quota limits |
| `contextd ask` returns empty results | Corpus not bootstrapped, or query doesn't match indexed content | Run `contextd index <corpus> --bootstrap` first; try simpler queries |
| `contextd status` shows "daemon not running" after reboot | Daemon was a shell child process, not a service | Install the [systemd unit](#running-under-systemd-optional) for persistence across reboots |
| Daemon crashes on git commit | Git temp files (e.g., `index.lock`) entering the watcher pipeline | Update to latest; fixed in `2b6d33d` — the daemon now excludes `.git/` events and handles vanished temp files |
| Section-granular index skips headings | `heading_min_level` / `heading_max_level` too narrow | Adjust range in corpus TOML (default: levels 2–4); level 1 (`#`) is excluded by default |
| `contextd index --incremental` re-processes everything | Graph predates `inferred_at` marker migration | Run `contextd index <corpus> --bootstrap` once — migration `_0004` backfills markers on existing nodes |
| Phantom or duplicate `Section`/`File` nodes clutter `query_graph` / `related` / `inbound` results | Older graphs accumulated path-less `Section` / hash-less `File` stubs (inference used to mint a node per edge target) and mixed `\` vs `/` path identities from indexing under different OSes | Update to latest and run `contextd up` — migration `_0005` purges the phantom stubs (and their dangling edges) and canonicalises path identity to forward slashes. Going forward the indexer resolves inferred `File`/`Section` targets to existing nodes instead of stubbing |
| Neo4j container won't start on port 7687 | Another backend already bound to the port | `contextd down` first, then switch `backend` in config and `contextd up` |
| Edits from Windows don't trigger re-indexing | WSL2 inotify doesn't see Windows-side writes | Lower `sweep_interval_seconds` in config (see [periodic sweep](#periodic-sweep-wsl2--windows-side-edits)); or touch the file from WSL |
| Large MCP search payloads | Embedding vectors included in results (pre-`96c409a`) | Update to latest; the `search` tool now strips embedding vectors from results |

---

## Roadmap / Known gaps

Items that are designed and partially built but not yet wired or shipped:

| Gap | Detail |
|---|---|
| **Hybrid search** | `search()` is full-text only. Vector-similarity fallback (hybrid ranking) is deferred; callers needing vector-space matches can call `GraphStore.vector_search` directly for now. |
| **`CONTEXTD_INFERENCE_DAILY_BUDGET`** | The design specifies an env var cap on daily Gemini calls. Not implemented; manual cost monitoring via `contextd costs` is the current guard. |
| **Per-corpus MCP tool `$` false positives** | `extract_placeholders` uses a simple regex and will match `$` inside Cypher string literals as spurious parameters. Proper Cypher tokenisation is deferred. |
| **Stale `CheckpointStore` entries on section refresh** | The per-file checkpoint records phase-level completion (summarise, relate) but not per-section granularity. After a differential re-index updates only some sections, the checkpoint still reflects the prior full-file state. Reporting via `contextd status` will show the old completion time until the next full bootstrap. |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, the architectural invariants that CI enforces, and the commit-message conventions.

---

## Licence

Contextd is released under the MIT Licence. See [LICENSE](LICENSE).
