# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Contextd is a locally-hosted GraphRAG knowledge layer. It indexes markdown, code, and structured-data corpora into a hybrid graph + vector store (Memgraph or KùzuDB, pluggable via config), generates per-file (or per-section) summaries and AI-inferred typed relationships via external LLM APIs (Gemini for inference, Voyage AI for embeddings), and exposes the result to AI assistants through a Model Context Protocol server. The tool is designed for single-user local use — everything runs on one machine; no multi-user concerns.

## Current Phase: Active Implementation

The project is mid-build against a detailed milestone plan. The plan drives build order deterministically — do not skip or reorder milestones.

**As of 2026-04-20 (HEAD `803328c`; all pushed to origin/main — spec-delta cleanup batch `c9bc877..803328c` resolved 25 of 27 deferred items):**

- **M0** (repo scaffold) — complete (`e752200`). CI green.
- **M1** (config + ontology foundations) — complete 5/5. Closing commit `6551a71`.
- **M2** (external AI providers) — complete 5/5. `InferenceProvider`/`EmbeddingProvider` ABCs, `GeminiProvider` with retry + BLOCK_NONE safety + usage accounting, `VoyageProvider` with batched embedding + retry, factory with env-var-driven keys, append-only `CostLog`. Closing commit `f1cecb3`.
- **M3** (storage backends) — complete 4/4 with post-closure bug fixes. `GraphStore` factory + forward-only `MigrationRunner`, `MemgraphBackend` (Bolt via gqlalchemy) + baseline migration, `KuzuBackend` (embedded v0.11) + baseline migration, parametrized cross-backend integration suite. Closing commit `fd6d477`; subsequent hardening in `eb28a41`, `cabe6f7`, `7d1c285`, `088069b`, `cab529f`, `8988a12`, `c4c8cac` (see spec-delta log below).
- **M4** (indexing primitives) — complete 7/7. `FileHasher` (MD5 + JSON state), `TokenChunker` (word-count sliding window), `HeadingParser` (markdown-it-py + GitHub anchor dedup), `EntityResolver` (vector-similarity via `GraphStore.vector_search`), `PromptRenderer` (`{{var}}` mustache over `.md` templates) + three default templates (`summarise`/`relate`/`translate`), `Summariser` + `RelationshipInferrer` (provider+renderer+parser; ontology-validated silent discard on inferred edges). Shared `contextd/inference/_json_body.py` extractor. M3 storage received a cross-cutting fix (`vector_search` unified to return `score` on both backends — spec-delta #14). Closing commit `a3f3bd6`.
- **M5** (indexing pipeline) — complete 5/5. `DebouncedQueue` (path-set aggregator with monotonic idle window), `CorpusWatcher` (watchdog wrapper over inotify/FSEvents/ReadDirectoryChangesW), `is_git_busy` (`.git/index.lock` / `HEAD.lock` polling detector), `phase_enumerate`+`phase_embed`+`phase_summarise`+`phase_relate`+`phase_close` + `run_bootstrap` runner + `enumerate_corpus_files` glob expander (5-phase bootstrap per spec §5.9 step 5; phase_enumerate absorbs embedding-at-CREATE because Kuzu `File.embedding` is IMMUTABLE_AFTER_CREATE — see spec-delta #20), `CheckpointStore` (per-corpus JSON resume state). Integration test parametrized on both backends asserts phase ordering, file counts, and one REFERENCES-edge case that exercises the spec-delta-(c) label kwargs. Closing commit `80abfde` (Task 5.4 follow-up refactor at `1591925` dropped phase_embed dead params, removed discarded `count_files` query, expanded test coverage, and added `reason STRING` to Kuzu REFERENCES/BELONGS_TO REL tables — extension of spec-delta #11).
- **M6** (CLI) — complete 5/5. Click + rich CLI shipped as `contextd/cli.py` with commands `init` / `up` / `down` / `status` / `add-corpus` / `list-corpora` / `index` / `ask` / `logs` / `costs`. Shared `_load_cfg()` helper centralises the user-config-or-default fallback. Wheel now bundles `contextd/docker_compose.yml` (memgraph:latest image + port 7687 only — spec-delta #25) and force-includes repo-root `prompts/` (spec-delta #26). `tomli-w>=1.0` added to main deps. `index --bootstrap` wires the full M5 pipeline end-to-end (real embedder + summariser + inferrer + hasher + store); `index --estimate-only` is filesystem-only for cheap cost previews. `ask` command is registered via deferred import of `contextd.inference.translate.QueryTranslator` (module built in M8) — `contextd ask --help` works now, `contextd ask "<question>"` raises ImportError until M8 lands. Closing commits `56b1541` + `9ffa069` (ruff RUF100 cleanup follow-up).
- **M7** (MCP server) — complete 3/3. `ReadOnlyGuardError` + `assert_read_only(cypher)` — keyword-regex gate rejects CREATE/MERGE/DELETE/SET/REMOVE/DROP/DETACH (plan-verbatim). 8 tool functions in `contextd/mcp/tools.py` thinly wrap `GraphStore` — `describe_project` (with spec-delta #30 AND-joined WHERE fix), `search` (full-text only; vector fallback deferred to M9), `related` (variable-length paths with caller-controlled depth, now descriptor-clamped to 1-5 per spec-delta #32), `inbound` / `outbound` (type-aware citation listings), `get_file_summary`, `query_graph` (read-only guarded raw Cypher), `section_tree`. Stdio MCP server at `contextd/mcp_server.py` registers all 8 tools over the `mcp>=0.9` SDK; `pyproject.toml` already registers `contextd-mcp` console script. Integration tests exercise describe_project (ordering by inbound citations) and query_graph (write rejection) on both backends. Closing commit `a52b8f7`.
- **M8** (NL → Cypher) — complete 1/1. `QueryTranslator(provider, renderer, ontology)` in `contextd/inference/translate.py` renders the `translate` prompt with sorted node/edge-type lists, calls the inference provider, extracts Cypher via `_CYPHER_FENCE` regex (with keyword-line fallback for prose-wrapped responses), and passes the result through `assert_read_only` before returning. Follow-up `b10a02a` tightened `_extract_cypher` to raise `ValueError` when the LLM returns empty/prose-only content — previously that would silently hit the backend as an opaque syntax error. The `corpus` kwarg is plumbed through from `cli.py ask --corpus` but is a no-op TODO pending M9/M10 cross-corpus routing design (spec-delta #34). Bonus: removed the now-dead `# type: ignore[import-untyped]` on `contextd/cli.py:283` (mypy-strict `unused-ignore` forced it once `translate.py` existed with real types). Closing commits `e5f5ef8` + `b10a02a`.
- **M9** (Section-granularity) — complete 2/2. `phase_enumerate_sections` walks markdown with `HeadingParser`, batch-computes Section embeddings at CREATE time (spec-delta #37 — Section.embedding is IMMUTABLE_AFTER_CREATE on Kuzu), and emits File + Section nodes with `CONTAINS(File→Section)` / `PARENT_OF(Section→Section)` / `NEXT_SIBLING(Section→Section)` structural edges (labels required on Kuzu per spec-delta #38). `phase_embed_sections` is an accounting stub (spec-delta #37 analog). `phase_summarise_sections` and `phase_relate_sections` re-parse the source file per Section to recover body text, then SET summary/key_points/confidence or wipe-and-replace inferred edges with `src_label="Section"` (spec-delta #38). `phase_derive_file_level` sets File.summary only — File.embedding is skipped in section-mode corpora because Kuzu's File.embedding is IMMUTABLE_AFTER_CREATE (spec-delta #39, known limitation). `run_bootstrap` branches on `corpus.corpus.granularity`. Follow-up `334c682` added coverage for Section summaries, File summary derivation, and Section→Section inferred edges — the plan-verbatim test only asserted titles, leaving the Delta-B label-plumbing untested at runtime. Closing commits `971c9e0` + `7a96290` + `334c682`.

**Cursor:** M10 Task 10.1 (Runeledger-PRD adapter + fixtures).

**Spec-delta cleanup:** between M9 close and M10, a 20-commit sweep resolved 25 of 27 deferred items logged across M1–M9 (safety-critical guards, mutation-defence tightening, CLI error-handling, MCP server hygiene, label-to-PK unification, CONTEXTD_HOME extraction, prompts repackaging, Corpus-stats migration, etc.). See the spec-delta log below for per-item details. Three items intentionally deferred to future sessions: **(#72)** real `QueryTranslator` corpus injection (cross-corpus routing design work), **(#74)** stale Section garbage collection (architectural new phase; blocks M11 incremental re-index), **(#80)** `cli.py` module split into four sub-modules (broad mechanical refactor). One item — **(#71)** File/Section embedding mutability on Kuzu — research-resolved as a permanent upstream limitation (see "Permanent limitations" below).

**Test suite:** 243 unit + 65 integration = 308 collected, 302 executed (6 backend-specific skips for Kuzu-vs-Memgraph semantics). `ruff check`, `ruff format --check`, `mypy --strict`, and the abstraction-invariant grep all clean. Integration suite runs Memgraph via Docker (memgraph:latest v3.x) + Kuzu embedded in `tmp_path`.

**Local CI discipline:** the four local gates (ruff check / ruff format --check / mypy --strict / pytest) do not cover every GitHub Actions job. Before pushing, also run the abstraction-invariant grep locally — the exact command is in `.github/workflows/ci.yml` under the `abstraction-invariant` job. A prior-session M3 push went out with the abstraction-invariant job red for 3 commits because the local check was skipped.

**Memgraph / Docker:** Docker Desktop WSL2 integration is now enabled. Use `memgraph:latest` (v3.9), NOT `memgraph-platform:latest` (pinned at v2.14 — predates vector-index support).

## Session-Start Required Reads

**On session start, before taking any action on this project, read in order:**

1. [`docs/design.md`](docs/design.md) — the architectural design spec. Source of truth for what the system is and how it's structured. Read in full, or skim §1 + §2 + §5 for minimum orientation.
2. [`docs/implementation-plan.md`](docs/implementation-plan.md) — milestone-by-milestone build plan with TDD-style per-task instructions. Find the current cursor via `git log --oneline -n 10`.
3. `git log --oneline -n 10` and `git status --short` — current HEAD, uncommitted state.

These are non-negotiable. Skipping them means working without context that's load-bearing for every decision.

## Repository

- **Remote:** `git@github.com:giuseppecardenas/contextd.git` (HTTPS form: `https://github.com/giuseppecardenas/contextd.git`)
- **Local path:** `/home/giuseppe/src/contextd/`
- **Branch:** `main` (single-branch project; all work lands here)
- **Visibility:** currently private; flip to public at M13 per the plan
- **Auth:** global `credential.helper=store` reads `~/.git-credentials`; `git push origin main` / `git fetch origin` authenticate silently against github.com/giuseppecardenas. Rotate the PAT by editing that single line — git no longer consults `$GITHUB_PAT` (the env var is still exported in `~/.bashrc` for the `gh` CLI).

## Environment

- Platform: Windows 11 + WSL2 Ubuntu
- Python 3.11+ via a project-local `uv` venv at `/home/giuseppe/src/contextd/.venv/`
- **Activate the venv first on every session** that runs code: `source .venv/bin/activate`
- Dev deps already installed (126 packages as of 2026-04-20); no need to re-`uv pip install` unless `pyproject.toml` changes
- Required env vars (all in `~/.bashrc`): `GEMINI_API_KEY`, `VOYAGE_API_KEY`, `GITHUB_PAT`

## Architecture Invariants (non-negotiable)

These constraints are enforced in CI and are load-bearing for correctness:

- **Backend-specific modules must not be imported outside `contextd/storage/`.** Enforced by a grep step in `.github/workflows/ci.yml`. Consumers (indexer, MCP server, CLI) depend on the `GraphStore` ABC only, never on `memgraph.py` or `kuzu.py` directly. The factory in `contextd/storage/factory.py` returns the concrete via deferred imports; this is the only place backend modules are named.
- **Every edge carries `origin ∈ {inferred, structural, manual}`.** Wipe-and-replace on re-index operates only on `origin="inferred"`; structural and manual edges are preserved.
- **AI-inferred edges are ontology-validated at write time.** `Ontology.validate_edge()` rejects types not declared in `contextd/ontology/base.json`. This is the primary defence against hallucinated relationship types.
- **Section-granular mode is opt-in per corpus.** The file-granular default treats whole files as first-class nodes; section mode promotes subheadings to first-class nodes with `CONTAINS`/`PARENT_OF`/`NEXT_SIBLING` edges. See design §5.11.

## Tech Stack

- Language: Python 3.11+ (lowercase generics, walrus, `Self` type, exception groups)
- Build: `hatch` via `pyproject.toml`
- Dev env: `uv` (`uv venv`, `uv pip install -e ".[dev]"`)
- CLI: `click` + `rich` for TUI output
- MCP: `mcp` SDK (stdio transport to Claude Desktop / Cursor)
- Inference: `google-genai` SDK, Gemini Flash tier by default
- Embeddings: `voyageai` SDK, `voyage-3` (1024-dim)
- Storage: `gqlalchemy` (Memgraph Bolt) / `kuzu` (embedded), behind `GraphStore` ABC
- Parsing: `markdown-it-py` for section extraction
- Testing: `pytest` + `testcontainers-python` (Memgraph); VCR cassettes mock external APIs
- Lint / format / type: `ruff check`, `ruff format`, `mypy --strict`
- CI: GitHub Actions (lint-and-type, unit matrix on Ubuntu/macOS × Python 3.11/3.12, abstraction-invariant grep)

## Testing Discipline

**Every commit must leave the full CI triad + unit suite green.** Run before committing:

```bash
source .venv/bin/activate
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -v
```

All four must exit 0. No exceptions, no "I'll fix it in the next commit", no relaxing rules. If a check fails:

- Lint failure → fix inline; run again.
- Format failure → run `ruff format contextd tests` (write mode), re-run `--check` to confirm.
- Type failure → add the narrowest possible `type: ignore[code]` only if the failure is a genuine library limitation; otherwise fix the type error.
- Test failure → do not commit until all pass.

## Commit Conventions

- **One task = one commit.** The implementation plan's per-task commit command is the authority on commit message.
- **Message shape:** `type(scope): summary (spec §X.Y)` — e.g. `feat(storage): GraphStore ABC with typed origin property (spec §2.5.1)`.
- **No amending pushed commits.** If a fix is needed after push, land a new follow-up commit (e.g., `fix(storage): rename delete_edges edge_type→label`) rather than rewriting history.
- **Never skip hooks.** `--no-verify`, `--no-gpg-sign`, etc., are off-limits.

## Subagent Execution Contract

When this project is driven via `superpowers:subagent-driven-development`, these rules bind the implementer, spec-reviewer, and code-quality-reviewer subagents:

- **Escalate rather than deviate.** If a plan's literal instruction conflicts with a correct outcome (e.g., the plan's test regex doesn't match the library's actual output), report `BLOCKED` with a clear description of the conflict — do not silently resolve by relaxing the instruction. The controller (controller-level Claude) updates the plan via a spec-delta and re-dispatches, maintaining traceability.
- **Negative instructions are hard gates.** If the prompt says "do NOT do X", and the only path forward requires X, stop and escalate. A rationalised "the spirit of the instruction allowed X" is insufficient.
- **TDD rigor.** If a test unexpectedly passes *before* the implementation is written (Step 2 of the TDD cycle), STOP — the test is not exercising what it claims. Report `BLOCKED`.
- **CI triad is the final gate.** Every task commits only after all four checks (ruff check, ruff format --check, mypy --strict, pytest) exit 0.
- **Working-directory discipline.** Operate inside `/home/giuseppe/src/contextd/` only. Do not touch `/home/giuseppe/src/games/runeledger/` or any other directory. The Runeledger-PRD adapter work in M10 produces *example config* inside `contextd/examples/runeledger-prd/`; the external Runeledger corpus is read-only input, never a write target.

This contract exists because a prior-session subagent silently relaxed a negative instruction in Task 1.4. The substance of that deviation was defensible (pydantic v2 Literal behaviour vs. the plan's expected error message), but the *process* was not — the correct remedy was to escalate via `BLOCKED` and let the controller apply a plan-level spec-delta. The rule above is the tightened agreement.

## Key Files

- [`docs/design.md`](docs/design.md) — architectural design spec (source of truth for architecture)
- [`docs/implementation-plan.md`](docs/implementation-plan.md) — milestone-by-milestone build plan (source of truth for build order)
- `contextd/` — Python package (indexer, inference, mcp, ontology, providers, storage, migrations)
- `tests/` — unit / integration / e2e / fixtures
- `pyproject.toml` — hatch build + ruff + mypy + pytest config
- `.github/workflows/ci.yml` — CI workflow (lint-and-type, unit matrix, abstraction-invariant)

## Known Limitations / Deferred Items

The 2026-04-20 spec-delta cleanup batch (`c9bc877..803328c`) resolved 25 of 27 deferred items logged across M1–M9. See the spec-delta log for per-item details of what changed (entries #58–#70 + #73–#79 + #81–#84). The three items below intentionally deferred to future sessions — each is substantial design or research work, not a narrow fix. A fourth item, SD #71, has since been reclassified as a permanent upstream limitation (see "Permanent limitations" below).

### Open — substantial follow-up work

- **SD #72 — `QueryTranslator.translate(corpus=...)` real Cypher injection.** CLI `ask --corpus <name>` threads the flag all the way to the translator, but the translator's `if corpus:` body is a `pass` + TODO pending cross-corpus routing design. Need to pick a shape — likely `WHERE n.corpus = $corpus` appended to the first MATCH's WHERE clause, or a prompt-level directive — plus write a test. Design-gap work, not a narrow bug fix.
- **SD #74 — Stale Section garbage collection in section-mode re-index.** `phase_summarise_sections` and `phase_relate_sections` silently skip sections whose anchor no longer appears in the source (e.g., heading renamed between re-indexes). The graph retains the stale Section node. Needs a new GC phase that DETACH DELETEs Section nodes whose `id` isn't in the current parser output. Blocks M11 incremental re-index.
- **SD #80 — `cli.py` module split.** `contextd/cli.py` is ~410 lines after all the M6 cleanups; splitting into `cli/__init__.py` (group + `init` + `main`) + `cli/infra.py` (up/down/status) + `cli/corpora.py` (add-corpus/list-corpora/index) + `cli/query.py` (ask/logs/costs) would keep each module focused. Broad mechanical refactor with test-file ripple across every `tests/unit/test_cli_*.py`.

### Permanent limitations (upstream, no fix available)

- **SD #71 — Kuzu embedding columns are immutable after first write, permanently.**
  On Kuzu 0.11.3 (the final release — the project is archived upstream), vector-indexed
  `DOUBLE[1024]` columns reject `SET` after node creation. The three workarounds —
  `CALL DROP_VECTOR_INDEX` + SET + re-CREATE, DB close/reopen between steps, and DETACH
  DELETE + re-CREATE per node — are all broken in Kuzu 0.11.3: DROP leaves orphan catalog
  state (`_<N>_<idx>_UPPER does not exist in catalog` on subsequent CHECKPOINT / CREATE /
  re-CREATE_VECTOR_INDEX), and DETACH DELETE on an indexed table silently corrupts the HNSW
  index for the entire label thereafter (vector search returns empty for all subsequently-
  inserted nodes, regardless of primary key). Ground-truth verified 2026-04-21 via direct
  Kuzu experiments in a throwaway DB on the project's venv. Consequence: in section-mode
  corpora on Kuzu, `File.embedding` remains NULL — `phase_derive_file_level` sets only
  `File.summary` (see `contextd/indexer/phases.py::phase_derive_file_level`). Memgraph has
  no such restriction; File.embedding-as-centroid could be wired there independently if
  the "consistent behaviour across backends" posture is ever relaxed. The only full-
  resolution path on Kuzu would be swapping to a post-Kuzu graph DB entirely (out of
  scope).

### Spec-deltas applied in-flight (plan text still needs patching)

Items 1–24 (M1–M5 deviations: Memgraph/Kuzu connection quirks, vector-index DDL, `GraphStore` label kwargs, `_keys.py` module, FTS migration, `upsert_node`/`upsert_edge` shape, score-shape unification, `FileHasher`/`HeadingParser`/`PromptRenderer` defect fixes, fence-stripper, watcher types, `phase_embed` restructure, `phase_close` stats, `phase_relate` labels, REL `reason STRING`) were archived 2026-04-21 to trim this file — the architectural rules they produced are encoded in ABC docstrings, migration DDL, `contextd/storage/_keys.py`, and `contextd/inference/_json_body.py`. Full narrative lives in git log `e752200..1591925`; commit messages carry `spec-delta #N` references for code-comment lookups.

25. **M6.1 docker-compose template uses `memgraph:latest` (`096f537`)** — plan text specifies `memgraph/memgraph-platform:latest` which is pinned at v2.14 and predates vector-index support (inherited issue from spec-delta #2). Template changed to `memgraph/memgraph:latest` (v3.x) and port 3000 dropped (the Lab UI port the plain `memgraph` image doesn't serve).

26. **M6.1 Hatch `force-include` ships repo-root `prompts/` in the wheel (`096f537`)** — plan's `init` command copies `resources.files("prompts")` but the directory lives at repo root, NOT inside the `contextd/` package. In a wheel install `resources.files("prompts")` would `ModuleNotFoundError`; the plan's `try: ... except Exception: pass` wrapper masked the failure silently, which in turn would have caused `contextd index` to break with "no templates" on a fresh install. Resolution: added `[tool.hatch.build.targets.wheel.force-include] "prompts" = "prompts"` to `pyproject.toml`, dropped the bare `try/except`, dropped the dead `prompts_src` assignment. Tested via `tests/unit/test_cli_init.py::test_init_creates_layout` which now asserts `~/.contextd/prompts/summarise.md` exists after `init`.

27. **M6.2 `up` kuzu branch creates `db_path.parent`, not `db_path` (`075164e`)** — plan says `db_path.mkdir(parents=True, exist_ok=True)` but Kuzu stores the DB in a single FILE (spec-delta #3), not a directory. Creating a directory at `db_path` breaks `KuzuBackend.connect()` with "Database path cannot be a directory". Backend internally does `db_path.parent.mkdir(parents=True, exist_ok=True)`; CLI `up` now matches. Test `test_up_kuzu_creates_db_path` uses a nested path (`{home}/nested/graph`) so the parent dir is actually created by `up` (not pre-existing from fixture setup — follow-up `9ebc26e` made the test falsifiable).

28. **M6.3 `_load_cfg()` helper extracted (`a431f7e`)** — the 4-line `Config.load(path) if path.exists() else Config.load_default()` ternary was duplicated across `up`, `down`, `status` per the plan. M6.2 code-review flagged the duplication. `_load_cfg() -> Config` helper placed near the top of `cli.py` using `TYPE_CHECKING` to avoid a cycle; `from contextd.config import Config` stays lazy inside the helper body. `index` (M6.4) and `ask` (M6.5) adopted the helper on landing.

29. **M6.5 `ask` command imports `QueryTranslator` via deferred-and-typed-ignored import (`56b1541`, `9ffa069`)** — `contextd.inference.translate` does not exist yet (built in M8). The import lives inside the `ask` function body so CLI registration and `contextd ask --help` work today; `contextd ask "<question>"` raises `ImportError` at invocation until M8 lands. Annotated `# type: ignore[import-untyped]` to keep mypy-strict green. Initial attempt at an additional `# noqa: PGH003` tripped `RUF100` (project's ruff select list doesn't include `PGH`); follow-up `9ffa069` dropped the dead `noqa`. When M8 lands, remove the `type: ignore` comment and exercise the command with a real integration test.

30. **M7.2 `describe_project` double-WHERE fix (`a32f6e2`)** — plan's verbatim Cypher had two consecutive `WHERE` clauses after a single `MATCH`: `WHERE n.corpus = $corpus` (conditional f-string expansion) followed by `WHERE n.summary IS NOT NULL`. Both Memgraph and Kuzu parse this as a syntax error. Fixed by collecting predicates into a `filters: list[str]` and AND-joining them into a single `WHERE`. Params dict populated conditionally to match. Verified passing on both backends via `tests/integration/test_mcp_tools.py::test_describe_project_returns_summaries`.

31. **M7.2 integration test `upsert_edge` label kwargs (`a32f6e2`)** — plan's `tests/integration/test_mcp_tools.py::test_describe_project_returns_summaries` called `backend.upsert_edge("a.md", "b.md", "REFERENCES", origin="structural")` which fails on Kuzu per spec-delta #4 (schema-first REL tables require explicit src/dst labels). Added `src_label="File", dst_label="File"` to the call. Plan-verbatim elsewhere.

32. **M7.3 `related` tool descriptor depth clamped 1-5 (`a52b8f7`)** — plan's `related` Tool descriptor declared `depth: {"type": "integer", "default": 2}` with no min/max. Caller-controlled depth combined with the f-string interpolation in `tools.related` meant an MCP caller could pass `depth=1000` and hang Memgraph or crash Kuzu. Added `"minimum": 1, "maximum": 5"` to the JSON schema and updated the description to "(1-5)". The 5-hop ceiling aligns with design §5's "reasonable neighbourhood" framing. Function-level defensive clamp is a deferred Known-Limitation item above.

33. **M7.3 narrow mypy-strict ignores on mcp SDK decorators (`a52b8f7`)** — `@server.list_tools()` and `@server.call_tool()` decorators are untyped in mcp v1.27. Mypy-strict requires `# type: ignore[no-untyped-call,untyped-decorator]` on `list_tools` (double error: untyped decorator AND untyped call of decorator factory) and `# type: ignore[untyped-decorator]` on `call_tool`. Asymmetric but correct — matches what mypy actually reports. Remove both when mcp SDK ships type stubs.

34. **M8.1 broken corpus injection replaced with TODO no-op (`e5f5ef8`)** — plan's `cypher.replace("MATCH", f"MATCH ", 1)` inside `if corpus:` just adds a trailing space; it doesn't inject a corpus filter. `if "corpus" not in cypher:` guard was plausibly meant to skip the replacement when the LLM already filtered, but the replacement itself was incomplete. Replaced the body with `pass` + TODO comment deferring the real filter shape to M9/M10 (cross-corpus routing design). `corpus` kwarg kept on `translate()` signature because `cli.py::ask` plumbs it through from `--corpus`.

35. **M8.1 empty/prose-only translator response raises (`b10a02a`)** — code-quality review caught that `_extract_cypher` returned `""` on prose-only LLM output, which then passed `assert_read_only` silently and hit the backend as an opaque syntax error. Added `if not cypher: raise ValueError(...)` with a message naming the failure mode. Consistent with the tolerant-but-traceable parsing posture of M4.6/4.7.

36. **M8.1 dead `# type: ignore[import-untyped]` on `cli.py:283` removed (`e5f5ef8`)** — once `contextd/inference/translate.py` existed with real types, mypy-strict's `warn_unused_ignores` fired `unused-ignore` on the now-dead directive. Removal was forced by CI, not a planned deviation. Spec-delta #29 predicted this: "When M8 lands, remove the `type: ignore` comment." Done.

37. **M9.1 `phase_enumerate_sections` pre-computes Section embeddings at CREATE time (`971c9e0`)** — plan's separation of enumeration from embedding (M9.2's `phase_embed_sections` did `SET s.embedding = $vec`) fails on Kuzu because `Section.embedding` is in `contextd/storage/_keys.py::IMMUTABLE_AFTER_CREATE_BY_LABEL`. Resolution mirrors M5.4 spec-delta #21 exactly: `phase_enumerate_sections` gains an `embedder: EmbeddingProvider` parameter, batch-computes all Section body embeddings upfront, and passes them to `upsert_node("Section", {..., "embedding": vec})` at CREATE time. `phase_embed_sections` becomes an accounting-only stub (count of Section rows, no SET). `pipeline.py run_bootstrap` section branch updated to pass `embedder` to `phase_enumerate_sections`.

38. **M9.1/9.2 structural + inferred edges gain label kwargs for section phases (`971c9e0`, `7a96290`)** — per spec-deltas #4 and #23, Kuzu requires `src_label` / `dst_label` on `upsert_edge` and `src_label` on `delete_edges`. Applied uniformly:
    - `phase_enumerate_sections` — CONTAINS(`File→Section`), PARENT_OF(`Section→Section`), NEXT_SIBLING(`Section→Section`).
    - `phase_relate_sections` — `delete_edges(..., src_label="Section")` + `upsert_edge(..., src_label="Section", dst_label=rel.target_type)`.
    Section→non-Section inferred edges may still hit Kuzu REL-table FROM/TO constraints (logged as deferred).

39. **M9.2 `phase_derive_file_level` skips File.embedding (`7a96290`)** — plan's `SET f.summary = $summary, f.embedding = $embedding` fails on Kuzu because `File.embedding` is in `IMMUTABLE_AFTER_CREATE_BY_LABEL`. Since the phase layer cannot branch on Kuzu-vs-Memgraph without violating the abstraction invariant, the resolution is to skip `f.embedding` entirely: only `SET f.summary`. File.embedding stays NULL in section-mode corpora on both backends (consistency > duplication). `_centroid` helper retained for when a migration flips the column to mutable or a DETACH-DELETE-reCREATE pattern lands. Query simplified from `collect({summary, embedding})` map-literal to `collect(s.summary)` — reduces Kuzu map-literal exposure surface.

40. **M9.2 integration-test coverage expansion (`334c682`)** — plan-prescribed `test_section_granular_bootstrap` only asserted Section titles, leaving `phase_summarise_sections` writes, `phase_derive_file_level` File.summary derivation, and `phase_relate_sections` Delta-B label kwargs all unexercised at runtime. Added read-back assertions for Section.summary + File.summary, plus a new `test_section_granular_inferred_edges` that seeds Section→Section REFERENCES via a side_effect'd inferrer and asserts `count(r)==2` — runtime-validating the `src_label="Section", dst_label="Section"` kwargs on both backends.

41. **M9 subagent-driven push discipline note** — the M9.1 and M9.2 implementer subagents pushed to `origin/main` as part of their tasks. The controller prompt did not explicitly prohibit pushing, but previous milestones had implicit controller-only push convention. Accepted post-hoc (local CI gates all green, no damage), but future implementer prompts should explicitly state "do not push — leave pushing to the controller" to maintain single-point push discipline.

42. **2026-04-20 spec-delta cleanup batch (`c9bc877..803328c`, 20 commits).** Between M9 close and M10 start, 25 of 27 deferred items logged across M1–M9 were resolved in a single controller-driven sweep. Concrete changes:

    - **Defensive `__init__` guards** (SD #58/#59/#60/#61/#62/#63): `TokenChunker` raises on `overlap_tokens >= max_tokens` (prevents infinite loop); `FileHasher._load_state` validates the JSON shape with `isinstance` instead of an unchecked `cast`; `DebouncedQueue` rejects `window_seconds <= 0` and calls `path.resolve()` on `add()` for deterministic dedup; `CorpusWatcher.start()` raises `RuntimeError` on second call + `stop()` logs a warning when the 5 s join times out + module docstring names the dispatch-thread callback precondition; `CheckpointStore.save()` is now atomic (`tmp.write_text` + `os.replace`) with per-field `isinstance` guards in `load()` + corpus-name validation; `Ontology` fields converted to `frozenset`/`MappingProxyType`/`tuple` so consumers cannot mutate shared state, and `validate_node` dropped its unused `properties: dict[str, Any]` parameter.
    - **Error-path hardening** (SD #64/#65/#66/#67): `Summariser` raises descriptive KeyError/TypeError on missing or wrong-typed `summary` field instead of late failures via `cast`; `PromptRenderer` now guards `template` against path traversal via `is_relative_to` before `read_text()`; `translate.md` allow-list expanded to name DETACH DELETE / DROP / FOREACH / CALL-with-side-effects, and `_FORBIDDEN` regex gained `FOREACH` plus a negative-lookbehind `(?<![.\w])` so `RETURN n.set AS prop` no longer false-positives; `cli.py` `up` pre-checks `shutil.which("docker")` and `ask` wraps `translator.translate` + `store.exec_read` in `click.ClickException` so missing-Docker / translator-failure / backend-error paths render as "Error: …" instead of Python tracebacks.
    - **Architectural unification** (SD #68/#69): `contextd/indexer/phases.py::_infer_key` and `contextd/indexer/entity_resolver.py::resolve` both delegate to `contextd.storage._keys.primary_key_for` — the authoritative label→PK map that mirrors the migration DDL. Unknown labels raise `ValueError`; `phase_relate`/`phase_relate_sections` catch and skip so hallucinated targets don't abort the batch. New `contextd/_paths.py::contextd_home()` replaces the module-import-time `CONTEXTD_HOME` capture in `contextd.cli`; every CLI test dropped its `importlib.reload(contextd.cli)` dance, and `contextd.mcp_server` imports from `_paths` instead of `cli` (~200 ms click/rich overhead gone from MCP startup).
    - **MCP + CLI feature tightening** (SD #73/#75/#76/#77/#78/#79/#81/#82/#83/#84): `phase_enumerate_sections` now writes real MD5 hashes via `FileHasher` (unblocks M11 incremental in section mode); coverage gaps filled for `Ontology.with_aliases` error path + `is_git_busy` branches + `add-corpus` duplicate-guard body-unchanged + `list-corpora` missing-vs-empty branches + `index --bootstrap` CLI path; `is_git_busy` resolves `.git` gitfiles for worktrees + submodules + `--separate-git-dir`; `contextd/mcp_server.py` wraps `stdio_server` in `try/finally: store.close()`, switched every tool result from `str(obj)` to `json.dumps(obj, default=str)`, catches tool-dispatch exceptions into `{"error": "…"}`, and promoted the inline tool list to a module-level `TOOL_DESCRIPTORS` constant so unit tests can assert the 8-tool surface; `tools.related` gained a defensive function-level depth clamp; `tools.describe_project` narrowed to `:File` + `tools.search` docstring pruned; dead `@field_validator("backend")` on `StorageConfig` removed, `phase_embed_sections` signature shrunk to `(corpus_cfg, store)`, `_centroid` deleted; `enumerate_corpus_files` skips `.git`/`.venv`/`__pycache__`/`node_modules` and symlinks by default; `logs --follow` traps `KeyboardInterrupt`; translator fence regex accepts any language tag + keyword-line fallback slices from first keyword to end-of-text (preserves multi-line continuations); `prompts/` moved from repo-root to `contextd/prompts/` (Hatch `force-include` replaced with ordinary `include`); section phases cache `ParsedSection` per-file; `index` body refactored to `_build_pipeline_deps` + `--estimate-only` now counts UTF-8 chars not bytes.
    - **New Kuzu migration 2** (SD #70): `Corpus.node_count` + `Corpus.edge_count` columns added via `ALTER TABLE Corpus ADD … INT64`. `phase_close` now persists both counts at registration time. Memgraph's sibling migration is a no-op (schema-free) maintained for `schema_version` parity.

    Test suite grew 183 → 243 unit tests (60 new) across this batch; integration stayed at 65 with two new assertions (Corpus-stats read-back, File.hash 32-hex-char). All four CI gates + the abstraction-invariant grep stayed green throughout. Four items deferred at the time — see **Known Limitations / Deferred Items** above for SD #72/#74/#80 rationales; SD #71 was subsequently reclassified as a permanent upstream Kuzu limitation (see "Permanent limitations" above).
