# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Contextd is a locally-hosted GraphRAG knowledge layer. It indexes markdown, code, and structured-data corpora into a hybrid graph + vector store (Memgraph or KùzuDB, pluggable via config), generates per-file (or per-section) summaries and AI-inferred typed relationships via external LLM APIs (Gemini for inference, Voyage AI for embeddings), and exposes the result to AI assistants through a Model Context Protocol server. The tool is designed for single-user local use — everything runs on one machine; no multi-user concerns.

## Current Phase: Active Implementation

The project is mid-build against a detailed milestone plan. The plan drives build order deterministically — do not skip or reorder milestones.

**As of 2026-04-20 (HEAD `088069b`, all pushed to origin/main):**

- **M0** (repo scaffold) — complete (`e752200`). CI green.
- **M1** (config + ontology foundations) — complete 5/5. Closing commit `6551a71`.
- **M2** (external AI providers) — complete 5/5. `InferenceProvider`/`EmbeddingProvider` ABCs, `GeminiProvider` with retry + BLOCK_NONE safety + usage accounting, `VoyageProvider` with batched embedding + retry, factory with env-var-driven keys, append-only `CostLog`. Closing commit `f1cecb3`.
- **M3** (storage backends) — complete 4/4 with post-closure bug fixes. `GraphStore` factory + forward-only `MigrationRunner`, `MemgraphBackend` (Bolt via gqlalchemy) + baseline migration, `KuzuBackend` (embedded v0.11) + baseline migration, parametrized cross-backend integration suite. Closing commit `fd6d477`; subsequent hardening in `eb28a41`, `cabe6f7`, `7d1c285`, `088069b`, `cab529f` (see spec-delta log below).

**Cursor:** M4 Task 4.1 (file hasher with hash-based change detection).

**Test suite:** 44 unit + 25 integration = 69 collected, 68 executed (1 skipped for Kuzu distance-vs-similarity threshold semantics). `ruff check`, `ruff format --check`, `mypy --strict` all clean. Integration suite runs Memgraph via Docker (memgraph:latest v3.x) + Kuzu embedded in `tmp_path`.

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
- **Auth for push:** `GITHUB_PAT` env var (exported in `~/.bashrc`). Origin URL stays clean — inline the PAT per-push only: `git push "https://giuseppecardenas:${GITHUB_PAT}@github.com/giuseppecardenas/contextd.git" main:main`.

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

### Non-blocking — revisit during M4/M5 pipeline work

- **Task 1.3 code-quality review flagged two items** to address before backends cache `Ontology`:
  - `Ontology`'s mutable fields (`dict`/`set`) are exposed directly; a consumer could mutate `onto.edge_types.add(...)` and silently corrupt validation. Fix: `frozenset` for sets, `types.MappingProxyType` for `node_types`.
  - `validate_node(node_type, properties)` ignores `properties` entirely. Either drop the param (YAGNI) or add real property-key validation.
- **Task 1.4 `@field_validator("backend")` is dead code** — pydantic v2's `Literal["memgraph", "kuzu"]` check fires first. Harmless but removable.
- **Test for `Ontology.with_aliases` error path** (raises on unknown target) is not present. Add before integration tests rely on it.

### Spec-deltas applied in-flight (plan text still needs patching)

1. **Task 2.2** — plan's test used `genai_errors.APIError(code=429, message="quota", ...)` but the real `google-genai` API uses `response_json={"error": {"message": ...}}` (no `message` kwarg). Adapted the test (`577be77`); retry-path semantics unchanged.
2. **Task 3.2 (Memgraph)**
   - `testcontainers.memgraph.MemgraphContainer` does not exist in `testcontainers-python 4.14.2`. Use generic `DockerContainer("memgraph/memgraph:latest").with_exposed_ports(7687).waiting_for(LogMessageWaitStrategy("You are running Memgraph"))` instead.
   - `memgraph-platform:latest` is pinned at v2.14.1, which predates vector-index support (introduced in v2.18). Use `memgraph:latest` (v3.x).
   - Memgraph vector-index `CONFIG` metric must be `"cos"`, not `"cosine"`.
   - `gqlalchemy`'s `Memgraph` class has no `drop_connection()`; `close()` just releases the reference and lets the cached Bolt connection GC.
   - Registered `integration` pytest marker in `pyproject.toml` and a gqlalchemy-specific mypy override (untyped third-party lib).
3. **Task 3.3 (Kuzu 0.11.3)**
   - Kuzu stores the DB in a single file, not a directory — create the parent dir, pass the file path.
   - Kuzu is schema-first: `MigrationRunner._current_applied()` runs before migration 1, so the `Meta` table is bootstrapped in `KuzuBackend.connect()` with `CREATE NODE TABLE IF NOT EXISTS Meta(...)`. The migration 1 DDL omits `Meta` on Kuzu.
   - Runner uses a fixed singleton PK `schema_version: 0` so `MERGE (m:Meta {schema_version: 0})` satisfies Kuzu's primary-key requirement while remaining valid on Memgraph.
   - `_record_applied` inlines the migration id as a literal in Cypher; Python `int` binds as `INT8` and Kuzu rejects `INT8 ∪ INT64[]` list-concat. `CAST(... AS INT64)` works on Kuzu but not Memgraph; `toInteger(...)` works on Memgraph but not Kuzu — literal is the portable path.
   - `embedding` must be `DOUBLE[1024]` (fixed-size ARRAY), not `DOUBLE[]` list, for `CREATE_VECTOR_INDEX`.
   - Drop `dim := 1024` from `CREATE_VECTOR_INDEX`; dimension is inferred from the column type.
   - Backtick-escape reserved words `start` and `end` in `WorkSession` DDL.
4. **Task 3.4 (cross-backend tests)** — ABC change: `GraphStore.upsert_edge` and `delete_edges` now take optional `src_label` / `dst_label` kwargs. Memgraph treats them as advisory MATCH-narrowing hints; Kuzu requires them (REL tables have fixed FROM/TO label pairs, and node tables don't share a common key-property shape across labels). Kuzu raises `ValueError` if the labels are omitted. The plan's `test_delete_edges_by_origin` uses `REFERENCES` + `BELONGS_TO` but Kuzu's `BELONGS_TO` is declared `FROM File TO Ticket`, not `File→File` — the test was rewritten to use `REFERENCES` + `CONTAINS (File→Section)` which is valid on both backends.

5. **`delete_edges` contract tightening (`cabe6f7`)** — both backends now raise `ValueError` when both `origin` and `label` are None. An unfiltered wipe would violate the design invariant that wipe-and-replace operates only on `origin="inferred"`; callers must opt in. The ABC docstring prescribes this as required behaviour.

6. **New `contextd/storage/_keys.py` module (`cabe6f7`, `088069b`)** — centralises `PRIMARY_KEY_BY_LABEL` (the label-to-PK-property map mirroring the migration DDL) and `IMMUTABLE_AFTER_CREATE_BY_LABEL` (properties Kuzu rejects via `SET` after node creation — `File.embedding`, `Section.embedding`). Both backends import from here. Drift with the migration DDL is currently caught only by integration tests; a dedicated unit test is in the backlog. Plan does not mention this module.

7. **Kuzu `upsert_node` is two-phase check-then-CREATE-or-SET (`088069b`)** — plan's single `MERGE (n:L {all_props})` shape has two problems on Kuzu: (a) re-upsert with a changed non-PK property trips the PK uniqueness constraint; (b) vector-indexed columns cannot be assigned via `SET` after creation (even in `ON CREATE SET`). Resolution: if the node does not exist, `CREATE` with all props inline; if it does, `SET` only the mutable (non-PK, non-vector-indexed) properties. Correct under Kuzu's single-writer capability; would need a lock if a multi-writer backend reuses this code.

8. **Memgraph FTS migration + backend rewritten (`088069b`)** — plan's `CREATE INDEX ON :File(summary)` creates a B-tree label-property index, which the `text_search` procedure cannot find by name. Changed to `CREATE TEXT INDEX File_summary_ft ON :File` (Lucene-backed). The backend's `full_text_search` switched from `text_search.search(idx, q)` (requires Lucene-expression syntax) to `text_search.search_all(idx, q)` (plain keyword over all indexed properties). search_all does not return a score; the ABC signature is unchanged but the result dict has `node` only, no `score`. M7 callers wanting ranked output must pass a Lucene expression via `exec_read` directly.

9. **Memgraph `vector_search` threshold clause (`088069b`)** — plan's `YIELD ... WHERE ...` form parses the WHERE token outside a valid Cypher position. The backend now re-projects via `WITH node, score WHERE score >= <threshold>` before RETURN. The threshold is currently f-strung into the Cypher; parameterising it is in the backlog (float injection surface + `nan`/`inf` bad-format risk).

10. **Kuzu `vector_search` arity (`088069b`)** — plan's `CALL QUERY_VECTOR_INDEX(idx, $q, $k)` has the wrong arity. Kuzu's signature is `QUERY_VECTOR_INDEX(table, idx, query, k)` — 4 args. Also `k` is inlined as a literal because Python `int` binds as `INT8` and the procedure expects `INT64` (same pattern as `_record_applied`).

11. **Kuzu `upsert_edge` per-property SET + REL-table schema (`7d1c285`)** — plan's `SET r.origin = $origin` silently dropped every other edge property. The backend now enumerates one `SET r.<k> = $<k>` per supplied property. Kuzu REL tables are schema-first, so any property beyond the declared columns surfaces as a binder exception — `REFERENCES` and `BELONGS_TO` gained `confidence DOUBLE` (nullable) to carry the inferred-edge review signal; adding further edge properties requires a migration.

12. **CI `abstraction-invariant` grep narrowed to `contextd/` (`eb28a41`)** — originally `grep ... contextd/ tests/ --exclude-dir=storage`, which caught the plan-prescribed `from contextd.storage.memgraph import ...` in integration test files and failed CI for three consecutive commits on main before being fixed. The invariant's intent is to protect runtime consumers (indexer, MCP, CLI) from coupling to a concrete backend; integration tests are legitimately backend-specific and stay outside the fence.

13. **pytest filterwarnings suppression (`088069b`)** — `gqlalchemy.exceptions.GQLAlchemySubclassNotFoundWarning` is raised whenever gqlalchemy materialises a node whose label has no Python ORM class. `GraphStore` deals in dicts, not ORM models; this warning is structural and harmless. Added to `[tool.pytest.ini_options].filterwarnings` alongside the `"error"` strictness default.

14. **`vector_search` score-shape unification (`323c179`)** — during M4 Task 4.4 (`EntityResolver`) code review, we found the backends diverged on `vector_search` return shape: Memgraph returned `"score"`, Kuzu returned `"distance"`. The ABC docstring documented the divergence. `EntityResolver.resolve` reads `top.get("score", 0.0)` which defaulted to 0.0 on Kuzu and always skipped dedup. Normalised both backends to return `{"node", "score"}` where `score` is cosine similarity in `[0, 1]`. The Kuzu backend's internal server-side filter still operates on distance; only the public return shape changed.
