# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Contextd is a locally-hosted GraphRAG knowledge layer. It indexes markdown, code, and structured-data corpora into a hybrid graph + vector store (Neo4j Community), generates per-file (or per-section) summaries and AI-inferred typed relationships via external LLM APIs (Gemini for inference, Voyage AI for embeddings), and exposes the result to AI assistants through a Model Context Protocol server. The tool is designed for single-user local use — everything runs on one machine; no multi-user concerns.

## Current Phase: Active Implementation

The project is mid-build against a detailed milestone plan. The plan drives build order deterministically — do not skip or reorder milestones.

**As of 2026-04-24 (HEAD `fe3650d`; all pushed to origin/main):**

- **M0** (repo scaffold) — complete (`e752200`). CI green.
- **M1** (config + ontology foundations) — complete 5/5. Closing commit `6551a71`.
- **M2** (external AI providers) — complete 5/5. `InferenceProvider`/`EmbeddingProvider` ABCs, `GeminiProvider` (retry + BLOCK_NONE + usage accounting), `VoyageProvider` (batched + retry), factory, `CostLog`. Closing commit `f1cecb3`.
- **M3** (storage backends) — complete 4/4. `GraphStore` ABC + forward-only `MigrationRunner`, `MemgraphBackend` (Bolt via gqlalchemy). Note: KùzuDB excised in M11.9. Closing commit `fd6d477`.
- **M4** (indexing primitives) — complete 7/7. `FileHasher`, `TokenChunker`, `HeadingParser`, `EntityResolver`, `PromptRenderer` + three default templates, `Summariser`, `RelationshipInferrer`. Closing commit `a3f3bd6`.
- **M5** (indexing pipeline) — complete 5/5. `DebouncedQueue`, `CorpusWatcher`, `is_git_busy`, 5-phase bootstrap (`phase_enumerate`/`embed`/`summarise`/`relate`/`close`) + `run_bootstrap`, `CheckpointStore`. Closing commit `80abfde`.
- **M6** (CLI) — complete 5/5. Click + rich CLI (`contextd/cli/`) with `init`/`up`/`down`/`status`/`add-corpus`/`list-corpora`/`index`/`ask`/`logs`/`costs`. Closing commits `56b1541`.
- **M7** (MCP server) — complete 3/3. `ReadOnlyGuardError` + `assert_read_only`. 8 MCP tools in `contextd/mcp/tools.py`; stdio server at `contextd/mcp_server.py`. Closing commit `a52b8f7`.
- **M8** (NL → Cypher) — complete 1/1. `QueryTranslator` in `contextd/inference/translate.py`; `_extract_cypher` raises on empty/prose-only output. Closing commits `e5f5ef8` + `b10a02a`.
- **M9** (Section-granularity) — complete 2/2. `phase_enumerate_sections` + Section nodes with `CONTAINS`/`PARENT_OF`/`NEXT_SIBLING` edges; `run_bootstrap` branches on `corpus.corpus.granularity`. Closing commits `971c9e0` + `7a96290` + `334c682`.
- **M10** (Acme-PRD adapter + critical wiring) — complete 13/13. Wired all unconsumed corpus TOML config surfaces: ontology aliases (`M10.3`), overrides loader (`M10.4`), prompt-override (`M10.5`), per-corpus MCP tools (`M10.6`), `add-corpus --from TEMPLATE` (`M10.7`), non-markdown routing in section-granular corpora (`M10.9`). Key deviation: plan's `corpus.toml` paths must be template-parent-relative (not repo-root-anchored) after `8fa7f76` fix. Commit range `94e3af3..ab62d80` (18 commits).
- **Appendix-M11** (Neo4j backend stand-up + Kuzu excision) — complete 9/9. Neo4j Community 5.x replaces KùzuDB. `Neo4jBackend` (Bolt via `neo4j>=5.15`), baseline migration (constraints + vector + FTS indexes), docker-compose `--profile memgraph`/`--profile neo4j`, default `backend="neo4j"` (M11.8), Kuzu fully excised (M11.9). Commit range `57fdb39..4d69272`.
- **Plan-M11** (Windows / WSL2 integration) — complete 1/1. Four PowerShell wrappers under `scripts/windows/` forwarding to `wsl -d $Distro -- contextd <cmd>`. Closing commit `c9db050`.
- **M12** (End-to-end tests + CI finalisation) — complete 2/2. `tests/e2e/test_full_cycle.py` parametrized on both backends; `backend` fixture in `tests/conftest.py`; `integration` + `e2e` CI jobs added. Closing commits `7f059da..f0077be`.

- **M14** (Incremental indexer daemon) — complete 12/12. `contextd-indexer` console script + `contextd/daemon.py` main loop; `CorpusWatcher`/`DebouncedQueue`/`is_git_busy`/`CheckpointStore`/`FileHasher` wired into production; `PendingUpsertBuffer` for crash-safe retry; IPC socket (`contextd/daemon_ipc.py`) for `contextd status` enrichment; `contextd index --incremental` one-shot scan; `allowed_branches` gate; `incremental_workers` + log-rotation config fields. 417 unit tests. Commit range `0b1f255..fe3650d`.

**Cursor:** plan-M0 through plan-M12 + M14 complete, plus appendix-M11 (Neo4j). Next: M13 (docs, README, publish).

**Spec-delta cleanup:** between M9 and M10, a 20-commit sweep resolved 25 of 27 deferred items (git log `c9bc877..803328c`). Three remaining items resolved pre-M11: SD #72 (corpus injection, `6a5c4a0`), SD #74 (stale Section GC, `f65abab`), SD #80 (`cli.py` module split, `5a278b0`). SD #71 (Kuzu embedding mutability) resolved by Kuzu excision in M11.

**Test suite:** `ruff check`, `ruff format --check`, `mypy --strict`, and the abstraction-invariant grep all clean. Integration + e2e suites run Neo4j Community via Docker (`neo4j:5` image) and also run in CI on every push/PR to main. Integration + e2e tests require Docker; they fail at fixture setup if the Docker daemon is unreachable (expected in dev shells without Docker — use `pytest tests/unit` for fast iteration). The `backend` fixture is single-valued (Neo4j) since Memgraph was excised.

**Local CI discipline:** the four local gates (ruff check / ruff format --check / mypy --strict / pytest) do not cover every GitHub Actions job. Before pushing, also run the abstraction-invariant grep locally — the exact command is in `.github/workflows/ci.yml` under the `abstraction-invariant` job. A prior-session M3 push went out with the abstraction-invariant job red for 3 commits because the local check was skipped.

**Neo4j / Docker:** Docker Desktop WSL2 integration is enabled. The backend runs as `neo4j:5` via `contextd up` (`docker compose --profile neo4j`). Memgraph was excised — Neo4j Community is the sole storage backend; the `GraphStore` ABC, factory, and abstraction-invariant grep are kept so a second backend could be re-added without recoupling consumers.

## Session-Start Required Reads

**On session start, before taking any action on this project, read in order:**

1. [`docs/architecture.md`](docs/architecture.md) — the architecture overview (three-layer decomposition, bootstrap phases, storage abstraction). Source of truth for how the system is structured. Pair with [`README.md`](README.md) for the user-facing overview.
2. `git log --oneline -n 10` and `git status --short` — current HEAD, uncommitted state.

These are non-negotiable. Skipping them means working without context that's load-bearing for every decision.

## Repository

- **Remote:** `git@github.com:giuseppecardenas/contextd.git` (HTTPS form: `https://github.com/giuseppecardenas/contextd.git`)
- **Local path:** `~/src/contextd/`
- **Branch:** `main` (single-branch project; all work lands here)

## Environment

- Platform: Windows 11 + WSL2 Ubuntu
- Python 3.11+ via a project-local `uv` venv at `~/src/contextd/.venv/`
- **Activate the venv first on every session** that runs code: `source .venv/bin/activate`
- Dev deps already installed (126 packages as of 2026-04-20); no need to re-`uv pip install` unless `pyproject.toml` changes
- Required env vars: `GEMINI_API_KEY`, `VOYAGE_API_KEY`

## Architecture Invariants (non-negotiable)

These constraints are enforced in CI and are load-bearing for correctness:

- **Backend-specific modules must not be imported outside `contextd/storage/`.** Enforced by a grep step in `.github/workflows/ci.yml`. Consumers (indexer, MCP server, CLI) depend on the `GraphStore` ABC only, never on `neo4j.py` directly. The factory in `contextd/storage/factory.py` returns the concrete via a deferred import; this is the only place the backend module is named.
- **Every edge carries `origin ∈ {inferred, structural, manual}`.** Wipe-and-replace on re-index operates only on `origin="inferred"`; structural and manual edges are preserved.
- **AI-inferred edges are ontology-validated at write time.** `Ontology.validate_edge()` rejects types not declared in `contextd/ontology/base.json`. This is the primary defence against hallucinated relationship types.
- **Section-granular mode is opt-in per corpus.** The file-granular default treats whole files as first-class nodes; section mode promotes subheadings to first-class nodes with `CONTAINS`/`PARENT_OF`/`NEXT_SIBLING` edges. See `docs/architecture.md`.

## Tech Stack

- Language: Python 3.11+ (lowercase generics, walrus, `Self` type, exception groups)
- Build: `hatch` via `pyproject.toml`
- Dev env: `uv` (`uv venv`, `uv pip install -e ".[dev]"`)
- CLI: `click` + `rich` for TUI output
- MCP: `mcp` SDK (stdio transport to Claude Desktop / Cursor)
- Inference: `google-genai` SDK, `gemma-4-31b-it` by default (Gemma family; 15 RPM free-tier quota is the typical binding constraint, not per-call latency)
- Embeddings: `voyageai` SDK, `voyage-4-large` (1024-dim)
- Storage: `neo4j-driver` 5.x (Neo4j Community), behind `GraphStore` ABC
- Parsing: `markdown-it-py` for section extraction
- Testing: `pytest` + `testcontainers-python` (Neo4j); VCR cassettes mock external APIs
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
- **Working-directory discipline.** Operate inside `~/src/contextd/` by default. The Acme-PRD adapter was moved out of contextd (see log entry #48); its config + prompts now live at `/home/you/src/acme/.contextd/`, which is a legitimate write target *for that adapter's own files* — the acme corpus content itself is still read-only input. Do not touch any other directory without explicit user instruction.

This contract exists because a prior-session subagent silently relaxed a negative instruction in Task 1.4. The substance of that deviation was defensible (pydantic v2 Literal behaviour vs. the plan's expected error message), but the *process* was not — the correct remedy was to escalate via `BLOCKED` and let the controller apply a plan-level spec-delta. The rule above is the tightened agreement.

## Key Files

- [`docs/architecture.md`](docs/architecture.md) — architecture overview (source of truth for architecture)
- `contextd/` — Python package (indexer, inference, mcp, ontology, providers, storage, migrations)
- `tests/` — unit / integration / e2e / fixtures
- `pyproject.toml` — hatch build + ruff + mypy + pytest config
- `.github/workflows/ci.yml` — CI workflow (lint-and-type, unit matrix, abstraction-invariant)

## Known Limitations / Deferred Items

All 27 deferred items from M1–M9 are resolved. Items 1–24 (M1–M5 deviations) archived 2026-04-21 — outcomes encoded in ABC docstrings, migration DDL, `_keys.py`, `_json_body.py`; git log `e752200..1591925`. Items 25–48 below are spec-deltas applied in-flight (plan text still needs patching).

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

37. **M9.1 `phase_enumerate_sections` pre-computes Section embeddings at CREATE time (`971c9e0`)** — plan's separation of enumeration from embedding (M9.2's `phase_embed_sections` did `SET s.embedding = $vec`) was restructured so that `phase_enumerate_sections` gains an `embedder: EmbeddingProvider` parameter, batch-computes all Section body embeddings upfront, and passes them to `upsert_node("Section", {..., "embedding": vec})` at CREATE time. `phase_embed_sections` becomes an accounting-only stub (count of Section rows, no SET). `pipeline.py run_bootstrap` section branch updated to pass `embedder` to `phase_enumerate_sections`. (Originally motivated by Kuzu's IMMUTABLE_AFTER_CREATE constraint; the embed-at-CREATE pattern is retained with Neo4j because it simplifies the phase boundary and avoids a separate SET pass.)

38. **M9.1/9.2 structural + inferred edges gain label kwargs for section phases (`971c9e0`, `7a96290`)** — `src_label` / `dst_label` on `upsert_edge` and `src_label` on `delete_edges` are advisory on schema-free backends (Memgraph, Neo4j) but required for correctness. Applied uniformly:
    - `phase_enumerate_sections` — CONTAINS(`File→Section`), PARENT_OF(`Section→Section`), NEXT_SIBLING(`Section→Section`).
    - `phase_relate_sections` — `delete_edges(..., src_label="Section")` + `upsert_edge(..., src_label="Section", dst_label=rel.target_type)`.

39. **M9.2 `phase_derive_file_level` skips File.embedding (`7a96290`)** — plan's `SET f.summary = $summary, f.embedding = $embedding` was simplified to only `SET f.summary`, keeping both backends consistent. File.embedding stays NULL in section-mode corpora. `_centroid` helper was retained but later deleted in the M9 spec-delta cleanup batch (SD #84). Query simplified from `collect({summary, embedding})` map-literal to `collect(s.summary)`.

40. **M9.2 integration-test coverage expansion (`334c682`)** — plan-prescribed `test_section_granular_bootstrap` only asserted Section titles, leaving `phase_summarise_sections` writes, `phase_derive_file_level` File.summary derivation, and `phase_relate_sections` Delta-B label kwargs all unexercised at runtime. Added read-back assertions for Section.summary + File.summary, plus a new `test_section_granular_inferred_edges` that seeds Section→Section REFERENCES via a side_effect'd inferrer and asserts `count(r)==2` — runtime-validating the `src_label="Section", dst_label="Section"` kwargs on both backends.

41. **M9 subagent-driven push discipline note** — the M9.1 and M9.2 implementer subagents pushed to `origin/main` as part of their tasks. The controller prompt did not explicitly prohibit pushing, but previous milestones had implicit controller-only push convention. Accepted post-hoc (local CI gates all green, no damage), but future implementer prompts should explicitly state "do not push — leave pushing to the controller" to maintain single-point push discipline.

42. **2026-04-20 spec-delta cleanup batch** (`c9bc877..803328c`, 20 commits, 60 new unit tests). Defensive `__init__` guards (TokenChunker, FileHasher, DebouncedQueue, CorpusWatcher, CheckpointStore, Ontology immutability). Error-path hardening (Summariser KeyError/TypeError, PromptRenderer path-traversal guard, `_FORBIDDEN` regex gains FOREACH + negative-lookbehind, CLI docker-check + ClickException wrapping). Architectural unification: `primary_key_for` in `_keys.py`; `contextd_home()` in `_paths.py` (removes 200ms click/rich overhead from MCP startup). MCP/CLI tightening: TOOL_DESCRIPTORS constant, `enumerate_corpus_files` skips hidden dirs/symlinks, `prompts/` moved into `contextd/prompts/` package, `_build_pipeline_deps` refactor, `--estimate-only` counts UTF-8 chars.

43. **M11 — Neo4j stand-up + Kuzu excision** (`57fdb39..4d69272`, 9 tasks). See Appendix-M11 milestone bullet for full details. `BackendName = Literal["memgraph", "neo4j"]`; `src_label`/`dst_label` kwargs advisory-only on Neo4j (no schema-first REL tables). Kuzu fully excised in M11.9.

44. **M10 — Acme adapter + critical wiring** (`94e3af3..ab62d80`, 18 commits). See M10 milestone bullet for full details. Critical plan deviation: `corpus.toml` declared paths must be template-parent-relative (not repo-root-anchored) after `8fa7f76` fix — `add-corpus --from` round-trips correctly only with the corrected template. `CorpusConfig.load` wraps `TOMLDecodeError` → `CorpusConfigError` so one bad corpus can't brick MCP startup.

45. **M12.2 — `docker:dind` service block dropped from CI workflow (`f0077be`).** The plan §12.2 verbatim included `services: docker: image: docker:dind` on the `integration` job. On GitHub-hosted `ubuntu-24.04` runners this is a no-op: Docker is pre-installed on the host, testcontainers-python connects via `/var/run/docker.sock` by default, and `tests/conftest.py`'s `backend` fixture has no `DOCKER_HOST` override to redirect at a nested daemon. The DinD service block was a GitLab-CI-recipe copy-paste that adds a ghost container without helping. Dropped; M12.2 ships with plain `runs-on: ubuntu-24.04` on both the `integration` and `e2e` jobs.

46. **M12.1 — `backend` fixture factored up from `tests/integration/conftest.py` to `tests/conftest.py` (`2fb77e4`).** M12.1 initially landed `tests/e2e/conftest.py` with `from tests.integration.conftest import backend as backend` to share the fixture across both test roots. Code-quality review flagged this as a pytest antipattern (conftest files aren't guaranteed importable via `sys.path`; pytest's docs discourage importing from them). Resolution: moved the fixture to the nearest common ancestor (`tests/conftest.py`), deleted `tests/integration/conftest.py` and `tests/e2e/conftest.py` entirely. Pytest's normal upward conftest discovery serves the fixture to both subdirectories without any explicit import. Side benefit: if future work adds `tests/` subdirectories (e.g., `tests/perf/`), they inherit `backend` automatically.

47. **2026-04-22 post-M12 hardening wave (`730ec39..9ed77ae`, 4 commits).** Four out-of-plan fixes driven by real-world usage of the Acme corpus index:

    - **`730ec39` — `[indexer] inference_concurrency` knob.** Threads a `ThreadPoolExecutor(max_workers=N)` through the four LLM-bound phases (`phase_summarise`, `phase_relate`, `phase_summarise_sections`, `phase_relate_sections`) via a new `_parallel_map` helper in `contextd/indexer/phases.py`. Default `N=1` preserves serial call ordering; higher values parallelise I/O-bound Gemini calls. Section phases pre-populate `_parse_cached` serially before dispatching workers (dict writes would race; reads are safe). Concurrency is useful to saturate API latency but the quota ceiling (~15 RPM for Gemma free tier) is the real cap — `N=5` is the practical sweet spot. 8 new unit tests + 3 config tests.

    - **`468db33` — idempotent resume + `--refresh` scope.** Partial bootstraps now resume without re-doing completed work. `phase_summarise*` filters on `s.summary IS NULL`; `phase_relate*` filters on `s.inferred_at IS NULL` and writes the marker after a successful upsert loop (zero-edge sections still marked so they aren't retried every restart; error paths leave it unset for retry). New `--refresh <scope>` option on `contextd index` with four values: `inferred`, `summaries`, `llm`, `all` — each wiping a specific layer of the dependency stack. Helper `_wipe_for_refresh(corpus, store, scope)` in `pipeline.py`. 14 new unit tests.

    - **`ea2c446` — migration `_0003` (Section full-text index).** Baseline only created `File_summary_ft`; `search(kind='Section')` crashed with `ProcedureCallFailed: no such fulltext schema index: Section_summary_ft`. New migration on both backends (Neo4j: `CREATE FULLTEXT INDEX ... ON EACH [s.summary]`; Memgraph: `CREATE TEXT INDEX ... ON :Section`). Idempotent via `IF NOT EXISTS` / name-idempotence. 1 new integration test parametrized on both backends.

    - **`9ed77ae` — migration `_0004` (backfill `inferred_at` on existing graphs).** The marker code in `468db33` was forward-only; graphs bootstrapped before it landed had zero markers, so a plain resume would re-run relate on the entire corpus. Migration `_0004` marks any Section/File with ≥1 outgoing `origin='inferred'` edge as processed. Uses `coalesce(s.inferred_at, datetime())` for idempotence. Live acme Neo4j: 0 → 501 sections marked (155 remain as zero-edge / never-ran cases — re-attempted on next bootstrap). 1 new integration test.

    Combined: 25 new unit tests (350 total), ~4 new integration tests (~89 total). All four CI gates + abstraction-invariant grep clean. The concurrency + idempotent + migration stack together means a partial run → user-kill → restart cycle costs only the un-processed remainder + quota delta, not a full corpus re-do.

48. **MCP `search` payload fix + Acme adapter relocation (2026-04-22).** Two structural cleanups after real use surfaced issues:

    - **`96c409a` — `tools.search` drops `embedding` + flattens node onto row.** Real-session acme eval showed `search(kind='Section', limit=8)` returning a 192 KB blob that overflowed the MCP client's per-result token budget; diagnosis found each row was carrying the full 1024-dim `embedding` float vector (~12 KB/row) inline. `contextd/mcp/tools.py::search` now strips `embedding` and flattens the node dict onto each row, so the shape matches what `query_graph` with an explicit `RETURN node.id AS id, node.summary AS summary, …` already produced. Three new unit tests: no `embedding` key leaks, default label is `File`, Pattern-style rows without an embedding field pass through unchanged.

    - **Acme-PRD adapter moved out of `examples/` into the Acme repo.** The adapter's three `[mcp.tools]` Cypher files (`four_surface`, `find_dangling_registrations`, `audit_stale_shas`) were evaluated against the live graph and judged not worth the tunability cost vs. composing the same queries out of generic tools + `query_graph`. Rather than tighten the tools in place, the decision was: drop the per-corpus MCP tools entirely, and relocate the remaining adapter config (`corpus.toml`, `ontology.json`, `prompts/summary.md`, `README.md`) from `examples/acme-prd/` to `/home/you/src/acme/.contextd/` — next to the corpus it describes. Setup flow is now `contextd add-corpus . --name acme-prd --from .contextd/corpus.toml` from the acme repo root.

      Contextd-side impact: `examples/acme-prd/` deleted, `docs/acme-adapter.md` deleted, `tests/integration/test_acme_adapter.py` deleted (the M10.12 canary; feature-level coverage survives in dedicated unit tests for each of ontology aliases, overrides, prompt override, corpus MCP tools, and non-md routing), `tests/integration/test_acme_tools.py` deleted (tools are gone; the one invariant worth keeping — Risk PK merge-on-identical-description — moved to `tests/integration/test_upsert_roundtrip.py`). `README.md`, `CHANGELOG.md`, `docs/cli.md`, `docs/design.md` updated to describe the generic adapter pattern (`.contextd/` directory colocated with the corpus) instead of the now-deleted example. `docs/design.md §13` rewritten from "Acme-PRD adapter" to "Writing a domain adapter" with generic guidance on when to register per-corpus MCP tools (rule of thumb: don't, unless a query runs often enough that tunability-cost pays back).

      Acme-side impact: new `/home/you/src/acme/.contextd/` directory with the four adapter files. Adapter README includes a "Query patterns" section showing the three previously-baked queries (four-surface coverage, dangling registrations filtered by `*Def$`, stale SHA audit) expressed as ad-hoc `query_graph` Cypher the caller composes in-session — tunable without a server restart.

    Tests: 353 unit (+3 from search fix, +0 net from relocation since one moved and two deleted), integration count dropped by ~5 (deleted `test_acme_adapter.py` had 1 backend-parametrized test = 2 variants; `test_acme_tools.py` had 5 parametrized tests = 10 variants, minus 1 preserved = 9 removed; plus 2 new for the Risk preserved test). All four CI gates + abstraction-invariant grep clean.
