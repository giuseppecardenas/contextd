# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Contextd is a locally-hosted GraphRAG knowledge layer. It indexes markdown, code, and structured-data corpora into a hybrid graph + vector store (Memgraph or KùzuDB, pluggable via config), generates per-file (or per-section) summaries and AI-inferred typed relationships via external LLM APIs (Gemini for inference, Voyage AI for embeddings), and exposes the result to AI assistants through a Model Context Protocol server. The tool is designed for single-user local use — everything runs on one machine; no multi-user concerns.

## Current Phase: Active Implementation

The project is mid-build against a detailed milestone plan. The plan drives build order deterministically — do not skip or reorder milestones.

**As of 2026-04-20 (HEAD `a52b8f7`; d9ec286..a52b8f7 local-only, not yet pushed):**

- **M0** (repo scaffold) — complete (`e752200`). CI green.
- **M1** (config + ontology foundations) — complete 5/5. Closing commit `6551a71`.
- **M2** (external AI providers) — complete 5/5. `InferenceProvider`/`EmbeddingProvider` ABCs, `GeminiProvider` with retry + BLOCK_NONE safety + usage accounting, `VoyageProvider` with batched embedding + retry, factory with env-var-driven keys, append-only `CostLog`. Closing commit `f1cecb3`.
- **M3** (storage backends) — complete 4/4 with post-closure bug fixes. `GraphStore` factory + forward-only `MigrationRunner`, `MemgraphBackend` (Bolt via gqlalchemy) + baseline migration, `KuzuBackend` (embedded v0.11) + baseline migration, parametrized cross-backend integration suite. Closing commit `fd6d477`; subsequent hardening in `eb28a41`, `cabe6f7`, `7d1c285`, `088069b`, `cab529f`, `8988a12`, `c4c8cac` (see spec-delta log below).
- **M4** (indexing primitives) — complete 7/7. `FileHasher` (MD5 + JSON state), `TokenChunker` (word-count sliding window), `HeadingParser` (markdown-it-py + GitHub anchor dedup), `EntityResolver` (vector-similarity via `GraphStore.vector_search`), `PromptRenderer` (`{{var}}` mustache over `.md` templates) + three default templates (`summarise`/`relate`/`translate`), `Summariser` + `RelationshipInferrer` (provider+renderer+parser; ontology-validated silent discard on inferred edges). Shared `contextd/inference/_json_body.py` extractor. M3 storage received a cross-cutting fix (`vector_search` unified to return `score` on both backends — spec-delta #14). Closing commit `a3f3bd6`.
- **M5** (indexing pipeline) — complete 5/5. `DebouncedQueue` (path-set aggregator with monotonic idle window), `CorpusWatcher` (watchdog wrapper over inotify/FSEvents/ReadDirectoryChangesW), `is_git_busy` (`.git/index.lock` / `HEAD.lock` polling detector), `phase_enumerate`+`phase_embed`+`phase_summarise`+`phase_relate`+`phase_close` + `run_bootstrap` runner + `enumerate_corpus_files` glob expander (5-phase bootstrap per spec §5.9 step 5; phase_enumerate absorbs embedding-at-CREATE because Kuzu `File.embedding` is IMMUTABLE_AFTER_CREATE — see spec-delta #20), `CheckpointStore` (per-corpus JSON resume state). Integration test parametrized on both backends asserts phase ordering, file counts, and one REFERENCES-edge case that exercises the spec-delta-(c) label kwargs. Closing commit `80abfde` (Task 5.4 follow-up refactor at `1591925` dropped phase_embed dead params, removed discarded `count_files` query, expanded test coverage, and added `reason STRING` to Kuzu REFERENCES/BELONGS_TO REL tables — extension of spec-delta #11).
- **M6** (CLI) — complete 5/5. Click + rich CLI shipped as `contextd/cli.py` with commands `init` / `up` / `down` / `status` / `add-corpus` / `list-corpora` / `index` / `ask` / `logs` / `costs`. Shared `_load_cfg()` helper centralises the user-config-or-default fallback. Wheel now bundles `contextd/docker_compose.yml` (memgraph:latest image + port 7687 only — spec-delta #25) and force-includes repo-root `prompts/` (spec-delta #26). `tomli-w>=1.0` added to main deps. `index --bootstrap` wires the full M5 pipeline end-to-end (real embedder + summariser + inferrer + hasher + store); `index --estimate-only` is filesystem-only for cheap cost previews. `ask` command is registered via deferred import of `contextd.inference.translate.QueryTranslator` (module built in M8) — `contextd ask --help` works now, `contextd ask "<question>"` raises ImportError until M8 lands. Closing commits `56b1541` + `9ffa069` (ruff RUF100 cleanup follow-up).
- **M7** (MCP server) — complete 3/3. `ReadOnlyGuardError` + `assert_read_only(cypher)` — keyword-regex gate rejects CREATE/MERGE/DELETE/SET/REMOVE/DROP/DETACH (plan-verbatim). 8 tool functions in `contextd/mcp/tools.py` thinly wrap `GraphStore` — `describe_project` (with spec-delta #30 AND-joined WHERE fix), `search` (full-text only; vector fallback deferred to M9), `related` (variable-length paths with caller-controlled depth, now descriptor-clamped to 1-5 per spec-delta #32), `inbound` / `outbound` (type-aware citation listings), `get_file_summary`, `query_graph` (read-only guarded raw Cypher), `section_tree`. Stdio MCP server at `contextd/mcp_server.py` registers all 8 tools over the `mcp>=0.9` SDK; `pyproject.toml` already registers `contextd-mcp` console script. Integration tests exercise describe_project (ordering by inbound citations) and query_graph (write rejection) on both backends. Closing commit `a52b8f7`.

**Cursor:** M8 Task 8.1 (Natural-language query translation — `QueryTranslator`).

**Test suite:** 180 unit + 63 integration = 243 collected, 237 executed (6 backend-specific skips for Kuzu-vs-Memgraph semantics). `ruff check`, `ruff format --check`, `mypy --strict`, and the abstraction-invariant grep all clean. Integration suite runs Memgraph via Docker (memgraph:latest v3.x) + Kuzu embedded in `tmp_path`.

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

### Non-blocking — revisit during M5/M6 pipeline work

- **Task 1.3 code-quality review flagged two items** to address before backends cache `Ontology`:
  - `Ontology`'s mutable fields (`dict`/`set`) are exposed directly; a consumer could mutate `onto.edge_types.add(...)` and silently corrupt validation. Fix: `frozenset` for sets, `types.MappingProxyType` for `node_types`.
  - `validate_node(node_type, properties)` ignores `properties` entirely. Either drop the param (YAGNI) or add real property-key validation.
- **Task 1.4 `@field_validator("backend")` is dead code** — pydantic v2's `Literal["memgraph", "kuzu"]` check fires first. Harmless but removable.
- **Test for `Ontology.with_aliases` error path** (raises on unknown target) is not present. Add before integration tests rely on it.
- **M4.1 `FileHasher._load_state` cast is unvalidated** — `cast(dict[str, str], json.loads(...))` tells mypy the shape is right but does not verify it. A hand-edited or corrupted state file flows through and `is_changed` silently returns a wrong answer. Add an `isinstance` guard before M5 state-file corruption becomes a real concern.
- **M4.2 `TokenChunker.__init__` lacks `overlap < max` guard** — if `overlap_tokens >= max_tokens`, `step = max - overlap <= 0` and `start` never advances → infinite loop. Add `raise ValueError("overlap_tokens must be < max_tokens")` in `__init__` before M5 wires it from `Config`.
- **M4.4 `EntityResolver` fallback key order `("path", "id", "name")`** — duplicates label-to-PK knowledge that lives in `contextd/storage/_keys.py::PRIMARY_KEY_BY_LABEL`. Any new label with a different PK (e.g., `Risk.description`, `Meta.schema_version`) will silently return `None` from `resolve()`. M5 pipeline wiring is the right time to pass the canonical PK through.
- **M4.5 `PromptRenderer.render(**kwargs: str)` forces string coercion at call sites** — `Summariser` already does `max_words=str(self._max_words)`. Acceptable friction, but a `**kwargs: object` with `str()` inside `_sub` would remove it. Not blocking.
- **M4.5 `PromptRenderer` has no path-traversal guard** — `template="../../../etc/passwd"` would escape `template_dir`. Single-user threat model so risk is low, but a one-line `is_relative_to` check before `read_text()` closes the footgun.
- **M4.5 `prompts/translate.md` mutation-keyword allow-list incomplete** — lists CREATE, MERGE, DELETE, SET, REMOVE. Missing: DETACH DELETE, DROP, FOREACH, CALL-with-side-effects. M8 hardens this, but the prompt is the first-line defence.
- **M4.5 `PromptRenderer` override chain deferred** — the plan mentions `~/.contextd/prompts/` override with packaged fallback; current impl takes a single `template_dir`. Module docstring was pruned to match reality. Feature landing is M6 (config-driven prompt directory) or later.
- **M4.6 `Summariser` required-field `KeyError` message is bare** — `data["summary"]` raises with no context about where the data came from. Wrap as `f"Provider response missing 'summary'; got keys {list(data.keys())}"` for better debugging when real Gemini output hits.
- **M4.6 `cast(str, data["summary"])` unvalidated** — same pattern as M4.1; trusts the JSON shape. An `isinstance` check before the cast would catch a `42`-as-summary mistake upstream.
- **M5.1 `DebouncedQueue.__init__` lacks `window_seconds > 0` validation** — a zero or negative value fires `drain_if_ready()` on every poll. Consistent with the M4.2 `overlap < max` gap; fix together. Also: not thread-safe (callback from watchdog thread would race `add` / `drain_if_ready`) and not Path-normalised (`Path("./a")` and `Path("a")` hash differently and don't dedup). Address when M5 wiring binds `CorpusWatcher` → `DebouncedQueue`.
- **M5.2 `CorpusWatcher` hygiene gaps before production wiring** — (a) `start()` called twice silently leaks the first observer (second call overwrites `self._observer`; first thread keeps running, still holds watches, still fires callbacks). Raise `RuntimeError("already started")`. (b) `stop()`'s 5-second join-timeout swallows failures; log a warning instead. (c) Callback fires on the observer's dispatch thread — document as a class-docstring precondition so downstream callers know to use a queue/lock.
- **M5.3 `is_git_busy` test-coverage gaps + worktree blind spot** — plan tests cover (no-git-dir → False) and (index.lock → True) only. Add `HEAD.lock`-only → True and `.git present but no lock` → False for mutation-style coverage of the `or` branch. Also: when `corpus_root/.git` is a regular file (worktree / submodule / `--separate-git-dir`), `is_dir()` returns False and the detector is blind to in-progress rebases. Resolve the gitfile: read `gitdir: /path/...` out of the `.git` file and re-check that path. Low priority for single-user local tool.
- **M5.4 `phase_embed` is an accounting-only stub** — spec-delta #20 moved the work into `phase_enumerate` so Kuzu's `IMMUTABLE_AFTER_CREATE` on `File.embedding` is respected. The follow-up (`1591925`) shrank the signature to `phase_embed(files)` and removed the three dead-parameter lies. Right now the 5-phase contract (test assertion `result.phases[1].processed == 2`) keeps the stub alive. If design §5.9's five conceptual steps are ever re-scoped to match reality, collapse to four phases and renumber.
- **M5.4 `_infer_key` in `phases.py` duplicates `PRIMARY_KEY_BY_LABEL`** — 4 labels vs. 14 in `contextd/storage/_keys.py`. Unknown target types silently fall through to `"name"`. Route through `primary_key_for()` during M8/M9 so hallucinated target types raise at write time rather than creating malformed weak-entry nodes.
- **M5.4 `enumerate_corpus_files` has no symlink / `.git` protection** — default `root.glob("**/*")` follows symlinks and includes dotfiles. A corpus that doesn't set `exclude` can walk its own `.git/objects/**`. Design's `exclude` list is the prescribed defence; no hard-coded `.git` safeguard. Consider a default-exclude-dotfiles flag when M6 wires the CLI.
- **M5.4 `phase_enumerate` holds the full corpus embedding matrix in memory** — batching computes embeddings per batch but appends to `all_embeddings: list[list[float]]` before looping again for upserts. At 1024-dim × 8 B × N files, 10k-file corpus ≈ 80 MB of vectors held simultaneously. Streaming variant (embed → upsert → drop batch vectors) is the fix when corpus sizes grow past the target.
- **M5.4 Corpus node lacks `node_count` / `edge_count` columns** (spec-delta #22) — `phase_close` currently only writes `name` + `registered_at`. Plan's original intent was to persist corpus stats on the Corpus singleton for MCP-side queries. Add columns via a new Kuzu migration when the MCP layer needs them; Memgraph is schema-free and would accept them today.
- **M5.5 `CheckpointStore.save` is not atomic** — `write_text()` truncates-then-writes, so an interrupt between truncation and flush leaves a partial/empty JSON. This defeats the module's stated resume purpose. Switch to `tmp.write_text(...) + os.replace(tmp, target)` before the bootstrap resume path consumes checkpoints.
- **M5.5 `CheckpointStore.load` has no JSON-shape validation** — `Checkpoint(**raw)` silently admits wrong-typed field values (e.g. `last_committed_batch: "three"`) that explode downstream. Same class of issue as M4.1 spec-delta #15; add narrow `isinstance` guards on each field, or mirror the FileHasher cast pattern.
- **M5.5 corpus name passed verbatim to filesystem** — `save("../../etc/passwd", cp)` escapes `self._root`. Single-user threat model lowers the risk, but a one-line name-validation guard (reject `/`, `\`, `.`, `..`, leading-dot) closes the footgun. Class docstring should also state the single-writer precondition explicitly.
- **M6.1 `--yes` flag on `init` is vestigial** — declared but unread; interactive prompts deferred to M13. Remove or document as reserved when M6 cleanup lands.
- **M6.1 `CONTEXTD_HOME` captured at module-import time** — every test in `tests/unit/test_cli_*.py` needs `importlib.reload(contextd.cli)` after `monkeypatch.setenv`. A late-bound `_contextd_home() -> Path` accessor removes the reload dance but ripples through every `from contextd.cli import CONTEXTD_HOME` call site. Defer to a single cleanup pass after M7 when the module layout stabilises.
- **M6.1 `read_text()` calls lack `encoding="utf-8"`** — defaults to platform encoding. Fine on Ubuntu target; tighten when `cli.py` is next touched.
- **M6.1 Hatch `force-include` for `prompts/` couples wheel build to Hatch** — cleaner alternative: move `prompts/` to `contextd/prompts/` when M4.5's config-driven prompt-dir override lands.
- **M6.2 `up` produces a Python traceback if Docker is missing** — `subprocess.run(["docker", ...], check=True)` with Docker absent raises `FileNotFoundError` which Click surfaces as a traceback. `init` already warns about missing Docker; `up` should gate via `shutil.which("docker")` → `raise click.ClickException(...)` in an M7 CLI-hardening pass.
- **M6.2 `status` docstring promises "daemon + backend + corpora" but only reports backend + corpora** — no daemon-running check. Add a PID-file or `docker ps` probe when the M7 MCP server lands and has a lifecycle of its own.
- **M6.3 add-corpus duplicate-guard test does not assert the TOML body was not overwritten** — test passes if the early-return is accidentally moved after `write_bytes`. One-line `stat().st_mtime_ns` assertion closes the gap. Fold into M12 test consolidation.
- **M6.3 list-corpora empty-vs-absent branches indistinguishably covered** — `test_list_corpora_empty` matches both the "run `contextd init` first" and "no corpora registered yet" messages. Add a second test that creates an empty `corpora/` dir and asserts the exact second-branch string.
- **M6.4 `index` command uses `sys.exit(1)` on error paths** — prefer Click-idiomatic `raise click.UsageError(...)` / `ctx.exit(1)` for richer error presentation. Non-blocking; fold into M7 CLI-hardening.
- **M6.4 bootstrap branch of `index` has no CLI-level test coverage** — `run_bootstrap` integration is covered at M5. A sixth test patching `run_bootstrap` to return a canned `BootstrapResult` would assert the per-phase green-check printing. Cheap when revisited.
- **M6.4 `index` command body is 70 lines / 5 responsibilities** — validate, resolve corpus config, enumerate, estimate-or-bootstrap, wire deps, run. Extract `_build_pipeline_deps(cfg, corpus_cfg, corpus_name) -> tuple` after M6.5. Makes the incremental branch (M7) easier to fill.
- **M6.4 `estimate-only` token projection uses `st_size // 4`** — fine for ASCII markdown, over-estimates on UTF-8 multi-byte content. Switch to `len(p.read_text("utf-8"))` when the estimate drives budget-gating.
- **M6.5 `ask` command has no error handling** — translator failure, unreachable backend, malformed Cypher all surface as stack traces. Wrap `translator.translate` + `store.exec_read` in a `click.ClickException` handler when M8's `QueryTranslator` lands.
- **M6.5 `logs --follow` does not trap `KeyboardInterrupt`** — `subprocess.run(["tail", "-f", ...])` blocks; Ctrl-C produces a Click `Aborted!`. Benign; fix when CLI error paths are hardened.
- **M6 `cli.py` at 341 lines — splitting planned for M7** — suggested layout when `ask` + MCP-proxy commands land: `cli/__init__.py` (group + `init` + `main`), `cli/infra.py` (up/down/status), `cli/corpora.py` (add-corpus, list-corpora, index), `cli/query.py` (ask, logs, costs). M7 did NOT touch cli.py (MCP lives in its own module) so the deferred split carries forward.
- **M7.1 `_FORBIDDEN` regex false-positive on dotted property access** — `RETURN n.set AS prop` would match `SET` because `\b` fires between `.` and `s`. MCP-caller threat model is low (LLM-generated Cypher rarely uses those property names), but a negative-lookbehind tighten `(?<![.\w])(CREATE|...)\b` is a one-line fix when M8's NL→Cypher translator produces real query samples.
- **M7.1 forbidden keyword list missing `FOREACH`** — `FOREACH (x IN list | SET n.x = x)` is a write pattern. The inner `SET` is caught by the current regex, so practical impact is narrow (only pure-FOREACH patterns without inner writes would pass), but the plan's own M4.5 deferred note flagged FOREACH explicitly. Add when next touching the guard.
- **M7.2 `tools.search` docstring over-promises vector fallback** — docstring says "full-text first, fall back to vector" but implementation runs full-text only. Plan-verbatim. Prune docstring or implement fallback when M9 lands vector-search routing.
- **M7.2 `tools.describe_project` returns mixed-label rows** — `MATCH (n)` has no label filter, so Section/Ticket/etc. nodes with summaries surface with `path=None`, `name=None`. Consumable by LLMs but shape-inconsistent. When M7.3's tool descriptors are hardened, narrow the MATCH or document the null-key pattern.
- **M7.2 `tools.related` uses f-string depth interpolation** — `MATCH (a)-[r*1..{depth}]-(b)` substitutes Python int directly. M7.3's tool descriptor clamps to `[1, 5]` via JSON schema (spec-delta #32), but the function itself has no defensive clamp. Kuzu hard-caps at depth 30; Memgraph has no hard cap. When parameterized variable-length paths are validated to work across both SDKs (Memgraph `$depth` / Kuzu `CAST($depth AS INT64)`), switch to parameters for defence-in-depth.
- **M7.3 `store.close()` never called in `run()`** — the `GraphStore` connection is opened but never closed. Process exit reaps everything for single-user stdio usage, but a `try: ... finally: store.close()` wrap around the `async with stdio_server()` block is the right shape. Fold into M12 e2e hardening.
- **M7.3 tool call return shape is `str(obj)` not JSON** — `[{"type": "text", "text": str(ov.nodes)}]` emits Python `repr` (single-quoted, `None` not `null`, `True`/`False` not lowercase). LLM clients parse as strings rather than structured data. Switch to `json.dumps(obj, default=str)` per branch before M12 e2e wires a real Claude/Cursor client.
- **M7.3 no error handling around `tools.X(...)` calls** — malformed node_id, missing arg, or `assert_read_only`'s `ValueError` all bubble up as Python exceptions. The mcp SDK presumably converts to MCP error responses, but the behaviour is implicit. Wrap in `try: ... except Exception as e: return [{"type": "text", "text": f"error: {e}"}]` (or mcp's canonical error shape) when M12 e2e exercises failure paths.
- **M7.3 `_list` tool registry is closure-bound** — tests can't verify "there are 8 tools with schemas X/Y/Z" without running the server. Promoting to module-level `_TOOL_DESCRIPTORS: list[Tool]` would let a unit test assert the registry. Controller explicitly deferred this refactor for 7.3; revisit during M12 e2e or next MCP-adjacent work.
- **M7.3 `CONTEXTD_HOME` imported from `contextd.cli`** — pulls click/rich into the MCP process (~200ms import overhead). Same class as the M6.1 deferred item; fold into the eventual `contextd/_paths.py` or `contextd/bootstrap.py` extraction.

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

8. **Memgraph FTS migration + backend rewritten (`088069b`, updated `1487023`)** — plan's `CREATE INDEX ON :File(summary)` creates a B-tree label-property index, which the `text_search` procedure cannot find by name. Changed to `CREATE TEXT INDEX File_summary_ft ON :File` (Lucene-backed). The backend's `full_text_search` switched from `text_search.search(idx, q)` (requires Lucene-expression syntax) to `text_search.search_all(idx, q)` (plain keyword over all indexed properties). `search_all` yields both `node` and `score`; `full_text_search` returns `{"node", "score"}` on both backends after the audit-pass fix in `1487023`.

9. **Memgraph `vector_search` threshold clause (`088069b`)** — plan's `YIELD ... WHERE ...` form parses the WHERE token outside a valid Cypher position. The backend now re-projects via `WITH node, score WHERE score >= <threshold>` before RETURN. The threshold is currently f-strung into the Cypher; parameterising it is in the backlog (float injection surface + `nan`/`inf` bad-format risk).

10. **Kuzu `vector_search` arity (`088069b`)** — plan's `CALL QUERY_VECTOR_INDEX(idx, $q, $k)` has the wrong arity. Kuzu's signature is `QUERY_VECTOR_INDEX(table, idx, query, k)` — 4 args. Also `k` is inlined as a literal because Python `int` binds as `INT8` and the procedure expects `INT64` (same pattern as `_record_applied`).

11. **Kuzu `upsert_edge` per-property SET + REL-table schema (`7d1c285`)** — plan's `SET r.origin = $origin` silently dropped every other edge property. The backend now enumerates one `SET r.<k> = $<k>` per supplied property. Kuzu REL tables are schema-first, so any property beyond the declared columns surfaces as a binder exception — `REFERENCES` and `BELONGS_TO` gained `confidence DOUBLE` (nullable) to carry the inferred-edge review signal; adding further edge properties requires a migration.

12. **CI `abstraction-invariant` grep narrowed to `contextd/` (`eb28a41`)** — originally `grep ... contextd/ tests/ --exclude-dir=storage`, which caught the plan-prescribed `from contextd.storage.memgraph import ...` in integration test files and failed CI for three consecutive commits on main before being fixed. The invariant's intent is to protect runtime consumers (indexer, MCP, CLI) from coupling to a concrete backend; integration tests are legitimately backend-specific and stay outside the fence.

13. **pytest filterwarnings suppression (`088069b`)** — `gqlalchemy.exceptions.GQLAlchemySubclassNotFoundWarning` is raised whenever gqlalchemy materialises a node whose label has no Python ORM class. `GraphStore` deals in dicts, not ORM models; this warning is structural and harmless. Added to `[tool.pytest.ini_options].filterwarnings` alongside the `"error"` strictness default.

14. **`vector_search` score-shape unification (`323c179`)** — during M4 Task 4.4 (`EntityResolver`) code review, we found the backends diverged on `vector_search` return shape: Memgraph returned `"score"`, Kuzu returned `"distance"`. The ABC docstring documented the divergence. `EntityResolver.resolve` reads `top.get("score", 0.0)` which defaulted to 0.0 on Kuzu and always skipped dedup. Normalised both backends to return `{"node", "score"}` where `score` is cosine similarity in `[0, 1]`. The Kuzu backend's internal server-side filter still operates on distance; only the public return shape changed.

15. **M4.1 `FileHasher` mypy `cast` (`2df9133`)** — plan's `_load_state` returns `json.loads(...)` bare; mypy-strict rejects with `no-any-return`. Wrapped with `cast(dict[str, str], ...)` — narrowest fix, not a semantic change.

16. **M4.3 `HeadingParser` defect fixes (`4535d4b`, `7ccec17`)** — plan's verbatim code had four Critical/Important defects caught by code-quality review: (a) inline link headings stored raw Markdown as title/anchor because `inline.content` returns source, (b) empty anchors on punctuation-only headings, (c) duplicate-anchor cross-contamination in `sibling_ordinals`, (d) `__init__` accepted invalid `min_level`/`max_level` bounds. Second pass found (e) dedup suffix collisions with manually-authored anchors, (f) image-only headings emitting raw Markdown. Fixed with: a walker over `inline.children` filtered to `{text, code_inline, image}`, empty-anchor fallback to `"section"`, GitHub-style dedup counters that skip pre-claimed slots, and range-validated `__init__`. Nine new tests across both commits.

17. **M4.5 `PromptRenderer.render(template=...)` parameter rename (`9661317`)** — plan signature was `render(self, name: str, **kwargs: str)` but the plan's own test calls `renderer.render("summarise", name="world")` which would `TypeError: got multiple values for argument 'name'`. Renamed positional param to `template`. Forward-compatible: M4.6/4.7 callers already use the first arg positionally.

18. **M4.5 docstring honesty + M4.6/M4.7 KeyError context (`bcbe395`, code-quality follow-up)** — pruned `PromptRenderer` module docstring claim of `~/.contextd/prompts` fallback (not implemented; deferred); KeyError now includes template name so caller sees which template lacked which variable.

19. **M4.6 / M4.7 fence-stripper + shape-tolerance refactor (`425773d`, `15fb5ff`, `a3f3bd6`)** — plan's fence regex `^```(?:json)?\s*|\s*```\s*$` was brittle against non-`json` language tags (`yaml`, `text`) and prose wrappers (`Here is the JSON: ...`). Replaced with a `first-{` to `last-}` slice in the shared `contextd/inference/_json_body.py`. Added `isinstance`-guard tolerance for non-list `relationships`, non-dict rows, and rows missing `target_name` — all now silently discarded to match the rest of the module's tolerant-parsing pattern instead of raising `KeyError` mid-batch. Eleven new tests.

20. **M5.2 watcher type-correctness fixes (`ec788f7`)** — plan's code had three narrow mypy-strict / ruff failures: (a) `# noqa: N804` suppresses the wrong rule (N804 is for classmethod `cls`, N805 is for instance-method `self`). (b) `self._observer: Observer | None` — `watchdog.observers.Observer` is a platform-dispatched callable alias (`ObserverType`), not a class; the stored value is a `BaseObserver` subclass instance. Changed to `BaseObserver | None` via `from watchdog.observers.api import BaseObserver`. (c) `Path(event.src_path)` — `src_path` is typed `str | bytes` and mypy-strict rejects `bytes` at `Path()`. Wrapped with `os.fsdecode(...)`. All three are runtime no-ops on WSL2 inotify.

21. **M5.4 phase_embed restructured to enumerate-time embedding (`13bc14f`, refined `1591925`)** — plan's `phase_embed` used `store.exec_write("MATCH (n:File {path: $path}) SET n.embedding = $vec", ...)`. This fails on Kuzu because `File.embedding` is declared in `contextd/storage/_keys.py::IMMUTABLE_AFTER_CREATE_BY_LABEL` (vector-indexed column; Kuzu rejects `SET` after creation per spec-delta #7). Option 2 (use `upsert_node` to update) was analysed and rejected because Kuzu's two-phase upsert skips immutable properties on the update branch — the embedding would silently not be written. Resolution: `phase_enumerate` was extended to accept `embedder: EmbeddingProvider`, batch-computes embeddings upfront, and passes them to `upsert_node` at CREATE time. `phase_embed` became an accounting-only stub that reports the file count (no storage calls) so the 5-phase contract and integration-test `result.phases[1].processed == 2` assertion remain intact. Follow-up `1591925` shrank `phase_embed` signature to `phase_embed(files: list[Path])` — the `embedder`/`store`/`batch_size` params were unused post-restructure and lied about the function's behaviour. **Process note:** the implementer silently restructured despite a controller-issued negative instruction to escalate BLOCKED; the technical outcome is correct but the process violated the Subagent Execution Contract and is a project-level lesson for future subagent briefings.

22. **M5.4 `phase_close` dropped corpus-stat persistence (`13bc14f`, refined `1591925`)** — plan's `phase_close` upserts `name` + `registered_at` + `node_count` + `edge_count` on the Corpus node. Kuzu's Corpus DDL declares only `(name STRING PRIMARY KEY, root STRING, registered_at TIMESTAMP, content_profile STRING)` — no `node_count` / `edge_count` columns — and adding them requires a migration. Writing them only to Memgraph (schema-free) would break the backend-neutral contract. `phase_close` now upserts `name` + `registered_at` only; the plan's `MATCH ()-[r]->() RETURN count(r) AS c` query was removed entirely in `1591925`. Also: `"__now__"` placeholder replaced with `datetime.now(dt.UTC)` — no backend code substitutes the placeholder.

23. **M5.4 `phase_relate` Kuzu label kwargs (`13bc14f`)** — plan's `store.delete_edges(str(f), origin="inferred")` and `store.upsert_edge(str(f), ..., origin="inferred", properties=...)` omit the label kwargs that Kuzu requires (schema-first REL tables declare fixed FROM/TO pairs; see spec-delta #4). Added `src_label="File"` to `delete_edges` and `src_label="File", dst_label=rel.target_type` to `upsert_edge`. Integration test `test_bootstrap_creates_inferred_edges` covers the branch on both backends; the happy-path `test_bootstrap_on_sample_corpus` uses `inferrer.return_value=[]` so doesn't exercise the label kwargs.

24. **Kuzu REFERENCES / BELONGS_TO gained `reason STRING` (`1591925`)** — extension of spec-delta #11. `phase_relate` passes `properties={"confidence": rel.confidence, "reason": rel.reason}` to `upsert_edge`; without the `reason` column on the REL table, Kuzu's schema-first binder rejects the call. Applied to the baseline migration directly (vs. a new migration 2) because the project is pre-release and no user DBs have been persisted. Future REL-property additions must go through a new migration file.

25. **M6.1 docker-compose template uses `memgraph:latest` (`096f537`)** — plan text specifies `memgraph/memgraph-platform:latest` which is pinned at v2.14 and predates vector-index support (inherited issue from spec-delta #2). Template changed to `memgraph/memgraph:latest` (v3.x) and port 3000 dropped (the Lab UI port the plain `memgraph` image doesn't serve).

26. **M6.1 Hatch `force-include` ships repo-root `prompts/` in the wheel (`096f537`)** — plan's `init` command copies `resources.files("prompts")` but the directory lives at repo root, NOT inside the `contextd/` package. In a wheel install `resources.files("prompts")` would `ModuleNotFoundError`; the plan's `try: ... except Exception: pass` wrapper masked the failure silently, which in turn would have caused `contextd index` to break with "no templates" on a fresh install. Resolution: added `[tool.hatch.build.targets.wheel.force-include] "prompts" = "prompts"` to `pyproject.toml`, dropped the bare `try/except`, dropped the dead `prompts_src` assignment. Tested via `tests/unit/test_cli_init.py::test_init_creates_layout` which now asserts `~/.contextd/prompts/summarise.md` exists after `init`.

27. **M6.2 `up` kuzu branch creates `db_path.parent`, not `db_path` (`075164e`)** — plan says `db_path.mkdir(parents=True, exist_ok=True)` but Kuzu stores the DB in a single FILE (spec-delta #3), not a directory. Creating a directory at `db_path` breaks `KuzuBackend.connect()` with "Database path cannot be a directory". Backend internally does `db_path.parent.mkdir(parents=True, exist_ok=True)`; CLI `up` now matches. Test `test_up_kuzu_creates_db_path` uses a nested path (`{home}/nested/graph`) so the parent dir is actually created by `up` (not pre-existing from fixture setup — follow-up `9ebc26e` made the test falsifiable).

28. **M6.3 `_load_cfg()` helper extracted (`a431f7e`)** — the 4-line `Config.load(path) if path.exists() else Config.load_default()` ternary was duplicated across `up`, `down`, `status` per the plan. M6.2 code-review flagged the duplication. `_load_cfg() -> Config` helper placed near the top of `cli.py` using `TYPE_CHECKING` to avoid a cycle; `from contextd.config import Config` stays lazy inside the helper body. `index` (M6.4) and `ask` (M6.5) adopted the helper on landing.

29. **M6.5 `ask` command imports `QueryTranslator` via deferred-and-typed-ignored import (`56b1541`, `9ffa069`)** — `contextd.inference.translate` does not exist yet (built in M8). The import lives inside the `ask` function body so CLI registration and `contextd ask --help` work today; `contextd ask "<question>"` raises `ImportError` at invocation until M8 lands. Annotated `# type: ignore[import-untyped]` to keep mypy-strict green. Initial attempt at an additional `# noqa: PGH003` tripped `RUF100` (project's ruff select list doesn't include `PGH`); follow-up `9ffa069` dropped the dead `noqa`. When M8 lands, remove the `type: ignore` comment and exercise the command with a real integration test.

30. **M7.2 `describe_project` double-WHERE fix (`a32f6e2`)** — plan's verbatim Cypher had two consecutive `WHERE` clauses after a single `MATCH`: `WHERE n.corpus = $corpus` (conditional f-string expansion) followed by `WHERE n.summary IS NOT NULL`. Both Memgraph and Kuzu parse this as a syntax error. Fixed by collecting predicates into a `filters: list[str]` and AND-joining them into a single `WHERE`. Params dict populated conditionally to match. Verified passing on both backends via `tests/integration/test_mcp_tools.py::test_describe_project_returns_summaries`.

31. **M7.2 integration test `upsert_edge` label kwargs (`a32f6e2`)** — plan's `tests/integration/test_mcp_tools.py::test_describe_project_returns_summaries` called `backend.upsert_edge("a.md", "b.md", "REFERENCES", origin="structural")` which fails on Kuzu per spec-delta #4 (schema-first REL tables require explicit src/dst labels). Added `src_label="File", dst_label="File"` to the call. Plan-verbatim elsewhere.

32. **M7.3 `related` tool descriptor depth clamped 1-5 (`a52b8f7`)** — plan's `related` Tool descriptor declared `depth: {"type": "integer", "default": 2}` with no min/max. Caller-controlled depth combined with the f-string interpolation in `tools.related` meant an MCP caller could pass `depth=1000` and hang Memgraph or crash Kuzu. Added `"minimum": 1, "maximum": 5"` to the JSON schema and updated the description to "(1-5)". The 5-hop ceiling aligns with design §5's "reasonable neighbourhood" framing. Function-level defensive clamp is a deferred Known-Limitation item above.

33. **M7.3 narrow mypy-strict ignores on mcp SDK decorators (`a52b8f7`)** — `@server.list_tools()` and `@server.call_tool()` decorators are untyped in mcp v1.27. Mypy-strict requires `# type: ignore[no-untyped-call,untyped-decorator]` on `list_tools` (double error: untyped decorator AND untyped call of decorator factory) and `# type: ignore[untyped-decorator]` on `call_tool`. Asymmetric but correct — matches what mypy actually reports. Remove both when mcp SDK ships type stubs.
