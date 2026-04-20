# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Contextd is a locally-hosted GraphRAG knowledge layer. It indexes markdown, code, and structured-data corpora into a hybrid graph + vector store (Memgraph or KùzuDB, pluggable via config), generates per-file (or per-section) summaries and AI-inferred typed relationships via external LLM APIs (Gemini for inference, Voyage AI for embeddings), and exposes the result to AI assistants through a Model Context Protocol server. The tool is designed for single-user local use — everything runs on one machine; no multi-user concerns.

## Current Phase: Active Implementation

The project is mid-build against a detailed milestone plan. As of 2026-04-20, M0 (repo scaffold) is complete (commit `e752200`, CI green on GitHub), and M1 (config + ontology foundations) is 4/5 done with local commits only. The plan drives build order deterministically — do not skip or reorder milestones.

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

- **Task 1.3 code-quality review flagged two non-blocking items** to address before M3/M5 backends cache `Ontology`:
  - `Ontology`'s mutable fields (`dict`/`set`) are exposed directly; a consumer could mutate `onto.edge_types.add(...)` and silently corrupt validation. Fix: `frozenset` for sets, `types.MappingProxyType` for `node_types`.
  - `validate_node(node_type, properties)` ignores `properties` entirely. Either drop the param (YAGNI) or add real property-key validation.
- **Task 1.4 `@field_validator("backend")` is dead code** — pydantic v2's `Literal["memgraph", "kuzu"]` check fires first. Harmless but removable.
- **Test for `Ontology.with_aliases` error path** (raises on unknown target) is not present. Add before M3 integration tests rely on it.

These are tracked in the implementation plan's "Known Limitations" section at the end of M1 closure — revisit when touching those modules during M2/M3.
