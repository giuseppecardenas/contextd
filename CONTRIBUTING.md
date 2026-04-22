# Contributing to Contextd

## Development setup

```bash
git clone https://github.com/giuseppecardenas/contextd.git
cd contextd
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

Required environment variables (export in `~/.bashrc` or `.env`):

```
GEMINI_API_KEY   # https://aistudio.google.com/app/apikey
VOYAGE_API_KEY   # https://www.voyageai.com/
```

## Running tests

Before committing, the full CI gate chain must be green:

```bash
ruff check contextd tests
ruff format --check contextd tests
mypy --strict contextd
pytest tests/unit -q
```

All four must exit 0. No exceptions.

Additional test scopes (require Docker):

- `pytest tests/integration` — runs against live Memgraph + Neo4j containers
  (testcontainers spins them up automatically)
- `pytest tests/e2e` — full pipeline against both backends

## Architectural invariants

These constraints are enforced in CI and are load-bearing for correctness:

- **Backend-specific modules must not be imported outside `contextd/storage/`.**
  The abstraction-invariant grep in `.github/workflows/ci.yml` enforces this.
  Consumers depend on the `GraphStore` ABC only; the factory in
  `contextd/storage/factory.py` is the sole place backend modules are named.
- **Every edge carries `origin ∈ {inferred, structural, manual}`.**
  Wipe-and-replace on re-index operates only on `origin="inferred"`;
  structural and manual edges are preserved.
- **AI-inferred edges are validated against the ontology at write time.**
  `Ontology.validate_edge()` rejects types not declared in
  `contextd/ontology/base.json`. This is the primary defence against
  hallucinated relationship types.

## Adding a new backend

1. Implement the `GraphStore` ABC in `contextd/storage/<name>.py`.
2. Add migrations under `contextd/migrations/<name>/`.
3. Register in `contextd/storage/factory.py` (deferred import pattern).
4. Add the backend name to the `backend` fixture parametrisation in
   `tests/conftest.py`.

## Pull requests

- **One task, one commit.** Match the commit subject style:
  `type(scope): summary (spec §X.Y)` — e.g.
  `feat(storage): GraphStore ABC with typed origin property (spec §2.5.1)`.
  The `(spec §X.Y)` suffix cross-references the implementation plan when applicable.
- **All four CI gates must pass** before a PR is mergeable (ruff check,
  ruff format --check, mypy --strict, pytest tests/unit).
- **No amending pushed commits.** If a fix is needed post-push, land a new
  follow-up commit rather than rewriting history.
- **Never skip hooks** (`--no-verify` etc. are off-limits).
- Open an issue first for substantial changes so the design can be agreed
  before implementation starts.

## Filing issues

Please include:

- Contextd version (`contextd --version`) and Python version
- Backend in use (Neo4j or Memgraph) and container image tag
- Minimal reproduction steps
- Full error output (stack trace or log lines)
