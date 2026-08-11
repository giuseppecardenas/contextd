# Runbook: runeledger corpus repair (wipe + re-bootstrap)

The 2026-08-11 graph audit (`docs/investigations/runeledger-edge-summary-quality.md`)
found the runeledger corpus's inferred layer degraded beyond in-place repair:
~4,600 entity stubs (84–91% degree≤1), zero entity content, hallucinated
`Corpus`/`Meta` nodes, and an empty Section→Section web. The indexing-quality
overhaul fixes the pipeline; this runbook replays the corpus through it.
Exclusive section bodies changed every stored section hash anyway, so a full
wipe + re-bootstrap is both the simplest and the required path.

Run the steps in order.

## 1. Upgrade + migrate

```bash
# from the contextd repo root, venv active, on the overhaul HEAD
contextd up          # starts backend if down; applies migration _0007
                     # (entity name_norm btree + vector indexes)
```

## 2. Refresh the home prompt templates — FIRST, before any indexing

`contextd init` copies templates once and never updates them; the stale
`~/.contextd/prompts/relate.md` (2026-04-30 vintage) was the single biggest
quality defect found. The new identity/candidate/glean/rollup/code templates
are inert until this step runs:

```bash
contextd init --refresh-prompts
contextd status      # every template must show "matches packaged"
```

If you had deliberate customisations in `~/.contextd/prompts`, port them onto
the new packaged text instead of skipping this step.

## 3. Update the runeledger adapter template

Edit `~/src/runeledger/.contextd/corpus.toml` (the `--from` template) before
re-adding. Recommended additions:

```toml
[summarization.overrides]
"**/*.lua" = "builtin:summarise_code"   # stop summarising Lua as PRD prose

[resolution]
# defaults are fine for runeledger; shown for visibility
# case_insensitive_labels = ["Pattern", "Technology", "Client", "Risk", "Service", "Integration"]
# fuzzy_threshold = 90.0

[[lexical.patterns]]
regex = "\\bFR-[A-Z]+-\\d+\\b"
edge_type = "REFERENCES"
target_type = "FRRow"        # ontology alias → Ticket

[[lexical.patterns]]
formats = ["lua"]
regex = "register_(\\w+)"
edge_type = "REFERENCES"
target_type = "Registry"     # ontology alias → Pattern
capture = 0
```

Also set in `~/.contextd/config.toml`:

```toml
[providers.openai_compat]
temperature = 0.2            # JSON extraction; translation keeps default

[inference]
relate_gleaning_rounds = 1   # already the default; 0 to halve relate spend
```

## 4. Remove the polluted corpus

```bash
contextd remove-corpus runeledger
```

This now also deletes the corpus-scoped entity population (entities carry
`corpus` since mint time).

## 5. Sweep legacy strays

Entities minted before corpus tagging have no `corpus` property and survive
step 4:

```bash
contextd prune-entities --dry-run          # inspect
contextd prune-entities                    # reap zero-degree strays
```

(`--max-degree 1` also exists for sweeping single-use stubs later, after the
new pipeline has run — don't use it on a healthy graph without a dry run.)

## 6. Re-register and bootstrap

```bash
contextd add-corpus ~/src/runeledger --name runeledger \
  --from ~/src/runeledger/.contextd/corpus.toml
contextd index runeledger --bootstrap
```

Expect the phase list: enumerate → gc → embed → summarise → rollup → relate →
derive_file_level (+ file-granular phases for the Lua files) →
merge_descriptions → close. Watch `contextd logs --follow` for
`relate drop:` / `resolve:` lines — every discard and merge decision is now
logged.

## 7. Re-measure against the 2026-08-11 baselines

Re-run the audit queries from
`docs/investigations/runeledger-edge-summary-quality.md` §5. Success criteria:

| Metric | Baseline (2026-08-11) | Target |
|---|---|---|
| Entity nodes | ~4,600 | order-of-magnitude fewer |
| Entities degree≤1 | 84–91% | well below 50% |
| Entity content (description) fill | 0% (except Risk) | majority filled |
| Section→Section inferred edges | ~0 | present; §-refs resolve |
| Corpus/Meta nodes minted by inference | 35 / 129 | 0 / 0 |
| Confidence histogram | 83% at 0.9 | floor-enforced, lexical at 1.0 |
| Sections with ≥1 inferred edge | 78% | higher |
| md File nodes with embeddings | 0/40 | 40/40 |
| Lua summaries | "This section..." misframed | code-prompt framed |
| Same text in two Section nodes | every parent/child pair | none |

Drop-log counts by reason (`grep "relate drop:" ~/.contextd/logs/contextd.log`)
replace the graph audit as the ongoing quality signal.
