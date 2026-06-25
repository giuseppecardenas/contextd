# Ontology Reference

The base ontology defines which node types and edge types are valid across all corpora. Per-corpus configuration can introduce aliases (alternate names for base types) without adding new canonical types. AI-inferred edges are validated against the ontology at write time; unrecognised types are silently discarded.

The base ontology file: `contextd/ontology/base.json`.

---

## Node types

| Node type | Primary key | Key properties |
|---|---|---|
| `File` | `path` | `path`, `name`, `type`, `hash`, `size`, `updated`, `embedding`, `summary`, `key_points`, `summary_generated_at`, `summary_confidence`, `corpus` |
| `Section` | `id` | `id`, `anchor`, `title`, `level`, `path`, `corpus`, `file_id`, `ordinal`, `embedding`, `summary`, `key_points`, `entities_mentioned`, `summary_generated_at`, `summary_confidence` |
| `Artifact` | `id` | `id`, `title`, `description`, `reusable`, `created`, `updated`, `corpus` |
| `Ticket` | `id` | `id`, `title`, `status`, `created`, `updated`, `corpus` |
| `Pattern` | `name` | `name`, `description`, `when_to_use`, `examples`, `corpus` |
| `Technology` | `name` | `name`, `version` |
| `Client` | `name` | `name` |
| `Repo` | `name` | `name`, `url` |
| `Service` | `name` | `name`, `repo` |
| `Integration` | `name` | `name`, `type` |
| `Risk` | `description` | `description`, `severity`, `corpus` |
| `WorkSession` | `id` | `id`, `start`, `end`, `focus_area` |
| `Corpus` | `name` | `name`, `root`, `registered_at`, `content_profile` |
| `Meta` | `schema_version` | `schema_version`, `contextd_version`, `backend_name`, `initialised_at` |

Primary keys are enforced via uniqueness constraints in both backend migrations. The canonical map is `contextd/storage/_keys.py::PRIMARY_KEY_BY_LABEL`.

`Section` nodes are only populated in section-granular corpora (`[corpus] granularity = "section"`).

`Risk.description` is the primary key by design: the inferrer emits `Risk` nodes identified solely by their description text, and MERGE semantics collapse identical-description upserts into one node. Two audit-gap entries with identical phrasing will merge — considered correct for the Acme use case; if co-existing same-phrased Risks are needed in future, a content-hash-derived `id` field with a companion migration is the remedy.

---

## Edge types

| Edge type | Canonical usage | Origin |
|---|---|---|
| `CONTAINS` | File → Section | structural |
| `PARENT_OF` | Section → Section | structural |
| `NEXT_SIBLING` | Section → Section | structural |
| `BELONGS_TO` | File → Ticket, Artifact → Ticket | inferred |
| `CREATED_BY` | Artifact → Ticket | inferred |
| `DOCUMENTS` | Artifact → Pattern, Artifact → Technology | inferred |
| `DOCUMENTED_IN` | File → Artifact | inferred |
| `APPLIES_TO` | Artifact → Client, Artifact → Repo | inferred |
| `PART_OF` | File → Repo | inferred |
| `SIMILAR_TO` | Ticket → Ticket, File → File | inferred |
| `RELATED_TO` | Ticket → Ticket | inferred |
| `REFERENCES` | Artifact → Artifact, File → File, Section → Section | inferred |
| `SUPERSEDES` | Artifact → Artifact | inferred |
| `CONTRADICTS` | Artifact → Artifact | inferred |
| `USES` | File → Technology | inferred |
| `MODIFIES` | File → File | inferred |
| `DEPENDS_ON` | Service → Service | inferred |
| `IDENTIFIES_RISK` | Artifact → Risk | inferred |
| `RECOMMENDS` | Artifact → Pattern | inferred |
| `EDITED_DURING` | File → WorkSession | structural |
| `VERIFIED_ON` | Artifact → Artifact | manual |
| `NEEDS_UPDATE` | Artifact → Artifact | manual |

`CONTAINS`, `PARENT_OF`, `NEXT_SIBLING`, and `EDITED_DURING` are always `structural`. `VERIFIED_ON` and `NEEDS_UPDATE` are always `manual`. All others are typically `inferred` but any origin value is technically legal.

---

## Edge origin values

Every edge carries an `origin` property. Valid values:

| Value | Meaning |
|---|---|
| `structural` | Derived deterministically by a parser (heading hierarchy, markdown links). Preserved across re-index. |
| `inferred` | Produced by the AI relationship-inference step. Wiped and replaced on re-index. |
| `manual` | Inserted by a human via direct Cypher or a future `contextd review-relationships` command. Preserved across re-index. |

Wipe-and-replace on re-index (`delete_edges(..., origin="inferred")`) operates only on `inferred` edges. The ABC's `delete_edges` raises `ValueError` when both `origin` and `edge_type` are `None` — an unfiltered delete would violate this invariant.

---

## Per-corpus aliases

Aliases let a corpus use domain vocabulary in its Cypher queries and inferred relationships without introducing new canonical types.

### Node-label aliases

Declared inline in the corpus TOML under `[ontology.aliases]`. Each alias maps a domain name to a canonical node type. The alias is resolved transparently at inference time — the underlying storage always uses the canonical label.

```toml
[ontology.aliases]
Registry = "Pattern"
FRRow    = "Ticket"
LuaFile  = "File"
GapEntry = "Risk"
```

Applied via `Ontology.with_aliases(aliases)` in `contextd/ontology/schema.py`. Raises `OntologyError` if any target is not a declared canonical node type.

### Edge-label aliases

Declared in a separate JSON file pointed to by `[ontology] overrides = "path/to/overrides.json"`. The JSON object's `edge_label_aliases` key maps domain edge names to canonical edge types.

```json
{
  "edge_label_aliases": {
    "CITES":      "REFERENCES",
    "CONSUMES":   "USES",
    "SCHEMA_FOR": "DOCUMENTS",
    "REGISTERS":  "DOCUMENTS",
    "CLOSES_GAP": "SUPERSEDES"
  }
}
```

Applied via `apply_overrides(ontology, overrides_path)` in `contextd/ontology/overrides.py`, which calls `Ontology.with_edge_aliases(edge_aliases)`. Raises `OntologyError` if any target is not a declared canonical edge type.

**Distinction from node aliases:** Node aliases are inline in the corpus TOML; edge aliases require a separate file. Both are applied in `_build_pipeline_deps` in `contextd/cli/corpora.py` — `with_aliases()` first, then `apply_overrides()`.

---

## Validation

```python
ontology.validate_node("Pattern")   # passes
ontology.validate_node("Registry")  # passes if alias declared; raises OntologyError otherwise
ontology.validate_edge("REFERENCES", origin="inferred")  # passes
ontology.validate_edge("HALLUCINATED", origin="inferred")  # raises OntologyError
```

The `RelationshipInferrer` calls `ontology.validate_edge()` on every AI-inferred relationship before writing it to the graph. Invalid types are silently discarded (logged at DEBUG level) — not raised — so a single hallucinated edge type does not abort the batch.

---

## Extending the ontology

To add a new node type or edge type:

1. **`contextd/ontology/base.json`** — add the node type (with property list) or edge type name.
2. **If it is a PK-bearing node type:** add it to `contextd/storage/_keys.py::PRIMARY_KEY_BY_LABEL`.
3. **Migration DDL (both backends):**
   - `contextd/migrations/memgraph/` — add a new migration file with `CREATE CONSTRAINT ON (n:NewType) ASSERT n.<pk> IS UNIQUE`.
   - `contextd/migrations/neo4j/` — add a matching migration with Neo4j constraint syntax.
4. **Tests** — add a unit test to `tests/unit/test_ontology.py` asserting the new type validates correctly.

Node types that do not require a primary-key uniqueness constraint (e.g. types always created subordinate to another node) still need an entry in `PRIMARY_KEY_BY_LABEL` for `upsert_node` and `delete_edges` to work correctly — use whichever property best uniquely identifies the node within its label.
