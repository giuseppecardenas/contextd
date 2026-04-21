"""Integration tests for Runeledger-PRD Cypher tool files.

Exercises the four Cypher files in examples/runeledger-prd/tools/ against
both storage backends (parametrized via the ``backend`` fixture in conftest.py).

These tests load the *actual* Cypher from disk so that accidental future edits
to the .cypher files are caught immediately rather than silently diverging from
a hand-copied string in the test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

# Resolve the tools directory relative to this file so the tests work from any
# working directory.  examples/ lives at the repo root, three levels above tests/.
_REPO_ROOT = Path(__file__).parent.parent.parent
_TOOLS_DIR = _REPO_ROOT / "examples" / "runeledger-prd" / "tools"


def _run(
    backend: Any, cypher_file: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Load a Cypher file from the tools directory and run it."""
    cypher = (_TOOLS_DIR / cypher_file).read_text()
    return backend.exec_read(cypher, params or {})


def _seed_full_graph(backend: Any) -> None:
    """Seed a minimal four-surface Runeledger graph.

    Pattern: combat_roll
      - (Section s-consumer) -[:USES]-> (Pattern)
      - (Section s-schema)   -[:DOCUMENTS]-> (Pattern)
      - (File   combat.lua)  -[:DOCUMENTS]-> (Pattern)
      - (Ticket FR-COMBAT-001) -[:DOCUMENTS]-> (Pattern)
    """
    pattern_id = backend.upsert_node("Pattern", {"name": "combat_roll"})
    consumer_id = backend.upsert_node(
        "Section",
        {"id": "s-consumer", "title": "Consumer", "corpus": "runeledger"},
    )
    schema_id = backend.upsert_node(
        "Section",
        {"id": "s-schema", "title": "Schema", "corpus": "runeledger"},
    )
    lua_id = backend.upsert_node(
        "File",
        {"path": "mods/base/combat.lua", "hash": "abc", "corpus": "runeledger"},
    )
    fr_id = backend.upsert_node(
        "Ticket",
        {"id": "FR-COMBAT-001", "corpus": "runeledger"},
    )
    backend.upsert_edge(
        consumer_id, pattern_id, "USES", "structural", src_label="Section", dst_label="Pattern"
    )
    backend.upsert_edge(
        schema_id, pattern_id, "DOCUMENTS", "structural", src_label="Section", dst_label="Pattern"
    )
    backend.upsert_edge(
        lua_id, pattern_id, "DOCUMENTS", "structural", src_label="File", dst_label="Pattern"
    )
    backend.upsert_edge(
        fr_id, pattern_id, "DOCUMENTS", "structural", src_label="Ticket", dst_label="Pattern"
    )


def test_four_surface_returns_all_four_surfaces(backend: Any) -> None:
    """four_surface.cypher returns one row with all four surfaces populated."""
    _seed_full_graph(backend)

    rows = _run(backend, "four_surface.cypher", {"registry_name": "combat_roll"})

    assert len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}"
    row = rows[0]
    assert row["consumer_section"] == "s-consumer"
    assert row["schema_section"] == "s-schema"
    assert row["lua_file"] == "mods/base/combat.lua"
    assert row["fr_row"] == "FR-COMBAT-001"


def test_four_surface_missing_surface_shows_null(backend: Any) -> None:
    """four_surface.cypher returns NULL for absent lua and fr surfaces."""
    # Seed only consumer + schema; no Lua file, no Ticket.
    pattern_id = backend.upsert_node("Pattern", {"name": "combat_roll"})
    consumer_id = backend.upsert_node(
        "Section",
        {"id": "s-consumer", "title": "Consumer", "corpus": "runeledger"},
    )
    schema_id = backend.upsert_node(
        "Section",
        {"id": "s-schema", "title": "Schema", "corpus": "runeledger"},
    )
    backend.upsert_edge(
        consumer_id, pattern_id, "USES", "structural", src_label="Section", dst_label="Pattern"
    )
    backend.upsert_edge(
        schema_id, pattern_id, "DOCUMENTS", "structural", src_label="Section", dst_label="Pattern"
    )

    rows = _run(backend, "four_surface.cypher", {"registry_name": "combat_roll"})

    assert len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}"
    row = rows[0]
    assert row["consumer_section"] == "s-consumer"
    assert row["schema_section"] == "s-schema"
    assert row["lua_file"] is None
    assert row["fr_row"] is None


def test_dangling_finds_missing_surfaces(backend: Any) -> None:
    """dangling.cypher identifies a Pattern missing the Lua surface."""
    # Seed consumer + schema + fr, but NOT the lua File.
    pattern_id = backend.upsert_node("Pattern", {"name": "combat_roll"})
    consumer_id = backend.upsert_node(
        "Section",
        {"id": "s-consumer", "title": "Consumer", "corpus": "runeledger"},
    )
    schema_id = backend.upsert_node(
        "Section",
        {"id": "s-schema", "title": "Schema", "corpus": "runeledger"},
    )
    fr_id = backend.upsert_node(
        "Ticket",
        {"id": "FR-COMBAT-001", "corpus": "runeledger"},
    )
    backend.upsert_edge(
        consumer_id, pattern_id, "USES", "structural", src_label="Section", dst_label="Pattern"
    )
    backend.upsert_edge(
        schema_id, pattern_id, "DOCUMENTS", "structural", src_label="Section", dst_label="Pattern"
    )
    backend.upsert_edge(
        fr_id, pattern_id, "DOCUMENTS", "structural", src_label="Ticket", dst_label="Pattern"
    )

    rows = _run(backend, "dangling.cypher")

    # The dangling query only returns rows for Patterns with at least one missing surface.
    assert len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}"
    row = rows[0]
    assert row["has_consumer"] is True
    assert row["has_schema"] is True
    assert row["has_lua"] is False
    assert row["has_fr"] is True


def test_stale_shas_filters_correctly(backend: Any) -> None:
    """stale_shas.cypher returns only Risk nodes whose description contains SHA=pending."""
    backend.upsert_node(
        "Risk",
        {"description": "SHA=pending: foo", "severity": 2},
    )
    backend.upsert_node(
        "Risk",
        {"description": "SHA=abc123: bar", "severity": 1},
    )
    backend.upsert_node(
        "Risk",
        {"description": "SHA=pending: baz", "severity": 3},
    )

    rows = _run(backend, "stale_shas.cypher")

    descriptions = [r["entry"] for r in rows]
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}: {rows}"
    assert all("SHA=pending" in d for d in descriptions)
    # Ordered by severity ascending.
    severities = [r["severity"] for r in rows]
    assert severities == sorted(severities)
