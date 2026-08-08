from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_per_corpus_cypher_tool_appears_in_list_tools_and_is_callable(
    backend: object, tmp_path: Path
) -> None:
    """Seed a File node, register a per-corpus Cypher tool via build_tool_descriptors,
    dispatch it in-process, and assert the returned rows are correct.

    This test exercises the full corpus-tool loader + dispatch path without
    spinning up the full async stdio server.  Parametrized on both backends
    via the ``backend`` fixture in conftest.py.
    """
    from contextd.mcp.corpus_tools import build_tool_descriptors
    from contextd.mcp_server import _dispatch_tool
    from contextd.storage.base import GraphStore

    assert isinstance(backend, GraphStore)
    backend.upsert_node(
        "File",
        {"path": "/docs/readme.md", "hash": "abc", "corpus": "test-corpus"},
    )

    # Write a minimal Cypher tool file.
    cypher = "MATCH (n:File {path: $path}) RETURN n.path AS path"
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()
    cypher_file = tmp_path / "find_file.cypher"
    cypher_file.write_text(cypher, encoding="utf-8")

    # as_posix(): a Windows path interpolated into a TOML *basic* string turns
    # its backslashes into escape sequences (\\U, \\t, ...), so the file fails
    # to parse and the loader silently yields zero descriptors.
    toml_content = f"""
[corpus]
name = "test-corpus"
root = "/tmp"
[mcp.tools]
find_file = "{cypher_file.as_posix()}"
"""
    (corpora_dir / "test-corpus.toml").write_text(toml_content, encoding="utf-8")

    # Load the per-corpus tool descriptors.
    corpus_descriptors, corpus_registry = build_tool_descriptors(tmp_path)
    assert len(corpus_descriptors) == 1
    tool_desc = corpus_descriptors[0]
    assert tool_desc.name == "test-corpus.find_file"
    assert "test-corpus.find_file" in corpus_registry

    # Dispatch via _dispatch_tool.
    result = _dispatch_tool(
        "test-corpus.find_file",
        {"path": "/docs/readme.md"},
        backend,
        corpus_registry,
    )

    assert len(result) == 1
    parsed = json.loads(result[0]["text"])
    assert isinstance(parsed, list)
    assert any(row.get("path") == "/docs/readme.md" for row in parsed)


def test_describe_project_returns_summaries(backend) -> None:  # type: ignore[no-untyped-def]
    backend.upsert_node(
        "File", {"path": "a.md", "hash": "h", "corpus": "c", "summary": "summary of a"}
    )
    backend.upsert_node(
        "File", {"path": "b.md", "hash": "h", "corpus": "c", "summary": "summary of b"}
    )
    backend.upsert_edge(
        "a.md", "b.md", "REFERENCES", origin="structural", src_label="File", dst_label="File"
    )

    from contextd.mcp import tools

    overview = tools.describe_project(backend, corpus="c")
    assert len(overview.nodes) == 2
    # Most-cited node (b.md has inbound=1) should appear before a.md.
    paths = [row["path"] for row in overview.nodes]
    assert paths[0] == "b.md"


def test_query_graph_rejects_writes(backend) -> None:
    from contextd.mcp import tools
    from contextd.mcp.readonly_guard import ReadOnlyGuardError

    with pytest.raises(ReadOnlyGuardError):
        tools.query_graph(backend, "CREATE (n:File {path: 'x'})")


def test_get_node_and_list_entities_read_populated_entity(backend) -> None:  # type: ignore[no-untyped-def]
    """A populated entity node is fully readable via get_node and enumerable
    via list_entities — the gap the entity-content work closes."""
    from contextd.mcp import tools

    backend.upsert_node(
        "Ticket",
        {"id": "INTENG-1", "title": "Fix auth", "status": "open", "corpus": "c"},
    )

    node = tools.get_node(backend, "INTENG-1")
    assert node is not None
    assert node["labels"] == ["Ticket"]
    assert node["title"] == "Fix auth"
    assert node["status"] == "open"

    listed = tools.list_entities(backend, "Ticket", prop="status", value="open", corpus="c")
    assert [row["id"] for row in listed] == ["INTENG-1"]


def test_search_reaches_entity_content(backend) -> None:  # type: ignore[no-untyped-def]
    """search(kind='Ticket') hits Ticket_title_ft (migration _0006) via the
    per-label property mapping — entity content is actually searchable."""
    from contextd.mcp import tools

    backend.upsert_node("Ticket", {"id": "INTENG-1", "title": "authentication bug", "corpus": "c"})
    backend.upsert_node("Ticket", {"id": "INTENG-2", "title": "unrelated billing", "corpus": "c"})

    hits = tools.search(backend, "authentication", kind="Ticket", limit=5)
    assert [h["id"] for h in hits] == ["INTENG-1"]


def test_explain_relationship_returns_edge_provenance(backend) -> None:  # type: ignore[no-untyped-def]
    from contextd.mcp import tools

    backend.upsert_node("File", {"path": "a.md", "hash": "h", "corpus": "c"})
    backend.upsert_node("Ticket", {"id": "INTENG-1", "title": "t", "corpus": "c"})
    backend.upsert_edge(
        "a.md",
        "INTENG-1",
        "REFERENCES",
        origin="inferred",
        properties={"confidence": 0.9, "reason": "the file references the ticket"},
        src_label="File",
        dst_label="Ticket",
    )

    rows = tools.explain_relationship(backend, "a.md", "INTENG-1")
    assert len(rows) == 1
    assert rows[0]["edge_type"] == "REFERENCES"
    assert rows[0]["origin"] == "inferred"
    assert rows[0]["confidence"] == 0.9
    assert rows[0]["reason"] == "the file references the ticket"
    assert rows[0]["outbound"] is True


def test_find_reusable_filters_by_reusable_flag(backend) -> None:  # type: ignore[no-untyped-def]
    """Both Artifacts match the query text; only the reusable one is returned."""
    from contextd.mcp import tools

    backend.upsert_node(
        "Artifact",
        {"id": "A1", "description": "reusable deployment script", "reusable": True, "corpus": "c"},
    )
    backend.upsert_node(
        "Artifact",
        {"id": "A2", "description": "one-off deployment note", "reusable": False, "corpus": "c"},
    )

    rows = tools.find_reusable(backend, "deployment")
    assert [r["id"] for r in rows] == ["A1"]


def test_check_freshness_surfaces_needs_update(backend) -> None:  # type: ignore[no-untyped-def]
    from contextd.mcp import tools

    backend.upsert_node("File", {"path": "old.md", "hash": "h", "corpus": "c"})
    backend.upsert_node("File", {"path": "new.md", "hash": "h", "corpus": "c"})
    backend.upsert_edge(
        "old.md",
        "new.md",
        "NEEDS_UPDATE",
        origin="inferred",
        properties={"confidence": 0.8, "reason": "superseded guidance"},
        src_label="File",
        dst_label="File",
    )

    rows = tools.check_freshness(backend, node_id="old.md")
    assert any(
        r["edge_type"] == "NEEDS_UPDATE" and r["reason"] == "superseded guidance" for r in rows
    )


def test_whats_new_returns_recently_updated_nodes(backend) -> None:  # type: ignore[no-untyped-def]
    import datetime as dt

    from contextd.mcp import tools

    backend.upsert_node(
        "File",
        {
            "path": "fresh.md",
            "hash": "h",
            "corpus": "c",
            "summary": "s",
            "updated": dt.datetime(2026, 6, 1, tzinfo=dt.UTC),
        },
    )

    hits = tools.whats_new(backend, "2026-01-01T00:00:00Z", corpus="c")
    assert [h["node"] for h in hits] == ["fresh.md"]

    none = tools.whats_new(backend, "2026-12-01T00:00:00Z", corpus="c")
    assert none == []
