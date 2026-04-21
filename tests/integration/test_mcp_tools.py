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
    cypher_file.write_text(cypher)

    toml_content = f"""
[corpus]
name = "test-corpus"
root = "/tmp"
[mcp.tools]
find_file = "{cypher_file}"
"""
    (corpora_dir / "test-corpus.toml").write_text(toml_content)

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
