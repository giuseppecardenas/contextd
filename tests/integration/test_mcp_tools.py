import pytest

pytestmark = pytest.mark.integration


def test_describe_project_returns_summaries(backend) -> None:
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
