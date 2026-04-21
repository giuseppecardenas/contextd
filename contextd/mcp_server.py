"""stdio MCP server for Contextd.

Spawned by MCP clients (Claude Desktop, Cursor) over stdio. Each tool
is registered with a JSON schema so clients can introspect the
surface. Tool bodies delegate to contextd.mcp.tools.

Known limitation (Delta C): importing CONTEXTD_HOME from contextd.cli pulls in
click, rich, and all CLI-command side-effects at MCP startup. Moving
CONTEXTD_HOME to a standalone utility module is deferred to the M6
refactor backlog — acceptable for single-user local use.
"""

from __future__ import annotations

from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from contextd.cli import CONTEXTD_HOME
from contextd.config import Config
from contextd.mcp import tools
from contextd.storage.factory import build_graph_store


async def run() -> None:
    cfg = (
        Config.load(CONTEXTD_HOME / "config.toml")
        if (CONTEXTD_HOME / "config.toml").exists()
        else Config.load_default()
    )
    store = build_graph_store(cfg)
    store.connect()

    server: Server[Any] = Server("contextd")

    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def _list() -> list[Tool]:
        return [
            Tool(
                name="describe_project",
                description="Compact project primer — top-N most-cited nodes with summaries.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "corpus": {"type": "string"},
                        "n": {"type": "integer", "default": 40},
                    },
                },
            ),
            Tool(
                name="search",
                description="Hybrid search — full-text first, vector fallback.",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
            Tool(
                name="related",
                description="Outbound+inbound traversal within N hops (1-5).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string"},
                        "depth": {
                            "type": "integer",
                            "default": 2,
                            "minimum": 1,
                            "maximum": 5,
                        },
                    },
                    "required": ["node_id"],
                },
            ),
            Tool(
                name="inbound",
                description="What cites this node?",
                inputSchema={
                    "type": "object",
                    "properties": {"node_id": {"type": "string"}},
                    "required": ["node_id"],
                },
            ),
            Tool(
                name="outbound",
                description="What does this node cite?",
                inputSchema={
                    "type": "object",
                    "properties": {"node_id": {"type": "string"}},
                    "required": ["node_id"],
                },
            ),
            Tool(
                name="get_file_summary",
                description="Summary + key points for a single file.",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            Tool(
                name="section_tree",
                description="Outline of a file (section-granular corpora).",
                inputSchema={
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="query_graph",
                description="Raw read-only Cypher escape hatch.",
                inputSchema={
                    "type": "object",
                    "properties": {"cypher": {"type": "string"}},
                    "required": ["cypher"],
                },
            ),
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def _call(name: str, arguments: dict[str, Any]) -> Any:
        match name:
            case "describe_project":
                ov = tools.describe_project(
                    store,
                    corpus=arguments.get("corpus"),
                    n=arguments.get("n", 40),
                )
                return [{"type": "text", "text": str(ov.nodes)}]
            case "search":
                return [{"type": "text", "text": str(tools.search(store, **arguments))}]
            case "related":
                return [{"type": "text", "text": str(tools.related(store, **arguments))}]
            case "inbound":
                return [{"type": "text", "text": str(tools.inbound(store, **arguments))}]
            case "outbound":
                return [{"type": "text", "text": str(tools.outbound(store, **arguments))}]
            case "get_file_summary":
                return [{"type": "text", "text": str(tools.get_file_summary(store, **arguments))}]
            case "section_tree":
                return [{"type": "text", "text": str(tools.section_tree(store, **arguments))}]
            case "query_graph":
                return [
                    {"type": "text", "text": str(tools.query_graph(store, arguments["cypher"]))}
                ]
            case _:
                raise ValueError(f"Unknown tool: {name}")

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main() -> None:
    import asyncio

    asyncio.run(run())


if __name__ == "__main__":
    main()
