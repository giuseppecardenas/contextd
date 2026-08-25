"""stdio MCP server for Contextd.

Spawned by MCP clients (Claude Desktop, Cursor) over stdio. Each tool
is registered with a JSON schema so clients can introspect the
surface. Tool bodies delegate to contextd.mcp.tools.

The home-directory accessor ``contextd_home()`` is imported from
``contextd._paths`` rather than ``contextd.cli`` so the MCP process
doesn't pull in click/rich — SD #69 fixed the Delta-C import coupling
that existed in the initial M7.3 implementation.

Per-corpus tools are registered at startup by scanning
``~/.contextd/corpora/*.toml`` for ``[mcp.tools]`` entries.  Each
entry maps a tool name to a Cypher file; the resulting tools are
namespaced ``<corpus>.<tool>`` to avoid collisions with the 8 generic
tools (which never contain a dot in their names).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from contextd._paths import contextd_home
from contextd.config import Config, SearchConfig
from contextd.inference.prompts import PromptRenderer
from contextd.inference.translate import QueryTranslator
from contextd.mcp import tools
from contextd.mcp.corpus_tools import (
    CorpusTool,
    build_tool_descriptors,
    dispatch_corpus_tool,
)
from contextd.ontology.schema import Ontology
from contextd.providers.base import EmbeddingProvider
from contextd.providers.factory import (
    ProviderFactoryError,
    build_embedding_provider,
    build_inference_provider,
)
from contextd.storage.base import GraphStore
from contextd.storage.factory import build_graph_store

_GENERIC_TOOL_DESCRIPTORS: list[Tool] = [
    Tool(
        name="describe_project",
        description="Compact project primer — top-N most-cited File nodes with summaries.",
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
        description=(
            "Hybrid search (vector + full-text, RRF-fused) over retrieval chunks "
            "(default kind: Chunk), collapsed small-to-big to the best enclosing "
            "Section/File with an `evidence` block (matched text, line range, "
            "neighbour context). Other kinds (File, Section, Topic, Artifact, "
            "Ticket, Pattern, Risk) return flat node rows. Falls back to "
            "full-text when no embedder is configured or the kind has no vector "
            "index."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "kind": {
                    "type": "string",
                    "description": "Node label to search (default: Chunk).",
                },
                "corpus": {
                    "type": "string",
                    "description": "Restrict to one corpus (default: all).",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum rows to return.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "fulltext", "vector"],
                    "description": "Override ranking mode (default from server config: hybrid).",
                },
                "profiles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        'Chunk profiles to query, e.g. ["fine", "coarse"] '
                        "(default: server config, else every profile present)."
                    ),
                },
                "return_unit": {
                    "type": "string",
                    "enum": ["chunk", "section", "file", "auto"],
                    "description": (
                        "Unit to collapse chunk hits to (default from server config: auto "
                        "= the Section when at least auto_merge_threshold of its chunks hit, "
                        "else the chunk)."
                    ),
                },
                "window": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Neighbour chunks attached as evidence context per side.",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="expand_chunk",
        description=(
            "A retrieval chunk with its neighbouring chunks (same parent and "
            "profile, by ordinal) and the parent Section/File summary — show me "
            "more around this search hit."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "window": {
                    "type": "integer",
                    "default": 2,
                    "minimum": 0,
                    "maximum": 10,
                },
            },
            "required": ["chunk_id"],
        },
    ),
    Tool(
        name="topics",
        description=(
            "Cross-document topics (RAPTOR-style cluster summaries over Sections/"
            "Files) with their members. Ranked by hybrid search when `query` is "
            "given, else listed by layer and size. Empty unless the corpus has "
            "[topics] enabled."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "corpus": {"type": "string"},
                "query": {"type": "string"},
                "layer": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "default": 20},
            },
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
    Tool(
        name="get_node",
        description=(
            "Full labels + properties for one node, matched by path/id/name. "
            "Entity-aware counterpart to get_file_summary (which is File-only)."
        ),
        inputSchema={
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    ),
    Tool(
        name="explain_relationship",
        description=(
            "Direct edges between two nodes with provenance: edge type, "
            "direction, origin (inferred/structural/manual), confidence, and the "
            "inferrer's reason."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["source", "target"],
        },
    ),
    Tool(
        name="ticket_dossier",
        description=(
            "A ticket's whole neighborhood in one call: connected files, risks, "
            "artifacts, and related tickets, each with edge type and summary."
        ),
        inputSchema={
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
    ),
    Tool(
        name="find_reusable",
        description=(
            "Reusable Artifact nodes ranked by full-text relevance to the query "
            "(reusable=true only). Check before creating new artifacts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="list_entities",
        description=(
            "List nodes of an entity kind (e.g. Integration, Ticket) with their "
            "properties, optionally filtered by a property equality and/or corpus."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "kind": {"type": "string"},
                "prop": {"type": "string"},
                "value": {"type": "string"},
                "corpus": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["kind"],
        },
    ),
    Tool(
        name="check_freshness",
        description=(
            "Freshness signals (SUPERSEDES/CONTRADICTS/NEEDS_UPDATE edges) for a "
            "node or across a corpus, so a caller can judge whether a hit is "
            "still current. Supply node_id or corpus."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "corpus": {"type": "string"},
                "limit": {"type": "integer", "default": 200},
            },
        },
    ),
    Tool(
        name="find_contradictions",
        description=(
            "CONTRADICTS edge pairs, optionally narrowed to a topic, so "
            "conflicting guidance can be reconciled. Sparse; often empty."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    ),
    Tool(
        name="whats_new",
        description=(
            "Nodes changed at or after an ISO-8601 timestamp, newest first — "
            "changed source documents for catching up on an evolving corpus. "
            "Optionally scoped to a corpus."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "ISO-8601 timestamp."},
                "corpus": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["since"],
        },
    ),
    Tool(
        name="timeline",
        description=(
            "Chronological view of nodes relevant to a node or topic, plus the "
            "SUPERSEDES chains among them, to show how a decision evolved."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "topic": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    ),
    Tool(
        name="ask",
        description=(
            "Natural-language question answered by translating to Cypher and "
            "running it. Returns the generated cypher and the rows. Requires an "
            "inference provider; one LLM call per invocation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "corpus": {"type": "string"},
            },
            "required": ["question"],
        },
    ),
    Tool(
        name="grep_corpus",
        description=(
            "Regex search over corpus file contents on disk, for exact strings "
            "(flag names, ids, config keys) that summaries paraphrase away. "
            "Scoped to one corpus or all."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "corpus": {"type": "string"},
                "limit": {"type": "integer", "default": 100},
            },
            "required": ["pattern"],
        },
    ),
]


def _text(obj: Any) -> list[dict[str, str]]:
    """MCP tool result shape — JSON-serialised payload under 'text'.

    Previously we used ``str(obj)`` which emits Python repr (single-quoted,
    Python True/None, etc.) — LLM clients couldn't parse it as structured
    data. ``json.dumps(..., default=str)`` renders real JSON and falls back
    to ``str()`` for non-serialisable objects (datetimes, Path).
    """
    return [{"type": "text", "text": json.dumps(obj, default=str)}]


def _dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    store: GraphStore,
    corpus_registry: dict[str, CorpusTool] | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    search_cfg: SearchConfig | None = None,
    translator: QueryTranslator | None = None,
    home: Path | None = None,
    profile_weights: dict[str, float] | None = None,
) -> Any:
    """Route a tool-call to the right tools.X body.

    Extracted from _call so tests can assert dispatch behaviour without
    spinning up the full async stdio loop.

    Per-corpus tools are dispatched when ``name`` contains a dot (the
    ``<corpus>.<tool>`` namespace separator).  The ``corpus_registry``
    must be provided for those calls; if it is None or the name is not
    registered, a ValueError is raised.

    ``embedder`` and ``search_cfg`` are threaded through to the ``search``
    tool for hybrid ranking; both are optional so tests can dispatch the
    other tools without constructing them (``search`` then runs full-text
    only against ``SearchConfig`` defaults).
    """
    if "." in name:
        reg = corpus_registry or {}
        result = dispatch_corpus_tool(name, arguments, reg, store.exec_read)
        return _text(result)

    match name:
        case "describe_project":
            ov = tools.describe_project(
                store,
                corpus=arguments.get("corpus"),
                n=arguments.get("n", 40),
            )
            return _text(ov.nodes)
        case "search":
            cfg_s = search_cfg or SearchConfig()
            # Server-side knobs (rrf_k/fetch_k/weights) come from config, never
            # from client arguments; only `mode` is a client-facing override.
            return _text(
                tools.search(
                    store,
                    arguments["query"],
                    kind=arguments.get("kind"),
                    limit=arguments.get("limit", 20),
                    embedder=embedder,
                    mode=arguments.get("mode", cfg_s.mode),
                    rrf_k=cfg_s.rrf_k,
                    fetch_k=cfg_s.fetch_k,
                    vector_weight=cfg_s.vector_weight,
                    fulltext_weight=cfg_s.fulltext_weight,
                    corpus=arguments.get("corpus"),
                    profiles=arguments.get("profiles") or cfg_s.chunk_profiles,
                    profile_weights=profile_weights,
                    return_unit=arguments.get("return_unit", cfg_s.return_unit),
                    auto_merge_threshold=cfg_s.auto_merge_threshold,
                    window=arguments.get("window", cfg_s.window),
                    max_evidence_chars=cfg_s.max_evidence_chars,
                )
            )
        case "expand_chunk":
            return _text(
                tools.expand_chunk(store, arguments["chunk_id"], window=arguments.get("window", 2))
            )
        case "topics":
            cfg_s = search_cfg or SearchConfig()
            return _text(
                tools.topics(
                    store,
                    corpus=arguments.get("corpus"),
                    query=arguments.get("query"),
                    layer=arguments.get("layer"),
                    limit=arguments.get("limit", 20),
                    embedder=embedder,
                    mode=cfg_s.mode,
                    rrf_k=cfg_s.rrf_k,
                    fetch_k=cfg_s.fetch_k,
                )
            )
        case "related":
            return _text(tools.related(store, **arguments))
        case "inbound":
            return _text(tools.inbound(store, **arguments))
        case "outbound":
            return _text(tools.outbound(store, **arguments))
        case "get_file_summary":
            return _text(tools.get_file_summary(store, **arguments))
        case "section_tree":
            return _text(tools.section_tree(store, **arguments))
        case "query_graph":
            return _text(tools.query_graph(store, arguments["cypher"]))
        case "get_node":
            return _text(tools.get_node(store, **arguments))
        case "explain_relationship":
            return _text(tools.explain_relationship(store, **arguments))
        case "ticket_dossier":
            return _text(tools.ticket_dossier(store, **arguments))
        case "find_reusable":
            return _text(tools.find_reusable(store, **arguments))
        case "list_entities":
            return _text(tools.list_entities(store, **arguments))
        case "check_freshness":
            return _text(tools.check_freshness(store, **arguments))
        case "find_contradictions":
            return _text(tools.find_contradictions(store, **arguments))
        case "whats_new":
            return _text(tools.whats_new(store, **arguments))
        case "timeline":
            return _text(tools.timeline(store, **arguments))
        case "ask":
            return _text(
                tools.ask(
                    store,
                    translator,
                    arguments["question"],
                    corpus=arguments.get("corpus"),
                )
            )
        case "grep_corpus":
            if home is None:
                raise ValueError("grep_corpus requires the contextd home directory")
            return _text(
                tools.grep_corpus(
                    home,
                    arguments["pattern"],
                    corpus=arguments.get("corpus"),
                    limit=arguments.get("limit", 100),
                )
            )
        case _:
            raise ValueError(f"Unknown tool: {name}")


async def run() -> None:
    home = contextd_home()
    cfg = (
        Config.load(home / "config.toml")
        if (home / "config.toml").exists()
        else Config.load_default()
    )
    store = build_graph_store(cfg)
    store.connect()
    try:
        # Build the query-time embedder for hybrid search. A missing API key
        # (or unset api_key_env for a local server) raises ProviderFactoryError;
        # we swallow it and leave embedder=None so the server still starts and
        # `search` degrades to full-text. A *down* local server does not raise
        # here — it raises at first embed(), which tools.search catches.
        embedder: EmbeddingProvider | None
        try:
            embedder = build_embedding_provider(cfg)
        except ProviderFactoryError:
            embedder = None

        # Build the NL->Cypher translator for the `ask` tool, mirroring the
        # embedder's degrade-to-None pattern: a missing inference provider (no
        # API key, etc.) leaves the server running with `ask` disabled rather
        # than failing startup.
        translator: QueryTranslator | None
        try:
            translator = QueryTranslator(
                provider=build_inference_provider(cfg),
                renderer=PromptRenderer(home / "prompts"),
                ontology=Ontology.load_base(),
            )
        except ProviderFactoryError:
            translator = None

        server: Server[Any] = Server("contextd")

        # Build the full tool list (generic 8 + per-corpus) and the
        # corpus-tool dispatch registry.  Done after store.connect() so
        # that the home-directory is accessible and any per-corpus TOML
        # parse failures are surfaced before the server loop starts.
        corpus_descriptors, corpus_registry = build_tool_descriptors(home)
        all_descriptors: list[Tool] = _GENERIC_TOOL_DESCRIPTORS + corpus_descriptors
        profile_weights = _load_profile_weights(home)

        @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
        async def _list() -> list[Tool]:
            return all_descriptors

        @server.call_tool()  # type: ignore[untyped-decorator]
        async def _call(name: str, arguments: dict[str, Any]) -> Any:
            try:
                return _dispatch_tool(
                    name,
                    arguments,
                    store,
                    corpus_registry,
                    embedder=embedder,
                    search_cfg=cfg.search,
                    translator=translator,
                    home=home,
                    profile_weights=profile_weights,
                )
            except Exception as exc:
                # Render the error as the tool's text payload so the MCP
                # client sees a structured response instead of a protocol
                # exception. Read-only-guard rejections, malformed args,
                # and backend errors all flow through here.
                return _text({"error": f"{type(exc).__name__}: {exc}"})

        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        store.close()


def _load_profile_weights(home: Path) -> dict[str, float]:
    """Chunk-profile RRF weights from every registered corpus (last wins).

    The server does not otherwise load corpus configs; a corpus whose TOML
    fails to parse is skipped (it would have failed indexing too) rather than
    blocking startup.
    """
    from contextd.corpus_config import CorpusConfig

    weights: dict[str, float] = {}
    corpora_dir = home / "corpora"
    if not corpora_dir.is_dir():
        return weights
    for toml_path in sorted(corpora_dir.glob("*.toml")):
        try:
            cfg = CorpusConfig.load(toml_path)
        except Exception:
            continue
        for profile in cfg.chunking.profiles:
            weights[profile.name] = profile.weight
    return weights


def main() -> None:
    import asyncio

    asyncio.run(run())


if __name__ == "__main__":
    main()


def _build_all_tool_descriptors(home: Path) -> tuple[list[Tool], dict[str, CorpusTool]]:
    """Public helper: generic tools + per-corpus tools from *home*.

    Intended for tests and tooling that need the full surface without
    running the async server.
    """
    corpus_descriptors, corpus_registry = build_tool_descriptors(home)
    all_descriptors = _GENERIC_TOOL_DESCRIPTORS + corpus_descriptors
    return all_descriptors, corpus_registry
