r"""Purge inference-minted phantom File/Section stubs; canonicalize path identity.

Two one-shot cleanups for graphs built before the node-identity fix landed.

1. **Phantom stubs.** Earlier ``phase_relate*`` minted a ``File``/``Section``
   node for every inferred-edge target the LLM named — including references to
   things that were never real files/sections (relative paths, mangled section
   numbers like ``§12.2.5``). Those stubs carry no ``hash`` (File) / no ``path``
   (Section) and polluted ``query_graph`` / ``related`` / ``inbound`` results
   with confusing path-less "sections". The indexer no longer creates them (it
   resolves the reference to an existing node or drops the edge); this removes
   the ones already in the graph. ``DETACH DELETE`` also drops the now-dangling
   inferred edges that pointed at them.

2. **Path-separator drift.** Node identity is a path string. A tree indexed by
   more than one code path / OS accumulated both ``C:\x`` (backslash) and
   ``C:/x`` (forward-slash) identities for *different* files; the fix pins the
   forward-slash form via ``canonical_path``. Re-writing the legacy backslash
   identities here means the next re-index MERGEs against the existing node and
   updates it in place instead of creating a forward-slash twin and orphaning
   the original.

Idempotent: after one run no ``hash``-null File / ``path``-null Section remains
and no identity contains a backslash, so a re-run matches zero rows. A real
``File`` always has a ``hash`` and a real ``Section`` always has a ``path``
(both set at enumerate time), so the DELETE predicates never touch legitimate
nodes.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    # 1. Drop phantom stubs (DETACH also removes their dangling inferred edges).
    "MATCH (s:Section) WHERE s.path IS NULL DETACH DELETE s",
    "MATCH (f:File) WHERE f.hash IS NULL DETACH DELETE f",
    # 2. Canonicalize legacy backslash identities to forward slashes. Raw
    # strings: Cypher needs the two-character token '\\' (one backslash) — a
    # plain Python literal would collapse it to '\' and Cypher would read the
    # backslash as escaping the closing quote.
    r"MATCH (f:File) WHERE f.path CONTAINS '\\' SET f.path = replace(f.path, '\\', '/')",
    r"MATCH (s:Section) WHERE s.id CONTAINS '\\' "
    r"SET s.id = replace(s.id, '\\', '/'), "
    r"s.path = replace(s.path, '\\', '/'), "
    r"s.file_id = replace(s.file_id, '\\', '/')",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=5, name="purge_phantom_stubs_canonicalize_paths_neo4j", up=up)
