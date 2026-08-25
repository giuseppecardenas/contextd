"""Graph-backed per-unit candidate retrieval for the relate phase.

Implements the ``CandidateRetriever`` protocol with four sources, all served
from the graph via the ``GraphStore`` ABC — no provider API calls:

1. **Entities by label** — for each mintable label, existing entity names
   ranked by degree (hub concepts, not stubs, get advertised). Cached per
   ``(corpus, label)`` with a short TTL: the relate phases run hundreds of
   workers and must not issue thousands of identical queries, while the TTL
   still lets mid-phase mints converge into later prompts.
2. **Same-file sections** — every section of the source file (id + title, in
   document order). This is what makes intra-document targets emittable at
   all: Section ids are unguessable absolute ``path#anchor`` strings.
3. **Similar sections** — the unit's *stored* embedding (written at CREATE
   time by enumerate) queried against the Section vector index, fused with a
   full-text leg over Section summaries via reciprocal rank fusion — the same
   MCP-independent fusion the search tool uses.
4. **Neighbour files** — files already edged with this file, topped up with
   vector-similar Files.

Every source is exception-guarded: candidate retrieval degrades to less
context, never blocks the LLM call.
"""

from __future__ import annotations

import logging
import re
import threading
import time

from contextd.inference.context import (
    CandidateBundle,
    FileCandidate,
    SectionCandidate,
    UnitIdentity,
)
from contextd.ontology.schema import Ontology
from contextd.search import reciprocal_rank_fusion
from contextd.storage._keys import primary_key_for
from contextd.storage.base import GraphStore

_log = logging.getLogger(__name__)

# Lucene query-syntax metacharacters. Section titles are fed to
# db.index.fulltext.queryNodes verbatim as the full-text leg's query; a title
# like "Save/Load Format" starts a Lucene regex at the "/" and crashes the
# parser (TokenMgrError), silently killing the leg for every such title.
# Titles are prose, never intentional query syntax — escape everything.
_LUCENE_SPECIALS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def _lucene_escape(text: str) -> str:
    return _LUCENE_SPECIALS.sub(lambda m: "\\" + m.group(0), text)


class GraphCandidateRetriever:
    def __init__(
        self,
        ontology: Ontology,
        *,
        per_label_cap: int = 15,
        section_cap: int = 12,
        file_cap: int = 8,
        vector_k: int = 10,
        cache_ttl_seconds: float = 60.0,
    ) -> None:
        self._onto = ontology
        self._per_label_cap = per_label_cap
        self._section_cap = section_cap
        self._file_cap = file_cap
        self._vector_k = vector_k
        self._ttl = cache_ttl_seconds
        self._entity_cache: dict[tuple[str, str], tuple[float, tuple[str, ...]]] = {}
        self._lock = threading.Lock()

    def for_unit(self, store: GraphStore, *, identity: UnitIdentity) -> CandidateBundle:
        return CandidateBundle(
            entities_by_label=self._entities(store, identity.corpus),
            sections=self._sections(store, identity),
            files=self._files(store, identity),
        )

    # -- source 1: degree-ranked entities, cached -----------------------------

    def _entities(self, store: GraphStore, corpus: str) -> dict[str, tuple[str, ...]]:
        out: dict[str, tuple[str, ...]] = {}
        now = time.monotonic()
        for label in sorted(self._onto.mintable_labels()):
            key = (corpus, label)
            with self._lock:
                cached = self._entity_cache.get(key)
                if cached is not None and now - cached[0] < self._ttl:
                    if cached[1]:
                        out[label] = cached[1]
                    continue
            names = self._load_entity_names(store, corpus, label)
            with self._lock:
                self._entity_cache[key] = (now, names)
            if names:
                out[label] = names
        return out

    def _load_entity_names(self, store: GraphStore, corpus: str, label: str) -> tuple[str, ...]:
        pk = primary_key_for(label)
        try:
            rows = store.exec_read(
                # label/pk come from the ontology + _keys map, not user input.
                f"MATCH (n:{label} {{corpus: $c}}) "
                f"RETURN n.{pk} AS name ORDER BY COUNT {{ (n)--() }} DESC LIMIT $cap",
                {"c": corpus, "cap": self._per_label_cap},
            )
        except Exception as exc:
            _log.warning("candidates: entity lookup failed for %s: %s", label, exc)
            return ()
        return tuple(str(r["name"]) for r in rows if r.get("name"))

    # -- source 2 + 3: sections ------------------------------------------------

    def _sections(self, store: GraphStore, identity: UnitIdentity) -> tuple[SectionCandidate, ...]:
        seen: dict[str, SectionCandidate] = {}
        try:
            same_file = store.exec_read(
                "MATCH (s:Section {corpus: $c, path: $p}) "
                "RETURN s.id AS id, s.title AS title ORDER BY s.ordinal",
                {"c": identity.corpus, "p": identity.file_path},
            )
            for r in same_file:
                if r["id"] != identity.src_id:
                    seen[r["id"]] = SectionCandidate(id=r["id"], title=str(r.get("title") or ""))
        except Exception as exc:
            _log.warning("candidates: same-file section lookup failed: %s", exc)
        for row in self._similar(store, identity, label="Section", search_prop="summary"):
            sid = str(row.get("id") or "")
            if sid and sid != identity.src_id and sid not in seen:
                seen[sid] = SectionCandidate(id=sid, title=str(row.get("title") or ""))
            if len(seen) >= self._section_cap:
                break
        # Chunk leg: body-level similarity finds neighbours whose *summaries*
        # differ but whose text overlaps (an identifier, a quoted config key),
        # which the summary-only legs above miss.
        if len(seen) < self._section_cap:
            for sid, title in self._chunk_neighbour_sections(store, identity):
                if sid != identity.src_id and sid not in seen:
                    seen[sid] = SectionCandidate(id=sid, title=title)
                if len(seen) >= self._section_cap:
                    break
        return tuple(list(seen.values())[: self._section_cap])

    def _chunk_neighbour_sections(
        self, store: GraphStore, identity: UnitIdentity
    ) -> list[tuple[str, str]]:
        """Sections whose chunks are vector-near this unit's stored embedding."""
        vec = self._stored_embedding(store, identity)
        if vec is None:
            return []
        try:
            rows = store.vector_search(
                label="Chunk",
                property_name="embedding",
                query=vec,
                k=self._vector_k,
                filters={"corpus": identity.corpus, "parent_label": "Section"},
            )
        except Exception as exc:
            _log.warning("candidates: chunk leg failed: %s", exc)
            return []
        parent_ids: list[str] = []
        for r in rows:
            pid = r["node"].get("parent_id")
            if isinstance(pid, str) and pid not in parent_ids:
                parent_ids.append(pid)
        if not parent_ids:
            return []
        try:
            titles = store.exec_read(
                "MATCH (s:Section) WHERE s.id IN $ids RETURN s.id AS id, s.title AS title",
                {"ids": parent_ids},
            )
        except Exception as exc:
            _log.warning("candidates: chunk parent lookup failed: %s", exc)
            return []
        by_id = {str(t["id"]): str(t.get("title") or "") for t in titles}
        return [(pid, by_id[pid]) for pid in parent_ids if pid in by_id]

    # -- source 4: files -------------------------------------------------------

    def _files(self, store: GraphStore, identity: UnitIdentity) -> tuple[FileCandidate, ...]:
        seen: dict[str, FileCandidate] = {}
        try:
            linked = store.exec_read(
                "MATCH (f:File {path: $p})--(o:File) WHERE o.hash IS NOT NULL "
                "RETURN DISTINCT o.path AS path, o.name AS name LIMIT $cap",
                {"p": identity.file_path, "cap": self._file_cap},
            )
            for r in linked:
                seen[r["path"]] = FileCandidate(path=r["path"], name=str(r.get("name") or ""))
        except Exception as exc:
            _log.warning("candidates: linked-file lookup failed: %s", exc)
        for row in self._similar(store, identity, label="File", search_prop="summary"):
            fpath = str(row.get("path") or "")
            if fpath and fpath != identity.file_path and fpath not in seen:
                seen[fpath] = FileCandidate(path=fpath, name=str(row.get("name") or ""))
            if len(seen) >= self._file_cap:
                break
        return tuple(list(seen.values())[: self._file_cap])

    # -- shared hybrid-similarity leg -----------------------------------------

    def _stored_embedding(self, store: GraphStore, identity: UnitIdentity) -> list[float] | None:
        pk = primary_key_for(identity.src_label)
        try:
            rows = store.exec_read(
                f"MATCH (n:{identity.src_label}) WHERE n.{pk} = $v "
                "RETURN n.embedding AS embedding LIMIT 1",
                {"v": identity.src_id},
            )
        except Exception as exc:
            _log.warning("candidates: stored-embedding read failed: %s", exc)
            return None
        if rows and isinstance(rows[0].get("embedding"), list):
            return [float(x) for x in rows[0]["embedding"]]
        return None

    def _similar(
        self, store: GraphStore, identity: UnitIdentity, *, label: str, search_prop: str
    ) -> list[dict[str, object]]:
        """Hybrid (vector + full-text, RRF-fused) neighbours of this unit.

        Uses the unit's stored embedding — no embedder needed in the relate
        phases — and the unit's title (or filename) as the full-text query.
        Both legs are scoped to the unit's corpus server-side via ``filters``
        (the backend over-fetches before applying the predicate, so ``k``
        rows of the *right* corpus can come back). Either leg failing
        degrades to the other; both failing yields [].
        """
        vector_rows: list[dict[str, object]] = []
        fulltext_rows: list[dict[str, object]] = []
        vec = self._stored_embedding(store, identity)
        if vec is not None:
            try:
                vector_rows = store.vector_search(
                    label=label,
                    property_name="embedding",
                    query=vec,
                    k=self._vector_k,
                    filters={"corpus": identity.corpus},
                )
            except Exception as exc:
                _log.warning("candidates: vector leg failed for %s: %s", label, exc)
        query_text = _lucene_escape(identity.title or identity.rel_path.rsplit("/", 1)[-1])
        if query_text:
            try:
                fulltext_rows = store.full_text_search(
                    label=label,
                    property_name=search_prop,
                    query=query_text,
                    k=self._vector_k,
                    filters={"corpus": identity.corpus},
                )
            except Exception as exc:
                _log.warning("candidates: full-text leg failed for %s: %s", label, exc)
        if not vector_rows and not fulltext_rows:
            return []
        try:
            return reciprocal_rank_fusion(
                vector_rows, fulltext_rows, label=label, limit=self._vector_k
            )
        except Exception as exc:
            _log.warning("candidates: fusion failed for %s: %s", label, exc)
            return []
