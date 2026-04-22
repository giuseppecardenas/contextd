"""Backfill ``inferred_at`` on Section/File nodes that already have inferred edges.

The idempotent-resume logic introduced in the same commit wave as this
migration filters ``phase_relate*`` input on ``WHERE n.inferred_at IS NULL``
and writes the marker inside the worker after a successful upsert loop.
Graphs bootstrapped before that code landed have zero ``inferred_at``
properties set, so a naïve resume would treat every section as
un-processed and re-run relate on the entire corpus — exactly what the
marker was supposed to prevent.

This migration does a one-shot backfill: any Section or File with at
least one outgoing ``origin='inferred'`` edge is assumed to have been
processed by a prior relate pass, and gets its marker set. Sections
that legitimately returned zero edges on the first pass are *not*
detectable retroactively — they remain unmarked and will be
re-attempted on the next bootstrap. Acceptable: a one-time small
over-run (usually ≤10% of sections) vs. re-doing the full batch.

Idempotent: ``SET`` of an existing property to ``datetime()`` overwrites
with a nearly-identical timestamp; semantically a no-op. Fresh corpora
(no inferred edges yet) match zero nodes and write nothing.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    # Sections with any outgoing inferred edge → mark as processed.
    "MATCH (s:Section) WHERE (s)-[{origin: 'inferred'}]->() "
    "SET s.inferred_at = coalesce(s.inferred_at, datetime())",
    # Files with any outgoing inferred edge → same.
    "MATCH (f:File) WHERE (f)-[{origin: 'inferred'}]->() "
    "SET f.inferred_at = coalesce(f.inferred_at, datetime())",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=4, name="backfill_inferred_at_neo4j", up=up)
