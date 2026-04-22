"""Full-text index on Section.summary for section-granular corpora.

The baseline (_0001) created ``File_summary_ft`` so ``search(kind='File')``
could call ``db.index.fulltext.queryNodes``. Section-granular corpora
(spec §5.11) promote H2/H3/H4 headings to first-class :Section nodes with
their own summaries, which ``search(kind='Section')`` needs a matching
index for. Without this, the call raises ``ProcedureCallFailed: There is
no such fulltext schema index: Section_summary_ft``.

Idempotent: ``IF NOT EXISTS`` guards the DDL so replays are safe.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    "CREATE FULLTEXT INDEX Section_summary_ft IF NOT EXISTS FOR (s:Section) ON EACH [s.summary]",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=3, name="section_fulltext_neo4j", up=up)
