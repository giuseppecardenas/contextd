"""Full-text indexes on typed-entity content properties.

Until entity content extraction (the indexer now populates ``Ticket.title``,
``Pattern.description``, ``Risk.description`` and the other declared entity
fields instead of leaving bare ``{pk, corpus}`` stubs), the only content-bearing
full-text indexes were on ``File.summary`` / ``Section.summary`` (_0001, _0003)
and ``Artifact.description`` (_0001). This migration adds the remaining
full-text indexes so ``search`` can reach the newly-populated entity content via
``db.index.fulltext.queryNodes``.

Index names follow the ``{Label}_{property}_ft`` convention that
``full_text_search`` interpolates (``contextd/storage/neo4j.py``); a name that
does not match is unreachable. ``Artifact_description_ft`` already exists from
the baseline migration and is intentionally not re-declared here. Entity types
that are effectively just a name (Technology, Client, Repo, Service,
Integration) get no full-text index; they are reached by exact name via
``list_entities`` / ``get_node``.

Idempotent: ``IF NOT EXISTS`` guards the DDL so replays are safe.
"""

from typing import Any

from contextd.storage.migration import Migration

_DDL = [
    "CREATE FULLTEXT INDEX Ticket_title_ft IF NOT EXISTS FOR (t:Ticket) ON EACH [t.title]",
    "CREATE FULLTEXT INDEX Pattern_description_ft IF NOT EXISTS "
    "FOR (p:Pattern) ON EACH [p.description]",
    "CREATE FULLTEXT INDEX Risk_description_ft IF NOT EXISTS FOR (r:Risk) ON EACH [r.description]",
]


def up(store: Any, version: int) -> None:
    for stmt in _DDL:
        store.exec_write(stmt, None)


migration = Migration(id=6, name="entity_content_indexes_neo4j", up=up)
