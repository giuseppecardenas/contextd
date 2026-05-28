"""Ordered list of Neo4j migrations."""

from contextd.migrations.neo4j._0001_baseline import migration as _0001
from contextd.migrations.neo4j._0002_corpus_indexes import migration as _0002
from contextd.migrations.neo4j._0003_section_fulltext import migration as _0003
from contextd.migrations.neo4j._0004_backfill_inferred_at import migration as _0004
from contextd.migrations.neo4j._0005_purge_phantom_stubs import migration as _0005
from contextd.storage.migration import Migration

ALL_MIGRATIONS: list[Migration] = [_0001, _0002, _0003, _0004, _0005]
