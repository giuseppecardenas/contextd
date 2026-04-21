"""Ordered list of Neo4j migrations."""

from contextd.migrations.neo4j._0001_baseline import migration as _0001
from contextd.migrations.neo4j._0002_corpus_indexes import migration as _0002
from contextd.storage.migration import Migration

ALL_MIGRATIONS: list[Migration] = [_0001, _0002]
