"""Ordered list of Memgraph migrations."""

from contextd.migrations.memgraph._0001_baseline import migration as _0001
from contextd.migrations.memgraph._0002_corpus_stats import migration as _0002
from contextd.migrations.memgraph._0003_section_fulltext import migration as _0003
from contextd.storage.migration import Migration

ALL_MIGRATIONS: list[Migration] = [_0001, _0002, _0003]
