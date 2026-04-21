"""Ordered list of Kuzu migrations."""

from contextd.migrations.kuzu._0001_baseline import migration as _0001
from contextd.migrations.kuzu._0002_corpus_stats import migration as _0002
from contextd.storage.migration import Migration

ALL_MIGRATIONS: list[Migration] = [_0001, _0002]
