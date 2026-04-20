"""Ordered list of Memgraph migrations."""

from contextd.migrations.memgraph._0001_baseline import migration as _0001
from contextd.storage.migration import Migration

ALL_MIGRATIONS: list[Migration] = [_0001]
