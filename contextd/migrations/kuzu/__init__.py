"""Ordered list of Kuzu migrations."""

from contextd.migrations.kuzu._0001_baseline import migration as _0001
from contextd.storage.migration import Migration

ALL_MIGRATIONS: list[Migration] = [_0001]
