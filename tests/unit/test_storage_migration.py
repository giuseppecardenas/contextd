from pathlib import Path
from unittest.mock import MagicMock

from contextd.storage.migration import Migration, MigrationRunner


def test_runner_applies_pending_migrations(tmp_path: Path) -> None:
    applied: list[str] = []

    def mig_up(store: object, version: int) -> None:
        applied.append(f"up-{version}")

    migrations = [
        Migration(id=1, name="baseline", up=mig_up),
        Migration(id=2, name="second", up=mig_up),
    ]
    mock_store = MagicMock()
    mock_store.exec_read.return_value = [{"applied": [1]}]  # migration 1 already applied

    runner = MigrationRunner(mock_store, migrations)
    runner.apply()

    assert applied == ["up-2"]  # only migration 2 was pending
