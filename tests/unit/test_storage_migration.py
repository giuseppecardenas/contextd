from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


def test_runner_skips_when_all_applied() -> None:
    """Re-applying an up-to-date migration list is a no-op: up is never called
    and no new _record_applied write is issued."""
    calls: list[int] = []

    def mig_up(store: object, version: int) -> None:
        calls.append(version)

    migrations = [Migration(id=1, name="baseline", up=mig_up)]
    mock_store = MagicMock()
    mock_store.exec_read.return_value = [{"applied": [1]}]

    MigrationRunner(mock_store, migrations).apply()

    assert calls == []
    mock_store.exec_write.assert_not_called()


def test_runner_halts_on_failure_and_does_not_record() -> None:
    """When a migration's up() raises, the runner propagates the error and
    the failing migration's id is NOT recorded — so a subsequent re-run
    retries that migration (relying on idempotent DDL)."""
    applied: list[int] = []

    def mig1_up(store: object, version: int) -> None:
        applied.append(version)

    def mig2_up(store: object, version: int) -> None:
        raise RuntimeError("DDL failed mid-apply")

    def mig3_up(store: object, version: int) -> None:
        applied.append(version)  # pragma: no cover — must not run

    migrations = [
        Migration(id=1, name="one", up=mig1_up),
        Migration(id=2, name="two", up=mig2_up),
        Migration(id=3, name="three", up=mig3_up),
    ]
    mock_store = MagicMock()
    mock_store.exec_read.return_value = []  # nothing applied yet

    with pytest.raises(RuntimeError, match="DDL failed"):
        MigrationRunner(mock_store, migrations).apply()

    # mig1 ran and was recorded; mig2 ran and raised before recording; mig3 never ran.
    assert applied == [1]
    # _record_applied inlines the id as a Cypher literal; recorded Cyphers end in `+ [<id>]`
    recorded_cyphers = [call.args[0] for call in mock_store.exec_write.call_args_list]
    assert len(recorded_cyphers) == 1
    assert "+ [1]" in recorded_cyphers[0]


def test_runner_sorts_by_id_regardless_of_input_order() -> None:
    """apply() must apply in ascending id order even if the caller passed them
    out of order — otherwise a higher-id migration could run against a schema
    that its predecessor has not yet set up."""
    order: list[int] = []

    def mig_up(store: object, version: int) -> None:
        order.append(version)

    migrations = [
        Migration(id=3, name="three", up=mig_up),
        Migration(id=1, name="one", up=mig_up),
        Migration(id=2, name="two", up=mig_up),
    ]
    mock_store = MagicMock()
    mock_store.exec_read.return_value = []

    MigrationRunner(mock_store, migrations).apply()

    assert order == [1, 2, 3]
