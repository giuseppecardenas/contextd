import pytest

from contextd.mcp.readonly_guard import ReadOnlyGuardError, assert_read_only


def test_allows_match_return() -> None:
    assert_read_only("MATCH (n:File) RETURN n LIMIT 10")


def test_rejects_create() -> None:
    with pytest.raises(ReadOnlyGuardError, match="CREATE"):
        assert_read_only("CREATE (n:File) RETURN n")


def test_rejects_merge() -> None:
    with pytest.raises(ReadOnlyGuardError, match="MERGE"):
        assert_read_only("MERGE (n:File {path: 'x'}) RETURN n")


def test_rejects_delete() -> None:
    with pytest.raises(ReadOnlyGuardError, match="DELETE"):
        assert_read_only("MATCH (n:File) DELETE n")


def test_rejects_set() -> None:
    with pytest.raises(ReadOnlyGuardError, match="SET"):
        assert_read_only("MATCH (n:File) SET n.x = 1")


def test_rejects_remove() -> None:
    with pytest.raises(ReadOnlyGuardError, match="REMOVE"):
        assert_read_only("MATCH (n:File) REMOVE n.x")


def test_rejects_detach_delete() -> None:
    with pytest.raises(ReadOnlyGuardError, match="DETACH"):
        assert_read_only("MATCH (n:File) DETACH DELETE n")


def test_rejects_drop() -> None:
    with pytest.raises(ReadOnlyGuardError, match="DROP"):
        assert_read_only("DROP INDEX File_summary_ft")


def test_rejects_foreach() -> None:
    with pytest.raises(ReadOnlyGuardError, match="FOREACH"):
        assert_read_only("MATCH (n:File) FOREACH (x IN n.key_points | SET n.last_touched = x)")


def test_allows_dotted_property_named_set() -> None:
    """Negative-lookbehind must prevent n.set / n.create / n.remove from
    false-positively matching the forbidden keyword list."""
    assert_read_only("MATCH (n:File) RETURN n.set AS s, n.create AS c LIMIT 5")


def test_allows_property_named_delete() -> None:
    assert_read_only("MATCH (n:File) WHERE n.delete IS NULL RETURN n.remove AS r LIMIT 1")
