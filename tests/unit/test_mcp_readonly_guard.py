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
