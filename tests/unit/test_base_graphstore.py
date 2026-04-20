from contextd.storage.base import GraphStore


def test_graphstore_is_abstract() -> None:
    # Calling abstract methods on a concrete subclass that does not override
    # them should fail with TypeError at instantiation.
    try:
        GraphStore()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("GraphStore must be abstract.")


def test_graphstore_required_abstracts() -> None:
    required = {
        "connect",
        "close",
        "apply_migrations",
        "upsert_node",
        "upsert_edge",
        "delete_edges",
        "exec_read",
        "exec_write",
        "vector_search",
        "full_text_search",
        "capabilities",
    }
    abstract_names = getattr(GraphStore, "__abstractmethods__", frozenset())
    missing = required - abstract_names
    assert not missing, f"GraphStore missing required abstract members: {missing}"
