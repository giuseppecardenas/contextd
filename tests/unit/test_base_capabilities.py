from contextd.storage.base import BackendCapabilities


def test_capabilities_is_frozen() -> None:
    caps = BackendCapabilities(
        name="memgraph",
        concurrent_writers=-1,  # unlimited sentinel
        supports_vector_index=True,
        supports_full_text_index=True,
        supports_graph_algorithms=True,
        requires_docker=True,
        default_connection="bolt://127.0.0.1:7687",
    )
    # Frozen dataclasses reject attribute assignment.
    import dataclasses

    try:
        caps.name = "kuzu"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("BackendCapabilities must be frozen.")


def test_unlimited_sentinel() -> None:
    caps = BackendCapabilities(
        name="memgraph",
        concurrent_writers=-1,
        supports_vector_index=True,
        supports_full_text_index=True,
        supports_graph_algorithms=True,
        requires_docker=True,
        default_connection="bolt://127.0.0.1:7687",
    )
    assert caps.unlimited_writers is True

    caps_kuzu = BackendCapabilities(
        name="kuzu",
        concurrent_writers=1,
        supports_vector_index=True,
        supports_full_text_index=True,
        supports_graph_algorithms=False,
        requires_docker=False,
        default_connection="~/.contextd/graph/",
    )
    assert caps_kuzu.unlimited_writers is False
