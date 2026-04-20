"""Abstract base for storage backends plus capability introspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BackendName = Literal["memgraph", "kuzu"]


@dataclass(frozen=True)
class BackendCapabilities:
    """Static declaration of what a storage backend supports.

    Callers adapt behaviour via these flags rather than trying an
    operation and reacting to failures — e.g., the MCP ``query_graph``
    tool rejects Cypher that calls procedures the backend lacks, with
    a clear error citing the capability gap.
    """

    name: BackendName
    concurrent_writers: int
    """-1 means unlimited; integers >= 1 are the maximum concurrent writer count."""
    supports_vector_index: bool
    supports_full_text_index: bool
    supports_graph_algorithms: bool
    requires_docker: bool
    default_connection: str

    @property
    def unlimited_writers(self) -> bool:
        return self.concurrent_writers == -1
