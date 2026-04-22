"""Shared fixtures for the e2e test suite.

Re-exports the ``backend`` fixture from ``tests/integration/conftest.py``
so that the e2e suite can share the same parametrized Memgraph + Neo4j
testcontainer setup without duplicating it.
"""

from __future__ import annotations

# Re-export the parametrized backend fixture so pytest discovers it here.
from tests.integration.conftest import backend as backend

__all__ = ["backend"]
