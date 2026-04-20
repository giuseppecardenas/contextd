"""Smoke test that proves the test runner and package import work."""

import contextd


def test_package_has_version() -> None:
    assert contextd.__version__ == "0.1.0"


def test_math_still_works() -> None:
    assert 1 + 1 == 2
