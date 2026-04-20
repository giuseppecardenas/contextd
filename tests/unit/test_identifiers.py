"""Unit tests for contextd.storage._identifiers."""

from __future__ import annotations

import math

import pytest

from contextd.storage._identifiers import (
    validate_identifier,
    validate_property_keys,
    validate_search_k,
    validate_threshold,
)


class TestValidateIdentifier:
    @pytest.mark.parametrize("value", ["File", "File_summary_ft", "_private", "X1", "a"])
    def test_accepts_identifier_shape(self, value: str) -> None:
        assert validate_identifier(value, kind="label") == value

    @pytest.mark.parametrize(
        "value",
        [
            "",  # empty
            "1File",  # leading digit
            "File-summary",  # hyphen
            "File summary",  # space
            "File;DROP",  # statement-terminator
            "File') RETURN 1 //",  # injection attempt
            "File.summary",  # dotted
            "名前",  # non-ASCII
        ],
    )
    def test_rejects_bad_shapes(self, value: str) -> None:
        with pytest.raises(ValueError, match=r"label must match"):
            validate_identifier(value, kind="label")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValueError, match=r"label must match"):
            validate_identifier(123, kind="label")  # type: ignore[arg-type]

    def test_error_names_the_kind(self) -> None:
        with pytest.raises(ValueError, match=r"edge_type must match"):
            validate_identifier("bad-type", kind="edge_type")


class TestValidatePropertyKeys:
    def test_accepts_valid_keys(self) -> None:
        validate_property_keys({"path": "a.md", "hash": "h1"}, context="test")

    def test_rejects_key_with_cypher_break(self) -> None:
        with pytest.raises(ValueError, match=r"property key"):
            validate_property_keys(
                {"x'; DROP TABLE File; --": 1},
                context="upsert_node(label='File')",
            )

    def test_rejects_non_string_key(self) -> None:
        with pytest.raises(ValueError, match=r"property key"):
            validate_property_keys({1: "value"}, context="test")  # type: ignore[dict-item]

    def test_echoes_context_in_error(self) -> None:
        with pytest.raises(ValueError, match=r"upsert_edge\(edge_type='REFERENCES'\)"):
            validate_property_keys(
                {"bad-key": 1},
                context="upsert_edge(edge_type='REFERENCES')",
            )


class TestValidateSearchK:
    @pytest.mark.parametrize("k", [1, 10, 1000])
    def test_accepts_positive_int(self, k: int) -> None:
        assert validate_search_k(k) == k

    def test_rejects_bool(self) -> None:
        # bool is a subclass of int; isinstance(True, int) is True. The check
        # must specifically reject bool or we silently coerce True→1.
        with pytest.raises(ValueError, match=r"non-bool int"):
            validate_search_k(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"non-bool int"):
            validate_search_k(False)  # type: ignore[arg-type]

    def test_rejects_float(self) -> None:
        with pytest.raises(ValueError, match=r"non-bool int"):
            validate_search_k(9.5)  # type: ignore[arg-type]

    def test_rejects_string(self) -> None:
        with pytest.raises(ValueError, match=r"non-bool int"):
            validate_search_k("10")  # type: ignore[arg-type]

    @pytest.mark.parametrize("k", [0, -1, -100])
    def test_rejects_non_positive(self, k: int) -> None:
        with pytest.raises(ValueError, match=r">= 1"):
            validate_search_k(k)


class TestValidateThreshold:
    def test_accepts_none(self) -> None:
        assert validate_threshold(None) is None

    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
    def test_accepts_in_range(self, t: float) -> None:
        assert validate_threshold(t) == t

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"finite"):
            validate_threshold(bad)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
    def test_rejects_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_threshold(bad)

    def test_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match=r"finite float"):
            validate_threshold(True)  # type: ignore[arg-type]

    def test_rejects_string(self) -> None:
        with pytest.raises(ValueError, match=r"finite float"):
            validate_threshold("0.5")  # type: ignore[arg-type]
