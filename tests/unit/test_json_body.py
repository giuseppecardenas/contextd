import json

import pytest

from contextd.inference._json_body import (
    _strip_trailing_commas,
    extract_json_body,
    loads_json_body,
)


def test_bare_json_round_trips() -> None:
    assert extract_json_body('{"x": 1}') == '{"x": 1}'


def test_strips_json_fence() -> None:
    assert extract_json_body('```json\n{"x": 1}\n```') == '{"x": 1}'


def test_strips_any_language_fence() -> None:
    assert extract_json_body('```yaml\n{"x": 1}\n```') == '{"x": 1}'


def test_strips_surrounding_prose() -> None:
    assert extract_json_body('Here it is: {"x": 1} thanks!') == '{"x": 1}'


def test_no_json_raises() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_body("no json here")


def test_inverted_braces_raises() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_body("} leads before { starts")


def test_loads_plain_object() -> None:
    assert loads_json_body('{"x": 1}') == {"x": 1}


def test_loads_through_fence_and_prose() -> None:
    assert loads_json_body('Here you go:\n```json\n{"x": [1, 2]}\n```') == {"x": [1, 2]}


def test_loads_repairs_trailing_comma_in_array() -> None:
    # The observed deepseek-v4-flash failure: a trailing comma closing the
    # `key_points` array. json.loads rejects it; the summary should survive.
    raw = '{"summary": "s", "key_points": ["a", "b", ]}'
    assert loads_json_body(raw) == {"summary": "s", "key_points": ["a", "b"]}


def test_loads_repairs_trailing_comma_in_object() -> None:
    assert loads_json_body('{"a": 1, "b": 2, }') == {"a": 1, "b": 2}


def test_loads_repairs_nested_and_multiline_trailing_commas() -> None:
    raw = '{\n "a": [1, 2,\n ],\n "b": {"c": 3,\n },\n}'
    assert loads_json_body(raw) == {"a": [1, 2], "b": {"c": 3}}


def test_repair_preserves_commas_inside_strings() -> None:
    # The reason this is a scanner and not a regex: prose values legitimately
    # contain a `, ]` / `, }` sequence, and a blind substitution would rewrite
    # the model's own content.
    # The `k` array forces the repair path; the `summary` value must come
    # through byte-identical despite containing `, ]` and `, }`.
    raw = '{"summary": "lists look like [a, b, ] in prose, }", "k": [1, ], }'
    assert loads_json_body(raw) == {"summary": "lists look like [a, b, ] in prose, }", "k": [1]}


def test_repair_handles_escaped_quotes_before_trailing_comma() -> None:
    # An escaped quote must not read as the end of the string, or the scanner
    # would treat following text as structural and corrupt the value.
    raw = '{"a": ["he said \\"hi, }\\"", ]}'
    assert loads_json_body(raw) == {"a": ['he said "hi, }"']}


def test_valid_json_is_never_rewritten() -> None:
    # Strict parse comes first, so well-formed output never touches the repair.
    raw = '{"summary": "trailing comma text: , }", "n": [1, 2]}'
    assert loads_json_body(raw) == json.loads(raw)


def test_strip_trailing_commas_is_identity_on_clean_json() -> None:
    raw = '{"a": [1, 2], "b": {"c": 3}}'
    assert _strip_trailing_commas(raw) == raw


def test_loads_propagates_unrepairable_json() -> None:
    # A truncated response is not a trailing-comma problem; it must still fail
    # rather than be silently coerced into a partial summary.
    with pytest.raises(json.JSONDecodeError):
        loads_json_body('{"summary": "ok", "key_points": ["a"}')
