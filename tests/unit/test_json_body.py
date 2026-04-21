import pytest

from contextd.inference._json_body import extract_json_body


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
