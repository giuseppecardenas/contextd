from __future__ import annotations

from contextd.chunking.sentences import sentence_spans


def _sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


def test_basic_split() -> None:
    assert _sentences("One. Two! Three? Four.") == ["One.", "Two!", "Three?", "Four."]


def test_abbreviations_and_decimals_do_not_split() -> None:
    text = "Dr. Smith ran 1.5x faster, e.g. on Mondays. Then he stopped."
    assert _sentences(text) == ["Dr. Smith ran 1.5x faster, e.g. on Mondays.", "Then he stopped."]


def test_initials_do_not_split() -> None:
    assert _sentences("J. Smith arrived. Good.") == ["J. Smith arrived.", "Good."]


def test_blank_line_is_always_a_break() -> None:
    assert _sentences("no punctuation here\n\nsecond para") == [
        "no punctuation here",
        "second para",
    ]


def test_closing_quotes_stay_attached() -> None:
    assert _sentences('He said "stop." Then left.') == ['He said "stop."', "Then left."]


def test_cjk_terminators() -> None:
    text = "\u7b2c\u4e00\u53e5\u3002 \u7b2c\u4e8c\u53e5\uff01"
    assert _sentences(text) == text.split(" ")


def test_spans_are_trimmed_and_ordered() -> None:
    text = "  ab.   cd.  "
    spans = sentence_spans(text)
    assert spans == [(2, 5), (8, 11)]


def test_empty() -> None:
    assert sentence_spans("") == []
    assert sentence_spans("   \n ") == []
