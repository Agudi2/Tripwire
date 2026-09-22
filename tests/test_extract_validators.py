"""Format validators must accept well-formed field values and score malformed ones lower."""

import pytest

from extract.validators import FIELDS, validate


@pytest.mark.parametrize("value", ["18/03/2018", "2018-03-18", "18 Mar 2018", "18.03.2018"])
def test_date_parses(value: str) -> None:
    """A date that a real calendar accepts scores a full 1.0."""
    assert validate("date", value) == 1.0


def test_date_shaped_but_impossible_scores_partial() -> None:
    """A date-shaped string with an impossible month is suspicious, not gibberish."""
    score = validate("date", "18/33/2018")
    assert 0.0 < score < 1.0


@pytest.mark.parametrize("value", ["", "SUPER MART", "n/a"])
def test_date_rejects_non_dates(value: str) -> None:
    """Text with no date shape at all scores zero."""
    assert validate("date", value) == 0.0


@pytest.mark.parametrize("value", ["18.50", "1,234.56", "$18.50", "RM 18.50", "RM18.50"])
def test_total_accepts_currency(value: str) -> None:
    """Currency-shaped totals, with or without a symbol, score a full 1.0."""
    assert validate("total", value) == 1.0


def test_total_without_cents_scores_partial() -> None:
    """A bare integer is plausibly a total but is missing the cents receipts always print."""
    assert 0.0 < validate("total", "18") < 1.0


@pytest.mark.parametrize("value", ["", "eighteen fifty", "18.5.0", "TOTAL"])
def test_total_rejects_non_currency(value: str) -> None:
    """Anything that is not a number scores zero."""
    assert validate("total", value) == 0.0


def test_company_accepts_names_and_rejects_empty_or_symbolic() -> None:
    """A company needs actual letters; blank or punctuation-only values score zero."""
    assert validate("company", "SUPER MART SDN BHD") == 1.0
    assert validate("company", "") == 0.0
    assert validate("company", "***") == 0.0


def test_address_needs_more_than_a_word() -> None:
    """A full street address scores 1.0; a single bare word is only partly credible."""
    assert validate("address", "NO 12, JALAN BESAR, 43000 KAJANG") == 1.0
    assert 0.0 < validate("address", "KAJANG") < 1.0
    assert validate("address", "  ") == 0.0


def test_every_field_validates_to_a_usable_float() -> None:
    """Validators double as a confidence signal, so each returns a float in [0, 1]."""
    for name in FIELDS:
        score = validate(name, "18/03/2018")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_unknown_field_falls_back_to_non_empty_check() -> None:
    """A field with no dedicated validator still yields a usable signal."""
    assert validate("line_items", "2 x COFFEE") == 1.0
    assert validate("line_items", "") == 0.0
