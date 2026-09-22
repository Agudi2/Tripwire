"""The OCR+rules baseline must find the obvious fields and report the signals behind them."""

from extract import rules
from extract.validators import FIELDS

RECEIPT = (
    "SUPER MART SDN BHD",
    "NO 12, JALAN BESAR, 43000 KAJANG",
    "DATE: 18/03/2018",
    "COFFEE 4.50",
    "TOTAL 18.50",
)


def _values(extraction) -> dict:
    """Collapse an Extraction into a plain name -> value mapping for assertions."""
    return {field.name: field.value for field in extraction.fields}


def test_extracts_the_four_sroie_fields(make_document) -> None:
    """A plain receipt yields company, date, address, and total."""
    extraction = rules.extract(make_document(RECEIPT))
    assert extraction.doc_id == "sroie-001"
    assert extraction.method == "rules"
    values = _values(extraction)
    assert values["company"] == "SUPER MART SDN BHD"
    assert values["address"] == "NO 12, JALAN BESAR, 43000 KAJANG"
    assert values["date"] == "18/03/2018"
    assert values["total"] == "18.50"


def test_every_field_is_emitted_once_with_the_expected_signals(make_document) -> None:
    """Each of the fixed fields appears exactly once and carries the three baseline signals."""
    extraction = rules.extract(make_document(RECEIPT))
    assert tuple(f.name for f in extraction.fields) == FIELDS
    for field in extraction.fields:
        assert set(field.signals) >= {"ocr_conf", "format_valid", "match_strength"}
        for value in field.signals.values():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0


def test_format_valid_signal_tracks_the_validators(make_document) -> None:
    """Well-formed values get format_valid 1.0 from the shared validators."""
    extraction = rules.extract(make_document(RECEIPT))
    signals = {f.name: f.signals for f in extraction.fields}
    assert signals["date"]["format_valid"] == 1.0
    assert signals["total"]["format_valid"] == 1.0


def test_ocr_conf_reflects_only_the_tokens_actually_used(make_document) -> None:
    """A badly-read total line drags down that field's ocr_conf and nothing else."""
    doc = make_document((
        ("SUPER MART SDN BHD", 0.99),
        ("NO 12, JALAN BESAR, 43000 KAJANG", 0.99),
        ("DATE: 18/03/2018", 0.99),
        ("TOTAL 18.50", 0.40),
    ))
    signals = {f.name: f.signals for f in rules.extract(doc).fields}
    assert signals["total"]["ocr_conf"] == 0.40
    assert signals["company"]["ocr_conf"] == 0.99


def test_missing_field_is_reported_as_empty_with_no_match_strength(make_document) -> None:
    """A receipt with no total still emits the field, flagged as unfound for review."""
    doc = make_document(("SUPER MART SDN BHD", "NO 12, JALAN BESAR, 43000 KAJANG"))
    signals = {f.name: f.signals for f in rules.extract(doc).fields}
    values = _values(rules.extract(doc))
    assert values["total"] == ""
    assert signals["total"]["match_strength"] == 0.0
    assert signals["total"]["format_valid"] == 0.0


def test_total_keyword_beats_an_unanchored_amount(make_document) -> None:
    """The amount on the TOTAL line wins over other currency-shaped tokens on the page."""
    doc = make_document((
        "SUPER MART SDN BHD",
        "SUBTOTAL 99.99",
        "TOTAL 18.50",
    ))
    values = _values(rules.extract(doc))
    assert values["total"] == "18.50"


def test_line_texts_groups_tokens_back_into_reading_order(make_document) -> None:
    """The shared line grouper reconstructs the page one line at a time, top to bottom."""
    assert rules.line_texts(make_document(RECEIPT).tokens) == RECEIPT
