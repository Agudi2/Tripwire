"""Each failure cause must be assigned for the right reason, not by accident of ordering."""

from core.schema import Document, Extraction, FieldPrediction, OCRToken
from eval.errors import analyse, classify, document_text, edit_distance, strip_punctuation

GOLD = {
    "company": "SUPER MART SDN BHD",
    "address": "NO 12, JALAN BESAR, KAJANG",
    "date": "18/03/2018",
    "total": "18.50",
}

PAGE = "supermartsdnbhdno12jalanbesarkajangdate18032018total1850servicecharge12500"


def test_empty_prediction_is_missing() -> None:
    """A blank value means the rule found nothing, which is not the same as finding it wrong."""
    assert classify("date", "", GOLD, PAGE) == "missing"
    assert classify("date", "   ", GOLD, PAGE) == "missing"


def test_same_characters_different_punctuation_is_formatting() -> None:
    """Right value, wrong punctuation: the scorer is strict, the extractor is not at fault."""
    assert classify("address", "NO 12 JALAN BESAR KAJANG", GOLD, PAGE) == "formatting"


def test_gold_plus_trailing_text_is_over_capture() -> None:
    """The address rule running past its span is a distinct defect from misreading it."""
    predicted = "NO 12, JALAN BESAR, KAJANG DATE: 18/03/2018"
    assert classify("address", predicted, GOLD, PAGE) == "over_capture"


def test_another_fields_value_is_wrong_field() -> None:
    """Returning the company as the address means anchoring picked the wrong line."""
    assert classify("address", "SUPER MART SDN BHD", GOLD, PAGE) == "wrong_field"


def test_near_miss_string_is_ocr_misread() -> None:
    """A couple of substituted characters is consistent with OCR, not with a wrong line."""
    assert classify("company", "SUPER MART S0N BH0", GOLD, PAGE) == "ocr_misread"


def test_value_absent_from_the_page_is_hallucination() -> None:
    """A value printed nowhere on the document was not read off it."""
    assert classify("company", "TOTALLY DIFFERENT COMPANY LTD", GOLD, PAGE) == "hallucination"


def test_value_present_on_page_but_wrong_is_wrong_value() -> None:
    """Picking a different printed amount is an anchoring failure, not a hallucination."""
    # 125.00 is on the page but nowhere near 18.50, so no misread explains it. A value only a
    # digit away, like 17.50, is deliberately NOT used here: one substitution is exactly what
    # an OCR confusion looks like, and the classifier rightly calls that ocr_misread.
    assert classify("total", "125.00", GOLD, PAGE) == "wrong_value"


def test_single_digit_difference_prefers_misread_over_wrong_value() -> None:
    """Pins the boundary: one substituted digit reads as OCR noise, not as a wrong line."""
    assert classify("total", "18.60", GOLD, PAGE) == "ocr_misread"


def test_edit_distance_matches_known_values() -> None:
    """Levenshtein on worked examples, including the identical-string short circuit."""
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("", "abc") == 3


def test_strip_punctuation_keeps_only_alphanumerics() -> None:
    """Normalisation used by the formatting and containment checks."""
    assert strip_punctuation("NO 12, Jalan-Besar!") == "no12jalanbesar"


def test_analyse_counts_only_wrong_fields() -> None:
    """Correct fields never enter the tally, and every wrong one gets exactly one cause."""
    tokens = (OCRToken(text="SUPER", bbox=(0, 0, 10, 10), confidence=0.9),)
    doc = Document(doc_id="d1", dataset="sroie", split="test", image_path="d1.jpg",
                   tokens=tokens, gold=GOLD)
    extraction = Extraction(doc_id="d1", method="rules", fields=(
        FieldPrediction(name="company", value="SUPER MART SDN BHD"),   # correct
        FieldPrediction(name="date", value=""),                        # missing
        FieldPrediction(name="total", value="99.99"),                  # not on page
    ))
    result = analyse([doc], [extraction])
    assert result["n_fields"] == 3
    assert result["n_errors"] == 2
    assert result["overall"]["missing"] == 1
    assert result["overall"]["hallucination"] == 1


def test_document_text_concatenates_tokens() -> None:
    """Page text used for containment checks comes from the OCR tokens, normalised."""
    tokens = (OCRToken(text="NO", bbox=(0, 0, 5, 5), confidence=0.9),
              OCRToken(text="12,", bbox=(6, 0, 12, 5), confidence=0.9))
    doc = Document(doc_id="d", dataset="sroie", split="test", image_path="d.jpg", tokens=tokens)
    assert document_text(doc) == "no12"
