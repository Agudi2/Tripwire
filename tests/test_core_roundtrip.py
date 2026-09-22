"""Stage 0 contract tests: every shared type must survive a JSONL round-trip unchanged."""

from pathlib import Path

from core.io import load_documents, load_extractions, load_scored_fields, write_jsonl
from core.schema import Document, Extraction, FieldPrediction, OCRToken, ScoredField, is_correct


def test_document_roundtrip(tmp_path: Path) -> None:
    """A Document with nested OCR tokens reads back exactly as written."""
    doc = Document(
        doc_id="sroie-001",
        dataset="sroie",
        split="train",
        image_path="data/raw/sroie/001.jpg",
        tokens=(OCRToken(text="TOTAL", bbox=(10, 20, 60, 35), confidence=0.94),),
        gold={"total": "18.50", "company": "ACME SDN BHD"},
    )
    path = tmp_path / "docs.jsonl"
    assert write_jsonl(path, [doc]) == 1
    assert load_documents(path) == (doc,)


def test_extraction_roundtrip(tmp_path: Path) -> None:
    """An Extraction keeps its per-field confidence signals across a round-trip."""
    extraction = Extraction(
        doc_id="sroie-001",
        method="llm",
        fields=(FieldPrediction(name="total", value="18.50",
                                signals={"agreement": 1.0, "ocr_conf": 0.94}),),
    )
    path = tmp_path / "ex.jsonl"
    write_jsonl(path, [extraction])
    assert load_extractions(path) == (extraction,)


def test_scored_field_roundtrip(tmp_path: Path) -> None:
    """A ScoredField preserves its calibrated confidence and accept decision."""
    scored = ScoredField(doc_id="sroie-001", name="total", value="18.50",
                         confidence=0.97, accepted=True, correct=True)
    path = tmp_path / "scored.jsonl"
    write_jsonl(path, [scored])
    assert load_scored_fields(path) == (scored,)


def test_is_correct_ignores_case_and_spacing() -> None:
    """Normalised match treats casing and extra whitespace as equivalent, not as errors."""
    assert is_correct("ACME  Sdn Bhd", "acme sdn bhd")
    assert not is_correct("18.50", "18.15")
