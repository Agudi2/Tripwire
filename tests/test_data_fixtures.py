"""Synthetic fixtures must be deterministic, schema-correct, and carry real OCR noise."""

from pathlib import Path

import pytest

from core.io import load_documents, write_jsonl
from core.schema import Document, OCRToken
from data.fixtures import generate_documents

DATASETS = ("sroie", "cord", "funsd")


@pytest.mark.parametrize("dataset", DATASETS)
def test_same_seed_gives_identical_documents(dataset: str) -> None:
    """Regenerating with the same seed reproduces byte-identical documents."""
    assert generate_documents(dataset, count=12, seed=42) == generate_documents(
        dataset, count=12, seed=42
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_different_seed_gives_different_documents(dataset: str) -> None:
    """The seed genuinely varies the generated content."""
    assert generate_documents(dataset, count=12, seed=42) != generate_documents(
        dataset, count=12, seed=7
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_documents_survive_a_core_io_roundtrip(dataset: str, tmp_path: Path) -> None:
    """Fixtures written with core.io read back exactly, tokens and gold included."""
    docs = generate_documents(dataset, count=8, seed=42)
    path = tmp_path / f"{dataset}.jsonl"
    assert write_jsonl(path, docs) == len(docs)
    assert load_documents(path) == docs


@pytest.mark.parametrize("dataset", DATASETS)
def test_documents_are_well_formed(dataset: str) -> None:
    """Every fixture is a Document with a unique id, tokens, and non-empty gold."""
    docs = generate_documents(dataset, count=20, seed=42)
    assert len({d.doc_id for d in docs}) == len(docs)
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.dataset == dataset
        assert doc.tokens and all(isinstance(t, OCRToken) for t in doc.tokens)
        assert doc.gold and all(v for v in doc.gold.values())


def test_gold_fields_match_each_dataset() -> None:
    """Each dataset produces the gold fields the README describes."""
    sroie = generate_documents("sroie", count=5, seed=42)
    cord = generate_documents("cord", count=5, seed=42)
    funsd = generate_documents("funsd", count=5, seed=42)

    for doc in sroie:
        assert set(doc.gold) == {"company", "date", "address", "total"}
    for doc in cord:
        assert {"company", "date", "address", "total"} <= set(doc.gold)
        assert any(k.startswith("item_") for k in doc.gold)
    for doc in funsd:
        assert len(doc.gold) >= 2  # form key/value pairs


@pytest.mark.parametrize("dataset", DATASETS)
def test_confidences_are_valid_and_span_a_range(dataset: str) -> None:
    """Confidences are probabilities and are not all pinned at one value."""
    tokens = [t for d in generate_documents(dataset, count=30, seed=42) for t in d.tokens]
    assert all(0.0 <= t.confidence <= 1.0 for t in tokens)
    assert len({round(t.confidence, 2) for t in tokens}) > 5


@pytest.mark.parametrize("dataset", DATASETS)
def test_some_tokens_are_low_confidence_misreads(dataset: str) -> None:
    """The corpus contains OCR errors, and they skew low-confidence.

    Without this the downstream confidence stage has nothing to learn from.
    """
    docs = generate_documents(dataset, count=40, seed=42)
    tokens = [t for d in docs for t in d.tokens]
    gold_words = {w for d in docs for v in d.gold.values() for w in v.split()}
    low = [t for t in tokens if t.confidence < 0.6]

    assert low, "expected some low-confidence tokens"
    assert any(t.text not in gold_words for t in low)
    mean_low = sum(t.confidence for t in low) / len(low)
    mean_all = sum(t.confidence for t in tokens) / len(tokens)
    assert mean_low < mean_all


@pytest.mark.parametrize("dataset", DATASETS)
def test_bboxes_are_ordered_integer_pixels(dataset: str) -> None:
    """Every box is (x0, y0, x1, y1) with x0 < x1 and y0 < y1."""
    for doc in generate_documents(dataset, count=10, seed=42):
        for token in doc.tokens:
            x0, y0, x1, y1 = token.bbox
            assert all(isinstance(v, int) for v in token.bbox)
            assert x0 < x1 and y0 < y1


def test_unknown_dataset_is_rejected() -> None:
    """An unsupported dataset name fails loudly rather than silently returning nothing."""
    with pytest.raises(ValueError, match="mnist"):
        generate_documents("mnist", count=3, seed=42)
