"""Shared test fixtures for every stage.

Holds two independent families: hand-built Documents for the extract-stage tests, and a
complete synthetic on-disk project for the confidence and eval CLIs. Both exist so those
stages can be tested without running the stage upstream of them.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from core import paths
from core.io import write_jsonl
from core.schema import Document, Extraction, FieldPrediction, OCRToken

# Nominal geometry for a synthetic page: each line gets its own 20px band, 10px apart.
_LINE_HEIGHT = 20
_LINE_PITCH = 30
_CHAR_WIDTH = 8

FIELD_NAMES = ("company", "date", "address", "total")


@pytest.fixture
def make_document():
    """Return a factory that lays text out as OCR tokens, one y-band per line."""

    def _make(lines, *, doc_id="sroie-001", confidence=0.95, gold=None) -> Document:
        """Build a Document from lines of text; a line may be (text, confidence)."""
        tokens = []
        for row, line in enumerate(lines):
            text, conf = line if isinstance(line, tuple) else (line, confidence)
            y0 = 10 + row * _LINE_PITCH
            x = 10
            for word in text.split():
                width = len(word) * _CHAR_WIDTH
                tokens.append(
                    OCRToken(text=word, bbox=(x, y0, x + width, y0 + _LINE_HEIGHT),
                             confidence=conf)
                )
                x += width + _CHAR_WIDTH
        return Document(
            doc_id=doc_id,
            dataset="sroie",
            split="train",
            image_path=f"data/raw/sroie/{doc_id}.jpg",
            tokens=tuple(tokens),
            gold=dict(gold or {}),
        )

    return _make


@pytest.fixture
def no_network(monkeypatch):
    """Make any attempt to open a socket fail loudly, so a test can prove it stayed offline."""

    def _blocked(*args, **kwargs):
        """Stand in for the socket constructors and fail the test on any use."""
        raise AssertionError("network access attempted during an offline test")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@dataclass(frozen=True)
class SyntheticProject:
    """Handle to a temporary project tree holding documents and extractions on disk."""

    dataset: str
    method: str
    root: Path


def _build_split(dataset: str, split: str, n_docs: int, seed: int) -> tuple[list, list]:
    """Make documents and matching extractions whose signals really do predict correctness."""
    rng = np.random.default_rng(seed)
    documents, extractions = [], []
    for d in range(n_docs):
        doc_id = f"{dataset}-{split}-{d:03d}"
        gold = {name: f"{name}-{d}" for name in FIELD_NAMES}
        documents.append(Document(doc_id=doc_id, dataset=dataset, split=split,
                                  image_path=f"{doc_id}.jpg", tokens=(), gold=gold))
        fields = []
        for name in FIELD_NAMES:
            agreement = float(rng.uniform(0.0, 1.0))
            correct = bool(rng.random() < agreement)
            fields.append(FieldPrediction(
                name=name,
                value=gold[name] if correct else f"wrong-{d}",
                # ocr_conf is deliberately absent so the missing-signal path is exercised.
                signals={"agreement": agreement, "format_valid": 1.0},
            ))
        extractions.append(Extraction(doc_id=doc_id, method="llm", fields=tuple(fields)))
    return documents, extractions


@pytest.fixture()
def synthetic_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticProject:
    """Point every core path at a tmp tree and fill the calibration and test splits."""
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    dataset, method = "sroie", "llm"
    for split, seed in (("calibration", 1), ("test", 2)):
        documents, extractions = _build_split(dataset, split, n_docs=80, seed=seed)
        write_jsonl(paths.documents_path(dataset, split), documents)
        write_jsonl(paths.extraction_path(dataset, method, split), extractions)
    return SyntheticProject(dataset=dataset, method=method, root=tmp_path)
