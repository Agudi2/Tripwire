"""Shared data contracts passed between the data, extract, confidence, and eval stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

# The three fixed splits. Test is reserved for final numbers only.
Split = Literal["train", "calibration", "test"]

# Datasets named in the project README.
Dataset = Literal["sroie", "cord", "funsd"]


@dataclass(frozen=True)
class OCRToken:
    """A single word recognised by OCR, with its box and the engine's own confidence."""

    text: str
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixels
    confidence: float  # 0.0-1.0 as reported by the OCR engine


@dataclass(frozen=True)
class Document:
    """One scanned page: its OCR tokens plus the gold field values used for scoring."""

    doc_id: str
    dataset: str
    split: str
    image_path: str
    tokens: tuple[OCRToken, ...] = ()
    gold: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldPrediction:
    """One extracted field, carrying the raw signals that confidence scoring will consume.

    `signals` stays deliberately open-ended: each extractor contributes whatever it can
    measure (sample agreement, OCR confidence, format validity, token log-probs), and the
    confidence stage decides which of those to fit on.
    """

    name: str
    value: str
    signals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Extraction:
    """Everything one extractor produced for one document."""

    doc_id: str
    method: str  # "rules" or "llm"
    fields: tuple[FieldPrediction, ...] = ()


@dataclass(frozen=True)
class ScoredField:
    """A field after calibration, with the accept/review decision attached."""

    doc_id: str
    name: str
    value: str
    confidence: float  # calibrated P(value is correct)
    accepted: bool  # False means routed to human review
    correct: bool | None = None  # filled in only when gold is available


def is_correct(predicted: str, gold: str) -> bool:
    """Compare a predicted field value to gold using normalised exact match."""
    return _normalize(predicted) == _normalize(gold)


def _normalize(value: str) -> str:
    """Lowercase and collapse whitespace so trivial formatting gaps do not count as errors."""
    return " ".join(value.lower().split())
