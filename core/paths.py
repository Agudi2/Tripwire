"""Canonical on-disk layout. Every stage resolves its paths through here, never by hand."""

from __future__ import annotations

from pathlib import Path

# Project root, resolved relative to this file so the layout works from any cwd.
ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"  # untouched dataset downloads
PROCESSED = ROOT / "data" / "processed"  # normalised Documents, one JSONL per dataset
SPLITS = ROOT / "data" / "splits"  # train/calibration/test doc_id lists
ARTIFACTS = ROOT / "artifacts"  # extractions, fitted calibrators, thresholds
CACHE = ROOT / "cache"  # cached LLM responses, keyed by prompt hash
REPORTS = ROOT / "reports"  # generated tables and failure analysis


def extraction_path(dataset: str, method: str, split: str) -> Path:
    """Where one extractor's output for one dataset split is stored."""
    return ARTIFACTS / f"extractions.{dataset}.{method}.{split}.jsonl"


def documents_path(dataset: str, split: str) -> Path:
    """Where the normalised documents for one dataset split are stored."""
    return PROCESSED / f"{dataset}.{split}.jsonl"


def ensure_dirs() -> None:
    """Create every project directory that stages write into, if it does not exist yet."""
    for directory in (RAW, PROCESSED, SPLITS, ARTIFACTS, CACHE, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
