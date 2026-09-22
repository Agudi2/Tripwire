"""Fit the calibrator and the conformal threshold on the calibration split.

    python -m confidence.fit --split calibration

The calibration split is cut in half BY DOCUMENT: one half fits the calibrator, the other
supplies the conformal scores. The conformal guarantee assumes its scores are exchangeable
with fresh data, which fails if the calibrator was fit on the same fields; splitting by
document rather than by field also keeps correlated fields from straddling the two halves.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from confidence.calibrate import (
    Calibrator,
    calibrator_path,
    fit_isotonic,
    fit_platt,
    save_calibrator,
)
from confidence.conformal import conformal_threshold, save_threshold, threshold_path
from confidence.features import build_matrix, feature_names, unknown_signals
from core import paths
from core.io import load_documents, load_extractions
from core.schema import ScoredField, is_correct
from eval.bootstrap import group_indices


@dataclass(frozen=True)
class LabelledFields:
    """Every extracted field that has a gold value, flattened and ready to fit on."""

    doc_ids: tuple[str, ...]
    names: tuple[str, ...]
    values: tuple[str, ...]
    features: np.ndarray
    correct: np.ndarray
    gold_by_doc: Mapping[str, int]  # gold fields per document, emitted or not
    unknown: tuple[str, ...]  # signal names seen in the data but outside the fixed schema

    @property
    def n_gold(self) -> int:
        """Total gold fields across all documents -- the recall denominator."""
        return sum(self.gold_by_doc.values())


def load_labelled(dataset: str, method: str, split: str) -> LabelledFields:
    """Join one extractor's output against gold and build the feature matrix.

    Fields with no gold entry are dropped: they cannot be labelled, and guessing would put
    fabricated labels into the calibration fit.
    """
    documents = load_documents(_require(paths.documents_path(dataset, split), "documents"))
    extractions = load_extractions(
        _require(paths.extraction_path(dataset, method, split), "extractions"))
    gold_by_doc = {doc.doc_id: doc.gold for doc in documents}
    doc_ids, names, values, fields, correct = [], [], [], [], []
    for extraction in extractions:
        gold = gold_by_doc.get(extraction.doc_id)
        if gold is None:
            raise ValueError(f"extraction for unknown document {extraction.doc_id!r}")
        for field in extraction.fields:
            if field.name not in gold:
                continue
            doc_ids.append(extraction.doc_id)
            names.append(field.name)
            values.append(field.value)
            fields.append(field)
            correct.append(is_correct(field.value, gold[field.name]))
    return LabelledFields(
        doc_ids=tuple(doc_ids), names=tuple(names), values=tuple(values),
        features=build_matrix(fields), correct=np.asarray(correct, dtype=bool),
        gold_by_doc={doc.doc_id: len(doc.gold) for doc in documents},
        unknown=unknown_signals(fields))


def split_by_document(doc_ids: Sequence[str], frac: float,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Partition field indices into two halves that share no document.

    `frac` is the share of documents used for the first half (the calibrator fit).
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be strictly between 0 and 1")
    blocks = list(group_indices(doc_ids).values())
    order = np.random.default_rng(seed).permutation(len(blocks))
    cut = max(1, min(len(blocks) - 1, int(round(frac * len(blocks)))))
    first = np.concatenate([blocks[i] for i in order[:cut]])
    second = np.concatenate([blocks[i] for i in order[cut:]])
    return np.sort(first), np.sort(second)


def score_fields(labelled: LabelledFields, calibrator: Calibrator,
                 threshold: float) -> tuple[ScoredField, ...]:
    """Apply a fitted calibrator and threshold to produce the ScoredField contract."""
    confidence = calibrator.predict_proba(labelled.features)
    return tuple(
        ScoredField(doc_id=doc_id, name=name, value=value, confidence=float(conf),
                    accepted=bool(conf > threshold), correct=bool(correct))
        for doc_id, name, value, conf, correct in zip(
            labelled.doc_ids, labelled.names, labelled.values, confidence, labelled.correct))


def scored_path(dataset: str, method: str, split: str) -> Path:
    """Where the ScoredField output for one dataset/extractor/split is written."""
    return paths.ARTIFACTS / f"scored.{dataset}.{method}.{split}.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _parse_args(argv)
    try:
        labelled = load_labelled(args.dataset, args.method, args.split)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    if labelled.unknown:
        print(f"note: ignoring signals outside the fixed schema: {', '.join(labelled.unknown)}")

    fit_idx, conformal_idx = split_by_document(labelled.doc_ids, args.calibrator_frac, args.seed)
    fitter = fit_platt if args.calibrator == "platt" else fit_isotonic
    try:
        calibrator = fitter(labelled.features[fit_idx], labelled.correct[fit_idx],
                            feature_names())
    except ValueError as error:
        print(f"could not fit calibrator: {error}", file=sys.stderr)
        return 1

    held_out = calibrator.predict_proba(labelled.features[conformal_idx])
    threshold = conformal_threshold(held_out, labelled.correct[conformal_idx], args.alpha)
    cal_path = save_calibrator(calibrator,
                               calibrator_path(args.dataset, args.method, args.split))
    thr_path = save_threshold(threshold, args.alpha, int(conformal_idx.size),
                              threshold_path(args.dataset, args.method, args.split))
    print(f"fitted {args.calibrator} on {fit_idx.size} fields -> {cal_path}")
    print(f"threshold {threshold:.4f} from {conformal_idx.size} held-out fields "
          f"at alpha={args.alpha} -> {thr_path}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define the CLI surface documented in the README."""
    parser = argparse.ArgumentParser(description="Fit confidence calibration and threshold.")
    parser.add_argument("--split", default="calibration", help="split to fit on")
    parser.add_argument("--dataset", default="sroie")
    parser.add_argument("--method", default="llm", help="extractor whose output to calibrate")
    parser.add_argument("--alpha", type=float, default=0.02, help="target error rate")
    parser.add_argument("--calibrator", choices=("platt", "isotonic"), default="isotonic")
    parser.add_argument("--calibrator-frac", type=float, default=0.5,
                        help="share of documents used to fit the calibrator, rest for conformal")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _require(path: Path, what: str) -> Path:
    """Fail with a readable message instead of a traceback when a stage input is absent."""
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}; run the earlier stage first")
    return path


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
