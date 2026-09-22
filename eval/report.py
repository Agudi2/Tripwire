"""Evaluate a split with the fitted calibrator and write the markdown results table.

    python -m eval.report --split test

The calibrator and threshold come from the calibration split; this stage only applies them,
so the test split is never used to choose anything.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from confidence.calibrate import Calibrator, calibrator_path, load_calibrator
from confidence.conformal import load_threshold, threshold_path
from confidence.features import FEATURE_COLUMNS
from confidence.fit import LabelledFields, load_labelled, score_fields, scored_path
from core import paths
from core.io import write_jsonl
from eval.bootstrap import bootstrap_ci
from eval.metrics import (
    DEFAULT_N_BINS,
    brier_score,
    coverage_at,
    expected_calibration_error,
    precision_recall_f1,
    realized_error,
    selective_risk,
)

COLUMNS = ("Extractor", "Docs", "Fields", "Precision", "Recall", "F1 (95% CI)",
           "ECE before / after", "Brier", "Threshold", "Coverage",
           "Realized error (95% CI)", "Marginal risk")


def evaluate(labelled: LabelledFields, calibrator: Calibrator, threshold: float,
             n_boot: int, seed: int) -> dict:
    """Compute every headline metric for one extractor, with by-document bootstrap CIs."""
    confidence = calibrator.predict_proba(labelled.features)
    correct = labelled.correct
    precision, recall, f1 = precision_recall_f1(correct, labelled.n_gold)
    # Each field carries its document's share of gold, so a resample's recall denominator is
    # the exact gold count of the documents drawn. Documents from which the extractor emitted
    # nothing have no rows to resample and so sit outside the interval; they are still in the
    # point estimate's denominator.
    emitted = Counter(labelled.doc_ids)
    gold_weight = np.array([labelled.gold_by_doc[d] / emitted[d] for d in labelled.doc_ids])

    def f1_stat(idx: np.ndarray) -> float:
        """F1 on one document resample."""
        return precision_recall_f1(correct[idx], float(gold_weight[idx].sum()))[2]

    def error_stat(idx: np.ndarray) -> float:
        """Realized error among accepted fields on one document resample."""
        return realized_error(confidence[idx], correct[idx], threshold)

    # Computed once here: it scans for whichever raw signal this extractor actually carries.
    raw_ece, raw_signal = _raw_ece(labelled)

    return {
        "n_fields": int(correct.size),
        "n_docs": len(set(labelled.doc_ids)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_ci": bootstrap_ci(labelled.doc_ids, f1_stat, n_boot=n_boot, seed=seed),
        "ece_raw": raw_ece,
        "ece_raw_signal": raw_signal,
        "ece": expected_calibration_error(confidence, correct),
        "brier": brier_score(confidence, correct),
        "threshold": threshold,
        "coverage": coverage_at(confidence, threshold),
        "realized_error": realized_error(confidence, correct, threshold),
        "realized_error_ci": bootstrap_ci(labelled.doc_ids, error_stat, n_boot=n_boot, seed=seed),
        "selective_risk": selective_risk(confidence, correct, threshold),
    }


def build_report(split: str, alpha: float, results: Mapping[str, dict]) -> str:
    """Render the results mapping as a markdown table with its methodology stated."""
    lines = [
        f"# Tripwire results - {split} split",
        "",
        f"Target error rate: alpha = {alpha}. Calibrator and threshold were fit on the "
        "calibration split only.",
        "",
        f"Expected calibration error uses {DEFAULT_N_BINS} equal-width, right-closed bins over "
        "[0, 1]. *Before* is the extractor's best raw signal, named in the cell, because the "
        "two extractors emit different signals; *after* is the calibrated confidence.",
        "",
        "Intervals are 95% percentile bootstrap, resampled by document, because fields within "
        "a document are correlated.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for method, row in results.items():
        lines.append("| " + " | ".join([
            method,
            str(row["n_docs"]),
            str(row["n_fields"]),
            _fmt(row["precision"]),
            _fmt(row["recall"]),
            f"{_fmt(row['f1'])} {_fmt_ci(row['f1_ci'])}",
            f"{_fmt(row['ece_raw'])} ({row.get('ece_raw_signal', '?')}) / {_fmt(row['ece'])}",
            _fmt(row["brier"]),
            _fmt(row["threshold"]),
            _fmt(row["coverage"]),
            f"{_fmt(row['realized_error'])} {_fmt_ci(row['realized_error_ci'])}",
            _fmt(row["selective_risk"]),
        ]) + " |")
    lines += [
        "",
        "**Realized error** is the error rate among accepted fields. **Marginal risk** is the "
        "share of all fields that are accepted and wrong; that is the quantity the conformal "
        f"threshold bounds by alpha = {alpha}, so realized error can exceed alpha whenever "
        "coverage is below 1.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _parse_args(argv)
    results: dict[str, dict] = {}
    for method in args.method:
        try:
            labelled = load_labelled(args.dataset, method, args.split)
            calibrator = load_calibrator(
                _require(calibrator_path(args.dataset, method, args.fit_split)))
            record = load_threshold(_require(threshold_path(args.dataset, method,
                                                            args.fit_split)))
        except FileNotFoundError as error:
            print(f"{error}\nrun `python -m confidence.fit --split {args.fit_split}` first",
                  file=sys.stderr)
            return 1
        threshold = record["threshold"]
        results[method] = evaluate(labelled, calibrator, threshold, args.n_boot, args.seed)
        write_jsonl(scored_path(args.dataset, method, args.split),
                    score_fields(labelled, calibrator, threshold))

    out = args.out or (paths.REPORTS / f"results.{args.split}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(args.split, args.alpha, results), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _raw_ece(labelled: LabelledFields) -> tuple[float, str]:
    """ECE of the uncalibrated baseline signal, plus the name of the signal actually used.

    Different extractors emit different signals -- the rules baseline has no `agreement` and
    the LLM extractor has no `ocr_conf` -- so a fixed column gives `n/a` for whichever method
    lacks it. Instead take the first signal in FEATURE_COLUMNS order that this extractor
    really carries, and return its name so the report states which baseline it measured
    rather than leaving the reader to guess.
    """
    n_signals = len(FEATURE_COLUMNS)
    for column, name in enumerate(FEATURE_COLUMNS):
        present = labelled.features[:, n_signals + column] == 0.0
        if present.any():
            ece = expected_calibration_error(
                labelled.features[present, column], labelled.correct[present]
            )
            return ece, name
    return float("nan"), "none"


def _fmt(value: float) -> str:
    """Three decimals, or 'n/a' for an undefined metric."""
    return "n/a" if value is None or np.isnan(value) else f"{value:.3f}"


def _fmt_ci(bounds: tuple[float, float]) -> str:
    """Render a bootstrap interval as [low, high]."""
    return f"[{_fmt(bounds[0])}, {_fmt(bounds[1])}]"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define the CLI surface documented in the README."""
    parser = argparse.ArgumentParser(description="Evaluate a split and write the results table.")
    parser.add_argument("--split", default="test", help="split to evaluate")
    parser.add_argument("--dataset", default="sroie")
    parser.add_argument("--method", nargs="+", default=["llm"], help="extractors to report")
    parser.add_argument("--fit-split", default="calibration",
                        help="split the calibrator and threshold were fit on")
    parser.add_argument("--alpha", type=float, default=0.02, help="target error rate, for context")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def _require(path: Path) -> Path:
    """Fail with a readable message instead of a traceback when a fitted artifact is absent."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return path


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
