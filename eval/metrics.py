"""Field-level quality, calibration, and selective-risk metrics.

Every function takes plain arrays so it can be exercised on synthetic data without any
dataset or extractor being present.
"""

from __future__ import annotations

import numpy as np

# Equal-width binning is the convention throughout: n_bins bins of width 1/n_bins over
# [0, 1], each right-closed -- bin k covers (k/n_bins, (k+1)/n_bins] -- with confidence 0.0
# folded into the first bin. Reports must state this, because ECE is not comparable across
# different binning schemes.
DEFAULT_N_BINS = 10


def precision_recall_f1(correct: np.ndarray, n_gold: int) -> tuple[float, float, float]:
    """Field-level precision, recall and F1 from per-prediction correctness and gold count.

    `correct` holds one boolean per field the extractor actually emitted, so precision is
    over emitted fields and recall is over the gold fields that existed.
    """
    correct = _as_bool(correct)
    n_pred = int(correct.size)
    n_hit = int(correct.sum())
    precision = n_hit / n_pred if n_pred else 0.0
    recall = n_hit / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               n_bins: int = DEFAULT_N_BINS) -> float:
    """Count-weighted mean gap between accuracy and mean confidence inside each bin.

    Binning is equal-width and right-closed over [0, 1] (see DEFAULT_N_BINS).
    """
    confidence, correct = _checked(confidence, correct)
    if confidence.size == 0:
        return 0.0
    bins = _bin_index(confidence, n_bins)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        count = int(mask.sum())
        if count:
            gap = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
            total += count * gap
    return float(total / confidence.size)


def brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error between the confidence and the 0/1 outcome (lower is better)."""
    confidence, correct = _checked(confidence, correct)
    if confidence.size == 0:
        return 0.0
    return float(np.mean((confidence - correct.astype(float)) ** 2))


def risk_coverage_curve(confidence: np.ndarray,
                        correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep the most confident fields first; return (threshold, coverage, risk) per step k.

    Step k accepts the k most confident fields: coverage is k/n and risk is the error rate
    within those k. `threshold` is the confidence of the k-th field, i.e. the cutoff that
    produced that point.
    """
    confidence, correct = _checked(confidence, correct)
    if confidence.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty
    order = np.argsort(-confidence, kind="stable")
    ranked_conf = confidence[order]
    errors = (~correct[order]).astype(float)
    k = np.arange(1, confidence.size + 1, dtype=float)
    return ranked_conf, k / confidence.size, np.cumsum(errors) / k


def realized_error(confidence: np.ndarray, correct: np.ndarray, threshold: float) -> float:
    """Error rate *among accepted* fields (confidence > threshold); NaN if none are accepted.

    This is the number the README's "realized error at target" refers to. It is a
    conditional rate, so it is not the quantity the conformal threshold bounds -- see
    `selective_risk`.
    """
    confidence, correct = _checked(confidence, correct)
    accepted = confidence > threshold
    if not accepted.any():
        return float("nan")
    return float(np.mean(~correct[accepted]))


def selective_risk(confidence: np.ndarray, correct: np.ndarray, threshold: float) -> float:
    """Fraction of *all* fields that are accepted and wrong -- the marginal risk.

    This is exactly the quantity `confidence.conformal.conformal_threshold` bounds by alpha.
    """
    confidence, correct = _checked(confidence, correct)
    if confidence.size == 0:
        return 0.0
    return float(np.mean((confidence > threshold) & ~correct))


def coverage_at(confidence: np.ndarray, threshold: float) -> float:
    """Fraction of fields auto-accepted at a threshold."""
    confidence = np.asarray(confidence, dtype=float)
    if confidence.size == 0:
        return 0.0
    return float(np.mean(confidence > threshold))


def _bin_index(confidence: np.ndarray, n_bins: int) -> np.ndarray:
    """Map confidences to right-closed equal-width bin indices in [0, n_bins - 1]."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # side="left" puts a value sitting exactly on an edge into the lower bin, which is what
    # right-closed bins mean; it also avoids the float error that conf * n_bins would incur.
    return np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)


def _checked(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce the two parallel arrays and reject mismatched lengths or non-finite scores."""
    confidence = np.asarray(confidence, dtype=float)
    correct = _as_bool(correct)
    if confidence.shape != correct.shape:
        raise ValueError(f"confidence {confidence.shape} and correct {correct.shape} differ")
    if confidence.size and not np.isfinite(confidence).all():
        raise ValueError("confidence contains non-finite values")
    return confidence, correct


def _as_bool(values: np.ndarray) -> np.ndarray:
    """Coerce a correctness array to booleans without silently accepting None entries."""
    array = np.asarray(values)
    if array.dtype == object:
        raise ValueError("correctness array must be boolean, not object (unlabelled fields?)")
    return array.astype(bool)
