"""Split-conformal acceptance threshold for a target error rate alpha.

Following Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction": with n
exchangeable calibration scores s_1..s_n and a fresh score s_{n+1}, taking

    q = the k-th smallest calibration score, k = ceil((n + 1) * (1 - alpha))

gives P(s_{n+1} > q) <= alpha. The k uses (n + 1), not n -- the FINITE-SAMPLE CORRECTION.
Using ceil(n * (1 - alpha)) instead picks a threshold that is too low and quietly breaks the
guarantee; the gap is O(1/n), so it is invisible on large calibration sets and badly wrong on
small ones.

The score is built so that "s_{n+1} > q" is exactly the event we want to bound:

    s_i = confidence_i   if the field is wrong
    s_i = NEVER_ACCEPTED if the field is right   (below any confidence, so it can never
                                                  be the event being bounded)

so the controlled quantity is the MARGINAL risk P(field is accepted AND wrong), i.e.
`eval.metrics.selective_risk`, taken over all fields. It is not the error rate conditional on
acceptance (`eval.metrics.realized_error`), which is the marginal risk divided by coverage
and therefore larger. Reports must show both.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from core import paths

# Sentinel score for a correct field: strictly below any confidence in [0, 1].
NEVER_ACCEPTED = -1.0


def quantile_index(n: int, alpha: float) -> int:
    """The 1-based order statistic to take: ceil((n + 1) * (1 - alpha)).

    A value greater than n means alpha is smaller than 1/(n + 1) and no finite threshold can
    honour it with this many calibration points.
    """
    if n < 1:
        raise ValueError("need at least one calibration field")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return int(math.ceil((n + 1) * (1.0 - alpha)))


def conformal_threshold(confidence: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """Threshold t such that accepting fields with confidence > t keeps marginal risk <= alpha.

    Returns +inf when the calibration set is too small for alpha, which rejects every field --
    the only honest answer at that sample size.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct).astype(bool)
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same shape")
    if confidence.size and not np.isfinite(confidence).all():
        raise ValueError("confidence contains non-finite values")
    n = int(confidence.size)
    k = quantile_index(n, alpha)
    if k > n:
        return math.inf
    scores = np.where(correct, NEVER_ACCEPTED, confidence)
    return float(np.sort(scores)[k - 1])


def accept(confidence: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean accept mask. The comparison is strict (>), which is what the guarantee assumes."""
    return np.asarray(confidence, dtype=float) > threshold


def threshold_path(dataset: str, method: str, split: str) -> Path:
    """Where the fitted acceptance threshold for one dataset/extractor/split lives."""
    return paths.ARTIFACTS / f"threshold.{dataset}.{method}.{split}.json"


def save_threshold(threshold: float, alpha: float, n_calibration: int, path: Path) -> Path:
    """Persist the threshold together with the alpha and n it was derived from."""
    payload = {
        "threshold": threshold if math.isfinite(threshold) else None,
        "alpha": alpha,
        "n_calibration": n_calibration,
        "quantile_index": quantile_index(n_calibration, alpha),
        "correction": "ceil((n+1)*(1-alpha)) split-conformal finite-sample correction",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_threshold(path: Path) -> dict:
    """Read back a persisted threshold record, restoring +inf from its JSON null."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["threshold"] = math.inf if payload["threshold"] is None else float(payload["threshold"])
    return payload
