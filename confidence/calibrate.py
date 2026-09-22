"""Platt scaling and isotonic regression, fit on the calibration split and persisted as JSON.

Both calibrators map a feature matrix to P(field is correct). Platt is a logistic fit on the
features directly. Isotonic regression is one-dimensional by construction, so the features
are first reduced to a scalar by the same logistic fit and the isotonic map is fit on that
score -- the usual "base score, then non-parametric recalibration" arrangement.

Persisted files carry the fitted parameters only (no pickled estimator), so reloading is
version-independent and predictions are reproduced exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from core import paths

# Platt scaling is a maximum-likelihood logistic fit; sklearn regularises by default, so C is
# set high enough that the penalty is negligible while keeping separable data from diverging.
_PLATT_C = 1e6


@dataclass(frozen=True)
class PlattCalibrator:
    """Logistic map from features to calibrated probability."""

    feature_names: tuple[str, ...]
    coef: tuple[float, ...]
    intercept: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Calibrated P(correct) for each row of the feature matrix."""
        scores = _linear_score(features, self.feature_names, self.coef, self.intercept)
        return 1.0 / (1.0 + np.exp(-scores))


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone step map from a linear base score to a calibrated probability."""

    feature_names: tuple[str, ...]
    coef: tuple[float, ...]
    intercept: float
    knots_x: tuple[float, ...]  # base scores at which the fitted map changes
    knots_y: tuple[float, ...]  # calibrated probabilities at those knots

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Calibrated P(correct), interpolating between knots and clipping outside them."""
        scores = _linear_score(features, self.feature_names, self.coef, self.intercept)
        return np.interp(scores, np.asarray(self.knots_x), np.asarray(self.knots_y))


Calibrator = PlattCalibrator | IsotonicCalibrator


def fit_platt(features: np.ndarray, correct: np.ndarray,
              feature_names: tuple[str, ...]) -> PlattCalibrator:
    """Fit Platt scaling (a logistic regression) on the calibration split."""
    model = _fit_logistic(features, correct, feature_names)
    return PlattCalibrator(feature_names=tuple(feature_names),
                           coef=tuple(float(c) for c in model.coef_[0]),
                           intercept=float(model.intercept_[0]))


def fit_isotonic(features: np.ndarray, correct: np.ndarray,
                 feature_names: tuple[str, ...]) -> IsotonicCalibrator:
    """Fit an isotonic map on the calibration split, over a logistic base score."""
    model = _fit_logistic(features, correct, feature_names)
    base = model.decision_function(np.asarray(features, dtype=float))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(base, np.asarray(correct, dtype=float))
    return IsotonicCalibrator(feature_names=tuple(feature_names),
                              coef=tuple(float(c) for c in model.coef_[0]),
                              intercept=float(model.intercept_[0]),
                              knots_x=tuple(float(x) for x in iso.X_thresholds_),
                              knots_y=tuple(float(y) for y in iso.y_thresholds_))


def calibrator_path(dataset: str, method: str, split: str) -> Path:
    """Where the fitted calibrator for one dataset/extractor/split lives under ARTIFACTS."""
    return paths.ARTIFACTS / f"calibrator.{dataset}.{method}.{split}.json"


def save_calibrator(calibrator: Calibrator, path: Path) -> Path:
    """Write the fitted parameters as JSON, creating the artifacts directory if needed."""
    payload: dict[str, object] = {
        "kind": "isotonic" if isinstance(calibrator, IsotonicCalibrator) else "platt",
        "feature_names": list(calibrator.feature_names),
        "coef": list(calibrator.coef),
        "intercept": calibrator.intercept,
    }
    if isinstance(calibrator, IsotonicCalibrator):
        payload["knots_x"] = list(calibrator.knots_x)
        payload["knots_y"] = list(calibrator.knots_y)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_calibrator(path: Path) -> Calibrator:
    """Rebuild a calibrator from its JSON parameters."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = tuple(payload["feature_names"])
    coef = tuple(float(c) for c in payload["coef"])
    intercept = float(payload["intercept"])
    if payload["kind"] == "platt":
        return PlattCalibrator(feature_names=names, coef=coef, intercept=intercept)
    if payload["kind"] == "isotonic":
        return IsotonicCalibrator(feature_names=names, coef=coef, intercept=intercept,
                                  knots_x=tuple(float(x) for x in payload["knots_x"]),
                                  knots_y=tuple(float(y) for y in payload["knots_y"]))
    raise ValueError(f"unknown calibrator kind: {payload['kind']!r}")


def _fit_logistic(features: np.ndarray, correct: np.ndarray,
                  feature_names: tuple[str, ...]) -> LogisticRegression:
    """Shared logistic fit with the checks both calibrators need."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(correct).astype(int)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError(f"expected {len(feature_names)} feature columns, got {features.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels have different lengths")
    if np.unique(labels).size < 2:
        raise ValueError("calibration labels contain a single class; cannot fit a calibrator")
    return LogisticRegression(C=_PLATT_C, max_iter=1000).fit(features, labels)


def _linear_score(features: np.ndarray, feature_names: tuple[str, ...],
                  coef: tuple[float, ...], intercept: float) -> np.ndarray:
    """Apply the fitted linear map, refusing a matrix with the wrong number of columns."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError(f"expected {len(feature_names)} feature columns, got {features.shape}")
    return features @ np.asarray(coef) + intercept
