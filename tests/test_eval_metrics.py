"""Metric tests: every number is checked against a hand-computed worked example."""

import math

import numpy as np
import pytest

from eval.metrics import (
    brier_score,
    expected_calibration_error,
    precision_recall_f1,
    realized_error,
    risk_coverage_curve,
    selective_risk,
)


def test_precision_recall_f1_hand_computed() -> None:
    """3 of 4 emitted fields correct against 6 gold fields: P=0.75, R=0.5, F1=0.6."""
    correct = np.array([True, True, False, True])
    precision, recall, f1 = precision_recall_f1(correct, n_gold=6)
    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.6)


def test_precision_recall_f1_with_no_predictions_is_zero_not_nan() -> None:
    """Emitting nothing scores zero on all three, rather than dividing by zero."""
    assert precision_recall_f1(np.array([], dtype=bool), n_gold=5) == (0.0, 0.0, 0.0)


def test_expected_calibration_error_hand_computed() -> None:
    """Equal-width 10 bins; four points land alone in bins 0, 1, 8, 9 -> ECE = 0.1."""
    conf = np.array([0.05, 0.15, 0.85, 0.95])
    correct = np.array([False, False, True, True])
    assert expected_calibration_error(conf, correct, n_bins=10) == pytest.approx(0.1)


def test_expected_calibration_error_bin_weighting_is_by_count() -> None:
    """Bins are weighted by how many fields fall in them, not equally.

    Three points at 0.9 (two correct -> gap 0.2333) and one at 0.1 (incorrect -> gap 0.1)
    give (3 * 0.23333 + 1 * 0.1) / 4 = 0.2.
    """
    conf = np.array([0.9, 0.9, 0.9, 0.1])
    correct = np.array([True, True, False, False])
    assert expected_calibration_error(conf, correct, n_bins=10) == pytest.approx(0.2)


def test_perfectly_calibrated_scores_give_near_zero_ece() -> None:
    """Sanity check in the other direction: if P(correct) == confidence, ECE ~ 0."""
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.0, 1.0, 200_000)
    correct = rng.random(200_000) < conf
    assert expected_calibration_error(conf, correct, n_bins=10) < 0.01


def test_brier_score_hand_computed() -> None:
    """Mean squared error of confidence against the 0/1 outcome: 0.05 / 4 = 0.0125."""
    conf = np.array([0.05, 0.15, 0.85, 0.95])
    correct = np.array([False, False, True, True])
    assert brier_score(conf, correct) == pytest.approx(0.0125)


def test_risk_coverage_curve_hand_computed() -> None:
    """Sweeping the most confident fields first: risk is the error rate inside the top k."""
    conf = np.array([0.6, 0.9, 0.7, 0.8])
    correct = np.array([True, True, False, True])
    thresholds, coverage, risk = risk_coverage_curve(conf, correct)
    np.testing.assert_allclose(thresholds, [0.9, 0.8, 0.7, 0.6])
    np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0.0, 0.0, 1 / 3, 0.25])


def test_realized_error_and_selective_risk_differ_by_coverage() -> None:
    """Conditional error among accepted is 1/3; marginal accepted-error risk is 1/4."""
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, True])
    assert realized_error(conf, correct, threshold=0.65) == pytest.approx(1 / 3)
    assert selective_risk(conf, correct, threshold=0.65) == pytest.approx(0.25)


def test_realized_error_is_nan_when_nothing_is_accepted() -> None:
    """With an empty accepted set there is no error rate to report."""
    conf = np.array([0.1, 0.2])
    correct = np.array([True, False])
    assert math.isnan(realized_error(conf, correct, threshold=0.9))
    assert selective_risk(conf, correct, threshold=0.9) == 0.0


def test_metrics_reject_mismatched_lengths() -> None:
    """Confidence and correctness arrays must line up or the numbers are meaningless."""
    with pytest.raises(ValueError):
        brier_score(np.array([0.5, 0.5]), np.array([True]))
