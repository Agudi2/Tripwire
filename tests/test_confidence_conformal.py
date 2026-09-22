"""Conformal tests: the (n+1)-corrected split-conformal threshold must hold its guarantee."""

import math

import numpy as np

from confidence.conformal import conformal_threshold, quantile_index
from eval.metrics import selective_risk


def _trial(rng: np.random.Generator, n_cal: int, n_test: int, alpha: float) -> tuple[float, float]:
    """Draw one calibration/test pair, fit a threshold, and return (risk, coverage) on test.

    Confidences are drawn uniform on [0, 1] and a field is correct with probability equal to
    its confidence, so the scores are perfectly calibrated and the only thing under test is
    the conformal threshold itself.
    """
    conf_cal = rng.uniform(0.0, 1.0, n_cal)
    correct_cal = rng.random(n_cal) < conf_cal
    conf_test = rng.uniform(0.0, 1.0, n_test)
    correct_test = rng.random(n_test) < conf_test
    threshold = conformal_threshold(conf_cal, correct_cal, alpha)
    return selective_risk(conf_test, correct_test, threshold), float(np.mean(conf_test > threshold))


def test_quantile_index_uses_the_n_plus_one_correction() -> None:
    """Index is ceil((n+1)(1-alpha)); with n=99, alpha=0.05 that is 95, not 94."""
    assert quantile_index(99, 0.05) == 95
    assert quantile_index(19, 0.10) == 18
    assert quantile_index(100, 0.05) == 96
    # n=10, alpha=0.1 is where dropping the correction bites: 10 corrected vs 9 uncorrected.
    assert quantile_index(10, 0.10) == 10


def test_threshold_is_infinite_when_n_is_too_small_for_alpha() -> None:
    """With too few calibration points no finite threshold can honour alpha, so accept nothing."""
    rng = np.random.default_rng(0)
    conf = rng.random(10)
    correct = rng.random(10) < conf
    assert math.isinf(conformal_threshold(conf, correct, alpha=0.01))


def test_small_n_risk_holds_which_fails_without_the_correction() -> None:
    """At n=10, alpha=0.1 the corrected rule realises ~0.091 error and an uncorrected one ~0.18.

    4000 trials put the standard error of the mean near 0.002, so this separates the two
    rules decisively rather than by luck.
    """
    rng = np.random.default_rng(17)
    risks = [_trial(rng, n_cal=10, n_test=500, alpha=0.10)[0] for _ in range(4000)]
    assert float(np.mean(risks)) <= 0.10


def test_realized_error_stays_under_alpha_across_many_trials() -> None:
    """Over repeated draws the accepted-error rate on fresh data must not exceed alpha.

    The guarantee is marginal (it averages over calibration draws), so a single trial can
    exceed alpha by sampling noise. Documented finite-sample slack: the mean is allowed
    3 empirical standard errors of headroom, and the 95th percentile of per-trial risk is
    allowed 4 standard errors of a single trial's binomial noise.
    """
    alpha, n_cal, n_test, trials = 0.05, 2000, 2000, 200
    rng = np.random.default_rng(7)
    outcomes = [_trial(rng, n_cal, n_test, alpha) for _ in range(trials)]
    risks = np.array([r for r, _ in outcomes])
    coverages = np.array([c for _, c in outcomes])
    slack = 4 * math.sqrt(alpha * (1 - alpha) / n_test)
    assert risks.mean() <= alpha + 3 * risks.std(ddof=1) / math.sqrt(trials)
    assert float(np.percentile(risks, 95)) <= alpha + slack
    # The threshold must not pass by trivially rejecting everything: theory puts coverage
    # near 1 - sqrt(2 * alpha) = 0.68 rejected, i.e. about 0.32 accepted, for this generator.
    assert coverages.mean() > 0.25


def test_smaller_alpha_gives_a_stricter_threshold() -> None:
    """Tightening the error target can only raise the acceptance bar."""
    rng = np.random.default_rng(11)
    conf = rng.uniform(0, 1, 5000)
    correct = rng.random(5000) < conf
    assert conformal_threshold(conf, correct, 0.02) >= conformal_threshold(conf, correct, 0.10)


def test_perfect_confidences_accept_everything() -> None:
    """If no calibration field is wrong, nothing forces a cutoff and all fields are accepted."""
    conf = np.linspace(0.5, 1.0, 200)
    correct = np.ones(200, dtype=bool)
    assert conformal_threshold(conf, correct, alpha=0.05) < conf.min()
