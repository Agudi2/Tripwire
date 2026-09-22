"""Calibration tests: fitting must measurably reduce ECE and survive save/reload."""

import numpy as np
import pytest

from confidence.calibrate import (
    calibrator_path,
    fit_isotonic,
    fit_platt,
    load_calibrator,
    save_calibrator,
)
from eval.metrics import expected_calibration_error


def _miscalibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic scores that are badly overconfident: score = p**3 while P(correct) = p."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, size=n)
    correct = rng.random(n) < p
    score = p**3
    return score.reshape(-1, 1), correct, p


def test_platt_reduces_ece() -> None:
    """Platt scaling on a held-out half cuts ECE several-fold below the raw score.

    It does not drive ECE near zero here, and it cannot: Platt is a two-parameter sigmoid in
    the score, while the injected distortion is a cube root, which is not sigmoidal. Grid
    searching the population-optimal sigmoid against the true map p = s**(1/3) leaves a mean
    absolute error of 0.062, and the fit lands there, so a residual ECE around 0.06 is the
    correct answer for this model class rather than a sign of underfitting. The assertion is
    therefore on the size of the improvement; the isotonic comparison test below pins down
    the part Platt structurally cannot fix.
    """
    x_cal, y_cal, _ = _miscalibrated(4000, seed=1)
    x_test, y_test, _ = _miscalibrated(4000, seed=2)
    before = expected_calibration_error(x_test[:, 0], y_test)
    after = expected_calibration_error(fit_platt(x_cal, y_cal, ("score",)).predict_proba(x_test),
                                       y_test)
    assert after < before / 3
    assert after < 0.10


def test_isotonic_reduces_ece() -> None:
    """Isotonic regression, being non-parametric, tracks the true map far more closely."""
    x_cal, y_cal, _ = _miscalibrated(4000, seed=3)
    x_test, y_test, _ = _miscalibrated(4000, seed=4)
    before = expected_calibration_error(x_test[:, 0], y_test)
    after = expected_calibration_error(fit_isotonic(x_cal, y_cal, ("score",)).predict_proba(x_test),
                                       y_test)
    assert after < before / 6
    assert after < 0.05


def test_isotonic_beats_platt_when_the_distortion_is_not_sigmoidal() -> None:
    """On identical data the flexible calibrator must win, which is why both are offered.

    This is the direct evidence that Platt's residual ECE is a functional-form limit: given
    the same fit and test data, isotonic removes distortion that no sigmoid in the score can.
    """
    x_cal, y_cal, _ = _miscalibrated(4000, seed=1)
    x_test, y_test, _ = _miscalibrated(4000, seed=2)
    platt_ece = expected_calibration_error(
        fit_platt(x_cal, y_cal, ("score",)).predict_proba(x_test), y_test)
    isotonic_ece = expected_calibration_error(
        fit_isotonic(x_cal, y_cal, ("score",)).predict_proba(x_test), y_test)
    assert isotonic_ece < 0.75 * platt_ece


@pytest.mark.parametrize("fit", [fit_platt, fit_isotonic])
def test_save_and_reload_reproduces_predictions_exactly(fit, tmp_path) -> None:
    """Persisted parameters must round-trip bit-for-bit, or reported numbers drift."""
    x_cal, y_cal, _ = _miscalibrated(2000, seed=5)
    model = fit(x_cal, y_cal, ("score",))
    path = tmp_path / "cal.json"
    save_calibrator(model, path)
    reloaded = load_calibrator(path)
    assert reloaded.feature_names == model.feature_names
    np.testing.assert_array_equal(reloaded.predict_proba(x_cal), model.predict_proba(x_cal))


def test_predict_rejects_wrong_column_count() -> None:
    """A matrix built with a different feature set must not be scored by accident."""
    x_cal, y_cal, _ = _miscalibrated(500, seed=6)
    model = fit_platt(x_cal, y_cal, ("score",))
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((3, 2)))


def test_single_class_labels_raise() -> None:
    """All-correct calibration data cannot define a calibration map; fail loudly."""
    x = np.linspace(0, 1, 50).reshape(-1, 1)
    with pytest.raises(ValueError):
        fit_platt(x, np.ones(50, dtype=bool), ("score",))


def test_calibrator_path_is_under_artifacts() -> None:
    """Fitted parameters live under core.paths.ARTIFACTS, not next to the code."""
    from core.paths import ARTIFACTS

    path = calibrator_path("sroie", "llm", "calibration")
    assert path.parent == ARTIFACTS
    assert path.suffix == ".json"
