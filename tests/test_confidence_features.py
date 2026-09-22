"""Feature-matrix tests: the open `signals` mapping must become a fixed, explicit matrix."""

import numpy as np
import pytest

from confidence.features import (
    FEATURE_COLUMNS,
    MISSING_FILL,
    build_matrix,
    feature_names,
    unknown_signals,
)
from core.schema import FieldPrediction


def test_column_order_is_stable_and_explicit() -> None:
    """Names are value columns in FEATURE_COLUMNS order, then one missing flag per column."""
    names = feature_names()
    assert names[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    assert names[len(FEATURE_COLUMNS):] == tuple(f"{c}_missing" for c in FEATURE_COLUMNS)
    assert len(names) == 2 * len(FEATURE_COLUMNS)
    # Calling twice must not reorder anything.
    assert feature_names() == names


def test_all_signals_present_gives_values_and_zero_missing_flags() -> None:
    """A field carrying every known signal maps to its raw values with no missing flags set."""
    signals = {name: 0.1 * (i + 1) for i, name in enumerate(FEATURE_COLUMNS)}
    matrix = build_matrix([FieldPrediction(name="total", value="18.50", signals=signals)])
    assert matrix.shape == (1, 2 * len(FEATURE_COLUMNS))
    expected = [signals[c] for c in FEATURE_COLUMNS] + [0.0] * len(FEATURE_COLUMNS)
    np.testing.assert_allclose(matrix[0], expected)


def test_missing_signal_is_flagged_not_silently_filled() -> None:
    """A signal one extractor cannot produce is filled *and* marked, so the fit can tell."""
    present, absent = FEATURE_COLUMNS[0], FEATURE_COLUMNS[1]
    matrix = build_matrix([FieldPrediction(name="date", value="2024-01-01",
                                           signals={present: 0.75})])
    idx_absent = FEATURE_COLUMNS.index(absent)
    assert matrix[0, 0] == pytest.approx(0.75)
    assert matrix[0, idx_absent] == pytest.approx(MISSING_FILL)
    assert matrix[0, len(FEATURE_COLUMNS) + 0] == 0.0
    assert matrix[0, len(FEATURE_COLUMNS) + idx_absent] == 1.0


def test_unknown_signals_are_reported_not_dropped_silently() -> None:
    """Signals outside the fixed schema are surfaced by name so they can be added deliberately."""
    fields = [FieldPrediction(name="total", value="1", signals={"brand_new_signal": 1.0}),
              FieldPrediction(name="total", value="2", signals={FEATURE_COLUMNS[0]: 0.5})]
    assert unknown_signals(fields) == ("brand_new_signal",)


def test_non_finite_signal_raises() -> None:
    """NaN/inf would poison a fit silently, so it is rejected at the boundary."""
    bad = [FieldPrediction(name="total", value="1", signals={FEATURE_COLUMNS[0]: float("nan")})]
    with pytest.raises(ValueError):
        build_matrix(bad)


def test_empty_input_keeps_the_column_contract() -> None:
    """No fields still yields a correctly shaped (0, n_columns) matrix."""
    assert build_matrix([]).shape == (0, 2 * len(FEATURE_COLUMNS))
