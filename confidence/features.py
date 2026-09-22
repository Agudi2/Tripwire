"""Turn the open-ended `signals` mapping on a FieldPrediction into a fixed feature matrix.

Extractors are free to emit whatever signals they can measure, but a fitted calibrator needs
a stable column order and a fixed width. Signals an extractor cannot produce are filled with
a constant *and* marked by a companion indicator column, so a model can tell "signal absent"
apart from "signal present and low" instead of silently treating them as the same thing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from core.schema import FieldPrediction

# The fixed schema, in the order the README lists the confidence signals. Appending a new
# name here is safe; reordering or removing one invalidates every persisted calibrator.
# These names must match what the extractors actually emit -- tests/test_signal_contract.py
# fails if the two vocabularies drift apart, because a renamed signal is silently dropped
# rather than raising, and the calibrator then fits on one column less without saying so.
FEATURE_COLUMNS: tuple[str, ...] = (
    "agreement",       # LLM: fraction of sampled outputs agreeing on this value
    "ocr_conf",        # rules: mean OCR confidence of the tokens the rule consumed
    "format_valid",    # both: graded format plausibility from extract.validators
    "match_strength",  # rules: positional/keyword anchoring quality behind the match
    "mean_logprob",    # LLM: mean token log-prob, only if a cache payload supplies one
)

# Signals deliberately excluded from the matrix, with the reason. Listed rather than ignored
# silently so that `unknown_signals` stays a real alarm instead of routine noise.
IGNORED_SIGNALS: tuple[str, ...] = (
    "n_samples",  # constant per run, so it carries no per-field information
)

# Value used where a signal is absent. It is only meaningful alongside the matching
# "<name>_missing" indicator column.
MISSING_FILL = 0.0


def feature_names() -> tuple[str, ...]:
    """Column names of the matrix: the value columns, then one missing flag per value."""
    return FEATURE_COLUMNS + tuple(f"{name}_missing" for name in FEATURE_COLUMNS)


def build_matrix(fields: Sequence[FieldPrediction]) -> np.ndarray:
    """Build the (n_fields, 2 * n_signals) matrix described by `feature_names`.

    Raises ValueError on a non-finite signal value rather than letting NaN propagate into a
    fit and quietly produce a useless calibrator.
    """
    width = 2 * len(FEATURE_COLUMNS)
    matrix = np.zeros((len(fields), width), dtype=float)
    for row, field in enumerate(fields):
        for column, name in enumerate(FEATURE_COLUMNS):
            if name in field.signals:
                value = float(field.signals[name])
                if not np.isfinite(value):
                    raise ValueError(
                        f"signal {name!r} on field {field.name!r} is not finite: {value!r}")
                matrix[row, column] = value
            else:
                matrix[row, column] = MISSING_FILL
                matrix[row, len(FEATURE_COLUMNS) + column] = 1.0
    return matrix


def unknown_signals(fields: Sequence[FieldPrediction]) -> tuple[str, ...]:
    """Signal names present in the data but outside FEATURE_COLUMNS, sorted.

    Callers surface these so a newly added signal gets adopted deliberately instead of being
    dropped without a trace.
    """
    seen: set[str] = set()
    for field in fields:
        seen.update(field.signals)
    return tuple(sorted(seen - set(FEATURE_COLUMNS) - set(IGNORED_SIGNALS)))


def missing_rate(matrix: np.ndarray) -> dict[str, float]:
    """Fraction of rows missing each signal, for reporting which signals an extractor lacks."""
    n_signals = len(FEATURE_COLUMNS)
    if matrix.shape[1] != 2 * n_signals:
        raise ValueError(f"expected {2 * n_signals} columns, got {matrix.shape[1]}")
    if matrix.shape[0] == 0:
        return {name: 0.0 for name in FEATURE_COLUMNS}
    flags = matrix[:, n_signals:]
    return {name: float(flags[:, i].mean()) for i, name in enumerate(FEATURE_COLUMNS)}
