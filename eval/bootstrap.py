"""Bootstrap confidence intervals, resampled by document.

Fields inside one document share an image, one OCR pass and one LLM call, so their
correctness is correlated. Resampling individual fields would treat them as independent
observations and understate the interval; every resample here therefore draws whole
documents with replacement and keeps each document's fields together.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def document_resample_indices(doc_ids: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    """Draw n_documents documents with replacement and return their field indices, in blocks.

    The result is the concatenation of complete per-document index blocks, so no document is
    ever split across a resample.
    """
    groups = group_indices(doc_ids)
    if not groups:
        return np.zeros(0, dtype=int)
    blocks = list(groups.values())
    picked = rng.integers(0, len(blocks), len(blocks))
    return np.concatenate([blocks[i] for i in picked])


def group_indices(doc_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Index positions of each document's fields, keyed by doc_id in first-appearance order."""
    groups: dict[str, list[int]] = {}
    for position, doc_id in enumerate(doc_ids):
        groups.setdefault(doc_id, []).append(position)
    return {doc_id: np.asarray(positions, dtype=int) for doc_id, positions in groups.items()}


def bootstrap_ci(doc_ids: Sequence[str], statistic: Callable[[np.ndarray], float],
                 n_boot: int = 1000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for `statistic`, resampling by document.

    `statistic` receives the index array of one resample and returns a scalar. Resamples
    whose statistic is undefined (NaN, e.g. no accepted fields) are dropped before taking
    percentiles.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    values = np.array([statistic(document_resample_indices(doc_ids, rng))
                       for _ in range(n_boot)], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)
