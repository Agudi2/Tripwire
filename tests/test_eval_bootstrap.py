"""Bootstrap tests: resampling must happen by document, and that is verified, not assumed."""

import numpy as np

from eval.bootstrap import bootstrap_ci, document_resample_indices


def _grouped(doc_ids: list[str]) -> dict[str, list[int]]:
    """Index positions of each document, in first-appearance order (mirrors the module)."""
    groups: dict[str, list[int]] = {}
    for i, doc in enumerate(doc_ids):
        groups.setdefault(doc, []).append(i)
    return groups


def test_resample_takes_whole_documents_as_blocks() -> None:
    """Every stretch of the resample is one document's complete, in-order index block."""
    doc_ids = ["a", "a", "a", "b", "c", "c"]
    groups = _grouped(doc_ids)
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = document_resample_indices(doc_ids, rng)
        pos = 0
        n_blocks = 0
        while pos < idx.size:
            block = groups[doc_ids[idx[pos]]]
            assert idx[pos:pos + len(block)].tolist() == block, "fields split across a document"
            pos += len(block)
            n_blocks += 1
        assert n_blocks == len(groups), "must draw as many documents as the original has"


def test_resample_never_splits_a_document() -> None:
    """Each document appears a whole number of times, never a partial set of its fields."""
    doc_ids = ["a"] * 4 + ["b"] * 7 + ["c"] * 2
    groups = _grouped(doc_ids)
    rng = np.random.default_rng(1)
    for _ in range(50):
        idx = document_resample_indices(doc_ids, rng)
        drawn = [doc_ids[i] for i in idx]
        for doc, block in groups.items():
            assert drawn.count(doc) % len(block) == 0


def test_resample_is_deterministic_for_a_given_seed() -> None:
    """Same seed, same resample; reported CIs have to be reproducible."""
    doc_ids = ["a", "a", "b", "c", "c", "c"]
    first = document_resample_indices(doc_ids, np.random.default_rng(42))
    second = document_resample_indices(doc_ids, np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)


def test_by_document_interval_is_wider_than_by_field_when_fields_correlate() -> None:
    """The reason for resampling by document: within-document correlation.

    Each document here has 20 fields that are all right or all wrong together, so the
    effective sample size is 40 documents, not 800 fields. A by-field bootstrap would
    understate the interval; this asserts the by-document one is materially wider.
    """
    n_docs, per_doc = 40, 20
    rng = np.random.default_rng(5)
    outcome = rng.random(n_docs) < 0.5
    doc_ids = [f"d{d}" for d in range(n_docs) for _ in range(per_doc)]
    values = np.repeat(outcome.astype(float), per_doc)

    lo_doc, hi_doc = bootstrap_ci(doc_ids, lambda idx: float(values[idx].mean()),
                                  n_boot=500, seed=0)
    field_rng = np.random.default_rng(0)
    naive = [float(values[field_rng.integers(0, values.size, values.size)].mean())
             for _ in range(500)]
    lo_field, hi_field = np.percentile(naive, [2.5, 97.5])

    assert (hi_doc - lo_doc) > 2 * (hi_field - lo_field)


def test_interval_brackets_the_point_estimate() -> None:
    """A percentile interval must contain the statistic computed on the original sample."""
    rng = np.random.default_rng(9)
    doc_ids = [f"d{d}" for d in range(60) for _ in range(3)]
    values = rng.normal(loc=2.0, scale=1.0, size=180)
    point = float(values.mean())
    lo, hi = bootstrap_ci(doc_ids, lambda idx: float(values[idx].mean()), n_boot=400, seed=1)
    assert lo <= point <= hi
    assert lo < hi
