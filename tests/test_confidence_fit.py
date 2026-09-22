"""CLI tests for `python -m confidence.fit --split calibration`."""

from __future__ import annotations

import numpy as np
import pytest

from confidence.calibrate import calibrator_path, load_calibrator
from confidence.features import FEATURE_COLUMNS
from confidence.conformal import load_threshold, quantile_index, threshold_path
from confidence.fit import load_labelled, main, split_by_document
from core.io import write_jsonl
from core.schema import Document, Extraction, FieldPrediction


def test_readme_invocation_writes_calibrator_and_threshold(synthetic_project) -> None:
    """The documented command fits on the calibration split and persists both artifacts."""
    assert main(["--split", "calibration"]) == 0
    calibrator = load_calibrator(calibrator_path("sroie", "llm", "calibration"))
    assert calibrator.feature_names[0] == "agreement"
    record = load_threshold(threshold_path("sroie", "llm", "calibration"))
    assert record["alpha"] == pytest.approx(0.02)
    assert "(n+1)" in record["correction"]
    assert record["quantile_index"] == quantile_index(record["n_calibration"], record["alpha"])


def test_alpha_and_calibrator_choice_are_honoured(synthetic_project) -> None:
    """Both knobs reach the persisted artifacts rather than being silently ignored."""
    assert main(["--split", "calibration", "--alpha", "0.1", "--calibrator", "platt"]) == 0
    record = load_threshold(threshold_path("sroie", "llm", "calibration"))
    assert record["alpha"] == pytest.approx(0.1)
    assert load_calibrator(calibrator_path("sroie", "llm", "calibration")).coef


def test_calibrator_and_threshold_halves_are_disjoint_whole_documents() -> None:
    """The conformal scores must come from documents the calibrator never saw."""
    doc_ids = [f"d{d}" for d in range(10) for _ in range(3)]
    fit_idx, conformal_idx = split_by_document(doc_ids, frac=0.5, seed=0)
    assert set(fit_idx).isdisjoint(conformal_idx)
    assert sorted(np.concatenate([fit_idx, conformal_idx])) == list(range(len(doc_ids)))
    fit_docs = {doc_ids[i] for i in fit_idx}
    conformal_docs = {doc_ids[i] for i in conformal_idx}
    assert fit_docs.isdisjoint(conformal_docs), "a document leaked across both halves"


def test_load_labelled_joins_gold_and_skips_fields_without_it(tmp_path, monkeypatch) -> None:
    """Fields with no gold entry cannot be scored, so they are dropped, not guessed at."""
    from core import paths

    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    doc = Document(doc_id="d1", dataset="sroie", split="test", image_path="d1.jpg",
                   gold={"total": "18.50"})
    extraction = Extraction(doc_id="d1", method="llm", fields=(
        FieldPrediction(name="total", value="18.50", signals={"agreement": 0.9}),
        FieldPrediction(name="not_in_gold", value="x", signals={"agreement": 0.1}),
    ))
    write_jsonl(paths.documents_path("sroie", "test"), [doc])
    write_jsonl(paths.extraction_path("sroie", "llm", "test"), [extraction])
    labelled = load_labelled("sroie", "llm", "test")
    assert labelled.names == ("total",)
    assert labelled.doc_ids == ("d1",)
    assert labelled.correct.tolist() == [True]
    assert labelled.n_gold == 1
    # Width is two columns per signal (value + missing flag); derived so that adding a
    # signal to FEATURE_COLUMNS does not silently leave this assertion stale.
    assert labelled.features.shape == (1, 2 * len(FEATURE_COLUMNS))


def test_missing_inputs_fail_loudly(tmp_path, monkeypatch, capsys) -> None:
    """A missing extraction file is a clear non-zero exit, not a traceback or a silent pass."""
    from core import paths

    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    assert main(["--split", "calibration"]) != 0
    assert "not found" in capsys.readouterr().err
