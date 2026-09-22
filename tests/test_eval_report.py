"""CLI tests for `python -m eval.report --split test`."""

from __future__ import annotations

from confidence.fit import scored_path
from core import paths
from core.io import load_scored_fields
from eval.report import build_report, main


def test_readme_invocation_writes_a_markdown_table(synthetic_project) -> None:
    """The documented command evaluates the test split and writes a table under REPORTS."""
    from confidence.fit import main as fit_main

    assert fit_main(["--split", "calibration"]) == 0
    assert main(["--split", "test"]) == 0
    report = (paths.REPORTS / "results.test.md").read_text(encoding="utf-8")
    assert "| Extractor |" in report
    assert "llm" in report
    assert "ECE" in report and "Brier" in report and "Coverage" in report
    # CIs must be shown, and they are the by-document ones.
    assert "95% CI" in report or "[" in report
    assert "resampled by document" in report


def test_report_writes_scored_fields_that_respect_the_threshold(synthetic_project) -> None:
    """Every accepted field must sit strictly above the persisted conformal threshold."""
    from confidence.conformal import load_threshold, threshold_path
    from confidence.fit import main as fit_main

    fit_main(["--split", "calibration"])
    main(["--split", "test"])
    threshold = load_threshold(threshold_path("sroie", "llm", "calibration"))["threshold"]
    scored = load_scored_fields(scored_path("sroie", "llm", "test"))
    assert scored
    assert all(f.accepted == (f.confidence > threshold) for f in scored)
    assert all(f.correct is not None for f in scored)


def test_report_fails_clearly_without_a_fitted_calibrator(synthetic_project, capsys) -> None:
    """Reporting before fitting is an error with a pointer to the fit command."""
    assert main(["--split", "test"]) != 0
    assert "confidence.fit" in capsys.readouterr().err


def test_build_report_renders_the_numbers_it_is_given() -> None:
    """The table body comes from the results mapping, with no recomputation or rounding drift."""
    results = {"llm": {"n_fields": 10, "n_docs": 3, "f1": 0.5, "precision": 0.5, "recall": 0.5,
                       "ece_raw": 0.25, "ece_raw_signal": "agreement", "ece": 0.05, "brier": 0.1, "threshold": 0.4,
                       "coverage": 0.6, "realized_error": 0.02, "selective_risk": 0.012,
                       "f1_ci": (0.4, 0.6), "realized_error_ci": (0.0, 0.05)}}
    text = build_report("test", alpha=0.02, results=results)
    assert "| llm |" in text
    assert "0.500" in text and "0.250" in text
    assert "alpha = 0.02" in text
    assert "resampled by document" in text


def test_report_is_deterministic(synthetic_project) -> None:
    """Two runs over the same inputs produce byte-identical reports, CIs included."""
    from confidence.fit import main as fit_main

    fit_main(["--split", "calibration"])
    main(["--split", "test"])
    first = (paths.REPORTS / "results.test.md").read_text(encoding="utf-8")
    main(["--split", "test"])
    assert (paths.REPORTS / "results.test.md").read_text(encoding="utf-8") == first
