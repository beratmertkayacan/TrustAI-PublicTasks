"""Tests for explanation_drift.pipeline (end-to-end orchestration)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explanation_drift.drift import DRIFT_COMPONENTS
from explanation_drift.metrics import PERFORMANCE_METRICS
from explanation_drift import pipeline as pipeline_mod
from explanation_drift.pipeline import BenchmarkResult, main, run_benchmark
from explanation_drift.shift import ORIGINAL_LABEL
from explanation_drift.thresholds import (
    BOTH,
    EXPLANATION_ONLY,
    NEITHER,
    PERFORMANCE_ONLY,
    UNKNOWN,
)

#: Running the full grid in tests is not needed; two families and two levels are enough.
TEST_KINDS = ["age", "mixed"]
TEST_LEVELS = {"moderate": 1.0, "severe": 2.5}

#: A few resamples are enough to exercise the threshold code in tests.
BOOTSTRAP_IN_TESTS = 10


@pytest.fixture
def result(bundle) -> BenchmarkResult:
    return run_benchmark(
        bundle=bundle, kinds=TEST_KINDS, levels=TEST_LEVELS,
        max_samples=20, k=5, output_dir=None, bootstrap_samples=BOOTSTRAP_IN_TESTS,
    )


# in-memory results
def test_benchmark_covers_every_model_and_dataset(result):
    expected_datasets = 1 + len(TEST_KINDS) * len(TEST_LEVELS)
    assert set(result.performance["model"]) == {"logistic_regression", "gradient_boosting"}
    assert len(result.performance) == 2 * expected_datasets
    assert len(result.drift) == 2 * expected_datasets
    assert len(result.comparison) == 2 * expected_datasets
    assert len(result.shift) == expected_datasets # orijinal dahil


def test_benchmark_tables_have_expected_columns(result):
    for metric in PERFORMANCE_METRICS:
        assert metric in result.performance.columns
    for component in DRIFT_COMPONENTS:
        assert component in result.drift.columns
    assert {
        "early_warning_index", "drift_ratio", "verdict",
        "explanation_threshold", "performance_threshold", "gap_ci_low", "gap_ci_high",
    } <= set(result.comparison.columns)


def test_reference_dataset_has_zero_drift(result):
    """Comparing the reference with itself must give exactly zero (the start of the curves)."""
    original = result.comparison[result.comparison["dataset"] == ORIGINAL_LABEL]
    assert len(original) == 2                                       # one row per model
    assert original["explanation_drift"].abs().max() == pytest.approx(0.0)
    assert original["performance_drift"].abs().max() == pytest.approx(0.0)


def test_scores_stay_in_unit_range(result):
    """Every drift value must stay in [0, 1].

    The old version of this test asserted that a severe shift gives a higher
    drift score than a moderate one. That is an experimental result, not a
    correctness rule, so it is now measured and reported instead of asserted.
    """
    for column in [*DRIFT_COMPONENTS, "explanation_drift_score"]:
        values = result.drift[column].dropna()
        assert values.between(0.0, 1.0).all(), column
    for column in ["explanation_drift", "performance_drift"]:
        values = result.comparison[column].dropna()
        assert values.between(0.0, 1.0).all(), column


def test_importance_tables_per_model(result):
    assert set(result.importance) == {"logistic_regression", "gradient_boosting"}
    for frame in result.importance.values():
        assert ORIGINAL_LABEL in frame.columns
        assert "mixed_severe" in frame.columns


def test_verdict_and_rates(result):
    assert result.verdict() in {
        EXPLANATION_ONLY, PERFORMANCE_ONLY, BOTH, NEITHER, UNKNOWN,
    }
    assert 0.0 <= result.early_warning_rate() <= 1.0
    assert 0.0 <= result.significant_gap_rate() <= 1.0
    assert result.early_warning_count() >= 0


def test_thresholds_are_estimated_per_model(result):
    """Each model gets its own noise threshold, taken from unshifted data."""
    assert list(result.thresholds["model"]) == [
        "logistic_regression", "gradient_boosting",
    ]
    for column in ["explanation_threshold", "performance_threshold"]:
        values = result.thresholds[column]
        assert values.between(0.0, 1.0).all()
    assert (result.thresholds["bootstrap_samples"] == BOOTSTRAP_IN_TESTS).all()


def test_verdict_follows_the_threshold_of_its_model(result):
    """The label in each row must match a direct comparison with the thresholds."""
    for row in result.comparison.itertuples():
        above_explanation = row.explanation_drift > row.explanation_threshold
        above_performance = row.performance_drift > row.performance_threshold
        if above_explanation and not above_performance:
            assert row.verdict == EXPLANATION_ONLY
        elif above_performance and not above_explanation:
            assert row.verdict == PERFORMANCE_ONLY
        elif above_explanation and above_performance:
            assert row.verdict == BOTH
        else:
            assert row.verdict == NEITHER


def test_gap_interval_is_ordered(result):
    ordered = result.comparison["gap_ci_low"] <= result.comparison["gap_ci_high"]
    assert ordered.all()


def test_empty_comparison_is_handled():
    empty = BenchmarkResult(
        performance=pd.DataFrame(), shift=pd.DataFrame(), drift=pd.DataFrame(),
        comparison=pd.DataFrame({"dataset": [ORIGINAL_LABEL], "early_warning_index": [0.0],
                                 "verdict": [NEITHER]}),
    )
    assert empty.verdict() == "unknown"
    assert np.isnan(empty.early_warning_rate())


def test_early_warning_rate_nan_when_index_undefined():
    frame = pd.DataFrame(
        {"dataset": ["mixed_severe"], "early_warning_index": [float("nan")],
         "verdict": [UNKNOWN]}
    )
    result = BenchmarkResult(
        performance=pd.DataFrame(), shift=pd.DataFrame(), drift=pd.DataFrame(),
        comparison=frame,
    )
    assert np.isnan(result.early_warning_rate())


# artefacts on disk
def test_rates_are_nan_without_interval_columns():
    """An older table without the interval columns must not crash the summary."""
    result = BenchmarkResult(
        performance=pd.DataFrame(), shift=pd.DataFrame(), drift=pd.DataFrame(),
        comparison=pd.DataFrame(
            {"dataset": ["mixed_severe"], "early_warning_index": [0.1],
             "verdict": [EXPLANATION_ONLY]}
        ),
    )
    assert np.isnan(result.significant_gap_rate())
    assert result.early_warning_count() == 1


def test_rates_are_nan_when_every_interval_is_undefined():
    result = BenchmarkResult(
        performance=pd.DataFrame(), shift=pd.DataFrame(), drift=pd.DataFrame(),
        comparison=pd.DataFrame(
            {"dataset": ["mixed_severe"], "early_warning_index": [0.1],
             "gap_ci_low": [float("nan")], "verdict": [UNKNOWN]}
        ),
    )
    assert np.isnan(result.significant_gap_rate())


def test_counts_are_empty_without_shifted_rows():
    result = BenchmarkResult(
        performance=pd.DataFrame(), shift=pd.DataFrame(), drift=pd.DataFrame(),
        comparison=pd.DataFrame(
            {"dataset": [ORIGINAL_LABEL], "early_warning_index": [0.0],
             "gap_ci_low": [0.0], "verdict": [NEITHER]}
        ),
    )
    assert np.isnan(result.significant_gap_rate())
    assert result.early_warning_count() == 0


def test_benchmark_writes_every_artefact(bundle, tmp_path):
    result = run_benchmark(
        bundle=bundle, kinds=TEST_KINDS, levels=TEST_LEVELS,
        max_samples=20, k=5, output_dir=tmp_path, bootstrap_samples=BOOTSTRAP_IN_TESTS,
    )
    for name, path in result.artefacts.items():
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    saved = pd.read_csv(tmp_path / "early_warning.csv")
    pd.testing.assert_frame_equal(saved, result.comparison)

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "# Explanation Drift Benchmark" in report
    assert "## Early warning thresholds" in report
    assert "## Early warning comparison" in report
    assert "Majority verdict" in report


# command line
def test_main_runs_and_reports(monkeypatch, bundle, tmp_path, capsys):
    monkeypatch.setattr(pipeline_mod, "load_dataset", lambda **kwargs: bundle)
    exit_code = main(["--max-samples", "20", "--top-k", "5",
                      "--bootstrap-samples", str(BOOTSTRAP_IN_TESTS),
                      "--output-dir", str(tmp_path)])
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "Verdict:" in printed
    assert (tmp_path / "benchmark_report.md").exists()