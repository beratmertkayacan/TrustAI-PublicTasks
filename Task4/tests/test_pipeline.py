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

#: Testlerde tüm ızgarayı koşmak gereksiz; iki aile ve iki şiddet yeterli.
TEST_KINDS = ["age", "mixed"]
TEST_LEVELS = {"moderate": 1.0, "severe": 2.5}


@pytest.fixture
def result(bundle) -> BenchmarkResult:
    return run_benchmark(
        bundle=bundle, kinds=TEST_KINDS, levels=TEST_LEVELS,
        max_samples=20, k=5, output_dir=None,
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
    assert {"early_warning_index", "drift_ratio", "verdict"} <= set(result.comparison.columns)


def test_reference_dataset_has_zero_drift(result):
    """Referansın kendisiyle karşılaştırması tam sıfır olmalı (eğrilerin başlangıcı)"""
    original = result.comparison[result.comparison["dataset"] == ORIGINAL_LABEL]
    assert len(original) == 2                                       # model başına bir satır
    assert original["explanation_drift"].abs().max() == pytest.approx(0.0)
    assert original["performance_drift"].abs().max() == pytest.approx(0.0)


def test_drift_grows_with_severity(result):
    """Severe her zaman moderate'ten daha yüksek drift üretmeli."""
    for model in result.drift["model"].unique():
        subset = result.drift[result.drift["model"] == model].set_index("dataset")
        assert (
            subset.loc["mixed_severe", "explanation_drift_score"]
            > subset.loc["mixed_moderate", "explanation_drift_score"]
        )


def test_importance_tables_per_model(result):
    assert set(result.importance) == {"logistic_regression", "gradient_boosting"}
    for frame in result.importance.values():
        assert ORIGINAL_LABEL in frame.columns
        assert "mixed_severe" in frame.columns


def test_verdict_and_early_warning_rate(result):
    assert result.verdict() in {
        "explanation degrades first", "performance degrades first",
        "degrade together", "unknown",
    }
    rate = result.early_warning_rate()
    assert 0.0 <= rate <= 1.0


def test_empty_comparison_is_handled():
    empty = BenchmarkResult(
        performance=pd.DataFrame(), shift=pd.DataFrame(), drift=pd.DataFrame(),
        comparison=pd.DataFrame({"dataset": [ORIGINAL_LABEL], "early_warning_index": [0.0],
                                 "verdict": ["degrade together"]}),
    )
    assert empty.verdict() == "unknown"
    assert np.isnan(empty.early_warning_rate())


def test_early_warning_rate_nan_when_index_undefined():
    frame = pd.DataFrame(
        {"dataset": ["mixed_severe"], "early_warning_index": [float("nan")],
         "verdict": ["unknown"]}
    )
    result = BenchmarkResult(
        performance=pd.DataFrame(), shift=pd.DataFrame(), drift=pd.DataFrame(),
        comparison=frame,
    )
    assert np.isnan(result.early_warning_rate())


# artefacts on disk
def test_benchmark_writes_every_artefact(bundle, tmp_path):
    result = run_benchmark(
        bundle=bundle, kinds=TEST_KINDS, levels=TEST_LEVELS,
        max_samples=20, k=5, output_dir=tmp_path,
    )
    for name, path in result.artefacts.items():
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    saved = pd.read_csv(tmp_path / "early_warning.csv")
    pd.testing.assert_frame_equal(saved, result.comparison)

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "# Explanation Drift Benchmark" in report
    assert "## Early warning comparison" in report
    assert "Majority verdict" in report


# command line
def test_main_runs_and_reports(monkeypatch, bundle, tmp_path, capsys):
    monkeypatch.setattr(pipeline_mod, "load_dataset", lambda **kwargs: bundle)
    exit_code = main(["--max-samples", "20", "--top-k", "5",
                      "--output-dir", str(tmp_path)])
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "Verdict:" in printed
    assert (tmp_path / "benchmark_report.md").exists()