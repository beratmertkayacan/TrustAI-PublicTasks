"""Tests for explanation_drift.metrics (predictive performance side)."""

from __future__ import annotations

import numpy as np
import pytest

from explanation_drift.metrics import (
    PERFORMANCE_METRICS,
    compute_performance,
    performance_drift_score,
    performance_table,
    relative_degradation,
)


@pytest.fixture
def y_true():
    return np.array([0, 0, 0, 1, 1, 1, 0, 1])


@pytest.fixture
def perfect_prob():
    return np.array([0.01, 0.02, 0.10, 0.90, 0.95, 0.99, 0.05, 0.80])


# --------------------------------------------------------------------------- #
# compute_performance
# --------------------------------------------------------------------------- #
def test_perfect_predictions_score_one(y_true, perfect_prob):
    scores = compute_performance(y_true, perfect_prob)
    assert set(scores) == set(PERFORMANCE_METRICS)
    assert scores["accuracy"] == 1.0
    assert scores["roc_auc"] == 1.0
    assert scores["pr_auc"] == 1.0
    assert scores["brier"] < 0.02


def test_inverted_predictions_score_poorly(y_true, perfect_prob):
    scores = compute_performance(y_true, 1.0 - perfect_prob)
    assert scores["roc_auc"] == 0.0
    assert scores["accuracy"] == 0.0


def test_threshold_changes_hard_label_metrics(y_true, perfect_prob):
    strict = compute_performance(y_true, perfect_prob, threshold=0.99)
    assert strict["recall"] < 1.0
    # ranking metrics are threshold-free and therefore unchanged
    assert strict["roc_auc"] == 1.0


@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true:UserWarning")
def test_single_class_target_returns_nan_auc(perfect_prob):
    scores = compute_performance(np.zeros(8, dtype=int), perfect_prob)
    assert np.isnan(scores["roc_auc"])
    assert np.isnan(scores["pr_auc"])
    assert not np.isnan(scores["accuracy"])


@pytest.mark.parametrize(
    "y, prob, message",
    [
        ([], [], "empty"),
        ([0, 1, 1], [0.1, 0.2], "length mismatch"),
        ([0, 1], [[0.1, 0.9], [0.2, 0.8]], "1-D"),
        ([0, 1], [0.1, np.nan], "NaN"),
        ([0, 1], [0.1, 1.5], r"\[0, 1\]"),
        ([0, 2], [0.1, 0.9], "binary"),
    ],
)
def test_compute_performance_invalid_inputs(y, prob, message):
    with pytest.raises(ValueError, match=message):
        compute_performance(y, prob)


@pytest.mark.parametrize("threshold", [0.0, 1.0, 5.0])
def test_compute_performance_invalid_threshold(y_true, perfect_prob, threshold):
    with pytest.raises(ValueError, match="threshold"):
        compute_performance(y_true, perfect_prob, threshold=threshold)


# --------------------------------------------------------------------------- #
# performance_table
# --------------------------------------------------------------------------- #
def test_performance_table_orders_columns(y_true, perfect_prob):
    scores = compute_performance(y_true, perfect_prob)
    table = performance_table(
        [
            {"model": "m", "dataset": "original", **scores},
            {"model": "m", "dataset": "severe", **scores},
        ]
    )
    assert list(table.columns) == ["model", "dataset", *PERFORMANCE_METRICS]
    assert len(table) == 2


def test_performance_table_requires_rows():
    with pytest.raises(ValueError, match="No performance rows"):
        performance_table([])


def test_performance_table_requires_keys(y_true, perfect_prob):
    scores = compute_performance(y_true, perfect_prob)
    with pytest.raises(ValueError, match="'dataset'"):
        performance_table([{"model": "m", **scores}])


# --------------------------------------------------------------------------- #
# degradation
# --------------------------------------------------------------------------- #
def test_relative_degradation_is_zero_for_identical_scores(y_true, perfect_prob):
    scores = compute_performance(y_true, perfect_prob)
    degradation = relative_degradation(scores, scores)
    assert all(abs(v) < 1e-12 for v in degradation.values())


def test_relative_degradation_signs():
    reference = {"roc_auc": 0.80, "brier": 0.10}
    shifted = {"roc_auc": 0.60, "brier": 0.20}
    out = relative_degradation(reference, shifted, ["roc_auc", "brier"])
    assert out["roc_auc"] == pytest.approx(0.25)   # lost a quarter of the AUC
    assert out["brier"] == pytest.approx(1.0)      # error doubled -> positive too


def test_relative_degradation_handles_nan_and_zero_reference():
    out = relative_degradation(
        {"roc_auc": float("nan"), "brier": 0.0},
        {"roc_auc": 0.5, "brier": 0.1},
        ["roc_auc", "brier"],
    )
    assert np.isnan(out["roc_auc"])
    assert np.isnan(out["brier"])


def test_relative_degradation_missing_metric():
    with pytest.raises(KeyError, match="roc_auc"):
        relative_degradation({"brier": 0.1}, {"brier": 0.2}, ["roc_auc"])


def test_performance_drift_score_bounds():
    reference = {"roc_auc": 0.8, "pr_auc": 0.5, "balanced_accuracy": 0.7}
    identical = performance_drift_score(reference, reference)
    assert identical == pytest.approx(0.0)

    worse = performance_drift_score(
        reference, {"roc_auc": 0.6, "pr_auc": 0.4, "balanced_accuracy": 0.6}
    )
    assert 0.0 < worse < 1.0

    better = performance_drift_score(
        reference, {"roc_auc": 0.9, "pr_auc": 0.6, "balanced_accuracy": 0.8}
    )
    assert better == 0.0  # improvements clip to zero, never negative


def test_performance_drift_score_all_nan():
    nan_ref = {"roc_auc": float("nan")}
    assert np.isnan(performance_drift_score(nan_ref, {"roc_auc": 0.5}, ["roc_auc"]))
