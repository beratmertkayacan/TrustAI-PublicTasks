# What test_thresholds.py covers:
# a null distribution is built from resamples of unshifted data
# the threshold is the requested quantile of that distribution
# each score is compared with its own threshold
# the gap interval is ordered and reacts to a real gap
# invalid inputs are rejected

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explanation_drift.explain import ExplanationResult
from explanation_drift.thresholds import (
    BOTH,
    EXPLANATION_ONLY,
    NEITHER,
    PERFORMANCE_ONLY,
    UNKNOWN,
    DriftThresholds,
    bootstrap_gap_interval,
    estimate_thresholds,
    null_distributions,
    quantile_threshold,
    subset_explanation,
    threshold_table,
)

N_ROWS = 60


def make_explanation(seed: int = 0, scale: float = 1.0) -> ExplanationResult:
    """SHAP matrix with three features of clearly different importance."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "a": rng.normal(2.0 * scale, 0.5, N_ROWS),
        "b": rng.normal(1.0, 0.5, N_ROWS),
        "c": rng.normal(0.2, 0.1, N_ROWS),
    })
    return ExplanationResult(
        dataset="original", model_name="m", shap_values=frame, base_value=0.0
    )


@pytest.fixture
def reference() -> ExplanationResult:
    return make_explanation()


@pytest.fixture
def labels() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    y = (rng.random(N_ROWS) < 0.4).astype(int)
    prob = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.1, N_ROWS), 0.01, 0.99)
    return y, prob


# subsetting
def test_subset_explanation_keeps_columns(reference):
    subset = subset_explanation(reference, [0, 0, 1, 2])
    assert list(subset.shap_values.columns) == reference.feature_names
    assert subset.n_samples == 4
    assert subset.base_value == reference.base_value


def test_subset_explanation_rejects_empty(reference):
    with pytest.raises(ValueError, match="empty subset"):
        subset_explanation(reference, [])


# null distribution
def test_null_distributions_are_small_but_positive(reference, labels):
    y, prob = labels
    explanation, performance = null_distributions(reference, y, prob, n_samples=30)
    assert len(explanation) == 30 and len(performance) == 30
    assert (explanation >= 0).all()
    assert explanation.max() < 1.0
    assert explanation.mean() > 0.0  # resampling alone already moves the score


def test_null_distributions_are_reproducible(reference, labels):
    y, prob = labels
    first = null_distributions(reference, y, prob, n_samples=10, seed=5)[0]
    second = null_distributions(reference, y, prob, n_samples=10, seed=5)[0]
    np.testing.assert_allclose(first, second)


def test_null_distributions_check_row_counts(reference, labels):
    y, prob = labels
    with pytest.raises(ValueError, match="to match the explanation"):
        null_distributions(reference, y[:-1], prob[:-1], n_samples=5)


# thresholds
def test_quantile_threshold_picks_the_quantile():
    values = [0.0, 0.1, 0.2, 0.3, 0.4]
    assert quantile_threshold(values, 0.5) == pytest.approx(0.2)
    assert quantile_threshold(values, 1.0 - 1e-9) == pytest.approx(0.4, abs=1e-6)


def test_quantile_threshold_ignores_nan():
    assert quantile_threshold([0.1, float("nan"), 0.3], 0.5) == pytest.approx(0.2)


def test_quantile_threshold_invalid_inputs():
    with pytest.raises(ValueError, match="quantile"):
        quantile_threshold([0.1, 0.2], 1.5)
    with pytest.raises(ValueError, match="no usable values"):
        quantile_threshold([float("nan")], 0.9)


def test_estimate_thresholds_returns_both_scores(reference, labels):
    y, prob = labels
    thresholds = estimate_thresholds(reference, y, prob, n_samples=30, quantile=0.9)
    assert isinstance(thresholds, DriftThresholds)
    assert 0.0 <= thresholds.explanation <= 1.0
    assert 0.0 <= thresholds.performance <= 1.0
    assert thresholds.quantile == 0.9
    assert thresholds.n_samples == 30


def test_estimate_thresholds_invalid_inputs(reference, labels):
    y, prob = labels
    with pytest.raises(ValueError, match="n_samples"):
        estimate_thresholds(reference, y, prob, n_samples=1)


def test_a_higher_quantile_gives_a_higher_threshold(reference, labels):
    y, prob = labels
    low = estimate_thresholds(reference, y, prob, n_samples=40, quantile=0.5)
    high = estimate_thresholds(reference, y, prob, n_samples=40, quantile=0.99)
    assert high.explanation >= low.explanation


# classification
@pytest.mark.parametrize("explanation, performance, expected", [
    (0.20, 0.01, EXPLANATION_ONLY),
    (0.01, 0.20, PERFORMANCE_ONLY),
    (0.20, 0.20, BOTH),
    (0.01, 0.01, NEITHER),
    (float("nan"), 0.01, UNKNOWN),
])
def test_classify_uses_one_threshold_per_score(explanation, performance, expected):
    thresholds = DriftThresholds(explanation=0.05, performance=0.10, quantile=0.95, n_samples=10)
    assert thresholds.classify(explanation, performance) == expected


# gap interval
def test_gap_interval_is_ordered_and_detects_a_real_gap(reference, labels):
    y, prob = labels
    shifted = make_explanation(seed=0, scale=3.0)   # the same rows, larger attributions
    low, high = bootstrap_gap_interval(reference, shifted, y, prob, prob, n_samples=30)
    assert low <= high
    assert low > 0.0    # explanation moved while performance did not


def test_gap_interval_is_zero_without_a_shift(reference, labels):
    y, prob = labels
    low, high = bootstrap_gap_interval(reference, reference, y, prob, prob, n_samples=30)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.0, abs=1e-9)


def test_gap_interval_without_usable_samples(reference):
    """No replicate gives a usable score, so the interval is nan.

    One class and a prediction that is always wrong: the ranking metrics are
    undefined and balanced accuracy is 0, so the performance drift cannot be
    expressed as a relative loss.
    """
    y = np.zeros(N_ROWS, dtype=int)
    prob = np.full(N_ROWS, 0.99)
    low, high = bootstrap_gap_interval(reference, reference, y, prob, prob, n_samples=5)
    assert np.isnan(low) and np.isnan(high)


def test_gap_interval_checks_row_counts(reference, labels):
    y, prob = labels
    shifted = subset_explanation(reference, range(N_ROWS - 1))
    with pytest.raises(ValueError, match="same rows"):
        bootstrap_gap_interval(reference, shifted, y, prob, prob, n_samples=5)


# table
def test_threshold_table(reference, labels):
    y, prob = labels
    thresholds = {
        "m1": estimate_thresholds(reference, y, prob, n_samples=10),
        "m2": estimate_thresholds(reference, y, prob, n_samples=10, quantile=0.9),
    }
    table = threshold_table(thresholds)
    assert list(table["model"]) == ["m1", "m2"]
    assert list(table.columns) == [
        "model", "explanation_threshold", "performance_threshold",
        "quantile", "bootstrap_samples",
    ]


def test_threshold_table_requires_input():
    with pytest.raises(ValueError, match="No thresholds"):
        threshold_table({})
