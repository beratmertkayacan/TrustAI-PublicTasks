"""Tests for explanation_drift.drift (explanation drift metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explanation_drift.drift import (
    DRIFT_COMPONENTS,
    combine_components,
    drift_components,
    drift_table,
    early_warning_index,
    early_warning_table,
    explanation_drift_score,
    importance_reallocation,
    jaccard_index,
    rank_correlation,
    shap_distribution_shift,
    top_k_overlap,
)
from explanation_drift.explain import ExplanationResult

# small explanation objects

def make_result(values: dict[str, list[float]], dataset: str = "d") -> ExplanationResult:
    """Build an ExplanationResult straight from literal SHAP columns."""
    return ExplanationResult(
        dataset=dataset,
        model_name="m",
        shap_values=pd.DataFrame(values),
        base_value=0.0,
    )


@pytest.fixture
def reference() -> ExplanationResult:
    # importance order: a (2.0) > b (1.0) > c (0.5) > d (0.1)
    return make_result(
        {
            "a": [2.0, -2.0, 2.0, -2.0],
            "b": [1.0, -1.0, 1.0, -1.0],
            "c": [0.5, -0.5, 0.5, -0.5],
            "d": [0.1, -0.1, 0.1, -0.1],
        },
        dataset="original",
    )


@pytest.fixture
def reversed_result(reference) -> ExplanationResult:
    """Fully reversed importance order: d > c > b > a."""
    return make_result(
        {
            "a": [0.1, -0.1, 0.1, -0.1],
            "b": [0.5, -0.5, 0.5, -0.5],
            "c": [1.0, -1.0, 1.0, -1.0],
            "d": [2.0, -2.0, 2.0, -2.0],
        },
        dataset="reversed",
    )

# identity: an unchanged explanation must give zero drift
def test_identical_explanations_have_zero_drift(reference):
    components = drift_components(reference, reference)
    assert set(components) == set(DRIFT_COMPONENTS)
    for name, value in components.items():
        assert value == pytest.approx(0.0, abs=1e-12), name
    assert explanation_drift_score(reference, reference) == pytest.approx(0.0)


def test_reversed_explanations_have_high_drift(reference, reversed_result):
    score = explanation_drift_score(reference, reversed_result, k=2)
    assert score > 0.5

# 1- top-k overlap
def test_jaccard_index():
    assert jaccard_index(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard_index(["a", "b"], ["c", "d"]) == 0.0
    assert jaccard_index(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard_index([], []) == 1.0


def test_top_k_overlap_bounds(reference, reversed_result):
    assert top_k_overlap(reference, reference, k=2) == 1.0
    assert top_k_overlap(reference, reversed_result, k=2) == 0.0 # a,b vs d,c
    assert top_k_overlap(reference, reversed_result, k=4) == 1.0 # hepsi ortak


def test_top_k_overlap_rejects_bad_k(reference):
    with pytest.raises(ValueError, match="k must be positive"):
        top_k_overlap(reference, reference, k=0)

# 2- rank correlation (the test that catches an alignment bug)
def test_rank_correlation_detects_reversed_ranking(reference, reversed_result):
    """Both importance series arrive sorted by their own values. Without name alignment a reversed order would also give +1.0."""
    correlation = rank_correlation(
        reference.global_importance(), reversed_result.global_importance()
    )
    assert correlation == pytest.approx(-1.0)
 
# identity: a series compared with itself gives a perfect correlation
def test_rank_correlation_identity(reference):
    importance = reference.global_importance()
    assert rank_correlation(importance, importance) == pytest.approx(1.0)
    assert rank_correlation(importance, importance, method="kendall") == pytest.approx(1.0)


def test_rank_correlation_single_feature_is_nan():
    single = pd.Series([1.0], index=["a"])
    assert np.isnan(rank_correlation(single, single))


def test_rank_correlation_invalid_inputs(reference):
    importance = reference.global_importance()
    with pytest.raises(ValueError, match="Unknown method"):
        rank_correlation(importance, importance, method="pearson")
    with pytest.raises(TypeError, match="pandas Series"):
        rank_correlation(importance.values, importance)
    with pytest.raises(ValueError, match="Feature sets differ"):
        rank_correlation(importance, importance.drop("a"))

# 3- SHAP distribution shift
def test_distribution_shift_is_zero_for_identical_values(reference):
    report = shap_distribution_shift(reference, reference)
    assert list(report.columns) == [
        "feature", "wasserstein_norm", "js_distance",
        "mean_abs_reference", "mean_abs_shifted",
    ]
    assert report["js_distance"].max() == pytest.approx(0.0)
    assert report["wasserstein_norm"].max() == pytest.approx(0.0)


def test_distribution_shift_detects_a_moved_feature(reference):
    moved = make_result(
        {
            "a": [12.0, 10.0, 11.0, 13.0], # moved into a completely different range
            "b": [1.0, -1.0, 1.0, -1.0],
            "c": [0.5, -0.5, 0.5, -0.5],
            "d": [0.1, -0.1, 0.1, -0.1],
        }
    )
    report = shap_distribution_shift(reference, moved).set_index("feature")
    assert report.loc["a", "js_distance"] == pytest.approx(1.0) # no overlap
    assert report.loc["b", "js_distance"] == pytest.approx(0.0)


def test_distribution_shift_constant_feature_is_zero():
    flat = make_result({"a": [3.0, 3.0, 3.0]})
    report = shap_distribution_shift(flat, flat)
    assert report["js_distance"].iloc[0] == 0.0


def test_distribution_shift_invalid_inputs(reference):
    with pytest.raises(ValueError, match="bins must be"):
        shap_distribution_shift(reference, reference, bins=1)
    other = make_result({"a": [1.0], "z": [1.0]})
    with pytest.raises(ValueError, match="same features"):
        shap_distribution_shift(reference, other)

# 4- Importance reallocation
def test_importance_reallocation_bounds():
    left = pd.Series([1.0, 0.0], index=["a", "b"])
    right = pd.Series([0.0, 1.0], index=["a", "b"])
    assert importance_reallocation(left, left) == pytest.approx(0.0)
    assert importance_reallocation(left, right) == pytest.approx(1.0) # full handover
    half = pd.Series([0.5, 0.5], index=["a", "b"])
    assert importance_reallocation(left, half) == pytest.approx(0.5)


def test_importance_reallocation_handles_all_zero():
    zeros = pd.Series([0.0, 0.0], index=["a", "b"])
    other = pd.Series([1.0, 1.0], index=["a", "b"])
    assert importance_reallocation(zeros, other) == pytest.approx(0.0)

# composite score
def test_combine_components_weighted_mean():
    components = dict.fromkeys(DRIFT_COMPONENTS, 0.4)
    assert combine_components(components) == pytest.approx(0.4)
    weights = {"top_k_overlap_loss": 1.0}
    components["top_k_overlap_loss"] = 0.9
    assert combine_components(components, weights) == pytest.approx(0.9)


def test_combine_components_skips_nan():
    components = dict.fromkeys(DRIFT_COMPONENTS, 0.2)
    components["rank_disagreement"] = float("nan")
    assert combine_components(components) == pytest.approx(0.2)
    assert np.isnan(combine_components(dict.fromkeys(DRIFT_COMPONENTS, float("nan"))))


def test_combine_components_invalid_weights():
    components = dict.fromkeys(DRIFT_COMPONENTS, 0.2)
    with pytest.raises(ValueError, match="Unknown drift components"):
        combine_components(components, {"made_up": 1.0})
    with pytest.raises(ValueError, match="positive number"):
        combine_components(components, {"top_k_overlap_loss": 0.0})


def test_drift_table(reference, reversed_result):
    results = {"original": reference, "reversed": reversed_result}
    table = drift_table(results, k=2)
    assert list(table["dataset"]) == ["original", "reversed"]
    assert table.loc[0, "explanation_drift_score"] == pytest.approx(0.0)
    assert table.loc[1, "explanation_drift_score"] > 0.5
    for name in DRIFT_COMPONENTS:
        assert name in table.columns


def test_drift_table_requires_reference(reference):
    with pytest.raises(ValueError, match="Reference dataset 'original' missing"):
        drift_table({"shifted": reference})

# research question: which degrades first?
def test_early_warning_index_sign():
    assert early_warning_index(0.3, 0.1) == pytest.approx(0.2) # explanation ahead
    assert early_warning_index(0.1, 0.3) == pytest.approx(-0.2) # performance ahead
    assert np.isnan(early_warning_index(float("nan"), 0.1))


def test_early_warning_table_verdicts():
    table = early_warning_table(
        {"a": 0.30, "b": 0.10, "c": 0.20},
        {"a": 0.10, "b": 0.30, "c": 0.20},
    ).set_index("dataset")
    assert table.loc["a", "verdict"] == "explanation degrades first"
    assert table.loc["b", "verdict"] == "performance degrades first"
    assert table.loc["c", "verdict"] == "degrade together"
    assert table.loc["a", "drift_ratio"] == pytest.approx(3.0)


def test_early_warning_table_zero_performance_drift():
    table = early_warning_table({"a": 0.2}, {"a": 0.0})
    assert np.isnan(table.loc[0, "drift_ratio"])


def test_early_warning_table_unknown_verdict():
    """On a subset with a single class the performance score can be nan; then the verdict must be "unknown"."""
    table = early_warning_table({"a": 0.2}, {"a": float("nan")})
    assert table.loc[0, "verdict"] == "unknown"
    assert np.isnan(table.loc[0, "early_warning_index"])


def test_early_warning_table_invalid_inputs():
    with pytest.raises(ValueError, match="mismatch"):
        early_warning_table({"a": 0.1}, {"b": 0.1})
    with pytest.raises(ValueError, match="No datasets"):
        early_warning_table({}, {})


# invariants of the metric itself
def test_drift_score_is_symmetric(reference, reversed_result):
    """Swapping the two explanations must give the same score."""
    forward = explanation_drift_score(reference, reversed_result, k=2)
    backward = explanation_drift_score(reversed_result, reference, k=2)
    assert forward == pytest.approx(backward)


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 5.0, 100.0])
def test_components_stay_in_unit_range(reference, scale):
    """Whatever the input, every component and the score must stay in [0, 1]."""
    other = make_result({
        "a": [0.1 * scale, -0.2 * scale, 0.3 * scale, 0.0],
        "b": [1.0 * scale, 1.0, -1.0, 0.5],
        "c": [0.0, 0.0, 0.0, 0.0],
        "d": [-2.0 * scale, 2.0, 0.5, -0.5],
    })
    components = drift_components(reference, other, k=2)
    for name, value in components.items():
        if not np.isnan(value):
            assert 0.0 <= value <= 1.0, name
    score = explanation_drift_score(reference, other, k=2)
    assert 0.0 <= score <= 1.0


def test_larger_shift_can_lower_the_drift_score():
    """A bigger shift does not have to give a bigger drift score.

    For a linear model a SHAP value is weight * (x - background mean), so
    moving a feature past that mean can bring its attribution back to the
    original size. Global importance then returns to the reference even though
    the distribution moved further. The counterexample below is built from that
    idea: doubling the shift lowers the score.
    """
    ref = make_result({"f": [2.0, 2.0, 2.0, 2.0], "g": [1.0, 1.0, 1.0, 1.0]})
    small = make_result({"f": [0.0, 0.0, 0.0, 0.0], "g": [1.0, 1.0, 1.0, 1.0]})
    large = make_result({"f": [-2.0, -2.0, -2.0, -2.0], "g": [1.0, 1.0, 1.0, 1.0]})

    score_small = explanation_drift_score(ref, small, k=1)
    score_large = explanation_drift_score(ref, large, k=1)
    assert score_large < score_small

