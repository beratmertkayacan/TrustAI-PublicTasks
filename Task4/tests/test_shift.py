# Tests for explanation_drift.shift (shift generation + magnitude measurement).
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explanation_drift.shift import (
    AGE_BOUNDS,
    AGE_COLUMN,
    LIMIT_COLUMN,
    ORIGINAL_LABEL,
    PAYMENT_COLUMNS,
    SEVERITY_LEVELS,
    SHIFT_KINDS,
    apply_shift,
    classify_shift,
    generate_shifted_datasets,
    normalised_wasserstein,
    population_stability_index,
    shift_age,
    shift_credit_limit,
    shift_magnitude,
    shift_mixed,
    shift_payment_amount,
    shift_summary,
)


#shift families
@pytest.mark.parametrize("kind", SHIFT_KINDS)
def test_shift_does_not_mutate_input(X_synth, kind):
    # en kritik invariant: orijinal test seti hiçbir zaman bozulmamalı.
    before = X_synth.copy()
    apply_shift(X_synth, kind, intensity=2.0)
    pd.testing.assert_frame_equal(X_synth, before)


@pytest.mark.parametrize("kind", SHIFT_KINDS)
def test_zero_intensity_is_identity(X_synth, kind):
    # intensity = 0 -> hiçbir şey değişmemeli (kayma merdiveninin sıfır noktası).
    shifted = apply_shift(X_synth, kind, intensity=0.0)
    pd.testing.assert_frame_equal(shifted, X_synth)


@pytest.mark.parametrize("kind", SHIFT_KINDS)
def test_shift_preserves_shape_and_columns(X_synth, kind):
    shifted = apply_shift(X_synth, kind, intensity=1.0)
    assert shifted.shape == X_synth.shape
    assert list(shifted.columns) == list(X_synth.columns)
    assert list(shifted.index) == list(X_synth.index)


@pytest.mark.parametrize("kind", SHIFT_KINDS)
def test_shift_is_deterministic_given_a_seed(X_synth, kind):
    a = apply_shift(X_synth, kind, intensity=1.0, seed=123)
    b = apply_shift(X_synth, kind, intensity=1.0, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_produce_different_draws(X_synth):
    a = apply_shift(X_synth, "mixed", intensity=1.0, seed=1)
    b = apply_shift(X_synth, "mixed", intensity=1.0, seed=2)
    assert not np.allclose(a[LIMIT_COLUMN].values, b[LIMIT_COLUMN].values)


def test_seed_none_still_runs(X_synth):
    shifted = apply_shift(X_synth, "mixed", intensity=1.0, seed=None)
    assert shifted.shape == X_synth.shape


def test_age_shift_moves_the_mean_upwards(X_synth):
    shifted = shift_age(X_synth, intensity=1.0)
    assert shifted[AGE_COLUMN].mean() > X_synth[AGE_COLUMN].mean()
    # sadece yaş kolonu değişmeli
    untouched = [c for c in X_synth.columns if c != AGE_COLUMN]
    pd.testing.assert_frame_equal(shifted[untouched], X_synth[untouched])


def test_age_stays_within_bounds(X_synth):
    shifted = shift_age(X_synth, intensity=20.0)
    assert shifted[AGE_COLUMN].min() >= AGE_BOUNDS[0]
    assert shifted[AGE_COLUMN].max() <= AGE_BOUNDS[1]


def test_credit_limit_shift_inflates_limits(X_synth):
    shifted = shift_credit_limit(X_synth, intensity=1.0)
    assert shifted[LIMIT_COLUMN].mean() > X_synth[LIMIT_COLUMN].mean()
    assert (shifted[LIMIT_COLUMN] >= 0).all()


def test_payment_shift_shrinks_payments(X_synth):
    shifted = shift_payment_amount(X_synth, intensity=1.0)
    for column in PAYMENT_COLUMNS:
        assert shifted[column].mean() < X_synth[column].mean()
        assert (shifted[column] >= 0).all()


def test_mixed_shift_touches_every_family(X_synth):
    shifted = shift_mixed(X_synth, intensity=1.0)
    assert shifted[AGE_COLUMN].mean() > X_synth[AGE_COLUMN].mean()
    assert shifted[LIMIT_COLUMN].mean() > X_synth[LIMIT_COLUMN].mean()
    assert shifted["pay_amt1"].mean() < X_synth["pay_amt1"].mean()
    # kategorik kolonlara gürültü eklenmediğini doğrula
    assert set(np.unique(shifted["sex"])) <= set(np.unique(X_synth["sex"]))


def test_severity_is_monotonic(X_synth):
    # severe her zaman moderate'ten daha uzağa taşımalı.
    moderate = apply_shift(X_synth, "mixed", SEVERITY_LEVELS["moderate"])
    severe = apply_shift(X_synth, "mixed", SEVERITY_LEVELS["severe"])
    psi_moderate = shift_magnitude(X_synth, moderate)["psi"].mean()
    psi_severe = shift_magnitude(X_synth, severe)["psi"].mean()
    assert psi_severe > psi_moderate > 0.0


def test_constant_column_is_left_alone(X_synth):
    # std = 0 ise gürültü ekleme, atla (dağılımı bozmamak için)
    X = X_synth.copy()
    X["bill_amt1"] = 1000.0
    shifted = shift_mixed(X, intensity=1.0)
    assert (shifted["bill_amt1"] == 1000.0).all()


#invalid inputs
def test_apply_shift_unknown_kind(X_synth):
    with pytest.raises(ValueError, match="Unknown shift kind"):
        apply_shift(X_synth, "inflation")


@pytest.mark.parametrize("kind", SHIFT_KINDS)
def test_negative_intensity_rejected(X_synth, kind):
    with pytest.raises(ValueError, match="intensity"):
        apply_shift(X_synth, kind, intensity= -1.0)


@pytest.mark.parametrize("kind", SHIFT_KINDS)
def test_empty_frame_rejected(X_synth, kind):
    with pytest.raises(ValueError, match="empty"):
        apply_shift(X_synth.iloc[:0], kind)


@pytest.mark.parametrize("kind", SHIFT_KINDS)
def test_non_dataframe_rejected(X_synth, kind):
    with pytest.raises(TypeError, match="DataFrame"):
        apply_shift(X_synth.values, kind)


def test_missing_column_rejected(X_synth):
    with pytest.raises(ValueError, match="Missing columns"):
        shift_age(X_synth.drop(columns=[AGE_COLUMN]))
    with pytest.raises(ValueError, match="Missing columns"):
        shift_payment_amount(X_synth.drop(columns=["pay_amt3"]))


# dataset grid
def test_generate_shifted_datasets_grid(X_synth):
    datasets = generate_shifted_datasets(X_synth)
    expected = 1 + len(SHIFT_KINDS) * len(SEVERITY_LEVELS)
    assert len(datasets) == expected
    assert ORIGINAL_LABEL in datasets
    assert "mixed_severe" in datasets
    # "original" bir kopya olmalı, aynı nesne değil
    assert datasets[ORIGINAL_LABEL] is not X_synth
    pd.testing.assert_frame_equal(datasets[ORIGINAL_LABEL], X_synth)


def test_generate_shifted_datasets_without_original(X_synth):
    datasets = generate_shifted_datasets(
        X_synth, kinds=["age"], levels={"moderate": 1.0}, include_original=False
    )
    assert list(datasets) == ["age_moderate"]


def test_generate_shifted_datasets_requires_kinds_and_levels(X_synth):
    with pytest.raises(ValueError, match="shift kind"):
        generate_shifted_datasets(X_synth, kinds=[])
    with pytest.raises(ValueError, match="severity level"):
        generate_shifted_datasets(X_synth, levels={})





#magnitude metrics
def test_psi_is_zero_for_identical_samples():
    sample = np.random.default_rng(0).normal(size=1000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-12)


def test_psi_grows_with_distance():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    near = rng.normal(0.2, 1, 5000)
    far = rng.normal(2.0, 1, 5000)
    assert population_stability_index(reference, far) > population_stability_index(reference, near)
    assert population_stability_index(reference, far) > 0.25  # "major shift" eşiği


def test_psi_on_constant_reference_is_zero():
    assert population_stability_index(np.ones(100), np.arange(100)) == 0.0


def test_psi_invalid_inputs():
    with pytest.raises(ValueError, match="non-empty"): population_stability_index([], [1, 2, 3])
    with pytest.raises(ValueError, match="bins"): population_stability_index([1, 2, 3], [1, 2, 3], bins=1)


def test_normalised_wasserstein_behaviour():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 2000)
    assert normalised_wasserstein(reference, reference) == pytest.approx(0.0, abs=1e-12)
    shifted = reference + 1.0
    assert normalised_wasserstein(reference, shifted) == pytest.approx(1.0, rel=0.05)
    assert normalised_wasserstein(np.ones(10), np.arange(10)) == 0.0
    with pytest.raises(ValueError, match="non-empty"): normalised_wasserstein([], [1.0])


def test_shift_magnitude_report(X_synth):
    shifted = apply_shift(X_synth, "age", intensity=2.0)
    report = shift_magnitude(X_synth, shifted)
    assert list(report.columns) == ["feature", "psi", "wasserstein_norm", "mean_change_pct"]
    assert len(report) == X_synth.shape[1]
    assert report.iloc[0]["feature"] == AGE_COLUMN  # psi'ye göre azalan sıralı
    assert (report["psi"] >= 0).all()


def test_shift_magnitude_handles_zero_mean_feature(X_synth):
    X = X_synth.copy()
    X["bill_amt1"] = np.tile([-1.0, 1.0], len(X) // 2)  # ortalama tam 0
    report = shift_magnitude(X, X)
    row = report[report["feature"] == "bill_amt1"].iloc[0]
    assert np.isnan(row["mean_change_pct"])


def test_shift_magnitude_invalid_inputs(X_synth):
    with pytest.raises(ValueError, match="same columns"): shift_magnitude(X_synth, X_synth.drop(columns=[AGE_COLUMN]))
    with pytest.raises(ValueError, match="empty"): shift_magnitude(X_synth.iloc[:0], X_synth.iloc[:0])


@pytest.mark.parametrize("psi, label", [(0.05, "stable"), (0.15, "moderate"), (0.9, "major"), (float("nan"), "unknown")],)
def test_classify_shift(psi, label): assert classify_shift(psi) == label


def test_shift_summary_table(X_synth):
    datasets = generate_shifted_datasets(
        X_synth, kinds=["mixed"], levels=SEVERITY_LEVELS, include_original=False
    )
    summary = shift_summary(X_synth, datasets)
    assert list(summary["dataset"]) == ["mixed_moderate", "mixed_severe"]
    severe = summary[summary["dataset"] == "mixed_severe"].iloc[0]
    moderate = summary[summary["dataset"] == "mixed_moderate"].iloc[0]
    assert severe["mean_psi"] > moderate["mean_psi"]
    assert severe["severity_label"] in {"moderate", "major"}


def test_shift_summary_requires_datasets(X_synth):
    with pytest.raises(ValueError, match="No datasets"):
        shift_summary(X_synth, {})