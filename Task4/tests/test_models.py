"""Tests for explanation_drift.models (training, prediction, persistence)."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from explanation_drift.data import DataBundle, fit_scaler
from explanation_drift.models import (
    ADVANCED_NAME,
    BASELINE_NAME,
    MODEL_NAMES,
    TrainedModel,
    load_model,
    save_model,
    train_advanced,
    train_baseline,
    train_models,
)


@pytest.fixture
def bundle(X_synth, y_synth) -> DataBundle:
    """Small in-memory bundle: 150 train / 50 test rows."""
    X_train, X_test = X_synth.iloc[:150], X_synth.iloc[150:]
    y_train, y_test = y_synth.iloc[:150], y_synth.iloc[150:]
    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=fit_scaler(X_train),
    )


@pytest.fixture
def baseline(bundle) -> TrainedModel:
    return train_baseline(bundle.X_train, bundle.y_train, bundle.scaler)


@pytest.fixture
def advanced(bundle) -> TrainedModel:
    return train_advanced(bundle.X_train, bundle.y_train)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def test_train_models_returns_both_models(bundle):
    models = train_models(bundle)
    assert set(models) == set(MODEL_NAMES)
    assert models[BASELINE_NAME].requires_scaling
    assert not models[ADVANCED_NAME].requires_scaling


def test_baseline_is_reproducible(bundle):
    first = train_baseline(bundle.X_train, bundle.y_train, bundle.scaler)
    second = train_baseline(bundle.X_train, bundle.y_train, bundle.scaler)
    np.testing.assert_allclose(first.estimator.coef_, second.estimator.coef_)


def test_advanced_is_reproducible(bundle):
    a = train_advanced(bundle.X_train, bundle.y_train)
    b = train_advanced(bundle.X_train, bundle.y_train)
    np.testing.assert_allclose(
        a.predict_positive_proba(bundle.X_test), b.predict_positive_proba(bundle.X_test)
    )


@pytest.mark.parametrize("trainer", ["baseline", "advanced"])
def test_training_rejects_empty_frame(bundle, trainer):
    empty_X = bundle.X_train.iloc[:0]
    empty_y = bundle.y_train.iloc[:0]
    with pytest.raises(ValueError, match="empty"):
        if trainer == "baseline":
            train_baseline(empty_X, empty_y, bundle.scaler)
        else:
            train_advanced(empty_X, empty_y)


def test_training_rejects_length_mismatch(bundle):
    with pytest.raises(ValueError, match="length mismatch"):
        train_advanced(bundle.X_train, bundle.y_train.iloc[:-1])


def test_training_rejects_single_class_target(bundle):
    y = pd.Series(np.ones(len(bundle.X_train), dtype=int), index=bundle.X_train.index)
    with pytest.raises(ValueError, match="both classes"):
        train_advanced(bundle.X_train, y)


def test_training_rejects_non_dataframe(bundle):
    with pytest.raises(TypeError, match="DataFrame"):
        train_advanced(bundle.X_train.values, bundle.y_train)


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
def test_predict_proba_shape_and_range(baseline, advanced, bundle):
    for model in (baseline, advanced):
        proba = model.predict_proba(bundle.X_test)
        assert proba.shape == (len(bundle.X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        positive = model.predict_positive_proba(bundle.X_test)
        assert positive.shape == (len(bundle.X_test),)
        assert positive.min() >= 0.0 and positive.max() <= 1.0


def test_predict_thresholding(baseline, bundle):
    low = baseline.predict(bundle.X_test, threshold=0.2).sum()
    high = baseline.predict(bundle.X_test, threshold=0.8).sum()
    assert low >= high
    assert set(np.unique(baseline.predict(bundle.X_test))) <= {0, 1}


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.2])
def test_predict_rejects_invalid_threshold(baseline, bundle, threshold):
    with pytest.raises(ValueError, match="threshold"):
        baseline.predict(bundle.X_test, threshold=threshold)


def test_transform_scales_only_when_needed(baseline, advanced, bundle):
    scaled = baseline.transform(bundle.X_test)
    assert not np.allclose(scaled.values, bundle.X_test.values)
    assert list(scaled.columns) == list(bundle.X_test.columns)
    pd.testing.assert_frame_equal(advanced.transform(bundle.X_test), bundle.X_test)


def test_transform_rejects_non_dataframe(baseline, bundle):
    with pytest.raises(TypeError, match="DataFrame"):
        baseline.transform(bundle.X_test.values)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_save_and_load_model_roundtrip(baseline, bundle, tmp_path):
    path = save_model(baseline, tmp_path / "models")
    assert path.exists()
    reloaded = load_model(path)
    np.testing.assert_allclose(
        reloaded.predict_positive_proba(bundle.X_test),
        baseline.predict_positive_proba(bundle.X_test),
    )


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.joblib")


def test_load_model_wrong_payload(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(TypeError, match="TrainedModel"):
        load_model(path)
