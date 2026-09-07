"""Tests for explanation_drift.data (loading, validation, splitting, scaling)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from explanation_drift import data as data_mod
from explanation_drift.data import (
    FEATURE_NAMES,
    TARGET_NAME,
    DataBundle,
    fit_scaler,
    load_dataset,
    load_raw,
    scale_features,
    split_data,
    validate_dataset,
)

from .conftest import make_credit_frame, make_target


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_validate_dataset_accepts_clean_input(X_synth, y_synth):
    assert validate_dataset(X_synth, y_synth) is None


@pytest.mark.parametrize(
    "bad_X, bad_y, message",
    [
        ("not a frame", None, "DataFrame"),
        (None, "not a series", "Series"),
    ],
)
def test_validate_dataset_rejects_wrong_types(X_synth, y_synth, bad_X, bad_y, message):
    X = X_synth if bad_X is None else bad_X
    y = y_synth if bad_y is None else bad_y
    with pytest.raises(TypeError, match=message):
        validate_dataset(X, y)


def test_validate_dataset_rejects_empty_frame(y_synth):
    with pytest.raises(ValueError, match="empty"):
        validate_dataset(pd.DataFrame(columns=FEATURE_NAMES), y_synth)


def test_validate_dataset_rejects_length_mismatch(X_synth, y_synth):
    with pytest.raises(ValueError, match="length mismatch"):
        validate_dataset(X_synth, y_synth.iloc[:-5])


def test_validate_dataset_rejects_nan_features(X_synth, y_synth):
    X = X_synth.copy()
    X.loc[X.index[0], "age"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_dataset(X, y_synth)


def test_validate_dataset_rejects_nan_target(X_synth, y_synth):
    y = y_synth.astype(float).copy()
    y.iloc[0] = np.nan
    with pytest.raises(ValueError, match="y contains NaN"):
        validate_dataset(X_synth, y)


def test_validate_dataset_rejects_non_numeric_features(X_synth, y_synth):
    X = X_synth.copy()
    X["age"] = "old"
    with pytest.raises(ValueError, match="Non-numeric"):
        validate_dataset(X, y_synth)


def test_validate_dataset_rejects_non_binary_target(X_synth, y_synth):
    y = y_synth.copy()
    y.iloc[0] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_dataset(X_synth, y)


def test_validate_dataset_rejects_single_class_target(X_synth, y_synth):
    y = pd.Series(np.zeros(len(X_synth), dtype=int), index=X_synth.index, name=TARGET_NAME)
    with pytest.raises(ValueError, match="both classes"):
        validate_dataset(X_synth, y)


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def test_split_data_shapes_and_stratification(X_synth, y_synth):
    X_train, X_test, y_train, y_test = split_data(X_synth, y_synth, test_size=0.25)
    assert len(X_train) == 150 and len(X_test) == 50
    assert list(X_train.columns) == FEATURE_NAMES
    # stratified: class balance preserved within one percentage point
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_data_is_reproducible(X_synth, y_synth):
    first = split_data(X_synth, y_synth, random_state=7)[1]
    second = split_data(X_synth, y_synth, random_state=7)[1]
    pd.testing.assert_frame_equal(first, second)


def test_split_data_different_seed_changes_split(X_synth, y_synth):
    a = split_data(X_synth, y_synth, random_state=1)[1]
    b = split_data(X_synth, y_synth, random_state=2)[1]
    assert list(a.index) != list(b.index)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.2, 1.5])
def test_split_data_rejects_invalid_test_size(X_synth, y_synth, test_size):
    with pytest.raises(ValueError, match="test_size"):
        split_data(X_synth, y_synth, test_size=test_size)


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #
def test_fit_scaler_standardises_training_data(X_synth):
    scaled = scale_features(fit_scaler(X_synth), X_synth)
    assert np.allclose(scaled.mean().values, 0.0, atol=1e-9)
    assert np.allclose(scaled.std(ddof=0).values, 1.0, atol=1e-9)


def test_scale_features_preserves_index_and_columns(X_synth):
    subset = X_synth.iloc[10:20]
    scaled = scale_features(fit_scaler(X_synth), subset)
    assert list(scaled.columns) == FEATURE_NAMES
    assert list(scaled.index) == list(subset.index)


def test_fit_scaler_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        fit_scaler(pd.DataFrame(columns=FEATURE_NAMES))


def test_scale_features_rejects_column_mismatch(X_synth):
    scaler = fit_scaler(X_synth)
    with pytest.raises(ValueError, match="Column mismatch"):
        scale_features(scaler, X_synth[FEATURE_NAMES[::-1]])


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def test_load_raw_from_local_excel(monkeypatch, tmp_path):
    """The Excel reader is monkeypatched so the test needs no real .xls file."""
    frame = make_credit_frame(50)
    frame.columns = [c.upper() if c != "pay_0" else "PAY_0" for c in frame.columns]
    frame["default payment next month"] = make_target(make_credit_frame(50)).values
    frame.insert(0, "ID", range(1, 51))

    monkeypatch.setattr(data_mod.pd, "read_excel", lambda *a, **k: frame)
    path = tmp_path / "dataset.xls"
    path.write_bytes(b"stub")

    X, y = load_raw(local_path=path)
    assert list(X.columns) == FEATURE_NAMES
    assert y.name == TARGET_NAME
    assert len(X) == 50


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(local_path=tmp_path / "nope.xls")


def test_load_raw_local_excel_without_target(monkeypatch, tmp_path):
    frame = make_credit_frame(20)
    monkeypatch.setattr(data_mod.pd, "read_excel", lambda *a, **k: frame)
    path = tmp_path / "dataset.xls"
    path.write_bytes(b"stub")
    with pytest.raises(ValueError, match="default payment next month"):
        load_raw(local_path=path)


def test_load_raw_local_excel_missing_features(monkeypatch, tmp_path):
    frame = make_credit_frame(20).drop(columns=["age"])
    frame["default payment next month"] = 0
    frame.loc[frame.index[:5], "default payment next month"] = 1
    monkeypatch.setattr(data_mod.pd, "read_excel", lambda *a, **k: frame)
    path = tmp_path / "dataset.xls"
    path.write_bytes(b"stub")
    with pytest.raises(ValueError, match="Missing feature columns"):
        load_raw(local_path=path)


def test_load_raw_from_openml_is_renamed(monkeypatch):
    frame = make_credit_frame(30)
    inverse = {v: k for k, v in data_mod.OPENML_COLUMN_MAP.items()}
    openml_frame = frame.rename(columns=inverse)
    target = make_target(frame).astype(str)

    bunch = SimpleNamespace(data=openml_frame, target=target)
    monkeypatch.setattr(data_mod, "fetch_openml", lambda **kwargs: bunch)
    X, y = load_raw()
    assert list(X.columns) == FEATURE_NAMES
    assert y.dtype.kind == "i"


def test_load_raw_from_openml_missing_columns(monkeypatch):
    frame = make_credit_frame(30)
    bunch = SimpleNamespace(
        data=frame.rename(columns={"age": "x99"}),
        target=make_target(frame).astype(str),
    )
    monkeypatch.setattr(data_mod, "fetch_openml", lambda **kwargs: bunch)
    with pytest.raises(ValueError, match="missing columns"):
        load_raw()


def test_load_dataset_returns_bundle(monkeypatch, X_synth, y_synth):
    monkeypatch.setattr(data_mod, "load_raw", lambda **kwargs: (X_synth, y_synth))
    bundle = load_dataset(test_size=0.2)
    assert isinstance(bundle, DataBundle)
    assert bundle.feature_names == FEATURE_NAMES
    assert len(bundle.X_test) == 40
    scaled = bundle.scale(bundle.X_test)
    assert scaled.shape == bundle.X_test.shape
    # scaler was fitted on train only: test means are not exactly zero
    assert abs(scaled.mean().mean()) < 0.5
