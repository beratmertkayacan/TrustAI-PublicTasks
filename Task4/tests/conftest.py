"""Shared fixtures: small synthetic frames that mimic the real dataset schema.

The unit tests never touch the network or the 30k-row file; they exercise the
logic on tiny frames with the same columns and value ranges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explanation_drift.data import FEATURE_NAMES, TARGET_NAME, DataBundle, fit_scaler
from explanation_drift.models import TrainedModel, train_advanced, train_baseline


def make_credit_frame(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the same 23 columns as the UCI dataset."""
    rng = np.random.default_rng(seed)
    data = {
        "limit_bal": rng.integers(10_000, 500_000, n_rows).astype(float),
        "sex": rng.integers(1, 3, n_rows).astype(float),
        "education": rng.integers(1, 5, n_rows).astype(float),
        "marriage": rng.integers(1, 4, n_rows).astype(float),
        "age": rng.integers(21, 70, n_rows).astype(float),
    }
    for col in ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]:
        data[col] = rng.integers(-2, 4, n_rows).astype(float)
    for i in range(1, 7):
        data[f"bill_amt{i}"] = rng.normal(50_000, 30_000, n_rows).round(0)
        data[f"pay_amt{i}"] = rng.gamma(2.0, 3_000.0, n_rows).round(0)
    return pd.DataFrame(data)[FEATURE_NAMES]


def make_target(X: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Binary target that genuinely depends on pay_0 and limit_bal."""
    rng = np.random.default_rng(seed + 1)
    logit = 0.8 * X["pay_0"] - 1.2e-5 * X["limit_bal"] + 0.02 * X["age"] - 1.0
    prob = 1.0 / (1.0 + np.exp(-logit))
    return pd.Series((rng.random(len(X)) < prob).astype(int), index=X.index, name=TARGET_NAME)


@pytest.fixture
def X_synth() -> pd.DataFrame:
    return make_credit_frame()


@pytest.fixture
def y_synth(X_synth) -> pd.Series:
    return make_target(X_synth)


@pytest.fixture
def bundle(X_synth, y_synth) -> DataBundle:
    """Small in-memory bundle: 150 train / 50 test rows.

    Üç test modülü de buna ihtiyaç duyuyordu; conftest'te tek kopya tutmak
    tanımların birbirinden sessizce ayrışmasını engelliyor.
    """
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
    """Fitted logistic regression on the synthetic bundle."""
    return train_baseline(bundle.X_train, bundle.y_train, bundle.scaler)


@pytest.fixture
def advanced(bundle) -> TrainedModel:
    """Fitted gradient boosting on the synthetic bundle."""
    return train_advanced(bundle.X_train, bundle.y_train)