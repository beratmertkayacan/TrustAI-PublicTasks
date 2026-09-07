"""Shared fixtures: small synthetic frames that mimic the real dataset schema.

The unit tests never touch the network or the 30k-row file; they exercise the
logic on tiny frames with the same columns and value ranges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explanation_drift.data import FEATURE_NAMES, TARGET_NAME


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
