"""Baseline and advanced model training.

Two models are trained on the *original* training split and never refitted:

* ``logistic_regression`` (baseline) - a linear, inherently transparent model.
  It consumes standardised features, so the frozen training scaler travels with
  the model.
* ``gradient_boosting`` (advanced) - a non-linear ensemble that consumes raw
  features. Class imbalance is handled with balanced sample weights, matching
  Task 3.

A :class:`TrainedModel` bundles the estimator with the preprocessing it needs,
so the rest of the pipeline can hand every model the same raw feature frame and
stay agnostic about scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .data import RANDOM_STATE, DataBundle, scale_features

BASELINE_NAME = "logistic_regression"
ADVANCED_NAME = "gradient_boosting"
MODEL_NAMES = (BASELINE_NAME, ADVANCED_NAME)

DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class TrainedModel:
    """A fitted estimator plus the preprocessing it expects.

    Attributes
    ----------
    name:
        Identifier used in tables and output filenames.
    estimator:
        The fitted scikit-learn estimator.
    scaler:
        Frozen ``StandardScaler`` fitted on the training split, or ``None`` when
        the estimator consumes raw features.
    """

    name: str
    estimator: BaseEstimator
    scaler: Optional[StandardScaler] = None

    @property
    def requires_scaling(self) -> bool:
        return self.scaler is not None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map raw features into the space the estimator was fitted on."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if self.scaler is None:
            return X
        return scale_features(self.scaler, X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities, shape ``(n_samples, 2)``."""
        return self.estimator.predict_proba(self.transform(X))

    def predict_positive_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class (default), shape ``(n_samples,)``."""
        return self.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
        """Hard labels obtained by thresholding the positive-class probability."""
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        return (self.predict_positive_proba(X) >= threshold).astype(int)


def _check_training_input(X: pd.DataFrame, y: pd.Series) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if len(X) == 0:
        raise ValueError("Cannot train on an empty frame")
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
    if len(set(pd.unique(y))) < 2:
        raise ValueError("Training target must contain both classes")


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scaler: StandardScaler,
    random_state: int = RANDOM_STATE,
) -> TrainedModel:
    """Fit the baseline logistic regression on standardised features."""
    _check_training_input(X_train, y_train)
    estimator = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=random_state
    )
    estimator.fit(scale_features(scaler, X_train), y_train)
    return TrainedModel(name=BASELINE_NAME, estimator=estimator, scaler=scaler)


def train_advanced(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> TrainedModel:
    """Fit the advanced gradient boosting model on raw features."""
    _check_training_input(X_train, y_train)
    estimator = GradientBoostingClassifier(random_state=random_state)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    estimator.fit(X_train, y_train, sample_weight=sample_weight)
    return TrainedModel(name=ADVANCED_NAME, estimator=estimator, scaler=None)


def train_models(
    bundle: DataBundle, random_state: int = RANDOM_STATE
) -> dict[str, TrainedModel]:
    """Train both models from a :class:`~explanation_drift.data.DataBundle`."""
    return {
        BASELINE_NAME: train_baseline(
            bundle.X_train, bundle.y_train, bundle.scaler, random_state
        ),
        ADVANCED_NAME: train_advanced(bundle.X_train, bundle.y_train, random_state),
    }


def save_model(model: TrainedModel, directory: Union[str, Path]) -> Path:
    """Persist a trained model (estimator + scaler) as a joblib file."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model.name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(path: Union[str, Path]) -> TrainedModel:
    """Load a model previously written by :func:`save_model`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, TrainedModel):
        raise TypeError(f"{path} does not contain a TrainedModel")
    return model
