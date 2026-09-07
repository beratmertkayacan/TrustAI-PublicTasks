"""Predictive performance metrics.

These answer the first half of the research question: *does the model still
predict well after the input distribution moves?* Explanation-side metrics live
in :mod:`explanation_drift.drift`.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Metrics reported for every (model, dataset) pair. All are higher-is-better
#: except ``brier``, which is an error and therefore lower-is-better.
PERFORMANCE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
)

HIGHER_IS_BETTER = {name: name != "brier" for name in PERFORMANCE_METRICS}


def _validate_inputs(y_true: Sequence, y_prob: Sequence) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            f"y_true/y_prob length mismatch: {y_true.shape[0]} vs {y_prob.shape[0]}"
        )
    if y_prob.ndim != 1:
        raise ValueError("y_prob must be a 1-D array of positive-class probabilities")
    if np.isnan(y_prob).any():
        raise ValueError("y_prob contains NaN values")
    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise ValueError("y_prob values must lie in [0, 1]")
    labels = set(np.unique(y_true).tolist())
    if not labels <= {0, 1}:
        raise ValueError(f"y_true must be binary 0/1, found: {sorted(labels)}")
    return y_true, y_prob


def compute_performance(
    y_true: Sequence,
    y_prob: Sequence,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the full performance dictionary for one (model, dataset) pair.

    Ranking metrics (``roc_auc``, ``pr_auc``) are undefined when ``y_true``
    contains a single class; they are returned as ``nan`` instead of raising,
    so a degenerate shifted subset does not abort a whole benchmark run.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    y_true, y_prob = _validate_inputs(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float("nan")
        if single_class
        else float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def performance_table(rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Build a tidy comparison table from ``compute_performance`` results.

    Every row is expected to carry at least ``model`` and ``dataset`` keys plus
    the performance metrics.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        raise ValueError("No performance rows supplied")
    for key in ("model", "dataset"):
        if key not in frame.columns:
            raise ValueError(f"Performance rows must contain a '{key}' column")
    ordered = ["model", "dataset"] + [
        m for m in PERFORMANCE_METRICS if m in frame.columns
    ]
    return frame[ordered].reset_index(drop=True)


def relative_degradation(
    reference: Mapping[str, float],
    shifted: Mapping[str, float],
    metrics: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    """Fractional loss of performance relative to the unshifted reference.

    A value of ``0.0`` means "as good as on the original test set", ``0.25``
    means a quarter of the reference score was lost. Sign is normalised so that
    positive always means *worse*, including for the Brier error.
    """
    metrics = list(metrics or PERFORMANCE_METRICS)
    out: dict[str, float] = {}
    for name in metrics:
        if name not in reference or name not in shifted:
            raise KeyError(f"Metric '{name}' missing from reference or shifted scores")
        ref, cur = float(reference[name]), float(shifted[name])
        if np.isnan(ref) or np.isnan(cur):
            out[name] = float("nan")
        elif ref == 0.0:
            out[name] = float("nan")
        elif HIGHER_IS_BETTER[name]:
            out[name] = (ref - cur) / abs(ref)
        else:
            out[name] = (cur - ref) / abs(ref)
    return out


def performance_drift_score(
    reference: Mapping[str, float],
    shifted: Mapping[str, float],
    metrics: Sequence[str] = ("roc_auc", "pr_auc", "balanced_accuracy"),
) -> float:
    """Single 0-1 summary of how much predictive performance was lost.

    The mean relative degradation over discrimination-oriented metrics, clipped
    to ``[0, 1]`` so it is directly comparable with the explanation drift score.
    """
    degradation = relative_degradation(reference, shifted, metrics)
    values = [v for v in degradation.values() if not np.isnan(v)]
    if not values:
        return float("nan")
    return float(np.clip(np.mean(values), 0.0, 1.0))
