"""Data driven thresholds for the early warning check.

A drift score is never exactly zero, even when there is no shift at all: it also
moves because the test set is a finite sample. Comparing two scores against one
fixed number (0.05, for example) ignores that, and the two scores do not even
have the same amount of noise.

This module builds a null distribution for each score. It resamples customers
from the unshifted test set, measures the score between two independent
resamples, and repeats that many times. The threshold is a high quantile of that
distribution, so a value above it is unlikely to come from sampling noise alone
(5 percent at the default quantile).

The same resampling is used for a confidence interval on the gap between the two
scores, which tells us whether the explanation really moved more than the
performance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import RANDOM_STATE
from .drift import explanation_drift_score
from .explain import DEFAULT_TOP_K, ExplanationResult
from .metrics import compute_performance, performance_drift_score

BOOTSTRAP_SAMPLES = 200
NULL_QUANTILE = 0.95
CONFIDENCE_LEVEL = 0.95

#: Labels produced by DriftThresholds.classify.
EXPLANATION_ONLY = "explanation drift only"
PERFORMANCE_ONLY = "performance drift only"
BOTH = "both above threshold"
NEITHER = "no drift detected"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class DriftThresholds:
    """One noise threshold per score, estimated from unshifted data."""

    explanation: float
    performance: float
    quantile: float
    n_samples: int

    def classify(self, explanation_drift: float, performance_drift: float) -> str:
        """Compare each score with its own threshold."""
        if np.isnan(explanation_drift) or np.isnan(performance_drift):
            return UNKNOWN
        explanation_above = explanation_drift > self.explanation
        performance_above = performance_drift > self.performance
        if explanation_above and not performance_above:
            return EXPLANATION_ONLY
        if performance_above and not explanation_above:
            return PERFORMANCE_ONLY
        if explanation_above and performance_above:
            return BOTH
        return NEITHER


def _quiet_performance(y_true: Sequence, y_prob: Sequence) -> dict[str, float]:
    """Performance scores for one resample, without the degenerate-sample warning.

    A bootstrap resample can end up with a single class. sklearn warns about
    that, and the metrics already return nan for it, so the warning adds nothing
    here.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true"
        )
        warnings.filterwarnings("ignore", message="A single label was found")
        return compute_performance(y_true, y_prob)


def _check_inputs(n_samples: int, quantile: float) -> None:
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")


def subset_explanation(result: ExplanationResult, positions: Sequence[int]) -> ExplanationResult:
    """Take the given rows out of an explanation, keeping everything else."""
    positions = np.asarray(positions, dtype=int)
    if positions.size == 0:
        raise ValueError("Cannot take an empty subset of an explanation")
    return ExplanationResult(
        dataset=result.dataset,
        model_name=result.model_name,
        shap_values=result.shap_values.iloc[positions].reset_index(drop=True),
        base_value=result.base_value,
    )


def null_distributions(
    reference: ExplanationResult,
    y_true: Sequence,
    y_prob: Sequence,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_STATE,
    k: int = DEFAULT_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Drift scores between two resamples of the same unshifted data.

    Returns the explanation drift values and the performance drift values. Both
    describe what the scores do when nothing but the sample changes.
    """
    _check_inputs(n_samples, NULL_QUANTILE)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n_rows = reference.n_samples
    if len(y_true) != n_rows or len(y_prob) != n_rows:
        raise ValueError(
            f"y_true/y_prob must have {n_rows} rows to match the explanation"
        )

    rng = np.random.default_rng(seed)
    explanation_values: list[float] = []
    performance_values: list[float] = []

    for _ in range(n_samples):
        first = rng.integers(0, n_rows, n_rows)
        second = rng.integers(0, n_rows, n_rows)
        explanation_values.append(
            explanation_drift_score(
                subset_explanation(reference, first),
                subset_explanation(reference, second),
                k=k,
            )
        )
        scores_first = _quiet_performance(y_true[first], y_prob[first])
        scores_second = _quiet_performance(y_true[second], y_prob[second])
        performance_values.append(performance_drift_score(scores_first, scores_second))

    return np.asarray(explanation_values), np.asarray(performance_values)


def quantile_threshold(values: Sequence[float], quantile: float = NULL_QUANTILE) -> float:
    """High quantile of a null distribution, ignoring undefined values."""
    _check_inputs(2, quantile)
    clean = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if clean.size == 0:
        raise ValueError("Null distribution has no usable values")
    return float(np.quantile(clean, quantile))


def estimate_thresholds(
    reference: ExplanationResult,
    y_true: Sequence,
    y_prob: Sequence,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_STATE,
    k: int = DEFAULT_TOP_K,
    quantile: float = NULL_QUANTILE,
) -> DriftThresholds:
    """Estimate one threshold per score from unshifted data."""
    _check_inputs(n_samples, quantile)
    explanation_values, performance_values = null_distributions(
        reference, y_true, y_prob, n_samples=n_samples, seed=seed, k=k
    )
    return DriftThresholds(
        explanation=quantile_threshold(explanation_values, quantile),
        performance=quantile_threshold(performance_values, quantile),
        quantile=quantile,
        n_samples=n_samples,
    )


def bootstrap_gap_interval(
    reference: ExplanationResult,
    shifted: ExplanationResult,
    y_true: Sequence,
    prob_reference: Sequence,
    prob_shifted: Sequence,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_STATE,
    k: int = DEFAULT_TOP_K,
    level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Percentile interval for (explanation drift - performance drift).

    An interval fully above zero means the explanation moved more than the
    performance, and that this is not explained by the sample alone.
    """
    _check_inputs(n_samples, level)
    if reference.n_samples != shifted.n_samples:
        raise ValueError("Reference and shifted explanations must have the same rows")
    y_true = np.asarray(y_true)
    prob_reference = np.asarray(prob_reference, dtype=float)
    prob_shifted = np.asarray(prob_shifted, dtype=float)

    rng = np.random.default_rng(seed)
    n_rows = reference.n_samples
    gaps: list[float] = []

    for _ in range(n_samples):
        rows = rng.integers(0, n_rows, n_rows)
        explanation = explanation_drift_score(
            subset_explanation(reference, rows), subset_explanation(shifted, rows), k=k
        )
        performance = performance_drift_score(
            _quiet_performance(y_true[rows], prob_reference[rows]),
            _quiet_performance(y_true[rows], prob_shifted[rows]),
        )
        if not np.isnan(explanation) and not np.isnan(performance):
            gaps.append(explanation - performance)

    if not gaps:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return (
        float(np.quantile(gaps, tail)),
        float(np.quantile(gaps, 1.0 - tail)),
    )


def threshold_table(thresholds: Mapping[str, DriftThresholds]) -> pd.DataFrame:
    """One row per model, so the thresholds can be stored next to the results."""
    if not thresholds:
        raise ValueError("No thresholds supplied")
    rows = [
        {
            "model": model,
            "explanation_threshold": value.explanation,
            "performance_threshold": value.performance,
            "quantile": value.quantile,
            "bootstrap_samples": value.n_samples,
        }
        for model, value in thresholds.items()
    ]
    return pd.DataFrame(rows).reset_index(drop=True)
