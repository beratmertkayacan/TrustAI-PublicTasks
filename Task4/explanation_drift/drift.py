"""If the top 10 stay the same but ranks 11-23 get shuffled, top_k_overlap misses it and rank_disagreement catches it.
If the ranking stays the same but all contributions double, the ranking metrics miss it and distribution_shift catches it.
If pay_0 grows from 30% to 55% of the total without changing its rank, importance_reallocation catches it.


Explanation drift metrics.

Module question: how much did the explanation move while the model stayed the same?
Each metric compares the SHAP output of a reference (the original test set) with a shifted data set.

Four complementary views of the same question
---------------------------------------------
top_k_overlap: are the same features still in the top k?
rank_correlation: how much of the full ranking is kept? (Spearman / Kendall)
distribution_shift: did the shape of the contributions change? (Wasserstein + JS)
importance_reallocation: how much importance moved between features?

All four are folded into a single 0-1 "explanation_drift_score" where 0 means
"explanation unchanged" and 1 means "nothing in common". The same 0-1 scale is used on purpose, so this score can be compared directly with "performance_drift_score".
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, spearmanr

from .explain import DEFAULT_TOP_K, ExplanationResult
from .shift import normalised_wasserstein


# Component weights. All components are in the 0-1 range (close to 1 means more drift).
DRIFT_COMPONENTS = (
    "top_k_overlap_loss",
    "rank_disagreement",
    "distribution_shift",
    "importance_reallocation",
)

DEFAULT_WEIGHTS = {name: 0.25 for name in DRIFT_COMPONENTS}

HISTOGRAM_BINS = 20


#helpers
def _align(reference: pd.Series, shifted: pd.Series) -> tuple[pd.Series, pd.Series]:
    """put two importance series on the same feature order
    Both series arrive sorted by their own values, so comparing raw ".values" would always
    give a perfect correlation. Aligning by feature name prevents that.
    """

    if not isinstance(reference, pd.Series) or not isinstance(shifted, pd.Series):
        raise TypeError("Importances must be pandas Series indexed by feature name")
    if set(reference.index) != set(shifted.index):
        missing = set(reference.index) ^ set(shifted.index)
        raise ValueError(f"Feature sets differ between the two explanations: {missing}")
    ordered = reference.sort_index()
    return ordered, shifted.reindex(ordered.index)


def _normalise(importance: pd.Series) -> pd.Series:
    """Turn an importance vector into a distribution summing to 1"""
    total = float(importance.abs().sum())
    # All SHAP values are zero, so assume an equal split.
    if total == 0.0:
        return pd.Series(1.0 / len(importance), index=importance.index)
    return importance.abs() / total



# 1- Top-k overlap
def jaccard_index(left: Sequence[str], right: Sequence[str]) -> float:
    """|A n B| / |A u B| for two feature name collections."""
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def top_k_overlap(
    reference: ExplanationResult, shifted: ExplanationResult, k: int = DEFAULT_TOP_K
) -> float:
    """Fraction of the reference top-k features still present in the shifted top-k.

    Use |intersection| / k instead of Jaccard: both sets have the same size, so the value
    reads directly as "what share of the top k was kept".
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    reference_top = reference.top_k(k)
    shifted_top = shifted.top_k(k)
    return len(set(reference_top) & set(shifted_top)) / len(reference_top)


# 2- Ranking correlation
def rank_correlation(
    reference: pd.Series, shifted: pd.Series, method: str = "spearman"
) -> float:
    """Correlation between two importance rankings, aligned by feature name.

    Returns a value in "[-1, 1]"; 1 = identical ordering, 0 = unrelated, -1 = exactly reversed. 
    (With a single feature the correlation is undefined and "nan" is returned.)
    """
    reference, shifted = _align(reference, shifted)
    if len(reference) < 2:
        return float("nan")
    if method == "spearman":
        statistic = spearmanr(reference.values, shifted.values).statistic
    elif method == "kendall":
        statistic = kendalltau(reference.values, shifted.values).statistic
    else:
        raise ValueError(f"Unknown method '{method}', use 'spearman' or 'kendall'")
    return float(statistic)

# 3- SHAP value distribution comparison
def _js_divergence(reference: np.ndarray, actual: np.ndarray, bins: int) -> float:
    """Jensen-Shannon distance between two samples, binned on a shared grid."""
    low = min(reference.min(), actual.min())
    high = max(reference.max(), actual.max())
    if low == high: 
        return 0.0 # both distributions sit on a single point
    edges = np.linspace(low, high, bins + 1)
    p = np.histogram(reference, edges)[0] / reference.size
    q = np.histogram(actual, edges)[0] / actual.size
    distance = jensenshannon(p, q, base=2)
    return 0.0 if np.isnan(distance) else float(distance)


def shap_distribution_shift(
    reference: ExplanationResult,
    shifted: ExplanationResult,
    bins: int = HISTOGRAM_BINS,
) -> pd.DataFrame:
    """Per-feature comparison of the two SHAP value distributions."""
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    if list(reference.feature_names) != list(shifted.feature_names):
        raise ValueError("Explanations must describe the same features")

    rows = []
    for feature in reference.feature_names:
        ref = reference.shap_values[feature].values
        act = shifted.shap_values[feature].values
        rows.append(
            {
                "feature": feature,
                "wasserstein_norm": normalised_wasserstein(ref, act),
                "js_distance": _js_divergence(ref, act, bins),
                "mean_abs_reference": float(np.abs(ref).mean()),
                "mean_abs_shifted": float(np.abs(act).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("js_distance", ascending=False).reset_index(
        drop=True
    )

# 4- Importance reallocation
def importance_reallocation(reference: pd.Series, shifted: pd.Series) -> float:
    """Half the L1 distance between the two normalised importance vectors.

    The L1 distance between two probability vectors is in [0, 2]; dividing it by two
    gives "what share of the importance moved".
    """
    reference, shifted = _align(reference, shifted)
    p, q = _normalise(reference), _normalise(shifted)
    return float(np.abs(p.values - q.values).sum() / 2.0)


# composite score
def drift_components(
    reference: ExplanationResult,
    shifted: ExplanationResult,
    k: int = DEFAULT_TOP_K,
    bins: int = HISTOGRAM_BINS,
) -> dict[str, float]:
    """The four bounded 0-1 components, all oriented so that higher = more drift."""
    reference_importance = reference.global_importance()
    shifted_importance = shifted.global_importance()

    correlation = rank_correlation(reference_importance, shifted_importance)
    distribution = shap_distribution_shift(reference, shifted, bins=bins)

    return {
        "top_k_overlap_loss": 1.0 - top_k_overlap(reference, shifted, k),
        # Spearman [-1, 1] -> [0, 1]; a fully reversed ranking gives 1.0.
        "rank_disagreement": float("nan")
        if np.isnan(correlation)
        else (1.0 - correlation) / 2.0,
        "distribution_shift": float(distribution["js_distance"].mean()),
        "importance_reallocation": importance_reallocation(
            reference_importance, shifted_importance
        ),
    }


def explanation_drift_score(
    reference: ExplanationResult,
    shifted: ExplanationResult,
    k: int = DEFAULT_TOP_K,
    weights: Optional[Mapping[str, float]] = None,
    bins: int = HISTOGRAM_BINS,
) -> float:
    """Single 0-1 summary of how far the explanation moved."""
    components = drift_components(reference, shifted, k=k, bins=bins)
    return combine_components(components, weights)


def combine_components(
    components: Mapping[str, float], weights: Optional[Mapping[str, float]] = None
) -> float:
    """Weighted mean of the drift components, ignoring "nan" entries."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    unknown = set(weights) - set(DRIFT_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown drift components in weights: {sorted(unknown)}")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Weights must sum to a positive number")

    accumulated, used = 0.0, 0.0
    for name, weight in weights.items():
        value = components.get(name, float("nan"))
        if np.isnan(value):
            continue # skip an undefined component instead of letting it break the score
        accumulated += weight * value
        used += weight
    if used == 0.0:
        return float("nan")
    return float(np.clip(accumulated / used, 0.0, 1.0))


def drift_table(
    results: Mapping[str, ExplanationResult],
    reference_key: str = "original",
    k: int = DEFAULT_TOP_K,
    weights: Optional[Mapping[str, float]] = None,
    bins: int = HISTOGRAM_BINS,
) -> pd.DataFrame:
    """One row per dataset: every component plus the composite drift score."""
    if reference_key not in results:
        raise ValueError(f"Reference dataset '{reference_key}' missing from results")
    reference = results[reference_key]

    rows = []
    for name, result in results.items():
        components = drift_components(reference, result, k=k, bins=bins)
        rows.append(
            {
                "dataset": name,
                "model": result.model_name,
                **components,
                "explanation_drift_score": combine_components(components, weights),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)

# which degrades first? main research question

def early_warning_index(explanation_drift: float, performance_drift: float) -> float:
    """  "explanation_drift - performance_drift"
    A positive value means the explanation moved more than the performance, so SHAP
    gives an early warning. A negative value means the opposite.
    """
    if np.isnan(explanation_drift) or np.isnan(performance_drift):
        return float("nan")
    return float(explanation_drift - performance_drift)


def early_warning_table(
    drift_scores: Mapping[str, float], performance_scores: Mapping[str, float]
) -> pd.DataFrame:
    """Side-by-side comparison of the two drift curves.

    This table only holds numbers. The decision of what counts as drift needs a
    threshold, and that threshold is estimated from data in
    :mod:`explanation_drift.thresholds`.
    """
    missing = set(drift_scores) ^ set(performance_scores)
    if missing:
        raise ValueError(f"Datasets must match on both sides, got mismatch: {missing}")
    if not drift_scores:
        raise ValueError("No datasets supplied")

    rows = []
    for name in drift_scores:
        explanation = float(drift_scores[name])
        performance = float(performance_scores[name])
        index = early_warning_index(explanation, performance)
        rows.append(
            {
                "dataset": name,
                "explanation_drift": explanation,
                "performance_drift": performance,
                "early_warning_index": index,
                "drift_ratio": float("nan") if performance == 0.0 else explanation / performance,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)
