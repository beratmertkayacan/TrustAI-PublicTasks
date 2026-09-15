"""İlk 10 değişmeden 11-23 arası tamamen karışabilir -> top_k_overlap bunu göremez, rank_disagreement görür.
Sıralama aynı kalıp bütün katkılar iki katına çıkabilir (model daha keskin karar veriyordur) -> sıralama metrikleri bunu göremez, distribution_shift görür.
pay_0'ın payı %30'dan %55'e çıkıp sıra değişmeyebilir -> importance_reallocation bunu yakalar.


Explanation drift metrics.

Modül sorusu: -model aynı kalırken açıklaması ne kadar kaydı?- 
Her metrik bir referans (orijinal test seti) ile bir kaymış ver setinin SHAP çıktısını karşılaştırır.

Four complementary views of the same question
---------------------------------------------
top_k_overlap: Aynı değişkenler hala ilk k'da mı? 
rank_correlation: Sıralamanın tamamı ne kadar korundu? (Spearman / Kendall)
distribution_shift: Katkıların dağılımı değişti mi? (Wasserstein + JS)
importance_reallocation:Önem ağırlığı değişkenler arasında ne kadar el değiştirdi?

All four are folded into a single 0-1 "explanation_drift_score" where 0 means
"explanation unchanged" and 1 means "nothing in common". Aynı ölçekte olmaları kasıtlı: performans tarafındaki "performance_drift_score" ile doğrudan karşılaştırılabilsinler diye.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, spearmanr

from .explain import DEFAULT_TOP_K, ExplanationResult
from .shift import normalised_wasserstein


# Bileşenlerin ağırlıkları. Hepsi 0-1 aralığında (1'e yakınsa fazla kayma)
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
    !!her iki seri de kendi değerine göre sıralı geldiği için ham ".values" karşılaştırması her zaman mükemmel korelasyon verirdi.
    isimle hizalama bunu önler
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
    # Tüm SHAP değerleri sıfırsa eşit dağılım varsay.
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

    Jaccard yerine |kesişim| / k kullan: iki küme aynı boyutta olduğundan
    bu değer doğrudan "ilk k'nın yüzde kaçı korundu" diye okuyoruz.
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
    (Tek değişkenli girdide korelasyon tanımsızdır ve "nan" döner)
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
        return 0.0 # iki dağılım da tek bir noktada yığılmış
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

    İki olasılık vektörü arasındaki L1 mesafesi [0, 2] aralığındadır; ikiye
    bölünce "önem ağırlığının yüzde kaçı el değiştirdi" olarak okunur.
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
        # Spearman [-1, 1] -> [0, 1]; tam ters sıralama 1.0 verir.
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
            continue # tanımsız bileşen skoru bozmasın, sadece dışarıda kalsın
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
    Pozitif değer açıklamanın performanstan daha çok bozulduğunu, yani SHAP'ın
    erken uyarı verdiğini gösterir. Negatif değer tersini söyler.
    """
    if np.isnan(explanation_drift) or np.isnan(performance_drift):
        return float("nan")
    return float(explanation_drift - performance_drift)


def early_warning_table(
    drift_scores: Mapping[str, float], performance_scores: Mapping[str, float]
) -> pd.DataFrame:
    """Side-by-side comparison of the two drift curves, with a verdict column."""
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
                "verdict": _verdict(index),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _verdict(index: float, tolerance: float = 0.05) -> str:
    """Label the gap between the two drift curves."""
    if np.isnan(index):
        return "unknown"
    if index > tolerance:
        return "explanation degrades first"
    if index < -tolerance:
        return "performance degrades first"
    return "degrade together"