"""SHAP explanation generation for a frozen model across many datasets.

1. Exact explainers only. LogisticRegression is explained with "LinearExplainer" 
GradientBoosting with "TreeExplainer"; both are analytic, not sampled.
(Örnekleme tabanlı bir explainer kullansaydık ölçtüğümüz drift'in bir kısmı Monte Carlo gürültüsü olurdu)

2. Frozen background. The reference distribution handed to the explainer is
always the original training data (never the shifted frame being explained)
(SHAP değerleri "ortalama bir müşteriye göre fark" demektir; referansı kaymış veriyle birlikte kaydırırsak kaymayı kendi elimizle silmiş oluruz)

3. Same rows everywhere. When subsampling, the row indices are drawn once
and reused for every dataset, so differences between datasets come from the
shift alone and not from which customers happened to be sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.base import is_classifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .data import RANDOM_STATE
from .models import TrainedModel

DEFAULT_TOP_K = 10

# Estimator families we know how to explain exactly.
LINEAR_ESTIMATORS = (LogisticRegression,)
TREE_ESTIMATORS = (GradientBoostingClassifier, RandomForestClassifier)


@dataclass(frozen=True)
class ExplanationResult:
    """SHAP output for one model-dataset pair. 
    "shap_values" is a frame of signed contributions with the same shape and
    columns as the explained features; "base_value" is the explainer's
    expected model output (log-odds / margin space)"""

    dataset: str
    model_name: str
    shap_values: pd.DataFrame
    base_value: float

    @property
    def feature_names(self) -> list[str]:
        return list(self.shap_values.columns)

    @property
    def n_samples(self) -> int:
        return len(self.shap_values)

    def global_importance(self) -> pd.Series:
        # Mean absolute SHAP value per feature, sorted descending.
        return global_importance(self.shap_values)

    def ranking(self) -> list[str]:
        # Feature names ordered from most to least important.
        return list(self.global_importance().index)

    def top_k(self, k: int = DEFAULT_TOP_K) -> list[str]:
        # The ``k`` most important features.
        return top_k_features(self.global_importance(), k)





#Explainer construction

def build_explainer(model: TrainedModel, X_background: pd.DataFrame):
    """Return an exact SHAP explainer for "model", "X_background" must be raw features from the training split; 
    it is pushed through the model's own transform so the explainer sees the same space the estimator was fitted on.
    """
    if not isinstance(X_background, pd.DataFrame):
        raise TypeError("X_background must be a pandas DataFrame")
    if X_background.empty:
        raise ValueError("Background dataset cannot be empty")
    if not is_classifier(model.estimator):
        raise ValueError(f"{model.name} is not a classifier")

    background = model.transform(X_background)
    if isinstance(model.estimator, LINEAR_ESTIMATORS):
        return shap.LinearExplainer(model.estimator, background)
    if isinstance(model.estimator, TREE_ESTIMATORS):
        return shap.TreeExplainer(model.estimator)
    raise ValueError(
        f"No exact SHAP explainer registered for {type(model.estimator).__name__}"
    )


def _as_matrix(values, n_samples: int, n_features: int) -> np.ndarray:
    """Normalise every SHAP output shape into a plain (n_samples, n_features).
(shap sürümüne ve modele göre çıktı 2B dizi, 3B dizi (n, d, n_classes) veya
sınıf başına dizi listesi olabilir. İkili sınıflandırmada ilgilendiğimiz her zaman pozitif sınıftır (temerrüt))"""
    if isinstance(values, list): # eski API: sınıf başına bir dizi
        values = values[-1] # pozitif sınıf (temerrüt)
    array = np.asarray(values)
    if array.ndim == 3: # (n, d, n_classes)
        array = array[:, :, -1]# pozitif sınıf
    if array.shape != (n_samples, n_features):
        raise ValueError(
            f"Unexpected SHAP output shape {array.shape}, "
            f"expected ({n_samples}, {n_features})"
        )
    return array


def _base_value(explainer) -> float:
    # Collapse the explainer's expected value to a single float.
    return float(np.ravel(explainer.expected_value)[-1])


def model_margin(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    """Raw model output in the space SHAP explains (log-odds / margin).

    Bu fonksiyon üretimde kullanılmıyor; SHAP'ın toplanabilirlik (additivity)
    garantisini test edebilmek için var: sum(shap) + base == margin.
    """
    return np.asarray(model.estimator.decision_function(model.transform(X))).ravel()






#SHAP computation

def compute_shap_values(
    model: TrainedModel,
    X: pd.DataFrame,
    X_background: pd.DataFrame,
    dataset: str = "original",
    explainer=None,
) -> ExplanationResult:
    # Explain "X" with "model", using "X_background" as the reference.
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if X.empty:
        raise ValueError("Cannot explain an empty frame")
    if list(X.columns) != list(X_background.columns):
        raise ValueError("X and X_background must share the same columns")

    explainer = explainer if explainer is not None else build_explainer(model, X_background)
    raw = explainer.shap_values(model.transform(X))
    matrix = _as_matrix(raw, len(X), X.shape[1])

    return ExplanationResult(
        dataset=dataset,
        model_name=model.name,
        shap_values=pd.DataFrame(matrix, columns=X.columns, index=X.index),
        base_value=_base_value(explainer),
    )


def select_rows(
    datasets: Mapping[str, pd.DataFrame],
    max_samples: Optional[int] = None,
    seed: int = RANDOM_STATE,
) -> pd.Index:
    # Pick the row indices to explain, once, shared by every dataset.
    if not datasets:
        raise ValueError("No datasets supplied")
    frames = list(datasets.values())
    reference_index = frames[0].index
    for frame in frames[1:]:
        if not frame.index.equals(reference_index):
            raise ValueError("All datasets must share the same row index")
    if max_samples is None or max_samples >= len(reference_index):
        return reference_index
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(reference_index), size=max_samples, replace=False)
    return reference_index[np.sort(picked)]


def explain_datasets(
    model: TrainedModel,
    datasets: Mapping[str, pd.DataFrame],
    X_background: pd.DataFrame,
    max_samples: Optional[int] = None,
    seed: int = RANDOM_STATE,
) -> dict[str, ExplanationResult]:
    # Explain each dataset with the same frozen model and frozen background.
    rows = select_rows(datasets, max_samples=max_samples, seed=seed)
    explainer = build_explainer(model, X_background) # Explainer bir kez kurulur: arka plan dağılımı tüm veri setleri için aynı.
    return {
        name: compute_shap_values(
            model, frame.loc[rows], X_background, dataset=name, explainer=explainer
        )
        for name, frame in datasets.items()
    }


#Aggregation: SHAP matrix -> importance -> ranking

def global_importance(shap_values: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per feature, sorted descending.

    Mutlak değer alıyoruz çünkü global önem "ne kadar etkili" sorusudur; işaret
    (artırıyor mu azaltıyor mu) müşteri bazında anlamlıdır, popülasyon
    ortalamasında birbirini götürür.
    """
    if not isinstance(shap_values, pd.DataFrame):
        raise TypeError("shap_values must be a pandas DataFrame")
    if shap_values.empty:
        raise ValueError("shap_values is empty")
    importance = shap_values.abs().mean(axis=0)
    return importance.sort_values(ascending=False)


def top_k_features(importance: pd.Series, k: int = DEFAULT_TOP_K) -> list[str]:
    # First "k" names of an importance series (already sorted).
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return list(importance.index[:k])


def importance_table(results: Mapping[str, ExplanationResult]) -> pd.DataFrame:
    # Feature x dataset table of global importances, ordered by the first column.
    if not results:
        raise ValueError("No explanation results supplied")
    columns = {name: result.global_importance() for name, result in results.items()}
    table = pd.DataFrame(columns)
    first = next(iter(columns))
    return table.sort_values(first, ascending=False)


def ranking_table(
    results: Mapping[str, ExplanationResult], k: int = DEFAULT_TOP_K
) -> pd.DataFrame:
    # Side-by-side top-k rankings, one column per dataset.
    if not results:
        raise ValueError("No explanation results supplied")
    data = {name: result.top_k(k) for name, result in results.items()}
    return pd.DataFrame(data, index=[f"#{i + 1}" for i in range(k)])