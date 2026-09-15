"""Tests for explanation_drift.explain (SHAP generation and aggregation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from explanation_drift.data import DataBundle, fit_scaler
from explanation_drift.explain import (
    ExplanationResult,
    _as_matrix,
    build_explainer,
    compute_shap_values,
    explain_datasets,
    global_importance,
    importance_table,
    model_margin,
    ranking_table,
    select_rows,
    top_k_features,
)
from explanation_drift.models import TrainedModel, train_advanced, train_baseline
from explanation_drift.shift import generate_shifted_datasets


@pytest.fixture
def bundle(X_synth, y_synth) -> DataBundle:
    X_train, X_test = X_synth.iloc[:150], X_synth.iloc[150:]
    y_train, y_test = y_synth.iloc[:150], y_synth.iloc[150:]
    return DataBundle(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        scaler=fit_scaler(X_train),
    )


@pytest.fixture
def baseline(bundle) -> TrainedModel:
    return train_baseline(bundle.X_train, bundle.y_train, bundle.scaler)

@pytest.fixture
def advanced(bundle) -> TrainedModel:
    return train_advanced(bundle.X_train, bundle.y_train)



#explainer selection

def test_linear_model_gets_linear_explainer(baseline, bundle):
    assert isinstance(build_explainer(baseline, bundle.X_train), shap.LinearExplainer)


def test_tree_model_gets_tree_explainer(advanced, bundle):
    assert isinstance(build_explainer(advanced, bundle.X_train), shap.TreeExplainer)


def test_unknown_estimator_is_rejected(bundle):
    model = TrainedModel(name="knn", estimator=KNeighborsClassifier(), scaler=None)
    with pytest.raises(ValueError, match="No exact SHAP explainer"):
        build_explainer(model, bundle.X_train)


def test_regressor_is_rejected(bundle):
    model = TrainedModel(name="linreg", estimator=LinearRegression(), scaler=None)
    with pytest.raises(ValueError, match="not a classifier"):
        build_explainer(model, bundle.X_train)


def test_empty_background_is_rejected(baseline, bundle):
    with pytest.raises(ValueError, match="Background dataset cannot be empty"):
        build_explainer(baseline, bundle.X_train.iloc[:0])


def test_non_dataframe_background_is_rejected(baseline, bundle):
    with pytest.raises(TypeError, match="DataFrame"):
        build_explainer(baseline, bundle.X_train.values)




# the cornerstone: SHAP additivity
@pytest.mark.parametrize("model_name", ["baseline", "advanced"])
def test_shap_values_are_additive(request, bundle, model_name):
    """sum(shap) + base_value == model margin.

    SHAP'ın teorik garantisi budur. Tutmuyorsa ya yanlış uzayda (ölçeklenmemiş veri) hesaplıyoruzdur,
    ya yanlış sınıfı seçiyoruzdur, ya da base_value yanlıştır. Tek bir assert üç ayrı hata sınıfını birden yakalar.
    """
    model = request.getfixturevalue(model_name)
    result = compute_shap_values(model, bundle.X_test, bundle.X_train)
    reconstructed = result.shap_values.values.sum(axis=1) + result.base_value
    np.testing.assert_allclose(
        reconstructed, model_margin(model, bundle.X_test), atol=1e-8
    )


def test_shap_frame_shape_and_labels(baseline, bundle):
    result = compute_shap_values(baseline, bundle.X_test, bundle.X_train, dataset="orig")
    assert result.shap_values.shape == bundle.X_test.shape
    assert list(result.shap_values.columns) == list(bundle.X_test.columns)
    assert list(result.shap_values.index) == list(bundle.X_test.index)
    assert result.dataset == "orig"
    assert result.model_name == baseline.name
    assert result.n_samples == len(bundle.X_test)


def test_compute_shap_values_invalid_inputs(baseline, bundle):
    with pytest.raises(TypeError, match="DataFrame"):
        compute_shap_values(baseline, bundle.X_test.values, bundle.X_train)
    with pytest.raises(ValueError, match="empty"):
        compute_shap_values(baseline, bundle.X_test.iloc[:0], bundle.X_train)
    with pytest.raises(ValueError, match="same columns"):
        compute_shap_values(baseline, bundle.X_test.drop(columns=["age"]), bundle.X_train)




# shape normalisation

def test_as_matrix_passes_through_2d():
    values = np.zeros((4, 3))
    assert _as_matrix(values, 4, 3).shape == (4, 3)


def test_as_matrix_takes_positive_class_from_3d():
    values = np.zeros((4, 3, 2))
    values[:, :, 1] = 7.0 # pozitif sınıf
    np.testing.assert_allclose(_as_matrix(values, 4, 3), np.full((4, 3), 7.0))


def test_as_matrix_takes_positive_class_from_class_list():
 # Eski shap API'si sınıf başına bir dizi döndürür; np.asarray bunu (2, n, d) yapardı ve yanlış eksenden kesit alınırdı.
    values = [np.zeros((4, 3)), np.full((4, 3), 7.0)]
    np.testing.assert_allclose(_as_matrix(values, 4, 3), np.full((4, 3), 7.0))


def test_as_matrix_rejects_unexpected_shape():
    with pytest.raises(ValueError, match="Unexpected SHAP output shape"):
        _as_matrix(np.zeros((4, 5)), 4, 3)


# row selection (same customers in each dataset)
def test_select_rows_defaults_to_every_row(bundle):
    datasets = generate_shifted_datasets(bundle.X_test, kinds=["mixed"])
    assert select_rows(datasets).equals(bundle.X_test.index)


def test_select_rows_subsamples_deterministically(bundle):
    datasets = generate_shifted_datasets(bundle.X_test, kinds=["mixed"])
    first = select_rows(datasets, max_samples=10, seed=7)
    second = select_rows(datasets, max_samples=10, seed=7)
    assert len(first) == 10
    assert first.equals(second)
    assert set(first) <= set(bundle.X_test.index)


def test_select_rows_ignores_oversized_sample(bundle):
    datasets = {"original": bundle.X_test}
    assert len(select_rows(datasets, max_samples=10_000)) == len(bundle.X_test)


def test_select_rows_rejects_bad_input(bundle):
    with pytest.raises(ValueError, match="No datasets"):
        select_rows({})
    with pytest.raises(ValueError, match="max_samples"):
        select_rows({"a": bundle.X_test}, max_samples=0)
    with pytest.raises(ValueError, match="same row index"):
        select_rows({"a": bundle.X_test, "b": bundle.X_test.iloc[:-1]})


#explaining the whole grid
def test_explain_datasets_covers_every_dataset(baseline, bundle):
    datasets = generate_shifted_datasets(bundle.X_test, kinds=["mixed"])
    results = explain_datasets(baseline, datasets, bundle.X_train)
    assert set(results) == set(datasets)
    assert all(isinstance(r, ExplanationResult) for r in results.values())


def test_explain_datasets_uses_the_same_customers(baseline, bundle):
    """satır örneklemesi 1 kez: veri setleri arası farkın ne kadarı kayma ne kadarı farklı müşteri"""
    datasets = generate_shifted_datasets(bundle.X_test, kinds=["mixed"])
    results = explain_datasets(baseline, datasets, bundle.X_train, max_samples=15)
    indices = [tuple(r.shap_values.index) for r in results.values()]
    assert len(set(indices)) == 1
    assert len(indices[0]) == 15


def test_background_stays_frozen_across_datasets(baseline, bundle):
    """base_value sadece arka plan dağılımına bağlı
    her dataset için explpainer'ı o datasetle yeniden kursaydık base_value datasetler arası kayar, tek bir eşitlik kontrolü"""
    datasets = generate_shifted_datasets(bundle.X_test, kinds=["mixed"])
    results = explain_datasets(baseline, datasets, bundle.X_train)
    base_values = {round(r.base_value, 12) for r in results.values()}
    assert len(base_values) == 1


def test_shifted_explanations_actually_differ(baseline, bundle):
    """Kayma açıklamaları değiştirmiyorsa ölçecek bir şey yok"""
    datasets = generate_shifted_datasets(bundle.X_test, kinds=["mixed"])
    results = explain_datasets(baseline, datasets, bundle.X_train)
    original = results["original"].global_importance()
    severe = results["mixed_severe"].global_importance()
    assert not np.allclose(original.values, severe.reindex(original.index).values)


# aggregation (global importance)
def test_global_importance_is_sorted_and_non_negative(baseline, bundle):
    result = compute_shap_values(baseline, bundle.X_test, bundle.X_train)
    importance = result.global_importance()
    assert (importance.values >= 0).all()
    assert list(importance.values) == sorted(importance.values, reverse=True)
    assert set(importance.index) == set(bundle.X_test.columns)


def test_global_importance_matches_manual_mean_abs():
    frame = pd.DataFrame({"a": [1.0, -3.0], "b": [0.5, 0.5]})
    importance = global_importance(frame)
    assert importance["a"] == pytest.approx(2.0)
    assert importance["b"] == pytest.approx(0.5)
    assert list(importance.index) == ["a", "b"] # büyükten küçüğe


def test_global_importance_invalid_inputs():
    with pytest.raises(TypeError, match="DataFrame"):
        global_importance(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="empty"):
        global_importance(pd.DataFrame())


def test_top_k_features():
    importance = pd.Series([3.0, 2.0, 1.0], index=["c", "b", "a"])
    assert top_k_features(importance, 2) == ["c", "b"]
    assert top_k_features(importance, 99) == ["c", "b", "a"] # taşmayı kırp 
    with pytest.raises(ValueError, match="k must be positive"): top_k_features(importance, 0)


def test_result_ranking_and_top_k(baseline, bundle):
    result = compute_shap_values(baseline, bundle.X_test, bundle.X_train)
    ranking = result.ranking()
    assert len(ranking) == bundle.X_test.shape[1]
    assert result.top_k(5) == ranking[:5]
    assert result.feature_names == list(bundle.X_test.columns)


def test_importance_and_ranking_tables(baseline, bundle):
    datasets = generate_shifted_datasets(bundle.X_test, kinds=["mixed"])
    results = explain_datasets(baseline, datasets, bundle.X_train)

    table = importance_table(results)
    assert list(table.columns) == list(results)
    assert len(table) == bundle.X_test.shape[1]
    assert table.iloc[0, 0] == table.iloc[:, 0].max() # ilk column'a göre sıralı

    ranks = ranking_table(results, k=4)
    assert ranks.shape == (4, len(results))
    assert list(ranks.index) == ["#1", "#2", "#3", "#4"]


def test_tables_require_results():
    with pytest.raises(ValueError, match="No explanation results"):
        importance_table({})
    with pytest.raises(ValueError, match="No explanation results"):
        ranking_table({})