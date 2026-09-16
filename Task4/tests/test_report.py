"""Tests for explanation_drift.report (tables, figures, markdown)."""

from __future__ import annotations

import pandas as pd
import pytest
from matplotlib.figure import Figure

from explanation_drift.drift import DRIFT_COMPONENTS
from explanation_drift.report import (
    build_markdown_report,
    frame_to_markdown,
    parse_dataset_name,
    plot_component_breakdown,
    plot_drift_curves,
    plot_importance_shift,
    save_figure,
    save_markdown,
    save_table,
    severity_axis,
)


@pytest.fixture
def comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["original", "mixed_moderate", "mixed_severe"],
            "explanation_drift": [0.0, 0.088, 0.157],
            "performance_drift": [0.0, 0.042, 0.062],
        }
    )


@pytest.fixture
def drift_frame() -> pd.DataFrame:
    frame = pd.DataFrame({"dataset": ["original", "mixed_severe"]})
    for component in DRIFT_COMPONENTS:
        frame[component] = [0.0, 0.2]
    return frame

# dataset name -> severity
def test_parse_dataset_name():
    assert parse_dataset_name("original") == ("original", 0.0)
    assert parse_dataset_name("mixed_severe") == ("mixed", 2.5)
    assert parse_dataset_name("credit_limit_moderate") == ("credit_limit", 1.0)


def test_parse_dataset_name_rejects_unknown_level():
    with pytest.raises(ValueError, match="Cannot parse severity level"):
        parse_dataset_name("mixed_catastrophic")


def test_severity_axis_is_ordered(comparison):
    axis = severity_axis(comparison["dataset"])
    mixed = axis[axis["kind"] == "mixed"]
    assert list(mixed["intensity"]) == [1.0, 2.5] # artan şiddet
    with pytest.raises(ValueError, match="No datasets"):
        severity_axis([])


#persistence
def test_save_table_creates_parents_and_roundtrips(comparison, tmp_path):
    path = save_table(comparison, tmp_path / "nested" / "dir" / "t.csv")
    assert path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(path), comparison)


def test_save_table_refuses_empty(tmp_path):
    with pytest.raises(ValueError, match="empty table"):
        save_table(pd.DataFrame(), tmp_path / "t.csv")


def test_save_figure_writes_a_png(comparison, tmp_path):
    path = save_figure(plot_drift_curves(comparison), tmp_path / "f.png")
    assert path.exists()
    assert path.read_bytes()[:4] == b"\x89PNG" # gerçek PNG için imza


# figures - içeriği doğruluyoruz, piksellerini değil
def test_drift_curves_draw_two_lines(comparison):
    figure = plot_drift_curves(comparison)
    assert isinstance(figure, Figure)
    axis = figure.axes[0]
    assert len(axis.lines) == 2 # açıklama + performans
    labels = [line.get_label() for line in axis.lines]
    assert any("explanation" in label for label in labels)
    assert any("performance" in label for label in labels)
    # her iki eğri de orijinden (intensity 0) başlamalı
    for line in axis.lines:
        assert line.get_xdata()[0] == 0.0


def test_drift_curves_invalid_input(comparison):
    with pytest.raises(ValueError, match="empty frame"):
        plot_drift_curves(comparison.iloc[:0])
    with pytest.raises(ValueError, match="Missing required columns"):
        plot_drift_curves(comparison.drop(columns=["performance_drift"]))


def test_component_breakdown_has_one_bar_group_per_component(drift_frame):
    figure = plot_component_breakdown(drift_frame)
    axis = figure.axes[0]
    assert len(axis.patches) == len(DRIFT_COMPONENTS) * len(drift_frame)
    assert [t.get_text() for t in axis.get_xticklabels()] == list(drift_frame["dataset"])


def test_importance_shift_plot(tmp_path):
    importance = pd.DataFrame(
        {"original": [3.0, 2.0, 1.0], "mixed_severe": [1.0, 2.0, 3.0]},
        index=["a", "b", "c"],
    )
    figure = plot_importance_shift(importance, top_n=2)
    axis = figure.axes[0]
    assert len(axis.patches) == 4 # 2 değişken x 2 seri
    with pytest.raises(ValueError, match="top_n must be positive"):
        plot_importance_shift(importance, top_n=0)
    with pytest.raises(ValueError, match="Missing required columns"):
        plot_importance_shift(importance, shifted_column="nope")


#markdown üretimi doğrulama
def test_frame_to_markdown_structure():
    frame = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
    lines = frame_to_markdown(frame).splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "|---|---|"
    assert lines[3] == "|  | y |" # NaN boş hücreye dönüşür
    with pytest.raises(ValueError, match="empty frame"):
        frame_to_markdown(pd.DataFrame())


def test_build_markdown_report(comparison):
    text = build_markdown_report(
        {"Comparison": comparison, "Verdict": "explanation first"},
        title="T", intro="intro line",
    )
    assert text.startswith("# T")
    assert "intro line" in text
    assert "## Comparison" in text
    assert "## Verdict" in text
    assert "| dataset |" in text
    with pytest.raises(ValueError, match="No sections"):
        build_markdown_report({})


def test_save_markdown(tmp_path):
    path = save_markdown("# hello", tmp_path / "deep" / "r.md")
    assert path.read_text(encoding="utf-8") == "# hello"