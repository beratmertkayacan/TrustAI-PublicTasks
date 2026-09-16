from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .drift import DRIFT_COMPONENTS
from .shift import ORIGINAL_LABEL, SEVERITY_LEVELS

FIGURE_SIZE = (9, 5)
DPI = 150

# Çizimlerde kullanılan eğrilerin renk sabitleri (dosya boyu tutarlı ilerle) 
EXPLANATION_COLOUR = "#c2410c"
PERFORMANCE_COLOUR = "#1d4ed8"


# dataset naming -> severity axis

def parse_dataset_name(
    name: str, levels: Mapping[str, float] = SEVERITY_LEVELS
) -> tuple[str, float]:
    """Split "mixed_severe" into ("mixed", 2.5).

    "original" maps to intensity 0.0: kayma merdiveninin başlangıç noktası.
    """
    if name == ORIGINAL_LABEL:
        return ORIGINAL_LABEL, 0.0
    for level_name, intensity in levels.items():
        suffix = f"_{level_name}"
        if name.endswith(suffix):
            return name[: -len(suffix)], float(intensity)
    raise ValueError(f"Cannot parse severity level from dataset name '{name}'")


def severity_axis(
    datasets: Sequence[str], levels: Mapping[str, float] = SEVERITY_LEVELS
) -> pd.DataFrame:
    """Order dataset names along the shift-intensity axis."""
    if len(datasets) == 0:
        raise ValueError("No datasets supplied")
    rows = []
    for name in datasets:
        kind, intensity = parse_dataset_name(name, levels)
        rows.append({"dataset": name, "kind": kind, "intensity": intensity})
    return pd.DataFrame(rows).sort_values(["kind", "intensity"]).reset_index(drop=True)


# persistence helpers
def save_table(frame: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write a table to CSV, creating parent directories as needed."""
    if frame.empty:
        raise ValueError("Refusing to save an empty table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def save_figure(figure: Figure, path: Union[str, Path], dpi: int = DPI) -> Path:
    """Render a figure to PNG through the Agg canvas (no GUI, no global state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)# figürü Agg tuvaline bağla
    figure.savefig(path, dpi=dpi, format="png")
    return path


# figures
def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    if frame.empty:
        raise ValueError("Cannot plot an empty frame")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def plot_drift_curves(
    comparison: pd.DataFrame,
    title: str = "Explanation drift vs performance drift",
    levels: Mapping[str, float] = SEVERITY_LEVELS,
) -> Figure:
    """The headline figure: two drift curves on one shift-intensity axis.

    Eğrilerden hangisinin önce yükseldiği, tüm çalışmanın cevabıdır.
    """
    _require_columns(comparison, ["dataset", "explanation_drift", "performance_drift"])
    axis = severity_axis(comparison["dataset"], levels)
    data = axis.merge(comparison, on="dataset").sort_values("intensity")

    figure = Figure(figsize=FIGURE_SIZE)
    ax = figure.subplots()
    for kind, group in data.groupby("kind"):
        if kind == ORIGINAL_LABEL:
            continue # orijinal her ailenin başlangıç noktası, ayrı seri değil
        start = data[data["kind"] == ORIGINAL_LABEL]
        series = pd.concat([start, group]).sort_values("intensity")
        ax.plot(series["intensity"], series["explanation_drift"], marker="o",
                color=EXPLANATION_COLOUR, label=f"{kind}: explanation")
        ax.plot(series["intensity"], series["performance_drift"], marker="s",
                linestyle="--", color=PERFORMANCE_COLOUR, label=f"{kind}: performance")

    ax.set_xlabel("Shift intensity (0 = original, 1.0 = moderate, 2.5 = severe)")
    ax.set_ylabel("Drift score (0 = unchanged, 1 = fully drifted)")
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    figure.tight_layout()
    return figure


def plot_component_breakdown(drift_frame: pd.DataFrame, title: str = "Drift components") -> Figure:
    """Grouped bars: which of the four components carries the drift?"""
    _require_columns(drift_frame, ["dataset", *DRIFT_COMPONENTS])
    data = drift_frame.set_index("dataset")[list(DRIFT_COMPONENTS)]

    figure = Figure(figsize=FIGURE_SIZE)
    ax = figure.subplots()
    positions = range(len(data))
    width = 0.8 / len(DRIFT_COMPONENTS)
    for offset, component in enumerate(DRIFT_COMPONENTS):
        ax.bar(
            [p + offset * width for p in positions],
            data[component].values,
            width=width,
            label=component.replace("_", " "),
        )
    ax.set_xticks([p + 0.4 - width / 2 for p in positions])
    ax.set_xticklabels(data.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Component value (0-1)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    figure.tight_layout()
    return figure


def plot_importance_shift(
    importance: pd.DataFrame,
    reference_column: str = ORIGINAL_LABEL,
    shifted_column: str = "mixed_severe",
    top_n: int = 10,
    title: str = "Global SHAP importance: original vs shifted",
) -> Figure:
    """Horizontal bars comparing importance before and after the shift."""
    _require_columns(importance.reset_index(), [reference_column, shifted_column])
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    data = importance.nlargest(top_n, reference_column)[
        [reference_column, shifted_column]
    ].iloc[::-1]

    figure = Figure(figsize=FIGURE_SIZE)
    ax = figure.subplots()
    positions = range(len(data))
    ax.barh([p + 0.2 for p in positions], data[reference_column].values,
            height=0.4, color=PERFORMANCE_COLOUR, label=reference_column)
    ax.barh([p - 0.2 for p in positions], data[shifted_column].values,
            height=0.4, color=EXPLANATION_COLOUR, label=shifted_column)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8)
    figure.tight_layout()
    return figure


# markdown assembly
def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without extra dependencies.

    DataFrame.to_markdown gizli bir tabulate bağımlılığı istesin eksikse patlar, formatlayıcı yakalasın"""
    if frame.empty:
        raise ValueError("Cannot render an empty frame")
    header = [str(c) for c in frame.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in record]
            for record in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build_markdown_report(
    sections: Mapping[str, Union[pd.DataFrame, str]],
    title: str = "Explanation Drift Benchmark",
    intro: Optional[str] = None,
) -> str:
    """Assemble tables and prose into one markdown document."""
    if not sections:
        raise ValueError("No sections supplied")
    parts = [f"# {title}", ""]
    if intro:
        parts += [intro, ""]
    for heading, content in sections.items():
        parts.append(f"## {heading}")
        parts.append("")
        if isinstance(content, pd.DataFrame):
            parts.append(frame_to_markdown(content))
        else:
            parts.append(str(content))
        parts.append("")
    return "\n".join(parts)


def save_markdown(text: str, path: Union[str, Path]) -> Path:
    """Write a markdown document to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


    """Tables and figures for the benchmark.

Matplotlib burada pyplot üzerinden değil, doğrudan Figure nesnesiyle
kullanılıyor. Sebebi: pyplot global bir durum (state) tutar. 
Açık figürleri biriktirir, backend seçer, test koşarken pencere açmaya çalışabilir. 
Figure + Agg canvas ise saf bir nesnedir: fonksiyon figürü döndürür, çağıran isterse
kaydeder, testler de figürün içeriğini (eksen sayısı, çizgi sayısı, etiketler) dosyaya hiç dokunmadan doğrulayabilir.
"""
