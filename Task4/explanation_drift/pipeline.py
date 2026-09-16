from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import pandas as pd

from .data import RANDOM_STATE, DataBundle, load_dataset
from .drift import drift_table, early_warning_table
from .explain import DEFAULT_TOP_K, explain_datasets, importance_table
from .metrics import compute_performance, performance_drift_score, performance_table
from .models import train_models
from .report import (
    build_markdown_report,
    plot_component_breakdown,
    plot_drift_curves,
    plot_importance_shift,
    save_figure,
    save_markdown,
    save_table,
)
from .shift import (
    HEADLINE_KIND,
    ORIGINAL_LABEL,
    SEVERITY_LEVELS,
    SHIFT_KINDS,
    generate_shifted_datasets,
    shift_summary,
)

DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass
class BenchmarkResult:

    performance: pd.DataFrame
    shift: pd.DataFrame
    drift: pd.DataFrame
    comparison: pd.DataFrame
    importance: dict[str, pd.DataFrame] = field(default_factory=dict)
    artefacts: dict[str, Path] = field(default_factory=dict)

    def verdict(self) -> str:
        """Majority verdict across every shifted dataset and model."""
        shifted = self._shifted()
        if shifted.empty:
            return "unknown"
        return str(shifted["verdict"].mode().iloc[0])

    def early_warning_rate(self) -> float:
        """Share of shifted comparisons where explanation drift exceeded performance drift.

        Mutlak eşiğe dayanan "verdict" küçük sayılarda korumacı davranır.
        Bu oran işaret testidir: 1.0 demek "her koşulda açıklama daha çok kaydı".
        """
        shifted = self._shifted()
        if shifted.empty:
            return float("nan")
        index = shifted["early_warning_index"].dropna()
        if index.empty:
            return float("nan")
        return float((index > 0).sum() / len(index))

    def _shifted(self) -> pd.DataFrame:
        return self.comparison[self.comparison["dataset"] != ORIGINAL_LABEL]


def run_benchmark(
    bundle: Optional[DataBundle] = None,
    local_path: Optional[Union[str, Path]] = None,
    kinds: Sequence[str] = SHIFT_KINDS,
    levels: Mapping[str, float] = SEVERITY_LEVELS,
    max_samples: Optional[int] = None,
    k: int = DEFAULT_TOP_K,
    output_dir: Optional[Union[str, Path]] = DEFAULT_OUTPUT_DIR,
    seed: int = RANDOM_STATE,
) -> BenchmarkResult:
    """
    bundle:
        Pre-loaded data. Testlerde gerçek veri setini yüklemeden uçtan uca koşmak için var; 
        verilmezse "load_dataset" çağrılır. output_dir: "None" verilirse hiçbir dosya yazılmaz, sonuçlar yalnızca döndürülür.
    """
    bundle = bundle if bundle is not None else load_dataset(local_path=local_path)
    models = train_models(bundle, random_state=seed)
    datasets = generate_shifted_datasets(
        bundle.X_test, kinds=kinds, levels=levels, seed=seed
    )

    performance_rows: list[dict] = []
    drift_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    importance: dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        # performans tarafı 
        scores = {
            name: compute_performance(bundle.y_test, model.predict_positive_proba(frame))
            for name, frame in datasets.items()
        }
        performance_rows += [
            {"model": model_name, "dataset": name, **values}
            for name, values in scores.items()
        ]
        performance_drift = {
            name: performance_drift_score(scores[ORIGINAL_LABEL], values)
            for name, values in scores.items()
        }

        # açıklama tarafı 
        explanations = explain_datasets(
            model, datasets, bundle.X_train, max_samples=max_samples, seed=seed
        )
        model_drift = drift_table(explanations, reference_key=ORIGINAL_LABEL, k=k)
        drift_frames.append(model_drift)
        importance[model_name] = importance_table(explanations)

        # iki eğriyi yan yana koy 
        comparison = early_warning_table(
            model_drift.set_index("dataset")["explanation_drift_score"].to_dict(),
            performance_drift,
        )
        comparison.insert(1, "model", model_name)
        comparison_frames.append(comparison)

    result = BenchmarkResult(
        performance=performance_table(performance_rows),
        shift=shift_summary(bundle.X_test, datasets),
        drift=pd.concat(drift_frames, ignore_index=True),
        comparison=pd.concat(comparison_frames, ignore_index=True),
        importance=importance,
    )

    if output_dir is not None:
        result.artefacts = _write_artefacts(result, Path(output_dir), levels, kinds)
    return result


def _write_artefacts(
    result: BenchmarkResult,
    output_dir: Path,
    levels: Mapping[str, float],
    kinds: Sequence[str],
) -> dict[str, Path]:
    """Persist every table and figure; returns a name -> path map."""
    artefacts = {
        "performance_table": save_table(result.performance, output_dir / "performance.csv"),
        "shift_table": save_table(result.shift, output_dir / "shift_magnitude.csv"),
        "drift_table": save_table(result.drift, output_dir / "explanation_drift.csv"),
        "comparison_table": save_table(result.comparison, output_dir / "early_warning.csv"),
    }

    for model_name, frame in result.importance.items():
        artefacts[f"importance_{model_name}"] = save_table(
            frame.reset_index(names="feature"), output_dir / f"importance_{model_name}.csv"
        )

    headline = f"{HEADLINE_KIND}_{list(levels)[-1]}"
    for model_name in result.importance:
        subset = result.comparison[result.comparison["model"] == model_name]
        artefacts[f"curves_{model_name}"] = save_figure(
            plot_drift_curves(subset, title=f"Drift curves - {model_name}", levels=levels),
            output_dir / f"drift_curves_{model_name}.png",
        )
        artefacts[f"components_{model_name}"] = save_figure(
            plot_component_breakdown(
                result.drift[result.drift["model"] == model_name],
                title=f"Drift components - {model_name}",
            ),
            output_dir / f"drift_components_{model_name}.png",
        )
        if HEADLINE_KIND in kinds and headline in result.importance[model_name].columns:
            artefacts[f"importance_plot_{model_name}"] = save_figure(
                plot_importance_shift(
                    result.importance[model_name], shifted_column=headline,
                    title=f"SHAP importance shift - {model_name}",
                ),
                output_dir / f"importance_shift_{model_name}.png",
            )

    artefacts["report"] = save_markdown(
        build_markdown_report(
            {
                "Shift magnitude": result.shift.round(3),
                "Model performance": result.performance.round(4),
                "Explanation drift": result.drift.round(3),
                "Early warning comparison": result.comparison.round(3),
                "Verdict": (
                    f"Majority verdict across shifted datasets: **{result.verdict()}**\n\n"
                    f"Explanation drift exceeded performance drift in "
                    f"**{result.early_warning_rate():.0%}** of shifted comparisons."
                ),
            },
            intro="Generated by `python -m explanation_drift.pipeline`.",
        ),
        output_dir / "benchmark_report.md",
    )
    return artefacts


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Run the explanation drift benchmark")
    parser.add_argument("--local-path", default=None,
                        help="Raw UCI .xls file to use instead of OpenML")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Explain only this many test rows (same rows everywhere)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args(argv)

    result = run_benchmark(
        local_path=args.local_path,
        max_samples=args.max_samples,
        k=args.top_k,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(result.comparison.round(3).to_string(index=False))
    print(f"\nVerdict: {result.verdict()}")
    print(f"Artefacts written to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__": # pragma: no cover
    raise SystemExit(main())


"""End-to-end benchmark runner.

Tek komut: veriyi yükle, iki modeli eğit, dokuz değerlendirme seti üret, her biri
için SHAP hesapla, drift ve performans metriklerini çıkar, tabloları ve figürleri outputs/ altına yaz.

    python -m explanation_drift.pipeline --max-samples 1500
"""