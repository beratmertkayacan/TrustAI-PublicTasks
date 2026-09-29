from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import pandas as pd

from .data import RANDOM_STATE, DataBundle, load_dataset
from .drift import drift_table, early_warning_table
from .thresholds import (
    BOOTSTRAP_SAMPLES,
    EXPLANATION_ONLY,
    bootstrap_gap_interval,
    estimate_thresholds,
    threshold_table,
)
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
    thresholds: pd.DataFrame = field(default_factory=pd.DataFrame)
    artefacts: dict[str, Path] = field(default_factory=dict)

    def verdict(self) -> str:
        """Majority verdict across every shifted dataset and model."""
        shifted = self._shifted()
        if shifted.empty:
            return "unknown"
        return str(shifted["verdict"].mode().iloc[0])

    def early_warning_rate(self) -> float:
        """Share of shifted comparisons where explanation drift was the larger one.

        This is a sign test: 1.0 means the explanation moved more in every case.
        It says nothing about significance, so read it next to
        :meth:`significant_gap_rate`.
        """
        shifted = self._shifted()
        if shifted.empty:
            return float("nan")
        index = shifted["early_warning_index"].dropna()
        if index.empty:
            return float("nan")
        return float((index > 0).sum() / len(index))

    def significant_gap_rate(self) -> float:
        """Share of shifted comparisons whose gap interval stays above zero."""
        shifted = self._shifted()
        if shifted.empty or "gap_ci_low" not in shifted.columns:
            return float("nan")
        low = shifted["gap_ci_low"].dropna()
        if low.empty:
            return float("nan")
        return float((low > 0).sum() / len(low))

    def early_warning_count(self) -> int:
        """How many shifted comparisons were labelled as explanation drift only."""
        shifted = self._shifted()
        if shifted.empty or "verdict" not in shifted.columns:
            return 0
        return int((shifted["verdict"] == EXPLANATION_ONLY).sum())

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
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> BenchmarkResult:
    """
    bundle:
        Pre-loaded data. It exists so tests can run end to end without loading the real data set;
        if it is not given, "load_dataset" is called. output_dir: with "None" no file is written and only the results are returned.
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
    thresholds: dict[str, object] = {}

    for model_name, model in models.items():
        # model quality on the full test set
        performance_rows += [
            {
                "model": model_name,
                "dataset": name,
                **compute_performance(bundle.y_test, model.predict_positive_proba(frame)),
            }
            for name, frame in datasets.items()
        ]

        # explanation side
        explanations = explain_datasets(
            model, datasets, bundle.X_train, max_samples=max_samples, seed=seed
        )
        model_drift = drift_table(explanations, reference_key=ORIGINAL_LABEL, k=k)
        drift_frames.append(model_drift)
        importance[model_name] = importance_table(explanations)

        # The comparison uses the rows that were explained, so both scores are
        # measured on the same sample and share the same noise level.
        rows = explanations[ORIGINAL_LABEL].shap_values.index
        y_rows = bundle.y_test.loc[rows]
        probabilities = {
            name: model.predict_positive_proba(frame.loc[rows])
            for name, frame in datasets.items()
        }
        scores = {
            name: compute_performance(y_rows, prob)
            for name, prob in probabilities.items()
        }
        performance_drift = {
            name: performance_drift_score(scores[ORIGINAL_LABEL], values)
            for name, values in scores.items()
        }

        model_thresholds = estimate_thresholds(
            explanations[ORIGINAL_LABEL],
            y_rows,
            probabilities[ORIGINAL_LABEL],
            n_samples=bootstrap_samples,
            seed=seed,
            k=k,
        )
        thresholds[model_name] = model_thresholds

        # put the two curves side by side
        comparison = early_warning_table(
            model_drift.set_index("dataset")["explanation_drift_score"].to_dict(),
            performance_drift,
        )
        comparison.insert(1, "model", model_name)
        comparison["explanation_threshold"] = model_thresholds.explanation
        comparison["performance_threshold"] = model_thresholds.performance

        intervals = [
            bootstrap_gap_interval(
                explanations[ORIGINAL_LABEL],
                explanations[name],
                y_rows,
                probabilities[ORIGINAL_LABEL],
                probabilities[name],
                n_samples=bootstrap_samples,
                seed=seed,
                k=k,
            )
            for name in comparison["dataset"]
        ]
        comparison["gap_ci_low"] = [low for low, _ in intervals]
        comparison["gap_ci_high"] = [high for _, high in intervals]
        comparison["verdict"] = [
            model_thresholds.classify(explanation, performance)
            for explanation, performance in zip(
                comparison["explanation_drift"], comparison["performance_drift"]
            )
        ]
        comparison_frames.append(comparison)

    result = BenchmarkResult(
        performance=performance_table(performance_rows),
        shift=shift_summary(bundle.X_test, datasets),
        drift=pd.concat(drift_frames, ignore_index=True),
        comparison=pd.concat(comparison_frames, ignore_index=True),
        importance=importance,
        thresholds=threshold_table(thresholds),
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
        "threshold_table": save_table(result.thresholds, output_dir / "thresholds.csv"),
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
                "Early warning thresholds": result.thresholds.round(4),
                "Early warning comparison": result.comparison.round(4),
                "Verdict": (
                    f"Majority verdict across shifted datasets: **{result.verdict()}**\n\n"
                    f"Explanation drift was the larger score in "
                    f"**{result.early_warning_rate():.0%}** of shifted comparisons, "
                    f"and the gap interval stayed above zero in "
                    f"**{result.significant_gap_rate():.0%}** of them. "
                    f"{result.early_warning_count()} comparisons were labelled "
                    f"'{EXPLANATION_ONLY}'."
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
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES,
                        help="Resamples used for the thresholds and the gap interval")
    args = parser.parse_args(argv)

    result = run_benchmark(
        local_path=args.local_path,
        max_samples=args.max_samples,
        k=args.top_k,
        output_dir=args.output_dir,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(result.comparison.round(3).to_string(index=False))
    print(f"\nVerdict: {result.verdict()}")
    print(f"Artefacts written to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__": # pragma: no cover
    raise SystemExit(main())


"""End-to-end benchmark runner.

One command: load the data, train two models, build nine evaluation sets, compute SHAP for
each of them, produce the drift and performance metrics, and write the tables and figures under outputs/.

    python -m explanation_drift.pipeline --max-samples 1500
"""