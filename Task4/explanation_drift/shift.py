"""
Distribution shift generation and shift-magnitude measurement.

Shift families
--------------
age: Popülasyon yaşlanıyor (additive, in years).
credit_limit: Enflasyon / limit artışı (multiplicative).
payment_amount: Müşteriler daha az ödeme yapıyor (multiplicative decay).
mixed: Hepsi birlikte + tüm sürekli değişkenlere Gauss gürültüsü.

moderate and severe are same transformation at two strengths.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .data import RANDOM_STATE

#shift parametreleri (moderate = intensity 1.0)

AGE_SHIFT_YEARS = 5.0        # moderate seviyede popülasyon ~5 yaş yaşlanıyor
LIMIT_GROWTH = 0.35          # moderate: kredi limitleri %35 büyüyor
PAYMENT_DECLINE = 0.35       # moderate: ödeme tutarları %35 azalıyor
NOISE_SCALE = 0.10           # mixed: her değişkenin std'sinin %10'u kadar gürültü

# Severity ladder -> Aynı dönüşüm, iki farklı şiddet.
SEVERITY_LEVELS: dict[str, float] = {"moderate": 1.0, "severe": 2.5}

ORIGINAL_LABEL = "original"

AGE_COLUMN = "age"
LIMIT_COLUMN = "limit_bal"
PAYMENT_COLUMNS = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]
BILL_COLUMNS = ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]

# Kategorik(ordinal) kolonlar -> bunlara gürültü eklemek anlamsız olurdu.
CATEGORICAL_COLUMNS = ["sex", "education", "marriage", "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]

AGE_BOUNDS = (18.0, 100.0)   # gerçekçi yaş aralığı

# helper func frame is valid
def _check_frame(X: pd.DataFrame, required: Sequence[str]) -> None:
    # Guard clause shared by every shift function.
    if not isinstance(X, pd.DataFrame): 
        raise TypeError("X must be a pandas DataFrame")
    if X.empty:
        raise ValueError("Cannot shift an empty frame")
    missing = [c for c in required if c not in X.columns]
    if missing:
        raise ValueError(f"Missing columns required for this shift: {missing}")


def _check_intensity(intensity: float) -> float:
    if intensity < 0:
        raise ValueError(f"intensity must be >= 0, got {intensity}")
    return float(intensity)


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


# shift families

def shift_age(
    X: pd.DataFrame, intensity: float = 1.0, seed: Optional[int] = RANDOM_STATE
) -> pd.DataFrame:
    # Age the population by intensity * AGE_SHIFT_YEARS years.
    _check_frame(X, [AGE_COLUMN])
    intensity = _check_intensity(intensity)
    rng = _rng(seed)
    out = X.copy()  # girdi frame'i asla mutasyona uğramaz
    # Additive kayma + küçük bireysel varyasyon (herkes aynı anda yaşlanmaz)
    jitter = rng.normal(0.0, 1.0 * intensity, len(out))
    out[AGE_COLUMN] = np.clip(
        out[AGE_COLUMN] + intensity * AGE_SHIFT_YEARS + jitter, *AGE_BOUNDS
    )
    return out


def shift_credit_limit(
    X: pd.DataFrame, intensity: float = 1.0, seed: Optional[int] = RANDOM_STATE
) -> pd.DataFrame:
    # Credit limits to be inflated multiplicatively by (1 + LIMIT_GROWTH) ** intensity.
    _check_frame(X, [LIMIT_COLUMN])
    intensity = _check_intensity(intensity)
    rng = _rng(seed)
    out = X.copy()
    # Çarpımsal büyüme (negatife düşmesi matematiksel olarak imkânsız)
    factor = (1.0 + LIMIT_GROWTH) ** intensity
    noise = rng.normal(1.0, 0.05 * intensity, len(out))
    out[LIMIT_COLUMN] = np.maximum(out[LIMIT_COLUMN] * factor * noise, 0.0)
    return out


def shift_payment_amount(
    X: pd.DataFrame, intensity: float = 1.0, seed: Optional[int] = RANDOM_STATE
) -> pd.DataFrame:
    # Repayment amounts to be shrunk multiplicatively by (1 - PAYMENT_DECLINE) ** intensity.
    _check_frame(X, PAYMENT_COLUMNS)
    intensity = _check_intensity(intensity)
    rng = _rng(seed)
    out = X.copy()
    factor = (1.0 - PAYMENT_DECLINE) ** intensity
    for column in PAYMENT_COLUMNS:
        noise = rng.normal(1.0, 0.05 * intensity, len(out))
        # Ödemeler negatif olamaz  (alt sınır 0)
        out[column] = np.maximum(out[column] * factor * noise, 0.0)
    return out


def shift_mixed(
    X: pd.DataFrame, intensity: float = 1.0, seed: Optional[int] = RANDOM_STATE
) -> pd.DataFrame:
    # Three shifts at once, plus Gaussian noise on the continuous columns. headline scenario
    _check_frame(X, [AGE_COLUMN, LIMIT_COLUMN, *PAYMENT_COLUMNS])
    intensity = _check_intensity(intensity)
    # Aynı seed'i üç kez kullanmamak için her adıma farklı offset veriyoruz
    base = 0 if seed is None else int(seed)
    out = shift_age(X, intensity, None if seed is None else base)
    out = shift_credit_limit(out, intensity, None if seed is None else base + 1)
    out = shift_payment_amount(out, intensity, None if seed is None else base + 2)

    rng = _rng(None if seed is None else base + 3)
    continuous = [
        c for c in out.columns
        if c not in CATEGORICAL_COLUMNS and pd.api.types.is_numeric_dtype(out[c])
    ]
    for column in continuous:
        std = float(X[column].std())
        if std == 0.0:
            continue  # kolon sabitse gürültü ekleme, atla (dağılımı bozmamak için)
        out[column] = out[column] + rng.normal(0.0, NOISE_SCALE * intensity * std, len(out))
    # Gürültü sonrası fiziksel sınırları tekrar uygula.
    # Not: bill_amt negatif olabilir (fazla ödeme -> alacaklı bakiye), onu kırpmamak için
    for column in [LIMIT_COLUMN, *PAYMENT_COLUMNS]:
        out[column] = np.maximum(out[column], 0.0)
    out[AGE_COLUMN] = np.clip(out[AGE_COLUMN], *AGE_BOUNDS)
    return out


SHIFT_FUNCTIONS = {
    "age": shift_age,
    "credit_limit": shift_credit_limit,
    "payment_amount": shift_payment_amount,
    "mixed": shift_mixed,
}

SHIFT_KINDS = tuple(SHIFT_FUNCTIONS)

# Ana eksen: tek değişkenli kaymalar ablation, mixed ise kritik.
HEADLINE_KIND = "mixed"


def apply_shift(
    X: pd.DataFrame,
    kind: str,
    intensity: float = 1.0,
    seed: Optional[int] = RANDOM_STATE,
) -> pd.DataFrame:
    # Dispatch to one of the shift families by name.
    if kind not in SHIFT_FUNCTIONS:
        raise ValueError(
            f"Unknown shift kind '{kind}'. Available: {sorted(SHIFT_FUNCTIONS)}"
        )
    return SHIFT_FUNCTIONS[kind](X, intensity=intensity, seed=seed)


def generate_shifted_datasets(
    X: pd.DataFrame,
    kinds: Iterable[str] = SHIFT_KINDS,
    levels: Mapping[str, float] = SEVERITY_LEVELS,
    seed: Optional[int] = RANDOM_STATE,
    include_original: bool = True,
) -> dict[str, pd.DataFrame]:
# Build the full evaluation grid, returns a mapping {"original": X, "mixed_moderate": ..., ...}. Keys are f"{kind}_{level}" so downstream tables stay readable.
    kinds = list(kinds)
    if not kinds:
        raise ValueError("At least one shift kind is required")
    if not levels:
        raise ValueError("At least one severity level is required")

    datasets: dict[str, pd.DataFrame] = {}
    if include_original:
        datasets[ORIGINAL_LABEL] = X.copy()
    for kind in kinds:
        for level_name, intensity in levels.items():
            datasets[f"{kind}_{level_name}"] = apply_shift(
                X, kind=kind, intensity=intensity, seed=seed
            )
    return datasets



# shift magnitude: "moderate" ve "severe" etiketlerini sayısallaştıran kısım

def population_stability_index(
    reference: Sequence[float],
    actual: Sequence[float],
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
# Population Stability Index between a reference and an actual sample. rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift. Bins are reference quantiles, so the metric is scale-free.

    ref = np.asarray(reference, dtype=float)
    act = np.asarray(actual, dtype=float)
    if ref.size == 0 or act.size == 0:
        raise ValueError("PSI needs non-empty reference and actual samples")
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")

    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0  # sabit referans dağılımı (kayma yok)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_share = np.clip(np.histogram(ref, edges)[0] / ref.size, epsilon, None)
    act_share = np.clip(np.histogram(act, edges)[0] / act.size, epsilon, None)
    return float(np.sum((act_share - ref_share) * np.log(act_share / ref_share)))


def normalised_wasserstein(reference: Sequence[float], actual: Sequence[float]) -> float:
    # Wasserstein distance expressed in reference standard deviations.
    ref = np.asarray(reference, dtype=float)
    act = np.asarray(actual, dtype=float)
    if ref.size == 0 or act.size == 0:
        raise ValueError("Wasserstein needs non-empty samples")
    std = float(np.std(ref))
    distance = float(wasserstein_distance(ref, act))
    return 0.0 if std == 0.0 else distance / std


def shift_magnitude(
    X_reference: pd.DataFrame, X_shifted: pd.DataFrame, bins: int = 10
) -> pd.DataFrame:
# Per-feature shift report: PSI, normalised Wasserstein, mean change.
    if list(X_reference.columns) != list(X_shifted.columns):
        raise ValueError("Reference and shifted frames must share the same columns")
    if X_reference.empty or X_shifted.empty:
        raise ValueError("Cannot measure shift on an empty frame")

    rows = []
    for column in X_reference.columns:
        ref, act = X_reference[column].values, X_shifted[column].values
        ref_mean = float(np.mean(ref))
        rows.append(
            {
                "feature": column,
                "psi": population_stability_index(ref, act, bins=bins),
                "wasserstein_norm": normalised_wasserstein(ref, act),
                "mean_change_pct": float("nan")
                if ref_mean == 0.0
                else (float(np.mean(act)) - ref_mean) / abs(ref_mean) * 100.0,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def classify_shift(psi: float) -> str:
    # Map a PSI value onto the conventional stability labels.
    if np.isnan(psi):
        return "unknown"
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "major"


def shift_summary(
    X_reference: pd.DataFrame,
    datasets: Mapping[str, pd.DataFrame],
    bins: int = 10,
) -> pd.DataFrame:
    # One row per shifted dataset: how far did the input distribution move?
    if not datasets:
        raise ValueError("No datasets supplied")
    rows = []
    for name, frame in datasets.items():
        report = shift_magnitude(X_reference, frame, bins=bins)
        max_psi = float(report["psi"].max())
        rows.append(
            {
                "dataset": name,
                "mean_psi": float(report["psi"].mean()),
                "max_psi": max_psi,
                "mean_wasserstein_norm": float(report["wasserstein_norm"].mean()),
                "n_features_major_shift": int((report["psi"] >= 0.25).sum()),
                "severity_label": classify_shift(max_psi),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)