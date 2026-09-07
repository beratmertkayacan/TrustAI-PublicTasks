"""
Loading and preprocessing of the Taiwan Credit Card Default dataset.
The dataset is the UCI "default of credit card clients" data (30,000 clients,
23 features). It is loaded from OpenML by default so the pipeline reproduces
Tasks 1-3, with an offline fallback to the raw UCI Excel file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
OPENML_DATA_ID = 42477
TARGET_NAME = "default"

#: OpenML serves the features as x1..x23; map them to the official UCI names.
OPENML_COLUMN_MAP = {
    "x1": "limit_bal", 
    "x2": "sex", 
    "x3": "education", 
    "x4": "marriage",
    "x5": "age", 
    "x6": "pay_0", 
    "x7": "pay_2", 
    "x8": "pay_3", 
    "x9": "pay_4", 
    "x10": "pay_5", 
    "x5": "age", 
    "x6": "pay_0", 
    "x7": "pay_2", 
    "x8": "pay_3", 
    "x9": "pay_4",
    "x10": "pay_5",
    "x11": "pay_6",
    "x12": "bill_amt1",
    "x13": "bill_amt2",
    "x14": "bill_amt3",
    "x15": "bill_amt4",
    "x10": "pay_5", 
    "x11": "pay_6", 
    "x12": "bill_amt1", 
    "x13": "bill_amt2",
    "x14": "bill_amt3", 
    "x15": "bill_amt4", 
    "x16": "bill_amt5",
    "x17": "bill_amt6", 
    "x18": "pay_amt1", 
    "x19": "pay_amt2",
    "x20": "pay_amt3", 
    "x21": "pay_amt4", 
    "x22": "pay_amt5", 
    "x17": "bill_amt6", 
    "x18": "pay_amt1", 
    "x19": "pay_amt2", 
    "x20": "pay_amt3", 
    "x21": "pay_amt4", 
    "x22": "pay_amt5", 
    "x23": "pay_amt6",
    "x20": "pay_amt3", 
    "x21": "pay_amt4", 
    "x22": "pay_amt5", 
    "x23": "pay_amt6",
}

FEATURE_NAMES = [
    "limit_bal", "sex", "education", "marriage", "age",
    "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
]

CLASS_NAMES = ["No Default", "Default"]


@dataclass(frozen=True) 
class DataBundle:
    """Train/test split plus the frozen scaler fitted on the training split."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    def scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen training scaler to any frame with the same columns."""
        return scale_features(self.scaler, X)


def load_raw(
    local_path: Optional[Union[str, Path]] = None,
    data_id: int = OPENML_DATA_ID,
    data_home: Optional[Union[str, Path]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    
    if local_path is not None:
        X, y = _load_local_excel(Path(local_path))
    else:
        X, y = _load_openml(data_id=data_id, data_home=data_home)
    validate_dataset(X, y)
    return X, y


def _load_local_excel(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    # The UCI file carries a two-row header; the real column names are on row 2.
    frame = pd.read_excel(path, header=1)
    frame = frame.rename(columns={c: str(c).strip().lower() for c in frame.columns})
    target_col = "default payment next month"
    if target_col not in frame.columns:
        raise ValueError(f"Column '{target_col}' missing in {path}")
    missing = [c for c in FEATURE_NAMES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {path}: {missing}")
    X = frame[FEATURE_NAMES].astype(float)
    y = frame[target_col].astype(int).rename(TARGET_NAME)
    return X, y


def _load_openml(data_id: int, data_home) -> tuple[pd.DataFrame, pd.Series]:
    raw = fetch_openml(data_id=data_id, as_frame=True, parser="auto", data_home=data_home)
    X = raw.data.rename(columns=OPENML_COLUMN_MAP)
    missing = [c for c in FEATURE_NAMES if c not in X.columns]
    if missing:
        raise ValueError(f"OpenML dataset {data_id} is missing columns: {missing}")
    X = X[FEATURE_NAMES].astype(float)
    y = raw.target.astype(int).rename(TARGET_NAME)
    y.index = X.index
    return X, y


def validate_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    """Raise a descriptive error if the loaded data is unusable downstream."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    if X.empty:
        raise ValueError("X is empty")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"X contains NaN values in columns: {bad}")
    if y.isna().any():
        raise ValueError("y contains NaN values")
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")
    classes = set(pd.unique(y))
    if not classes <= {0, 1}:
        raise ValueError(f"y must be binary 0/1, found labels: {sorted(classes)}")
    if len(classes) < 2:
        raise ValueError("y must contain both classes")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split, identical in spirit to Tasks 1-3."""
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    validate_dataset(X, y)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training features only."""
    if X_train.empty:
        raise ValueError("Cannot fit a scaler on an empty frame")
    return StandardScaler().fit(X_train)


def scale_features(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Transform ``X`` with a fitted scaler, preserving columns and index."""
    expected = list(getattr(scaler, "feature_names_in_", X.columns))
    if list(X.columns) != expected:
        raise ValueError(
            "Column mismatch between scaler and frame: "
            f"expected {expected}, got {list(X.columns)}"
        )
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


def load_dataset(
    local_path: Optional[Union[str, Path]] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    data_home: Optional[Union[str, Path]] = None,
) -> DataBundle:
    """Load, validate and split the dataset in one call."""
    X, y = load_raw(local_path=local_path, data_home=data_home)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=fit_scaler(X_train),
    )
