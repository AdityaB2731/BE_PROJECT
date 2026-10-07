"""CSV loading, cleaning, target encoding, splitting, and scaling."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TextIO

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_PATH = Path("data/raw/CICIDS2017.csv")
DEFAULT_ARTIFACT_DIR = Path("artifacts")


def load_csv(source: str | Path | BinaryIO | TextIO) -> pd.DataFrame:
    """Load and clean a CSV from a path or an uploaded file-like object."""
    frame = pd.read_csv(source)
    frame.columns = frame.columns.astype(str).str.strip()
    frame = frame.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    if frame.empty:
        raise ValueError("The CSV has no rows after cleaning.")
    return frame


def find_target_column(frame: pd.DataFrame, target_column: str | None = None) -> str:
    """Find a target column, preferring common CICIDS naming conventions."""
    if target_column:
        if target_column not in frame.columns:
            raise ValueError(f"Target column '{target_column}' was not found.")
        return target_column

    candidates = {"label", "target", "class", "attack", "y"}
    for column in frame.columns:
        if column.lower() in candidates:
            return column
    raise ValueError(
        "Could not find the target column. Expected one of: Label, Target, Class, Attack."
    )


def encode_target(values: pd.Series) -> pd.Series:
    """Map BENIGN (case-insensitive) to 0 and every other label to 1."""
    return values.astype(str).str.strip().str.casefold().ne("benign").astype(int)


def prepare_dataset(
    frame: pd.DataFrame, target_column: str | None = None
) -> tuple[pd.DataFrame, pd.Series, str, dict[str, float]]:
    """Create numeric features and a binary target with median imputation values."""
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    target_name = find_target_column(frame, target_column)
    target = encode_target(frame.pop(target_name))

    features = frame.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")
    if features.shape[1] == 0:
        raise ValueError("No numeric feature columns were found in the CSV.")
    fill_values = features.median(numeric_only=True).fillna(0.0).to_dict()
    features = features.fillna(fill_values).fillna(0.0).astype(float)
    if target.nunique() < 2:
        raise ValueError("Training requires both BENIGN and attack rows.")
    return features, target, target_name, fill_values


def split_and_scale(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Perform a stratified 80/20 split and fit scaling only on training data."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train), columns=features.columns, index=x_train.index
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test), columns=features.columns, index=x_test.index
    )
    return x_train_scaled, x_test_scaled, y_train, y_test, scaler


def save_preprocessing(
    scaler: StandardScaler,
    feature_columns: list[str],
    fill_values: dict[str, float],
    target_column: str,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> None:
    """Persist preprocessing artifacts used by both models and inference."""
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, directory / "scaler.joblib")
    joblib.dump(
        {
            "feature_columns": feature_columns,
            "fill_values": fill_values,
            "target_column": target_column,
        },
        directory / "preprocessing.joblib",
    )