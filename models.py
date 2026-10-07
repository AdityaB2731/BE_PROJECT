"""Supervised and unsupervised NIDS models plus evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_random_forest(x_train: pd.DataFrame, y_train: pd.Series, artifact_dir: str | Path) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(x_train, y_train)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(artifact_dir) / "random_forest.joblib")
    return model


def train_isolation_forest(
    x_train: pd.DataFrame, y_train: pd.Series, artifact_dir: str | Path
) -> IsolationForest:
    normal_rows = x_train.loc[y_train == 0]
    if normal_rows.empty:
        raise ValueError("Isolation Forest requires at least one BENIGN training row.")
    model = IsolationForest(random_state=42, contamination="auto", n_estimators=100)
    model.fit(normal_rows)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(artifact_dir) / "isolation_forest.joblib")
    return model


def evaluate_classifier(
    model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    false_positives = matrix[0, 1]
    normal_rows = matrix[0].sum()
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() > 1 else 0.0,
        "false_positive_rate": float(false_positives / normal_rows) if normal_rows else 0.0,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["BENIGN", "ATTACK"], output_dict=True, zero_division=0
        ),
    }


def predict_flow(features_dict: dict[str, float], artifact_dir: str | Path = "artifacts") -> float:
    """Return the supervised attack probability for one feature dictionary."""
    directory = Path(artifact_dir)
    model = joblib.load(directory / "random_forest.joblib")
    metadata = joblib.load(directory / "preprocessing.joblib")
    scaler = joblib.load(directory / "scaler.joblib")
    row = pd.DataFrame([{column: features_dict.get(column, np.nan) for column in metadata["feature_columns"]}])
    row = row.apply(pd.to_numeric, errors="coerce").fillna(metadata["fill_values"]).fillna(0.0)
    scaled = pd.DataFrame(scaler.transform(row), columns=row.columns)
    return float(model.predict_proba(scaled)[:, 1][0])


def predict_anomaly(
    features_dict: dict[str, float], artifact_dir: str | Path = "artifacts"
) -> dict[str, float | int]:
    """Return binary anomaly status and a normalized anomaly score."""
    directory = Path(artifact_dir)
    model = joblib.load(directory / "isolation_forest.joblib")
    metadata = joblib.load(directory / "preprocessing.joblib")
    scaler = joblib.load(directory / "scaler.joblib")
    row = pd.DataFrame([{column: features_dict.get(column, np.nan) for column in metadata["feature_columns"]}])
    row = row.apply(pd.to_numeric, errors="coerce").fillna(metadata["fill_values"]).fillna(0.0)
    scaled = pd.DataFrame(scaler.transform(row), columns=row.columns)
    raw_prediction = int(model.predict(scaled)[0])
    decision_score = float(model.decision_function(scaled)[0])
    anomaly_score = float(np.clip(0.5 - decision_score, 0.0, 1.0))
    return {"is_anomaly": int(raw_prediction == -1), "anomaly_score": anomaly_score}