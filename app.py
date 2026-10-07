"""Streamlit dashboard for training and inspecting network flows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from data_pipeline import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_PATH,
    load_csv,
    prepare_dataset,
    save_preprocessing,
    split_and_scale,
)
from models import (
    evaluate_classifier,
    predict_anomaly,
    predict_flow,
    train_isolation_forest,
    train_random_forest,
)
from risk_engine import calculate_risk


st.set_page_config(page_title="NIDS Workbench", page_icon="🛡️", layout="wide")
st.title("Network Intrusion Detection System")
st.caption("Local-first training and hybrid flow inspection")


def artifact_ready() -> bool:
    required = ["random_forest.joblib", "isolation_forest.joblib", "scaler.joblib", "preprocessing.joblib"]
    return all((DEFAULT_ARTIFACT_DIR / name).exists() for name in required)


def load_source(uploaded_file, path_text: str):
    if uploaded_file is not None:
        return load_csv(uploaded_file)
    path = Path(path_text.strip())
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return load_csv(path)


with st.sidebar:
    st.header("Configuration")
    path_text = st.text_input("Local CSV path", value=str(DEFAULT_DATA_PATH))
    uploaded_file = st.file_uploader("Or upload a CSV", type=["csv"])
    target_column = st.text_input("Target column (optional)", placeholder="Label")

tab_training, tab_inspector, tab_metrics = st.tabs(
    ["Data & Model Training", "Live Flow Inspector", "Performance & Metrics"]
)

if "metrics" not in st.session_state:
    st.session_state.metrics = None

with tab_training:
    st.subheader("Dataset")
    st.write("Upload a CSV above or place one at the local path. The default path is ready for your dataset.")
    if st.button("Load and preview data", type="secondary"):
        try:
            preview = load_source(uploaded_file, path_text)
            st.session_state.preview = preview
        except Exception as error:
            st.error(str(error))
    if "preview" in st.session_state:
        preview = st.session_state.preview
        left, right = st.columns(2)
        left.metric("Rows", f"{len(preview):,}")
        right.metric("Columns", len(preview.columns))
        st.dataframe(preview.head(10), use_container_width=True)
        if st.button("Train models", type="primary"):
            try:
                features, target, target_name, fill_values = prepare_dataset(
                    preview, target_column.strip() or None
                )
                x_train, x_test, y_train, y_test, scaler = split_and_scale(features, target)
                save_preprocessing(scaler, list(features.columns), fill_values, target_name)
                forest = train_random_forest(x_train, y_train, DEFAULT_ARTIFACT_DIR)
                train_isolation_forest(x_train, y_train, DEFAULT_ARTIFACT_DIR)
                st.session_state.metrics = evaluate_classifier(forest, x_test, y_test)
                st.session_state.feature_columns = list(features.columns)
                st.success("Both models trained and saved in artifacts/.")
            except Exception as error:
                st.error(str(error))

with tab_inspector:
    st.subheader("Hybrid flow inference")
    if not artifact_ready():
        st.info("Train the models first. The inspector will then create inputs for every numeric feature.")
    else:
        metadata = __import__("joblib").load(DEFAULT_ARTIFACT_DIR / "preprocessing.joblib")
        feature_values = {}
        columns = st.columns(3)
        for index, feature in enumerate(metadata["feature_columns"]):
            feature_values[feature] = columns[index % 3].number_input(feature, value=0.0, format="%.6f")
        if st.button("Inspect flow", type="primary"):
            supervised_probability = predict_flow(feature_values)
            anomaly = predict_anomaly(feature_values)
            risk = calculate_risk(supervised_probability, float(anomaly["anomaly_score"]))
            first, second, third = st.columns(3)
            first.metric("Risk score", f'{risk["risk_score"]:.1%}')
            second.metric("Attack probability", f'{risk["supervised_probability"]:.1%}')
            third.metric("Anomaly score", f'{risk["anomaly_score"]:.1%}')
            st.markdown(f"### Severity: :{'red' if risk['severity'] == 'HIGH' else 'orange' if risk['severity'] == 'MEDIUM' else 'green'}[{risk['severity']}]")
            st.write("Isolation Forest status:", "ANOMALY" if anomaly["is_anomaly"] else "NORMAL")

with tab_metrics:
    st.subheader("Model performance")
    metrics = st.session_state.metrics
    if metrics is None:
        st.info("Train the models to populate performance metrics.")
    else:
        metric_columns = st.columns(6)
        for column, name in zip(metric_columns, ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]):
            column.metric(name.replace("_", " ").title(), f"{metrics[name]:.3f}")
        matrix = pd.DataFrame(metrics["confusion_matrix"], index=["Actual BENIGN", "Actual ATTACK"], columns=["Predicted BENIGN", "Predicted ATTACK"])
        figure, axis = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_xlabel("")
        axis.set_ylabel("")
        st.pyplot(figure, clear_figure=True)
        st.json(metrics["classification_report"])