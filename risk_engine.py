"""Hybrid supervised and unsupervised risk scoring."""

from __future__ import annotations


def calculate_risk(supervised_probability: float, anomaly_score: float) -> dict[str, float | str]:
    supervised_probability = max(0.0, min(1.0, supervised_probability))
    anomaly_score = max(0.0, min(1.0, anomaly_score))
    risk_score = (0.7 * supervised_probability) + (0.3 * anomaly_score)
    if risk_score >= 0.7:
        severity = "HIGH"
    elif risk_score >= 0.4:
        severity = "MEDIUM"
    else:
        severity = "LOW"
    return {
        "risk_score": risk_score,
        "severity": severity,
        "supervised_probability": supervised_probability,
        "anomaly_score": anomaly_score,
    }