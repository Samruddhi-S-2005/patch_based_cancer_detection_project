"""
Objective 2 Anomaly Analysis and Risk Prediction Module.

Implements model-based anomaly scoring and project-level risk categorization
derived from the trained EfficientNet-B0 tumor probability.

DISCLAIMER:
"MODEL-BASED ANOMALY/RISK SCORE — NOT A CLINICALLY VALIDATED MEDICAL RISK SCORE."
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

DISCLAIMER_TEXT = "MODEL-BASED ANOMALY/RISK SCORE — NOT A CLINICALLY VALIDATED MEDICAL RISK SCORE."

LOW_RISK_MAX = 0.30
MED_RISK_MAX = 0.70


def compute_anomaly_scores(logits: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Computes patch-level model-based anomaly score:
    A = sigmoid(logit) = P(tumor)
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    else:
        logits_arr = np.asarray(logits).astype(float)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits_arr, -50.0, 50.0)))
    return probs.astype(float).ravel()


def categorize_risk(probs: np.ndarray) -> List[str]:
    """
    Categorizes predicted probabilities into project-level risk categories:
    - Low Risk:    P(tumor) < 0.30
    - Medium Risk: 0.30 <= P(tumor) < 0.70
    - High Risk:   P(tumor) >= 0.70
    """
    probs_flat = np.asarray(probs).ravel()
    categories = []
    for p in probs_flat:
        if p < LOW_RISK_MAX:
            categories.append("Low")
        elif p < MED_RISK_MAX:
            categories.append("Medium")
        else:
            categories.append("High")
    return categories


def build_predictions_dataframe(
    samples_metadata: List[Dict],
    logits: np.ndarray,
    classification_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Assembles complete prediction, anomaly, and risk records for evaluation.
    """
    logits_flat = np.asarray(logits).astype(float).ravel()
    probs = compute_anomaly_scores(logits_flat)
    risk_categories = categorize_risk(probs)
    predicted_labels = (probs >= classification_threshold).astype(int)

    rows = []
    for i, meta in enumerate(samples_metadata):
        rows.append({
            "filename": meta.get("filename", ""),
            "wsi_id": meta.get("wsi_id", ""),
            "x": int(meta.get("x", -1)),
            "y": int(meta.get("y", -1)),
            "true_label": int(meta["label"]),
            "true_class": "tumor" if int(meta["label"]) == 1 else "normal",
            "logit": float(logits_flat[i]),
            "tumor_prob": float(probs[i]),
            "anomaly_score": float(probs[i]),  # A = P(tumor)
            "predicted_label": int(predicted_labels[i]),
            "predicted_class": "tumor" if predicted_labels[i] == 1 else "normal",
            "risk_category": risk_categories[i],
        })

    df = pd.DataFrame(rows)
    return df


def generate_risk_summary(df: pd.DataFrame) -> Dict:
    """
    Generates summary of risk distribution cross-tabulated by ground truth class.
    """
    crosstab = pd.crosstab(df["true_class"], df["risk_category"], margins=True)
    crosstab_dict = crosstab.to_dict()

    total_patches = len(df)
    risk_counts = df["risk_category"].value_counts().to_dict()

    summary = {
        "disclaimer": DISCLAIMER_TEXT,
        "risk_thresholds": {
            "low": f"< {LOW_RISK_MAX:.2f}",
            "medium": f"[{LOW_RISK_MAX:.2f}, {MED_RISK_MAX:.2f})",
            "high": f">= {MED_RISK_MAX:.2f}",
        },
        "overall_distribution": {
            k: {"count": int(v), "percentage": float(f"{(v / total_patches) * 100:.2f}")}
            for k, v in risk_counts.items()
        },
        "breakdown_by_true_class": crosstab_dict,
    }
    return summary
