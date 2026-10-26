"""
Objective 2 Metrics & Visualization Module.

Computes comprehensive diagnostic metrics for binary tumor detection:
- PR-AUC (Average Precision) - Primary model selection metric
- F1-Score - Secondary / tie-break metric
- ROC-AUC
- Accuracy, Precision, Recall / Sensitivity, Specificity
- Confusion Matrix (TN, FP, FN, TP)
- Publication-quality diagnostic plots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Union[float, int, Dict]]:
    """
    Computes all standard binary classification and ranking metrics.
    y_true: 1D binary ground-truth array (0 or 1)
    y_probs: 1D predicted tumor probability array (0.0 to 1.0)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_probs = np.asarray(y_probs).astype(float).ravel()
    y_pred = (y_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "sensitivity": rec,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "support": {
            "normal": int((y_true == 0).sum()),
            "tumor": int((y_true == 1).sum()),
            "total": int(len(y_true)),
        },
    }


def plot_training_curves(
    history: List[Dict],
    output_dir: Path,
) -> None:
    """Plots training/validation loss and F1/PR-AUC metric trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss (Weighted BCE)", color="#1f77b4", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], marker="s", label="Validation Loss", color="#d62728", linewidth=2)
    plt.axvline(x=3.5, color="gray", linestyle="--", alpha=0.7, label="Stage 1 -> Stage 2 Transition")
    plt.title("EfficientNet-B0: Training and Validation Loss Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    # 2. Validation Metrics Trajectory (F1 & PR-AUC)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_pr_auc"], marker="^", label="Validation PR-AUC (Primary Metric)", color="#2ca02c", linewidth=2)
    plt.plot(df["epoch"], df["val_f1"], marker="D", label="Validation F1-Score", color="#9467bd", linewidth=2)
    plt.plot(df["epoch"], df["val_recall"], marker="x", label="Validation Recall (Sensitivity)", color="#ff7f0e", linestyle=":", linewidth=1.5)
    plt.axvline(x=3.5, color="gray", linestyle="--", alpha=0.7, label="Stage 1 -> Stage 2 Transition")
    plt.title("EfficientNet-B0: Validation Metrics Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "f1_prauc_curve.png", dpi=300)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    roc_auc: float,
    output_path: Path,
) -> None:
    """Plots publication-quality ROC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"EfficientNet-B0 (ROC-AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    pr_auc: float,
    output_path: Path,
) -> None:
    """Plots publication-quality Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    prevalence = (y_true == 1).sum() / len(y_true)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#2ca02c", lw=2.5, label=f"EfficientNet-B0 (PR-AUC = {pr_auc:.4f})")
    plt.axhline(y=prevalence, color="gray", lw=1.5, linestyle="--", label=f"Baseline Prevalence ({prevalence:.4f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall (PR) Curve (Validation Set)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    output_path: Path,
) -> None:
    """Plots annotated Confusion Matrix heatmap."""
    tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    total = np.sum(matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["Normal (0)", "Tumor (1)"]
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Validation Confusion Matrix',
    )

    thresh = matrix.max() / 2.
    labels_names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            pct = count / total * 100
            name = labels_names[i][j]
            ax.text(
                j, i, f"{name}\n{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontweight="bold", fontsize=11,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_anomaly_and_risk_distribution(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    output_path: Path,
) -> None:
    """Plots histogram of model-based anomaly scores by class."""
    plt.figure(figsize=(8, 5))
    normal_scores = anomaly_scores[y_true == 0]
    tumor_scores = anomaly_scores[y_true == 1]

    plt.hist(normal_scores, bins=50, alpha=0.6, color="#1f77b4", label=f"Normal Patches (N={len(normal_scores)})", density=True)
    plt.hist(tumor_scores, bins=50, alpha=0.6, color="#d62728", label=f"Tumor Patches (N={len(tumor_scores)})", density=True)

    plt.axvline(x=0.30, color="orange", linestyle="--", linewidth=1.5, label="Risk Threshold (Low < 0.30 <= Med)")
    plt.axvline(x=0.70, color="red", linestyle="--", linewidth=1.5, label="Risk Threshold (Med < 0.70 <= High)")

    plt.title("Model-Based Anomaly Score Distribution by Ground Truth Class\n(P(tumor) = Anomaly Score)", fontsize=11, fontweight="bold")
    plt.xlabel("Anomaly Score: P(tumor)", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
