"""
Objective 2 Part B: Test Set Evaluation for Anomaly Linear Heads.

Evaluates the trained multi-head anomaly model on held-out TEST sets
using strictly frozen, validation-calibrated thresholds.

Tasks Evaluated:
  1. BACH (Ductal / DCIS Architecture Proxy)
  2. NuCLS (Tumor Nuclei Hyper-Density)
  3. BCSS (Tumor Necrosis / Debris)

Outputs generated under:
  result/objective_2/anomaly_model/
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.objective_2.anomaly_model import EfficientNetB0MultiHeadAnomaly
from src.objective_2.train_anomaly_heads import AnomalyManifestDataset, precompute_dataset_features


def compute_metrics_for_task(
    true_labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict:
    """Computes full set of binary classification metrics."""
    preds = (probs >= threshold).astype(int)
    preds_50 = (probs >= 0.50).astype(int)

    roc_auc = float(roc_auc_score(true_labels, probs))
    pr_auc = float(average_precision_score(true_labels, probs))

    cm = confusion_matrix(true_labels, preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(true_labels, preds, zero_division=0))
    f1 = float(f1_score(true_labels, preds, zero_division=0))
    f1_50 = float(f1_score(true_labels, preds_50, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(true_labels, preds))
    acc = float(accuracy_score(true_labels, preds))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "optimal_threshold": float(threshold),
        "f1_at_optimal_threshold": f1,
        "f1_at_0.50_threshold": f1_50,
        "balanced_accuracy": balanced_acc,
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "support": {
            "total_samples": int(len(true_labels)),
            "positive_samples": int(np.sum(true_labels == 1)),
            "negative_samples": int(np.sum(true_labels == 0)),
        },
    }


def main() -> None:
    out_dir = PROJECT_ROOT / "result" / "objective_2" / "anomaly_model"
    ckpt_dir = out_dir / "checkpoints"
    metrics_dir = out_dir / "metrics"
    preds_dir = out_dir / "predictions"
    logs_dir = out_dir / "logs"

    for d in [ckpt_dir, metrics_dir, preds_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = logging.getLogger("EvaluateAnomalyHeads")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(logs_dir / "evaluate_anomaly_heads.log", mode="w", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 70)
    logger.info("OBJECTIVE 2 PART B: HELD-OUT TEST EVALUATION FOR ANOMALY HEADS")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")

    # Load trained checkpoint — use clean 3-head checkpoint (remediation: atypia removed)
    ckpt_path = ckpt_dir / "anomaly_linear_heads_clean.pt"
    if not ckpt_path.exists():
        # Fallback: check for original name (will log a warning)
        ckpt_path = ckpt_dir / "anomaly_linear_heads_best.pt"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading anomaly checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model = EfficientNetB0MultiHeadAnomaly(device=device)
    # strict=False: clean checkpoint omits atypia weights by design (remediation).
    missing_keys, _ = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    permitted_missing = [k for k in missing_keys if any(t in k for t in ["ductal", "density", "necrosis"])]
    if permitted_missing:
        raise RuntimeError(f"Missing keys for permitted heads: {permitted_missing}")
    logger.info(f"  Note: {len(missing_keys)} key(s) left at random init (atypia head — not used)")
    model.to(device)
    model.eval()

    optimal_thresholds = checkpoint["optimal_thresholds"]
    logger.info(f"Loaded validation-calibrated thresholds: {optimal_thresholds}")

    datasets_info = {
        "atypia": {
            "name": "MITOS-ATYPIA14 (Severe Nuclear Pleomorphism)",
            "manifest": PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "mitos_atypia" / "mitos_atypia_patch_manifest.csv",
            "base_dir": PROJECT_ROOT,
            "pred_file": preds_dir / "mitos_atypia_test_predictions.csv",
            "pos_label_name": "severe_atypia_score3",
            "neg_label_name": "mild_mod_atypia_score1_2",
        },
        "ductal": {
            "name": "BACH (Ductal/DCIS Proxy)",
            "manifest": PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "bach_ductal_proxy" / "bach_ductal_proxy_patch_manifest.csv",
            "base_dir": PROJECT_ROOT,
            "pred_file": preds_dir / "bach_ductal_test_predictions.csv",
            "pos_label_name": "ductal_architecture_proxy",
            "neg_label_name": "non_ductal_proxy",
        },
        "density": {
            "name": "NuCLS (High Tumor Density)",
            "manifest": PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "nucls_density" / "nucls_density_patch_manifest.csv",
            "base_dir": PROJECT_ROOT,
            "pred_file": preds_dir / "nucls_density_test_predictions.csv",
            "pos_label_name": "high_tumor_density",
            "neg_label_name": "no_tumor_nuclei",
        },
        "necrosis": {
            "name": "BCSS (Necrosis/Debris)",
            "manifest": PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "bcss_necrosis" / "bcss_necrosis_patch_manifest.csv",
            "base_dir": PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "bcss_necrosis",
            "pred_file": preds_dir / "bcss_necrosis_test_predictions.csv",
            "pos_label_name": "necrosis",
            "neg_label_name": "non_necrosis",
        },
    }

    all_test_metrics = {}
    summary_rows = []
    classification_reports_text = []

    for task, info in datasets_info.items():
        logger.info("=" * 70)
        logger.info(f"EVALUATING TEST SET: {task.upper()} — {info['name']}")
        logger.info("=" * 70)

        tau = optimal_thresholds[task]

        if info["pred_file"].exists():
            logger.info(f"  [REUSING] Loading existing test predictions from {info['pred_file'].name}")
            df_pred = pd.read_csv(info["pred_file"])
            lbl_arr = (df_pred["true_label"] if "true_label" in df_pred.columns else df_pred["label"]).values.astype(int)
            probs = (df_pred["anomaly_probability"] if "anomaly_probability" in df_pred.columns else df_pred["pred_prob"]).values.astype(float)
            test_meta = []
        else:
            test_ds = AnomalyManifestDataset(info["manifest"], split="test", base_dir=info["base_dir"])
            logger.info(f"  Test dataset size: {len(test_ds):,} patches")

            # Extract features
            t0 = time.time()
            test_feat, test_lbl, test_meta = precompute_dataset_features(model, test_ds, device)
            logger.info(f"  Feature extraction complete in {time.time() - t0:.2f}s")

            # Inference through task head
            test_feat_dev = test_feat.to(device)
            with torch.no_grad():
                logits = model.forward_features(test_feat_dev, task).cpu().numpy().ravel()
                probs = 1.0 / (1.0 + np.exp(-logits))

            lbl_arr = test_lbl.numpy().ravel().astype(int)

        metrics = compute_metrics_for_task(lbl_arr, probs, tau)
        all_test_metrics[task] = metrics

        logger.info(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC (AP)   : {metrics['pr_auc']:.4f}")
        logger.info(f"  Optimal Tau   : {metrics['optimal_threshold']:.4f}")
        logger.info(f"  F1 @ Tau      : {metrics['f1_at_optimal_threshold']:.4f}")
        logger.info(f"  F1 @ 0.50     : {metrics['f1_at_0.50_threshold']:.4f}")
        logger.info(f"  Balanced Acc  : {metrics['balanced_accuracy']:.4f}")
        logger.info(f"  Sensitivity   : {metrics['sensitivity']:.4f}")
        logger.info(f"  Specificity   : {metrics['specificity']:.4f}")
        logger.info(f"  Precision     : {metrics['precision']:.4f}")
        logger.info(f"  Confusion Mat : TN={metrics['confusion_matrix']['true_negative']}, FP={metrics['confusion_matrix']['false_positive']}, FN={metrics['confusion_matrix']['false_negative']}, TP={metrics['confusion_matrix']['true_positive']}")

        # Build prediction dataframe
        pred_labels = (probs >= tau).astype(int)
        risk_cats = []
        for p in probs:
            if p < 0.30:
                risk_cats.append("Low")
            elif p < 0.70:
                risk_cats.append("Medium")
            else:
                risk_cats.append("High")

        if test_meta:
            pred_rows = []
            for i, meta in enumerate(test_meta):
                pred_rows.append({
                    "patch_path": meta["path"],
                    "source_id": meta["source_id"],
                    "x": meta["x"],
                    "y": meta["y"],
                    "true_label": int(lbl_arr[i]),
                    "true_class": info["pos_label_name"] if lbl_arr[i] == 1 else info["neg_label_name"],
                    "logit": float(logits[i]),
                    "anomaly_probability": float(probs[i]),
                    "predicted_label": int(pred_labels[i]),
                    "predicted_class": info["pos_label_name"] if pred_labels[i] == 1 else info["neg_label_name"],
                    "risk_category": risk_cats[i],
                })

            df_preds = pd.DataFrame(pred_rows)
            df_preds.to_csv(info["pred_file"], index=False)
            logger.info(f"  Saved Predictions CSV: {info['pred_file']}")

        # Summary row
        summary_rows.append({
            "task": task,
            "dataset": info["name"],
            "test_samples": len(lbl_arr),
            "test_positives": metrics["support"]["positive_samples"],
            "test_negatives": metrics["support"]["negative_samples"],
            "optimal_threshold": round(metrics["optimal_threshold"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "pr_auc": round(metrics["pr_auc"], 4),
            "f1_at_optimal_tau": round(metrics["f1_at_optimal_threshold"], 4),
            "f1_at_0.50": round(metrics["f1_at_0.50_threshold"], 4),
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            "sensitivity": round(metrics["sensitivity"], 4),
            "specificity": round(metrics["specificity"], 4),
            "precision": round(metrics["precision"], 4),
            "true_negatives": metrics["confusion_matrix"]["true_negative"],
            "false_positives": metrics["confusion_matrix"]["false_positive"],
            "false_negatives": metrics["confusion_matrix"]["false_negative"],
            "true_positives": metrics["confusion_matrix"]["true_positive"],
        })

        # Classification report string
        cr = classification_report(
            lbl_arr,
            pred_labels,
            target_names=[info["neg_label_name"], info["pos_label_name"]],
            digits=4,
        )
        classification_reports_text.append(f"TASK: {task.upper()} ({info['name']})\nOptimal Threshold: {tau:.4f}\n{cr}\n" + "=" * 70 + "\n")

    # Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = metrics_dir / "anomaly_test_metrics_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info(f"\nSaved Metrics Summary CSV: {summary_csv_path}")

    # Save JSON metrics
    json_metrics_path = metrics_dir / "anomaly_test_metrics.json"
    with open(json_metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_test_metrics, f, indent=2)
    logger.info(f"Saved Test Metrics JSON: {json_metrics_path}")

    # Save classification reports text
    cr_path = metrics_dir / "classification_reports.txt"
    with open(cr_path, "w", encoding="utf-8") as f:
        f.writelines(classification_reports_text)
    logger.info(f"Saved Classification Reports: {cr_path}")

    logger.info("=" * 70)
    logger.info("ALL HELD-OUT ANOMALY TEST EVALUATIONS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
