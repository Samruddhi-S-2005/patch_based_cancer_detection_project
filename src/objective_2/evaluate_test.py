"""
Objective 2 Phase 2B: Held-Out TEST Set Evaluation.

Evaluates the best saved EfficientNet-B0 checkpoint (Epoch 4, best validation
PR-AUC) on the completely held-out TEST set.

IMPORTANT CONSTRAINTS:
- Loads ONLY result/objective_2/checkpoints/best_model.pt (Epoch 4).
- Runs on data/PATCHES/TEST ONLY (2,881 patches, 8 WSIs).
- Uses deterministic get_eval_transforms() -- NO augmentation.
- Cross-checks labels against test_patch_manifest.csv;
  any mismatch raises an error (not silently corrected).
- Does NOT touch training or validation data, splits, or checkpoints.
- All outputs saved under result/objective_2/ with test_ prefix.

DISCLAIMER: MODEL-BASED ANOMALY/RISK SCORE -- NOT A CLINICALLY VALIDATED
MEDICAL RISK SCORE.
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
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.objective_2.anomaly_and_risk import (
    DISCLAIMER_TEXT,
    build_predictions_dataframe,
    compute_anomaly_scores,
    generate_risk_summary,
)
from src.objective_2.dataset import HistopathPatchDataset, get_eval_transforms
from src.objective_2.metrics import (
    compute_metrics,
    plot_anomaly_and_risk_distribution,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from src.objective_2.model import EfficientNetB0TumorClassifier


# ---------------------------------------------------------------------------
# Expected constants (from frozen Objective-1 + verified audit)
# ---------------------------------------------------------------------------
EXPECTED_TEST_TOTAL    = 2881
EXPECTED_TEST_NORMAL   = 2563
EXPECTED_TEST_TUMOR    = 318
EXPECTED_TEST_WSI_IDS  = {
    "normal_004", "normal_012", "normal_083", "normal_103",
    "tumor_005",  "tumor_058",  "tumor_093",  "tumor_101",
}
EXPECTED_BEST_EPOCH    = 4


def setup_logger(log_file: Path) -> logging.Logger:
    """Configures structured logging to console and file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Objective2_TestEval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Runs deterministic inference over the DataLoader.
    Returns (all_logits, all_labels, all_metadata).
    No loss computed -- test evaluation is label-blind for loss purposes.
    """
    model.eval()
    all_logits_list  = []
    all_labels_list  = []
    all_metadata: List[Dict] = []

    for images, labels, meta in loader:
        images = images.to(device)
        logits = model(images)

        all_logits_list.append(logits.cpu().numpy())
        all_labels_list.append(labels.cpu().numpy())

        b_size = images.size(0)
        for i in range(b_size):
            all_metadata.append({
                "filename": meta["filename"][i],
                "wsi_id":   meta["wsi_id"][i],
                "x":        int(meta["x"][i]),
                "y":        int(meta["y"][i]),
                "label":    int(meta["label"][i]),
            })

    all_logits = np.concatenate(all_logits_list, axis=0).ravel()
    all_labels = np.concatenate(all_labels_list, axis=0).ravel()
    return all_logits, all_labels, all_metadata


def run_test_evaluation() -> None:

    # ------------------------------------------------------------------
    # 1. Paths
    # ------------------------------------------------------------------
    test_dir      = PROJECT_ROOT / "data"   / "PATCHES" / "TEST"
    test_manifest = test_dir / "test_patch_manifest.csv"
    ckpt_path     = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt"

    out_base   = PROJECT_ROOT / "result" / "objective_2"
    metrics_dir = out_base / "metrics"
    plots_dir   = out_base / "plots"
    preds_dir   = out_base / "predictions"
    logs_dir    = out_base / "logs"

    for d in [metrics_dir, plots_dir, preds_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "test_eval.log")

    logger.info("=" * 70)
    logger.info("OBJECTIVE 2 PHASE 2B: HELD-OUT TEST SET EVALUATION")
    logger.info("=" * 70)
    logger.info(f"DISCLAIMER: {DISCLAIMER_TEXT}")
    logger.info("")

    # ------------------------------------------------------------------
    # 2. Pre-flight path checks
    # ------------------------------------------------------------------
    logger.info("[CHECK] Verifying required paths exist...")
    for p in [test_dir, test_manifest, ckpt_path]:
        if not p.exists():
            logger.error(f"MISSING required path: {p}")
            sys.exit(1)
        logger.info(f"  [OK] {p}")

    # ------------------------------------------------------------------
    # 3. Load checkpoint -- STRICT: must be epoch 4
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[CHECKPOINT] Loading best_model.pt...")
    device = torch.device("cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    ckpt_epoch = ckpt.get("epoch")
    ckpt_stage = ckpt.get("stage")
    ckpt_metrics = ckpt.get("metrics", {})
    logger.info(f"  Checkpoint epoch : {ckpt_epoch} (expected: {EXPECTED_BEST_EPOCH})")
    logger.info(f"  Checkpoint stage : {ckpt_stage}")
    logger.info(f"  Val PR-AUC stored: {ckpt_metrics.get('pr_auc', 'N/A')}")
    logger.info(f"  Val F1 stored    : {ckpt_metrics.get('f1_score', 'N/A')}")

    if ckpt_epoch != EXPECTED_BEST_EPOCH:
        logger.error(
            f"CHECKPOINT EPOCH MISMATCH: expected {EXPECTED_BEST_EPOCH}, "
            f"found {ckpt_epoch}. STOPPING."
        )
        sys.exit(1)

    logger.info("  [OK] Checkpoint epoch verified.")

    # ------------------------------------------------------------------
    # 4. Model -- load weights, set eval mode
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[MODEL] Initializing EfficientNet-B0 (pretrained=False for weight loading)...")
    model = EfficientNetB0TumorClassifier(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    param_s = model.get_parameter_summary()
    logger.info(f"  Total parameters: {param_s['total_parameters']:,}")
    logger.info("  [OK] Model weights loaded. model.eval() set.")

    # ------------------------------------------------------------------
    # 5. Test Dataset -- deterministic, no augmentation
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[DATASET] Loading TEST dataset with eval transforms (no augmentation)...")
    logger.info(f"  patch_dir    : {test_dir}")
    logger.info(f"  manifest     : {test_manifest}")
    logger.info("  transform    : get_eval_transforms() [ToTensor + ImageNet Normalize only]")
    logger.info("  Label source : filename regex label{N} + manifest cross-check")

    test_dataset = HistopathPatchDataset(
        patch_dir=test_dir,
        manifest_path=test_manifest,
        split_name="test",
        transform=get_eval_transforms(),
    )

    counts = test_dataset.get_class_counts()
    wsi_ids = set(test_dataset.get_wsi_ids())

    logger.info(f"  Normal patches : {counts['normal']:,} (expected: {EXPECTED_TEST_NORMAL:,})")
    logger.info(f"  Tumor patches  : {counts['tumor']:,} (expected: {EXPECTED_TEST_TUMOR:,})")
    logger.info(f"  Total patches  : {counts['total']:,} (expected: {EXPECTED_TEST_TOTAL:,})")
    logger.info(f"  WSI IDs found  : {sorted(wsi_ids)}")

    # Strict count assertion
    if counts["total"] != EXPECTED_TEST_TOTAL:
        logger.error(
            f"TEST PATCH COUNT MISMATCH: expected {EXPECTED_TEST_TOTAL}, "
            f"got {counts['total']}. STOPPING."
        )
        sys.exit(1)
    if counts["normal"] != EXPECTED_TEST_NORMAL:
        logger.error(
            f"NORMAL PATCH COUNT MISMATCH: expected {EXPECTED_TEST_NORMAL}, "
            f"got {counts['normal']}. STOPPING."
        )
        sys.exit(1)
    if counts["tumor"] != EXPECTED_TEST_TUMOR:
        logger.error(
            f"TUMOR PATCH COUNT MISMATCH: expected {EXPECTED_TEST_TUMOR}, "
            f"got {counts['tumor']}. STOPPING."
        )
        sys.exit(1)

    # WSI disjointness assertion
    unexpected_wsi = wsi_ids - EXPECTED_TEST_WSI_IDS
    if unexpected_wsi:
        logger.error(f"UNEXPECTED WSI IDs in test set: {unexpected_wsi}. STOPPING.")
        sys.exit(1)
    missing_wsi = EXPECTED_TEST_WSI_IDS - wsi_ids
    if missing_wsi:
        logger.error(f"MISSING WSI IDs from test set: {missing_wsi}. STOPPING.")
        sys.exit(1)

    logger.info("  [OK] All count and WSI ID checks passed.")

    # DataLoader -- shuffle=False for determinism
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    logger.info(f"  DataLoader batches: {len(test_loader)} (batch_size=64, shuffle=False)")

    # ------------------------------------------------------------------
    # 6. Inference
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[INFERENCE] Running model inference on TEST set...")
    t0 = time.time()
    all_logits, all_labels, all_metadata = run_inference(model, test_loader, device)
    elapsed = time.time() - t0
    logger.info(f"  Inference complete in {elapsed:.1f}s for {len(all_labels)} patches.")

    # ------------------------------------------------------------------
    # 7. Metrics
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[METRICS] Computing test set metrics (threshold=0.5)...")
    test_probs = compute_anomaly_scores(all_logits)
    metrics = compute_metrics(all_labels, test_probs, threshold=0.5)

    cm = metrics["confusion_matrix"]
    logger.info(f"  Accuracy        : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision       : {metrics['precision']:.4f}")
    logger.info(f"  Recall/Sensitiv : {metrics['recall']:.4f}")
    logger.info(f"  Specificity     : {metrics['specificity']:.4f}")
    logger.info(f"  F1-Score        : {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC         : {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC          : {metrics['pr_auc']:.4f}")
    logger.info(f"  Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")

    # ------------------------------------------------------------------
    # 8. Predictions dataframe + anomaly/risk
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[ANOMALY/RISK] Building per-patch predictions, anomaly scores, risk categories...")
    df_preds = build_predictions_dataframe(all_metadata, all_logits, classification_threshold=0.5)
    risk_summary = generate_risk_summary(df_preds)

    overall = risk_summary["overall_distribution"]
    breakdown = risk_summary["breakdown_by_true_class"]
    logger.info(f"  Low Risk    : {overall.get('Low', {}).get('count', 0)} patches ({overall.get('Low', {}).get('percentage', 0)}%)")
    logger.info(f"  Medium Risk : {overall.get('Medium', {}).get('count', 0)} patches ({overall.get('Medium', {}).get('percentage', 0)}%)")
    logger.info(f"  High Risk   : {overall.get('High', {}).get('count', 0)} patches ({overall.get('High', {}).get('percentage', 0)}%)")
    logger.info(f"  Risk breakdown by true class:\n{json.dumps(breakdown, indent=4)}")

    # ------------------------------------------------------------------
    # 9. Save predictions CSV
    # ------------------------------------------------------------------
    test_pred_path = preds_dir / "test_predictions.csv"
    df_preds.to_csv(test_pred_path, index=False)
    logger.info(f"\n[SAVE] Test predictions CSV saved: {test_pred_path} (Rows: {len(df_preds)})")

    # ------------------------------------------------------------------
    # 10. Classification report (text)
    # ------------------------------------------------------------------
    y_pred_binary = (test_probs >= 0.5).astype(int)
    clf_report_str = classification_report(
        all_labels,
        y_pred_binary,
        target_names=["Normal (0)", "Tumor (1)"],
        digits=4,
        zero_division=0,
    )
    clf_report_path = metrics_dir / "test_classification_report.txt"
    with open(clf_report_path, "w", encoding="utf-8") as f:
        f.write("OBJECTIVE 2 PHASE 2B: EFFICIENTNET-B0 TEST SET CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint Used : best_model.pt (Epoch {ckpt_epoch}, Stage {ckpt_stage})\n")
        f.write(f"Test Set        : data/PATCHES/TEST ({counts['total']} patches, {len(wsi_ids)} WSIs)\n")
        f.write(f"Transform       : get_eval_transforms() [NO augmentation]\n")
        f.write(f"Threshold       : 0.5\n")
        f.write("\n--- TEST SET METRICS ---\n")
        f.write(f"PR-AUC          : {metrics['pr_auc']:.4f}\n")
        f.write(f"ROC-AUC         : {metrics['roc_auc']:.4f}\n")
        f.write(f"F1-Score        : {metrics['f1_score']:.4f}\n")
        f.write(f"Recall          : {metrics['recall']:.4f}\n")
        f.write(f"Precision       : {metrics['precision']:.4f}\n")
        f.write(f"Specificity     : {metrics['specificity']:.4f}\n")
        f.write(f"Accuracy        : {metrics['accuracy']:.4f}\n")
        f.write("\n")
        f.write(clf_report_str + "\n")
        f.write("CONFUSION MATRIX:\n")
        f.write(f"TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}, TP: {cm['tp']}\n\n")
        f.write("--- VALIDATION SET METRICS (for comparison, from best checkpoint) ---\n")
        f.write(f"Val PR-AUC  : {ckpt_metrics.get('pr_auc', 'N/A'):.4f}\n")
        f.write(f"Val F1      : {ckpt_metrics.get('f1_score', 'N/A'):.4f}\n")
        f.write(f"Val Recall  : {ckpt_metrics.get('recall', 'N/A'):.4f}\n")
        f.write(f"Val Accuracy: {ckpt_metrics.get('accuracy', 'N/A'):.4f}\n")
        val_cm = ckpt_metrics.get("confusion_matrix", {})
        f.write(f"Val CM      : TN={val_cm.get('tn')}, FP={val_cm.get('fp')}, FN={val_cm.get('fn')}, TP={val_cm.get('tp')}\n")
        f.write(f"\nDISCLAIMER: {DISCLAIMER_TEXT}\n")
    logger.info(f"[SAVE] Test classification report saved: {clf_report_path}")

    # ------------------------------------------------------------------
    # 11. Test metrics JSON
    # ------------------------------------------------------------------
    test_metrics_payload = {
        "phase": "test_evaluation",
        "checkpoint": {
            "path": str(ckpt_path),
            "epoch": ckpt_epoch,
            "stage": ckpt_stage,
            "val_pr_auc": ckpt_metrics.get("pr_auc"),
            "val_f1": ckpt_metrics.get("f1_score"),
        },
        "test_set": {
            "path": str(test_dir),
            "total_patches": int(counts["total"]),
            "normal_patches": int(counts["normal"]),
            "tumor_patches": int(counts["tumor"]),
            "wsi_ids": sorted(list(wsi_ids)),
        },
        "test_metrics": {
            "accuracy":     float(metrics["accuracy"]),
            "precision":    float(metrics["precision"]),
            "recall":       float(metrics["recall"]),
            "sensitivity":  float(metrics["recall"]),
            "specificity":  float(metrics["specificity"]),
            "f1_score":     float(metrics["f1_score"]),
            "roc_auc":      float(metrics["roc_auc"]),
            "pr_auc":       float(metrics["pr_auc"]),
            "threshold":    0.5,
            "confusion_matrix": {
                "tn": cm["tn"], "fp": cm["fp"],
                "fn": cm["fn"], "tp": cm["tp"],
                "matrix": cm["matrix"],
            },
            "support": metrics["support"],
        },
        "risk_summary": risk_summary,
        "disclaimer": DISCLAIMER_TEXT,
    }
    test_metrics_path = metrics_dir / "test_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics_payload, f, indent=2)
    logger.info(f"[SAVE] Test metrics JSON saved: {test_metrics_path}")

    # ------------------------------------------------------------------
    # 12. Test metrics summary CSV (mirrors validation_metrics_summary.csv)
    # ------------------------------------------------------------------
    df_test_summary = pd.DataFrame([{
        "phase":            "test",
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_stage": ckpt_stage,
        "test_pr_auc":      metrics["pr_auc"],
        "test_f1":          metrics["f1_score"],
        "test_recall":      metrics["recall"],
        "test_precision":   metrics["precision"],
        "test_specificity": metrics["specificity"],
        "test_accuracy":    metrics["accuracy"],
        "test_roc_auc":     metrics["roc_auc"],
        "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
        "total_test_samples": int(counts["total"]),
        "normal_samples":     int(counts["normal"]),
        "tumor_samples":      int(counts["tumor"]),
        # Validation reference (from checkpoint)
        "val_pr_auc_ref":   ckpt_metrics.get("pr_auc"),
        "val_f1_ref":       ckpt_metrics.get("f1_score"),
    }])
    test_summary_path = metrics_dir / "test_metrics_summary.csv"
    df_test_summary.to_csv(test_summary_path, index=False)
    logger.info(f"[SAVE] Test metrics summary CSV saved: {test_summary_path}")

    # ------------------------------------------------------------------
    # 13. Diagnostic plots
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[PLOTS] Generating test set diagnostic plots...")

    # ROC Curve
    plot_roc_curve(
        all_labels, test_probs, metrics["roc_auc"],
        plots_dir / "test_roc_curve.png",
    )
    # Overwrite title for test context -- done via patched call below
    _patch_test_roc_title(all_labels, test_probs, metrics["roc_auc"], plots_dir / "test_roc_curve.png")
    logger.info("  [OK] test_roc_curve.png")

    # PR Curve
    _plot_test_pr_curve(all_labels, test_probs, metrics["pr_auc"], plots_dir / "test_pr_curve.png")
    logger.info("  [OK] test_pr_curve.png")

    # Confusion Matrix
    _plot_test_confusion_matrix(cm, plots_dir / "test_confusion_matrix.png")
    logger.info("  [OK] test_confusion_matrix.png")

    # Anomaly / Risk distribution
    plot_anomaly_and_risk_distribution(
        all_labels, test_probs,
        plots_dir / "test_anomaly_risk_distribution.png",
    )
    logger.info("  [OK] test_anomaly_risk_distribution.png")

    logger.info("[PLOTS] All test diagnostic plots generated.")

    # ------------------------------------------------------------------
    # 14. Final summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("OBJECTIVE 2 PHASE 2B -- TEST SET EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Test PR-AUC  : {metrics['pr_auc']:.4f}  (Val PR-AUC: {ckpt_metrics.get('pr_auc', 'N/A'):.4f})")
    logger.info(f"  Test ROC-AUC : {metrics['roc_auc']:.4f}  (Val ROC-AUC: {ckpt_metrics.get('roc_auc', 'N/A'):.4f})")
    logger.info(f"  Test F1      : {metrics['f1_score']:.4f}  (Val F1: {ckpt_metrics.get('f1_score', 'N/A'):.4f})")
    logger.info(f"  Test Recall  : {metrics['recall']:.4f}  (Val Recall: {ckpt_metrics.get('recall', 'N/A'):.4f})")
    logger.info(f"  Test Acc     : {metrics['accuracy']:.4f}  (Val Acc: {ckpt_metrics.get('accuracy', 'N/A'):.4f})")
    logger.info(f"  Test CM      : TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    logger.info(f"  DISCLAIMER   : {DISCLAIMER_TEXT}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Local plot helpers -- test-specific titles/labels
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def _patch_test_roc_title(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    roc_auc: float,
    output_path: Path,
) -> None:
    """Replaces the ROC curve with test-set-specific title."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5,
             label=f"EfficientNet-B0 (ROC-AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
             label="Random Classifier (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve -- TEST SET",
              fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_test_pr_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    pr_auc: float,
    output_path: Path,
) -> None:
    """Plots PR Curve with test-set-specific title and prevalence."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    prevalence = float((y_true == 1).sum()) / len(y_true)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#2ca02c", lw=2.5,
             label=f"EfficientNet-B0 (PR-AUC = {pr_auc:.4f})")
    plt.axhline(y=prevalence, color="gray", lw=1.5, linestyle="--",
                label=f"Baseline Prevalence ({prevalence:.4f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall (PR) Curve -- TEST SET",
              fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_test_confusion_matrix(
    cm_dict: Dict[str, int],
    output_path: Path,
) -> None:
    """Plots confusion matrix with test-set-specific title."""
    tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    total = int(np.sum(matrix))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["Normal (0)", "Tumor (1)"]
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Test Set Confusion Matrix",
    )

    thresh = matrix.max() / 2.0
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


if __name__ == "__main__":
    run_test_evaluation()
