"""
Objective 2 Phase 2B: Master Comparative Test Evaluation for All 4 Backbones.

Evaluates the best saved checkpoints of all 4 architectures on the completely
held-out CAMELYON16 TEST set:
  1. EfficientNet-B0 (result/objective_2/checkpoints/best_model.pt)
  2. ResNet-50       (result/objective_2/checkpoints/resnet/best_model.pt)
  3. DenseNet-121    (result/objective_2/checkpoints/densenet/best_model.pt)
  4. VGG-16          (result/objective_2/checkpoints/vgg/best_model.pt)

Computes complete diagnostic metrics:
  - Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score,
    Balanced Accuracy, ROC-AUC, PR-AUC, Confusion Matrix (TN, FP, FN, TP),
    Parameter Count, Checkpoint Size, and Inference Latency.

Saves:
  - result/objective_2/metrics/master_test_benchmark_comparison.csv
  - result/objective_2/metrics/master_test_benchmark_comparison.json
  - Individual test_metrics.json & test_classification_report.txt per model folder
  - result/objective_2/plots/master_comparative_roc_pr_curves.png
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.objective_2.anomaly_and_risk import compute_anomaly_scores
from src.objective_2.dataset import HistopathPatchDataset, get_eval_transforms
from src.objective_2.model import (
    DenseNet121TumorClassifier,
    EfficientNetB0TumorClassifier,
    ResNet50TumorClassifier,
    VGG16TumorClassifier,
)

EXPECTED_TEST_TOTAL = 2881
EXPECTED_TEST_NORMAL = 2563
EXPECTED_TEST_TUMOR = 318


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("MasterTestEval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits_list = []
    all_labels_list = []

    t0 = time.time()
    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)
        all_logits_list.append(logits.cpu().numpy())
        all_labels_list.append(labels.cpu().numpy())
    elapsed = time.time() - t0

    all_logits = np.concatenate(all_logits_list, axis=0).ravel()
    all_labels = np.concatenate(all_labels_list, axis=0).ravel()
    return all_logits, all_labels, elapsed


def evaluate_model_pipeline(
    model_name: str,
    model: torch.nn.Module,
    ckpt_path: Path,
    test_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    out_dir: Path,
) -> Dict:
    logger.info("=" * 70)
    logger.info(f"EVALUATING MODEL: {model_name.upper()}")
    logger.info(f"Checkpoint path: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ckpt_epoch = ckpt.get("epoch")
    ckpt_stage = ckpt.get("stage")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    param_summary = model.get_parameter_summary()
    total_params = param_summary["total_parameters"]
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)

    logger.info(f"Total parameters : {total_params:,}")
    logger.info(f"Checkpoint size  : {ckpt_size_mb:.2f} MB")
    logger.info(f"Checkpoint epoch : {ckpt_epoch} (Stage {ckpt_stage})")

    # Run inference
    all_logits, all_labels, elapsed_time = run_inference(model, test_loader, device)
    fps = len(all_labels) / elapsed_time if elapsed_time > 0 else 0

    probs = compute_anomaly_scores(all_logits)
    threshold = 0.5
    preds = (probs >= threshold).astype(int)

    # Calculate metrics
    acc = float(accuracy_score(all_labels, preds))
    prec = float(precision_score(all_labels, preds, zero_division=0))
    rec = float(recall_score(all_labels, preds, zero_division=0))
    f1 = float(f1_score(all_labels, preds, zero_division=0))
    bal_acc = float(balanced_accuracy_score(all_labels, preds))
    roc_auc = float(roc_auc_score(all_labels, probs))
    pr_auc = float(average_precision_score(all_labels, probs))

    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    logger.info(f"Inference Time   : {elapsed_time:.2f}s ({fps:.1f} patches/sec)")
    logger.info(f"Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"Balanced Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    logger.info(f"Precision        : {prec:.4f} ({prec*100:.2f}%)")
    logger.info(f"Recall (Sensitiv): {rec:.4f} ({rec*100:.2f}%)")
    logger.info(f"Specificity      : {specificity:.4f} ({specificity*100:.2f}%)")
    logger.info(f"F1-Score         : {f1:.4f}")
    logger.info(f"ROC-AUC          : {roc_auc:.4f}")
    logger.info(f"PR-AUC           : {pr_auc:.4f}")
    logger.info(f"Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    clf_report = classification_report(
        all_labels,
        preds,
        target_names=["Normal (0)", "Tumor (1)"],
        digits=4,
        zero_division=0,
    )

    # Save text report
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "test_classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"TEST SET CLASSIFICATION REPORT: {model_name.upper()}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint Epoch: {ckpt_epoch} (Stage {ckpt_stage})\n")
        f.write(f"Accuracy        : {acc:.4f}\n")
        f.write(f"Balanced Acc    : {bal_acc:.4f}\n")
        f.write(f"Precision       : {prec:.4f}\n")
        f.write(f"Recall          : {rec:.4f}\n")
        f.write(f"Specificity     : {specificity:.4f}\n")
        f.write(f"F1-Score        : {f1:.4f}\n")
        f.write(f"ROC-AUC         : {roc_auc:.4f}\n")
        f.write(f"PR-AUC          : {pr_auc:.4f}\n")
        f.write(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n\n")
        f.write(clf_report + "\n")

    result = {
        "model": model_name,
        "parameters": total_params,
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "best_epoch": ckpt_epoch,
        "best_stage": ckpt_stage,
        "inference_time_sec": round(elapsed_time, 2),
        "throughput_fps": round(fps, 1),
        "test_accuracy": acc,
        "test_balanced_accuracy": bal_acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_specificity": specificity,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "all_labels": all_labels,
        "all_probs": probs,
    }

    # Save model test metrics json
    json_path = out_dir / "test_metrics.json"
    clean_dict = {k: v for k, v in result.items() if k not in ["all_labels", "all_probs"]}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=4)

    return result


def main() -> None:
    test_dir = PROJECT_ROOT / "data" / "PATCHES" / "TEST"
    test_manifest = test_dir / "test_patch_manifest.csv"
    logs_dir = PROJECT_ROOT / "result" / "objective_2" / "logs"
    metrics_dir = PROJECT_ROOT / "result" / "objective_2" / "metrics"
    plots_dir = PROJECT_ROOT / "result" / "objective_2" / "plots"

    for d in [logs_dir, metrics_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "master_test_benchmark.log")
    logger.info("Initializing Master Benchmark Test Suite for All 4 Backbones...")

    device = torch.device("cpu")

    # Load dataset
    test_dataset = HistopathPatchDataset(
        patch_dir=test_dir,
        manifest_path=test_manifest,
        split_name="test",
        transform=get_eval_transforms(),
    )
    counts = test_dataset.get_class_counts()
    logger.info(f"Loaded {counts['total']} test patches ({counts['normal']} Normal, {counts['tumor']} Tumor)")

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    models_config = [
        (
            "efficientnet",
            EfficientNetB0TumorClassifier(pretrained=False),
            PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt",
            PROJECT_ROOT / "result" / "objective_2" / "metrics",
        ),
        (
            "resnet",
            ResNet50TumorClassifier(pretrained=False),
            PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "resnet" / "best_model.pt",
            PROJECT_ROOT / "result" / "objective_2" / "metrics" / "resnet",
        ),
        (
            "densenet",
            DenseNet121TumorClassifier(pretrained=False),
            PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "densenet" / "best_model.pt",
            PROJECT_ROOT / "result" / "objective_2" / "metrics" / "densenet",
        ),
        (
            "vgg",
            VGG16TumorClassifier(pretrained=False),
            PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "vgg" / "best_model.pt",
            PROJECT_ROOT / "result" / "objective_2" / "metrics" / "vgg",
        ),
    ]

    all_results = []
    for model_name, model_inst, ckpt_p, out_d in models_config:
        res = evaluate_model_pipeline(
            model_name=model_name,
            model=model_inst,
            ckpt_path=ckpt_p,
            test_loader=test_loader,
            device=device,
            logger=logger,
            out_dir=out_d,
        )
        all_results.append(res)

    # Compile master comparison table
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "Model": r["model"].upper(),
            "Parameters": r["parameters"],
            "Checkpoint (MB)": r["checkpoint_size_mb"],
            "Best Epoch": r["best_epoch"],
            "Inference Time (s)": r["inference_time_sec"],
            "Speed (FPS)": r["throughput_fps"],
            "Accuracy": round(r["test_accuracy"], 4),
            "Balanced Acc": round(r["test_balanced_accuracy"], 4),
            "Precision": round(r["test_precision"], 4),
            "Recall (Sensitivity)": round(r["test_recall"], 4),
            "Specificity": round(r["test_specificity"], 4),
            "F1-Score": round(r["test_f1"], 4),
            "ROC-AUC": round(r["test_roc_auc"], 4),
            "PR-AUC": round(r["test_pr_auc"], 4),
            "TP": r["tp"],
            "FP": r["fp"],
            "TN": r["tn"],
            "FN": r["fn"],
        })

    df_summary = pd.DataFrame(summary_rows)
    master_csv_path = metrics_dir / "master_test_benchmark_comparison.csv"
    master_json_path = metrics_dir / "master_test_benchmark_comparison.json"
    df_summary.to_csv(master_csv_path, index=False)
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=4)

    logger.info("\n" + "=" * 70)
    logger.info("MASTER COMPARATIVE BENCHMARK TABLE")
    logger.info("=" * 70)
    logger.info("\n" + df_summary.to_string(index=False))
    logger.info(f"\nMaster table saved to: {master_csv_path}")

    # Generate comparative ROC and PR curves
    plt.figure(figsize=(14, 6))

    # ROC Plot
    plt.subplot(1, 2, 1)
    colors = {"efficientnet": "#1f77b4", "resnet": "#2ca02c", "densenet": "#ff7f0e", "vgg": "#d62728"}
    for r in all_results:
        m = r["model"]
        fpr, tpr, _ = roc_curve(r["all_labels"], r["all_probs"])
        plt.plot(fpr, tpr, label=f"{m.upper()} (AUC = {r['test_roc_auc']:.4f})", color=colors.get(m, "gray"), lw=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Comparative ROC Curves (Held-Out Test Set)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # PR Plot
    plt.subplot(1, 2, 2)
    for r in all_results:
        m = r["model"]
        prec_curve, rec_curve, _ = precision_recall_curve(r["all_labels"], r["all_probs"])
        plt.plot(rec_curve, prec_curve, label=f"{m.upper()} (PR-AUC = {r['test_pr_auc']:.4f})", color=colors.get(m, "gray"), lw=2)
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (PPV)", fontsize=11)
    plt.title("Comparative Precision-Recall Curves (Held-Out Test Set)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    comp_plot_path = plots_dir / "master_comparative_roc_pr_curves.png"
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    logger.info(f"Master comparative plot saved to: {comp_plot_path}")
    logger.info("ALL BENCHMARK EVALUATIONS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
