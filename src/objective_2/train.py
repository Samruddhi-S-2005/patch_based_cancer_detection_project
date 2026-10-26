"""
Objective 2 Main Training Pipeline: EfficientNet-B0 Binary Tumor Classifier.

Executes:
- 2-Stage Transfer Learning:
  * Stage 1: Frozen backbone + Trainable Head (3 epochs, lr=1e-3, AdamW)
  * Stage 2: Unfreeze top feature blocks (features[6:]) (3 epochs, lr=1e-4, AdamW)
- Exact class-weighted BCE loss (pos_weight = 13129 / 1590 = 8.2572)
- Comprehensive validation evaluation after each epoch
- Model selection based on Validation PR-AUC (primary) and F1 (secondary)
- Post-training Anomaly Analysis and Project-Level Risk Categorization
- Publication-quality diagnostic plots and reproducibility logging
"""

from __future__ import annotations

import json
import logging
import os
import random
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
import torch.nn as nn
from sklearn.metrics import classification_report

from src.objective_2.anomaly_and_risk import (
    DISCLAIMER_TEXT,
    build_predictions_dataframe,
    compute_anomaly_scores,
    generate_risk_summary,
)
from src.objective_2.dataset import create_dataloaders
from src.objective_2.metrics import (
    compute_metrics,
    plot_anomaly_and_risk_distribution,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
)
from src.objective_2.model import (
    EfficientNetB0TumorClassifier,
    ResNet50TumorClassifier,
    DenseNet121TumorClassifier,
    VGG16TumorClassifier,
)


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for full determinism and reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file: Path) -> logging.Logger:
    """Configures structured logging to console and file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Objective2_Train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
) -> float:
    """Trains the model for one epoch and returns average weighted training loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    start_time = time.time()
    num_batches = len(loader)

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == num_batches:
            batch_loss = loss.item()
            progress_pct = (batch_idx + 1) / num_batches * 100
            elapsed = time.time() - start_time
            logger.info(
                f"  Epoch {epoch} [{batch_idx + 1}/{num_batches}] ({progress_pct:.1f}%) "
                f"- Batch Loss: {batch_loss:.4f} - Elapsed: {elapsed:.1f}s"
            )

    epoch_loss = running_loss / total_samples
    return epoch_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, List[Dict]]:
    """
    Evaluates model on validation loader.
    Returns: (val_loss, all_logits, all_labels, all_metadata)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_logits_list = []
    all_labels_list = []
    all_metadata: List[Dict] = []

    for images, labels, meta in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        all_logits_list.append(logits.cpu().numpy())
        all_labels_list.append(labels.cpu().numpy())

        # Extract metadata
        b_size = images.size(0)
        for i in range(b_size):
            all_metadata.append({
                "filename": meta["filename"][i],
                "wsi_id": meta["wsi_id"][i],
                "x": int(meta["x"][i]),
                "y": int(meta["y"][i]),
                "label": int(meta["label"][i]),
            })

    val_loss = running_loss / total_samples
    all_logits = np.concatenate(all_logits_list, axis=0).ravel()
    all_labels = np.concatenate(all_labels_list, axis=0).ravel()
    return val_loss, all_logits, all_labels, all_metadata


def run_training(arch: str = "efficientnet") -> None:
    # 1. Paths & Directory Setup
    train_dir = PROJECT_ROOT / "data" / "PATCHES" / "CLEANED"
    val_dir = PROJECT_ROOT / "data" / "PATCHES" / "VALIDATION"
    val_manifest = PROJECT_ROOT / "data" / "PATCHES" / "VALIDATION" / "validation_patch_manifest.csv"

    out_base = PROJECT_ROOT / "result" / "objective_2"
    ckpt_dir = out_base / "checkpoints" / arch
    metrics_dir = out_base / "metrics" / arch
    plots_dir = out_base / "plots" / arch
    preds_dir = out_base / "predictions" / arch
    logs_dir = out_base / "logs" / arch

    for d in [ckpt_dir, metrics_dir, plots_dir, preds_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "train.log")
    logger.info("=" * 70)
    logger.info(f"OBJECTIVE 2: {arch.upper()} TRANSFER LEARNING & ANOMALY ANALYSIS")
    logger.info("=" * 70)

    # 2. Hyperparameters & Configuration
    seed = 42
    batch_size = 64
    stage1_epochs = 3
    stage2_epochs = 3
    total_epochs = stage1_epochs + stage2_epochs
    stage1_lr = 1e-3
    stage2_lr = 1e-4
    weight_decay = 1e-4
    dropout_rate = 0.2

    set_seed(seed)
    device = torch.device("cpu")
    logger.info(f"Execution Device: {device} (PyTorch {torch.__version__})")
    logger.info(f"Random Seed: {seed}")
    logger.info(f"Batch Size: {batch_size}")

    # 3. DataLoaders
    logger.info("\nLoading Training & Validation Datasets...")
    train_loader, val_loader, train_ds, val_ds = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        val_manifest=val_manifest,
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )

    train_counts = train_ds.get_class_counts()
    val_counts = val_ds.get_class_counts()
    logger.info(f"Train Dataset: Normal={train_counts['normal']:,}, Tumor={train_counts['tumor']:,}, Total={train_counts['total']:,}")
    logger.info(f"Validation Dataset: Normal={val_counts['normal']:,}, Tumor={val_counts['tumor']:,}, Total={val_counts['total']:,}")

    # 4. Class Imbalance & Loss Function
    # Strictly calculated from training set: 13,129 normal / 1,590 tumor = 8.2572327
    pos_weight_val = train_counts["normal"] / train_counts["tumor"]
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"Training Class Weight pos_weight = {pos_weight_val:.4f} (BCEWithLogitsLoss)")

    # 5. Model Initialization
    logger.info(f"\nInitializing {arch.upper()} with ImageNet pretrained weights...")
    if arch == "efficientnet":
        model = EfficientNetB0TumorClassifier(dropout_rate=dropout_rate, pretrained=True)
    elif arch == "resnet":
        model = ResNet50TumorClassifier(dropout_rate=dropout_rate, pretrained=True)
    elif arch == "densenet":
        model = DenseNet121TumorClassifier(dropout_rate=dropout_rate, pretrained=True)
    elif arch == "vgg":
        model = VGG16TumorClassifier(dropout_rate=dropout_rate, pretrained=True)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
        
    model.to(device)

    # 6. Training Loop Setup
    history: List[Dict] = []
    best_val_pr_auc = -1.0
    best_val_f1 = -1.0
    best_epoch = -1
    best_val_metrics = {}
    best_logits = None
    best_metadata = None

    start_total_time = time.time()

    # -------------------------------------------------------------
    # STAGE 1: Frozen Backbone, Train Classification Head Only
    # -------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("STARTING STAGE 1: Feature Extractor Training (Classification Head Only)")
    logger.info(f"Epochs: 1 to {stage1_epochs}, Optimizer: AdamW, Learning Rate: {stage1_lr}")
    logger.info("-" * 70)

    model.freeze_backbone()
    stage1_params = model.get_parameter_summary()
    logger.info(f"Stage 1 Parameters -> Trainable: {stage1_params['trainable_parameters']:,}, Frozen: {stage1_params['frozen_parameters']:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage1_lr,
        weight_decay=weight_decay,
    )

    for epoch in range(1, stage1_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        val_loss, val_logits, val_labels, val_meta = evaluate(model, val_loader, criterion, device)
        val_probs = compute_anomaly_scores(val_logits)
        metrics = compute_metrics(val_labels, val_probs, threshold=0.5)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}/{total_epochs} [Stage 1] Completed in {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val PR-AUC: {metrics['pr_auc']:.4f} | Val F1: {metrics['f1_score']:.4f} | "
            f"Val Rec: {metrics['recall']:.4f} | Val Prec: {metrics['precision']:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f}"
        )

        cm = metrics["confusion_matrix"]
        logger.info(f"  Validation CM: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")

        record = {
            "epoch": epoch,
            "stage": 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_specificity": metrics["specificity"],
            "val_f1": metrics["f1_score"],
            "val_roc_auc": metrics["roc_auc"],
            "val_pr_auc": metrics["pr_auc"],
            "tn": cm["tn"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tp": cm["tp"],
            "epoch_time_sec": epoch_time,
        }
        history.append(record)

        # Model selection: Primary PR-AUC, Secondary F1
        is_best = False
        if metrics["pr_auc"] > best_val_pr_auc:
            is_best = True
        elif np.isclose(metrics["pr_auc"], best_val_pr_auc) and metrics["f1_score"] > best_val_f1:
            is_best = True

        if is_best:
            best_val_pr_auc = metrics["pr_auc"]
            best_val_f1 = metrics["f1_score"]
            best_epoch = epoch
            best_val_metrics = metrics
            best_logits = val_logits
            best_metadata = val_meta
            torch.save(
                {
                    "epoch": epoch,
                    "stage": 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "config": {
                        "pos_weight": pos_weight_val,
                        "batch_size": batch_size,
                        "lr": stage1_lr,
                        "seed": seed,
                    },
                },
                ckpt_dir / "best_model.pt",
            )
            logger.info(f"  >>> New Best Model Saved at Epoch {epoch} (PR-AUC = {best_val_pr_auc:.4f}, F1 = {best_val_f1:.4f})")

    # -------------------------------------------------------------
    # STAGE 2: Unfreeze Top Feature Blocks (features[6:]), Fine-tune
    # -------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("STARTING STAGE 2: Fine-Tuning Top MBConv Blocks (features[6:])")
    logger.info(f"Epochs: {stage1_epochs + 1} to {total_epochs}, Optimizer: AdamW, Learning Rate: {stage2_lr}")
    logger.info("-" * 70)

    model.unfreeze_top_blocks()
    stage2_params = model.get_parameter_summary()
    logger.info(f"Stage 2 Parameters -> Trainable: {stage2_params['trainable_parameters']:,}, Frozen: {stage2_params['frozen_parameters']:,}")

    optimizer2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage2_lr,
        weight_decay=weight_decay,
    )

    for epoch in range(stage1_epochs + 1, total_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer2, device, epoch, logger)
        val_loss, val_logits, val_labels, val_meta = evaluate(model, val_loader, criterion, device)
        val_probs = compute_anomaly_scores(val_logits)
        metrics = compute_metrics(val_labels, val_probs, threshold=0.5)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}/{total_epochs} [Stage 2] Completed in {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val PR-AUC: {metrics['pr_auc']:.4f} | Val F1: {metrics['f1_score']:.4f} | "
            f"Val Rec: {metrics['recall']:.4f} | Val Prec: {metrics['precision']:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f}"
        )

        cm = metrics["confusion_matrix"]
        logger.info(f"  Validation CM: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")

        record = {
            "epoch": epoch,
            "stage": 2,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_specificity": metrics["specificity"],
            "val_f1": metrics["f1_score"],
            "val_roc_auc": metrics["roc_auc"],
            "val_pr_auc": metrics["pr_auc"],
            "tn": cm["tn"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tp": cm["tp"],
            "epoch_time_sec": epoch_time,
        }
        history.append(record)

        # Model selection: Primary PR-AUC, Secondary F1
        is_best = False
        if metrics["pr_auc"] > best_val_pr_auc:
            is_best = True
        elif np.isclose(metrics["pr_auc"], best_val_pr_auc) and metrics["f1_score"] > best_val_f1:
            is_best = True

        if is_best:
            best_val_pr_auc = metrics["pr_auc"]
            best_val_f1 = metrics["f1_score"]
            best_epoch = epoch
            best_val_metrics = metrics
            best_logits = val_logits
            best_metadata = val_meta
            torch.save(
                {
                    "epoch": epoch,
                    "stage": 2,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer2.state_dict(),
                    "metrics": metrics,
                    "config": {
                        "pos_weight": pos_weight_val,
                        "batch_size": batch_size,
                        "lr": stage2_lr,
                        "seed": seed,
                    },
                },
                ckpt_dir / "best_model.pt",
            )
            logger.info(f"  >>> New Best Model Saved at Epoch {epoch} (PR-AUC = {best_val_pr_auc:.4f}, F1 = {best_val_f1:.4f})")

    # Save final model
    torch.save(
        {
            "epoch": total_epochs,
            "stage": 2,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        ckpt_dir / "final_model.pt",
    )

    total_training_time = time.time() - start_total_time
    logger.info("\n" + "=" * 70)
    logger.info(f"TRAINING COMPLETE in {total_training_time / 60:.2f} minutes.")
    logger.info(f"Best Validation Epoch: {best_epoch} with PR-AUC: {best_val_pr_auc:.4f}, F1: {best_val_f1:.4f}")
    logger.info("=" * 70)

    # -------------------------------------------------------------
    # 7. POST-TRAINING EVALUATION & ANOMALY / RISK GENERATION
    # -------------------------------------------------------------
    logger.info("\nGenerating post-training validation predictions, anomaly scores, and risk analysis...")

    # Load best checkpoint for final report and predictions
    best_checkpoint = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_loss, best_logits, val_labels, best_metadata = evaluate(model, val_loader, criterion, device)
    best_probs = compute_anomaly_scores(best_logits)

    # Assemble predictions dataframe
    df_preds = build_predictions_dataframe(best_metadata, best_logits, classification_threshold=0.5)
    df_preds.to_csv(preds_dir / "validation_predictions.csv", index=False)
    logger.info(f"Saved validation predictions to {preds_dir / 'validation_predictions.csv'} (Rows: {len(df_preds)})")

    # Risk analysis summary
    risk_summary = generate_risk_summary(df_preds)
    logger.info(f"Risk Categorization Summary:\n{json.dumps(risk_summary, indent=2)}")

    # Classification report
    y_pred_binary = (best_probs >= 0.5).astype(int)
    clf_report_str = classification_report(
        val_labels,
        y_pred_binary,
        target_names=["Normal (0)", "Tumor (1)"],
        digits=4,
        zero_division=0,
    )
    with open(metrics_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("OBJECTIVE 2: EFFICIENTNET-B0 VALIDATION CLASSIFICATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        f.write(f"PR-AUC (Primary Selection Metric): {best_val_pr_auc:.4f}\n")
        f.write(f"ROC-AUC: {best_val_metrics['roc_auc']:.4f}\n")
        f.write(f"F1-Score: {best_val_metrics['f1_score']:.4f}\n")
        f.write(f"Recall (Sensitivity): {best_val_metrics['recall']:.4f}\n")
        f.write(f"Precision: {best_val_metrics['precision']:.4f}\n")
        f.write(f"Specificity: {best_val_metrics['specificity']:.4f}\n")
        f.write(f"Accuracy: {best_val_metrics['accuracy']:.4f}\n\n")
        f.write(clf_report_str + "\n\n")
        f.write("CONFUSION MATRIX:\n")
        cm = best_val_metrics["confusion_matrix"]
        f.write(f"TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}, TP: {cm['tp']}\n\n")
        f.write(f"DISCLAIMER: {DISCLAIMER_TEXT}\n")

    # Save training history CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv(metrics_dir / "training_history.csv", index=False)

    # Save summary metrics JSON
    final_metrics_payload = {
        "best_epoch": best_epoch,
        "best_val_pr_auc": float(best_val_pr_auc),
        "best_val_f1": float(best_val_f1),
        "best_val_recall": float(best_val_metrics["recall"]),
        "best_val_precision": float(best_val_metrics["precision"]),
        "best_val_specificity": float(best_val_metrics["specificity"]),
        "best_val_accuracy": float(best_val_metrics["accuracy"]),
        "best_val_roc_auc": float(best_val_metrics["roc_auc"]),
        "confusion_matrix": best_val_metrics["confusion_matrix"],
        "support": best_val_metrics["support"],
        "risk_summary": risk_summary,
        "disclaimer": DISCLAIMER_TEXT,
    }
    with open(metrics_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2)

    # Save summary metrics CSV
    df_summary = pd.DataFrame([{
        "best_epoch": best_epoch,
        "val_pr_auc": best_val_pr_auc,
        "val_f1": best_val_f1,
        "val_recall": best_val_metrics["recall"],
        "val_precision": best_val_metrics["precision"],
        "val_specificity": best_val_metrics["specificity"],
        "val_accuracy": best_val_metrics["accuracy"],
        "val_roc_auc": best_val_metrics["roc_auc"],
        "val_loss": val_loss,
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
        "tp": cm["tp"],
        "total_val_samples": len(df_preds),
        "train_samples": len(train_ds),
        "pos_weight": float(pos_weight_val),
    }])
    df_summary.to_csv(metrics_dir / "validation_metrics_summary.csv", index=False)

    # Save run config
    run_config = {
        "model": f"{arch.upper()} (ImageNet Pretrained)",
        "framework": f"PyTorch {torch.__version__}",
        "device": "CPU",
        "random_seed": seed,
        "batch_size": batch_size,
        "total_epochs": total_epochs,
        "stage1_epochs": stage1_epochs,
        "stage1_lr": stage1_lr,
        "stage2_epochs": stage2_epochs,
        "stage2_lr": stage2_lr,
        "optimizer": "AdamW (weight_decay=1e-4)",
        "loss_function": f"BCEWithLogitsLoss(pos_weight={pos_weight_val:.4f})",
        "train_samples": train_counts,
        "val_samples": val_counts,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "train_augmentation": [
            "RandomHorizontalFlip(0.5)",
            "RandomVerticalFlip(0.5)",
            "RandomRotation(15)",
            "ColorJitter(brightness=0.1, contrast=0.1)",
        ],
        "best_epoch": best_epoch,
        "best_checkpoint": str(ckpt_dir / "best_model.pt"),
        "disclaimer": DISCLAIMER_TEXT,
    }
    with open(metrics_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # -------------------------------------------------------------
    # 8. GENERATE DIAGNOSTIC PLOTS
    # -------------------------------------------------------------
    logger.info("\nGenerating diagnostic plots in result/objective_2/plots/...")
    plot_training_curves(history, plots_dir)
    plot_roc_curve(val_labels, best_probs, best_val_metrics["roc_auc"], plots_dir / "roc_curve.png")
    plot_pr_curve(val_labels, best_probs, best_val_pr_auc, plots_dir / "pr_curve.png")
    plot_confusion_matrix(best_val_metrics["confusion_matrix"], plots_dir / "confusion_matrix.png")
    plot_anomaly_and_risk_distribution(val_labels, best_probs, plots_dir / "anomaly_risk_distribution.png")

    logger.info("All diagnostic plots generated successfully.")
    logger.info("=" * 70)
    logger.info("OBJECTIVE 2 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Tumor Classifier")
    parser.add_argument("--arch", type=str, default="efficientnet", choices=["efficientnet", "resnet", "densenet", "vgg"], help="Model architecture")
    args = parser.parse_args()
    
    run_training(arch=args.arch)
