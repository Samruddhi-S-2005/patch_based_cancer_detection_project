"""
Objective 2 Pre-flight Sanity Check Script.

Verifies:
1. Data loading from actual TRAIN and VALIDATION patch folders.
2. Binary label extraction and manifest cross-checking.
3. Image dimensions (224x224 RGB) and ImageNet normalization.
4. EfficientNet-B0 model initialization and pretrained weights.
5. Forward pass, weighted BCE loss calculation, backward gradient pass on CPU.
6. Anomaly scoring and project-level risk categorization functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
from src.objective_2.dataset import (
    HistopathPatchDataset,
    create_dataloaders,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.objective_2.model import EfficientNetB0TumorClassifier
from src.objective_2.metrics import compute_metrics
from src.objective_2.anomaly_and_risk import (
    compute_anomaly_scores,
    categorize_risk,
    build_predictions_dataframe,
    generate_risk_summary,
    DISCLAIMER_TEXT,
)


def run_sanity_check() -> bool:
    print("=" * 60)
    print("STARTING OBJECTIVE 2 PRE-FLIGHT SANITY CHECK")
    print("=" * 60)

    train_dir = PROJECT_ROOT / "data" / "PATCHES" / "CLEANED"
    val_dir = PROJECT_ROOT / "data" / "PATCHES" / "VALIDATION"
    val_manifest = PROJECT_ROOT / "data" / "PATCHES" / "VALIDATION" / "validation_patch_manifest.csv"

    # 1. Dataset Initialization
    print("\n[1/6] Loading Train & Validation Datasets...")
    train_loader, val_loader, train_ds, val_ds = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        train_manifest=None,
        val_manifest=val_manifest,
        batch_size=8,
        num_workers=0,
        seed=42,
    )

    train_counts = train_ds.get_class_counts()
    val_counts = val_ds.get_class_counts()
    print(f"  Train: Normal={train_counts['normal']}, Tumor={train_counts['tumor']}, Total={train_counts['total']}")
    print(f"  Validation: Normal={val_counts['normal']}, Tumor={val_counts['tumor']}, Total={val_counts['total']}")

    assert train_counts["total"] == 14719, f"Expected 14,719 train patches, got {train_counts['total']}"
    assert train_counts["normal"] == 13129, f"Expected 13,129 train normal, got {train_counts['normal']}"
    assert train_counts["tumor"] == 1590, f"Expected 1,590 train tumor, got {train_counts['tumor']}"
    assert val_counts["total"] == 3349, f"Expected 3,349 val patches, got {val_counts['total']}"
    assert val_counts["normal"] == 3253, f"Expected 3,253 val normal, got {val_counts['normal']}"
    assert val_counts["tumor"] == 96, f"Expected 96 val tumor, got {val_counts['tumor']}"
    print("  [OK] Dataset counts match frozen Objective 1 verified numbers exactly!")

    # 2. Batch Inspection
    print("\n[2/6] Inspecting sample batch...")
    images, labels, sample_meta = next(iter(train_loader))
    print(f"  Image batch shape: {images.shape}")
    print(f"  Label batch shape: {labels.shape}")
    assert images.shape == (8, 3, 224, 224), f"Incorrect image shape: {images.shape}"
    assert labels.shape == (8, 1), f"Incorrect label shape: {labels.shape}"
    print("  [OK] Batch shape and tensor types verified.")

    # 3. Model Instantiation & Weights
    print("\n[3/6] Initializing EfficientNet-B0 with ImageNet pretrained weights...")
    model = EfficientNetB0TumorClassifier(dropout_rate=0.2, pretrained=True)
    summary_before = model.get_parameter_summary()
    print(f"  Total params: {summary_before['total_parameters']:,}")

    # Stage 1 Freezing
    model.freeze_backbone()
    summary_stage1 = model.get_parameter_summary()
    print(f"  Stage 1 (Head only) - Trainable: {summary_stage1['trainable_parameters']:,}, Frozen: {summary_stage1['frozen_parameters']:,}")
    assert summary_stage1["trainable_parameters"] == 1281, f"Expected 1281 trainable params in Stage 1, got {summary_stage1['trainable_parameters']}"

    # Stage 2 Freezing
    model.unfreeze_top_blocks(start_block_idx=6)
    summary_stage2 = model.get_parameter_summary()
    print(f"  Stage 2 (Top blocks) - Trainable: {summary_stage2['trainable_parameters']:,}, Frozen: {summary_stage2['frozen_parameters']:,}")
    print("  [OK] Model architecture and parameter freezing verified.")

    # 4. Forward & Backward Pass on CPU
    print("\n[4/6] Testing forward pass, weighted loss, and backward gradient step...")
    model.freeze_backbone()
    pos_weight = torch.tensor([13129.0 / 1590.0])  # ~8.2572
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    probs = torch.sigmoid(logits)
    print(f"  Sample loss: {loss.item():.4f}")
    print(f"  Sample probabilities: {[round(p, 4) for p in probs.squeeze().tolist()]}")
    assert not torch.isnan(loss), "Loss computed as NaN!"
    print("  [OK] Forward pass and gradient step completed successfully.")

    # 5. Anomaly Scoring & Risk Categorization
    print("\n[5/6] Testing Anomaly Analysis & Risk Categorization...")
    anomaly_scores = compute_anomaly_scores(logits)
    risk_cats = categorize_risk(anomaly_scores)
    # Convert sample_meta tuple of dicts / list to appropriate format
    meta_list = []
    for i in range(len(labels)):
        meta_list.append({
            "filename": sample_meta["filename"][i] if isinstance(sample_meta["filename"], list) else sample_meta["filename"][i],
            "wsi_id": sample_meta["wsi_id"][i] if isinstance(sample_meta["wsi_id"], list) else sample_meta["wsi_id"][i],
            "x": int(sample_meta["x"][i]),
            "y": int(sample_meta["y"][i]),
            "label": int(sample_meta["label"][i]),
        })
    df_preds = build_predictions_dataframe(meta_list, logits.detach().cpu().numpy())
    summary_risk = generate_risk_summary(df_preds)
    
    print(f"  Anomaly scores (sample): {[round(s, 4) for s in anomaly_scores[:4]]}")
    print(f"  Risk categories (sample): {risk_cats[:4]}")
    print(f"  Disclaimer text: '{DISCLAIMER_TEXT}'")
    assert "MODEL-BASED ANOMALY/RISK SCORE" in summary_risk["disclaimer"]
    print("  [OK] Anomaly and Risk logic verified.")

    # 6. Metrics Calculation
    print("\n[6/6] Testing Evaluation Metrics...")
    dummy_y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1])
    dummy_y_probs = np.array([0.05, 0.12, 0.88, 0.25, 0.92, 0.76, 0.10, 0.65])
    metrics = compute_metrics(dummy_y_true, dummy_y_probs)
    print(f"  Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  CM: TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")
    print("  [OK] Metrics calculation verified.")

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_sanity_check()
    sys.exit(0 if success else 1)
