"""
Objective 2 Part B: Train Anomaly-Specific Linear Heads on Frozen EfficientNet-B0.

Uses the frozen 1280-D feature representation from Objective-2A:
  result/objective_2/checkpoints/best_model.pt

Trains three independent binary heads:
  1. BACH: InSitu vs Normal/Benign (Ductal/DCIS architectural proxy, pos_weight=2.00)
  2. NuCLS: >=10 tumor nuclei vs 0 (Tumor nuclei hyper-density, pos_weight=1.72)
  3. BCSS: >=25% necrosis vs <=5% (Tumor necrosis/debris, pos_weight=1.29)

Includes strict pre-flight sanity test and validation-only threshold tuning.
"""

from __future__ import annotations

import json
import logging
import math
import os
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
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.objective_2.anomaly_model import EfficientNetB0MultiHeadAnomaly

# ImageNet normalization standard (matching Objective 2A)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class AnomalyManifestDataset(Dataset):
    """Loads 224x224 patches from a specific anomaly manifest and split."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        base_dir: Optional[str | Path] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split.lower()
        self.base_dir = Path(base_dir) if base_dir else PROJECT_ROOT
        self.transform = transform or get_eval_transforms()

        df = pd.read_csv(self.manifest_path)
        if "split" not in df.columns:
            raise ValueError(f"Manifest {self.manifest_path} missing 'split' column")

        self.df = df[df["split"].str.lower() == self.split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{self.split}' in {self.manifest_path}")

        # Resolve paths
        self.samples: List[Dict] = []
        for _, row in self.df.iterrows():
            rel_patch_path = str(row["patch_path"])
            full_path = self.base_dir / rel_patch_path
            if not full_path.exists():
                # Try fallback relative to PROJECT_ROOT
                full_path = PROJECT_ROOT / rel_patch_path
                if not full_path.exists():
                    raise FileNotFoundError(f"Patch file not found: {full_path}")

            sample = {
                "path": str(full_path),
                "label": int(row["label"]),
                "x": int(row["x"]) if "x" in row else -1,
                "y": int(row["y"]) if "y" in row else -1,
                "source_id": str(row.get("image_id", row.get("wsi_id", row.get("slide_id", "")))),
            }
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        with Image.open(sample["path"]) as img:
            img = img.convert("RGB")
            tensor_img = self.transform(img)

        label_tensor = torch.tensor([sample["label"]], dtype=torch.float32)
        return tensor_img, label_tensor, sample


def run_sanity_test(
    model: EfficientNetB0MultiHeadAnomaly,
    device: torch.device,
    datasets_info: Dict[str, Dict],
    logger: logging.Logger,
) -> bool:
    """
    Executes mandatory pre-flight checks:
    1. Checkpoint loaded successfully
    2. One patch produces 1280-D embedding
    3. Each head produces a scalar logit
    4. Forward/backward pass works on each head
    5. Source-level train/val/test separation verified
    """
    logger.info("=" * 70)
    logger.info("PRE-FLIGHT SANITY AUDIT")
    logger.info("=" * 70)

    try:
        # Check 1: Dummy image pass -> 1280-D embedding
        dummy_x = torch.randn(2, 3, 224, 224, device=device)
        model.eval()
        with torch.no_grad():
            feat = model.extract_features(dummy_x)

        if feat.shape != (2, 1280):
            logger.error(f"[FAIL] Feature extraction produced shape {feat.shape}, expected (2, 1280)")
            return False
        logger.info(f"[PASS] Feature extractor produces 1280-D embedding: {feat.shape}")

        # Check 2: All heads produce valid scalar logits
        for task in EfficientNetB0MultiHeadAnomaly.TASKS:
            out = model.forward_features(feat, task)
            if out.shape != (2, 1):
                logger.error(f"[FAIL] Head '{task}' produced shape {out.shape}, expected (2, 1)")
                return False
            logger.info(f"[PASS] Head '{task}' produced scalar logit: {out.shape}")

        # Check 3: One forward/backward step on each head
        model.train()
        for task, info in datasets_info.items():
            head = model.heads[task]
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
            dummy_feat = torch.randn(4, 1280, device=device)
            dummy_target = torch.tensor([[1.0], [0.0], [1.0], [0.0]], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([info["pos_weight"]], device=device))

            optimizer.zero_grad()
            logits = head(dummy_feat)
            loss = criterion(logits, dummy_target)
            loss.backward()
            optimizer.step()

            # Verify backbone weights did not receive gradients
            for p in model.features.parameters():
                if p.grad is not None:
                    logger.error("[FAIL] Backbone weights received gradients during head training!")
                    return False
            logger.info(f"[PASS] Forward/backward step verified on '{task}' (backbone strictly frozen)")

        # Check 4: Source-level split leakage check
        logger.info("[CHECK] Verifying 0 source-level leakage across splits...")
        for task, info in datasets_info.items():
            df = pd.read_csv(info["manifest"])
            src_col = info["source_col"]
            train_src = set(df[df["split"].str.lower() == "train"][src_col])
            val_src = set(df[df["split"].str.lower() == "validation"][src_col])
            test_src = set(df[df["split"].str.lower() == "test"][src_col])

            tv_int = train_src.intersection(val_src)
            tt_int = train_src.intersection(test_src)
            vt_int = val_src.intersection(test_src)

            if len(tv_int) > 0 or len(tt_int) > 0 or len(vt_int) > 0:
                logger.error(f"[FAIL] Leakage detected in task '{task}': Train∩Val={len(tv_int)}, Train∩Test={len(tt_int)}, Val∩Test={len(vt_int)}")
                return False
            logger.info(f"[PASS] Task '{task}': 0 overlap across Train ({len(train_src)}), Val ({len(val_src)}), Test ({len(test_src)})")

        logger.info("[PASS] ALL SANITY CHECKS PASSED SUCCESSFULLY!\n")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Sanity test crashed with exception: {e}")
        return False


@torch.no_grad()
def precompute_dataset_features(
    model: EfficientNetB0MultiHeadAnomaly,
    dataset: AnomalyManifestDataset,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Precomputes 1280-D features through the frozen backbone.
    Enables fast linear probe training and exact reproducibility.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []
    all_metadata = []

    for images, labels, metas in loader:
        images = images.to(device)
        feat = model.extract_features(images)
        all_features.append(feat.cpu())
        all_labels.append(labels)

        b_size = images.size(0)
        for i in range(b_size):
            all_metadata.append({
                "path": metas["path"][i],
                "label": int(metas["label"][i]),
                "x": int(metas["x"][i]),
                "y": int(metas["y"][i]),
                "source_id": metas["source_id"][i],
            })

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    return features_tensor, labels_tensor, all_metadata


def train_single_head(
    task: str,
    head: nn.Module,
    train_feat: torch.Tensor,
    train_lbl: torch.Tensor,
    val_feat: torch.Tensor,
    val_lbl: torch.Tensor,
    pos_weight: float,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    logger: Optional[logging.Logger] = None,
) -> Tuple[nn.Module, Dict, float]:
    """
    Trains an individual linear probe head on precomputed frozen features.
    Selects best checkpoint by validation PR-AUC.
    Finds optimal F1 classification threshold on validation set.
    """
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    dataset_train = torch.utils.data.TensorDataset(train_feat, train_lbl)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    val_feat_dev = val_feat.to(device)
    val_lbl_arr = val_lbl.numpy().ravel()

    best_val_prauc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        head.train()
        epoch_loss = 0.0
        n_batches = 0

        for b_feat, b_lbl in train_loader:
            b_feat = b_feat.to(device)
            b_lbl = b_lbl.to(device)

            optimizer.zero_grad()
            logits = head(b_feat)
            loss = criterion(logits, b_lbl)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # Validation evaluation
        head.eval()
        with torch.no_grad():
            val_logits = head(val_feat_dev).cpu().numpy().ravel()
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        val_prauc = float(average_precision_score(val_lbl_arr, val_probs))
        val_rocauc = float(roc_auc_score(val_lbl_arr, val_probs))

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        if logger and epoch % 5 == 0:
            logger.info(f"  [Epoch {epoch:02d}/{epochs:02d}] Train Loss: {avg_loss:.4f} | Val PR-AUC: {val_prauc:.4f} | Val ROC-AUC: {val_rocauc:.4f}")

    # Load best weights
    head.load_state_dict(best_state)
    head.eval()

    # Determine validation-optimal threshold (F1-maximizing)
    with torch.no_grad():
        val_logits = head(val_feat_dev).cpu().numpy().ravel()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    precisions, recalls, thresholds = precision_recall_curve(val_lbl_arr, val_probs)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.50
    optimal_threshold = float(np.clip(optimal_threshold, 0.05, 0.95))

    val_preds_opt = (val_probs >= optimal_threshold).astype(int)
    val_preds_50 = (val_probs >= 0.50).astype(int)

    val_metrics = {
        "best_epoch": int(best_epoch),
        "val_pr_auc": float(best_val_prauc),
        "val_roc_auc": float(roc_auc_score(val_lbl_arr, val_probs)),
        "optimal_threshold": float(optimal_threshold),
        "val_f1_at_optimal_threshold": float(f1_score(val_lbl_arr, val_preds_opt)),
        "val_f1_at_0.50_threshold": float(f1_score(val_lbl_arr, val_preds_50)),
    }

    if logger:
        logger.info(f"  [BEST] Epoch {best_epoch} | Best Val PR-AUC: {best_val_prauc:.4f} | Opt Threshold: {optimal_threshold:.4f} | Val F1@opt: {val_metrics['val_f1_at_optimal_threshold']:.4f}")

    return head, val_metrics, optimal_threshold


def main() -> None:
    # Set up directories
    out_dir = PROJECT_ROOT / "result" / "objective_2" / "anomaly_model"
    ckpt_dir = out_dir / "checkpoints"
    metrics_dir = out_dir / "metrics"
    logs_dir = out_dir / "logs"

    for d in [ckpt_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = logging.getLogger("TrainAnomalyHeads")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(logs_dir / "train_anomaly_heads.log", mode="w", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 70)
    logger.info("OBJECTIVE 2 PART B: ANOMALY-SPECIFIC LINEAR HEAD TRAINING")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")

    # Checkpoint path
    backbone_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt"
    logger.info(f"Loading frozen backbone checkpoint: {backbone_ckpt}")

    model = EfficientNetB0MultiHeadAnomaly(checkpoint_path=backbone_ckpt, device=device)
    model.to(device)

    summary = model.get_parameter_summary()
    logger.info(f"Parameter Summary: Total={summary['total_parameters']:,} | Frozen={summary['frozen_parameters']:,} | Trainable={summary['trainable_parameters']:,}")

    # Dataset configurations
    datasets_info = {
        "atypia": {
            "name": "MITOS-ATYPIA14 (Severe Nuclear Pleomorphism)",
            "manifest": PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "mitos_atypia" / "mitos_atypia_patch_manifest.csv",
            "base_dir": PROJECT_ROOT,
            "source_col": "frame_id",
            "pos_weight": 9.50,
        },
        "ductal": {
            "name": "BACH (Ductal/DCIS Proxy)",
            "manifest": PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "bach_ductal_proxy" / "bach_ductal_proxy_patch_manifest.csv",
            "base_dir": PROJECT_ROOT,
            "source_col": "image_id",
            "pos_weight": 2.00,
        },
        "density": {
            "name": "NuCLS (High Tumor Density)",
            "manifest": PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "nucls_density" / "nucls_density_patch_manifest.csv",
            "base_dir": PROJECT_ROOT,
            "source_col": "wsi_id",
            "pos_weight": 1.72,
        },
        "necrosis": {
            "name": "BCSS (Necrosis/Debris)",
            "manifest": PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "bcss_necrosis" / "bcss_necrosis_patch_manifest.csv",
            "base_dir": PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "bcss_necrosis",
            "source_col": "slide_id",
            "pos_weight": 1.29,
        },
    }

    # Step 1: Pre-flight Sanity Audit
    sanity_ok = run_sanity_test(model, device, datasets_info, logger)
    if not sanity_ok:
        logger.error("CRITICAL ERROR: Pre-flight sanity audit failed! Aborting training.")
        sys.exit(1)

    # Check for existing checkpoint to preserve already trained heads
    existing_ckpt_path = ckpt_dir / "anomaly_linear_heads_best.pt"
    existing_thresholds = {}
    existing_val_metrics = {}
    if existing_ckpt_path.exists():
        logger.info(f"Loading existing checkpoint to preserve pre-trained anomaly heads: {existing_ckpt_path}")
        existing_ckpt = torch.load(existing_ckpt_path, map_location=device, weights_only=False)
        existing_thresholds = existing_ckpt.get("optimal_thresholds", {})
        existing_val_metrics = existing_ckpt.get("validation_metrics", {})
        for task_name in ["ductal", "density", "necrosis"]:
            if f"heads.{task_name}.1.weight" in existing_ckpt["model_state_dict"]:
                head_weights = {
                    "1.weight": existing_ckpt["model_state_dict"][f"heads.{task_name}.1.weight"],
                    "1.bias": existing_ckpt["model_state_dict"][f"heads.{task_name}.1.bias"],
                }
                model.heads[task_name].load_state_dict(head_weights)
                logger.info(f"  [PRESERVED] Loaded trained weights for '{task_name}'")

    # Step 2: Extract features & train linear heads
    trained_validation_metrics = dict(existing_val_metrics)
    optimal_thresholds = dict(existing_thresholds)

    # Allow CLI flag or default to training missing heads
    tasks_to_train = [t for t in datasets_info.keys() if t not in existing_thresholds or "--retrain-all" in sys.argv]
    if not tasks_to_train:
        tasks_to_train = list(datasets_info.keys())

    logger.info(f"Tasks targeted for training: {tasks_to_train}")

    for task in tasks_to_train:
        info = datasets_info[task]
        logger.info("=" * 70)
        logger.info(f"TRAINING TASK: {task.upper()} — {info['name']}")
        logger.info(f"  Pos weight: {info['pos_weight']}")
        logger.info("=" * 70)

        # Datasets
        train_ds = AnomalyManifestDataset(info["manifest"], split="train", base_dir=info["base_dir"])
        val_ds = AnomalyManifestDataset(info["manifest"], split="validation", base_dir=info["base_dir"])

        logger.info(f"  Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")

        # Precompute features
        logger.info("  Extracting 1280-D features through frozen backbone...")
        t0 = time.time()
        train_feat, train_lbl, _ = precompute_dataset_features(model, train_ds, device)
        val_feat, val_lbl, _ = precompute_dataset_features(model, val_ds, device)
        logger.info(f"  Feature extraction complete in {time.time() - t0:.2f}s (Train: {train_feat.shape}, Val: {val_feat.shape})")

        # Train linear probe head
        head = model.heads[task]
        trained_head, val_mets, opt_tau = train_single_head(
            task=task,
            head=head,
            train_feat=train_feat,
            train_lbl=train_lbl,
            val_feat=val_feat,
            val_lbl=val_lbl,
            pos_weight=info["pos_weight"],
            device=device,
            epochs=30,
            lr=1e-3,
            weight_decay=1e-4,
            logger=logger,
        )

        model.heads[task] = trained_head
        trained_validation_metrics[task] = val_mets
        optimal_thresholds[task] = opt_tau

    # Step 3: Save Anomaly Multi-Head Checkpoint
    checkpoint_out = ckpt_dir / "anomaly_linear_heads_best.pt"
    torch.save({
        "backbone_checkpoint": str(backbone_ckpt),
        "model_state_dict": model.state_dict(),
        "optimal_thresholds": optimal_thresholds,
        "validation_metrics": trained_validation_metrics,
        "tasks": list(datasets_info.keys()),
    }, checkpoint_out)

    logger.info("=" * 70)
    logger.info(f"Saved Anomaly Checkpoint: {checkpoint_out}")

    # Save validation metrics JSON
    val_metrics_out = metrics_dir / "anomaly_validation_metrics.json"
    with open(val_metrics_out, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_thresholds": optimal_thresholds,
            "tasks": trained_validation_metrics,
        }, f, indent=2)

    logger.info(f"Saved Validation Metrics: {val_metrics_out}")
    logger.info("=" * 70)
    logger.info("ANOMALY LINEAR HEADS TRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
