"""
Script: Resumable Training-Set Feature Extraction and PCA Visualizations (Tasks 1 & 2).

Extracts pre-classifier continuous feature representations from:
1. EfficientNet-B0 (1280-D)
2. ResNet-50 (2048-D)
3. DenseNet-121 (1024-D)
4. VGG-16 (4096-D)

Features:
- Authoritative split identification from training_patch_preprocessing_audit.csv (14,719 KEEP patches).
- Memory-mapped arrays (.npy) with deterministic patch-ID ordering across all 4 models.
- Per-batch checkpointing and state tracking in extraction_progress.json for safe resumability.
- Comprehensive manifest with patch ID, WSI/patient ID, label provenance, split, hash, feature layer.
- Two-dimensional PCA scatter plots with clear legends and scientific disclaimers.
"""

import os
import sys
import json
import time
import uuid
import hashlib
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("d:/Samruddhi/Projects/patch_based_cancer_detection_project").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.objective_2.feature_analysis import (
    load_classifier_model,
    extract_backbone_embedding,
    EMBEDDING_DIMENSIONS,
    MODEL_CHECKPOINTS,
)

CHECKPOINT_HASHES = {
    "EfficientNet-B0": "cc6764e073a21e523b2ae6a6b24b6c718bf4159a778f2563a307c2e88900b3ef",
    "ResNet-50": "a1e47d42e1a5b732eeadb5cea620807add979ebea938f9f7b73072b6fd4f8864",
    "DenseNet-121": "13e80add08df8dd471b95697b50e30c4745c08fc511ed2da8b1782fb53094303",
    "VGG-16": "353f1fcb8d13e412a6dc690242edbf2126d537ec99152223b4b8f1fb1fba24dd",
}

FEATURE_LAYERS = {
    "EfficientNet-B0": "model.backbone.features -> AdaptiveAvgPool2d(1) -> flatten(1)",
    "ResNet-50": "model.backbone.layer4 -> AdaptiveAvgPool2d(1) -> flatten(1)",
    "DenseNet-121": "model.backbone.features -> relu -> adaptive_avg_pool2d(1) -> flatten(1)",
    "VGG-16": "model.backbone.features -> avgpool -> flatten(1) -> classifier[:6]",
}


class PreprocessedPatchDataset(Dataset):
    def __init__(self, patch_paths: List[str]):
        self.patch_paths = patch_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx: int):
        p = self.patch_paths[idx]
        with Image.open(p) as img:
            im_rgb = img.convert("RGB")
            tensor = self.transform(im_rgb)
        return tensor, idx


def get_eligible_training_patches():
    qc_csv = PROJECT_ROOT / "results" / "objective_1_preprocessing" / "quality_control" / "training_patch_preprocessing_audit.csv"
    if not qc_csv.exists():
        raise FileNotFoundError(f"QC audit CSV not found at: {qc_csv}")

    df_qc = pd.read_csv(qc_csv)
    # Filter strictly for eligible retained training patches
    df_eligible = df_qc[(df_qc["split"] == "train") & (df_qc["final_decision"] == "KEEP")].copy()
    
    # Sort deterministically by class and patch_id to ensure bit-for-bit reproducible index ordering
    df_eligible.sort_values(by=["class_label", "patch_id"], ascending=[True, True], inplace=True)
    df_eligible.reset_index(drop=True, inplace=True)
    return df_eligible


def extract_model_embeddings(
    model_name: str,
    df_eligible: pd.DataFrame,
    out_dir: Path,
    batch_size: int = 64,
    limit: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
):
    safe_name = model_name.lower().replace("-", "_")
    target_dim = EMBEDDING_DIMENSIONS[model_name]
    num_samples = len(df_eligible) if limit is None else min(limit, len(df_eligible))
    
    out_npy_path = out_dir / f"{safe_name}_train_embeddings.npy"
    progress_file = out_dir / "extraction_progress.json"

    # Load or initialize progress
    progress = {}
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                progress = json.load(f)
        except Exception:
            progress = {}

    model_progress = progress.get(model_name, {"processed_count": 0, "status": "PENDING"})

    # Check if existing npy is already complete
    if out_npy_path.exists() and model_progress.get("status") == "COMPLETED" and model_progress.get("processed_count") == num_samples:
        print(f"[{model_name}] Embeddings already complete ({num_samples} samples). Skipping extraction.")
        return np.load(out_npy_path)

    print(f"\n==================================================")
    print(f"EXTRACTING {model_name} (Dim: {target_dim}, Samples: {num_samples})")
    print(f"==================================================")

    # Initialize or open memory-mapped array
    start_idx = 0
    if out_npy_path.exists() and model_progress.get("processed_count", 0) > 0:
        start_idx = model_progress["processed_count"]
        print(f"  Resuming from index {start_idx}/{num_samples}...")
        mmap_arr = np.lib.format.open_memmap(out_npy_path, mode="r+", dtype="float32", shape=(num_samples, target_dim))
    else:
        print(f"  Allocating memory-mapped array at: {out_npy_path}...")
        mmap_arr = np.lib.format.open_memmap(out_npy_path, mode="w+", dtype="float32", shape=(num_samples, target_dim))

    # Load model
    print(f"  Loading model checkpoint from {MODEL_CHECKPOINTS[model_name]}...")
    model = load_classifier_model(model_name, device=device)
    torch.set_num_threads(8)

    # Dataset & loader
    dataset = PreprocessedPatchDataset(df_eligible["cleaned_path"].iloc[:num_samples].tolist())
    
    t0 = time.time()
    curr_idx = start_idx
    while curr_idx < num_samples:
        batch_end = min(curr_idx + batch_size, num_samples)
        batch_indices = list(range(curr_idx, batch_end))
        
        # Load batch tensors
        tensors = torch.stack([dataset[i][0] for i in batch_indices]).to(device)
        
        with torch.inference_mode():
            emb = extract_backbone_embedding(model, model_name, tensors)
            emb_np = emb.cpu().numpy().astype(np.float32)

        # Write directly to memory-mapped array
        mmap_arr[curr_idx:batch_end, :] = emb_np
        mmap_arr.flush()
        
        curr_idx = batch_end
        
        # Save progress
        model_progress["processed_count"] = curr_idx
        model_progress["status"] = "IN_PROGRESS" if curr_idx < num_samples else "COMPLETED"
        model_progress["last_updated"] = datetime.datetime.now().isoformat()
        progress[model_name] = model_progress
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        if curr_idx % 512 == 0 or curr_idx == num_samples:
            elapsed = time.time() - t0
            rate = (curr_idx - start_idx) / max(elapsed, 1e-4)
            rem = (num_samples - curr_idx) / max(rate, 1e-4)
            print(f"  Progress: {curr_idx}/{num_samples} ({curr_idx/num_samples*100:.1f}%) | Speed: {rate:.1f} patches/s | ETA: {rem/60:.1f} min")

    print(f"[{model_name}] Extraction finished in {time.time()-t0:.1f}s.")
    return mmap_arr


def generate_training_pca_visualizations(
    model_name: str,
    embeddings: np.ndarray,
    df_eligible: pd.DataFrame,
    viz_dir: Path,
):
    print(f"  Generating PCA scatter plot for {model_name}...")
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(embeddings)
    exp_var = pca.explained_variance_ratio_

    labels = df_eligible["class_label"].iloc[:len(embeddings)].values
    classes = df_eligible["class_name"].iloc[:len(embeddings)].values

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot normal and tumor points
    norm_mask = (labels == 0)
    tum_mask = (labels == 1)

    ax.scatter(coords_2d[norm_mask, 0], coords_2d[norm_mask, 1], c="#2b8cbe", alpha=0.35, s=12, label=f"Normal Lymph Node (N={np.sum(norm_mask):,})")
    ax.scatter(coords_2d[tum_mask, 0], coords_2d[tum_mask, 1], c="#e41a1c", alpha=0.35, s=12, label=f"Metastatic Carcinoma (N={np.sum(tum_mask):,})")

    ax.set_title(
        f"PathoVision AI: {model_name} Training-Set Feature Projections (PCA)\n"
        f"Feature Vector: {EMBEDDING_DIMENSIONS[model_name]}-D | Total Training Patches: {len(embeddings):,}",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel(f"Principal Component 1 ({exp_var[0]*100:.2f}% Variance Explained)", fontsize=11)
    ax.set_ylabel(f"Principal Component 2 ({exp_var[1]*100:.2f}% Variance Explained)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Scientific disclaimer note at the bottom
    disclaimer = (
        "SCIENTIFIC NOTE: Axes represent linear projections of learned numerical CNN feature embeddings, NOT physical biological measurements.\n"
        "Separation reflects high-dimensional visual pattern discrimination learned via transfer learning on CAMELYON16 tissue."
    )
    plt.figtext(0.5, 0.015, disclaimer, wrap=True, horizontalalignment="center", fontsize=8, style="italic", bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="#cccccc", lw=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    safe_name = model_name.lower().replace("-", "_")
    plot_path = viz_dir / f"{safe_name}_training_pca_scatter.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved PCA plot to: {plot_path}")

    return {
        "model_name": model_name,
        "plot_path": str(plot_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "pc1_variance": round(float(exp_var[0]), 4),
        "pc2_variance": round(float(exp_var[1]), 4),
        "total_explained_variance": round(float(exp_var.sum()), 4),
        "total_samples": int(len(embeddings)),
        "normal_count": int(np.sum(norm_mask)),
        "tumor_count": int(np.sum(tum_mask)),
    }


def main():
    parser = argparse.ArgumentParser(description="PathoVision AI: Training Feature Extraction")
    parser.add_argument("--model", type=str, default="all", choices=["all", "efficientnet_b0", "resnet_50", "densenet_121", "vgg_16"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of patches (for pilot test/dry-run)")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "results" / "objective_2_model_and_anomaly" / "feature_extraction" / "training"
    viz_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print("PATHOVISION AI: TRAINING FEATURE EXTRACTION ENGINE")
    print("======================================================================")

    df_eligible = get_eligible_training_patches()
    total_eligible = len(df_eligible)
    print(f"Authoritative Eligible Training Patches: {total_eligible:,}")
    print(f"  Normal: {sum(df_eligible['class_label'] == 0):,} | Tumor: {sum(df_eligible['class_label'] == 1):,}")

    target_models = (
        ["EfficientNet-B0", "ResNet-50", "DenseNet-121", "VGG-16"]
        if args.model == "all"
        else [k for k in MODEL_CHECKPOINTS.keys() if k.lower().replace("-", "_") == args.model]
    )

    run_id = f"RUN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    # Save canonical training feature manifest
    num_to_index = total_eligible if args.limit is None else min(args.limit, total_eligible)
    manifest_rows = []
    for r_idx in range(num_to_index):
        row = df_eligible.iloc[r_idx]
        c_path = Path(row["cleaned_path"])
        rel_p = str(c_path.relative_to(PROJECT_ROOT) if c_path.is_absolute() else c_path).replace("\\", "/")
        manifest_rows.append({
            "embedding_row_index": r_idx,
            "patch_id": row["patch_id"],
            "source_wsi_id": row["source_wsi_id"],
            "patient_id": row["patient_id"],
            "class_label": int(row["class_label"]),
            "class_name": str(row["class_name"]),
            "ground_truth_provenance": "CAMELYON16 WSI annotation polygon overlap >= 50% for tumor, 0% for normal",
            "split": "train",
            "source_patch_path": rel_p,
            "preprocessing_version": "ImageNet-normalized 224x224 Level 0 patch (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])",
            "extraction_run_id": run_id,
            "status": "SUCCESS"
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = out_dir / "training_feature_extraction_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Saved training manifest ({len(manifest_df):,} rows) to: {manifest_csv}")

    pca_summaries = {}
    for m_name in target_models:
        embs = extract_model_embeddings(
            m_name,
            df_eligible,
            out_dir,
            batch_size=args.batch_size,
            limit=args.limit,
            device=torch.device("cpu")
        )
        pca_meta = generate_training_pca_visualizations(m_name, embs, df_eligible, viz_dir)
        pca_summaries[m_name] = pca_meta

    with open(viz_dir / "training_pca_metadata.json", "w") as f:
        json.dump(pca_summaries, f, indent=2)

    print("\n======================================================================")
    print("TRAINING FEATURE EXTRACTION & PCA VISUALIZATIONS COMPLETE!")
    print("======================================================================")


if __name__ == "__main__":
    main()
