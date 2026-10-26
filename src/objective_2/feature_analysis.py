"""
Feature Extraction, Latent Space Analysis, and Tissue Morphology Audit.

Extracts pre-classifier neural embeddings across all 4 convolutional architectures:
- EfficientNet-B0: 1280-D
- ResNet-50: 2048-D
- DenseNet-121: 1024-D
- VGG-16: 4096-D

Performs 2D dimensionality reduction (PCA) on held-out CAMELYON16 test cohorts.
Provides a clear, transparent scientific morphology audit distinguishing learned
neural embeddings from physical continuous tissue measurements.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.objective_2.model import (
    EfficientNetB0TumorClassifier,
    ResNet50TumorClassifier,
    DenseNet121TumorClassifier,
    VGG16TumorClassifier,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Verified Checkpoint Paths
MODEL_CHECKPOINTS = {
    "EfficientNet-B0": PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt",
    "ResNet-50": PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "resnet" / "best_model.pt",
    "DenseNet-121": PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "densenet" / "best_model.pt",
    "VGG-16": PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "vgg" / "best_model.pt",
}

EMBEDDING_DIMENSIONS = {
    "EfficientNet-B0": 1280,
    "ResNet-50": 2048,
    "DenseNet-121": 1024,
    "VGG-16": 4096,
}

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_classifier_model(model_name: str, device: torch.device = torch.device("cpu")) -> nn.Module:
    """Instantiates and loads weights for any of the 4 classifier architectures."""
    canonical_name = model_name
    for k in MODEL_CHECKPOINTS:
        if k.lower() in model_name.lower():
            canonical_name = k
            break

    ckpt_path = MODEL_CHECKPOINTS[canonical_name]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {canonical_name}: {ckpt_path}")

    if canonical_name == "EfficientNet-B0":
        model = EfficientNetB0TumorClassifier(pretrained=False)
    elif canonical_name == "ResNet-50":
        model = ResNet50TumorClassifier(pretrained=False)
    elif canonical_name == "DenseNet-121":
        model = DenseNet121TumorClassifier(pretrained=False)
    elif canonical_name == "VGG-16":
        model = VGG16TumorClassifier(pretrained=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    ckpt_data = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def extract_backbone_embedding(
    model: nn.Module,
    model_name: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Extracts the continuous pre-classifier feature embedding vector:
    - EfficientNet-B0: 1280-D (features -> avgpool -> flatten)
    - ResNet-50: 2048-D (conv1..layer4 -> avgpool -> flatten)
    - DenseNet-121: 1024-D (features -> relu -> adaptive_avg_pool2d -> flatten)
    - VGG-16: 4096-D (features -> avgpool -> flatten -> classifier[:6])
    """
    canonical_name = model_name
    for k in MODEL_CHECKPOINTS:
        if k.lower() in model_name.lower():
            canonical_name = k
            break

    with torch.no_grad():
        if canonical_name == "EfficientNet-B0":
            feat = model.backbone.features(tensor)
            pool = model.backbone.avgpool(feat)
            emb = torch.flatten(pool, 1)
        elif canonical_name == "ResNet-50":
            x = model.backbone.conv1(tensor)
            x = model.backbone.bn1(x)
            x = model.backbone.relu(x)
            x = model.backbone.maxpool(x)
            x = model.backbone.layer1(x)
            x = model.backbone.layer2(x)
            x = model.backbone.layer3(x)
            x = model.backbone.layer4(x)
            pool = model.backbone.avgpool(x)
            emb = torch.flatten(pool, 1)
        elif canonical_name == "DenseNet-121":
            feat = model.backbone.features(tensor)
            out = F.relu(feat, inplace=True)
            pool = F.adaptive_avg_pool2d(out, (1, 1))
            emb = torch.flatten(pool, 1)
        elif canonical_name == "VGG-16":
            feat = model.backbone.features(tensor)
            pool = model.backbone.avgpool(feat)
            flat = torch.flatten(pool, 1)
            emb = model.backbone.classifier[:6](flat)
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")

    return emb


def run_pca_reduction(embeddings: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs standard Principal Component Analysis (PCA) on embedding matrices.
    Returns (coordinates_2d, explained_variance_ratio).
    """
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embeddings)
    return coords, pca.explained_variance_ratio_


def get_feature_cache_path() -> Path:
    """Returns standard path to the feature projections cache JSON."""
    cache_dir = PROJECT_ROOT / "result" / "objective_2" / "feature_analysis"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "feature_projections.json"


def generate_feature_projections_cache(
    sample_size_per_class: int = 75,
    device: Optional[torch.device] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Extracts embeddings across all 4 architectures on a balanced subset of held-out CAMELYON16 test patches,
    computes 2D PCA projections, and persists the results.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_normal_dir = PROJECT_ROOT / "data" / "PATCHES" / "TEST" / "normal"
    test_tumor_dir = PROJECT_ROOT / "data" / "PATCHES" / "TEST" / "tumor"

    normal_files = sorted(list(test_normal_dir.glob("*.png")))
    tumor_files = sorted(list(test_tumor_dir.glob("*.png")))

    if not normal_files or not tumor_files:
        raise FileNotFoundError("Held-out test patch directories empty or missing.")

    # Stratified deterministic selection with seed 42
    rng = np.random.RandomState(42)
    n_sample = min(sample_size_per_class, len(normal_files), len(tumor_files))

    sel_normal_indices = rng.choice(len(normal_files), size=n_sample, replace=False)
    sel_tumor_indices = rng.choice(len(tumor_files), size=n_sample, replace=False)

    sampled_patches: List[Tuple[Path, int, str]] = []
    for idx in sorted(sel_normal_indices):
        sampled_patches.append((normal_files[idx], 0, "Normal Lymph Node"))
    for idx in sorted(sel_tumor_indices):
        sampled_patches.append((tumor_files[idx], 1, "Metastatic Carcinoma"))

    patch_paths = [p[0] for p in sampled_patches]
    patch_labels = [p[1] for p in sampled_patches]
    patch_class_names = [p[2] for p in sampled_patches]
    patch_filenames = [p[0].name for p in sampled_patches]

    # Pre-load all preprocessed images into a tensor batch to minimize I/O overhead
    tensors_list = []
    for p in patch_paths:
        img = Image.open(p).convert("RGB")
        t = EVAL_TRANSFORMS(img)
        tensors_list.append(t)
    batch_tensors = torch.stack(tensors_list).to(device)

    arch_order = ["EfficientNet-B0", "ResNet-50", "DenseNet-121", "VGG-16"]
    projections_data: Dict[str, Any] = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "CAMELYON16 Held-out Test Split",
            "total_samples": len(patch_paths),
            "normal_samples": n_sample,
            "tumor_samples": n_sample,
            "methodology": "Pre-classifier continuous feature embedding extraction followed by PCA 2D projection",
            "provenance_note": "Evaluated freshly on frozen model checkpoints without synthetic simulation.",
        },
        "samples": [
            {
                "index": i,
                "filename": patch_filenames[i],
                "true_label": patch_labels[i],
                "true_class": patch_class_names[i],
            }
            for i in range(len(patch_paths))
        ],
        "models": {},
    }

    for arch in arch_order:
        print(f"[FEATURE ANALYSIS] Extracting {arch} embeddings on {len(patch_paths)} test patches...")
        t0 = time.time()
        model = load_classifier_model(arch, device=device)

        # Batch inference in chunks of 32
        embeddings_chunks = []
        scores_chunks = []
        chunk_size = 32
        for start_idx in range(0, len(patch_paths), chunk_size):
            end_idx = min(start_idx + chunk_size, len(patch_paths))
            sub_batch = batch_tensors[start_idx:end_idx]

            emb = extract_backbone_embedding(model, arch, sub_batch)
            embeddings_chunks.append(emb.cpu().numpy())

            with torch.no_grad():
                logits = model(sub_batch).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                scores_chunks.append(probs)

        all_embs = np.concatenate(embeddings_chunks, axis=0)
        all_probs = np.concatenate(scores_chunks, axis=0)

        coords_2d, exp_var = run_pca_reduction(all_embs, n_components=2)
        elapsed = time.time() - t0

        projections_data["models"][arch] = {
            "embedding_dimension": int(all_embs.shape[1]),
            "pca_explained_variance_ratio": [round(float(v), 4) for v in exp_var],
            "total_explained_variance": round(float(exp_var.sum()), 4),
            "extraction_time_seconds": round(elapsed, 2),
            "points": [
                {
                    "pc1": round(float(coords_2d[i, 0]), 4),
                    "pc2": round(float(coords_2d[i, 1]), 4),
                    "pred_prob": round(float(all_probs[i]), 4),
                    "pred_class": "Metastatic Carcinoma" if all_probs[i] >= 0.50 else "Normal Lymph Node",
                    "is_correct": bool((all_probs[i] >= 0.50) == (patch_labels[i] == 1)),
                }
                for i in range(len(patch_paths))
            ],
        }

    save_path = output_path or get_feature_cache_path()
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(projections_data, f, indent=2)

    print(f"[FEATURE ANALYSIS] Saved {len(patch_paths)} patch feature projections to: {save_path}")
    return projections_data


def load_cached_feature_projections() -> Optional[Dict[str, Any]]:
    """Loads cached feature projections if available, else returns None."""
    p = get_feature_cache_path()
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_morphology_audit_report() -> Dict[str, Any]:
    """
    Returns an authoritative scientific audit regarding morphological feature claims.
    Clarifies what features are measured vs inferred via CNN latent representations.
    """
    return {
        "audit_version": "PathoVision AI v2.6 Scientific Audit",
        "primary_inference_architecture": "Unimodal Deep Convolutional Neural Network (EfficientNet-B0 + ResNet + DenseNet + VGG)",
        "feature_representation": {
            "latent_vector_dimension": 1280,
            "pooling_type": "Adaptive Average Pooling (GAP)",
            "nature_of_representation": (
                "Continuous high-dimensional distributed neural embedding (z in R^1280). "
                "The latent space captures texture, color gradients, and hierarchical spatial patterns."
            ),
        },
        "continuous_physical_measurements": {
            "cell_segmentation_masks": False,
            "continuous_nuclear_diameters_microns": False,
            "nuclear_cytoplasmic_ratio_measured": False,
            "necrotic_pixel_surface_area_measured": False,
            "explicit_mitotic_counter": False,
            "explanation": (
                "PathoVision AI does NOT execute instance segmentation (e.g. StarDist or Cellpose) "
                "or continuous micrometer measurements during live inference. No physical cell borders "
                "or continuous micrometric geometric dimensions are dynamically computed."
            ),
        },
        "anomaly_probing_mechanism": {
            "mechanism": "Frozen Backbone Linear Probing",
            "active_heads": [
                {
                    "key": "ductal_distortion",
                    "name": "Ductal Structure Distortion",
                    "proxy_dataset": "BACH InSitu Breast Cohort",
                    "decision_threshold": 0.4038,
                    "interpretation": "Evaluates linear alignment with in-situ glandular disruption patterns.",
                },
                {
                    "key": "density_clusters",
                    "name": "Tumor Cell Density Clusters",
                    "proxy_dataset": "NuCLS High Cellularity Cohort (>=10 nuclei)",
                    "decision_threshold": 0.5315,
                    "interpretation": "Evaluates linear alignment with hyper-cellular tumor clustering patterns.",
                },
                {
                    "key": "tumor_necrosis",
                    "name": "Tumor Necrosis & Cellular Debris",
                    "proxy_dataset": "BCSS Necrosis Area Cohort (>=25% necrosis)",
                    "decision_threshold": 0.2715,
                    "interpretation": "Evaluates linear alignment with eosinophilic necrotic debris patterns.",
                },
            ],
            "quarantined_heads": [
                {
                    "key": "pleomorphism_atypia",
                    "name": "Nuclear Pleomorphism / Atypia",
                    "quarantine_reason": (
                        "Permanently quarantined due to cross-magnification divergence, subjective MITOS annotation "
                        "boundaries, and risk of confusing high grade atypia with physiological germinal center lymphocytes."
                    ),
                    "active_in_production": False,
                }
            ],
        },
        "historical_offline_data_clarification": {
            "offline_csv_files": ["nucls_spatial_features.csv", "nucls_features.csv"],
            "status": "Historical offline proxy dataset generation artifacts",
            "used_in_live_inference": False,
            "clarification": (
                "These CSV files contain offline spatial statistics computed from external NuCLS bounding boxes "
                "during exploratory training. They are NOT queried or computed during live inference on patient patches."
            ),
        },
    }
