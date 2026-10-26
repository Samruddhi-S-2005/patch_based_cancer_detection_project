"""
Objective 2 Part B — Anomaly Inference Pipeline.

Runs both frozen models over a single patch or a directory of 224×224 patches
and produces per-patch predictions for four signals:

  1. tumor_probability      — from Objective-2A EfficientNet-B0 tumor classifier
  2. ductal_probability     — DCIS/ductal architectural proxy (BACH-trained head)
  3. density_probability    — high tumor-nucleus density (NuCLS-trained head)
  4. necrosis_probability   — necrosis/debris (BCSS-trained head)

Anomaly thresholds are the validation-selected values from training.
Do NOT recompute thresholds using inference data.

DISCLAIMER:
  Anomaly outputs are model predictions learned from external datasets.
  CAMELYON16 inference is exploratory because corresponding ground-truth labels
  for these anomaly types are unavailable. These outputs are not clinically
  validated.

FROZEN RULES (must NOT be violated):
  - Do NOT modify, overwrite, or retrain either model checkpoint.
  - Do NOT fine-tune any EfficientNet backbone layer.
  - Do NOT calculate accuracy/AUC on CAMELYON anomaly predictions
    (no ground truth available for these anomaly types on CAMELYON).
  - Do NOT claim these are clinical risk scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.objective_2.anomaly_model import EfficientNetB0MultiHeadAnomaly
from src.objective_2.dataset import get_eval_transforms
from src.objective_2.model import EfficientNetB0TumorClassifier

# ---------------------------------------------------------------------------
# Constants — validation-selected thresholds (must not change)
# ---------------------------------------------------------------------------
ANOMALY_THRESHOLDS: Dict[str, float] = {
    "ductal":   0.40379902720451355,   # BACH validation optimal
    "density":  0.5314773321151733,    # NuCLS validation optimal
    "necrosis": 0.27145883440971375,   # BCSS validation optimal
}

TUMOR_CLASSIFIER_CKPT  = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt"
# Clean checkpoint — 3 permitted heads only (ductal/density/necrosis).
# The contaminated 4-head file is preserved for forensic inspection at:
#   backups/pre_remediation_20260902/anomaly_checkpoints/anomaly_linear_heads_best.pt
ANOMALY_MODEL_CKPT     = PROJECT_ROOT / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_clean.pt"
INFERENCE_OUT_DIR      = PROJECT_ROOT / "result" / "objective_2" / "anomaly_model" / "inference"

DISCLAIMER = (
    "Anomaly outputs are model predictions learned from external datasets. "
    "CAMELYON16 inference is exploratory because corresponding ground-truth labels "
    "for these anomaly types are unavailable. These outputs are not clinically validated."
)

# ---------------------------------------------------------------------------
# Preprocessing — imported from the shared dataset module to guarantee
# exact consistency with training and evaluation (Task 4 fix).
# get_eval_transforms() = ToTensor + ImageNet Normalize, NO augmentation.
# ---------------------------------------------------------------------------
EVAL_TRANSFORM = get_eval_transforms()


# ---------------------------------------------------------------------------
# Lightweight Dataset for inference (no labels required)
# ---------------------------------------------------------------------------
class InferencePatchDataset(Dataset):
    """
    Loads 224×224 PNG patches from a list of paths.
    Applies the same deterministic eval transform used in training/evaluation.
    No labels are required or produced.
    """

    def __init__(self, patch_paths: List[Path]) -> None:
        self.patch_paths = patch_paths

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        p = self.patch_paths[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
            tensor = EVAL_TRANSFORM(img)
        return tensor, str(p)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_tumor_classifier(device: torch.device) -> EfficientNetB0TumorClassifier:
    """
    Loads Objective-2A best_model.pt into EfficientNetB0TumorClassifier.
    Strictly read-only — checkpoint is never written.
    """
    if not TUMOR_CLASSIFIER_CKPT.exists():
        raise FileNotFoundError(f"Tumor classifier checkpoint not found: {TUMOR_CLASSIFIER_CKPT}")

    ckpt = torch.load(str(TUMOR_CLASSIFIER_CKPT), map_location="cpu", weights_only=False)
    model = EfficientNetB0TumorClassifier(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Freeze everything — inference only
    for p in model.parameters():
        p.requires_grad_(False)

    return model


def load_anomaly_model(device: torch.device) -> Tuple[EfficientNetB0MultiHeadAnomaly, Dict[str, float]]:
    """
    Loads the trained anomaly linear-probe checkpoint.
    Returns (model, optimal_thresholds_dict).
    Strictly read-only — checkpoint is never written.
    """
    if not ANOMALY_MODEL_CKPT.exists():
        raise FileNotFoundError(f"Anomaly model checkpoint not found: {ANOMALY_MODEL_CKPT}")

    ckpt = torch.load(str(ANOMALY_MODEL_CKPT), map_location="cpu", weights_only=False)
    model = EfficientNetB0MultiHeadAnomaly(device=device)
    # strict=False: the clean checkpoint omits atypia head weights by design.
    # The atypia slot remains at random init and is never called in production paths.
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Error only if any of the three PERMITTED heads are missing
    permitted_missing = [k for k in missing if any(t in k for t in ["ductal", "density", "necrosis"])]
    if permitted_missing:
        raise RuntimeError(f"Unexpected missing keys for permitted heads: {permitted_missing}")
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    # Use the stored thresholds as a cross-check, but always defer to
    # the module-level ANOMALY_THRESHOLDS constants (validation-calibrated).
    stored_thresholds = ckpt.get("optimal_thresholds", {})
    return model, stored_thresholds


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_batch_inference(
    tumor_model: EfficientNetB0TumorClassifier,
    anomaly_model: EfficientNetB0MultiHeadAnomaly,
    patch_paths: List[Path],
    device: torch.device,
    batch_size: int = 32,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Runs both models over all patch_paths in batches.
    Returns a DataFrame with one row per patch, containing all four
    probabilities, binary flags, and anomaly type columns.
    """
    dataset = InferencePatchDataset(patch_paths)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_rows: List[Dict] = []
    t0 = time.time()

    for batch_idx, (images, paths) in enumerate(loader):
        images = images.to(device)

        # --- Tumor classifier ---
        tumor_logits = tumor_model(images).cpu().numpy().ravel()
        tumor_probs  = 1.0 / (1.0 + np.exp(-tumor_logits))

        # --- Anomaly: extract 1280-D features once, reuse for all three heads ---
        feat = anomaly_model.extract_features(images)  # [B, 1280]

        ductal_logits   = anomaly_model.forward_features(feat, "ductal").cpu().numpy().ravel()
        density_logits  = anomaly_model.forward_features(feat, "density").cpu().numpy().ravel()
        necrosis_logits = anomaly_model.forward_features(feat, "necrosis").cpu().numpy().ravel()

        ductal_probs   = 1.0 / (1.0 + np.exp(-ductal_logits))
        density_probs  = 1.0 / (1.0 + np.exp(-density_logits))
        necrosis_probs = 1.0 / (1.0 + np.exp(-necrosis_logits))

        for i, path_str in enumerate(paths):
            d_flag = int(ductal_probs[i]   >= ANOMALY_THRESHOLDS["ductal"])
            n_flag = int(density_probs[i]  >= ANOMALY_THRESHOLDS["density"])
            k_flag = int(necrosis_probs[i] >= ANOMALY_THRESHOLDS["necrosis"])

            # Multiple anomalies?
            n_positive = d_flag + n_flag + k_flag
            multiple_anomalies = int(n_positive > 1)

            # Primary anomaly type — highest scoring among those above threshold;
            # if none above threshold → "none_above_threshold"
            scores_above = {}
            if d_flag:
                scores_above["ductal"]   = float(ductal_probs[i])
            if n_flag:
                scores_above["density"]  = float(density_probs[i])
            if k_flag:
                scores_above["necrosis"] = float(necrosis_probs[i])

            if scores_above:
                primary_type = max(scores_above, key=scores_above.__getitem__)
                highest_anomaly_score = scores_above[primary_type]
            else:
                primary_type = "none_above_threshold"
                highest_anomaly_score = float(max(
                    ductal_probs[i], density_probs[i], necrosis_probs[i]
                ))

            all_rows.append({
                "patch_path":            path_str,
                "tumor_probability":     float(tumor_probs[i]),
                "ductal_probability":    float(ductal_probs[i]),
                "density_probability":   float(density_probs[i]),
                "necrosis_probability":  float(necrosis_probs[i]),
                "ductal_flag":           d_flag,
                "density_flag":          n_flag,
                "necrosis_flag":         k_flag,
                "primary_anomaly_type":  primary_type,
                "multiple_anomalies_flag": multiple_anomalies,
                "highest_anomaly_score": highest_anomaly_score,
            })

        if logger and (batch_idx + 1) % 5 == 0:
            elapsed = time.time() - t0
            done = (batch_idx + 1) * batch_size
            logger.info(f"  Processed ~{min(done, len(patch_paths))}/{len(patch_paths)} patches "
                        f"in {elapsed:.1f}s")

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Sanity test
# ---------------------------------------------------------------------------
def run_sanity_test(
    tumor_model: EfficientNetB0TumorClassifier,
    anomaly_model: EfficientNetB0MultiHeadAnomaly,
    sample_patches: List[Path],
    device: torch.device,
    logger: logging.Logger,
) -> bool:
    """
    Runs a targeted pre-flight check on 5 patches:
      - model loading (already done by caller)
      - tensor shape [1, 3, 224, 224] into backbone
      - 1280-D embedding shape
      - all four probabilities in [0, 1]
      - determinism: two identical runs produce identical logits
    Returns True on pass, False on failure.
    """
    logger.info("=" * 60)
    logger.info("SANITY TEST — 5 patches")
    logger.info("=" * 60)

    n = min(5, len(sample_patches))
    test_paths = sample_patches[:n]
    logger.info(f"  Patches used: {[p.name for p in test_paths]}")

    # Single image shape check
    with Image.open(test_paths[0]) as img:
        img = img.convert("RGB")
        t = EVAL_TRANSFORM(img).unsqueeze(0).to(device)

    if t.shape != (1, 3, 224, 224):
        logger.error(f"[FAIL] Input tensor shape: {t.shape} (expected [1, 3, 224, 224])")
        return False
    logger.info(f"  [PASS] Input tensor shape: {t.shape}")

    with torch.no_grad():
        # 1280-D embedding
        feat = anomaly_model.extract_features(t)
        if feat.shape != (1, 1280):
            logger.error(f"[FAIL] Embedding shape: {feat.shape} (expected [1, 1280])")
            return False
        logger.info(f"  [PASS] 1280-D embedding shape: {feat.shape}")

        # Probabilities in [0,1]
        tumor_prob   = float(torch.sigmoid(tumor_model(t)).item())
        ductal_prob  = float(torch.sigmoid(anomaly_model.forward_features(feat, "ductal")).item())
        density_prob = float(torch.sigmoid(anomaly_model.forward_features(feat, "density")).item())
        necrosis_prob= float(torch.sigmoid(anomaly_model.forward_features(feat, "necrosis")).item())

    for name, val in [
        ("tumor",    tumor_prob),
        ("ductal",   ductal_prob),
        ("density",  density_prob),
        ("necrosis", necrosis_prob),
    ]:
        if not (0.0 <= val <= 1.0):
            logger.error(f"[FAIL] {name}_probability={val:.6f} not in [0,1]")
            return False
        logger.info(f"  [PASS] {name}_probability={val:.4f} ∈ [0,1]")

    # Determinism check — run twice, compare logits
    with torch.no_grad():
        logit1 = tumor_model(t).item()
        logit2 = tumor_model(t).item()
    if logit1 != logit2:
        logger.error(f"[FAIL] Determinism check failed: {logit1} vs {logit2}")
        return False
    logger.info(f"  [PASS] Determinism: tumor logit={logit1:.8f} (identical on repeat)")

    # Quick run through all 5 patches
    df_sanity = run_batch_inference(tumor_model, anomaly_model, test_paths, device)
    if len(df_sanity) != n:
        logger.error(f"[FAIL] Expected {n} rows, got {len(df_sanity)}")
        return False
    required_cols = [
        "patch_path", "tumor_probability", "ductal_probability",
        "density_probability", "necrosis_probability",
        "ductal_flag", "density_flag", "necrosis_flag",
        "primary_anomaly_type", "multiple_anomalies_flag",
    ]
    missing = [c for c in required_cols if c not in df_sanity.columns]
    if missing:
        logger.error(f"[FAIL] Missing columns: {missing}")
        return False
    logger.info(f"  [PASS] Output CSV shape: {df_sanity.shape}, all required columns present")
    logger.info(f"  [PASS] Sample output:")
    for _, row in df_sanity.iterrows():
        logger.info(
            f"    {Path(row.patch_path).name} | "
            f"tumor={row.tumor_probability:.3f} | "
            f"ductal={row.ductal_probability:.3f} | "
            f"density={row.density_probability:.3f} | "
            f"necrosis={row.necrosis_probability:.3f} | "
            f"primary={row.primary_anomaly_type}"
        )

    logger.info("  [PASS] ALL SANITY CHECKS PASSED")
    return True


# ---------------------------------------------------------------------------
# Manifest-based patch selection (Task 1 fix)
# ---------------------------------------------------------------------------
def select_patches_from_manifest(
    manifest_path: Path,
    max_patches: int,
    logger: logging.Logger,
) -> Tuple[List[Path], List[int], List[str]]:
    """
    Selects patches from the CAMELYON16 test manifest using VERIFIED labels,
    NOT directory names.  Ensures both normal (label 0) and tumour (label 1)
    patches are included in the sample.

    Returns (patch_paths, reference_labels, wsi_ids).
    Raises RuntimeError if either class is missing from the manifest.
    """
    df = pd.read_csv(manifest_path)
    keep = df[df["final_decision"] == "KEEP"].copy()

    label0 = keep[keep["label"] == 0]
    label1 = keep[keep["label"] == 1]

    logger.info(f"[MANIFEST] {manifest_path.name}: {len(keep)} KEEP patches "
                f"(label0={len(label0)}, label1={len(label1)})")

    if len(label0) == 0:
        raise RuntimeError("Manifest has ZERO normal (label 0) patches — cannot build balanced sample.")
    if len(label1) == 0:
        raise RuntimeError("Manifest has ZERO tumour (label 1) patches — cannot build balanced sample.")

    # Split max_patches roughly equally between the two classes
    half = max_patches // 2
    n_label0 = min(half, len(label0))
    n_label1 = min(max_patches - n_label0, len(label1))
    # If label1 has fewer than half, give the remainder to label0
    if n_label0 + n_label1 < max_patches:
        n_label0 = min(max_patches - n_label1, len(label0))

    # Deterministic, reproducible sampling (seed=42)
    sampled_0 = label0.sample(n=n_label0, random_state=42)
    sampled_1 = label1.sample(n=n_label1, random_state=42)

    patch_paths: List[Path] = []
    reference_labels: List[int] = []
    wsi_ids: List[str] = []

    for _, row in pd.concat([sampled_0, sampled_1]).iterrows():
        rel_path = str(row["saved_path"])
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            logger.warning(f"  Patch file not found on disk: {full_path}")
            continue
        patch_paths.append(full_path)
        reference_labels.append(int(row["label"]))
        wsi_ids.append(str(row["wsi_id"]))

    # Log selection details
    logger.info(f"[SELECT] Patches selected for inference: {len(patch_paths)}")
    logger.info(f"  Normal (label 0) : {sum(1 for l in reference_labels if l == 0)}")
    logger.info(f"  Tumour (label 1) : {sum(1 for l in reference_labels if l == 1)}")
    logger.info(f"  Unique WSIs      : {sorted(set(wsi_ids))}")
    logger.info("  Per-patch selection log:")
    for i, (p, lbl, wid) in enumerate(zip(patch_paths, reference_labels, wsi_ids)):
        logger.info(f"    [{i:3d}] {p.name} | label={lbl} | wsi={wid}")

    if not any(l == 0 for l in reference_labels):
        raise RuntimeError("Sample contains ZERO normal patches after file-existence check.")
    if not any(l == 1 for l in reference_labels):
        raise RuntimeError("Sample contains ZERO tumour patches after file-existence check.")

    return patch_paths, reference_labels, wsi_ids
def build_inference_summary(df: pd.DataFrame) -> Dict:
    """
    Builds a summary of inference results.
    Deliberately excludes accuracy/AUC — there is no ground-truth anomaly
    label for CAMELYON16 patches.
    """
    n = len(df)

    def _range(col: str) -> Dict:
        return {
            "min":  float(df[col].min()),
            "max":  float(df[col].max()),
            "mean": float(df[col].mean()),
            "std":  float(df[col].std()),
        }

    ductal_pos   = int(df["ductal_flag"].sum())
    density_pos  = int(df["density_flag"].sum())
    necrosis_pos = int(df["necrosis_flag"].sum())
    multi_pos    = int(df["multiple_anomalies_flag"].sum())
    none_pos     = int((df["primary_anomaly_type"] == "none_above_threshold").sum())

    primary_counts = df["primary_anomaly_type"].value_counts().to_dict()

    # Reference label breakdown (Task 1 — from manifest, not folder names)
    ref_counts = {}
    if "reference_label" in df.columns:
        ref_counts = {
            "normal_label0": int((df["reference_label"] == 0).sum()),
            "tumor_label1": int((df["reference_label"] == 1).sum()),
            "unknown": int((df["reference_label"] == -1).sum()),
        }

    return {
        "disclaimer": DISCLAIMER,
        "n_patches_processed": n,
        "reference_label_counts": ref_counts,
        "tumor_probability":   _range("tumor_probability"),
        "ductal_probability":  _range("ductal_probability"),
        "density_probability": _range("density_probability"),
        "necrosis_probability":_range("necrosis_probability"),
        "anomaly_thresholds_used": ANOMALY_THRESHOLDS,
        "anomaly_flag_counts": {
            "ductal_positive":   ductal_pos,
            "density_positive":  density_pos,
            "necrosis_positive": necrosis_pos,
        },
        "patches_with_multiple_anomalies": multi_pos,
        "patches_no_anomaly_above_threshold": none_pos,
        "primary_anomaly_type_distribution": primary_counts,
        "note_no_auc_computed": (
            "No accuracy, AUC, or confusion matrix is computed for CAMELYON16 anomaly "
            "predictions because corresponding ground-truth labels for ductal, density, "
            "and necrosis anomaly types are unavailable in this dataset."
        ),
    }


# ---------------------------------------------------------------------------
# Visualization — one real example figure
# ---------------------------------------------------------------------------
def save_example_figure(
    df: pd.DataFrame,
    out_path: Path,
    n_examples: int = 6,
) -> None:
    """
    Shows n_examples real CAMELYON16 patches with all four probabilities
    and predicted anomaly type annotated.  No synthetic images.
    """
    # Select a diverse set: try to include different primary_anomaly_type values
    samples = []
    for ptype in ["ductal", "density", "necrosis", "none_above_threshold"]:
        subset = df[df["primary_anomaly_type"] == ptype]
        if len(subset) > 0:
            samples.append(subset.sample(1, random_state=42))
    # Fill remaining slots from full df
    already = set(s.index[0] for s in samples)
    remaining = df.drop(index=list(already))
    need = max(0, n_examples - len(samples))
    if need > 0 and len(remaining) > 0:
        samples.append(remaining.sample(min(need, len(remaining)), random_state=42))

    df_plot = pd.concat(samples).head(n_examples).reset_index(drop=True)
    n = len(df_plot)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    for i, row in df_plot.iterrows():
        ax = axes[i]
        try:
            with Image.open(row["patch_path"]) as img:
                ax.imshow(np.array(img.convert("RGB")))
        except Exception:
            ax.set_facecolor("#cccccc")

        title = (
            f"tumor={row['tumor_probability']:.2f}  "
            f"ductal={row['ductal_probability']:.2f}\n"
            f"density={row['density_probability']:.2f}  "
            f"necrosis={row['necrosis_probability']:.2f}\n"
            f"primary: {row['primary_anomaly_type']}"
        )
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.axis("off")

        # Colour border by primary anomaly type
        colour_map = {
            "ductal":              "#e07b00",
            "density":             "#1a7abf",
            "necrosis":            "#d62728",
            "none_above_threshold":"#555555",
        }
        border_col = colour_map.get(row["primary_anomaly_type"], "#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(3)

    # Hide unused axes if n < 6
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "CAMELYON16 Test Patches — Exploratory Anomaly Inference\n"
        "(DISCLAIMER: no ground-truth anomaly labels; not clinically validated)",
        fontsize=10, style="italic", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("AnomalyInference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Objective 2B Anomaly Inference — frozen models, no retraining."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Single patch path or directory of patches. "
             "Default: 100 patches from CAMELYON16 TEST set.",
    )
    parser.add_argument(
        "--max_patches", type=int, default=100,
        help="Maximum number of patches to process when --input is a directory. "
             "Default: 100.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Inference batch size. Default: 32.",
    )
    parser.add_argument(
        "--sanity_only", action="store_true",
        help="Run sanity test only (5 patches), then exit.",
    )
    args = parser.parse_args()

    INFERENCE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(INFERENCE_OUT_DIR / "predict_anomalies.log")

    logger.info("=" * 70)
    logger.info("OBJECTIVE 2 PART B — ANOMALY INFERENCE PIPELINE")
    logger.info("=" * 70)
    logger.info(f"DISCLAIMER: {DISCLAIMER}")
    logger.info("")
    logger.info(f"Tumor classifier  : {TUMOR_CLASSIFIER_CKPT}")
    logger.info(f"Anomaly model     : {ANOMALY_MODEL_CKPT}")
    logger.info(f"Output directory  : {INFERENCE_OUT_DIR}")
    logger.info("")
    logger.info("Preprocessing (imported from dataset.get_eval_transforms):")
    logger.info("  transforms.ToTensor()")
    logger.info("  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
    logger.info("  No augmentation applied.")
    logger.info("")
    logger.info(f"Anomaly thresholds (validation-selected, immutable):")
    for k, v in ANOMALY_THRESHOLDS.items():
        logger.info(f"  {k:10s}: {v:.6f}")
    logger.info("")

    # ------------------------------------------------------------------
    # 1. Device
    # ------------------------------------------------------------------
    device = torch.device("cpu")
    logger.info(f"Compute device: {device}")

    # ------------------------------------------------------------------
    # 2. Load models
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[LOAD] Loading tumor classifier (Objective-2A best_model.pt)...")
    tumor_model = load_tumor_classifier(device)
    logger.info("  [OK] EfficientNetB0TumorClassifier loaded, eval mode, fully frozen.")

    logger.info("[LOAD] Loading anomaly linear-probe model...")
    anomaly_model, stored_thresholds = load_anomaly_model(device)

    # Cross-check stored thresholds against module constants
    threshold_match = all(
        abs(stored_thresholds.get(k, 0) - ANOMALY_THRESHOLDS[k]) < 1e-9
        for k in ANOMALY_THRESHOLDS
    )
    if threshold_match:
        logger.info("  [OK] Stored thresholds match module constants exactly.")
    else:
        logger.warning(
            f"  [WARN] Threshold mismatch: stored={stored_thresholds}, "
            f"using module constants={ANOMALY_THRESHOLDS}"
        )

    param_s = anomaly_model.get_parameter_summary()
    logger.info(
        f"  [OK] Anomaly model: total={param_s['total_parameters']:,} | "
        f"frozen={param_s['frozen_parameters']:,} | "
        f"trainable={param_s['trainable_parameters']:,} (heads only)"
    )

    # ------------------------------------------------------------------
    # 3. Collect patches
    # ------------------------------------------------------------------
    logger.info("")
    if args.input is not None:
        input_path = Path(args.input)
        if input_path.is_file():
            all_patches = [input_path]
            logger.info(f"[INPUT] Single patch: {input_path}")
        elif input_path.is_dir():
            all_patches = sorted(input_path.rglob("*.png"))
            logger.info(f"[INPUT] Directory: {input_path} → {len(all_patches)} PNG files found")
        else:
            logger.error(f"Input path does not exist: {input_path}")
            sys.exit(1)
    else:
        # Default: CAMELYON16 TEST patches
        default_test_dir = PROJECT_ROOT / "data" / "PATCHES" / "TEST"
        if not default_test_dir.exists():
            logger.error(f"Default CAMELYON16 TEST directory not found: {default_test_dir}")
            sys.exit(1)
        all_patches = sorted(default_test_dir.rglob("*.png"))
        logger.info(
            f"[INPUT] Default CAMELYON16 TEST directory: "
            f"{default_test_dir} → {len(all_patches)} patches total"
        )

    if len(all_patches) == 0:
        logger.error("No PNG patches found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Sanity test (always run first, regardless of --sanity_only)
    # ------------------------------------------------------------------
    sanity_patches = all_patches[:5]
    passed = run_sanity_test(tumor_model, anomaly_model, sanity_patches, device, logger)

    if not passed:
        logger.error("SANITY TEST FAILED. Aborting inference.")
        sys.exit(1)
    logger.info("")

    if args.sanity_only:
        logger.info("--sanity_only flag set. Stopping after sanity test.")
        return

    # ------------------------------------------------------------------
    # 5. Select patches for inference
    # ------------------------------------------------------------------
    if args.input is None:
        # Default: CAMELYON16 TEST — use manifest-based selection (Task 1 fix)
        test_manifest = PROJECT_ROOT / "data" / "PATCHES" / "TEST" / "test_patch_manifest.csv"
        if not test_manifest.exists():
            logger.error(f"Test manifest not found: {test_manifest}")
            sys.exit(1)
        selected, ref_labels, wsi_ids = select_patches_from_manifest(
            test_manifest, args.max_patches, logger,
        )
    else:
        # Custom input: use directory/file listing (no manifest labels available)
        selected = all_patches[:args.max_patches]
        ref_labels = [-1] * len(selected)  # unknown label
        wsi_ids = ["unknown"] * len(selected)
        logger.info(f"[SELECT] Custom input: {len(selected)} patches (labels unknown)")

    logger.info(
        "  NOTE: subdir names reflect CAMELYON WSI type, NOT anomaly ground truth. "
        "No AUC/accuracy is computed for anomaly outputs."
    )

    # ------------------------------------------------------------------
    # 6. Full inference
    # ------------------------------------------------------------------
    logger.info("")
    logger.info(f"[INFERENCE] Running inference on {len(selected)} patches...")
    t_start = time.time()
    df_results = run_batch_inference(
        tumor_model, anomaly_model, selected, device,
        batch_size=args.batch_size, logger=logger,
    )
    elapsed = time.time() - t_start
    logger.info(f"  Done in {elapsed:.1f}s ({elapsed/len(selected):.3f}s/patch)")

    # Add manifest-verified reference labels and WSI IDs (Task 1)
    df_results["reference_label"] = ref_labels
    df_results["wsi_id"] = wsi_ids

    # ------------------------------------------------------------------
    # 7. Save predictions CSV
    # ------------------------------------------------------------------
    csv_path = INFERENCE_OUT_DIR / "camelyon16_anomaly_predictions.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"  Saved predictions CSV: {csv_path} ({len(df_results)} rows)")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    summary = build_inference_summary(df_results)
    summary_path = INFERENCE_OUT_DIR / "inference_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved summary JSON: {summary_path}")

    # ------------------------------------------------------------------
    # 9. Print summary to log
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("INFERENCE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Patches processed        : {summary['n_patches_processed']}")
    logger.info("")
    for signal in ["tumor", "ductal", "density", "necrosis"]:
        r = summary[f"{signal}_probability"]
        logger.info(
            f"  {signal:10s} prob range: "
            f"[{r['min']:.4f}, {r['max']:.4f}] "
            f"mean={r['mean']:.4f} std={r['std']:.4f}"
        )
    logger.info("")
    fc = summary["anomaly_flag_counts"]
    logger.info(f"  Ductal flag positive     : {fc['ductal_positive']}  (τ={ANOMALY_THRESHOLDS['ductal']:.4f})")
    logger.info(f"  Density flag positive    : {fc['density_positive']}  (τ={ANOMALY_THRESHOLDS['density']:.4f})")
    logger.info(f"  Necrosis flag positive   : {fc['necrosis_positive']}  (τ={ANOMALY_THRESHOLDS['necrosis']:.4f})")
    logger.info(f"  Multiple anomalies       : {summary['patches_with_multiple_anomalies']}")
    logger.info(f"  No anomaly above threshold: {summary['patches_no_anomaly_above_threshold']}")
    logger.info("")
    logger.info("  Primary anomaly type distribution:")
    for k, v in sorted(summary["primary_anomaly_type_distribution"].items()):
        logger.info(f"    {k:30s}: {v}")
    logger.info("")
    logger.info(f"  {summary['note_no_auc_computed']}")
    logger.info("")
    logger.info(f"  DISCLAIMER: {DISCLAIMER}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 10. Example figure
    # ------------------------------------------------------------------
    fig_path = INFERENCE_OUT_DIR / "example_predictions.png"
    try:
        save_example_figure(df_results, fig_path)
        logger.info(f"  Saved example figure: {fig_path}")
    except Exception as e:
        logger.warning(f"  Could not save example figure: {e}")

    logger.info("")
    logger.info("INFERENCE PIPELINE COMPLETE.")


if __name__ == "__main__":
    main()
