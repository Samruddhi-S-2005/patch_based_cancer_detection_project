"""
Objective 2 Production Module: Whole Slide Image (WSI) Reconstruction & Heatmap Stitcher.

Implements medical-grade spatial stitching of patch predictions back into the
original coordinate space of Whole Slide Images:
  1. Reconstructs spatial coordinate grids (X, Y) from WSI metadata or streaming patch manifests.
  2. Runs 4-Layer Diagnostic Evaluation on every patch across the slide:
     - Layer 1: Quality filter
     - Layer 2: Malignancy probability P(Tumor)
     - Layer 3: 4-Way Anomaly indices (Pleomorphism, Ductal Distortion, Density, Necrosis)
     - Layer 4: Composite Risk level
  3. Reconstructs multi-color spatial heatmaps:
     - Continuous Malignancy Probability Heatmap (Red intensity)
     - Multi-Color Anomaly Contours:
       * Orange: Nuclear Pleomorphism (Atypia)
       * Purple: Ductal Structure Distortion
       * Yellow: Cell Density Clusters
       * Cyan: Tumor Necrosis & Debris
  4. Computes slide-level clinical aggregation metrics:
     - Overall WSI Classification (Malignant vs Normal)
     - Tumor Burden (% of total biopsy area involved)
     - Dominant Anomaly Type
     - Spatial Heatmap Overlay image saved to result/wsi_heatmaps/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.objective_2.layered_pipeline import (
    ANOMALY_COLORS,
    LayeredDiagnosticPipeline,
)


def reconstruct_wsi_heatmaps(
    wsi_id: str,
    manifest_csv: Path,
    pipeline: LayeredDiagnosticPipeline,
    output_dir: Path,
    patch_size: int = 224,
) -> Dict:
    """
    Reconstructs continuous WSI heatmaps and anomaly segmentations for a specific slide.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_csv)
    df_slide = df[(df["wsi_id"] == wsi_id) & (df["selected"] == True)].copy()

    if len(df_slide) == 0:
        raise ValueError(f"No selected test patches found for WSI: {wsi_id}")

    print(f"\n{'='*70}")
    print(f"PROCESSING WSI HEATMAP RECONSTRUCTION: {wsi_id.upper()}")
    print(f"Total Selected Patches: {len(df_slide)} (Ground Truth Tumors: {len(df_slide[df_slide['label']==1])})")
    print(f"{'='*70}")

    # Coordinate boundaries
    min_x, max_x = df_slide["x"].min(), df_slide["x"].max() + patch_size
    min_y, max_y = df_slide["y"].min(), df_slide["y"].max() + patch_size

    width_span = max_x - min_x
    height_span = max_y - min_y

    # Downsampling scale for macro-level 2D canvas (scale factor: 1/16 or 1/32)
    scale = 0.05  # downsampled grid representation
    grid_w = max(100, int(width_span * scale))
    grid_h = max(100, int(height_span * scale))

    # Initialize 2D spatial maps
    tissue_canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255  # white background
    prob_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    gt_map = np.zeros((grid_h, grid_w), dtype=np.uint8)

    anomaly_maps = {
        "atypia": np.zeros((grid_h, grid_w), dtype=np.float32),
        "ductal": np.zeros((grid_h, grid_w), dtype=np.float32),
        "density": np.zeros((grid_h, grid_w), dtype=np.float32),
        "necrosis": np.zeros((grid_h, grid_w), dtype=np.float32),
    }

    patch_results = []
    print("Evaluating patches through 4-Layer Diagnostic Engine...")

    for idx, row in df_slide.iterrows():
        patch_path_str = str(row["saved_path"]).replace("\\", "/")
        full_patch_path = PROJECT_ROOT / patch_path_str
        if not full_patch_path.exists():
            continue

        x_orig, y_orig = int(row["x"]), int(row["y"])
        gt_label = int(row["label"])

        # Run pipeline
        report = pipeline.analyze_patch(full_patch_path)
        p_tumor = report["layer_2_malignancy"]["probability"]
        anomalies = report["layer_3_anomalies"]

        patch_img = report["raw_image"]

        # Map to downscaled grid coordinates
        gx = int((x_orig - min_x) * scale)
        gy = int((y_orig - min_y) * scale)
        gw = max(2, int(patch_size * scale))
        gh = max(2, int(patch_size * scale))

        # Clamp boundaries
        gx_end = min(grid_w, gx + gw)
        gy_end = min(grid_h, gy + gh)

        # Place downsampled tissue patch on canvas
        resized_patch = cv2.resize(patch_img, (gx_end - gx, gy_end - gy))
        tissue_canvas[gy:gy_end, gx:gx_end] = resized_patch

        # Populate probability map (max pooling for overlapping strides)
        prob_map[gy:gy_end, gx:gx_end] = np.maximum(prob_map[gy:gy_end, gx:gx_end], p_tumor)
        gt_map[gy:gy_end, gx:gx_end] = np.maximum(gt_map[gy:gy_end, gx:gx_end], gt_label)

        if p_tumor >= 0.50:
            anomaly_maps["atypia"][gy:gy_end, gx:gx_end] = np.maximum(
                anomaly_maps["atypia"][gy:gy_end, gx:gx_end], anomalies["pleomorphism_atypia"]["score"]
            )
            anomaly_maps["ductal"][gy:gy_end, gx:gx_end] = np.maximum(
                anomaly_maps["ductal"][gy:gy_end, gx:gx_end], anomalies["ductal_distortion"]["score"]
            )
            anomaly_maps["density"][gy:gy_end, gx:gx_end] = np.maximum(
                anomaly_maps["density"][gy:gy_end, gx:gx_end], anomalies["density_clusters"]["score"]
            )
            anomaly_maps["necrosis"][gy:gy_end, gx:gx_end] = np.maximum(
                anomaly_maps["necrosis"][gy:gy_end, gx:gx_end], anomalies["tumor_necrosis"]["score"]
            )

        patch_results.append({
            "x": x_orig,
            "y": y_orig,
            "gt_label": gt_label,
            "pred_prob": p_tumor,
            "pred_label": 1 if p_tumor >= 0.50 else 0,
            "atypia_score": anomalies["pleomorphism_atypia"]["score"],
            "ductal_score": anomalies["ductal_distortion"]["score"],
            "density_score": anomalies["density_clusters"]["score"],
            "necrosis_score": anomalies["tumor_necrosis"]["score"],
        })

    # Slide-level summary calculations
    df_res = pd.DataFrame(patch_results)
    total_patches = len(df_res)
    tumor_patches = int((df_res["pred_prob"] >= 0.50).sum())
    tumor_fraction_pct = round((tumor_patches / total_patches) * 100, 2)
    wsi_verdict = "MALIGNANT" if tumor_patches >= 3 else "BENIGN / NORMAL"

    # Identify dominant anomaly if malignant
    dominant_anomaly = "None (Benign)"
    if wsi_verdict == "MALIGNANT":
        mean_anom = {
            "Nuclear Pleomorphism": df_res[df_res["pred_prob"] >= 0.5]["atypia_score"].mean(),
            "Ductal Structure Distortion": df_res[df_res["pred_prob"] >= 0.5]["ductal_score"].mean(),
            "Tumor Cell Density Clusters": df_res[df_res["pred_prob"] >= 0.5]["density_score"].mean(),
            "Tumor Necrosis / Debris": df_res[df_res["pred_prob"] >= 0.5]["necrosis_score"].mean(),
        }
        dominant_anomaly = max(mean_anom, key=mean_anom.get)

    print(f"\nWSI Slide Diagnostic Result:")
    print(f"  Diagnosis        : {wsi_verdict}")
    print(f"  Tumor Burden     : {tumor_fraction_pct}% of biopsy area ({tumor_patches}/{total_patches} patches)")
    print(f"  Dominant Anomaly : {dominant_anomaly}")

    # Generate 4-panel publication visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor="#f8f9fa")

    # Panel 1: Reconstructed Biopsy Tissue Canvas
    axes[0, 0].imshow(tissue_canvas)
    axes[0, 0].set_title(f"1. Reconstructed Biopsy Tissue (Macro-Level)\nWSI: {wsi_id} (Grid: {grid_w}x{grid_h})", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: Ground Truth Annotation Map
    axes[0, 1].imshow(tissue_canvas)
    gt_overlay = np.zeros_like(tissue_canvas)
    gt_overlay[gt_map == 1] = [214, 39, 40]  # Red
    axes[0, 1].imshow(gt_overlay, alpha=0.55)
    axes[0, 1].set_title(f"2. Ground Truth Pathologist Labels\n({int(df_slide['label'].sum())} Verified Tumor Patches)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel 3: Model Malignancy Probability Heatmap
    axes[1, 0].imshow(tissue_canvas)
    im_prob = axes[1, 0].imshow(prob_map, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[1, 0].set_title(f"3. AI Primary Malignancy Heatmap P(Tumor)\nVerdict: {wsi_verdict} (Burden: {tumor_fraction_pct}%)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im_prob, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Malignancy Confidence")

    # Panel 4: Multi-Color Anomaly Profiling Overlay
    axes[1, 1].imshow(tissue_canvas)
    legend_handles = []

    # Overlay anomaly contours if present
    anomaly_specs = [
        ("density", "Cell Density Clusters", [188, 189, 34], 0.5315),
        ("ductal", "Ductal Distortion", [148, 103, 189], 0.4038),
        ("necrosis", "Tumor Necrosis", [23, 190, 207], 0.2715),
        ("atypia", "Nuclear Pleomorphism", [255, 127, 14], 0.5000),
    ]

    for key, label, rgb, thresh in anomaly_specs:
        mask = anomaly_maps[key] >= thresh
        if np.any(mask):
            anom_layer = np.zeros_like(tissue_canvas)
            anom_layer[mask] = rgb
            axes[1, 1].imshow(anom_layer, alpha=0.45)
            legend_handles.append(mpatches.Patch(color=np.array(rgb)/255.0, label=label))

    axes[1, 1].set_title(f"4. Multi-Color Histopathological Anomaly Map\nDominant: {dominant_anomaly}", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")
    if legend_handles:
        axes[1, 1].legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    plt.suptitle(f"Whole Slide Biopsy Diagnostic Reconstruction Dashboard — {wsi_id.upper()}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = output_dir / f"{wsi_id}_reconstructed_diagnostic_heatmap.png"
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"[SAVED] WSI Heatmap saved: {save_path}")

    # Save summary JSON
    summary_data = {
        "wsi_id": wsi_id,
        "total_patches": total_patches,
        "tumor_patches": tumor_patches,
        "tumor_fraction_percent": tumor_fraction_pct,
        "wsi_diagnosis": wsi_verdict,
        "dominant_anomaly": dominant_anomaly,
        "heatmap_file": str(save_path.name),
    }
    with open(output_dir / f"{wsi_id}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    return summary_data


def main():
    manifest_csv = PROJECT_ROOT / "data" / "PATCHES" / "TEST" / "test_patch_manifest.csv"
    output_dir = PROJECT_ROOT / "result" / "wsi_heatmaps"

    cls_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt"
    anom_ckpt = PROJECT_ROOT / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_best.pt"

    pipeline = LayeredDiagnosticPipeline(cls_ckpt, anom_ckpt)

    # Reconstruct 1 Normal WSI and 1 Tumor WSI for demonstration:
    # tumor_058 (96 tumor patches, 196 normal patches)
    # normal_004 (300 normal patches)
    test_slides = ["tumor_058", "normal_004"]

    for sid in test_slides:
        reconstruct_wsi_heatmaps(
            wsi_id=sid,
            manifest_csv=manifest_csv,
            pipeline=pipeline,
            output_dir=output_dir,
        )

    print("\n[ALL WSI RECONSTRUCTIONS COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
