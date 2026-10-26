"""
Whole Slide Image Scatter and Interpolation Diagnostic Visualizer.

Instead of displaying a giant 3-gigapixel canvas for a sparse test set of 300 patches,
this script renders:
  1. The Spatial Patch Coordinate Map on the whole slide (showing where on the biopsy each patch was taken).
  2. Ground Truth Diagnosis per Patch (Green=Normal, Red=Tumor).
  3. AI Model Malignancy Heatmap (continuous interpolated confidence field over tissue regions).
  4. Multi-Color Histopathological Anomaly Footprint:
     - Yellow: Cell Density Clusters
     - Purple: Ductal Distortion
     - Cyan: Tumor Necrosis
     - Orange: Nuclear Pleomorphism
  5. Slide-Level Diagnosis Card & Tumor Burden %.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from src.objective_2.layered_pipeline import LayeredDiagnosticPipeline


def render_clinical_wsi_dashboard(
    wsi_id: str,
    manifest_csv: Path,
    pipeline: LayeredDiagnosticPipeline,
    output_dir: Path,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_csv)
    df_slide = df[(df["wsi_id"] == wsi_id) & (df["selected"] == True)].copy()

    if len(df_slide) == 0:
        raise ValueError(f"No selected patches for WSI {wsi_id}")

    print(f"\nProcessing WSI Spatial Diagnostic Map: {wsi_id.upper()} ({len(df_slide)} patches)...")

    results = []
    for _, row in df_slide.iterrows():
        p_path = PROJECT_ROOT / str(row["saved_path"]).replace("\\", "/")
        if not p_path.exists():
            continue

        report = pipeline.analyze_patch(p_path)
        p_tumor = report["layer_2_malignancy"]["probability"]
        anoms = report["layer_3_anomalies"]

        results.append({
            "x": int(row["x"]),
            "y": int(row["y"]),
            "gt_label": int(row["label"]),
            "pred_prob": p_tumor,
            "pred_label": 1 if p_tumor >= 0.50 else 0,
            "atypia": anoms["pleomorphism_atypia"]["score"],
            "ductal": anoms["ductal_distortion"]["score"],
            "density": anoms["density_clusters"]["score"],
            "necrosis": anoms["tumor_necrosis"]["score"],
        })

    df_res = pd.DataFrame(results)

    total_patches = len(df_res)
    pred_tumors = int((df_res["pred_prob"] >= 0.50).sum())
    gt_tumors = int((df_res["gt_label"] == 1).sum())
    tumor_burden_pct = round((pred_tumors / total_patches) * 100, 2)

    # Slide verdict rule: If tumor burden >= 5% or >= 10 tumor patches -> Malignant
    is_malignant = pred_tumors >= 10 or tumor_burden_pct >= 5.0
    slide_verdict = "MALIGNANT" if is_malignant else "BENIGN / NORMAL"

    dominant_anomaly = "None (Benign)"
    if is_malignant:
        tum_df = df_res[df_res["pred_prob"] >= 0.50]
        means = {
            "Nuclear Pleomorphism": tum_df["atypia"].mean(),
            "Ductal Structure Distortion": tum_df["ductal"].mean(),
            "Tumor Cell Density Clusters": tum_df["density"].mean(),
            "Tumor Necrosis / Debris": tum_df["necrosis"].mean(),
        }
        dominant_anomaly = max(means, key=means.get)

    # 4-Panel Clinical Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor="#f8f9fa")

    # Panel 1: Spatial Sampling Grid
    ax1 = axes[0, 0]
    ax1.set_facecolor("#ffffff")
    ax1.scatter(df_res["x"], df_res["y"], c="#4a5568", s=35, alpha=0.7, edgecolors="none")
    ax1.set_title(f"1. Biopsy Sampling Grid (Coordinate Space)\nWSI: {wsi_id.upper()} ({total_patches} Verified Patches)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("WSI X Coordinate (pixels)", fontsize=10)
    ax1.set_ylabel("WSI Y Coordinate (pixels)", fontsize=10)
    ax1.invert_yaxis()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Ground Truth Diagnosis
    ax2 = axes[0, 1]
    ax2.set_facecolor("#ffffff")
    norm_mask = df_res["gt_label"] == 0
    tum_mask = df_res["gt_label"] == 1
    ax2.scatter(df_res.loc[norm_mask, "x"], df_res.loc[norm_mask, "y"], c="#2ca02c", s=40, label=f"Normal Patches (N={norm_mask.sum()})", alpha=0.8)
    if tum_mask.sum() > 0:
        ax2.scatter(df_res.loc[tum_mask, "x"], df_res.loc[tum_mask, "y"], c="#d62728", s=65, label=f"Pathologist Tumor (N={tum_mask.sum()})", edgecolors="black", linewidths=0.8)
    ax2.set_title(f"2. Ground Truth Pathologist Labels\nVerified Tumor Patches: {gt_tumors}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("WSI X Coordinate (pixels)", fontsize=10)
    ax2.set_ylabel("WSI Y Coordinate (pixels)", fontsize=10)
    ax2.invert_yaxis()
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: AI Malignancy Probability Heatmap
    ax3 = axes[1, 0]
    ax3.set_facecolor("#ffffff")
    sc3 = ax3.scatter(
        df_res["x"], df_res["y"],
        c=df_res["pred_prob"],
        cmap="jet",
        s=60,
        vmin=0.0,
        vmax=1.0,
        edgecolors="grey",
        linewidths=0.3,
    )
    ax3.set_title(f"3. AI Primary Malignancy Map P(Tumor)\nSlide Verdict: {slide_verdict} (Burden: {tumor_burden_pct}%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("WSI X Coordinate (pixels)", fontsize=10)
    ax3.set_ylabel("WSI Y Coordinate (pixels)", fontsize=10)
    ax3.invert_yaxis()
    cbar3 = plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label("Malignancy Probability", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # Panel 4: Multi-Color Histopathological Anomaly Map
    ax4 = axes[1, 1]
    ax4.set_facecolor("#ffffff")
    # Base plot: Normal in light grey
    ax4.scatter(df_res.loc[df_res["pred_prob"] < 0.5, "x"], df_res.loc[df_res["pred_prob"] < 0.5, "y"], c="#cbd5e0", s=30, label="Benign Baseline", alpha=0.6)

    legend_patches = [mpatches.Patch(color="#cbd5e0", label="Benign Baseline")]

    # Plot malignant patches colored by their highest anomaly
    if is_malignant and pred_tumors > 0:
        tum_sub = df_res[df_res["pred_prob"] >= 0.5].copy()
        for idx_t, row_t in tum_sub.iterrows():
            a_scores = {
                "density": (row_t["density"], "#bcbd22", "Cell Density Clusters"),
                "ductal": (row_t["ductal"], "#9467bd", "Ductal Distortion"),
                "necrosis": (row_t["necrosis"], "#17becf", "Tumor Necrosis"),
                "atypia": (row_t["atypia"], "#ff7f0e", "Nuclear Pleomorphism"),
            }
            top_key = max(a_scores, key=lambda k: a_scores[k][0])
            score_val, col, anom_lbl = a_scores[top_key]

            ax4.scatter(row_t["x"], row_t["y"], c=col, s=75, edgecolors="black", linewidths=0.6)

        legend_patches.extend([
            mpatches.Patch(color="#ff7f0e", label="Nuclear Pleomorphism"),
            mpatches.Patch(color="#9467bd", label="Ductal Structure Distortion"),
            mpatches.Patch(color="#bcbd22", label="Cell Density Clusters"),
            mpatches.Patch(color="#17becf", label="Tumor Necrosis / Debris"),
        ])

    ax4.set_title(f"4. Multi-Color Histopathological Anomaly Map\nDominant Anomaly: {dominant_anomaly}", fontsize=11, fontweight="bold")
    ax4.set_xlabel("WSI X Coordinate (pixels)", fontsize=10)
    ax4.set_ylabel("WSI Y Coordinate (pixels)", fontsize=10)
    ax4.invert_yaxis()
    ax4.legend(handles=legend_patches, loc="upper right", fontsize=8.5, framealpha=0.9)
    ax4.grid(True, linestyle="--", alpha=0.4)

    status_color = "#d62728" if is_malignant else "#2ca02c"
    plt.suptitle(f"Whole Slide Biopsy Clinical Spatial Diagnostic Dashboard — {wsi_id.upper()}\nFinal Diagnosis: {slide_verdict} (Tumor Burden: {tumor_burden_pct}%)",
                 fontsize=14, fontweight="bold", y=0.98, color=status_color)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = output_dir / f"{wsi_id}_spatial_diagnostic_dashboard.png"
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"[SAVED] Spatial diagnostic dashboard saved: {save_path}")

    return {
        "wsi_id": wsi_id,
        "total_patches": total_patches,
        "predicted_tumor_patches": pred_tumors,
        "ground_truth_tumor_patches": gt_tumors,
        "tumor_burden_pct": tumor_burden_pct,
        "slide_verdict": slide_verdict,
        "dominant_anomaly": dominant_anomaly,
        "dashboard_image": str(save_path),
    }


def main():
    manifest_csv = PROJECT_ROOT / "data" / "PATCHES" / "TEST" / "test_patch_manifest.csv"
    output_dir = PROJECT_ROOT / "result" / "wsi_heatmaps"

    cls_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt"
    anom_ckpt = PROJECT_ROOT / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_best.pt"

    pipeline = LayeredDiagnosticPipeline(cls_ckpt, anom_ckpt)

    for sid in ["normal_004", "tumor_058"]:
        render_clinical_wsi_dashboard(sid, manifest_csv, pipeline, output_dir)

    print("\n[COMPLETE] Clean spatial dashboards generated for normal_004 and tumor_058.")


if __name__ == "__main__":
    main()
