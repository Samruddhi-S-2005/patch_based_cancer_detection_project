"""
Live Whole Slide Image (WSI) Ingestion, Tiling, Batched Multi-Layer Inference,
and Spatial Heatmap / Anomaly Map Reconstruction Engine.

Supports:
1. Uploading full WSI / Biopsy Tissue Scans (.tif, .tiff, .svs, .ndpi, .png, .jpg, .jpeg)
2. Uploading a Batch of WSI Patch Files (multiple .png/.jpg patches)
3. Fast vectorized PyTorch batch inference across Layer 1 (Quality), Layer 2 (Malignancy),
   Layer 3 (4-Way Anomaly Profiling), and on-demand Layer 4 (Grad-CAM).
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

# Disable DecompressionBombError for large gigapixel WSI .tif files
Image.MAX_IMAGE_PIXELS = None

from src.objective_2.layered_pipeline import LayeredDiagnosticPipeline

__all__ = [
    "ANOMALY_META",
    "blend_wsi_heatmap",
    "compute_multimodel_wsi_heatmaps",
    "create_sample_stitched_wsi_scans",
    "export_qupath_geojson",
    "export_qupath_tsv",
    "find_tcga_svs_slides",
    "load_wsi_image_from_upload",
    "process_openslide_wsi_file",
    "process_wsi_canvas",
    "reconstruct_wsi_probability_field",
    "stitch_uploaded_patch_batch",
]

# Standard hex & RGB colors for the 4 histopathological anomalies
ANOMALY_META = {
    "pleomorphism_atypia": {
        "label": "Nuclear Pleomorphism [Quarantined]",
        "short": "Nuclear Pleomorphism (Quarantined)",
        "hex": "#94a3b8",
        "rgb": (148, 163, 184),
        "threshold": 0.0500,
        "is_experimental": False,
        "is_quarantined": True,
    },
    "ductal_distortion": {
        "label": "Ductal Structure Distortion (BACH)",
        "short": "Ductal Distortion",
        "hex": "#8b5cf6",
        "rgb": (139, 92, 246),
        "threshold": 0.4038,
        "is_experimental": False,
    },
    "density_clusters": {
        "label": "Tumor Cell Density Clusters (NuCLS)",
        "short": "Cell Density Clusters",
        "hex": "#eab308",
        "rgb": (234, 179, 8),
        "threshold": 0.5315,
        "is_experimental": False,
    },
    "tumor_necrosis": {
        "label": "Tumor Necrosis & Debris (BCSS)",
        "short": "Tumor Necrosis",
        "hex": "#06b6d4",
        "rgb": (6, 182, 212),
        "threshold": 0.2715,
        "is_experimental": False,
    },
}


def load_wsi_image_from_upload(file_bytes: bytes, filename: str, max_canvas_dim: int = 2240) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Loads an uploaded WSI (.tif, .tiff, .svs, .ndpi, .png, .jpg) into an RGB numpy array
    suitable for 224x224 spatial tiling, while preserving original dimensions metadata.
    """
    suffix = Path(filename).suffix.lower()
    orig_dims = (0, 0)

    # Try standard PIL first (handles PNG, JPG, standard TIFF)
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        orig_w, orig_h = pil_img.size
        orig_dims = (orig_w, orig_h)

        # If image is extremely large (e.g. > max_canvas_dim), downsample cleanly to max_canvas_dim
        if max(orig_w, orig_h) > max_canvas_dim:
            scale = max_canvas_dim / float(max(orig_w, orig_h))
            new_w = max(224, (int(orig_w * scale) // 224) * 224)
            new_h = max(224, (int(orig_h * scale) // 224) * 224)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        elif min(orig_w, orig_h) < 672:
            # If user uploaded a small thumbnail or single region, scale to a clean 6x6 or 8x6 grid (at least 1344px)
            scale = 1344.0 / float(min(orig_w, orig_h))
            new_w = max(896, min(2240, (int(orig_w * scale) // 224) * 224))
            new_h = max(896, min(2240, (int(orig_h * scale) // 224) * 224))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        else:
            # Snap dimensions to clean multiples of 224
            new_w = max(224, (orig_w // 224) * 224)
            new_h = max(224, (orig_h // 224) * 224)
            if (new_w, new_h) != (orig_w, orig_h):
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return np.array(pil_img), orig_dims
    except Exception:
        pass

    # Fallback for pyramidal .svs / .ndpi / bigtiff via OpenSlide or tifffile
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        try:
            import openslide
            slide = openslide.OpenSlide(str(tmp_path))
            orig_w, orig_h = slide.dimensions
            orig_dims = (orig_w, orig_h)
            scale = max_canvas_dim / float(max(orig_w, orig_h))
            thumb_w = max(672, (int(orig_w * scale) // 224) * 224)
            thumb_h = max(672, (int(orig_h * scale) // 224) * 224)
            thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
            thumb = thumb.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            slide.close()
            return np.array(thumb), orig_dims
        except Exception:
            import tifffile
            arr = tifffile.imread(str(tmp_path))
            if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            orig_h, orig_w = arr.shape[:2]
            orig_dims = (orig_w, orig_h)
            scale = min(1.0, max_canvas_dim / float(max(orig_w, orig_h)))
            new_w = max(672, (int(orig_w * scale) // 224) * 224)
            new_h = max(672, (int(orig_h * scale) // 224) * 224)
            resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized, orig_dims
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def process_wsi_canvas(
    wsi_rgb: np.ndarray,
    pipeline: LayeredDiagnosticPipeline,
    orig_dims: Optional[Tuple[int, int]] = None,
    patch_size: int = 224,
    batch_size: int = 32,
    progress_callback=None,
) -> Dict:
    """
    Tiles a WSI RGB canvas into 224x224 patches, runs Layer 1 quality screening,
    executes batched Layer 2 (Malignancy) & Layer 3 (4-Anomaly) inference on viable tissue tiles,
    and reconstructs spatial heatmaps and anomaly maps.
    """
    h, w = wsi_rgb.shape[:2]
    n_rows = h // patch_size
    n_cols = w // patch_size
    if orig_dims is None:
        orig_dims = (w, h)

    scale_x = orig_dims[0] / float(w)
    scale_y = orig_dims[1] / float(h)

    tiles_meta = []
    tile_crops: Dict[str, np.ndarray] = {}
    valid_indices = []
    valid_tensors = []

    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * patch_size, (r + 1) * patch_size
            x0, x1 = c * patch_size, (c + 1) * patch_size
            crop = wsi_rgb[y0:y1, x0:x1]
            tile_id = f"R{r:02d}_C{c:02d}"

            l1 = pipeline.assess_layer_1_quality(crop)
            # Consider tile as tissue if tissue_fraction >= 15%
            is_tissue = l1["tissue_fraction"] >= 15.0

            tile_info = {
                "tile_id": tile_id,
                "row": r,
                "col": c,
                "canvas_x": x0,
                "canvas_y": y0,
                "wsi_x": int(x0 * scale_x),
                "wsi_y": int(y0 * scale_y),
                "is_tissue": is_tissue,
                "qc_status": l1["status"] if is_tissue else "BACKGROUND_GLASS",
                "blur_score": l1["blur_score"],
                "tissue_fraction": l1["tissue_fraction"],
                "pred_prob": 0.0,
                "pred_label": 0,
                "verdict": "Background" if not is_tissue else "Normal",
                "atypia": None,  # Quarantined / excluded from active inference
                "ductal": 0.0,
                "density": 0.0,
                "necrosis": 0.0,
                "dominant_anomaly": "None (Benign)",
                "dominant_anomaly_key": "none",
                "risk_tier": "Background" if not is_tissue else "Low Risk",
            }
            tiles_meta.append(tile_info)

            if is_tissue:
                tile_crops[tile_id] = crop
                pil_crop = Image.fromarray(crop)
                tensor_crop = pipeline.eval_transform(pil_crop)
                valid_tensors.append(tensor_crop)
                valid_indices.append(idx)

            idx += 1

    # Batched PyTorch Inference for Layer 2 & Layer 3
    total_valid = len(valid_tensors)
    if total_valid > 0:
        for b_start in range(0, total_valid, batch_size):
            b_end = min(total_valid, b_start + batch_size)
            batch_tensor = torch.stack(valid_tensors[b_start:b_end]).to(pipeline.device)

            with torch.no_grad():
                cls_logits = pipeline.classifier(batch_tensor).view(-1)
                cls_probs = torch.sigmoid(cls_logits).cpu().numpy()

                anom_logits = pipeline.anomaly_model.forward_all_heads(batch_tensor)
                p_ductal = torch.sigmoid(anom_logits["ductal"].view(-1)).cpu().numpy()
                p_density = torch.sigmoid(anom_logits["density"].view(-1)).cpu().numpy()
                p_necrosis = torch.sigmoid(anom_logits["necrosis"].view(-1)).cpu().numpy()
                has_atypia = ("atypia" in anom_logits) and not ANOMALY_META.get("pleomorphism_atypia", {}).get("is_quarantined", False)
                p_atypia = torch.sigmoid(anom_logits["atypia"].view(-1)).cpu().numpy() if has_atypia else None

            t_atypia = getattr(pipeline, "anomaly_thresholds", {}).get("atypia", 0.0500)
            t_ductal = getattr(pipeline, "anomaly_thresholds", {}).get("ductal", 0.4038)
            t_density = getattr(pipeline, "anomaly_thresholds", {}).get("density", 0.5315)
            t_necrosis = getattr(pipeline, "anomaly_thresholds", {}).get("necrosis", 0.2715)

            for i_local, global_tile_idx in enumerate(valid_indices[b_start:b_end]):
                prob_tum = float(cls_probs[i_local])
                is_mal = prob_tum >= 0.50

                d_val = float(p_ductal[i_local])
                c_val = float(p_density[i_local])
                n_val = float(p_necrosis[i_local])
                a_val = float(p_atypia[i_local]) if (has_atypia and p_atypia is not None) else None

                # Active anomaly scores strictly exclude quarantined atypia
                anom_scores = {
                    "ductal_distortion": d_val,
                    "density_clusters": c_val,
                    "tumor_necrosis": n_val,
                }
                if has_atypia and a_val is not None:
                    anom_scores["pleomorphism_atypia"] = a_val

                # Permitted anomaly count strictly uses active heads (excludes quarantined atypia)
                active_cnt = sum([
                    d_val >= t_ductal,
                    c_val >= t_density,
                    n_val >= t_necrosis,
                ])
                if has_atypia and a_val is not None:
                    active_cnt += int(a_val >= t_atypia)

                if is_mal:
                    top_k = max(anom_scores, key=anom_scores.get)
                    dom_label = ANOMALY_META[top_k]["short"]
                    risk = "High Suspicion (Multi-Phenotype)" if (active_cnt >= 2 or n_val >= 0.60) else "High Suspicion (Focal Malignancy)"
                else:
                    if active_cnt > 0:
                        top_k = max(anom_scores, key=anom_scores.get)
                        dom_label = f"Isolated {ANOMALY_META[top_k]['short']}"
                        risk = "Intermediate / Equivocal"
                    else:
                        top_k = "none"
                        dom_label = "Morphologically Benign"
                        risk = "Low Suspicion (Benign)" if prob_tum < 0.30 else "Intermediate / Equivocal"

                t_entry = tiles_meta[global_tile_idx]
                t_entry["pred_prob"] = round(prob_tum, 4)
                t_entry["pred_label"] = 1 if is_mal else 0
                t_entry["verdict"] = "Malignant" if is_mal else "Normal"
                t_entry["atypia"] = round(a_val, 4) if (has_atypia and a_val is not None) else None
                t_entry["ductal"] = round(d_val, 4)
                t_entry["density"] = round(c_val, 4)
                t_entry["necrosis"] = round(n_val, 4)
                t_entry["dominant_anomaly"] = dom_label
                t_entry["dominant_anomaly_key"] = top_k
                t_entry["risk_tier"] = risk

            if progress_callback is not None:
                progress_callback(min(1.0, b_end / float(total_valid)))

    df_tiles = pd.DataFrame(tiles_meta)

    # Reconstruct Visual Overlays on the WSI Canvas
    # 1. Grid Overlay (Green = Benign Tissue, Red = Malignant Patch, Faint Gray = Glass)
    grid_overlay = wsi_rgb.copy()
    # 2. Low-res probability grid (n_rows x n_cols) upsampled smoothly to (h, w)
    prob_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    tissue_mask = np.zeros((h, w), dtype=np.float32)
    # 3. Anomaly Phenotype Canvas
    anomaly_overlay = wsi_rgb.copy()

    for _, row in df_tiles.iterrows():
        r, c = int(row["row"]), int(row["col"])
        y0, y1 = r * patch_size, (r + 1) * patch_size
        x0, x1 = c * patch_size, (c + 1) * patch_size

        if not row["is_tissue"]:
            cv2.rectangle(grid_overlay, (x0, y0), (x1 - 1, y1 - 1), (203, 213, 225), 1)
            continue

        prob_grid[r, c] = float(row["pred_prob"])
        tissue_mask[y0:y1, x0:x1] = 1.0

        if row["pred_label"] == 1:
            # Malignant border in crimson red
            cv2.rectangle(grid_overlay, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), (220, 38, 38), 4)
            # Color anomaly overlay by dominant anomaly
            dom_k = row["dominant_anomaly_key"]
            if dom_k in ANOMALY_META:
                col_rgb = ANOMALY_META[dom_k]["rgb"]
                patch_block = anomaly_overlay[y0:y1, x0:x1].astype(np.float32)
                color_block = np.full_like(patch_block, col_rgb, dtype=np.float32)
                anomaly_overlay[y0:y1, x0:x1] = np.clip(0.52 * patch_block + 0.48 * color_block, 0, 255).astype(np.uint8)
                cv2.rectangle(anomaly_overlay, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), col_rgb, 3)
        else:
            # Benign tissue border in emerald green
            cv2.rectangle(grid_overlay, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), (34, 197, 94), 2)

    # Smooth continuous heatmap generation
    prob_upsampled = cv2.resize(prob_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    prob_upsampled = cv2.GaussianBlur(prob_upsampled, (0, 0), sigmaX=patch_size * 0.35)
    prob_upsampled = np.clip(prob_upsampled, 0.0, 1.0)

    # Summary statistics
    df_tissue = df_tiles[df_tiles["is_tissue"] == True]
    n_tissue = len(df_tissue)
    n_mal = int((df_tissue["pred_label"] == 1).sum()) if n_tissue > 0 else 0
    burden_pct = round((n_mal / n_tissue) * 100.0, 2) if n_tissue > 0 else 0.0
    is_slide_malignant = (n_mal >= 2 and burden_pct >= 5.0) or (n_mal >= 5)
    slide_verdict = "MALIGNANT METASTASIS" if is_slide_malignant else ("SUSPICIOUS FOCUS" if n_mal > 0 else "BENIGN / NORMAL")

    if n_mal > 0:
        df_mal = df_tissue[df_tissue["pred_label"] == 1]
        mean_anoms = {
            "Ductal Structure Distortion": float(df_mal["ductal"].mean()),
            "Tumor Cell Density Clusters": float(df_mal["density"].mean()),
            "Tumor Necrosis & Debris": float(df_mal["necrosis"].mean()),
        }
        dominant_slide_anomaly = max(mean_anoms, key=mean_anoms.get)
    else:
        mean_anoms = {
            "Ductal Structure Distortion": 0.0,
            "Tumor Cell Density Clusters": 0.0,
            "Tumor Necrosis & Debris": 0.0,
        }
        dominant_slide_anomaly = "None (Benign Baseline)"

    pct_exceeding = {
        "Ductal Structure Distortion": round(float((df_tissue["ductal"] >= t_ductal).sum() / n_tissue * 100), 2) if n_tissue > 0 else 0.0,
        "Tumor Cell Density Clusters": round(float((df_tissue["density"] >= t_density).sum() / n_tissue * 100), 2) if n_tissue > 0 else 0.0,
        "Tumor Necrosis & Debris": round(float((df_tissue["necrosis"] >= t_necrosis).sum() / n_tissue * 100), 2) if n_tissue > 0 else 0.0,
    }

    return {
        "wsi_rgb": wsi_rgb,
        "grid_overlay": grid_overlay,
        "prob_upsampled": prob_upsampled,
        "tissue_mask": tissue_mask,
        "anomaly_overlay": anomaly_overlay,
        "df_tiles": df_tiles,
        "tile_crops": tile_crops,
        "summary": {
            "orig_dimensions": f"{orig_dims[0]} × {orig_dims[1]} px",
            "grid_dimensions": f"{n_cols} × {n_rows} tiles ({len(df_tiles)} total)",
            "total_tiles": len(df_tiles),
            "tissue_tiles": n_tissue,
            "malignant_tiles": n_mal,
            "benign_tiles": n_tissue - n_mal,
            "tumor_burden_pct": burden_pct,
            "burden_formula": f"{n_mal} malignant / {n_tissue} viable tissue tiles",
            "denominator_definition": "Viable tissue tiles (excluding non-tissue glass)",
            "decision_threshold": 0.5000,
            "slide_verdict": slide_verdict,
            "is_malignant": is_slide_malignant or (n_mal > 0),
            "dominant_anomaly": dominant_slide_anomaly,
            "mean_anomalies": mean_anoms,
            "pct_exceeding_threshold": pct_exceeding,
            "atypia_status": "QUARANTINED (Excluded from active inference and clinical risk)",
        },
    }


def blend_wsi_heatmap(
    wsi_rgb: np.ndarray,
    prob_upsampled: np.ndarray,
    tissue_mask: np.ndarray,
    alpha: float = 0.50,
    colormap_name: str = "jet",
    blend_mode: str = "hotspot",
) -> np.ndarray:
    """
    Blends the spatial malignancy probability heatmap onto the WSI RGB canvas.

    Modes:
      - 'hotspot': (Default clinical pathology view) The opacity is weighted by
        malignancy probability so normal/benign tissue retains its natural H&E
        morphology without being obscured by dark blue tint, while suspicious
        and malignant foci glow vividly in warm amber/orange and crimson red.
      - 'uniform': Applies the colormap uniformly across all tissue regions with
        constant opacity alpha (classic full-field continuous colormap).
      - 'pure_field': Returns pure continuous colormap field on clean white canvas.
    """
    target_h, target_w = wsi_rgb.shape[:2]
    if prob_upsampled.shape[:2] != (target_h, target_w):
        prob_upsampled = cv2.resize(prob_upsampled.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if tissue_mask.shape[:2] != (target_h, target_w):
        tissue_mask = cv2.resize(tissue_mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    cmap = matplotlib.colormaps[colormap_name]
    heat_rgb = (cmap(prob_upsampled)[:, :, :3] * 255.0).astype(np.float32)
    base_rgb = wsi_rgb.astype(np.float32)
    t_mask = tissue_mask[:, :, None]

    if blend_mode == "pure_field":
        pure = np.clip(heat_rgb * t_mask + 255.0 * (1.0 - t_mask), 0, 255).astype(np.uint8)
        return pure

    if blend_mode == "hotspot":
        # Smoothly ramp opacity from 0.0 on benign baseline to full alpha on tumor hotspots
        # P < 0.15: alpha = 0.0 (natural optical H&E)
        # P in [0.15, 0.55]: soft transition through cool/amber
        # P >= 0.55: full alpha on malignant carcinoma foci
        prob_weight = np.clip((prob_upsampled - 0.15) / 0.40, 0.0, 1.0)[:, :, None]
        effective_alpha = alpha * t_mask * prob_weight
    else:
        effective_alpha = alpha * t_mask

    blended = base_rgb * (1.0 - effective_alpha) + heat_rgb * effective_alpha
    return np.clip(blended, 0, 255).astype(np.uint8)



def reconstruct_wsi_probability_field(
    df_tissue: pd.DataFrame,
    probs: np.ndarray,
    canvas_shape: Tuple[int, int],
    tissue_mask: np.ndarray,
    patch_size: int = 224,
    scale: Optional[float] = None,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstructs a smooth, continuous spatial malignancy probability field
    across viable tissue regions from discrete tile inferences.
    Uses 2D Delaunay spatial interpolation with nearest-neighbor convex hull
    completion and cubic Gaussian smoothing, strictly clipped to tissue boundaries.
    Eliminates disconnected polka-dot artifacts completely.
    """
    from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

    # Enforce strict shape alignment to tissue_mask (ground-truth canvas boundary)
    if tissue_mask is not None and hasattr(tissue_mask, "shape") and len(tissue_mask.shape) >= 2:
        thumb_h, thumb_w = tissue_mask.shape[:2]
    else:
        thumb_h, thumb_w = canvas_shape[:2]

    n_tiles = len(df_tissue)
    if n_tiles == 0:
        return np.zeros((thumb_h, thumb_w), dtype=np.float32)

    eff_scale = float(scale) if (scale is not None and scale > 0) else 1.0
    tw = max(4, int(patch_size * eff_scale))
    th = max(4, int(patch_size * eff_scale))

    # Fast 2D Delaunay spatial interpolation when coordinate metadata is available
    if n_tiles >= 3 and "canvas_x" in df_tissue.columns and "canvas_y" in df_tissue.columns:
        pts = df_tissue[["canvas_x", "canvas_y"]].to_numpy().astype(np.float32) + np.array([tw / 2.0, th / 2.0], dtype=np.float32)
        vals = np.array(probs, dtype=np.float32)

        dw, dh = max(32, thumb_w // 4), max(32, thumb_h // 4)
        grid_x, grid_y = np.meshgrid(np.linspace(0, thumb_w, dw), np.linspace(0, thumb_h, dh))
        eval_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

        try:
            lin_interp = LinearNDInterpolator(pts, vals)
            near_interp = NearestNDInterpolator(pts, vals)
            lin_vals = lin_interp(eval_points)
            nan_mask = np.isnan(lin_vals)
            if np.any(nan_mask):
                lin_vals[nan_mask] = near_interp(eval_points[nan_mask])

            grid_2d = lin_vals.reshape((dh, dw))
            smooth_field = cv2.resize(grid_2d.astype(np.float32), (thumb_w, thumb_h), interpolation=cv2.INTER_CUBIC)
            smooth_field = cv2.GaussianBlur(smooth_field, (0, 0), sigmaX=max(6.0, float(tw * 2.0)))
            if tissue_mask is not None and smooth_field.shape[:2] != tissue_mask.shape[:2]:
                smooth_field = cv2.resize(smooth_field.astype(np.float32), (tissue_mask.shape[1], tissue_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            return np.clip(smooth_field * tissue_mask, 0.0, 1.0)
        except Exception:
            pass

    # Adaptive density accumulator for small N or canvas grids
    accum_prob = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    accum_weight = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    footprint = max(tw * 3, 24)
    half_fp = footprint // 2

    for i, (_, row) in enumerate(df_tissue.iterrows()):
        cx = int(row.get("canvas_x", row.get("col", 0) * patch_size) + tw // 2)
        cy = int(row.get("canvas_y", row.get("row", 0) * patch_size) + th // 2)
        p = float(probs[i])
        x1, x2 = max(0, cx - half_fp), min(thumb_w, cx + half_fp + 1)
        y1, y2 = max(0, cy - half_fp), min(thumb_h, cy + half_fp + 1)
        accum_prob[y1:y2, x1:x2] += p
        accum_weight[y1:y2, x1:x2] += 1.0

    mask_sampled = accum_weight > 0
    prob_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    prob_map[mask_sampled] = accum_prob[mask_sampled] / accum_weight[mask_sampled]
    blurred = cv2.GaussianBlur(prob_map, (0, 0), sigmaX=max(5.0, float(tw * 1.5)))
    if tissue_mask is not None and blurred.shape[:2] != tissue_mask.shape[:2]:
        blurred = cv2.resize(blurred.astype(np.float32), (tissue_mask.shape[1], tissue_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(blurred * tissue_mask, 0.0, 1.0)


def compute_multimodel_wsi_heatmaps(
    wsi_result: Dict,
    pipeline: LayeredDiagnosticPipeline,
    colormap_name: str = "jet",
    alpha: float = 0.50,
    blend_mode: str = "hotspot",
) -> Dict[str, Dict]:
    """
    Computes comparable whole-slide spatial malignancy score heatmaps across all four architectures:
    - EfficientNet-B0 (Primary)
    - ResNet-50
    - DenseNet-121
    - VGG-16

    Evaluates the exact same accepted tissue tiles on identical coordinates without mutating
    the pipeline's active primary classifier state.
    """
    from src.objective_2.layered_pipeline import resolve_model_checkpoint

    wsi_rgb = wsi_result["wsi_rgb"]
    tissue_mask = wsi_result["tissue_mask"]
    df_tiles = wsi_result["df_tiles"]
    tile_crops = wsi_result["tile_crops"]
    patch_size = 224
    h, w = wsi_rgb.shape[:2]
    n_rows = h // patch_size
    n_cols = w // patch_size

    # Filter to viable tissue tiles only
    df_tissue = df_tiles[df_tiles["is_tissue"] == True].copy()
    tissue_count = len(df_tissue)

    architectures = ["EfficientNet-B0", "ResNet-50", "DenseNet-121", "VGG-16"]
    multimodel_maps = {}

    # Prepare batch of valid tissue crops
    valid_tile_ids = [row["tile_id"] for _, row in df_tissue.iterrows() if row["tile_id"] in tile_crops]
    if not valid_tile_ids:
        for arch in architectures:
            multimodel_maps[arch] = {
                "status": "NO_TISSUE_TILES",
                "blended_heatmap": wsi_rgb,
                "prob_upsampled": np.zeros((h, w), dtype=np.float32),
                "malignant_tiles": 0,
                "tissue_tiles": 0,
                "tumor_burden_pct": 0.0,
                "slide_verdict": "BENIGN / NORMAL",
                "checkpoint": "N/A",
            }
        return multimodel_maps

    # Stack valid tensor crops
    tensor_list = []
    for tid in valid_tile_ids:
        crop = tile_crops[tid]
        if crop.shape[:2] != (patch_size, patch_size):
            crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        tensor_list.append(pipeline.eval_transform(Image.fromarray(crop)))
    stacked_tensors = torch.stack(tensor_list).to(pipeline.device)

    active_arch = getattr(pipeline, "model_name", "EfficientNet-B0")

    for arch in architectures:
        ckpt_path = resolve_model_checkpoint(arch)
        if not ckpt_path.exists():
            multimodel_maps[arch] = {
                "status": "UNAVAILABLE",
                "error": f"Checkpoint missing: {ckpt_path.name}",
                "blended_heatmap": wsi_rgb,
                "prob_upsampled": np.zeros((h, w), dtype=np.float32),
                "malignant_tiles": 0,
                "tissue_tiles": tissue_count,
                "tumor_burden_pct": 0.0,
                "slide_verdict": "UNAVAILABLE",
                "checkpoint": ckpt_path.name,
            }
            continue

        try:
            # Ensure caching container for model predictions exists in wsi_result
            if "_multimodel_probs" not in wsi_result:
                wsi_result["_multimodel_probs"] = {}

            # If this is the active model that already ran on the full canvas, reuse its exact results
            if arch == active_arch and "prob_upsampled" in wsi_result:
                prob_up = wsi_result["prob_upsampled"]
                blended = blend_wsi_heatmap(wsi_rgb, prob_up, tissue_mask, alpha=alpha, colormap_name=colormap_name, blend_mode=blend_mode)
                s = wsi_result["summary"]
                multimodel_maps[arch] = {
                    "status": "AVAILABLE",
                    "blended_heatmap": blended,
                    "prob_upsampled": prob_up,
                    "malignant_tiles": int(s.get("malignant_tiles", 0)),
                    "tissue_tiles": int(s.get("tissue_tiles", tissue_count)),
                    "tumor_burden_pct": float(s.get("tumor_burden_pct", 0.0)),
                    "slide_verdict": s.get("slide_verdict", "BENIGN / NORMAL"),
                    "checkpoint": ckpt_path.name,
                }
            else:
                # Check if this model's raw probabilities were already computed for this slide
                if arch in wsi_result["_multimodel_probs"]:
                    all_probs = wsi_result["_multimodel_probs"][arch]
                else:
                    # Load the model via pipeline's internal comparison cache
                    model, _ = pipeline._get_comparison_model_and_cam(arch)
                    # Batched forward pass in chunks of 32 using inference_mode for maximum speed
                    probs_chunks = []
                    chunk_sz = 32
                    for c_start in range(0, len(stacked_tensors), chunk_sz):
                        c_end = min(len(stacked_tensors), c_start + chunk_sz)
                        sub_batch = stacked_tensors[c_start:c_end]
                        with torch.inference_mode():
                            logits = model(sub_batch).view(-1)
                            probs = torch.sigmoid(logits).cpu().numpy()
                            probs_chunks.append(probs)

                    all_probs = np.concatenate(probs_chunks, axis=0)
                    wsi_result["_multimodel_probs"][arch] = all_probs

                is_openslide = ("canvas_x" in df_tissue.columns and "canvas_y" in df_tissue.columns)
                scale = wsi_result.get("scale", None)
                stride = wsi_result.get("stride", 224)

                if is_openslide:
                    df_ordered = df_tissue[df_tissue["tile_id"].isin(valid_tile_ids)].set_index("tile_id").loc[valid_tile_ids].reset_index()
                    prob_up_arch = reconstruct_wsi_probability_field(
                        df_tissue=df_ordered,
                        probs=all_probs,
                        canvas_shape=(h, w),
                        tissue_mask=tissue_mask,
                        patch_size=patch_size,
                        scale=scale,
                        stride=stride,
                    )
                else:
                    max_r = int(df_tissue["row"].max()) if len(df_tissue) > 0 else 0
                    max_c = int(df_tissue["col"].max()) if len(df_tissue) > 0 else 0
                    grid_rows = max(n_rows, max_r + 1)
                    grid_cols = max(n_cols, max_c + 1)
                    prob_grid_arch = np.zeros((grid_rows, grid_cols), dtype=np.float32)
                    for i_tile, tid in enumerate(valid_tile_ids):
                        t_row = df_tissue[df_tissue["tile_id"] == tid].iloc[0]
                        r, c = int(t_row["row"]), int(t_row["col"])
                        if 0 <= r < grid_rows and 0 <= c < grid_cols:
                            prob_grid_arch[r, c] = float(all_probs[i_tile])

                    prob_up_arch = cv2.resize(prob_grid_arch, (w, h), interpolation=cv2.INTER_CUBIC)
                    prob_up_arch = cv2.GaussianBlur(prob_up_arch, (0, 0), sigmaX=patch_size * 0.35)
                    prob_up_arch = np.clip(prob_up_arch * tissue_mask, 0.0, 1.0)

                blended_arch = blend_wsi_heatmap(wsi_rgb, prob_up_arch, tissue_mask, alpha=alpha, colormap_name=colormap_name, blend_mode=blend_mode)

                n_mal_arch = int((all_probs >= 0.50).sum())
                n_tiss_arch = len(all_probs)
                burden_arch = round((n_mal_arch / n_tiss_arch) * 100.0, 2) if n_tiss_arch > 0 else 0.0
                is_mal_slide = (n_mal_arch >= 2 and burden_arch >= 5.0) or (n_mal_arch >= 5)
                verdict_arch = "MALIGNANT METASTASIS" if is_mal_slide else ("SUSPICIOUS FOCUS" if n_mal_arch > 0 else "BENIGN / NORMAL")

                multimodel_maps[arch] = {
                    "status": "AVAILABLE",
                    "blended_heatmap": blended_arch,
                    "prob_upsampled": prob_up_arch,
                    "malignant_tiles": n_mal_arch,
                    "tissue_tiles": n_tiss_arch,
                    "tumor_burden_pct": burden_arch,
                    "slide_verdict": verdict_arch,
                    "checkpoint": ckpt_path.name,
                }
        except Exception as e:
            multimodel_maps[arch] = {
                "status": "ERROR",
                "error": str(e),
                "blended_heatmap": wsi_rgb,
                "prob_upsampled": np.zeros((h, w), dtype=np.float32),
                "malignant_tiles": 0,
                "tissue_tiles": tissue_count,
                "tumor_burden_pct": 0.0,
                "slide_verdict": "ERROR",
                "checkpoint": ckpt_path.name,
            }

    return multimodel_maps


def create_sample_stitched_wsi_scans(output_dir: Path) -> Dict[str, Path]:
    """
    Creates two realistic high-resolution stitched WSI tissue biopsy scans (.tif)
    from CAMELYON16 test patches so the user can test WSI upload & live tiling immediately:
      1. sample_wsi_tumor_058_biopsy.tif (1792 x 1344 px, 8x6 = 48 patches with metastatic focus)
      2. sample_wsi_normal_004_biopsy.tif (1792 x 1344 px, 8x6 = 48 patches of benign lymph node)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tumor_out = output_dir / "sample_wsi_tumor_058_biopsy.tif"
    normal_out = output_dir / "sample_wsi_normal_004_biopsy.tif"

    if tumor_out.exists() and normal_out.exists():
        return {"tumor_wsi": tumor_out, "normal_wsi": normal_out}

    test_dir = PROJECT_ROOT / "data" / "PATCHES" / "TEST"
    norm_patches = sorted((test_dir / "normal").glob("normal_004_*.png"))
    tum_pos_patches = sorted((test_dir / "tumor").glob("tumor_058_*label1.png"))
    tum_neg_patches = sorted((test_dir / "tumor").glob("tumor_058_*label0.png"))

    if not tum_pos_patches:
        tum_pos_patches = sorted((test_dir / "tumor").glob("*label1.png"))
    if not tum_neg_patches:
        tum_neg_patches = norm_patches

    n_cols, n_rows = 8, 6
    ps = 224

    # 1. Build Tumor WSI Biopsy Scan (metastatic core in center-right surrounded by lymphoid stroma)
    canvas_tum = np.full((n_rows * ps, n_cols * ps, 3), 246, dtype=np.uint8)
    pos_idx, neg_idx = 0, 0
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * ps, (r + 1) * ps
            x0, x1 = c * ps, (c + 1) * ps
            # Leave corner tiles as realistic slide glass margin
            if (r in (0, n_rows - 1) and c in (0, n_cols - 1)):
                continue
            # Metastatic tumor core in rows 1..4, cols 3..6
            if 1 <= r <= 4 and 3 <= c <= 6 and pos_idx < len(tum_pos_patches):
                img_p = np.array(Image.open(tum_pos_patches[pos_idx % len(tum_pos_patches)]).convert("RGB"))
                pos_idx += 1
            else:
                img_p = np.array(Image.open(tum_neg_patches[neg_idx % len(tum_neg_patches)]).convert("RGB"))
                neg_idx += 1
            canvas_tum[y0:y1, x0:x1] = img_p

    Image.fromarray(canvas_tum).save(tumor_out, format="TIFF")

    # 2. Build Normal WSI Biopsy Scan (pure benign lymphoid tissue with glass corners)
    canvas_norm = np.full((n_rows * ps, n_cols * ps, 3), 246, dtype=np.uint8)
    n_idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * ps, (r + 1) * ps
            x0, x1 = c * ps, (c + 1) * ps
            if (r in (0, n_rows - 1) and c in (0, n_cols - 1)):
                continue
            if norm_patches:
                img_p = np.array(Image.open(norm_patches[n_idx % len(norm_patches)]).convert("RGB"))
                n_idx += 1
                canvas_norm[y0:y1, x0:x1] = img_p

    Image.fromarray(canvas_norm).save(normal_out, format="TIFF")
    return {"tumor_wsi": tumor_out, "normal_wsi": normal_out}


def stitch_uploaded_patch_batch(uploaded_files: List, patch_size: int = 224) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Takes a list of uploaded patch files (e.g., multiple 224x224 .png/.jpg files),
    arranges them into a clean 2D spatial biopsy grid canvas, and returns (wsi_rgb, orig_dims).
    """
    n_files = len(uploaded_files)
    n_cols = max(2, int(np.ceil(np.sqrt(n_files * 1.33))))
    n_rows = max(2, int(np.ceil(n_files / float(n_cols))))

    canvas = np.full((n_rows * patch_size, n_cols * patch_size, 3), 246, dtype=np.uint8)
    for idx, uf in enumerate(uploaded_files):
        r = idx // n_cols
        c = idx % n_cols
        if r >= n_rows:
            break
        try:
            uf.seek(0)
            pil_p = Image.open(uf).convert("RGB")
            if pil_p.size != (patch_size, patch_size):
                pil_p = pil_p.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
            y0, y1 = r * patch_size, (r + 1) * patch_size
            x0, x1 = c * patch_size, (c + 1) * patch_size
            canvas[y0:y1, x0:x1] = np.array(pil_p)
        except Exception:
            continue

    h, w = canvas.shape[:2]
    return canvas, (w, h)


def process_openslide_wsi_file(
    slide_path: Union[str, Path],
    pipeline: LayeredDiagnosticPipeline,
    stride: int = 224,
    max_tiles: int = 400,
    progress_callback=None,
) -> Dict:
    """
    Direct OpenSlide streaming pipeline:
    Opens whole-slide pyramidal scan, reads metadata (dimensions, mpp, levels),
    generates a low-resolution thumbnail for macro inspection and tissue detection,
    streams Level 0 patches at (x, y) coordinates with specified candidate stride (e.g. 224 or 112),
    filters glass background via Layer 1 QC, runs batched model inference,
    and returns coordinate-preserving spatial map without blowing up memory.
    """
    import openslide

    p_slide = Path(slide_path)
    if not p_slide.exists():
        raise FileNotFoundError(f"Slide file not found at: {p_slide}")

    slide = openslide.OpenSlide(str(p_slide))
    try:
        orig_w, orig_h = slide.dimensions
        level_count = slide.level_count
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25) or 0.25)
        mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0.25) or 0.25)

        # Overview thumbnail (max dim 1792px for clear display)
        max_thumb = 1792
        scale = max_thumb / float(max(orig_w, orig_h))
        thumb_w = max(224, int(orig_w * scale))
        thumb_h = max(224, int(orig_h * scale))
        thumbnail_pil = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
        thumb_np = np.array(thumbnail_pil)

        # Synchronize thumb_h, thumb_w directly with actual OpenSlide thumbnail array dimensions
        thumb_h, thumb_w = thumb_np.shape[:2]
        scale_x = thumb_w / float(orig_w)
        scale_y = thumb_h / float(orig_h)
        scale = (scale_x + scale_y) / 2.0

        # Candidate tissue detection on thumbnail using saturation + intensity
        thumb_hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
        s_channel = thumb_hsv[:, :, 1]
        v_channel = thumb_hsv[:, :, 2]
        tissue_mask_thumb = ((s_channel > 20) & (v_channel < 235)).astype(np.uint8)

        import concurrent.futures

        # Configure CPU inference threads for maximum performance
        if not torch.cuda.is_available():
            torch.set_num_threads(min(8, os.cpu_count() or 4))

        # Generate candidate (x, y) coordinates at Level 0
        patch_size = 224
        x_coords = np.arange(0, orig_w - patch_size, stride)
        y_coords = np.arange(0, orig_h - patch_size, stride)

        # Fast vectorized candidate search via area downsampling on thumbnail mask (0.005s)
        grid_mask = cv2.resize(tissue_mask_thumb.astype(np.float32), (len(x_coords), len(y_coords)), interpolation=cv2.INTER_AREA)
        candidate_indices = np.argwhere(grid_mask >= 0.15)
        candidates = [(int(x_coords[c]), int(y_coords[r])) for r, c in candidate_indices]

        if len(candidates) > max_tiles:
            # Deterministic uniform subsampling across slide
            step = len(candidates) / float(max_tiles)
            candidates = [candidates[int(i * step)] for i in range(max_tiles)]

        # Read patches in parallel via ThreadPoolExecutor for fast I/O throughput
        def _fetch_patch_rgb(coord):
            x0, y0 = coord
            rgba = slide.read_region((x0, y0), 0, (patch_size, patch_size))
            return np.array(rgba.convert("RGB"))

        n_workers = min(6, os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            raw_patches = list(executor.map(_fetch_patch_rgb, candidates))

        tiles_meta = []
        tile_crops = {}
        valid_tensors = []
        batch_size = 32

        for idx, (x0, y0) in enumerate(candidates):
            tile_id = f"T_{idx:03d}_X{x0}_Y{y0}"
            patch_rgb = raw_patches[idx]

            l1 = pipeline.assess_layer_1_quality(patch_rgb)
            is_tissue = bool(l1["passed"])

            tile_info = {
                "tile_id": tile_id,
                "row": int(y0 // stride),
                "col": int(x0 // stride),
                "canvas_x": int(x0 * scale_x),
                "canvas_y": int(y0 * scale_y),
                "wsi_x": x0,
                "wsi_y": y0,
                "is_tissue": is_tissue,
                "qc_status": l1["status"] if is_tissue else "REJECTED_LOW_TISSUE",
                "blur_score": l1["blur_score"],
                "tissue_fraction": l1["tissue_fraction"],
                "pred_prob": 0.0,
                "pred_label": 0,
                "verdict": "Background" if not is_tissue else "Normal",
                "atypia": None,  # Quarantined / excluded from active inference
                "ductal": 0.0,
                "density": 0.0,
                "necrosis": 0.0,
                "dominant_anomaly": "None (Benign)",
                "dominant_anomaly_key": "none",
                "risk_tier": "Background" if not is_tissue else "Low Suspicion (Benign)",
            }
            tiles_meta.append(tile_info)

            if is_tissue:
                tile_crops[tile_id] = patch_rgb
                tensor_crop = pipeline.eval_transform(Image.fromarray(patch_rgb))
                valid_tensors.append((idx, tensor_crop))

        # Batched inference using torch.inference_mode() for accelerated forward passes
        total_valid = len(valid_tensors)
        if total_valid > 0:
            for b_start in range(0, total_valid, batch_size):
                b_end = min(total_valid, b_start + batch_size)
                batch_items = valid_tensors[b_start:b_end]
                batch_tensor = torch.stack([item[1] for item in batch_items]).to(pipeline.device)

                with torch.inference_mode():
                    cls_logits = pipeline.classifier(batch_tensor).view(-1)
                    cls_probs = torch.sigmoid(cls_logits).cpu().numpy()

                    anom_logits = pipeline.anomaly_model.forward_all_heads(batch_tensor)
                    p_ductal = torch.sigmoid(anom_logits["ductal"].view(-1)).cpu().numpy()
                    p_density = torch.sigmoid(anom_logits["density"].view(-1)).cpu().numpy()
                    p_necrosis = torch.sigmoid(anom_logits["necrosis"].view(-1)).cpu().numpy()
                    has_atypia = ("atypia" in anom_logits) and not ANOMALY_META.get("pleomorphism_atypia", {}).get("is_quarantined", False)
                    p_atypia = torch.sigmoid(anom_logits["atypia"].view(-1)).cpu().numpy() if has_atypia else None

                t_atypia = getattr(pipeline, "anomaly_thresholds", {}).get("atypia", 0.0500)
                t_ductal = getattr(pipeline, "anomaly_thresholds", {}).get("ductal", 0.4038)
                t_density = getattr(pipeline, "anomaly_thresholds", {}).get("density", 0.5315)
                t_necrosis = getattr(pipeline, "anomaly_thresholds", {}).get("necrosis", 0.2715)

                for i_local, (global_idx, _) in enumerate(batch_items):
                    prob_tum = float(cls_probs[i_local])
                    is_mal = prob_tum >= 0.50

                    d_val = float(p_ductal[i_local])
                    c_val = float(p_density[i_local])
                    n_val = float(p_necrosis[i_local])
                    a_val = float(p_atypia[i_local]) if (has_atypia and p_atypia is not None) else None

                    # Active anomaly scores strictly exclude quarantined atypia
                    anom_scores = {
                        "ductal_distortion": d_val,
                        "density_clusters": c_val,
                        "tumor_necrosis": n_val,
                    }
                    if has_atypia and a_val is not None:
                        anom_scores["pleomorphism_atypia"] = a_val

                    # Permitted anomaly count strictly uses active heads (excludes quarantined atypia)
                    active_cnt = sum([
                        d_val >= t_ductal,
                        c_val >= t_density,
                        n_val >= t_necrosis,
                    ])
                    if has_atypia and a_val is not None:
                        active_cnt += int(a_val >= t_atypia)

                    if is_mal:
                        top_k = max(anom_scores, key=anom_scores.get)
                        dom_label = ANOMALY_META[top_k]["short"]
                        risk = "High Suspicion (Multi-Phenotype)" if (active_cnt >= 2 or n_val >= 0.60) else "High Suspicion (Focal Malignancy)"
                    else:
                        if active_cnt > 0:
                            top_k = max(anom_scores, key=anom_scores.get)
                            dom_label = f"Isolated {ANOMALY_META[top_k]['short']}"
                            risk = "Intermediate / Equivocal"
                        else:
                            top_k = "none"
                            dom_label = "Morphologically Benign"
                            risk = "Low Suspicion (Benign)" if prob_tum < 0.30 else "Intermediate / Equivocal"

                    t_entry = tiles_meta[global_idx]
                    t_entry["pred_prob"] = round(prob_tum, 4)
                    t_entry["pred_label"] = 1 if is_mal else 0
                    t_entry["verdict"] = "Malignant" if is_mal else "Normal"
                    t_entry["atypia"] = round(a_val, 4) if (has_atypia and a_val is not None) else None
                    t_entry["ductal"] = round(d_val, 4)
                    t_entry["density"] = round(c_val, 4)
                    t_entry["necrosis"] = round(n_val, 4)
                    t_entry["dominant_anomaly"] = dom_label
                    t_entry["dominant_anomaly_key"] = top_k
                    t_entry["risk_tier"] = risk

                if progress_callback is not None:
                    progress_callback(min(1.0, b_end / float(total_valid)))

        df_tiles = pd.DataFrame(tiles_meta)
        df_tissue = df_tiles[df_tiles["is_tissue"] == True]
        n_tissue = len(df_tissue)
        n_mal = int((df_tissue["pred_label"] == 1).sum()) if n_tissue > 0 else 0
        burden_pct = round((n_mal / n_tissue) * 100.0, 2) if n_tissue > 0 else 0.0
        is_slide_malignant = (n_mal >= 2 and burden_pct >= 5.0) or (n_mal >= 5)
        slide_verdict = "MALIGNANT METASTASIS" if is_slide_malignant else ("SUSPICIOUS FOCUS" if n_mal > 0 else "BENIGN / NORMAL")

        # Visual overlays on thumbnail
        grid_overlay = thumb_np.copy()
        for _, row in df_tiles.iterrows():
            tx = int(row["canvas_x"])
            ty = int(row["canvas_y"])
            tw = max(4, int(patch_size * scale))
            th = max(4, int(patch_size * scale))
            if row["pred_label"] == 1:
                cv2.rectangle(grid_overlay, (tx, ty), (tx + tw, ty + th), (220, 38, 38), 2)
            elif row["is_tissue"]:
                cv2.rectangle(grid_overlay, (tx, ty), (tx + tw, ty + th), (34, 197, 94), 1)

        # Real tissue mask from thumbnail segmentation
        clean_tissue_mask = cv2.morphologyEx(tissue_mask_thumb, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        tissue_mask = clean_tissue_mask.astype(np.float32)

        # Smooth, continuous probability field across slide
        if len(df_tissue) > 0:
            prob_upsampled = reconstruct_wsi_probability_field(
                df_tissue=df_tissue,
                probs=df_tissue["pred_prob"].to_numpy(),
                canvas_shape=(thumb_h, thumb_w),
                tissue_mask=tissue_mask,
                patch_size=patch_size,
                scale=scale,
                stride=stride,
            )
        else:
            prob_upsampled = np.zeros((thumb_h, thumb_w), dtype=np.float32)

        anomaly_overlay = thumb_np.copy()

        if n_mal > 0:
            df_mal = df_tissue[df_tissue["pred_label"] == 1]
            mean_anoms = {
                "Ductal Structure Distortion": float(df_mal["ductal"].mean()),
                "Tumor Cell Density Clusters": float(df_mal["density"].mean()),
                "Tumor Necrosis & Debris": float(df_mal["necrosis"].mean()),
            }
            dominant_slide_anomaly = max(mean_anoms, key=mean_anoms.get)
        else:
            mean_anoms = {
                "Ductal Structure Distortion": 0.0,
                "Tumor Cell Density Clusters": 0.0,
                "Tumor Necrosis & Debris": 0.0,
            }
            dominant_slide_anomaly = "Morphologically Benign Baseline"

        pct_exceeding = {
            "Ductal Structure Distortion": round(float((df_tissue["ductal"] >= t_ductal).sum() / n_tissue * 100), 2) if n_tissue > 0 else 0.0,
            "Tumor Cell Density Clusters": round(float((df_tissue["density"] >= t_density).sum() / n_tissue * 100), 2) if n_tissue > 0 else 0.0,
            "Tumor Necrosis & Debris": round(float((df_tissue["necrosis"] >= t_necrosis).sum() / n_tissue * 100), 2) if n_tissue > 0 else 0.0,
        }

        return {
            "wsi_rgb": thumb_np,
            "grid_overlay": grid_overlay,
            "prob_upsampled": prob_upsampled,
            "tissue_mask": tissue_mask,
            "anomaly_overlay": anomaly_overlay,
            "df_tiles": df_tiles,
            "tile_crops": tile_crops,
            "scale": scale,
            "stride": stride,
            "summary": {
                "orig_dimensions": f"{orig_w} × {orig_h} px",
                "grid_dimensions": f"{len(df_tiles)} sampled tiles",
                "total_tiles": len(df_tiles),
                "tissue_tiles": n_tissue,
                "malignant_tiles": n_mal,
                "benign_tiles": n_tissue - n_mal,
                "tumor_burden_pct": burden_pct,
                "burden_formula": f"{n_mal} malignant / {n_tissue} viable tissue tiles",
                "denominator_definition": "Viable tissue tiles (excluding non-tissue glass)",
                "decision_threshold": 0.5000,
                "slide_verdict": slide_verdict,
                "is_malignant": is_slide_malignant or (n_mal > 0),
                "dominant_anomaly": dominant_slide_anomaly,
                "mean_anomalies": mean_anoms,
                "pct_exceeding_threshold": pct_exceeding,
                "atypia_status": "QUARANTINED (Excluded from active inference and clinical risk)",
                "mpp": f"{mpp_x:.4f} × {mpp_y:.4f} μm/px",
                "level_count": level_count,
            },
        }
    finally:
        slide.close()


def export_qupath_geojson(df_tiles: pd.DataFrame, patch_size: int = 224, output_path: Optional[Union[str, Path]] = None) -> str:
    """
    Exports tile coordinates and prediction attributes into a standard QuPath GeoJSON FeatureCollection.
    Compatible with QuPath v0.4+ / v0.5+: drag and drop onto an opened WSI in QuPath.
    """
    features = []
    for _, row in df_tiles.iterrows():
        x0 = float(row.get("wsi_x", 0))
        y0 = float(row.get("wsi_y", 0))
        x1 = x0 + patch_size
        y1 = y0 + patch_size

        is_mal = int(row.get("pred_label", 0)) == 1
        is_tissue = bool(row.get("is_tissue", True))

        if not is_tissue:
            class_name = "Background"
            color_int = -8355712  # Gray
        elif is_mal:
            class_name = "Tumor"
            color_int = -2349530   # Red (#DC2626)
        else:
            class_name = "Normal"
            color_int = -15293878  # Green (#16A34A)

        feature = {
            "type": "Feature",
            "id": str(row.get("tile_id", f"tile_{int(x0)}_{int(y0)}")),
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1],
                        [x0, y0],
                    ]
                ],
            },
            "properties": {
                "objectType": "tile",
                "classification": {
                    "name": class_name,
                    "colorRGB": color_int,
                },
                "measurements": {
                    "Malignancy Probability": float(row.get("pred_prob", 0.0)),
                    "Nuclear Pleomorphism Score": float(row.get("atypia")) if (row.get("atypia") is not None and pd.notna(row.get("atypia"))) else None,
                    "Ductal Distortion Score": float(row.get("ductal", 0.0)),
                    "Cell Density Score": float(row.get("density", 0.0)),
                    "Tumor Necrosis Score": float(row.get("necrosis", 0.0)),
                    "Blur Score": float(row.get("blur_score", 0.0)),
                    "Tissue Fraction Pct": float(row.get("tissue_fraction", 0.0)),
                },
                "risk_tier": str(row.get("risk_tier", "Unknown")),
                "dominant_anomaly": str(row.get("dominant_anomaly", "None")),
                "qc_status": str(row.get("qc_status", "UNKNOWN")),
            },
        }
        features.append(feature)

    geojson_obj = {
        "type": "FeatureCollection",
        "features": features,
    }
    geojson_str = json.dumps(geojson_obj, indent=2)

    if output_path is not None:
        p_out = Path(output_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        with open(p_out, "w", encoding="utf-8") as f:
            f.write(geojson_str)

    return geojson_str


def export_qupath_tsv(df_tiles: pd.DataFrame, patch_size: int = 224, output_path: Optional[Union[str, Path]] = None) -> str:
    """
    Exports tile coordinates and measurements into QuPath Tile Measurement TSV format.
    Can be directly imported into QuPath via 'Measure -> Show measurement table' or 'File -> Import measurements'.
    """
    tsv_rows = []
    for _, row in df_tiles.iterrows():
        x0 = float(row.get("wsi_x", 0))
        y0 = float(row.get("wsi_y", 0))
        cx = x0 + patch_size / 2.0
        cy = y0 + patch_size / 2.0

        is_mal = int(row.get("pred_label", 0)) == 1
        is_tissue = bool(row.get("is_tissue", True))
        class_name = "Background" if not is_tissue else ("Tumor" if is_mal else "Normal")

        tsv_rows.append({
            "Tile ID": row.get("tile_id", f"tile_{int(x0)}_{int(y0)}"),
            "Class": class_name,
            "Centroid X px": cx,
            "Centroid Y px": cy,
            "Top Left X px": x0,
            "Top Left Y px": y0,
            "Width px": patch_size,
            "Height px": patch_size,
            "Malignancy Probability": round(float(row.get("pred_prob", 0.0)), 4),
            "Nuclear Pleomorphism": round(float(row.get("atypia")), 4) if (row.get("atypia") is not None and pd.notna(row.get("atypia"))) else "QUARANTINED",
            "Ductal Distortion": round(float(row.get("ductal", 0.0)), 4),
            "Cell Density": round(float(row.get("density", 0.0)), 4),
            "Tumor Necrosis": round(float(row.get("necrosis", 0.0)), 4),
            "QC Status": row.get("qc_status", "PASS"),
            "Tissue Fraction": round(float(row.get("tissue_fraction", 0.0)), 1),
            "Dominant Anomaly": row.get("dominant_anomaly", "None"),
            "Risk Tier": row.get("risk_tier", "Low"),
        })

    df_export = pd.DataFrame(tsv_rows)
    tsv_str = df_export.to_csv(sep="\t", index=False)

    if output_path is not None:
        p_out = Path(output_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        df_export.to_csv(p_out, sep="\t", index=False)

    return tsv_str


def find_tcga_svs_slides(base_dir: Optional[Union[str, Path]] = None) -> Dict[str, List[Path]]:
    """
    Discovers TCGA WSI .svs files under the project data directory.
    Categorizes slides into 'tumor', 'normal', and 'all'.
    Handles missing directories, empty results, and arbitrary subfolder structures gracefully.
    """
    if base_dir is None:
        candidate_dirs = [
            PROJECT_ROOT / "data" / "External_test" / "TCGA",
            PROJECT_ROOT / "data" / "external_test" / "tcga_brca_wsi",
            PROJECT_ROOT / "data" / "TCGA",
        ]
        tcga_dir = next((d for d in candidate_dirs if d.exists()), candidate_dirs[0])
    else:
        tcga_dir = Path(base_dir)

    result: Dict[str, List[Path]] = {"tumor": [], "normal": [], "all": []}
    if not tcga_dir.exists():
        return result

    all_svs = sorted(list(tcga_dir.glob("**/*.svs")), key=lambda p: p.stat().st_size)
    result["all"] = all_svs

    for p in all_svs:
        tokens = p.name.split("-")
        sample_code = tokens[3][:2] if len(tokens) > 3 else ""
        if sample_code.startswith("0") or "tumor" in p.parent.name.lower() or "tumor" in p.name.lower():
            result["tumor"].append(p)
        elif sample_code.startswith("1") or "normal" in p.parent.name.lower() or "normal" in p.name.lower():
            result["normal"].append(p)
        else:
            result["tumor"].append(p)

    result["normal"] = sorted(
        result["normal"],
        key=lambda p: 0 if "73-4677" in p.name else (1 if "49-4505" in p.name else 2)
    )
    return result




