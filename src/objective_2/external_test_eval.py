"""
PathoVision AI — Objective 2: Independent External Test Evaluation Runner.

This module evaluates the frozen PathoVision AI models on external, out-of-distribution
histopathology datasets (e.g. PatchCamelyon, BreakHis, TCGA-BRCA, CAMELYON17).

Features:
- Completely decoupled from training, validation, and hyperparameter tuning pipelines.
- Supports both labeled evaluation (calculates ROC-AUC, PR-AUC, Sensitivity, Specificity, F1)
  and unlabelled inference (predicts malignancy probability, risk tiers, and morphological anomalies).
- Flexible input schemas:
    A) Subdirectories: `data/external_test/tumor/` and `data/external_test/normal/`
    B) Manifest file: `data/external_test/external_manifest.csv` (columns: `image_path,label`)
    C) Flat folder: `data/external_test/images/` (unlabelled screening)
- Generates structured CSV predictions, JSON metric summaries, and an error audit log.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.objective_2.layered_pipeline import LayeredDiagnosticPipeline
from src.objective_2.wsi_live_engine import ANOMALY_META

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ExternalTestEval")

EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external_test"
EXTERNAL_OUTPUT_DIR = PROJECT_ROOT / "result" / "objective_2" / "external_test"


def discover_external_patches(base_dir: Path) -> Tuple[List[Dict], bool]:
    """
    Discovers test patches and determines whether ground-truth labels are present.
    Returns: (patch_entries_list, has_ground_truth)
    """
    manifest_path = base_dir / "external_manifest.csv"
    if manifest_path.exists():
        logger.info(f"Loading external manifest: {manifest_path}")
        df_m = pd.read_csv(manifest_path)
        entries = []
        has_gt = "label" in df_m.columns or "true_label" in df_m.columns
        lbl_col = "label" if "label" in df_m.columns else "true_label"
        for _, row in df_m.iterrows():
            p_str = row.get("image_path", row.get("patch_path", ""))
            p_img = Path(p_str)
            if not p_img.is_absolute():
                if (PROJECT_ROOT / p_str).exists():
                    p_img = PROJECT_ROOT / p_str
                else:
                    p_img = base_dir / p_str

            # Only evaluate patches (.png, .jpg, .tif), not full .svs WSIs
            if p_img.exists() and p_img.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                entries.append({
                    "path": str(p_img),
                    "filename": p_img.name,
                    "dataset_source": str(row.get("dataset_source", "External")),
                    "label": int(row[lbl_col]) if has_gt and pd.notna(row[lbl_col]) else None,
                })
        if entries:
            return entries, has_gt

    # Check for labeled subdirectories (including nested folders like patchcamelyon/ and breakhis/)
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    entries = []
    has_gt = False

    for sub_ds in [base_dir, base_dir / "patchcamelyon", base_dir / "breakhis"]:
        if not sub_ds.exists():
            continue
        t_dir = sub_ds / "tumor" if (sub_ds / "tumor").exists() else (sub_ds / "positive" if (sub_ds / "positive").exists() else sub_ds / "malignant")
        n_dir = sub_ds / "normal" if (sub_ds / "normal").exists() else (sub_ds / "negative" if (sub_ds / "negative").exists() else sub_ds / "benign")

        if t_dir.exists() and n_dir.exists():
            has_gt = True
            for p in t_dir.glob("*"):
                if p.suffix.lower() in valid_exts:
                    entries.append({"path": str(p), "filename": p.name, "dataset_source": sub_ds.name, "label": 1})
            for p in n_dir.glob("*"):
                if p.suffix.lower() in valid_exts:
                    entries.append({"path": str(p), "filename": p.name, "dataset_source": sub_ds.name, "label": 0})

    if entries:
        return entries, has_gt

    # Fallback flat search
    flat_dirs = [base_dir / "images", base_dir]
    for d in flat_dirs:
        if d.exists():
            for p in d.glob("*"):
                if p.suffix.lower() in valid_exts:
                    entries.append({"path": str(p), "filename": p.name, "dataset_source": "Unlabelled", "label": None})
            if entries:
                break

    return entries, False


def run_external_evaluation(
    data_dir: Path = EXTERNAL_DATA_DIR,
    output_dir: Path = EXTERNAL_OUTPUT_DIR,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Executes external evaluation on unseen images.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for data
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    entries, has_gt = discover_external_patches(data_dir)

    if not entries:
        logger.warning(
            f"No external test patches found in: {data_dir}\n"
            "Please download an external dataset (e.g. PatchCamelyon or BreakHis) and place patches in:\n"
            f"  - {data_dir / 'tumor'}  (for malignant patches)\n"
            f"  - {data_dir / 'normal'} (for benign patches)\n"
            "Or create a manifest CSV at:\n"
            f"  - {data_dir / 'external_manifest.csv'}"
        )
        readme_path = data_dir / "README.md"
        if not readme_path.exists():
            readme_path.write_text(
                "# External Test Dataset Directory\n\n"
                "Place unseen external test patches in this directory to evaluate model generalization.\n\n"
                "## Option 1: Labeled Folders\n"
                "- `data/external_test/tumor/` : Put 100-250 malignant patches here.\n"
                "- `data/external_test/normal/`: Put 100-250 normal/benign patches here.\n\n"
                "## Option 2: Manifest CSV\n"
                "Create `data/external_test/external_manifest.csv` with columns: `image_path,label` (1=Tumor, 0=Normal).\n\n"
                "## Option 3: Unlabelled Screening\n"
                "Place any histopathology patches in `data/external_test/images/`.\n",
                encoding="utf-8",
            )
        return {"status": "EMPTY_DATASET", "message": "No external test images found in data/external_test/."}

    logger.info(f"Discovered {len(entries)} external test images. Has ground truth: {has_gt}")

    # Load frozen pipeline
    cls_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt"
    anom_ckpt = PROJECT_ROOT / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_best.pt"

    pipeline = LayeredDiagnosticPipeline(
        classifier_ckpt=cls_ckpt,
        anomaly_ckpt=anom_ckpt,
        device=device,
    )

    pred_records = []
    logger.info("Beginning batch inference on external test set...")

    for i, item in enumerate(entries):
        p_path = item["path"]
        try:
            diag = pipeline.analyze_patch(p_path)
            l1 = diag["layer_1_quality"]
            l2 = diag["layer_2_malignancy"]
            l3 = diag["layer_3_anomalies"]
            l4 = diag["layer_4_risk_and_explainability"]

            rec = {
                "filename": item["filename"],
                "path": p_path,
                "dataset_source": item.get("dataset_source", "External"),
                "ground_truth_label": item["label"],
                "ground_truth_class": ("Tumor" if item["label"] == 1 else "Normal") if item["label"] is not None else "Unknown",
                "quality_passed": l1["passed"],
                "quality_status": l1["status"],
                "blur_score": l1["blur_score"],
                "tissue_fraction_pct": l1["tissue_fraction"],
                "malignancy_probability": l2["probability"],
                "predicted_label": 1 if l2["verdict"] == "Malignant" else 0,
                "predicted_verdict": l2["verdict"],
                "risk_tier": l4["risk_tier"],
                "atypia_score": l3["pleomorphism_atypia"]["score"],
                "atypia_active": l3["pleomorphism_atypia"]["active"],
                "ductal_score": l3["ductal_distortion"]["score"],
                "ductal_active": l3["ductal_distortion"]["active"],
                "density_score": l3["density_clusters"]["score"],
                "density_active": l3["density_clusters"]["active"],
                "necrosis_score": l3["tumor_necrosis"]["score"],
                "necrosis_active": l3["tumor_necrosis"]["active"],
            }
            pred_records.append(rec)
        except Exception as e:
            logger.error(f"Failed to analyze patch {p_path}: {e}")

    df_preds = pd.DataFrame(pred_records)
    preds_csv_path = output_dir / "external_test_predictions.csv"
    df_preds.to_csv(preds_csv_path, index=False)
    logger.info(f"Saved external test predictions: {preds_csv_path}")

    # Compute metrics if ground truth is available
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_images_evaluated": len(df_preds),
        "has_ground_truth": has_gt,
        "models_evaluated": {
            "primary_classifier": "EfficientNet-B0 (Trained on CAMELYON16)",
            "anomaly_heads": ["atypia (MITOS-ATYPIA14)", "ductal (BACH)", "density (NuCLS)", "necrosis (BCSS)"],
        },
    }

    if has_gt and df_preds["ground_truth_label"].notna().sum() > 0:
        df_eval = df_preds[df_preds["ground_truth_label"].notna()].copy()
        y_true = df_eval["ground_truth_label"].astype(int).values
        y_prob = df_eval["malignancy_probability"].astype(float).values
        y_pred = df_eval["predicted_label"].astype(int).values

        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))

        roc_auc = float(roc_auc_score(y_true, y_prob)) if (n_pos > 0 and n_neg > 0) else None
        pr_auc = float(average_precision_score(y_true, y_prob)) if n_pos > 0 else None
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * (prec * sens) / (prec + sens)) if (prec + sens) > 0 else 0.0

        metrics_dict = {
            "sample_counts": {"total": len(y_true), "tumor_positives": n_pos, "normal_negatives": n_neg},
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "balanced_accuracy": round(bal_acc, 4),
            "sensitivity_recall": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
        }
        report["metrics"] = metrics_dict

        # Identify False Positives & False Negatives for error audit
        df_fp = df_eval[(df_eval["ground_truth_label"] == 0) & (df_eval["predicted_label"] == 1)]
        df_fn = df_eval[(df_eval["ground_truth_label"] == 1) & (df_eval["predicted_label"] == 0)]

        report["error_analysis"] = {
            "false_positive_count": len(df_fp),
            "false_negative_count": len(df_fn),
            "false_positive_samples": df_fp["filename"].tolist()[:10],
            "false_negative_samples": df_fn["filename"].tolist()[:10],
        }

        logger.info(f"External Test ROC-AUC: {roc_auc:.4f} | Sensitivity: {sens:.4f} | Specificity: {spec:.4f}")
    else:
        # Unlabelled summary
        pred_pos = int((df_preds["predicted_label"] == 1).sum())
        report["unlabelled_summary"] = {
            "predicted_malignant_count": pred_pos,
            "predicted_normal_count": len(df_preds) - pred_pos,
            "mean_malignancy_prob": round(float(df_preds["malignancy_probability"].mean()), 4),
            "risk_tier_distribution": df_preds["risk_tier"].value_counts().to_dict(),
        }

    # Save JSON report
    report_json_path = output_dir / "external_test_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved external test report: {report_json_path}")

    return report


if __name__ == "__main__":
    run_external_evaluation()
