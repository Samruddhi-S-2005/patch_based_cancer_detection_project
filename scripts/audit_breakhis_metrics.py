"""
Independent Metric Verification and Reconciliation Script for BreakHis External Evaluation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

manifest_path = PROJECT_ROOT / "data" / "External_test" / "breakhis_manifest.csv"
preds_path = PROJECT_ROOT / "result" / "objective_2" / "external_test" / "breakhis" / "predictions" / "breakhis_predictions_all_models.csv"
summary_path = PROJECT_ROOT / "result" / "objective_2" / "external_test" / "breakhis" / "metrics" / "breakhis_model_transfer_summary.csv"
mag_path = PROJECT_ROOT / "result" / "objective_2" / "external_test" / "breakhis" / "metrics" / "breakhis_stratified_by_magnification.csv"
subtype_path = PROJECT_ROOT / "result" / "objective_2" / "external_test" / "breakhis" / "metrics" / "breakhis_stratified_by_subtype.csv"
anom_path = PROJECT_ROOT / "result" / "objective_2" / "external_test" / "breakhis" / "metrics" / "breakhis_anomaly_cross_activation_summary.csv"

df_manifest = pd.read_csv(manifest_path)
df_preds = pd.read_csv(preds_path)
df_summary = pd.read_csv(summary_path)
df_mag = pd.read_csv(mag_path)
df_subtype = pd.read_csv(subtype_path)
df_anom = pd.read_csv(anom_path)

print("=== 1. IMAGE ID ALIGNMENT & POLARITY ===")
manifest_img_ids = set(df_manifest["image_id"])
preds_img_ids = set(df_preds["image_id"])
print(f"Manifest IDs count: {len(manifest_img_ids)}")
print(f"Preds unique IDs count: {len(preds_img_ids)}")
print(f"Set difference: {manifest_img_ids ^ preds_img_ids}")

# Verify 1-to-1 matching across metadata columns
m_merged = pd.merge(
    df_preds[["image_id", "breakhis_label", "tumor_class", "tumor_subtype_code", "magnification"]].drop_duplicates(),
    df_manifest[["image_id", "breakhis_label", "tumor_class", "tumor_subtype_code", "magnification"]],
    on="image_id",
    suffixes=("_pred", "_man"),
)
print(f"Label mismatches: {(m_merged['breakhis_label_pred'] != m_merged['breakhis_label_man']).sum()}")
print(f"Class mismatches: {(m_merged['tumor_class_pred'] != m_merged['tumor_class_man']).sum()}")
print(f"Subtype mismatches: {(m_merged['tumor_subtype_code_pred'] != m_merged['tumor_subtype_code_man']).sum()}")
print(f"Magnification mismatches: {(m_merged['magnification_pred'] != m_merged['magnification_man']).sum()}")

# Polarity check
print("\nPolarity Check:")
print(f"Malignant count (label 1): {(df_manifest['breakhis_label'] == 1).sum()} (tumor_class == 'malignant': {(df_manifest['tumor_class'] == 'malignant').sum()})")
print(f"Benign count (label 0): {(df_manifest['breakhis_label'] == 0).sum()} (tumor_class == 'benign': {(df_manifest['tumor_class'] == 'benign').sum()})")

print("\n=== 2. INDEPENDENT RE-COMPUTATION OF OVERALL METRICS ===")
all_passed = True
for m_name in ["DenseNet-121", "EfficientNet-B0", "ResNet-50", "VGG-16"]:
    df_m = df_preds[df_preds["model_name"] == m_name].copy()
    y_true = df_m["breakhis_label"].values
    y_score = df_m["metastatic_probability"].values
    y_pred = (y_score >= 0.5).astype(int)

    roc_auc = round(float(roc_auc_score(y_true, y_score)), 4)
    pr_auc = round(float(average_precision_score(y_true, y_score)), 4)
    bacc = round(float(balanced_accuracy_score(y_true, y_pred)), 4)

    mal_mask = (y_true == 1)
    ben_mask = (y_true == 0)

    mal_trans = round(float(np.mean(y_score[mal_mask] >= 0.5)), 4)
    mal_mean = round(float(np.mean(y_score[mal_mask])), 4)
    mal_median = round(float(np.median(y_score[mal_mask])), 4)

    ben_cross = round(float(np.mean(y_score[ben_mask] >= 0.5)), 4)
    ben_mean = round(float(np.mean(y_score[ben_mask])), 4)
    ben_median = round(float(np.median(y_score[ben_mask])), 4)

    row = df_summary[df_summary["model_name"] == m_name].iloc[0]

    checks = [
        ("ROC-AUC", roc_auc, row["exploratory_ood_roc_auc"]),
        ("PR-AUC", pr_auc, row["exploratory_ood_pr_auc"]),
        ("Balanced Accuracy", bacc, row["exploratory_ood_balanced_accuracy"]),
        ("Malignant Transfer", mal_trans, row["malignant_metastatic_transfer_rate"]),
        ("Malignant Mean", mal_mean, row["malignant_mean_metastatic_score"]),
        ("Malignant Median", mal_median, row["malignant_median_metastatic_score"]),
        ("Benign Cross-Act", ben_cross, row["benign_neoplasm_cross_activation_rate"]),
        ("Benign Mean", ben_mean, row["benign_mean_metastatic_score"]),
        ("Benign Median", ben_median, row["benign_median_metastatic_score"]),
    ]

    print(f"\nModel: {m_name}")
    for name, calc, recorded in checks:
        diff = abs(calc - recorded)
        status = "MATCH" if diff < 1e-4 else f"MISMATCH (diff={diff})"
        if diff >= 1e-4:
            all_passed = False
        print(f"  {name:20s}: Calc={calc:.4f} | Recorded={recorded:.4f} -> {status}")

print(f"\nOverall Summary Match: {all_passed}")

print("\n=== 3. ANOMALY HEAD METRIC VERIFICATION ===")
for _, row in df_anom.iterrows():
    st = row["tumor_subtype"]
    tc = row["tumor_class"]
    sub_preds = df_preds[(df_preds["tumor_subtype_name"] == st) & (df_preds["model_name"] == "EfficientNet-B0")]

    calc_duct_flag = round(float((sub_preds["anomaly_ductal_score"] >= 0.4038).mean()), 4)
    calc_duct_mean = round(float(sub_preds["anomaly_ductal_score"].mean()), 4)
    calc_dens_flag = round(float((sub_preds["anomaly_density_score"] >= 0.5315).mean()), 4)
    calc_dens_mean = round(float(sub_preds["anomaly_density_score"].mean()), 4)
    calc_necr_flag = round(float((sub_preds["anomaly_necrosis_score"] >= 0.2715).mean()), 4)
    calc_necr_mean = round(float(sub_preds["anomaly_necrosis_score"].mean()), 4)

    assert abs(calc_duct_flag - row["ductal_flagged_rate"]) < 1e-3, f"Ductal flag mismatch for {st}"
    assert abs(calc_duct_mean - row["ductal_mean_score"]) < 1e-3, f"Ductal mean mismatch for {st}"
    assert abs(calc_dens_flag - row["density_flagged_rate"]) < 1e-3, f"Density flag mismatch for {st}"
    assert abs(calc_dens_mean - row["density_mean_score"]) < 1e-3, f"Density mean mismatch for {st}"
    assert abs(calc_necr_flag - row["necrosis_flagged_rate"]) < 1e-3, f"Necrosis flag mismatch for {st}"
    assert abs(calc_necr_mean - row["necrosis_mean_score"]) < 1e-3, f"Necrosis mean mismatch for {st}"

print("All Anomaly Head metrics verified and match raw predictions exactly!")
