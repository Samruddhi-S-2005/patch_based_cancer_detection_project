"""
Tests for BreakHis External Evaluation Pipeline.

Validates:
1. BreakHis manifest schema, record count (7,909 images), patient identifiers, and subtype taxonomy.
2. Immutability and integrity of all four classifier checkpoints and the clean anomaly model.
3. Immutability and preservation of existing CAMELYON16 registries, manifests, and benchmark metrics.
4. Correct output class naming ('Metastatic Carcinoma' vs 'Normal Lymph Node').
5. Absence of quarantined atypia head (only permitted heads: ductal, density, necrosis).
6. Existence and valid formatting of generated Grad-CAM paired heatmaps.
7. Verification of full predictions file (31,636 rows) and aggregated metrics files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Baseline SHA-256 hashes recorded before BreakHis evaluation began
BASELINE_CHECKPOINT_HASHES = {
    "EfficientNet-B0": (
        PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt",
        "cc6764e073a21e523b2ae6a6b24b6c718bf4159a778f2563a307c2e88900b3ef",
    ),
    "ResNet-50": (
        PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "resnet" / "best_model.pt",
        "a1e47d42e1a5b732eeadb5cea620807add979ebea938f9f7b73072b6fd4f8864",
    ),
    "DenseNet-121": (
        PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "densenet" / "best_model.pt",
        "13e80add08df8dd471b95697b50e30c4745c08fc511ed2da8b1782fb53094303",
    ),
    "VGG-16": (
        PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "vgg" / "best_model.pt",
        "353f1fcb8d13e412a6dc690242edbf2126d537ec99152223b4b8f1fb1fba24dd",
    ),
    "Clean-Anomaly-Heads": (
        PROJECT_ROOT / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_clean.pt",
        "c713a37146fe4a28bde27cc9631f6c213002214de677f8d299f33daa7bf368e1",
    ),
}

BASELINE_CAMELYON_HASHES = {
    "wsi_registry": (
        PROJECT_ROOT / "data" / "wsi_registry.csv",
        "b76a9b421ee818302db3642de14c87d2d416c1f1fbb693358be19132a4f88983",
    ),
    "dataset_manifest": (
        PROJECT_ROOT / "data" / "dataset_manifest.csv",
        "19ff549541c1cae6a018273808314d65a1586e3ff9cfd981f372c4aee9c30f5d",
    ),
    "master_test_benchmark": (
        PROJECT_ROOT / "result" / "objective_2" / "metrics" / "master_test_benchmark_comparison.csv",
        "308cf23119405b488a00187040e79e1319951fa2c3cb0f8c4596071a90c1641c",
    ),
}


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestBreakHisEvaluationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = PROJECT_ROOT / "data" / "External_test" / "breakhis_manifest.csv"
        cls.heatmaps_dir = PROJECT_ROOT / "result" / "objective_2" / "external_test" / "breakhis" / "heatmaps"
        cls.pilot_preds_path = PROJECT_ROOT / "result" / "objective_2" / "external_test" / "breakhis" / "predictions" / "breakhis_pilot_predictions.csv"
        cls.full_preds_path = PROJECT_ROOT / "result" / "objective_2" / "external_test" / "breakhis" / "predictions" / "breakhis_predictions_all_models.csv"
        cls.metrics_dir = PROJECT_ROOT / "result" / "objective_2" / "external_test" / "breakhis" / "metrics"
        cls.report_path = PROJECT_ROOT / "result" / "objective_2" / "external_test" / "breakhis" / "reports" / "BREAKHIS_EXTERNAL_TEST_REPORT.md"

    def test_01_manifest_existence_and_integrity(self):
        """Manifest must exist, have exactly 7,909 rows, 0 nulls, and verified slide/patient counts."""
        self.assertTrue(self.manifest_path.exists(), f"Manifest missing: {self.manifest_path}")
        df = pd.read_csv(self.manifest_path)
        self.assertEqual(len(df), 7909, f"Expected 7,909 images in manifest, found {len(df)}")

        expected_columns = [
            "biopsy_procedure", "tumor_class", "tumor_class_code", "tumor_subtype_code",
            "tumor_subtype_name", "year", "slide_id", "patient_id", "magnification",
            "magnification_str", "seq_num", "breakhis_label", "breakhis_label_name",
            "semantic_note", "image_id", "filename", "relative_path", "file_size_bytes",
            "width", "height"
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing column: {col}")
            self.assertEqual(df[col].isna().sum(), 0, f"Null values found in column {col}")

        # Unique counts
        self.assertEqual(df["image_id"].nunique(), 7909, "Duplicate image IDs found")
        self.assertEqual(df["tumor_class"].nunique(), 2)
        self.assertCountEqual(df["tumor_class"].unique(), ["benign", "malignant"])
        self.assertCountEqual(df["magnification"].unique(), [40, 100, 200, 400])

        # Subtypes: 4 benign + 4 malignant = 8 subtypes
        self.assertEqual(df["tumor_subtype_code"].nunique(), 8)
        self.assertEqual(df["patient_id"].nunique(), 82, "Expected 82 distinct patients")

        # Class counts
        benign_count = (df["tumor_class"] == "benign").sum()
        malignant_count = (df["tumor_class"] == "malignant").sum()
        self.assertEqual(benign_count, 2480, f"Expected 2480 benign images, got {benign_count}")
        self.assertEqual(malignant_count, 5429, f"Expected 5429 malignant images, got {malignant_count}")

    def test_02_checkpoint_immutability(self):
        """None of the four classifier checkpoints or the clean anomaly heads may be altered."""
        for name, (path, expected_sha) in BASELINE_CHECKPOINT_HASHES.items():
            self.assertTrue(path.exists(), f"Checkpoint missing for {name}: {path}")
            current_sha = compute_sha256(path)
            self.assertEqual(
                current_sha,
                expected_sha,
                f"CRITICAL: Checkpoint {name} ({path.name}) was modified! Hash mismatch: {current_sha} != {expected_sha}"
            )

    def test_03_camelyon16_registry_and_manifest_immutability(self):
        """CAMELYON16 manifests and registries must remain pristine."""
        for name, (path, expected_sha) in BASELINE_CAMELYON_HASHES.items():
            self.assertTrue(path.exists(), f"Benchmark file missing: {path}")
            current_sha = compute_sha256(path)
            self.assertEqual(
                current_sha,
                expected_sha,
                f"CRITICAL: {name} ({path.name}) was modified! Hash mismatch."
            )

    def test_04_pilot_predictions_schema_and_class_naming(self):
        """Verifies pilot predictions follow strict scientific naming and permitted heads."""
        self.assertTrue(self.pilot_preds_path.exists(), "Pilot predictions file not found")
        df_pilot = pd.read_csv(self.pilot_preds_path)

        # Check model names
        expected_models = {"EfficientNet-B0", "ResNet-50", "DenseNet-121", "VGG-16"}
        self.assertEqual(set(df_pilot["model_name"].unique()), expected_models)

        # Confirm predicted classes are strictly CAMELYON16 labels
        valid_classes = {"Normal Lymph Node", "Metastatic Carcinoma"}
        self.assertTrue(
            set(df_pilot["predicted_camelyon16_class"].unique()).issubset(valid_classes),
            f"Invalid predicted classes found: {set(df_pilot['predicted_camelyon16_class'].unique())}"
        )

        # Ensure no MITOS-ATYPIA-14 atypia head was evaluated
        col_names = list(df_pilot.columns)
        for col in col_names:
            self.assertNotIn("atypia", col.lower(), f"Quarantined atypia head found in predictions: {col}")

        # Ensure permitted heads are present
        for permitted in ["ductal", "density", "necrosis"]:
            self.assertIn(f"anomaly_{permitted}_score", col_names)
            self.assertIn(f"anomaly_{permitted}_flagged", col_names)

    def test_05_representative_heatmaps(self):
        """Verifies representative paired heatmaps exist and have non-zero dimensions."""
        self.assertTrue(self.heatmaps_dir.exists(), "Heatmaps directory not found")
        heatmap_files = list(self.heatmaps_dir.glob("*.png"))
        self.assertGreaterEqual(len(heatmap_files), 32, f"Expected at least 32 representative heatmaps, found {len(heatmap_files)}")

        for hmap_file in heatmap_files[:5]:
            with Image.open(hmap_file) as img:
                w, h = img.size
                self.assertGreater(w, 200)
                self.assertGreater(h, 200)

    def test_06_full_dataset_predictions_integrity(self):
        """Full evaluation predictions file must have 7,909 * 4 = 31,636 records, with no missing values."""
        self.assertTrue(self.full_preds_path.exists(), f"Full predictions missing: {self.full_preds_path}")
        df_full = pd.read_csv(self.full_preds_path)
        self.assertEqual(len(df_full), 7909 * 4, f"Expected 31,636 rows, found {len(df_full)}")
        self.assertEqual(df_full["image_id"].nunique(), 7909)

        # Confirm exactly 4 models evaluated equally
        for m_name in ["EfficientNet-B0", "ResNet-50", "DenseNet-121", "VGG-16"]:
            self.assertEqual((df_full["model_name"] == m_name).sum(), 7909)

        # Zero atypia in full predictions
        for col in df_full.columns:
            self.assertNotIn("atypia", col.lower())

        # Permitted heads present
        for permitted in ["ductal", "density", "necrosis"]:
            self.assertIn(f"anomaly_{permitted}_score", df_full.columns)
            self.assertIn(f"anomaly_{permitted}_flagged", df_full.columns)

    def test_07_generated_metrics_and_report_artifacts(self):
        """Summary CSV tables and markdown report must exist and be non-empty."""
        self.assertTrue(self.report_path.exists())
        self.assertGreater(self.report_path.stat().st_size, 1000)

        required_tables = [
            "breakhis_model_transfer_summary.csv",
            "breakhis_stratified_by_magnification.csv",
            "breakhis_stratified_by_subtype.csv",
            "breakhis_anomaly_cross_activation_summary.csv",
            "breakhis_patient_level_summary.csv",
            "breakhis_exploratory_ood_metrics.json",
        ]
        for tbl in required_tables:
            p = self.metrics_dir / tbl
            self.assertTrue(p.exists(), f"Missing required metrics artifact: {tbl}")
            self.assertGreater(p.stat().st_size, 50, f"Empty metrics artifact: {tbl}")

        # Recalculate exploratory OOD metrics directly from full predictions file
        from sklearn.metrics import roc_auc_score, average_precision_score
        df_full = pd.read_csv(self.full_preds_path)
        df_summary = pd.read_csv(self.metrics_dir / "breakhis_model_transfer_summary.csv")

        for m_name in ["EfficientNet-B0", "ResNet-50", "DenseNet-121", "VGG-16"]:
            sub = df_full[df_full["model_name"] == m_name]
            y_true = (sub["tumor_class"] == "malignant").astype(int)
            y_prob = sub["metastatic_probability"].values
            recalc_roc = roc_auc_score(y_true, y_prob)
            recalc_pr = average_precision_score(y_true, y_prob)

            expected_roc = df_summary[df_summary["model_name"] == m_name]["exploratory_ood_roc_auc"].values[0]
            expected_pr = df_summary[df_summary["model_name"] == m_name]["exploratory_ood_pr_auc"].values[0]
            self.assertAlmostEqual(recalc_roc, expected_roc, places=4)
            self.assertAlmostEqual(recalc_pr, expected_pr, places=4)


if __name__ == "__main__":
    unittest.main()
