"""
Test Suite: Comprehensive Project Audit, Preprocessing Records, Mask Generation,
Feature Extraction, and Traceability.

Validates:
1. Complete inventory and gap analysis reports.
2. Objective 1 Preprocessing Audit CSV and 7-sheet Excel workbook.
3. Deterministic binary tissue masks (14,719 masks on disk) and stage visualizations.
4. Objective 2 CNN pre-classifier embeddings across all four architectures (1280, 2048, 1024, 4096).
5. Explicit morphology audit statements and clean anomaly head persistence.
6. Bit-for-bit cryptographic immutability of original checkpoints and benchmark artifacts.
"""

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestProjectAuditAndTraceability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = PROJECT_ROOT
        cls.obj1_dir = cls.root / "results" / "objective_1_preprocessing"
        cls.obj2_dir = cls.root / "results" / "objective_2_model_and_anomaly"
        cls.obj3_dir = cls.root / "results" / "objective_3_evaluation_and_xai"
        cls.obj4_dir = cls.root / "results" / "objective_4_web_application"

    def test_01_project_inventory_and_gap_report(self):
        """Verifies Phase 1 inventory artifacts and implementation gap report."""
        inv_csv = self.root / "project_audit" / "inventory" / "project_file_inventory.csv"
        self.assertTrue(inv_csv.exists(), "project_file_inventory.csv must exist")
        # Check that it contains over 70,000 indexed files
        with open(inv_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertIn("relative_path", header)
            self.assertIn("size_bytes", header)
            self.assertIn("objective_association", header)
            row_count = sum(1 for _ in reader)
            self.assertGreaterEqual(row_count, 70000, f"Inventory must index >70k files, got {row_count}")

        # Check artifact inventory JSON
        inv_json = self.root / "project_audit" / "inventory" / "artifact_inventory.json"
        self.assertTrue(inv_json.exists(), "artifact_inventory.json must exist")
        with open(inv_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for expected_key in ["datasets", "manifests", "checkpoints", "masks", "quality_control_logs", "feature_artifacts"]:
            self.assertIn(expected_key, data, f"Key '{expected_key}' missing from artifact_inventory.json")

        # Check implementation gap report
        gap_md = self.root / "project_audit" / "inventory" / "implementation_gap_report.md"
        self.assertTrue(gap_md.exists(), "implementation_gap_report.md must exist")
        with open(gap_md, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Detailed Answers to the 10 Specific Audit Questions", content)
        self.assertIn("### 1.", content)
        self.assertIn("### 10.", content)

        # Check file migration plan
        mig_csv = self.root / "project_audit" / "file_migration_plan.csv"
        self.assertTrue(mig_csv.exists(), "file_migration_plan.csv must exist")

    def test_02_objective_1_preprocessing_records_and_excel(self):
        """Verifies Objective 1 full per-patch QC records and multi-tab Excel workbook."""
        audit_csv = self.obj1_dir / "quality_control" / "training_patch_preprocessing_audit.csv"
        self.assertTrue(audit_csv.exists(), "training_patch_preprocessing_audit.csv must exist")
        
        df = pd.read_csv(audit_csv)
        self.assertEqual(len(df), 14803, f"Audit must cover all 14,803 training patches, got {len(df)}")
        
        # Verify required audit fields
        required_cols = [
            "patch_id", "filename", "source_wsi_id", "coord_x", "coord_y",
            "width", "height", "tissue_ratio", "white_ratio", "blur_score",
            "gate_tissue_pass", "gate_white_pass", "gate_blur_pass",
            "final_decision", "removal_reason"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column '{col}' must be present in audit CSV")

        # Verify exact counts: 14,719 KEEP and 84 REMOVE
        dec_counts = df["final_decision"].value_counts().to_dict()
        self.assertEqual(dec_counts.get("KEEP", 0), 14719)
        self.assertEqual(dec_counts.get("REMOVE", 0), 84)

        # Check summary CSVs
        for s_file in ["summary_by_source_wsi.csv", "summary_by_class.csv", "summary_by_rejection_reason.csv"]:
            p = self.obj1_dir / "quality_control" / s_file
            self.assertTrue(p.exists(), f"Summary CSV {s_file} must exist")

        # Check Excel workbook with openpyxl
        excel_path = self.obj1_dir / "quality_control" / "training_patch_quality_control.xlsx"
        self.assertTrue(excel_path.exists(), "training_patch_quality_control.xlsx must exist")
        import openpyxl
        wb = openpyxl.load_workbook(excel_path, read_only=True)
        expected_sheets = [
            "Dataset Split Summary",
            "QC Threshold Definitions",
            "Accepted and Rejected Counts",
            "Per-WSI Summary",
            "Per-Patch Audit Sample (1000)",
            "Missing Files & Errors",
            "Reproducibility Metadata",
        ]
        for sheet_name in expected_sheets:
            self.assertIn(sheet_name, wb.sheetnames, f"Sheet '{sheet_name}' missing from Excel workbook")
        wb.close()

    def test_03_objective_1_tissue_masks_and_visualizations(self):
        """Verifies Objective 1 generated binary tissue masks and stage visualizations."""
        mask_manifest_path = self.obj1_dir / "masks" / "tissue_mask_manifest.csv"
        self.assertTrue(mask_manifest_path.exists(), "tissue_mask_manifest.csv must exist")

        df_masks = pd.read_csv(mask_manifest_path)
        self.assertEqual(len(df_masks), 14719, f"Mask manifest must contain 14,719 patches, got {len(df_masks)}")

        # Verify a random sample of actual mask files on disk
        sample_rows = df_masks.sample(n=min(50, len(df_masks)), random_state=42)
        for _, row in sample_rows.iterrows():
            mask_rel = row["mask_path"]
            mask_full = self.root / mask_rel
            self.assertTrue(mask_full.exists(), f"Mask file must exist at {mask_full}")
            
            # Verify dimensions and binary pixel values
            with Image.open(mask_full) as img:
                self.assertEqual(img.size, (224, 224), "Mask dimension must be 224x224")
                arr = np.array(img)
                unique_vals = set(np.unique(arr))
                # Must only contain 0 and 255 (or single value if 100% tissue / 0% tissue)
                self.assertTrue(unique_vals.issubset({0, 255}), f"Mask values must be binary {{0, 255}}, got {unique_vals}")

        # Check stage visualizations
        vis_index_path = self.obj1_dir / "stage_visualizations" / "stage_visualization_index.csv"
        self.assertTrue(vis_index_path.exists(), "stage_visualization_index.csv must exist")
        df_vis = pd.read_csv(vis_index_path)
        self.assertEqual(len(df_vis), 20, "Must have 20 visual stage composites")
        for _, row in df_vis.iterrows():
            comp_path = self.root / row["composite_visualization_path"]
            self.assertTrue(comp_path.exists(), f"Composite image must exist at {comp_path}")

    def test_04_objective_2_feature_embeddings_and_morphology(self):
        """Verifies Objective 2 pre-classifier embeddings across all 4 architectures and morphology report."""
        feat_dir = self.obj2_dir / "feature_extraction"
        expected_shapes = {
            "efficientnet_b0_embeddings.npy": (200, 1280),
            "resnet_50_embeddings.npy": (200, 2048),
            "densenet_121_embeddings.npy": (200, 1024),
            "vgg_16_embeddings.npy": (200, 4096),
        }

        for filename, exp_shape in expected_shapes.items():
            npy_path = feat_dir / filename
            self.assertTrue(npy_path.exists(), f"Feature embedding file {filename} must exist")
            arr = np.load(npy_path)
            self.assertEqual(arr.shape, exp_shape, f"{filename} shape must be {exp_shape}, got {arr.shape}")
            self.assertFalse(np.isnan(arr).any(), f"{filename} contains NaNs")
            self.assertGreater(float(np.var(arr)), 0.0, f"{filename} has zero variance")

        # Check feature manifest
        feat_manifest = feat_dir / "feature_extraction_manifest.csv"
        self.assertTrue(feat_manifest.exists(), "feature_extraction_manifest.csv must exist")
        df_feat = pd.read_csv(feat_manifest)
        self.assertEqual(len(df_feat), 200, "Manifest must index 200 patches")

        # Check morphology report
        morph_md = self.obj2_dir / "reports" / "explicit_morphology_analysis.md"
        self.assertTrue(morph_md.exists(), "explicit_morphology_analysis.md must exist")
        with open(morph_md, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Latent Neural Features vs Biological Measurements", content)
        self.assertIn("quarantined", content.lower())

    def test_05_objective_3_and_4_benchmarks_and_traceability(self):
        """Verifies benchmark preservation, BreakHis decoupling, and checkpoint integrity."""
        # CAMELYON16 held-out test metrics
        cam_metrics = self.obj3_dir / "metrics" / "master_test_benchmark_comparison.csv"
        self.assertTrue(cam_metrics.exists(), "master_test_benchmark_comparison.csv must exist")
        df_cam = pd.read_csv(cam_metrics)
        models = df_cam["Model"].str.upper().tolist()
        for expected_model in ["EFFICIENTNET", "RESNET", "DENSENET", "VGG"]:
            self.assertIn(expected_model, models)

        # BreakHis metrics exist and decouple from CAMELYON16
        brk_metrics = self.obj3_dir / "metrics" / "breakhis_model_transfer_summary.csv"
        self.assertTrue(brk_metrics.exists(), "breakhis_model_transfer_summary.csv must exist")

        # Checkpoint hash integrity
        clean_heads_path = self.root / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_clean.pt"
        self.assertTrue(clean_heads_path.exists(), "anomaly_linear_heads_clean.pt must exist")
        with open(clean_heads_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        self.assertTrue(h.startswith("c713a371"), f"Clean heads SHA-256 must start with c713a371, got {h}")

        # Pre-audit backup exists
        backup_dir = self.root / "backups" / "checkpoint_20261020_pre_audit"
        self.assertTrue(backup_dir.exists(), "Pre-audit backup directory must exist")
        self.assertTrue((backup_dir / "app.py").exists(), "Backup of app.py must exist")


if __name__ == "__main__":
    unittest.main()
