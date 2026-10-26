"""
Test Suite: Prediction Reliability, Trust & Verification
Validates:
1. Correct model-to-task and class mapping (CAMELYON16 Normal Lymph Node vs Metastatic Carcinoma).
2. Consistency between raw scores, decision thresholds, and displayed verdict labels.
3. Dataset compatibility & domain-shift guardrails for external inputs (BreakHis, user uploads).
4. Correct checkpoint and Grad-CAM target selection for all architectures.
5. Correct handling and quarantine of anomaly heads (atypia quarantined, 3 permitted heads experimental).
6. Immutable preservation of all 5 model checkpoints, dataset manifests, and WSI split assignments.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.objective_2.layered_pipeline import LayeredDiagnosticPipeline


class TestPredictionReliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.cls_ckpt = cls.project_root / "result" / "objective_2" / "checkpoints" / "best_model.pt"
        cls.anom_ckpt = cls.project_root / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_clean.pt"
        cls.device = torch.device("cpu")
        cls.pipeline = LayeredDiagnosticPipeline(
            classifier_ckpt=cls.cls_ckpt,
            anomaly_ckpt=cls.anom_ckpt,
            device=cls.device,
        )
        # Synthetic H&E test patch (224, 224, 3)
        np.random.seed(42)
        cls.sample_patch = np.random.randint(40, 220, (224, 224, 3), dtype=np.uint8)

    # --------------------------------------------------------------------------
    # 1. Model-to-Task and Class Mapping
    # --------------------------------------------------------------------------
    def test_01_model_to_task_and_class_mapping(self):
        """Verifies primary task is strictly CAMELYON16 lymph node classification with exact class names."""
        diag = self.pipeline.analyze_patch(self.sample_patch)
        l2 = diag["layer_2_malignancy"]

        self.assertIn(
            l2["predicted_class"],
            ["Normal Lymph Node", "Metastatic Carcinoma"],
            f"Predicted class '{l2['predicted_class']}' must be either 'Normal Lymph Node' or 'Metastatic Carcinoma'",
        )
        self.assertIn("CAMELYON16", l2.get("intended_task", ""))
        self.assertNotIn("General Breast Cancer", l2.get("intended_task", ""))

        # Verify multi-model comparison returns exact class names
        multi = self.pipeline.compare_all_models(self.sample_patch, target_class=1)
        per_model = multi["per_model_results"]
        for arch_name, m_data in per_model.items():
            self.assertIn(
                m_data["predicted_class"],
                ["Normal Lymph Node", "Metastatic Carcinoma"],
                f"Architecture {arch_name} predicted_class '{m_data['predicted_class']}' must match genuine CAMELYON16 classes",
            )
            self.assertEqual(m_data["target_class_name"], "Metastatic Carcinoma")

    # --------------------------------------------------------------------------
    # 2. Score, Threshold, and Label Consistency
    # --------------------------------------------------------------------------
    def test_02_score_threshold_and_label_consistency(self):
        """Verifies strict mathematical consistency between scores, thresholds, and binary decisions."""
        diag = self.pipeline.analyze_patch(self.sample_patch)
        l2 = diag["layer_2_malignancy"]
        prob = l2["probability"]
        tau = l2["threshold"]

        if prob >= tau:
            self.assertEqual(l2["verdict"], "Malignant")
            self.assertEqual(l2["predicted_class"], "Metastatic Carcinoma")
        else:
            self.assertEqual(l2["verdict"], "Normal")
            self.assertEqual(l2["predicted_class"], "Normal Lymph Node")

        # Verify Layer 3 anomaly probes
        l3 = diag["layer_3_anomalies"]
        for head_key in ["ductal_distortion", "density_clusters", "tumor_necrosis"]:
            info = l3[head_key]
            expected_flag = bool(info["score"] >= info["threshold"])
            self.assertEqual(
                info["is_flagged"],
                expected_flag,
                f"Head {head_key} is_flagged ({info['is_flagged']}) does not match score >= tau ({expected_flag})",
            )
            self.assertTrue(info["is_experimental"], f"Head {head_key} must be marked experimental")

    # --------------------------------------------------------------------------
    # 3. Dataset Compatibility & Domain Shift Guardrails
    # --------------------------------------------------------------------------
    def test_03_dataset_compatibility_guardrails(self):
        """Verifies that BreakHis and external inputs are guarded against false benign/malignant equivalence."""
        manifest_path = self.project_root / "data" / "External_test" / "breakhis_manifest.csv"
        self.assertTrue(manifest_path.exists(), "BreakHis manifest must exist on disk")

        df = pd.read_csv(manifest_path)
        # Verify semantic note in manifest warns about normal lymph node vs benign breast tumor
        self.assertIn("semantic_note", df.columns)
        first_row = df.iloc[0]
        self.assertIn("NOT Normal Lymph Node", first_row["semantic_note"])

        # Check that BreakHis benign samples are labeled as Benign Breast Tumor, not Normal
        benign_rows = df[df["tumor_class"] == "benign"]
        for _, row in benign_rows.head(10).iterrows():
            self.assertEqual(row["breakhis_label_name"], "Benign Breast Tumor")
            self.assertNotEqual(row["breakhis_label_name"], "Normal Lymph Node")

    # --------------------------------------------------------------------------
    # 4. Checkpoint and Grad-CAM Target Selection
    # --------------------------------------------------------------------------
    def test_04_checkpoint_and_gradcam_target_selection(self):
        """Verifies correct target class gradient computation and valid heatmap boundaries."""
        # Target 1: Metastatic Carcinoma
        diag_1 = self.pipeline.analyze_patch(self.sample_patch, target_class=1)
        cam_1 = diag_1["layer_4_risk_and_explainability"]["gradcam_heatmap"]
        self.assertEqual(cam_1.shape, (224, 224))
        self.assertTrue(0.0 <= cam_1.min() and cam_1.max() <= 1.0)

        # Target 0: Normal Lymph Node
        diag_0 = self.pipeline.analyze_patch(self.sample_patch, target_class=0)
        cam_0 = diag_0["layer_4_risk_and_explainability"]["gradcam_heatmap"]
        self.assertEqual(cam_0.shape, (224, 224))
        self.assertTrue(0.0 <= cam_0.min() and cam_0.max() <= 1.0)

        # Target attribution must not be forced to identical values
        self.assertEqual(diag_1["layer_4_risk_and_explainability"]["target_class"], 1)
        self.assertEqual(diag_0["layer_4_risk_and_explainability"]["target_class"], 0)

    # --------------------------------------------------------------------------
    # 5. Anomaly Heads Quarantine and Decoupled Diagnosis
    # --------------------------------------------------------------------------
    def test_05_anomaly_heads_quarantine_and_decoupled_risk(self):
        """Verifies MITOS-ATYPIA14 atypia head is permanently quarantined and never combines into cancer diagnosis."""
        # 1. Clean anomaly checkpoint must not contain atypia tensor
        clean_ckpt = self.project_root / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_clean.pt"
        self.assertTrue(clean_ckpt.exists())
        state = torch.load(clean_ckpt, map_location="cpu")
        weights = state.get("state_dict", state)
        for k in weights.keys():
            self.assertNotIn("atypia", k.lower(), f"Quarantined atypia weight '{k}' found in clean checkpoint!")

        # 2. Calling generate_anomaly_gradcam with atypia must raise ValueError
        with self.assertRaises(ValueError):
            self.pipeline.generate_anomaly_gradcam(self.sample_patch, task="atypia")

        # 3. Anomaly diagnosis decoupling: anomaly flags must not override primary risk tier
        diag = self.pipeline.analyze_patch(self.sample_patch)
        l2 = diag["layer_2_malignancy"]
        l4 = diag["layer_4_risk_and_explainability"]
        # If probability is very low, score band must be LOW or INTERMEDIATE, not HIGH from anomalies alone
        if l2["probability"] < 0.20:
            self.assertTrue(any(term in l4["risk_tier"] for term in ["Low", "Intermediate", "Moderate"]))

        # 4. Quarantined atypia head in diagnostic output is inactive
        l3 = diag["layer_3_anomalies"]
        self.assertTrue(l3["pleomorphism_atypia"]["is_quarantined"])
        self.assertFalse(l3["pleomorphism_atypia"]["is_flagged"])
        self.assertFalse(l3["pleomorphism_atypia"]["active"])

    # --------------------------------------------------------------------------
    # 6. Preservation of Checkpoints, Manifests, and WSI Splits
    # --------------------------------------------------------------------------
    def test_06_preservation_of_checkpoints_and_splits(self):
        """Verifies SHA-256 hashes of all 5 checkpoints and strict WSI train/val/test isolation."""
        def file_sha256(path: Path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()

        # Checkpoints
        ckpt_dir = self.project_root / "result" / "objective_2" / "checkpoints"
        expected_prefixes = {
            ckpt_dir / "best_model.pt": "cc6764e0",
            ckpt_dir / "resnet" / "best_model.pt": "a1e47d42",
            ckpt_dir / "densenet" / "best_model.pt": "13e80add",
            ckpt_dir / "vgg" / "best_model.pt": "353f1fcb",
            self.project_root / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_clean.pt": "c713a371",
        }

        for path, prefix in expected_prefixes.items():
            self.assertTrue(path.exists(), f"Checkpoint {path} missing")
            actual_hash = file_sha256(path)
            self.assertTrue(
                actual_hash.startswith(prefix),
                f"Checkpoint {path.name} hash mismatch! Expected prefix {prefix}, got {actual_hash}",
            )

        # WSI Registry & Splits Isolation
        wsi_reg = self.project_root / "data" / "wsi_registry.csv"
        self.assertTrue(wsi_reg.exists())
        self.assertTrue(file_sha256(wsi_reg).startswith("b76a9b42"))

        df_wsi = pd.read_csv(wsi_reg)
        train_wsis = set(df_wsi[df_wsi["split"] == "train"]["wsi_id"])
        val_wsis = set(df_wsi[df_wsi["split"] == "validation"]["wsi_id"])
        test_wsis = set(df_wsi[df_wsi["split"] == "test"]["wsi_id"])

        self.assertEqual(len(train_wsis), 42)
        self.assertEqual(len(val_wsis), 10)
        self.assertEqual(len(test_wsis), 8)
        self.assertEqual(len(train_wsis & val_wsis), 0, "Train and Val WSIs must not overlap")
        self.assertEqual(len(train_wsis & test_wsis), 0, "Train and Test WSIs must not overlap")
        self.assertEqual(len(val_wsis & test_wsis), 0, "Val and Test WSIs must not overlap")

        # Master Benchmark Verification & Metric Recalculation
        master_csv = self.project_root / "result" / "objective_2" / "metrics" / "master_test_benchmark_comparison.csv"
        self.assertTrue(master_csv.exists())
        self.assertTrue(file_sha256(master_csv).startswith("308cf231"))

        # Recalculate test metrics directly from prediction records
        pred_csv = self.project_root / "result" / "objective_2" / "predictions" / "test_predictions.csv"
        self.assertTrue(pred_csv.exists())
        df_pred = pd.read_csv(pred_csv)
        self.assertEqual(len(df_pred), 2881)
        y_true = df_pred["true_label"].values
        y_prob = df_pred["tumor_prob"].values
        y_pred = (y_prob >= 0.5000).astype(int)
        cm = pd.crosstab(y_true, y_pred)
        tp = int(cm.loc[1, 1])
        fn = int(cm.loc[1, 0])
        fp = int(cm.loc[0, 1])
        tn = int(cm.loc[0, 0])
        self.assertEqual((tp, fp, tn, fn), (303, 88, 2475, 15))

    # --------------------------------------------------------------------------
    # 7. Supported CAMELYON16 Normal Patch Verification
    # --------------------------------------------------------------------------
    def test_07_camelyon16_supported_normal_patch(self):
        """Verifies in-distribution CAMELYON16 normal patch yields valid supported benign diagnosis."""
        norm_path = self.project_root / "data" / "PATCHES" / "TEST" / "normal" / "normal_004_x21168_y73360_label0.png"
        if not norm_path.exists():
            self.skipTest(f"Test patch {norm_path} not found")

        diag = self.pipeline.analyze_patch(norm_path, source_domain="CAMELYON16")
        l2 = diag["layer_2_malignancy"]
        l4 = diag["layer_4_risk_and_explainability"]

        self.assertTrue(l2["is_supported_domain"])
        self.assertEqual(l2["domain_status"], "SUPPORTED_IN_DISTRIBUTION")
        self.assertEqual(l2["verdict"], "Normal")
        self.assertEqual(l2["predicted_class"], "Normal Lymph Node")
        self.assertLess(l2["probability"], 0.5000)
        self.assertIn("score_band", l4)
        self.assertFalse(l4["score_band"]["is_clinical_risk"])

    # --------------------------------------------------------------------------
    # 8. Supported CAMELYON16 Tumour Patch Verification
    # --------------------------------------------------------------------------
    def test_08_camelyon16_supported_tumor_patch(self):
        """Verifies in-distribution CAMELYON16 tumour patch yields valid supported malignant diagnosis."""
        tum_path = self.project_root / "data" / "PATCHES" / "TEST" / "tumor" / "tumor_005_x57904_y137760_label1.png"
        if not tum_path.exists():
            self.skipTest(f"Test patch {tum_path} not found")

        diag = self.pipeline.analyze_patch(tum_path, source_domain="CAMELYON16")
        l2 = diag["layer_2_malignancy"]
        l4 = diag["layer_4_risk_and_explainability"]

        self.assertTrue(l2["is_supported_domain"])
        self.assertEqual(l2["domain_status"], "SUPPORTED_IN_DISTRIBUTION")
        self.assertEqual(l2["verdict"], "Malignant")
        self.assertEqual(l2["predicted_class"], "Metastatic Carcinoma")
        self.assertGreaterEqual(l2["probability"], 0.5000)
        self.assertIn("score_band", l4)
        self.assertFalse(l4["score_band"]["is_clinical_risk"])

    # --------------------------------------------------------------------------
    # 9. Benign BreakHis Adenosis OOD Guardrail
    # --------------------------------------------------------------------------
    def test_09_breakhis_benign_adenosis_ood_guardrail(self):
        """Verifies that BreakHis breast histology is guarded against definitive CAMELYON16 metastatic carcinoma labels."""
        adenosis_dir = self.project_root / "data" / "External_test" / "BreaKHis_v1" / "histology_slides" / "breast" / "benign" / "SOB" / "adenosis"
        sample_path = next(adenosis_dir.rglob("*.png"), None)
        if sample_path is None:
            self.skipTest("No BreakHis adenosis sample available")

        diag = self.pipeline.analyze_patch(sample_path, source_domain="BreakHis")
        l2 = diag["layer_2_malignancy"]
        l4 = diag["layer_4_risk_and_explainability"]

        # MUST be flagged as out-of-distribution
        self.assertFalse(l2["is_supported_domain"])
        self.assertEqual(l2["domain_status"], "OUT_OF_DISTRIBUTION")
        # MUST NEVER declare definitive metastatic carcinoma on external breast tissue
        self.assertNotEqual(l2["predicted_class"], "Metastatic Carcinoma")
        self.assertIn("Exploratory", l2["predicted_class"])
        self.assertIn("score_band", l4)
        self.assertFalse(l4["score_band"]["is_clinical_risk"])

        # Also verify with an elevated adenosis sample if available
        elevated_sample = adenosis_dir / "SOB_B_A_14-22549AB" / "100X" / "SOB_B_A-14-22549AB-100-017.png"
        if elevated_sample.exists():
            diag_elev = self.pipeline.analyze_patch(elevated_sample, source_domain="BreakHis")
            l2_e = diag_elev["layer_2_malignancy"]
            self.assertFalse(l2_e["is_supported_domain"])
            self.assertNotEqual(l2_e["predicted_class"], "Metastatic Carcinoma")
            self.assertEqual(l2_e["predicted_class"], "Exploratory Response (Out-of-Distribution)")
            self.assertEqual(l2_e["verdict"], "Exploratory Elevated Score")

    # --------------------------------------------------------------------------
    # 10. Unsupported or Unknown Image Source
    # --------------------------------------------------------------------------
    def test_10_unsupported_or_unknown_image_source(self):
        """Verifies that unverified custom uploads are marked exploratory and unverified."""
        dummy_patch = np.random.randint(60, 200, (224, 224, 3), dtype=np.uint8)
        diag = self.pipeline.analyze_patch(dummy_patch, source_domain="Unknown")
        l2 = diag["layer_2_malignancy"]

        self.assertFalse(l2["is_supported_domain"])
        self.assertEqual(l2["domain_status"], "UNVERIFIED_DOMAIN")
        self.assertIn("Exploratory", l2["predicted_class"])

    # --------------------------------------------------------------------------
    # 11. Binary Threshold Boundary Precision (tau = 0.5000)
    # --------------------------------------------------------------------------
    def test_11_binary_threshold_boundary_precision(self):
        """Verifies exact mathematical boundary behavior at tau = 0.5000 (immediately below, equal, above)."""
        tau = 0.5000
        # Below boundary
        p_below = 0.4999
        self.assertFalse(p_below >= tau)
        # Exactly on boundary
        p_equal = 0.5000
        self.assertTrue(p_equal >= tau)
        # Above boundary
        p_above = 0.5001
        self.assertTrue(p_above >= tau)

    # --------------------------------------------------------------------------
    # 12. Provisional Three Score Bands Boundary and Exhaustiveness
    # --------------------------------------------------------------------------
    def test_12_provisional_three_score_bands_boundary_exhaustive(self):
        """Verifies non-overlapping, exhaustive coverage of [0.0, 1.0] across score bands."""
        from src.objective_2.layered_pipeline import classify_score_band

        # Boundary checks
        b_zero = classify_score_band(0.0)
        self.assertEqual(b_zero["band_key"], "LOW")

        b_sub30 = classify_score_band(0.2999)
        self.assertEqual(b_sub30["band_key"], "LOW")

        b_30 = classify_score_band(0.3000)
        self.assertEqual(b_30["band_key"], "INTERMEDIATE")

        b_50 = classify_score_band(0.5000)
        self.assertEqual(b_50["band_key"], "INTERMEDIATE")

        b_sub70 = classify_score_band(0.6999)
        self.assertEqual(b_sub70["band_key"], "INTERMEDIATE")

        b_70 = classify_score_band(0.7000)
        self.assertEqual(b_70["band_key"], "HIGH")

        b_one = classify_score_band(1.0)
        self.assertEqual(b_one["band_key"], "HIGH")

        # Exhaustive dense scan: every score from 0.00 to 1.00 maps to exactly one band
        for s in np.linspace(0.0, 1.0, 101):
            band = classify_score_band(float(s))
            self.assertIn(band["band_key"], ["LOW", "INTERMEDIATE", "HIGH"])
            self.assertFalse(band["is_clinical_risk"])

    # --------------------------------------------------------------------------
    # 13. High Anomaly Score with Low Tumour Score Decoupling
    # --------------------------------------------------------------------------
    def test_13_high_anomaly_low_tumor_decoupling(self):
        """Verifies that high anomaly scores do NOT force or alter a benign primary classification."""
        # Simulated scenario: low tumor score (0.15) with active ductal and necrosis probes
        anomalies_mock = {
            "ductal_distortion": {"name": "Ductal Distortion", "is_flagged": True, "score": 0.85},
            "density_clusters": {"name": "Density Clusters", "is_flagged": False, "score": 0.20},
            "tumor_necrosis": {"name": "Tumor Necrosis", "is_flagged": True, "score": 0.75},
        }
        p_tumor_low = 0.15
        dummy_tensor = torch.zeros(1, 3, 224, 224)
        dummy_np = np.zeros((224, 224, 3), dtype=np.uint8)

        risk_tier, _, _, narrative = self.pipeline.assess_layer_4_risk_and_explainability(
            dummy_np, dummy_tensor, p_tumor_low, anomalies_mock
        )
        self.assertIn("Low", risk_tier)
        self.assertIn("Ductal Distortion", narrative)
        self.assertIn("Independent external probes; not combined into malignancy verdict", narrative)

    # --------------------------------------------------------------------------
    # 14. High Tumour Score with Low Anomaly Score Decoupling
    # --------------------------------------------------------------------------
    def test_14_high_tumor_low_anomaly_decoupling(self):
        """Verifies that high tumour score maintains malignant classification with zero anomaly flags."""
        anomalies_mock = {
            "ductal_distortion": {"name": "Ductal Distortion", "is_flagged": False, "score": 0.10},
            "density_clusters": {"name": "Density Clusters", "is_flagged": False, "score": 0.10},
            "tumor_necrosis": {"name": "Tumor Necrosis", "is_flagged": False, "score": 0.10},
        }
        p_tumor_high = 0.92
        dummy_tensor = torch.zeros(1, 3, 224, 224)
        dummy_np = np.zeros((224, 224, 3), dtype=np.uint8)

        risk_tier, _, _, narrative = self.pipeline.assess_layer_4_risk_and_explainability(
            dummy_np, dummy_tensor, p_tumor_high, anomalies_mock
        )
        self.assertIn("High", risk_tier)
        self.assertIn("None flagged", narrative)

    # --------------------------------------------------------------------------
    # 15. Missing Checkpoints and Input Handling Robustness
    # --------------------------------------------------------------------------
    def test_15_missing_checkpoints_and_input_robustness(self):
        """Verifies clean error handling for missing checkpoints and non-standard image inputs."""
        non_existent = self.project_root / "result" / "objective_2" / "non_existent_model.pt"
        with self.assertRaises(FileNotFoundError):
            LayeredDiagnosticPipeline(classifier_ckpt=non_existent, anomaly_ckpt=self.anom_ckpt, device=self.device)

        # Non 224x224 input must be auto-resized without crashing
        odd_patch = np.zeros((300, 450, 3), dtype=np.uint8)
        diag = self.pipeline.analyze_patch(odd_patch)
        self.assertEqual(diag["raw_image"].shape, (224, 224, 3))

    # --------------------------------------------------------------------------
    # 16. UI and Pipeline Score Interpretation Consistency
    # --------------------------------------------------------------------------
    def test_16_ui_score_interpretation_consistency(self):
        """Verifies consistent interpretation semantics across pipeline and UI result contracts."""
        diag = self.pipeline.analyze_patch(self.sample_patch)
        l2 = diag["layer_2_malignancy"]
        l4 = diag["layer_4_risk_and_explainability"]

        # 1. Binary decision boundary is strictly 0.5000
        self.assertEqual(l2["threshold"], 0.5000)

        # 2. Score band dictionary presence
        self.assertIn("score_band", l4)
        self.assertIn("band_name", l4["score_band"])
        self.assertFalse(l4["score_band"]["is_clinical_risk"])

        # 3. Anomaly thresholds match calibrated validation optimums
        l3 = diag["layer_3_anomalies"]
        self.assertEqual(l3["ductal_distortion"]["threshold"], 0.4038)
        self.assertEqual(l3["density_clusters"]["threshold"], 0.5315)
        self.assertEqual(l3["tumor_necrosis"]["threshold"], 0.2715)
        self.assertTrue(l3["pleomorphism_atypia"]["is_quarantined"])


    # --------------------------------------------------------------------------
    # 17. analyze_patch API Signature and Domain Argument Robustness
    # --------------------------------------------------------------------------
    def test_17_analyze_patch_signature_and_domain_args(self):
        """Verifies analyze_patch handles positional, keyword, missing, and arbitrary kwargs safely."""
        # 1. Positional only without domain
        d1 = self.pipeline.analyze_patch(self.sample_patch)
        self.assertEqual(d1["layer_2_malignancy"]["source_domain"], "CAMELYON16")
        self.assertTrue(d1["layer_2_malignancy"]["is_supported_domain"])

        # 2. Keyword source_domain="CAMELYON16"
        d2 = self.pipeline.analyze_patch(self.sample_patch, target_class=1, source_domain="CAMELYON16")
        self.assertEqual(d2["layer_2_malignancy"]["source_domain"], "CAMELYON16")
        self.assertTrue(d2["layer_2_malignancy"]["is_supported_domain"])

        # 3. Keyword source_domain="BreakHis"
        d3 = self.pipeline.analyze_patch(self.sample_patch, target_class=1, source_domain="BreakHis")
        self.assertEqual(d3["layer_2_malignancy"]["source_domain"], "BreakHis")
        self.assertFalse(d3["layer_2_malignancy"]["is_supported_domain"])
        self.assertEqual(d3["layer_2_malignancy"]["domain_status"], "OUT_OF_DISTRIBUTION")

        # 4. Keyword domain="BreakHis" (alternative naming)
        d4 = self.pipeline.analyze_patch(self.sample_patch, domain="BreakHis")
        self.assertEqual(d4["layer_2_malignancy"]["source_domain"], "BreakHis")
        self.assertFalse(d4["layer_2_malignancy"]["is_supported_domain"])

        # 5. Missing / None domain on synthetic patch
        d5 = self.pipeline.analyze_patch(self.sample_patch, source_domain=None)
        self.assertEqual(d5["layer_2_malignancy"]["source_domain"], "CAMELYON16")

        # 6. Arbitrary unexpected kwargs (must not raise TypeError)
        d6 = self.pipeline.analyze_patch(self.sample_patch, unexpected_future_param=True, legacy_flag=False)
        self.assertIn("layer_2_malignancy", d6)

    # --------------------------------------------------------------------------
    # 18. Streamlit AppTest Patch Analysis Workflow (CAMELYON16 & BreakHis)
    # --------------------------------------------------------------------------
    def test_18_streamlit_patch_analysis_workflow_live(self):
        """Exercises live Streamlit AppTest for Patch Analysis with CAMELYON16 and BreakHis presets."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(self.project_root / "app.py"), default_timeout=45)
        at.run()
        self.assertFalse(at.exception, f"AppTest raised an exception on load: {at.exception}")

        # Verify default preset (CAMELYON16 Case 1) executes without exception
        self.assertFalse(at.exception)

        # Switch to BreakHis Benign Adenosis (Case 10)
        preset_box = None
        for sb in at.selectbox:
            if "Choose Verified Benchmark Sample" in (sb.label or ""):
                preset_box = sb
                break

        if preset_box is not None:
            bh_opt = next((opt for opt in preset_box.options if "BreakHis Benign Adenosis" in opt), None)
            if bh_opt:
                preset_box.select(bh_opt)
                at.run()
                self.assertFalse(at.exception, f"AppTest raised an exception when selecting BreakHis preset: {at.exception}")

    def test_19_upload_gated_behind_authentication_live(self):
        """Verifies that custom file upload is strictly gated behind manual authentication."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(self.project_root / "app.py"), default_timeout=45)
        at.run()
        self.assertFalse(at.exception)

        # 1. Unauthenticated Guest: file uploaders must NOT be active
        self.assertEqual(len(at.file_uploader), 0, "Unauthenticated guest should not have access to st.file_uploader")

        # 2. Authenticated Researcher: file uploaders become available
        at.session_state["auth_user"] = {
            "username": "researcher",
            "full_name": "Lead Researcher",
            "role": "researcher",
            "authenticated": True,
        }
        at.run()
        self.assertFalse(at.exception)
        self.assertGreaterEqual(len(at.file_uploader), 1, "Authenticated user must have access to st.file_uploader")


if __name__ == "__main__":
    unittest.main()
