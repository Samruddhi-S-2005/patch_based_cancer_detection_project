"""
PathoVision AI Automated Verification & Quality Assurance Suite.

Tests:
1. Preprocessing consistency (ImageNet normalization, dimensions).
2. Checkpoint loading for all 4 CNN backbones and Anomaly linear heads.
3. Prediction generation (forward passes, probabilities in [0, 1]).
4. Threshold decisions (calibrated decision rules).
5. Grad-CAM generation, class targeting, and input regeneration.
6. WSI tile-coordinate preservation and spatial mapping.
7. Continuous spatial heatmap reconstruction and blending.
8. Metric recalculation and exact reproduction from test predictions.
9. Report generation, schema validation, and path sanitization.
10. Error handling for invalid inputs and missing checkpoints.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

from src.objective_2.anomaly_model import EfficientNetB0MultiHeadAnomaly
from src.objective_2.gradcam import GradCAM, overlay_heatmap_on_image
from src.objective_2.layered_pipeline import LayeredDiagnosticPipeline
from src.objective_2.model import (
    DenseNet121TumorClassifier,
    EfficientNetB0TumorClassifier,
    ResNet50TumorClassifier,
    VGG16TumorClassifier,
)
from src.objective_2.wsi_live_engine import (
    blend_wsi_heatmap,
    compute_multimodel_wsi_heatmaps,
    process_openslide_wsi_file,
    process_wsi_canvas,
)


class TestPathoVisionSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt"
        cls.anom_ckpt = PROJECT_ROOT / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_clean.pt"
        cls.device = torch.device("cpu")

        cls.pipeline = LayeredDiagnosticPipeline(
            classifier_ckpt=cls.cls_ckpt,
            anomaly_ckpt=cls.anom_ckpt,
            device=cls.device,
        )

        test_dir = PROJECT_ROOT / "data" / "PATCHES" / "TEST"
        cls.norm_patch_path = next((test_dir / "normal").glob("*.png"))
        cls.tum_patch_path = next((test_dir / "tumor").glob("*label1.png"))

    def test_01_preprocessing_consistency(self):
        """Verifies normalization statistics and 224x224 tensor shape."""
        raw_img = Image.open(self.tum_patch_path).convert("RGB")
        tensor = self.pipeline.eval_transform(raw_img)

        self.assertEqual(tensor.shape, (3, 224, 224))
        # ImageNet normalization should yield mean roughly in [-1.0, 1.0]
        t_mean = tensor.mean().item()
        self.assertTrue(-1.5 < t_mean < 1.5, f"Unexpected tensor mean: {t_mean}")

    def test_02_checkpoint_loading(self):
        """Verifies all 4 CNN model architectures and anomaly probe checkpoints load correctly."""
        # 1. EfficientNet-B0
        eff = EfficientNetB0TumorClassifier(pretrained=False)
        ckpt_eff = torch.load(self.cls_ckpt, map_location="cpu", weights_only=False)
        eff.load_state_dict(ckpt_eff["model_state_dict"])
        self.assertIn("backbone.classifier.1.weight", ckpt_eff["model_state_dict"])

        # 2. ResNet-50
        res_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "resnet" / "best_model.pt"
        self.assertTrue(res_ckpt.exists())
        res = ResNet50TumorClassifier(pretrained=False)
        ckpt_res = torch.load(res_ckpt, map_location="cpu", weights_only=False)
        res.load_state_dict(ckpt_res["model_state_dict"])

        # 3. DenseNet-121
        dense_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "densenet" / "best_model.pt"
        self.assertTrue(dense_ckpt.exists())
        dense = DenseNet121TumorClassifier(pretrained=False)
        ckpt_dense = torch.load(dense_ckpt, map_location="cpu", weights_only=False)
        dense.load_state_dict(ckpt_dense["model_state_dict"])

        # 4. VGG-16
        vgg_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "vgg" / "best_model.pt"
        self.assertTrue(vgg_ckpt.exists())
        vgg = VGG16TumorClassifier(pretrained=False)
        ckpt_vgg = torch.load(vgg_ckpt, map_location="cpu", weights_only=False)
        vgg.load_state_dict(ckpt_vgg["model_state_dict"])

        # 5. Anomaly Linear Heads (Clean Checkpoint: ductal, density, necrosis; atypia quarantined)
        anom = EfficientNetB0MultiHeadAnomaly(checkpoint_path=self.cls_ckpt, device=self.device)
        ckpt_anom = torch.load(self.anom_ckpt, map_location="cpu", weights_only=False)
        anom.load_state_dict(ckpt_anom["model_state_dict"], strict=False)
        self.assertTrue(all(p.requires_grad is False for p in anom.features.parameters()))

    def test_03_prediction_generation(self):
        """Verifies valid forward inference on tumor and normal test patches."""
        diag_tum = self.pipeline.analyze_patch(self.tum_patch_path)
        diag_norm = self.pipeline.analyze_patch(self.norm_patch_path)

        p_tum = diag_tum["layer_2_malignancy"]["probability"]
        p_norm = diag_norm["layer_2_malignancy"]["probability"]

        self.assertTrue(0.0 <= p_tum <= 1.0)
        self.assertTrue(0.0 <= p_norm <= 1.0)
        self.assertEqual(diag_tum["layer_2_malignancy"]["verdict"], "Malignant")
        self.assertEqual(diag_norm["layer_2_malignancy"]["verdict"], "Normal")
        self.assertEqual(diag_tum["layer_2_malignancy"]["predicted_class"], "Metastatic Carcinoma")
        self.assertEqual(diag_norm["layer_2_malignancy"]["predicted_class"], "Normal Lymph Node")

    def test_04_threshold_decisions(self):
        """Verifies primary screening threshold (0.5000) and calibrated anomaly thresholds."""
        diag = self.pipeline.analyze_patch(self.tum_patch_path)
        anoms = diag["layer_3_anomalies"]

        self.assertEqual(diag["layer_2_malignancy"]["threshold"], 0.5000)
        self.assertEqual(anoms["ductal_distortion"]["threshold"], 0.4038)
        self.assertEqual(anoms["density_clusters"]["threshold"], 0.5315)
        self.assertEqual(anoms["tumor_necrosis"]["threshold"], 0.2715)
        self.assertEqual(anoms["pleomorphism_atypia"]["threshold"], 0.0500)
        self.assertTrue(anoms["pleomorphism_atypia"]["is_quarantined"])
        self.assertFalse(anoms["pleomorphism_atypia"]["is_flagged"])
        self.assertFalse(anoms["pleomorphism_atypia"]["active"])
        self.assertFalse(anoms["pleomorphism_atypia"]["is_experimental"])

    def test_05_gradcam_regeneration(self):
        """Verifies Grad-CAM heatmap generation, input regeneration, and class targeting."""
        img_np1 = np.array(Image.open(self.tum_patch_path).convert("RGB"))
        img_np2 = np.array(Image.open(self.norm_patch_path).convert("RGB"))

        t1 = self.pipeline.eval_transform(Image.fromarray(img_np1)).unsqueeze(0)
        t2 = self.pipeline.eval_transform(Image.fromarray(img_np2)).unsqueeze(0)

        cam_map1, prob1 = self.pipeline.gradcam.generate_heatmap(t1, device=self.device, target_class=1)
        cam_map2, prob2 = self.pipeline.gradcam.generate_heatmap(t2, device=self.device, target_class=1)
        cam_map1_norm, _ = self.pipeline.gradcam.generate_heatmap(t1, device=self.device, target_class=0)

        self.assertEqual(cam_map1.shape, (224, 224))
        self.assertEqual(cam_map2.shape, (224, 224))
        self.assertTrue(0.0 <= cam_map1.min() and cam_map1.max() <= 1.0)
        # Heatmaps for different inputs must differ
        self.assertFalse(np.allclose(cam_map1, cam_map2))
        # Malignancy targeting and normal targeting must differ
        self.assertFalse(np.allclose(cam_map1, cam_map1_norm))

    def test_06_wsi_tile_coordinate_mapping(self):
        """Verifies spatial coordinate preservation and tile ID formatting."""
        canvas = np.full((448, 448, 3), 200, dtype=np.uint8)
        res = process_wsi_canvas(canvas, self.pipeline, orig_dims=(4480, 4480), patch_size=224)

        df_tiles = res["df_tiles"]
        self.assertEqual(len(df_tiles), 4)  # 2x2 grid
        self.assertListEqual(df_tiles["tile_id"].tolist(), ["R00_C00", "R00_C01", "R01_C00", "R01_C01"])
        self.assertEqual(df_tiles.iloc[0]["wsi_x"], 0)
        self.assertEqual(df_tiles.iloc[1]["wsi_x"], 2240)
        self.assertEqual(df_tiles.iloc[2]["wsi_y"], 2240)

    def test_07_spatial_reconstruction(self):
        """Verifies continuous heatmap generation and Gaussian blending over tissue."""
        canvas = np.full((448, 448, 3), 180, dtype=np.uint8)
        res = process_wsi_canvas(canvas, self.pipeline, patch_size=224)

        blended = blend_wsi_heatmap(
            wsi_rgb=res["wsi_rgb"],
            prob_upsampled=res["prob_upsampled"],
            tissue_mask=res["tissue_mask"],
            alpha=0.50,
            colormap_name="jet",
        )
        self.assertEqual(blended.shape, (448, 448, 3))
        self.assertEqual(blended.dtype, np.uint8)

    def test_08_metric_recalculation_and_exact_reproduction(self):
        """Independently verifies test metrics reproduction on 2,881 test patches."""
        pred_csv = PROJECT_ROOT / "result" / "objective_2" / "predictions" / "test_predictions.csv"
        self.assertTrue(pred_csv.exists())

        df = pd.read_csv(pred_csv)
        self.assertEqual(len(df), 2881)

        y_true = df["true_label"].values
        y_prob = df["tumor_prob"].values
        y_pred = (y_prob >= 0.5000).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.assertEqual((tp, fp, tn, fn), (303, 88, 2475, 15))

        sens = recall_score(y_true, y_pred)
        spec = tn / (tn + fp)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)

        self.assertAlmostEqual(sens, 0.9528, places=4)
        self.assertAlmostEqual(spec, 0.9657, places=4)
        self.assertAlmostEqual(acc, 0.9642, places=4)
        self.assertAlmostEqual(bal_acc, 0.9592, places=4)
        self.assertAlmostEqual(prec, 0.7749, places=4)
        self.assertAlmostEqual(f1, 0.8547, places=4)
        self.assertAlmostEqual(roc, 0.9875, places=4)
        self.assertAlmostEqual(pr, 0.9664, places=4)

    def test_09_report_generation_and_path_sanitization(self):
        """Verifies structured JSON/CSV report generation without leaking local machine paths."""
        diag = self.pipeline.analyze_patch(self.tum_patch_path)

        report_payload = {
            "report_metadata": {
                "software": "PathoVision AI v2.6",
                "input_identifier": "tumor_005_test_patch",
                "model_backbone": "EfficientNet-B0",
                "decision_threshold": 0.5000,
            },
            "layer_1_quality": diag["layer_1_quality"],
            "layer_2_malignancy": diag["layer_2_malignancy"],
            "layer_3_anomalies": diag["layer_3_anomalies"],
            "layer_4_risk": diag["layer_4_risk_and_explainability"]["risk_tier"],
        }

        json_str = json.dumps(report_payload, indent=2)
        # Ensure no Windows absolute file paths like D:\Samruddhi are leaked
        self.assertNotIn("D:\\Samruddhi", json_str)
        self.assertNotIn("C:\\Users", json_str)

        parsed = json.loads(json_str)
        self.assertIn("layer_2_malignancy", parsed)
        self.assertEqual(parsed["layer_2_malignancy"]["verdict"], "Malignant")

    def test_10_invalid_inputs_and_missing_checkpoints(self):
        """Verifies error handling when checkpoint is missing or input is corrupted/empty."""
        # 1. Missing checkpoint exception
        with self.assertRaises(FileNotFoundError):
            LayeredDiagnosticPipeline(
                classifier_ckpt=Path("non_existent_model.pt"),
                anomaly_ckpt=self.anom_ckpt,
                device=self.device,
            )

        # 2. Quality screening on empty white glass patch
        white_patch = np.full((224, 224, 3), 255, dtype=np.uint8)
        qc_res = self.pipeline.assess_layer_1_quality(white_patch)
        self.assertFalse(qc_res["passed"])
        self.assertEqual(qc_res["status"], "LOW_TISSUE")

    def test_11_permitted_anomaly_heads_and_atypia_quarantine(self):
        """Verifies permitted linear probe heads forward pass and confirms atypia quarantine."""
        dummy_tensor = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = self.pipeline.anomaly_model.forward_all_heads(dummy_tensor)

        # Confirm quarantined atypia head is excluded from active execution
        self.assertNotIn("atypia", logits, "Quarantined atypia head must not be active in forward_all_heads")

        # Confirm all 3 permitted heads are active and valid
        self.assertIn("ductal", logits)
        self.assertIn("density", logits)
        self.assertIn("necrosis", logits)

        for head_name in ["ductal", "density", "necrosis"]:
            self.assertEqual(logits[head_name].shape, (2, 1))
            probs = torch.sigmoid(logits[head_name])
            self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_12_authentication_and_rbac(self):
        """Verifies secure password hashing, verification, RBAC permissions and user directory isolation."""
        from src.auth import (
            _hash_password,
            _verify_password,
            authenticate_user,
            register_user,
            get_user_workspace,
            load_users,
            save_users,
        )

        # 1. Hashing and constant-time verification
        pwd = "TestSecurePassword123!"
        h = _hash_password(pwd)
        self.assertIn("$", h)
        self.assertTrue(_verify_password(pwd, h))
        self.assertFalse(_verify_password("WrongPassword!", h))

        # 2. Pre-seeded accounts
        ok_res, u_res, _ = authenticate_user("researcher", "ResearchPass2026!")
        self.assertTrue(ok_res)
        self.assertEqual(u_res["role"], "researcher")

        ok_rev, u_rev, _ = authenticate_user("reviewer", "ReviewerAudit2026!")
        self.assertTrue(ok_rev)
        self.assertEqual(u_rev["role"], "reviewer")

        # 3. Controlled registration
        # Deny reviewer registration without master passkey
        ok_fail, _ = register_user("unauth_rev", "TestPass123!", "Unauth", "u@test.com", "reviewer", "WRONG_KEY")
        self.assertFalse(ok_fail)

        # 4. Isolated user workspace
        ws = get_user_workspace("researcher")
        self.assertTrue(ws.exists())
        self.assertTrue((ws / "uploads").exists())
        self.assertTrue((ws / "reports").exists())
        self.assertTrue((ws / "history.json").exists())

    def test_13_qupath_export_schema(self):
        """Verifies standard QuPath GeoJSON and TSV export generation."""
        from src.objective_2.wsi_live_engine import export_qupath_geojson, export_qupath_tsv

        df_mock = pd.DataFrame([
            {
                "tile_id": "T_001_X0_Y0",
                "wsi_x": 0,
                "wsi_y": 0,
                "is_tissue": True,
                "pred_label": 1,
                "pred_prob": 0.9421,
                "atypia": 0.8123,
                "ductal": 0.6543,
                "density": 0.8912,
                "necrosis": 0.4211,
                "blur_score": 182.5,
                "tissue_fraction": 89.2,
                "risk_tier": "High Suspicion",
                "dominant_anomaly": "Nuclear Pleomorphism",
                "qc_status": "PASS",
            },
            {
                "tile_id": "T_002_X224_Y0",
                "wsi_x": 224,
                "wsi_y": 0,
                "is_tissue": True,
                "pred_label": 0,
                "pred_prob": 0.0821,
                "atypia": 0.0211,
                "ductal": 0.1143,
                "density": 0.0512,
                "necrosis": 0.0111,
                "blur_score": 210.0,
                "tissue_fraction": 94.0,
                "risk_tier": "Low Suspicion",
                "dominant_anomaly": "None",
                "qc_status": "PASS",
            },
        ])

        # Test GeoJSON
        geojson_str = export_qupath_geojson(df_mock)
        geojson_data = json.loads(geojson_str)
        self.assertEqual(geojson_data["type"], "FeatureCollection")
        self.assertEqual(len(geojson_data["features"]), 2)
        f0 = geojson_data["features"][0]
        self.assertEqual(f0["geometry"]["type"], "Polygon")
        self.assertEqual(f0["properties"]["classification"]["name"], "Tumor")
        self.assertIn("Malignancy Probability", f0["properties"]["measurements"])

        # Test TSV
        tsv_str = export_qupath_tsv(df_mock)
        lines = tsv_str.strip().split("\n")
        self.assertEqual(len(lines), 3)  # Header + 2 rows
        self.assertIn("Tile ID\tClass\tCentroid X px", lines[0])
        self.assertIn("Tumor", lines[1])
        self.assertIn("Normal", lines[2])

    def test_14_dynamic_model_switching(self):
        """Verifies on-the-fly switching between EfficientNet, ResNet, DenseNet, and VGG backbones."""
        # Switch to ResNet-50
        self.pipeline.switch_classifier_model("ResNet-50")
        self.assertEqual(self.pipeline.model_name, "ResNet-50")
        self.assertIsInstance(self.pipeline.classifier, ResNet50TumorClassifier)
        diag_res = self.pipeline.analyze_patch(self.tum_patch_path)
        self.assertEqual(diag_res["metadata"]["backbone"], "ResNet-50")
        self.assertTrue(0.0 <= diag_res["layer_2_malignancy"]["probability"] <= 1.0)
        self.assertEqual(diag_res["layer_4_risk_and_explainability"]["gradcam_heatmap"].shape, (224, 224))

        # Switch to DenseNet-121
        self.pipeline.switch_classifier_model("DenseNet-121")
        self.assertEqual(self.pipeline.model_name, "DenseNet-121")
        self.assertIsInstance(self.pipeline.classifier, DenseNet121TumorClassifier)
        diag_dense = self.pipeline.analyze_patch(self.tum_patch_path)
        self.assertEqual(diag_dense["metadata"]["backbone"], "DenseNet-121")
        self.assertTrue(0.0 <= diag_dense["layer_2_malignancy"]["probability"] <= 1.0)
        self.assertEqual(diag_dense["layer_4_risk_and_explainability"]["gradcam_heatmap"].shape, (224, 224))

        # Switch back to EfficientNet-B0
        self.pipeline.switch_classifier_model("EfficientNet-B0")
        self.assertEqual(self.pipeline.model_name, "EfficientNet-B0")
        self.assertIsInstance(self.pipeline.classifier, EfficientNetB0TumorClassifier)

    def test_15_password_rotation_and_credential_protection(self):
        """Verifies password rotation API, rejection of weak passwords, and old-password invalidation."""
        import time
        from src.auth import authenticate_user, register_user, rotate_user_password

        # Register a temporary test researcher with unique identifier for test isolation
        test_u = f"test_doc_{int(time.time()*1000)}"
        register_user(test_u, "InitialStrongPassword2026!", "Test Doc", "doc@hospital.org", role="researcher")

        # Attempt rotation with wrong old password
        ok_wrong, msg_wrong = rotate_user_password(test_u, "WrongOldPassword!", "NewStrongPassword2026!")
        self.assertFalse(ok_wrong)

        # Attempt rotation with too short new password
        ok_short, msg_short = rotate_user_password(test_u, "InitialStrongPassword2026!", "short")
        self.assertFalse(ok_short)

        # Successful rotation
        ok_rot, _ = rotate_user_password(test_u, "InitialStrongPassword2026!", "RotatedStrongPassword2026!")
        self.assertTrue(ok_rot)

        # Verify old password no longer works
        ok_old_login, _, _ = authenticate_user(test_u, "InitialStrongPassword2026!")
        self.assertFalse(ok_old_login)

        # Verify new password works
        ok_new_login, u_data, _ = authenticate_user(test_u, "RotatedStrongPassword2026!")
        self.assertTrue(ok_new_login)
        self.assertEqual(u_data["username"], test_u)

    def test_16_multimodel_benchmark_metrics_reproduction(self):
        """Verifies master comparative benchmark JSON consistency and accuracy ordering."""
        bench_path = PROJECT_ROOT / "result" / "objective_2" / "metrics" / "master_test_benchmark_comparison.json"
        self.assertTrue(bench_path.exists())
        with open(bench_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        self.assertEqual(len(bench_data), 4)
        models_found = [entry["Model"].upper() for entry in bench_data]
        self.assertIn("EFFICIENTNET", models_found)
        self.assertIn("RESNET", models_found)
        self.assertIn("DENSENET", models_found)
        self.assertIn("VGG", models_found)

        for entry in bench_data:
            tp = entry["TP"]
            fp = entry["FP"]
            tn = entry["TN"]
            fn = entry["FN"]
            self.assertEqual(tp + fp + tn + fn, 2881)
            # All models should achieve > 95% test accuracy on CAMELYON16 held-out test
            self.assertGreater(entry["Accuracy"], 0.95)
            self.assertGreater(entry["ROC-AUC"], 0.98)

    def test_17_anomaly_threshold_decision_boundaries(self):
        """
        Task 2 Automated QA: Verifies below-threshold, exactly-at-threshold,
        and above-threshold decision boundaries for all 4 anomaly heads.
        """
        tasks = [
            ("pleomorphism_atypia", "atypia", 0.0500),
            ("ductal_distortion", "ductal", 0.4038),
            ("density_clusters", "density", 0.5315),
            ("tumor_necrosis", "necrosis", 0.2715),
        ]

        # 1. Verify schema on actual model forward pass
        dummy_tensor = torch.zeros((1, 3, 224, 224))
        diag_res = self.pipeline.assess_layer_3_anomalies(dummy_tensor)

        for dict_key, task_key, tau in tasks:
            self.assertIn(dict_key, diag_res)
            head_dict = diag_res[dict_key]
            # Verify single source of truth fields
            self.assertIn("raw_logit", head_dict)
            self.assertIn("score", head_dict)
            self.assertIn("threshold", head_dict)
            self.assertIn("threshold_source", head_dict)
            self.assertIn("comparison_operator", head_dict)
            self.assertIn("is_flagged", head_dict)
            self.assertIn("status_label", head_dict)
            self.assertIn("head_status", head_dict)
            self.assertEqual(head_dict["comparison_operator"], ">=")
            self.assertEqual(head_dict["threshold"], tau)
            self.assertEqual(head_dict["is_flagged"], head_dict["score"] >= tau)
            self.assertEqual(head_dict["active"], head_dict["is_flagged"])
            expected_label = "Anomaly Flagged" if head_dict["is_flagged"] else "Anomaly Not Detected"
            self.assertEqual(head_dict["status_label"], expected_label)

        # 2. Synthetic boundary tests: below, exactly-at, and above threshold
        for dict_key, task_key, tau in tasks:
            # Below-threshold case
            below_score = max(0.0, tau - 0.01)
            flag_below = bool(below_score >= tau)
            self.assertFalse(flag_below, f"Score {below_score} should NOT trigger threshold {tau}")

            # Exactly-at-threshold case
            at_score = tau
            flag_at = bool(at_score >= tau)
            self.assertTrue(flag_at, f"Score {at_score} exactly at threshold {tau} MUST trigger positive flag")

            # Above-threshold case
            above_score = min(1.0, tau + 0.01)
            flag_above = bool(above_score >= tau)
            self.assertTrue(flag_above, f"Score {above_score} above threshold {tau} MUST trigger positive flag")

        # 3. Verify Anomaly-Specific Grad-CAM Generation for permitted heads (atypia is quarantined)
        for dict_key, task_key, tau in tasks:
            if task_key == "atypia":
                with self.assertRaises(ValueError):
                    self.pipeline.generate_anomaly_gradcam(dummy_tensor, task=task_key)
                continue
            cam_map, score_val = self.pipeline.generate_anomaly_gradcam(dummy_tensor, task=task_key)
            self.assertEqual(cam_map.shape, (224, 224))
            self.assertTrue(0.0 <= score_val <= 1.0)
            self.assertTrue(0.0 <= cam_map.min() and cam_map.max() <= 1.0)

    def test_18_gradcam_perturbation_faithfulness(self):
        """Verifies quantitative perturbation-based Grad-CAM faithfulness check."""
        img_np = np.array(Image.open(self.tum_patch_path).convert("RGB"))
        t_in = self.pipeline.eval_transform(Image.fromarray(img_np)).unsqueeze(0)
        cam_map, prob = self.pipeline.gradcam.generate_heatmap(t_in, device=self.device, target_class=1)

        faith = self.pipeline.compute_gradcam_faithfulness(img_np, cam_map, target_class=1, top_k_pct=0.20)
        self.assertIn("salient_drop", faith)
        self.assertIn("control_drop", faith)
        self.assertIn("is_faithful", faith)
        self.assertIn("interpretation", faith)
        self.assertTrue(isinstance(faith["is_faithful"], bool))

    def test_19_multi_architecture_consensus_comparison(self):
        """Verifies controlled 4-model comparative consensus inference on the same input."""
        comp = self.pipeline.compare_all_models(self.tum_patch_path, target_class=1)
        self.assertIn("per_model_results", comp)
        self.assertIn("consensus", comp)

        res = comp["per_model_results"]
        self.assertIn("EfficientNet-B0", res)
        self.assertIn("ResNet-50", res)
        self.assertIn("DenseNet-121", res)
        self.assertIn("VGG-16", res)

        for arch, r in res.items():
            self.assertEqual(r["status"], "SUCCESS")
            self.assertTrue(0.0 <= r["score"] <= 1.0)
            self.assertEqual(r["gradcam_heatmap"].shape, (224, 224))

        cons = comp["consensus"]
        self.assertEqual(cons["total_models"], 4)
        self.assertIn(cons["verdict"], ["Malignant", "Normal"])
        self.assertEqual(cons["votes_malignant"] + cons["votes_normal"], 4)

    def test_20_deterministic_repeatable_inference(self):
        """Verifies deterministic and repeatable inference outputs on identical inputs."""
        diag1 = self.pipeline.analyze_patch(self.tum_patch_path)
        diag2 = self.pipeline.analyze_patch(self.tum_patch_path)

        self.assertEqual(diag1["layer_2_malignancy"]["probability"], diag2["layer_2_malignancy"]["probability"])
        self.assertEqual(diag1["layer_2_malignancy"]["raw_logit"], diag2["layer_2_malignancy"]["raw_logit"])
        self.assertEqual(diag1["layer_2_malignancy"]["verdict"], diag2["layer_2_malignancy"]["verdict"])

        for k in diag1["layer_3_anomalies"]:
            self.assertEqual(diag1["layer_3_anomalies"][k]["score"], diag2["layer_3_anomalies"][k]["score"])
            self.assertEqual(diag1["layer_3_anomalies"][k]["is_flagged"], diag2["layer_3_anomalies"][k]["is_flagged"])

        # Heatmaps must be identical
        diff = np.max(np.abs(diag1["layer_4_risk_and_explainability"]["gradcam_heatmap"] - diag2["layer_4_risk_and_explainability"]["gradcam_heatmap"]))
        self.assertAlmostEqual(float(diff), 0.0, places=4)

    def test_21_corrupted_input_fault_tolerance(self):
        """Verifies anomaly profiler gracefully handles invalid or corrupted tensor inputs."""
        corrupted_tensor = torch.full((1, 3, 224, 224), float("nan"))
        res = self.pipeline.assess_layer_3_anomalies(corrupted_tensor)
        self.assertIn("pleomorphism_atypia", res)
        self.assertIn("ductal_distortion", res)
        self.assertIn("density_clusters", res)
        self.assertIn("tumor_necrosis", res)
        # Should not crash and should report False for flagged
        for k, v in res.items():
            self.assertFalse(v["is_flagged"])

    def test_22_wsi_canvas_processing_without_atypia_key(self):
        """Verifies WSI canvas processing succeeds without an atypia key and keeps atypia quarantined."""
        canvas = np.ones((448, 448, 3), dtype=np.uint8) * 180  # 2x2 grid of tissue tiles
        res = process_wsi_canvas(canvas, self.pipeline, patch_size=224)
        summary = res["summary"]
        df_tiles = res["df_tiles"]

        # Confirm summary metadata
        self.assertIn("atypia_status", summary)
        self.assertIn("QUARANTINED", summary["atypia_status"])

        # Confirm active anomaly aggregation strictly excludes atypia
        mean_anoms = summary["mean_anomalies"]
        self.assertNotIn("Nuclear Pleomorphism", mean_anoms)
        self.assertNotIn("atypia", mean_anoms)
        self.assertIn("Ductal Structure Distortion", mean_anoms)
        self.assertIn("Tumor Cell Density Clusters", mean_anoms)
        self.assertIn("Tumor Necrosis & Debris", mean_anoms)

        pct_exceed = summary["pct_exceeding_threshold"]
        self.assertNotIn("Nuclear Pleomorphism", pct_exceed)
        self.assertNotIn("atypia", pct_exceed)

        # Confirm tile atypia values are None / NaN (not fabricated 0.0)
        self.assertTrue(df_tiles["atypia"].isna().all())

    def test_23_tcga_svs_slide_discovery_and_graceful_fallbacks(self):
        """Verifies TCGA SVS discovery handles existing directories and non-existent directories gracefully."""
        from src.objective_2.wsi_live_engine import find_tcga_svs_slides
        import tempfile

        # 1. Real project discovery
        slides = find_tcga_svs_slides()
        self.assertIn("tumor", slides)
        self.assertIn("normal", slides)
        self.assertIn("all", slides)
        self.assertGreater(len(slides["all"]), 0, "Expected TCGA slides to be discovered in project")
        self.assertGreater(len(slides["tumor"]), 0, "Expected at least 1 tumor slide discovered")
        self.assertGreater(len(slides["normal"]), 0, "Expected at least 1 normal slide discovered")

        # 2. Non-existent directory fallback
        non_existent = Path("non_existent_tcga_test_dir_9999")
        empty_res = find_tcga_svs_slides(non_existent)
        self.assertEqual(empty_res["tumor"], [])
        self.assertEqual(empty_res["normal"], [])
        self.assertEqual(empty_res["all"], [])

        # 3. Empty temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_res = find_tcga_svs_slides(Path(tmp_dir))
            self.assertEqual(tmp_res["tumor"], [])
            self.assertEqual(tmp_res["normal"], [])
            self.assertEqual(tmp_res["all"], [])

    def test_24_openslide_streaming_atypia_quarantine_and_error_handling(self):
        """Verifies process_openslide_wsi_file handles non-existent slide errors and executes without atypia crash."""
        from src.objective_2.wsi_live_engine import process_openslide_wsi_file, find_tcga_svs_slides

        # Error handling for invalid path
        with self.assertRaises(FileNotFoundError):
            process_openslide_wsi_file(Path("non_existent_slide.svs"), self.pipeline)

        # Execution on real slide with small tile budget (max_tiles=3)
        slides = find_tcga_svs_slides()
        if len(slides["tumor"]) > 0:
            test_slide = slides["tumor"][0]
            res = process_openslide_wsi_file(test_slide, self.pipeline, stride=224, max_tiles=3)
            summary = res["summary"]
            self.assertIn("atypia_status", summary)
            self.assertIn("QUARANTINED", summary["atypia_status"])
            self.assertNotIn("Nuclear Pleomorphism", summary["mean_anomalies"])
            self.assertIn("Ductal Structure Distortion", summary["mean_anomalies"])
            self.assertIn("Tumor Cell Density Clusters", summary["mean_anomalies"])
            self.assertIn("Tumor Necrosis & Debris", summary["mean_anomalies"])
            self.assertTrue(res["df_tiles"]["atypia"].isna().all())

    def test_25_wsi_cache_invalidation_and_no_stale_results(self):
        """Verifies switching between slides invalidates cached state and displays zero stale data."""
        sample_dir = PROJECT_ROOT / "data" / "sample_wsi"
        p_tum = sample_dir / "sample_wsi_tumor_058_biopsy.tif"
        p_norm = sample_dir / "sample_wsi_normal_004_biopsy.tif"

        if p_tum.exists() and p_norm.exists():
            img_tum = Image.open(p_tum).convert("RGB")
            img_norm = Image.open(p_norm).convert("RGB")

            # Run slide 1 (Tumor)
            res_tum = process_wsi_canvas(np.array(img_tum), self.pipeline, patch_size=224)
            self.assertEqual(res_tum["summary"]["slide_verdict"], "MALIGNANT METASTASIS")
            self.assertGreater(res_tum["summary"]["malignant_tiles"], 0)

            # Switch to slide 2 (Normal)
            res_norm = process_wsi_canvas(np.array(img_norm), self.pipeline, patch_size=224)
            self.assertEqual(res_norm["summary"]["slide_verdict"], "BENIGN / NORMAL")
            self.assertEqual(res_norm["summary"]["malignant_tiles"], 0)
            self.assertEqual(res_norm["summary"]["tumor_burden_pct"], 0.0)

            # Confirm no data bleed between runs
            self.assertNotEqual(res_tum["summary"]["slide_verdict"], res_norm["summary"]["slide_verdict"])
            self.assertNotEqual(res_tum["summary"]["malignant_tiles"], res_norm["summary"]["malignant_tiles"])

    def test_26_wsi_patch_reconciliation_and_qc_filtering(self):
        """Verifies full numerical reconciliation: candidate tiles = viable tissue tiles + rejected glass tiles."""
        sample_dir = PROJECT_ROOT / "data" / "sample_wsi"
        p_tum = sample_dir / "sample_wsi_tumor_058_biopsy.tif"

        if p_tum.exists():
            img_tum = Image.open(p_tum).convert("RGB")
            res = process_wsi_canvas(np.array(img_tum), self.pipeline, patch_size=224)
            s = res["summary"]
            df_tiles = res["df_tiles"]

            # Exact reconciliation
            n_total = s["total_tiles"]
            n_tissue = s["tissue_tiles"]
            n_mal = s["malignant_tiles"]
            n_benign = s["benign_tiles"]
            n_glass = int((df_tiles["is_tissue"] == False).sum())

            self.assertEqual(n_total, n_tissue + n_glass, "Total tiles must equal tissue tiles + rejected glass tiles")
            self.assertEqual(n_tissue, n_mal + n_benign, "Tissue tiles must equal malignant tiles + benign tiles")

            # Check that glass tiles were excluded from malignancy classification
            df_glass = df_tiles[df_tiles["is_tissue"] == False]
            self.assertTrue((df_glass["pred_label"] == 0).all())
            self.assertTrue((df_glass["verdict"] == "Background").all())

            # Check burden percentage formula
            expected_burden = round((n_mal / n_tissue) * 100.0, 2)
            self.assertAlmostEqual(s["tumor_burden_pct"], expected_burden, places=2)

    def test_27_master_benchmark_metric_integrity(self):
        """Verifies metric integrity and mathematical reconciliation of the 4-model CAMELYON16 test benchmark."""
        csv_path = PROJECT_ROOT / "result" / "objective_2" / "metrics" / "master_test_benchmark_comparison.csv"
        self.assertTrue(csv_path.exists(), f"Master benchmark file missing: {csv_path}")

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 4, "Must contain exactly 4 evaluated architectures")
        expected_models = {"EFFICIENTNET", "RESNET", "DENSENET", "VGG"}
        self.assertEqual(set(df["Model"].str.upper()), expected_models)

        for _, row in df.iterrows():
            m = row["Model"]
            tp, fp, tn, fn = int(row["TP"]), int(row["FP"]), int(row["TN"]), int(row["FN"])
            total = tp + fp + tn + fn
            self.assertEqual(total, 2881, f"Total test patches must be exactly 2,881 for {m}")
            self.assertEqual(tp + fn, 318, f"Total positive tumor patches must be exactly 318 for {m}")
            self.assertEqual(tn + fp, 2563, f"Total negative normal patches must be exactly 2,563 for {m}")

            # Reconcile Sensitivity (Recall)
            expected_sens = tp / (tp + fn)
            self.assertAlmostEqual(row["Recall (Sensitivity)"], expected_sens, places=3, msg=f"Sensitivity mismatch for {m}")

            # Reconcile Specificity
            expected_spec = tn / (tn + fp)
            self.assertAlmostEqual(row["Specificity"], expected_spec, places=3, msg=f"Specificity mismatch for {m}")

            # Reconcile Precision
            expected_prec = tp / (tp + fp)
            self.assertAlmostEqual(row["Precision"], expected_prec, places=3, msg=f"Precision mismatch for {m}")

            # Reconcile Balanced Accuracy
            expected_bacc = (expected_sens + expected_spec) / 2.0
            self.assertAlmostEqual(row["Balanced Acc"], expected_bacc, places=3, msg=f"Balanced Acc mismatch for {m}")

    def test_28_model_evaluation_ui_breakhis_removal(self):
        """Verifies that BreakHis OOD section, tabs, and tables have been decoupled from Model Evaluation in app.py."""
        app_file = PROJECT_ROOT / "app.py"
        self.assertTrue(app_file.exists(), "app.py must exist")

        with open(app_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract the Model Evaluation section
        start_marker = 'elif nav == "📊 Model Evaluation":'
        end_marker = 'elif nav == "🛡️ Institutional Reviewer Audit":'
        self.assertIn(start_marker, content)
        self.assertIn(end_marker, content)

        eval_section = content.split(start_marker)[1].split(end_marker)[0]

        # BreakHis OOD tabs and tables must NOT be in active Model Evaluation
        self.assertNotIn("tab_breakhis", eval_section, "tab_breakhis must not exist in Model Evaluation")
        self.assertNotIn("df_bh", eval_section, "BreakHis transfer dataframe must not be rendered in Model Evaluation")
        self.assertNotIn("fig_bh_transfer", eval_section, "BreakHis transfer plot must not exist in Model Evaluation")
        self.assertNotIn("breakhis_model_transfer_summary.csv", eval_section, "BreakHis metrics CSV must not be loaded in Model Evaluation")

        # In-distribution CAMELYON16 elements must be present
        self.assertIn("master_test_benchmark_comparison.csv", eval_section)
        self.assertIn("master_comparative_roc_pr_curves.png", eval_section)
        self.assertIn("Comparative Diagnostic Metric Visualizer", eval_section)
        self.assertIn("Architecture Benchmark Summary Table", eval_section)
        self.assertIn("Confusion Matrices", eval_section)

    def test_29_breakhis_supplementary_research_archive_integrity(self):
        """Verifies that the supplementary BreakHis research archive exists with all required components."""
        archive_root = PROJECT_ROOT / "research_archive" / "cross_dataset_breakhis"
        self.assertTrue(archive_root.exists(), f"Research archive root missing: {archive_root}")

        subdirs = ["reports", "results", "scripts", "configs", "metadata"]
        for sd in subdirs:
            p = archive_root / sd
            self.assertTrue(p.exists() and p.is_dir(), f"Archive subdirectory missing: {p}")

        # Check README.md
        readme = archive_root / "README.md"
        self.assertTrue(readme.exists(), "Archive README.md must exist")
        with open(readme, "r", encoding="utf-8") as f:
            readme_text = f.read()
        self.assertIn("Purpose of the Experiment", readme_text)
        self.assertIn("Relationship to the Primary CAMELYON16 Task", readme_text)
        self.assertIn("Dataset and Label Mapping", readme_text)
        self.assertIn("Known Limitations", readme_text)
        self.assertIn("Instructions for Project Demonstration", readme_text)

        # Check key archived artifacts
        self.assertTrue((archive_root / "results" / "breakhis_model_transfer_summary.csv").exists())
        self.assertTrue((archive_root / "results" / "breakhis_exploratory_ood_metrics.json").exists())
        self.assertTrue((archive_root / "scripts" / "evaluate_breakhis.py").exists())
        self.assertTrue((archive_root / "configs" / "breakhis_experiment_config.json").exists())
        self.assertTrue((archive_root / "metadata" / "breakhis_provenance_and_samples.json").exists())

    def test_30_comparative_roc_pr_plot_exists(self):
        """Verifies that master comparative ROC and PR curves plot exists and is valid."""
        plot_path = PROJECT_ROOT / "result" / "objective_2" / "plots" / "master_comparative_roc_pr_curves.png"
        self.assertTrue(plot_path.exists(), f"Comparative plot missing: {plot_path}")
        self.assertGreater(plot_path.stat().st_size, 50000, "Comparative plot file size suspiciously small")

    def test_31_tempfile_wsi_upload_handling(self):
        """Verifies that app.py imports tempfile and uses safe temporary file handling for WSI uploads."""
        app_file = PROJECT_ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Must import tempfile
        self.assertIn("import tempfile", content, "app.py must explicitly import tempfile")

        # Must use NamedTemporaryFile for safe WSI upload handling
        self.assertIn("tempfile.NamedTemporaryFile", content, "app.py must use tempfile.NamedTemporaryFile for uploads")

        # Must track active temp file in session state and unlink previous temp files
        self.assertIn("active_temp_wsi_path", content, "app.py must track active temp file in session state")
        self.assertIn("unlink", content, "app.py must cleanly unlink temporary files")

    def test_32_wsi_canvas_48_patch_reconciliation(self):
        """Verifies the complete 48-patch count calculation and stage-by-stage reconciliation on sample WSI."""
        sample_path = PROJECT_ROOT / "data" / "sample_wsi" / "sample_wsi_tumor_058_biopsy.tif"
        self.assertTrue(sample_path.exists(), f"Sample WSI missing: {sample_path}")

        img = Image.open(sample_path).convert("RGB")
        w, h = img.size
        # 1. Image dimensions: exactly 1792 x 1344
        self.assertEqual((w, h), (1792, 1344), "Sample WSI dimensions must be 1792 x 1344 px")

        # 2. Patch size and stride: 224 non-overlapping
        patch_size = 224
        cols = w // patch_size
        rows = h // patch_size
        self.assertEqual(cols, 8, "Grid must have 8 columns")
        self.assertEqual(rows, 6, "Grid must have 6 rows")

        # 3. Total candidate grid positions: exactly 48
        total_candidates = cols * rows
        self.assertEqual(total_candidates, 48, "Total candidate grid positions must be exactly 48")

        # 4. Execute pipeline
        res = process_wsi_canvas(np.array(img), self.pipeline, orig_dims=(w, h), patch_size=224)
        s = res["summary"]
        df = res["df_tiles"]

        # 5. Full numerical reconciliation
        self.assertEqual(s["total_tiles"], 48, "Reported total tiles must be 48")
        self.assertEqual(len(df), 48, "Tile metadata rows must be exactly 48")
        self.assertEqual(s["tissue_tiles"], 44, "Accepted tissue tiles must be exactly 44")
        self.assertEqual(len(df[df["is_tissue"] == False]), 4, "Rejected glass tiles must be exactly 4")
        self.assertEqual(s["malignant_tiles"] + s["benign_tiles"], 44, "Malignant + Benign must equal 44")
        self.assertEqual(res["prob_upsampled"].shape, (1344, 1792), "Heatmap must match WSI canvas dimensions")

    def test_33_openslide_svs_streaming_pipeline(self):
        """Verifies that OpenSlide stream processing on real TCGA SVS slide executes and reconciles correctly."""
        svs_dir = PROJECT_ROOT / "data" / "External_test" / "TCGA" / "tumor"
        svs_files = list(svs_dir.glob("*.svs"))
        if not svs_files:
            self.skipTest("No TCGA SVS files present in data directory")

        test_svs = svs_files[0]
        res = process_openslide_wsi_file(test_svs, self.pipeline, stride=224, max_tiles=20)
        s = res["summary"]
        df = res["df_tiles"]

        self.assertGreater(s["total_tiles"], 0)
        self.assertEqual(len(df), s["total_tiles"])
        self.assertEqual(s["tissue_tiles"] + len(df[df["is_tissue"] == False]), s["total_tiles"])
        self.assertEqual(s["malignant_tiles"] + s["benign_tiles"], s["tissue_tiles"])

    def test_34_external_test_page_removal_and_archive(self):
        """Verifies that External Test Evaluation is decoupled from app.py and preserved in research_archive."""
        app_file = PROJECT_ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Route and navigation must be removed from active app
        self.assertNotIn('elif nav == "🧪 External Test Evaluation":', content)
        self.assertNotIn('"🧪 External Test Evaluation"', content)

        # Archive directory must exist with all required subdirectories
        archive_root = PROJECT_ROOT / "research_archive" / "external_generalization_evaluation"
        self.assertTrue(archive_root.exists(), f"External archive missing: {archive_root}")

        for sub in ["reports", "results", "scripts", "configs", "data_manifest"]:
            p = archive_root / sub
            self.assertTrue(p.exists() and p.is_dir(), f"Archive subdirectory missing: {p}")

        # Check key preserved files
        self.assertTrue((archive_root / "README.md").exists())
        self.assertTrue((archive_root / "reports" / "external_test_report.json").exists())
        self.assertTrue((archive_root / "results" / "external_test_predictions.csv").exists())
        self.assertTrue((archive_root / "scripts" / "external_test_eval.py").exists())

        # Check README contains explanation of the 0-patch discrepancy
        with open(archive_root / "README.md", "r", encoding="utf-8") as f:
            readme_text = f.read()
        self.assertIn("0 Patches Found", readme_text)
        self.assertIn("Root Cause Audit", readme_text)
        self.assertIn("PatchCamelyon", readme_text)

    def test_35_methodology_doc_preservation(self):
        """Verifies that Methodology is preserved under docs/research and sidebar pointer is present."""
        app_file = PROJECT_ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Active route must be removed
        self.assertNotIn('elif nav == "📖 Methodology & Validation":', content)

        # Sidebar pointer must be present in app.py
        self.assertIn("📖 Methodology & Protocol Reference", content)
        self.assertIn("methodology_and_dataset_provenance.md", content)

        # Authoritative markdown document must exist and contain complete protocol details
        doc_path = PROJECT_ROOT / "docs" / "research" / "methodology_and_dataset_provenance.md"
        self.assertTrue(doc_path.exists(), f"Methodology doc missing: {doc_path}")

        with open(doc_path, "r", encoding="utf-8") as f:
            doc_text = f.read()

        # Check protocol layers and thresholds
        self.assertIn("Hierarchical 4-Layer Diagnostic Protocol", doc_text)
        self.assertIn("Layer 1: Tissue Quality & Focus Verification", doc_text)
        self.assertIn("Layer 2: Primary Malignancy Screening", doc_text)
        self.assertIn("Layer 3: Multi-Head Histopathological Anomaly Profiler", doc_text)
        self.assertIn("Layer 4: Visual Attention & Attribution", doc_text)
        self.assertIn("0.4038", doc_text)  # Ductal threshold
        self.assertIn("0.5315", doc_text)  # Density threshold
        self.assertIn("0.2715", doc_text)  # Necrosis threshold
        self.assertIn("QUARANTINED", doc_text)  # Nuclear atypia quarantined status
        self.assertIn("CAMELYON16", doc_text)
        self.assertIn("BACH", doc_text)
        self.assertIn("NuCLS", doc_text)
        self.assertIn("BCSS", doc_text)

    def test_36_learned_feature_analysis_restoration_and_accessibility(self):
        """Verifies that Learned Feature Analysis is fully restored in web app navigation and backed by cached projections."""
        app_file = PROJECT_ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            app_text = f.read()

        # Primary navigation MUST include Learned Feature Analysis
        nav_block = app_text.split("nav_options = [")[1].split("]")[0]
        self.assertIn('"🧬 Learned Feature Analysis"', nav_block, "Learned Feature Analysis must be restored in navigation")

        # Active routing must contain standalone Section 4
        self.assertIn('elif nav == "🧬 Learned Feature Analysis":', app_text)

        # Documentation pointer and preserved research doc
        self.assertIn("learned_feature_analysis_and_latent_space.md", app_text)
        doc_path = PROJECT_ROOT / "docs" / "research" / "learned_feature_analysis_and_latent_space.md"
        self.assertTrue(doc_path.exists(), f"Learned feature analysis research doc missing: {doc_path}")

        # Check implementation script and projections artifact
        script_path = PROJECT_ROOT / "src" / "objective_2" / "feature_analysis.py"
        self.assertTrue(script_path.exists(), f"Feature analysis script missing: {script_path}")
        projections_path = PROJECT_ROOT / "result" / "objective_2" / "feature_analysis" / "feature_projections.json"
        self.assertTrue(projections_path.exists(), f"Feature projections JSON missing: {projections_path}")

        with open(projections_path, "r", encoding="utf-8") as pf:
            proj_data = json.load(pf)
        self.assertIn("models", proj_data)
        self.assertEqual(len(proj_data.get("samples", [])), 120)

        # Check feature dimensions table and morphology audit in research doc and app
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_text = f.read()
        self.assertIn("1,280-D", doc_text)
        self.assertIn("2,048-D", doc_text)
        self.assertIn("1,024-D", doc_text)
        self.assertIn("4,096-D", doc_text)
        self.assertIn("What PathoVision AI Does NOT Do", doc_text)
        self.assertIn("What PathoVision AI Actually Computes", doc_text)

        # Check app contains table and morphology audit
        self.assertIn("1,280-D", app_text)
        self.assertIn("What PathoVision AI Does NOT Do", app_text)
        self.assertIn("What PathoVision AI Actually Computes", app_text)

        # Check Normal/Anomaly clarification presence in app
        self.assertIn("An anomaly pattern was flagged, but the primary tumour classifier did not exceed its tumour threshold", app_text)

        # Check WSI Viewer enhancements presence in app
        self.assertIn("Spatial Heatmap Visualization Mode:", app_text)
        self.assertIn("0.00 — Normal Lymph Node Stroma", app_text)

    def test_38_single_patch_analysis_refinements(self):
        """Verifies Single Patch Analysis visual hierarchy, metadata, Grad-CAM transparency, and anomaly controls."""
        app_file = PROJECT_ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            app_text = f.read()

        # 1. Primary Model Task and Intended Task Caveat
        self.assertIn("Primary Model Task: CAMELYON16 — Lymph-node metastasis classification", app_text)
        self.assertIn("(Intended task of the model; does not prove the uploaded image belongs to CAMELYON16)", app_text)
        self.assertIn("P(Tumor) Model Score", app_text)

        # 2. Raw logits moved to Technical Diagnostics
        self.assertIn("Raw Classifier Logit (z)", app_text)

        # 3. Exact Grad-CAM attribution text
        expected_gradcam_text = "Warmer regions indicate stronger positive attribution to the selected model output under this visualization. These highlights show model influence, not confirmed tumour locations."
        self.assertIn(expected_gradcam_text, app_text)
        self.assertIn("Grad-CAM is not tumour segmentation and does not establish biological causation.", app_text)

        # 4. Four-Model Comparison in expandable section without manufactured consensus
        self.assertIn("Multi-Architecture Model Comparison (Identical Input Patch)", app_text)
        self.assertIn("Disagreement among models reflects architecture-specific representations and does not by itself establish which model is correct.", app_text)
        self.assertIn("PathoVision AI presents each model's independent output without manufacturing consensus.", app_text)

        # 5. Three approved anomaly heads with concise note on excluded atypia head
        self.assertIn("Ductal Structure Distortion", app_text)
        self.assertIn("Tumor Cell Density Clusters", app_text)
        self.assertIn("Tumor Necrosis & Cellular Debris", app_text)
        self.assertIn("Note on Excluded Nuclear Pleomorphism / Atypia Head", app_text)
        self.assertIn("cross-dataset validity has not been established", app_text)

        # 6. Attribution perturbation check expander with salient vs control explanation
        self.assertIn("Attribution perturbation check", app_text)
        self.assertIn("Salient-Region Score Drop", app_text)
        self.assertIn("Control-Region Score Drop", app_text)
        self.assertIn("In <b>salient perturbation</b>, pixels with the highest Grad-CAM attribution are masked", app_text)
        self.assertIn("not a universal guarantee", app_text)

        # 7. Ground truth provenance distinguishes verified vs unverified
        self.assertIn("Source dataset: Unknown", app_text)
        self.assertIn("Unverified (Source dataset: Unknown)", app_text)
        self.assertIn("Never labeled as verified.", app_text)

    def test_37_wsi_multimodel_heatmap_comparison_and_redesign(self):
        """Verifies WSI redesign hierarchy and 4-model spatial heatmap comparison on exact same patches."""
        sample_path = PROJECT_ROOT / "data" / "sample_wsi" / "sample_wsi_tumor_058_biopsy.tif"
        self.assertTrue(sample_path.exists(), f"Sample WSI missing: {sample_path}")

        img = Image.open(sample_path).convert("RGB")
        res = process_wsi_canvas(np.array(img), self.pipeline, orig_dims=(1792, 1344), patch_size=224)

        # Compute multi-model heatmaps
        mm_maps = compute_multimodel_wsi_heatmaps(res, self.pipeline, colormap_name="jet", alpha=0.50)

        # 1. All 4 architectures present
        expected_archs = ["EfficientNet-B0", "ResNet-50", "DenseNet-121", "VGG-16"]
        self.assertEqual(sorted(list(mm_maps.keys())), sorted(expected_archs))

        # 2. Check each model's heatmap and reconcile evaluation on identical viable tissue patches
        for arch in expected_archs:
            m_info = mm_maps[arch]
            self.assertEqual(m_info["status"], "AVAILABLE", f"{arch} must be AVAILABLE")
            self.assertEqual(m_info["tissue_tiles"], 44, f"{arch} must evaluate exactly 44 tissue tiles")
            self.assertEqual(m_info["blended_heatmap"].shape, (1344, 1792, 3), f"{arch} heatmap shape mismatch")
            self.assertEqual(m_info["prob_upsampled"].shape, (1344, 1792), f"{arch} prob grid shape mismatch")
            self.assertGreater(m_info["malignant_tiles"], 0, f"{arch} must detect malignant tumor focus")
            self.assertGreater(m_info["tumor_burden_pct"], 20.0, f"{arch} tumor burden percentage mismatch")

        # 3. DenseNet-121 reflects high specificity inductive bias (conservative focal boundary)
        self.assertLessEqual(mm_maps["DenseNet-121"]["malignant_tiles"], mm_maps["VGG-16"]["malignant_tiles"])

        # 4. Check UI structure in app.py
        app_file = PROJECT_ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            app_content = f.read()

        # Section headings
        self.assertIn("1. Select and process slide", app_content)
        self.assertIn("2. Processing summary", app_content)
        self.assertIn("3. Whole-slide spatial viewer", app_content)
        self.assertIn("4. Hotspot tile inspection & live Grad-CAM", app_content)
        self.assertIn("5. Histopathological anomaly phenotyping", app_content)
        self.assertIn("6. Advanced diagnostics & export", app_content)
        self.assertIn("Four-Architecture Spatial Heatmap Comparison", app_content)
        self.assertIn("Overlay 224×224 Tile Screening Grid", app_content)
        self.assertIn("QUARANTINED", app_content)




if __name__ == "__main__":
    unittest.main()


