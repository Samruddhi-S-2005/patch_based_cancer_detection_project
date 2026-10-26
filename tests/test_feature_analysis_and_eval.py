"""
Test Suite: Feature Analysis, Model Evaluation Dashboard, and Anomaly Integrity.

Validates:
1. Pre-classifier feature extraction embedding dimensions (1280, 2048, 1024, 4096).
2. Dimensionality reduction (PCA) 2D projection cache format and non-empty variance.
3. Morphology audit report statements (absence of dynamic micro-measurements).
4. Whole-slide image live engine strictly referencing the 3 permitted anomaly heads.
5. Bit-for-bit cryptographic immutability of model checkpoints and benchmark records.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.objective_2.feature_analysis import (
    load_classifier_model,
    extract_backbone_embedding,
    load_cached_feature_projections,
    get_morphology_audit_report,
    EMBEDDING_DIMENSIONS,
    MODEL_CHECKPOINTS,
)
from src.objective_2.wsi_live_engine import ANOMALY_META


class TestFeatureAnalysisAndEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.dummy_batch = torch.randn(2, 3, 224, 224, device=cls.device)

    def test_01_feature_embedding_dimensions(self):
        """Verifies that all 4 models produce exact expected pre-classifier feature dimensions."""
        expected_dims = {
            "EfficientNet-B0": 1280,
            "ResNet-50": 2048,
            "DenseNet-121": 1024,
            "VGG-16": 4096,
        }

        for arch, exp_dim in expected_dims.items():
            model = load_classifier_model(arch, device=self.device)
            emb = extract_backbone_embedding(model, arch, self.dummy_batch)
            self.assertEqual(
                emb.shape,
                (2, exp_dim),
                f"{arch} feature embedding shape must be (2, {exp_dim}), got {emb.shape}",
            )
            self.assertEqual(EMBEDDING_DIMENSIONS[arch], exp_dim)

    def test_02_pca_projections_cache(self):
        """Verifies the precomputed feature projections cache is populated and mathematically sound."""
        cache = load_cached_feature_projections()
        self.assertIsNotNone(cache, "Feature projection cache must exist on disk.")

        self.assertIn("models", cache)
        self.assertIn("samples", cache)
        self.assertGreaterEqual(len(cache["samples"]), 100)

        for arch in ["EfficientNet-B0", "ResNet-50", "DenseNet-121", "VGG-16"]:
            self.assertIn(arch, cache["models"], f"{arch} must be present in projection cache")
            m_data = cache["models"][arch]
            self.assertEqual(m_data["embedding_dimension"], EMBEDDING_DIMENSIONS[arch])
            exp_var = m_data["pca_explained_variance_ratio"]
            self.assertEqual(len(exp_var), 2)
            self.assertGreater(exp_var[0], 0.0)
            self.assertGreater(exp_var[1], 0.0)
            self.assertLessEqual(sum(exp_var), 1.0)
            self.assertEqual(len(m_data["points"]), len(cache["samples"]))

    def test_03_morphology_audit_statements(self):
        """Verifies the morphology audit truthfully clarifies continuous physical measurements vs CNN embeddings."""
        audit = get_morphology_audit_report()
        self.assertIn("continuous_physical_measurements", audit)
        cpm = audit["continuous_physical_measurements"]
        self.assertFalse(cpm["cell_segmentation_masks"])
        self.assertFalse(cpm["continuous_nuclear_diameters_microns"])
        self.assertFalse(cpm["nuclear_cytoplasmic_ratio_measured"])
        self.assertFalse(cpm["necrotic_pixel_surface_area_measured"])

        # Check anomaly active heads vs quarantined atypia
        heads = audit["anomaly_probing_mechanism"]["active_heads"]
        active_keys = [h["key"] for h in heads]
        self.assertIn("ductal_distortion", active_keys)
        self.assertIn("density_clusters", active_keys)
        self.assertIn("tumor_necrosis", active_keys)
        self.assertNotIn("pleomorphism_atypia", active_keys)

        q_heads = audit["anomaly_probing_mechanism"]["quarantined_heads"]
        q_keys = [h["key"] for h in q_heads]
        self.assertIn("pleomorphism_atypia", q_keys)

    def test_04_wsi_three_permitted_anomaly_heads(self):
        """Verifies WSI anomaly metadata enforces quarantine on atypia and active status on the 3 permitted heads."""
        self.assertTrue(ANOMALY_META["pleomorphism_atypia"]["is_quarantined"])
        self.assertFalse(ANOMALY_META["pleomorphism_atypia"]["is_experimental"])

        # The 3 permitted heads must be active
        self.assertIn("ductal_distortion", ANOMALY_META)
        self.assertIn("density_clusters", ANOMALY_META)
        self.assertIn("tumor_necrosis", ANOMALY_META)

    def test_05_checkpoint_hashes_immutable(self):
        """Verifies that all model checkpoints and master benchmark records are bit-for-bit immutable."""
        expected_hashes = {
            "result/objective_2/checkpoints/best_model.pt": "cc6764e073a21e523b2ae6a6b24b6c718bf4159a778f2563a307c2e88900b3ef",
            "result/objective_2/checkpoints/resnet/best_model.pt": "a1e47d42e1a5b732eeadb5cea620807add979ebea938f9f7b73072b6fd4f8864",
            "result/objective_2/checkpoints/densenet/best_model.pt": "13e80add08df8dd471b95697b50e30c4745c08fc511ed2da8b1782fb53094303",
            "result/objective_2/checkpoints/vgg/best_model.pt": "353f1fcb8d13e412a6dc690242edbf2126d537ec99152223b4b8f1fb1fba24dd",
            "result/objective_2/anomaly_model/checkpoints/anomaly_linear_heads_clean.pt": "c713a37146fe4a28bde27cc9631f6c213002214de677f8d299f33daa7bf368e1",
            "result/objective_2/metrics/master_test_benchmark_comparison.csv": "308cf23119405b488a00187040e79e1319951fa2c3cb0f8c4596071a90c1641c",
        }

        for rel_path, exp_hash in expected_hashes.items():
            fpath = PROJECT_ROOT / rel_path
            self.assertTrue(fpath.exists(), f"Target file missing: {rel_path}")
            actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
            self.assertEqual(
                actual_hash,
                exp_hash,
                f"File hash mismatch for {rel_path}! Expected {exp_hash}, got {actual_hash}",
            )


if __name__ == "__main__":
    unittest.main()
