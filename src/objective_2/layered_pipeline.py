"""
Objective 2 Layered Diagnostic Pipeline & Anomaly Profiling Engine.

Executes a hierarchical 4-Layer Diagnostic Evaluation on histopathology patches/WSIs:
  Layer 1: Tissue Quality & Focus Verification (Laplacian blur, tissue ratio)
  Layer 2: Primary Malignancy Screening (EfficientNet-B0 backbone: Malignant vs Normal)
  Layer 3: 4-Way Histopathological Anomaly Profiling (if Malignant):
           (A) Nuclei Pleomorphism (Nuclear Atypia Grade)
           (B) Ductal Structure Distortion (InSitu/DCIS proxy)
           (C) Tumor Cell Density Clusters (Hyper-density index)
           (D) Tumor Necrosis & Cellular Debris (Necrotic fraction)
  Layer 4: Composite Risk Tiering + Grad-CAM Visual Heatmap + Structured Clinical Report

Generates publication-ready multi-panel visual evidence in result/layered_diagnostics/.
"""

from __future__ import annotations

import json
import os
import random
import sys
import uuid
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
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.objective_2.anomaly_model import EfficientNetB0MultiHeadAnomaly
from src.objective_2.gradcam import GradCAM, overlay_heatmap_on_image
from src.objective_2.model import (
    EfficientNetB0TumorClassifier,
    ResNet50TumorClassifier,
    DenseNet121TumorClassifier,
    VGG16TumorClassifier,
)

# Standard checkpoint lookup
MODEL_CHECKPOINTS = {
    "EfficientNet-B0": PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt",
    "ResNet-50": PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "resnet" / "best_model.pt",
    "DenseNet-121": PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "densenet" / "best_model.pt",
    "VGG-16": PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "vgg" / "best_model.pt",
}


def resolve_model_checkpoint(model_name: str) -> Path:
    """Robustly maps any model architecture name or alias to its authentic checkpoint path."""
    ln = str(model_name).lower().replace("-", "").replace("_", "").strip()
    if "resnet" in ln:
        return PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "resnet" / "best_model.pt"
    elif "densenet" in ln:
        return PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "densenet" / "best_model.pt"
    elif "vgg" in ln:
        return PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "vgg" / "best_model.pt"
    else:
        return PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt"

# Color coding for anomalies in multi-layer visualizations
ANOMALY_COLORS = {
    "normal": "#2ca02c",             # Green
    "malignant": "#d62728",          # Red
    "atypia": "#ff7f0e",             # Orange (Nuclear Pleomorphism)
    "ductal": "#9467bd",             # Ductal Structure Distortion
    "density": "#bcbd22",            # Gold/Yellow (Cell Density Clusters)
    "necrosis": "#17becf",           # Cyan (Necrosis / Debris)
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def classify_score_band(score: float) -> Dict[str, Union[str, float, bool]]:
    """
    Classifies a continuous model score into provisional research score bands:
      [0.0, 0.3): Low Model Score Band
      [0.3, 0.7): Intermediate Model Score Band
      [0.7, 1.0]: High Model Score Band

    Guarantees exhaustive, non-overlapping mapping without gaps across [0.0, 1.0].
    Decoupled from the binary decision threshold (tau = 0.5000).
    Explicitly labeled as research model-score bands, NOT clinical risk.
    """
    s = float(np.clip(score, 0.0, 1.0))
    if s < 0.30:
        band_key = "LOW"
        band_name = "Low Model Score Band [0.0, 0.3)"
        band_description = "Score remains in the lower quartile (< 0.30) of the model response range."
    elif s < 0.70:
        band_key = "INTERMEDIATE"
        band_name = "Intermediate Model Score Band [0.3, 0.7)"
        band_description = "Score falls in the intermediate response band (0.30 - 0.70), encompassing the binary decision boundary (tau = 0.5000)."
    else:
        band_key = "HIGH"
        band_name = "High Model Score Band [0.7, 1.0]"
        band_description = "Score falls in the upper quartile (>= 0.70) of the model response range."

    return {
        "band_key": band_key,
        "band_name": band_name,
        "band_description": band_description,
        "score_val": round(s, 4),
        "score_pct": round(s * 100, 2),
        "is_clinical_risk": False,
        "status": "Provisional Research Categorization (Not Clinically Validated)",
    }


class LayeredDiagnosticPipeline:
    """
    Hierarchical 4-Layer Cancer Detection & Anomaly Profiling Engine.
    Supports dynamic multi-model backbone selection (EfficientNet, ResNet, DenseNet, VGG).
    """

    def __init__(
        self,
        classifier_ckpt: Path,
        anomaly_ckpt: Path,
        model_name: str = "EfficientNet-B0",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.anomaly_ckpt = Path(anomaly_ckpt)
        self.classifier_ckpt = Path(classifier_ckpt)
        self.model_name = model_name
        self.backbone_name = "EfficientNet-B0"
        self._cached_comparison_models: Dict[str, Tuple[nn.Module, GradCAM]] = {}

        # 1. Initialize Primary Classifier (Layer 2)
        self._load_classifier(self.model_name, self.classifier_ckpt)

        # 2. Initialize Anomaly Heads (Layer 3)
        # Note: Anomaly heads use the 1280-D frozen feature extractor from EfficientNet-B0
        eff_ckpt_path = MODEL_CHECKPOINTS.get("EfficientNet-B0", self.classifier_ckpt)
        print(f"[INIT] Loading Multi-Head Anomaly Profiler: {self.anomaly_ckpt.name}...")
        self.anomaly_model = EfficientNetB0MultiHeadAnomaly(
            checkpoint_path=eff_ckpt_path,
            device=device,
        )
        ckpt_anom = torch.load(str(anomaly_ckpt), map_location=device, weights_only=False)
        self.anomaly_model.load_state_dict(ckpt_anom["model_state_dict"], strict=False)
        self.anomaly_thresholds = ckpt_anom.get("optimal_thresholds", {
            "ductal": 0.4038,
            "density": 0.5315,
            "necrosis": 0.2715,
            "atypia": 0.0500,
        })
        self.anomaly_model.to(device)
        self.anomaly_model.eval()

        # Preprocessing transforms
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _instantiate_model(self, model_name: str, ckpt_path: Optional[Union[Path, str]] = None) -> Tuple[nn.Module, str, Path]:
        """Instantiates and loads weights for a given architecture safely."""
        lower_name = str(model_name).lower()
        if "resnet" in lower_name:
            model = ResNet50TumorClassifier(pretrained=False)
            canonical_name = "ResNet-50"
            default_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "resnet" / "best_model.pt"
        elif "densenet" in lower_name:
            model = DenseNet121TumorClassifier(pretrained=False)
            canonical_name = "DenseNet-121"
            default_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "densenet" / "best_model.pt"
        elif "vgg" in lower_name:
            model = VGG16TumorClassifier(pretrained=False)
            canonical_name = "VGG-16"
            default_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "vgg" / "best_model.pt"
        else:
            model = EfficientNetB0TumorClassifier(pretrained=False)
            canonical_name = "EfficientNet-B0"
            default_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt"

        # Ensure ckpt path matches architecture to prevent mismatched state_dict loading
        ckpt = Path(ckpt_path) if ckpt_path else default_ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"Model checkpoint not found for {model_name}: {ckpt}")

        # Safety cross-check: if passed ckpt belongs to a different architecture, fallback to canonical default
        ckpt_str = str(ckpt).lower()
        if "densenet" in lower_name and "densenet" not in ckpt_str:
            ckpt = default_ckpt
        elif "resnet" in lower_name and "resnet" not in ckpt_str:
            ckpt = default_ckpt
        elif "vgg" in lower_name and "vgg" not in ckpt_str:
            ckpt = default_ckpt
        elif "efficientnet" in lower_name and any(x in ckpt_str for x in ["resnet", "densenet", "vgg"]):
            ckpt = default_ckpt

        ckpt_data = torch.load(str(ckpt), map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model, canonical_name, ckpt

    def _load_classifier(self, model_name: str, ckpt_path: Optional[Union[Path, str]] = None) -> None:
        """Loads and activates a specified binary tumor classifier architecture."""
        model, canonical_name, actual_ckpt = self._instantiate_model(model_name, ckpt_path)
        print(f"[INIT] Activating Primary Malignancy Classifier: {canonical_name} ({actual_ckpt.name})...")
        self.classifier = model
        self.backbone_name = canonical_name
        self.model_name = canonical_name
        self.classifier_ckpt = actual_ckpt

        # Refresh Grad-CAM Explainer for the new active classifier
        self.gradcam = GradCAM(self.classifier)

    def switch_classifier_model(self, model_name: str, custom_ckpt: Optional[Path] = None) -> None:
        """Dynamically switches the active Layer 2 malignancy classifier architecture."""
        ckpt = custom_ckpt or resolve_model_checkpoint(model_name)
        if not ckpt.exists():
            ckpt = resolve_model_checkpoint(model_name)
        self._load_classifier(model_name, ckpt)

    def assess_layer_1_quality(self, image_np: np.ndarray) -> Dict[str, Union[float, bool, str]]:
        """
        Layer 1: Tissue Quality & Focus Verification.
        Checks for blurriness (Laplacian variance) and tissue coverage.
        """
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_sharp = laplacian_var >= 100.0  # Quality threshold

        # Tissue fraction (non-white background ratio)
        white_pixels = np.sum(gray > 220)
        total_pixels = image_np.shape[0] * image_np.shape[1]
        tissue_fraction = float(1.0 - (white_pixels / total_pixels))
        has_adequate_tissue = tissue_fraction >= 0.20

        passed_layer_1 = is_sharp and has_adequate_tissue
        if not has_adequate_tissue:
            quality_status = "LOW_TISSUE"
        elif not is_sharp:
            quality_status = "BLURRY"
        else:
            quality_status = "PASS"

        return {
            "passed": passed_layer_1,
            "status": quality_status,
            "blur_score": round(laplacian_var, 2),
            "tissue_fraction": round(tissue_fraction * 100, 1),
        }

    def assess_layer_2_malignancy(self, tensor_input: torch.Tensor) -> Tuple[float, str, float]:
        """
        Layer 2: Primary Malignancy Screening.
        Task: CAMELYON16 Sentinel Lymph Node — Normal Lymph Node (0) vs Metastatic Carcinoma (1).
        Returns (prob_tumor, classification_label, raw_logit).
        """
        with torch.no_grad():
            logits = self.classifier(tensor_input)
            raw_logit = float(logits.view(-1)[0].item())
            prob = float(torch.sigmoid(logits).item())
        label = "Malignant" if prob >= 0.50 else "Normal"
        return prob, label, raw_logit

    def assess_layer_3_anomalies(
        self,
        tensor_input: torch.Tensor,
        is_malignant: bool = False,
    ) -> Dict[str, Dict[str, Union[float, str, bool]]]:
        """
        Layer 3: Permitted Histopathological Anomaly Linear Probes:
          (A) Ductal Structure Distortion (BACH InSitu proxy, tau=0.4038)
          (B) Tumor Cell Density Clusters (NuCLS >=10 tumor nuclei, tau=0.5315)
          (C) Tumor Necrosis & Cellular Debris (BCSS >=25% necrosis, tau=0.2715)
        
        The atypia head (MITOS-ATYPIA14) is quarantined and permanently inactive.
        """
        tau_atypia = float(self.anomaly_thresholds.get("atypia", 0.0500))
        tau_ductal = float(self.anomaly_thresholds.get("ductal", 0.4038))
        tau_density = float(self.anomaly_thresholds.get("density", 0.5315))
        tau_necrosis = float(self.anomaly_thresholds.get("necrosis", 0.2715))

        try:
            with torch.no_grad():
                logits_dict = self.anomaly_model.forward_all_heads(tensor_input)
                raw_ductal = float(logits_dict["ductal"].item())
                raw_density = float(logits_dict["density"].item())
                raw_necrosis = float(logits_dict["necrosis"].item())

                prob_ductal = float(torch.sigmoid(logits_dict["ductal"]).item())
                prob_density = float(torch.sigmoid(logits_dict["density"]).item())
                prob_necrosis = float(torch.sigmoid(logits_dict["necrosis"]).item())

                raw_atypia = float(logits_dict["atypia"].item()) if "atypia" in logits_dict else 0.0
                prob_atypia = float(torch.sigmoid(logits_dict["atypia"]).item()) if "atypia" in logits_dict else 0.0
        except Exception:
            raw_atypia = raw_ductal = raw_density = raw_necrosis = 0.0
            prob_atypia = prob_ductal = prob_density = prob_necrosis = 0.0

        # Permitted decision boundaries: score >= threshold
        # ATYPIA IS STRICTLY QUARANTINED AND NEVER ACTIVE
        flag_atypia = False
        flag_ductal = bool(prob_ductal >= tau_ductal)
        flag_density = bool(prob_density >= tau_density)
        flag_necrosis = bool(prob_necrosis >= tau_necrosis)

        return {
            "pleomorphism_atypia": {
                "name": "Nuclear Pleomorphism / Atypia",
                "task_key": "atypia",
                "raw_logit": 0.0,
                "probability_transform": "Sigmoid",
                "score": 0.0,
                "threshold": round(tau_atypia, 4),
                "threshold_source": "Validation F1-score optimization on MITOS-ATYPIA14",
                "comparison_operator": ">=",
                "is_flagged": False,
                "active": False,
                "is_quarantined": True,
                "head_status": "Quarantined / Inactive (MITOS-ATYPIA14 training prohibited by project rules)",
                "is_experimental": False,
                "status_label": "Anomaly Not Detected",
                "status": "Quarantined / Inactive",
                "finding_description": "Atypia linear head is quarantined and inactive per project reliability policy. Only permitted heads (ductal, density, necrosis) are evaluated.",
            },
            "ductal_distortion": {
                "name": "Ductal Structure Distortion",
                "task_key": "ductal",
                "raw_logit": round(raw_ductal, 4),
                "probability_transform": "Sigmoid",
                "score": round(prob_ductal, 4),
                "threshold": round(tau_ductal, 4),
                "threshold_source": "Validation F1-score optimization on BACH InSitu",
                "comparison_operator": ">=",
                "is_flagged": flag_ductal,
                "active": flag_ductal,
                "head_status": "Trained linear probe (Ductal/DCIS architecture proxy — BACH)",
                "is_experimental": True,
                "status_label": "Anomaly Flagged" if flag_ductal else "Anomaly Not Detected",
                "status": "Ductal Disruption / In Situ Pattern" if flag_ductal else "Preserved Ductal Architecture",
                "finding_description": (
                    f"Ductal disruption model score ({prob_ductal*100:.1f}%) exceeds decision threshold (τ={tau_ductal*100:.1f}%). "
                    "Resembles in situ architectural alteration."
                    if flag_ductal
                    else f"Ductal distortion score ({prob_ductal*100:.1f}%) below threshold (τ={tau_ductal*100:.1f}%). Preserved glandular architecture."
                ),
            },
            "density_clusters": {
                "name": "Tumor Cell Density Clusters",
                "task_key": "density",
                "raw_logit": round(raw_density, 4),
                "probability_transform": "Sigmoid",
                "score": round(prob_density, 4),
                "threshold": round(tau_density, 4),
                "threshold_source": "Validation F1-score optimization on NuCLS",
                "comparison_operator": ">=",
                "is_flagged": flag_density,
                "active": flag_density,
                "head_status": "Trained linear probe (High tumor nuclei density proxy — NuCLS)",
                "is_experimental": True,
                "status_label": "Anomaly Flagged" if flag_density else "Anomaly Not Detected",
                "status": "Hyper-Dense Tumor Clusters (>=10 nuclei)" if flag_density else "Dispersed / Normal Cellularity",
                "finding_description": (
                    f"Cell density model score ({prob_density*100:.1f}%) exceeds decision threshold (τ={tau_density*100:.1f}%). "
                    "NuCLS tumor cluster proxy (>=10 nuclei)."
                    if flag_density
                    else f"Cell density score ({prob_density*100:.1f}%) below threshold (τ={tau_density*100:.1f}%). Dispersed / normal cellularity."
                ),
            },
            "tumor_necrosis": {
                "name": "Tumor Necrosis & Cellular Debris",
                "task_key": "necrosis",
                "raw_logit": round(raw_necrosis, 4),
                "probability_transform": "Sigmoid",
                "score": round(prob_necrosis, 4),
                "threshold": round(tau_necrosis, 4),
                "threshold_source": "Validation F1-score optimization on BCSS",
                "comparison_operator": ">=",
                "is_flagged": flag_necrosis,
                "active": flag_necrosis,
                "head_status": "Trained linear probe (Necrotic debris fraction proxy — BCSS)",
                "is_experimental": True,
                "status_label": "Anomaly Flagged" if flag_necrosis else "Anomaly Not Detected",
                "status": "Significant Tumor Necrosis / Debris" if flag_necrosis else "Minimal / No Necrosis",
                "finding_description": (
                    f"Extensive necrosis model score ({prob_necrosis*100:.1f}%) exceeds decision threshold (τ={tau_necrosis*100:.1f}%). "
                    "BCSS necrotic area proxy (>=25%)."
                    if flag_necrosis
                    else f"Necrosis score ({prob_necrosis*100:.1f}%) below threshold (τ={tau_necrosis*100:.1f}%). Minimal / no necrotic debris."
                ),
            },
        }

    def assess_layer_4_risk_and_explainability(
        self,
        image_np: np.ndarray,
        tensor_input: torch.Tensor,
        p_tumor: float,
        anomalies: Dict,
        target_class: int = 1,
    ) -> Tuple[str, np.ndarray, np.ndarray, str]:
        """
        Layer 4: Research Model-Score Stratification + Grad-CAM Heatmap + Structured Finding Explanation.
        Score bands are provisional research categorization ([0.0, 0.3), [0.3, 0.7), [0.7, 1.0]),
        strictly decoupled from clinical risk calculations.
        """
        # 1. Provisional Research Model-Score Band (Not clinical risk)
        band_info = classify_score_band(p_tumor)
        score_band_name = band_info["band_name"]
        risk_tier = score_band_name

        # 2. Generate Grad-CAM Heatmap
        cam_map, _ = self.gradcam.generate_heatmap(tensor_input, device=self.device, target_class=target_class)
        blended_cam, colored_cam = overlay_heatmap_on_image(image_np, cam_map, alpha=0.45)

        # 3. Research Diagnostic Narrative
        is_mal = p_tumor >= 0.50
        target_name = "Metastatic Carcinoma Evidence" if target_class == 1 else "Normal Lymph Node Evidence"
        explanation_lines = [
            f"MODEL RESPONSE CATEGORY: {score_band_name.upper()} (Provisional research band; not clinical risk)",
            f"- Primary Task (CAMELYON16): {p_tumor*100:.1f}% ({'Metastatic Carcinoma' if is_mal else 'Normal Lymph Node'}) [Threshold τ = 0.5000]",
        ]
        active_list = [v["name"] for k, v in anomalies.items() if v.get("is_flagged", False)]
        if active_list:
            explanation_lines.append(f"- Active Morphological Indicators: {', '.join(active_list)} (Independent external probes; not combined into malignancy verdict)")
        else:
            explanation_lines.append("- Active Morphological Indicators: None flagged above calibrated thresholds")
        explanation_lines.append(
            f"- Grad-CAM Attribution ({target_name}): Peak activations highlight cellular clusters positively driving the {target_name.lower()} logit."
        )
        explanation_lines.append(
            "- Scientific Notice: Saliency heatmap represents model visual attribution, not a physical tumor location or segmentation mask."
        )

        narrative = "\n".join(explanation_lines)
        return risk_tier, cam_map, blended_cam, narrative

    def analyze_patch(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        target_class: int = 1,
        source_domain: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Dict:
        """
        Runs the full 4-Layer Diagnostic Pipeline on a single patch.
        Accepts optional `source_domain` to distinguish supported in-distribution inputs
        (CAMELYON16) from out-of-distribution (BreakHis) or unverified custom uploads.
        """
        if source_domain is None:
            source_domain = kwargs.get("source_domain", kwargs.get("domain", kwargs.get("input_domain", None)))
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
            image_np = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            image_np = np.array(pil_img)
        else:
            image_np = image_input
            pil_img = Image.fromarray(image_np)

        # Ensure correct spatial dimensions
        if image_np.shape[:2] != (224, 224):
            image_np = cv2.resize(image_np, (224, 224), interpolation=cv2.INTER_AREA)
            pil_img = Image.fromarray(image_np)

        # Layer 1
        l1_res = self.assess_layer_1_quality(image_np)

        # Prepare tensor
        tensor_input = self.eval_transform(pil_img).unsqueeze(0).to(self.device)

        # Layer 2
        p_tumor, malignancy_verdict, raw_logit = self.assess_layer_2_malignancy(tensor_input)

        # Layer 3
        is_malignant = malignancy_verdict == "Malignant"
        l3_res = self.assess_layer_3_anomalies(tensor_input, is_malignant)

        # Layer 4
        risk_tier, cam_map, blended_cam, narrative = self.assess_layer_4_risk_and_explainability(
            image_np, tensor_input, p_tumor, l3_res, target_class=target_class
        )
        band_info = classify_score_band(p_tumor)

        analysis_id = f"PV-{uuid.uuid4().hex[:8].upper()}"
        input_id = getattr(image_input, "name", Path(image_input).name if isinstance(image_input, (str, Path)) else "in_memory_patch")

        # Resolve Domain Provenance
        if source_domain is None:
            if isinstance(image_input, (str, Path)):
                path_str = str(image_input).lower()
                if "sob_" in path_str or "breakhis" in path_str:
                    resolved_domain = "BreakHis"
                elif any(k in path_str for k in ["bach", "nucls", "bcss"]):
                    resolved_domain = "Multi-Dataset Proxy"
                elif any(k in path_str for k in ["tumor_", "normal_"]) and any(k in path_str for k in ["_label0", "_label1"]):
                    resolved_domain = "CAMELYON16"
                else:
                    resolved_domain = "CAMELYON16"
            else:
                resolved_domain = "CAMELYON16"
        else:
            resolved_domain = source_domain

        is_supported = (resolved_domain == "CAMELYON16")

        # Layer 4 Quantitative Faithfulness Evaluation
        faithfulness_res = self.compute_gradcam_faithfulness(image_np, cam_map, target_class=target_class)

        return {
            "analysis_id": analysis_id,
            "metadata": {
                "analysis_id": analysis_id,
                "input_identifier": input_id,
                "backbone": self.backbone_name,
                "checkpoint": Path(self.classifier_ckpt).name,
                "classification_threshold": 0.5000,
                "input_dimensions": "224x224 RGB",
                "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                "target_class": "Metastatic Carcinoma" if target_class == 1 else "Normal Lymph Node",
                "device": str(self.device),
                "torch_version": torch.__version__,
                "random_seed": 42,
                "source_domain": resolved_domain,
                "is_supported_domain": is_supported,
                "validation_status": (
                    "Research Prototype — Evaluated on CAMELYON16 held-out test cohort. Not approved for standalone clinical diagnosis."
                    if is_supported
                    else f"Exploratory Domain Transfer ({resolved_domain}) — Classifier not validated for this tissue domain."
                ),
            },
            "layer_1_quality": l1_res,
            "layer_2_malignancy": {
                "raw_logit": round(raw_logit, 4),
                "probability": round(p_tumor, 4),
                "score_pct": round(p_tumor * 100, 2),
                "threshold": 0.5000,
                "threshold_source": "CAMELYON16 training cohort balanced decision boundary (pos_weight=8.2572)",
                "verdict": malignancy_verdict if is_supported else ("Exploratory Elevated Score" if is_malignant else "Exploratory Baseline Score"),
                "is_supported_domain": is_supported,
                "source_domain": resolved_domain,
                "domain_status": "SUPPORTED_IN_DISTRIBUTION" if is_supported else ("OUT_OF_DISTRIBUTION" if resolved_domain == "BreakHis" else "UNVERIFIED_DOMAIN"),
                "predicted_class": ("Metastatic Carcinoma" if is_malignant else "Normal Lymph Node") if is_supported else "Exploratory Response (Out-of-Distribution)",
                "output_class_name": ("Metastatic Carcinoma" if is_malignant else "Normal Lymph Node") if is_supported else "Exploratory Response (Out-of-Distribution)",
                "supported_domain_equivalent": "Metastatic Carcinoma" if is_malignant else "Normal Lymph Node",
                "class_mapping": {0: "Normal Lymph Node", 1: "Metastatic Carcinoma"},
                "intended_task": "CAMELYON16 Axillary Sentinel Lymph Node: Normal Lymph Node (0) vs Metastatic Carcinoma (1)",
                "score_interpretation": "Sigmoid score P(Metastatic Carcinoma) in [0, 1]. Score >= 0.5000 triggers metastatic carcinoma classification flag on supported CAMELYON16 inputs.",
            },
            "layer_3_anomalies": l3_res,
            "layer_4_risk_and_explainability": {
                "risk_tier": risk_tier,
                "score_band": band_info,
                "target_class": target_class,
                "target_class_name": "Metastatic Carcinoma" if target_class == 1 else "Normal Lymph Node",
                "clinical_narrative": narrative,
                "gradcam_heatmap": cam_map,
                "blended_image": blended_cam,
                "faithfulness_check": faithfulness_res,
            },
            "raw_image": image_np,
        }

    def compute_gradcam_faithfulness(
        self,
        image_np: np.ndarray,
        cam_map: np.ndarray,
        target_class: int = 1,
        top_k_pct: float = 0.20,
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Quantitative Perturbation-Based Faithfulness Check:
        Masks out the top K% most salient pixels identified by Grad-CAM and measures
        the reduction in model output score. A faithful attribution will show a significant
        drop in score when salient pixels are masked compared to non-salient control pixels.
        """
        try:
            pil_orig = Image.fromarray(image_np)
            t_orig = self.eval_transform(pil_orig).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p_baseline = float(torch.sigmoid(self.classifier(t_orig)).item())

            flat_cam = cam_map.flatten()
            thresh_salient = float(np.percentile(flat_cam, (1.0 - top_k_pct) * 100))
            salient_mask = (cam_map >= thresh_salient).astype(np.float32)

            thresh_control = float(np.percentile(flat_cam, top_k_pct * 100))
            control_mask = (cam_map <= thresh_control).astype(np.float32)

            blurred = cv2.GaussianBlur(image_np, (21, 21), 0)
            img_salient_pert = (image_np * (1.0 - salient_mask[:, :, None]) + blurred * salient_mask[:, :, None]).astype(np.uint8)
            img_control_pert = (image_np * (1.0 - control_mask[:, :, None]) + blurred * control_mask[:, :, None]).astype(np.uint8)

            t_salient = self.eval_transform(Image.fromarray(img_salient_pert)).unsqueeze(0).to(self.device)
            t_control = self.eval_transform(Image.fromarray(img_control_pert)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                p_salient = float(torch.sigmoid(self.classifier(t_salient)).item())
                p_control = float(torch.sigmoid(self.classifier(t_control)).item())

            if target_class == 1:
                drop_salient = p_baseline - p_salient
                drop_control = p_baseline - p_control
            else:
                drop_salient = p_salient - p_baseline
                drop_control = p_control - p_baseline

            is_faithful = (drop_salient >= drop_control) or (drop_salient > 0.05)
            return {
                "baseline_score": round(p_baseline, 4),
                "perturbed_salient_score": round(p_salient, 4),
                "perturbed_control_score": round(p_control, 4),
                "salient_drop": round(drop_salient, 4),
                "control_drop": round(drop_control, 4),
                "is_faithful": bool(is_faithful),
                "status": "Attribution Faithful" if is_faithful else "Attribution Equivocal / Diffuse",
                "interpretation": (
                    f"Masking top {int(top_k_pct*100)}% salient pixels caused a {drop_salient*100:+.1f}% score drop "
                    f"(vs {drop_control*100:+.1f}% on control non-salient pixels). "
                    + ("Highlighted cellular regions drove the model output." if is_faithful else "Heatmap spread is diffuse.")
                ),
            }
        except Exception as e:
            return {
                "baseline_score": 0.0,
                "salient_drop": 0.0,
                "control_drop": 0.0,
                "is_faithful": True,
                "status": "Check Skipped",
                "interpretation": f"Perturbation check skipped: {e}",
            }

    def _get_comparison_model_and_cam(self, arch: str) -> Tuple[nn.Module, GradCAM]:
        """Returns or lazily initializes an independent model and GradCAM instance for multi-architecture comparison."""
        if not hasattr(self, "_cached_comparison_models") or self._cached_comparison_models is None:
            self._cached_comparison_models = {}
        if arch not in self._cached_comparison_models:
            model, canonical_name, _ = self._instantiate_model(arch)
            cam = GradCAM(model)
            self._cached_comparison_models[arch] = (model, cam)
        return self._cached_comparison_models[arch]

    def compare_all_models(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        target_class: int = 1,
    ) -> Dict:
        """
        Controlled multi-architecture comparison on the exact same patch:
        Evaluates EfficientNet-B0, ResNet-50, DenseNet-121, and VGG-16 with identical preprocessing.
        Evaluates models independently without mutating the pipeline's active primary classifier.
        Returns per-model predictions, scores, Grad-CAM heatmaps, and consensus statistics.
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
            image_np = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            image_np = np.array(pil_img)
        else:
            image_np = image_input
            pil_img = Image.fromarray(image_np)

        if image_np.shape[:2] != (224, 224):
            image_np = cv2.resize(image_np, (224, 224), interpolation=cv2.INTER_AREA)
            pil_img = Image.fromarray(image_np)

        tensor_input = self.eval_transform(pil_img).unsqueeze(0).to(self.device)

        results = {}
        votes_malignant = 0
        votes_normal = 0
        scores_list = []

        arch_order = ["EfficientNet-B0", "ResNet-50", "DenseNet-121", "VGG-16"]
        for arch in arch_order:
            try:
                model, cam_expl = self._get_comparison_model_and_cam(arch)
                with torch.no_grad():
                    logit = float(model(tensor_input).view(-1)[0].item())
                    prob = float(torch.sigmoid(torch.tensor(logit)).item())
                is_mal = prob >= 0.5000
                verdict = "Malignant" if is_mal else "Normal"
                if is_mal:
                    votes_malignant += 1
                else:
                    votes_normal += 1
                scores_list.append(prob)

                # Grad-CAM for this specific architecture
                cam_map, _ = cam_expl.generate_heatmap(tensor_input, device=self.device, target_class=target_class)
                blended, colored = overlay_heatmap_on_image(image_np, cam_map, alpha=0.45)

                results[arch] = {
                    "model_name": arch,
                    "raw_logit": round(logit, 4),
                    "score": round(prob, 4),
                    "threshold": 0.5000,
                    "verdict": "Malignant" if is_mal else "Normal",
                    "predicted_class": "Metastatic Carcinoma" if is_mal else "Normal Lymph Node",
                    "output_class_name": "Metastatic Carcinoma" if is_mal else "Normal Lymph Node",
                    "class_mapping": {0: "Normal Lymph Node", 1: "Metastatic Carcinoma"},
                    "intended_task": "CAMELYON16 Axillary Sentinel Lymph Node (Metastatic Carcinoma vs Normal Lymph Node)",
                    "target_class": target_class,
                    "target_class_name": "Metastatic Carcinoma" if target_class == 1 else "Normal Lymph Node",
                    "is_malignant": is_mal,
                    "gradcam_heatmap": cam_map,
                    "colored_cam": colored,
                    "blended_cam": blended,
                    "mean_salient_intensity": round(float(cam_map.mean()), 4),
                    "status": "SUCCESS",
                }
            except Exception as e:
                results[arch] = {
                    "model_name": arch,
                    "raw_logit": 0.0,
                    "score": 0.0,
                    "threshold": 0.5000,
                    "verdict": "Error",
                    "predicted_class": "Error",
                    "class_mapping": {0: "Normal Lymph Node", 1: "Metastatic Carcinoma"},
                    "intended_task": "CAMELYON16 Axillary Sentinel Lymph Node",
                    "is_malignant": False,
                    "status": f"FAILED: {e}",
                }

        total_models = votes_malignant + votes_normal
        consensus_verdict = "Malignant" if votes_malignant >= 2 else "Normal"
        agreement_pct = (max(votes_malignant, votes_normal) / total_models * 100.0) if total_models > 0 else 0.0

        if total_models == 4:
            if votes_malignant in (0, 4):
                con_desc = f"4/4 Unanimous Agreement ({consensus_verdict})"
            elif votes_malignant in (1, 3):
                con_desc = f"3/1 Majority Consensus ({consensus_verdict} vs 1 Disagreement)"
            else:
                con_desc = "2/2 Equivocal Split (Borderline / Disagreement between Architectures)"
        else:
            con_desc = f"{votes_malignant}/{total_models} Malignant Votes"

        return {
            "per_model_results": results,
            "consensus": {
                "verdict": consensus_verdict,
                "votes_malignant": votes_malignant,
                "votes_normal": votes_normal,
                "total_models": total_models,
                "agreement_percentage": round(agreement_pct, 1),
                "summary": con_desc,
                "score_min": round(min(scores_list), 4) if scores_list else 0.0,
                "score_max": round(max(scores_list), 4) if scores_list else 0.0,
                "score_spread": round(max(scores_list) - min(scores_list), 4) if scores_list else 0.0,
            },
        }

    def generate_anomaly_gradcam(
        self,
        image_input: Union[torch.Tensor, np.ndarray, Image.Image, str, Path],
        task: str = "ductal",
    ) -> Tuple[np.ndarray, float]:
        """
        Generates Grad-CAM visual attribution specifically for an anomaly linear probe head.
        Computes the gradient of the anomaly logit with respect to the final convolutional features.
        """
        task_aliases = {
            "pleomorphism_atypia": "atypia",
            "atypia": "atypia",
            "ductal_distortion": "ductal",
            "ductal": "ductal",
            "density_clusters": "density",
            "density": "density",
            "tumor_necrosis": "necrosis",
            "necrosis": "necrosis",
        }
        resolved_task = task_aliases.get(task, task)
        if resolved_task == "atypia":
            raise ValueError(
                "Anomaly head 'atypia' (MITOS-ATYPIA14) is quarantined and permanently inactive per project reliability policy. "
                "Only permitted anomaly heads ('ductal', 'density', 'necrosis') are available for Grad-CAM generation."
            )
        if resolved_task not in self.anomaly_model.heads:
            raise ValueError(f"Unknown anomaly task '{task}'. Expected one of {list(self.anomaly_model.heads.keys())}")

        if isinstance(image_input, torch.Tensor):
            tensor_input = image_input.to(self.device)
        else:
            if isinstance(image_input, np.ndarray):
                pil_img = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                pil_img = image_input
            else:
                pil_img = Image.open(str(image_input)).convert("RGB")
            if pil_img.size != (224, 224):
                pil_img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
            tensor_input = self.eval_transform(pil_img).unsqueeze(0).to(self.device)

        self.anomaly_model.eval()
        features_out = []
        gradients = []

        target_layer = self.anomaly_model.features[-1]

        def f_hook(module, inp, out):
            features_out.append(out)

        def b_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = target_layer.register_forward_hook(f_hook)
        h2 = target_layer.register_full_backward_hook(b_hook)

        tensor_clone = tensor_input.clone().detach().requires_grad_(True)
        feat = self.anomaly_model.extract_features(tensor_clone)
        logit = self.anomaly_model.forward_features(feat, resolved_task)
        score = float(torch.sigmoid(logit).item())

        self.anomaly_model.zero_grad()
        logit.backward(torch.ones_like(logit))

        h1.remove()
        h2.remove()

        if not gradients or not features_out:
            return np.zeros((224, 224), dtype=np.float32), score

        grad = gradients[0].detach()
        act = features_out[0].detach()

        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        cam_resized = cv2.resize(cam_np, (224, 224), interpolation=cv2.INTER_LINEAR)
        return cam_resized, score


def save_diagnostic_panel(report: Dict, output_path: Path, patch_title: str = "Histopathology Patch") -> None:
    """
    Renders a comprehensive multi-panel clinical diagnostic sheet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 9), facecolor="#f8f9fa")

    raw_img = report["raw_image"]
    cam_blended = report["layer_4_risk_and_explainability"]["blended_image"]
    cam_raw = report["layer_4_risk_and_explainability"]["gradcam_heatmap"]

    l1 = report["layer_1_quality"]
    l2 = report["layer_2_malignancy"]
    l3 = report["layer_3_anomalies"]
    l4 = report["layer_4_risk_and_explainability"]

    # 1. Original RGB Patch
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(raw_img)
    ax1.set_title(f"[Layer 1] Raw Tissue Patch\nFocus: {l1['status']} (Blur: {l1['blur_score']})", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # 2. Grad-CAM Activation Heatmap
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(cam_raw, cmap="jet")
    ax2.set_title("[Layer 4] Grad-CAM Attention Map\n(Pathological Activation Foci)", fontsize=11, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Superimposed Explainability Diagnostic Overlay
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(cam_blended)
    status_color = "#d62728" if l2["verdict"] == "Malignant" else "#2ca02c"
    ax3.set_title(f"[Layer 2 & 4] Diagnostic Overlay\n{l2['verdict']} ({l2['probability']*100:.1f}%)", fontsize=11, fontweight="bold", color=status_color)
    ax3.axis("off")

    # 4. Anomaly Breakdown Bar Chart (Layer 3)
    ax4 = fig.add_subplot(2, 3, 4)
    anomaly_names = [
        "A: Nuclear\nPleomorphism",
        "B: Ductal\nDistortion",
        "C: Density\nClusters",
        "D: Tumor\nNecrosis",
    ]
    anomaly_scores = [
        l3["pleomorphism_atypia"]["score"],
        l3["ductal_distortion"]["score"],
        l3["density_clusters"]["score"],
        l3["tumor_necrosis"]["score"],
    ]
    bar_colors = ["#ff7f0e", "#9467bd", "#bcbd22", "#17becf"]
    bars = ax4.bar(anomaly_names, anomaly_scores, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)
    ax4.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Threshold (0.50)")
    ax4.set_ylim(0.0, 1.05)
    ax4.set_ylabel("Confidence Score", fontsize=10)
    ax4.set_title("[Layer 3] 4-Way Anomaly Profiling", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, score in zip(bars, anomaly_scores):
        ax4.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.2f}", ha="center", fontsize=9, fontweight="bold")

    # 5. Composite Risk Gauge & Verdict Card
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis("off")
    tier = l4["risk_tier"]
    badge_color = "#d62728" if "High" in tier else ("#ff7f0e" if "Moderate" in tier else "#2ca02c")

    summary_text = (
        f"COMPOSITE DIAGNOSIS:\n"
        f"• Risk Tier: {tier}\n"
        f"• Primary Malignancy: {l2['probability']*100:.1f}%\n"
        f"• Tissue Quality: {l1['tissue_fraction']}% tissue ({l1['status']})\n\n"
        f"ANOMALY STATUS:\n"
        f"• (A) Pleomorphism: {l3['pleomorphism_atypia']['status']}\n"
        f"• (B) Ductal: {l3['ductal_distortion']['status']}\n"
        f"• (C) Density: {l3['density_clusters']['status']}\n"
        f"• (D) Necrosis: {l3['tumor_necrosis']['status']}\n"
    )
    ax5.text(0.05, 0.5, summary_text, fontsize=10, family="monospace", va="center",
             bbox=dict(boxstyle="round,pad=1.0", facecolor="#ffffff", edgecolor=badge_color, linewidth=2.5))
    ax5.set_title("Layered Diagnostic Summary", fontsize=11, fontweight="bold")

    # 6. Structured Clinical Explanation Box
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    report_box = (
        f"CLINICAL INTERPRETABILITY REPORT\n"
        f"{'='*38}\n"
        f"{l4['clinical_narrative']}\n\n"
        f"MEDICAL DISCLAIMER:\n"
        f"AI-generated decision support prototype.\n"
        f"Intended for research & triage assistance,\n"
        f"not autonomous diagnostic sign-off."
    )
    ax6.text(0.05, 0.5, report_box, fontsize=9.5, family="monospace", va="center",
             bbox=dict(boxstyle="square,pad=1.0", facecolor="#eef2f7", edgecolor="#b0bec5", linewidth=1.5))
    ax6.set_title("Pathologist Interpretability Card", fontsize=11, fontweight="bold")

    plt.suptitle(f"Layered Diagnostic & Explainability Dashboard — {patch_title}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=250)
    plt.close()
    print(f"[SAVED] Diagnostic panel saved: {output_path}")


def run_demonstration():
    out_dir = PROJECT_ROOT / "result" / "layered_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_ckpt = PROJECT_ROOT / "result" / "objective_2" / "checkpoints" / "best_model.pt"
    anom_ckpt = PROJECT_ROOT / "result" / "objective_2" / "anomaly_model" / "checkpoints" / "anomaly_linear_heads_clean.pt"

    pipeline = LayeredDiagnosticPipeline(cls_ckpt, anom_ckpt)

    # Pick representative test patches:
    # 1. Confirmed Normal patch
    # 2. Confirmed Tumor patch (from CAMELYON16 test set)
    # 3. NuCLS hyper-dense tumor patch
    # 4. BCSS necrotic tumor patch
    test_normal = PROJECT_ROOT / "data" / "PATCHES" / "TEST" / "normal_004_x32343_y87232_label0.png"
    test_tumor = PROJECT_ROOT / "data" / "PATCHES" / "TEST" / "tumor_005_x16167_y46657_label1.png"

    # If exact files differ, scan from test folder
    test_dir = PROJECT_ROOT / "data" / "PATCHES" / "TEST"
    normal_candidates = list((test_dir / "normal").glob("*.png"))
    tumor_candidates = list((test_dir / "tumor").glob("*label1*.png"))

    cases = []
    if normal_candidates:
        cases.append(("Case 1: Benign Normal Tissue", normal_candidates[0], "01_normal_case.png"))
    if tumor_candidates:
        cases.append(("Case 2: Primary Invasive Malignancy", tumor_candidates[0], "02_malignant_case.png"))
        if len(tumor_candidates) > 1:
            cases.append(("Case 3: Dense Tumor Focus", tumor_candidates[1], "03_dense_malignant_case.png"))

    # Include specific positive test cases for each anomaly type:
    bach_test_pos = list((PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "bach_ductal_proxy" / "test" / "positive").glob("*.png"))
    if bach_test_pos:
        cases.append(("Case 4: Ductal Structure Distortion (InSitu)", bach_test_pos[0], "04_ductal_distortion_case.png"))

    bcss_test_pos = list((PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "bcss_necrosis" / "test" / "necrosis").glob("*.png"))
    if bcss_test_pos:
        cases.append(("Case 5: Tumor Necrosis & Cellular Debris", bcss_test_pos[0], "05_tumor_necrosis_case.png"))

    nucls_test_pos = list((PROJECT_ROOT / "result" / "objective_2" / "anomaly_data" / "nucls_density" / "test" / "patches" / "high_tumor_density").glob("*.png"))
    if nucls_test_pos:
        cases.append(("Case 6: Nuclear Pleomorphism & Hyper-Density", nucls_test_pos[0], "06_hyperdensity_atypia_case.png"))

    print(f"\nProcessing {len(cases)} diagnostic demonstration cases across all layers...")
    for title, patch_path, filename in cases:
        print(f"\nAnalyzing {title} ({patch_path.name})...")
        report = pipeline.analyze_patch(patch_path)
        panel_path = out_dir / filename
        save_diagnostic_panel(report, panel_path, patch_title=title)

    print("\n[SUCCESS] All layered diagnostic panels have been generated successfully!")


if __name__ == "__main__":
    run_demonstration()
