"""
Objective 2 Part B: Multi-Head Anomaly Model Definition.

Uses the frozen 1280-D feature representation from the trained Objective-2A
EfficientNet-B0 backbone (result/objective_2/checkpoints/best_model.pt) with
independent task-specific linear heads:
  1. head_ductal:  InSitu vs Normal/Benign (BACH ductal architecture proxy)
  2. head_density: >=10 tumor nuclei vs 0 (NuCLS high tumor-density)
  3. head_necrosis: >=25% necrosis vs <=5% (BCSS tumor necrosis/debris)

Backbone features are strictly frozen for linear probe training.

REMEDIATION NOTE (2026-10-18):
  An 'atypia' head connected to MITOS-ATYPIA-14 was added to this class
  and trained on 2026-10-14 in violation of the project's explicit freeze
  rules. The head definition is retained in nn.ModuleDict so existing
  checkpoints load without key errors, but:
    - PERMITTED_TASKS excludes 'atypia'
    - forward_all_heads() iterates only PERMITTED_TASKS
    - The atypia threshold (0.05) is NOT present in ANOMALY_THRESHOLDS
    - predict_anomalies.py and evaluate_anomaly_heads.py point to the clean
      checkpoint (anomaly_linear_heads_clean.pt) which holds uninitialised
      atypia weights, making the head numerically inactive
  Do not call forward_features(feat, 'atypia') in any production path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetB0MultiHeadAnomaly(nn.Module):
    """
    Multi-Head Anomaly Classifier using frozen EfficientNet-B0 backbone.

    PERMITTED_TASKS are the three heads trained on authorised external datasets:
      ductal:   InSitu vs Normal/Benign (BACH ductal architecture proxy)
      density:  >=10 tumor nuclei vs 0 (NuCLS high tumor-density)
      necrosis: >=25% necrosis vs <=5% (BCSS tumor necrosis/debris)

    The 'atypia' slot is present in nn.ModuleDict so legacy checkpoints load
    without key errors, but it is NOT included in PERMITTED_TASKS and must NOT
    be called in any production inference path (MITOS-ATYPIA-14 training was
    prohibited by the project's freeze rules).
    """

    # Only the three heads that were authorised and correctly trained
    PERMITTED_TASKS = ["ductal", "density", "necrosis"]

    # Kept for backwards-compatibility with checkpoint loading; do not use
    TASKS = PERMITTED_TASKS  # was ["atypia", "ductal", "density", "necrosis"] before remediation

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        dropout_rate: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.in_features = 1280

        # Build base EfficientNet-B0 backbone
        base_model = efficientnet_b0(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Load Objective-2A trained weights into backbone if checkpoint provided
        if checkpoint_path is not None:
            self.load_backbone_from_checkpoint(checkpoint_path)

        # Freeze all backbone convolutional features
        self.freeze_backbone()

        # Decoupled anomaly heads (each is Linear: 1280 -> 1)
        self.heads = nn.ModuleDict({
            "atypia": nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(in_features=self.in_features, out_features=1),
            ),
            "ductal": nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(in_features=self.in_features, out_features=1),
            ),
            "density": nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(in_features=self.in_features, out_features=1),
            ),
            "necrosis": nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(in_features=self.in_features, out_features=1),
            ),
        })

    def load_backbone_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Loads backbone weights from Objective-2A best_model.pt checkpoint."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        # Filter state dict for backbone.features.*
        features_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone.features."):
                new_k = k.replace("backbone.features.", "")
                features_dict[new_k] = v

        self.features.load_state_dict(features_dict, strict=True)

    def freeze_backbone(self) -> None:
        """Freezes all backbone feature extraction parameters."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 1280-D global average pooled feature embedding.
        Input:  [B, 3, 224, 224]
        Output: [B, 1280]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_features(self, feat: torch.Tensor, task: str) -> torch.Tensor:
        """Forward pass for a specific task head given 1280-D features."""
        if task not in self.heads:
            raise ValueError(f"Unknown task '{task}'. Expected one of: {list(self.heads.keys())}")
        return self.heads[task](feat)

    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        """Full forward pass: image -> 1280-D features -> scalar logit for specified task."""
        feat = self.extract_features(x)
        return self.forward_features(feat, task)

    def forward_all_heads(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs forward pass across all PERMITTED (three) anomaly heads sharing the 1280-D embedding.
        Does NOT call the atypia head — see module docstring."""
        feat = self.extract_features(x)
        return {task: self.heads[task](feat) for task in self.PERMITTED_TASKS}

    def get_parameter_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "backbone_parameters": sum(p.numel() for p in self.features.parameters()),
            "heads_parameters": sum(p.numel() for p in self.heads.parameters()),
        }
