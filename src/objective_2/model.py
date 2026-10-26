"""
Objective 2 Model Definition: Binary Classifiers for Histopathology.

Includes EfficientNet-B0, ResNet-50, DenseNet-121, and VGG-16.
All use ImageNet pretrained backbones with transfer learning:
- Stage 1: Frozen backbone + Trainable classification head.
- Stage 2: Unfreeze top feature blocks for fine-tuning.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    vgg16, VGG16_Weights
)


class BaseTumorClassifier(nn.Module):
    """Base class providing common utilities for parameter summary."""
    def get_parameter_summary(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
        }


class EfficientNetB0TumorClassifier(BaseTumorClassifier):
    def __init__(self, dropout_rate: float = 0.2, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=1),
        )

    @property
    def features(self) -> nn.Sequential:
        return self.backbone.features

    @property
    def classifier(self) -> nn.Sequential:
        return self.backbone.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, start_block_idx: int = 6) -> None:
        for i, block in enumerate(self.backbone.features):
            for param in block.parameters():
                param.requires_grad = (i >= start_block_idx)
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True


class ResNet50TumorClassifier(BaseTumorClassifier):
    def __init__(self, dropout_rate: float = 0.2, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features=in_features, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self) -> None:
        # Unfreeze layer4 (the top convolutional block)
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True


class DenseNet121TumorClassifier(BaseTumorClassifier):
    def __init__(self, dropout_rate: float = 0.2, pretrained: bool = True) -> None:
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features=in_features, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self) -> None:
        # Unfreeze denseblock4 and norm5
        for param in self.backbone.features.denseblock4.parameters():
            param.requires_grad = True
        for param in self.backbone.features.norm5.parameters():
            param.requires_grad = True
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True


class VGG16TumorClassifier(BaseTumorClassifier):
    def __init__(self, dropout_rate: float = 0.2, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.backbone = vgg16(weights=weights)

        # VGG16's classifier is a Sequential. We'll modify the final linear layer (index 6)
        in_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        # Unfreeze only the final classifier layers
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self) -> None:
        # Unfreeze the last convolutional block (layers 24 to 30)
        for i, layer in enumerate(self.backbone.features):
            if i >= 24:
                for param in layer.parameters():
                    param.requires_grad = True
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
