"""
Objective 2 Explainability Module: Multi-Architecture Grad-CAM Engine.

Supports:
- EfficientNet-B0 (model.backbone.features[-1])
- ResNet-50       (model.backbone.layer4[-1])
- DenseNet-121    (model.backbone.features.norm5)
- VGG-16          (model.backbone.features[28])

Computes class activation heatmaps via backward hook gradients and overlays
them onto raw histology RGB images with customizable alpha blending and colormaps.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ImageNet normalization standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) for binary classifiers.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.model.eval()

        # Disable in-place operations across model layers to prevent BackwardHook autograd view errors
        for mod in self.model.modules():
            if hasattr(mod, "inplace") and getattr(mod, "inplace", False):
                mod.inplace = False

        if target_layer is None:
            self.target_layer = self._auto_detect_target_layer(model)
        else:
            self.target_layer = target_layer

        self.gradients = []
        self.activations = []
        self._register_hooks()

    def _auto_detect_target_layer(self, model: nn.Module) -> nn.Module:
        """Automatically resolves the final convolutional layer for known backbones."""
        # 1. Backbones with .features Sequential
        if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
            features = model.backbone.features
            # DenseNet-121 has denseblock4 (use final Conv2d to avoid inplace ReLU issue on norm5)
            if hasattr(features, "denseblock4"):
                return features.denseblock4.denselayer16.conv2
            # VGG-16: 31 layers in features; layer 28 is the final Conv2d
            if len(features) > 28 and isinstance(features[28], nn.Conv2d):
                return features[28]
            # EfficientNet-B0 has features[-1]
            return features[-1]

        # 2. ResNet-50
        if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            return model.backbone.layer4[-1]

        # Fallback: scan for last Conv2d or BatchNorm2d
        last_conv = None
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                last_conv = module
        if last_conv is not None:
            return last_conv

        raise ValueError("Could not automatically locate the target convolutional layer.")

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        self.f_hook = self.target_layer.register_forward_hook(forward_hook)
        self.b_hook = self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        target_class: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """
        Generates 2D normalized Grad-CAM heatmap [0, 1] and prediction probability.
        input_tensor: [1, 3, H, W] normalized tensor
        target_class: 1 for Malignant / Tumor evidence, 0 for Normal / Benign evidence.
        Returns: (heatmap_2d, prob_score)
        """
        self.gradients.clear()
        self.activations.clear()

        input_tensor = input_tensor.to(device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Forward pass
        logits = self.model(input_tensor)
        prob = float(torch.sigmoid(logits).item())

        # Backward pass targeting the desired class
        self.model.zero_grad()
        if target_class == 1:
            target_scalar = logits
        else:
            # Targeting Normal (evidence that drives logit lower)
            target_scalar = -logits

        target_scalar.backward(retain_graph=True)

        if not self.gradients or not self.activations:
            raise RuntimeError("Grad-CAM hooks failed to collect activations/gradients.")

        grads = self.gradients[0].detach().cpu().numpy()[0]  # [C, H', W']
        acts = self.activations[0].detach().cpu().numpy()[0]   # [C, H', W']

        # Global average pooling on gradients -> channel weights
        weights = np.mean(grads, axis=(1, 2))  # [C]

        # Linear combination of activation maps
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # Apply ReLU to retain features that positively correlate with class
        cam = np.maximum(cam, 0)

        # Resize to input dimensions (224, 224)
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        if cam.shape != (h, w):
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, prob

    def remove_hooks(self) -> None:
        self.f_hook.remove()
        self.b_hook.remove()


def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays a 2D float heatmap [0, 1] onto an RGB uint8 image [H, W, 3].
    Returns (blended_rgb, colored_heatmap_rgb).
    """
    # Generate colored heatmap using matplotlib colormaps (e.g., 'jet' or 'turbo')
    import matplotlib
    cmap = matplotlib.colormaps[colormap_name]
    colored_heatmap = cmap(heatmap)[..., :3]  # drop alpha -> [H, W, 3] in [0, 1]
    colored_heatmap_uint8 = np.uint8(255 * colored_heatmap)

    if image_rgb.dtype != np.uint8:
        image_rgb = np.uint8(image_rgb)

    # Blend original image with heatmap
    blended = np.uint8(alpha * colored_heatmap_uint8 + (1.0 - alpha) * image_rgb)
    return blended, colored_heatmap_uint8
