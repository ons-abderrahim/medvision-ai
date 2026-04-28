"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for MedVision models.

Paper:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization." ICCV 2017.

Usage:
    explainer = GradCAMExplainer(model, target_layer="features.denseblock4")
    heatmap   = explainer.explain(image_tensor, class_idx=4)
    heatmap.save("heatmap.png")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAMExplainer:
    """
    Computes Grad-CAM heatmaps for a given model and target layer.

    Args:
        model:        PyTorch model (eval mode recommended).
        target_layer: Dotted attribute name of the convolutional layer to hook.
                      E.g. "features.denseblock4" for DenseNet-121.

    Example:
        >>> model = DenseNetChexpert.load("checkpoints/best.ckpt")
        >>> explainer = GradCAMExplainer(model, "features.denseblock4")
        >>> heatmap = explainer.explain(image_tensor, class_idx=10)  # Pleural Effusion
        >>> heatmap.save("gradcam.png")
    """

    def __init__(self, model: nn.Module, target_layer: str) -> None:
        self.model = model
        self.model.eval()

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Resolve layer by dotted name
        layer = self._get_layer(target_layer)
        self._register_hooks(layer)

    # ──────────────────────────────────────────────────────────
    # Hook registration
    # ──────────────────────────────────────────────────────────

    def _get_layer(self, name: str) -> nn.Module:
        parts = name.split(".")
        layer = self.model
        for p in parts:
            layer = getattr(layer, p)
        return layer

    def _register_hooks(self, layer: nn.Module) -> None:
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    # ──────────────────────────────────────────────────────────
    # Core computation
    # ──────────────────────────────────────────────────────────

    def explain(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        orig_size: Optional[tuple[int, int]] = None,
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Generate a Grad-CAM heatmap overlaid on the original image.

        Args:
            image_tensor: (1, 3, H, W) normalised input tensor.
            class_idx:    Target class index. If None, uses argmax of logits.
            orig_size:    (W, H) of the original image for resizing the overlay.
            alpha:        Heatmap overlay transparency (0 = no heatmap, 1 = only heatmap).

        Returns:
            PIL Image with heatmap overlaid on the original input.
        """
        assert image_tensor.ndim == 4, "Expected (1, C, H, W)"

        device = next(self.model.parameters()).device
        x = image_tensor.to(device).requires_grad_(True)

        # Forward pass
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM computation: global average pool the gradients
        gradients = self._gradients    # (1, C, h, w)
        activations = self._activations  # (1, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1).squeeze(0)  # (h, w)
        cam = torch.clamp(cam, min=0)  # ReLU

        # Normalise to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()

        return self._overlay(image_tensor.squeeze(0), cam_np, orig_size, alpha)

    # ──────────────────────────────────────────────────────────
    # Visualisation helper
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _overlay(
        tensor: torch.Tensor,
        cam: np.ndarray,
        size: Optional[tuple[int, int]],
        alpha: float,
    ) -> Image.Image:
        import cv2

        # Denormalise tensor to uint8
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * std + mean).clip(0, 1)
        img_uint8 = (img * 255).astype(np.uint8)

        target_size = size if size else (img_uint8.shape[1], img_uint8.shape[0])

        # Resize CAM to image size
        cam_resized = cv2.resize(cam, target_size)
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        base = cv2.resize(img_uint8, target_size)
        overlay = (alpha * heatmap_rgb + (1 - alpha) * base).astype(np.uint8)
        return Image.fromarray(overlay)
