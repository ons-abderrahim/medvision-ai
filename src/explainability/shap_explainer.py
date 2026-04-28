"""
SHAP (SHapley Additive exPlanations) wrapper using DeepSHAP for vision models.

Paper:
    Lundberg & Lee. "A Unified Approach to Interpreting Model Predictions."
    NeurIPS 2017.

Usage:
    explainer = SHAPExplainer(model, background_loader=val_loader)
    shap_img  = explainer.explain(image_tensor)
    shap_img.save("shap.png")
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader


class SHAPExplainer:
    """
    DeepSHAP pixel-level attribution for PyTorch image classifiers.

    Args:
        model:              Trained PyTorch model in eval mode.
        background_loader:  DataLoader for background reference samples.
        n_background:       Number of background samples to use (default: 50).
        device:             Device string ('cuda' or 'cpu'). Auto-detected if None.

    Example:
        >>> explainer = SHAPExplainer(model, background_loader=val_loader, n_background=50)
        >>> shap_image = explainer.explain(image_tensor, class_idx=4)
        >>> shap_image.save("shap_output.png")
    """

    def __init__(
        self,
        model: nn.Module,
        background_loader: DataLoader,
        n_background: int = 50,
        device: Optional[str] = None,
    ) -> None:
        try:
            import shap
        except ImportError:
            raise ImportError("Install shap: `pip install shap`")

        self.shap = shap
        self.model = model
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Collect background samples
        background = self._collect_background(background_loader, n_background)
        self.explainer = shap.DeepExplainer(model, background.to(self.device))

    # ──────────────────────────────────────────────────────────
    # Background collection
    # ──────────────────────────────────────────────────────────

    def _collect_background(self, loader: DataLoader, n: int) -> torch.Tensor:
        """Collect n images from a DataLoader as reference background."""
        batches = []
        collected = 0
        for images, _ in loader:
            batches.append(images)
            collected += len(images)
            if collected >= n:
                break
        return torch.cat(batches)[:n]

    # ──────────────────────────────────────────────────────────
    # Core computation
    # ──────────────────────────────────────────────────────────

    def explain(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        cmap: str = "RdBu_r",
    ) -> Image.Image:
        """
        Compute SHAP pixel attributions for a single image.

        Args:
            image_tensor: (1, 3, H, W) normalised input tensor.
            class_idx:    Target class index. If None, uses predicted class.
            cmap:         Matplotlib colormap for the attribution plot.

        Returns:
            PIL Image of the SHAP attribution map.
        """
        import matplotlib.pyplot as plt

        assert image_tensor.ndim == 4 and image_tensor.size(0) == 1

        x = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())

        shap_values = self.explainer.shap_values(x)

        # shap_values: list[num_classes] of (B, C, H, W)
        sv = shap_values[class_idx][0]  # (C, H, W)
        # Aggregate channels → (H, W)
        sv_agg = sv.mean(axis=0)

        # Normalise
        vmax = np.abs(sv_agg).max()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(sv_agg, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.axis("off")
        ax.set_title(f"SHAP — class {class_idx}", fontsize=10)
        plt.tight_layout(pad=0)

        # Render to PIL
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_arr = img_arr.reshape(height, width, 3)
        plt.close(fig)
        return Image.fromarray(img_arr)
