"""
DenseNet-121 fine-tuning for CheXpert multi-label chest X-ray classification.

Reference:
    Rajpurkar et al. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays
    with Deep Learning. arXiv:1711.05225.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


# CheXpert official 14-label pathology classes
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


class DenseNetChexpert(nn.Module):
    """
    DenseNet-121 backbone with a multi-label classification head for CheXpert.

    Architecture:
        - ImageNet-pretrained DenseNet-121 encoder
        - Global Average Pooling
        - Dropout for regularisation
        - Sigmoid-activated linear head (multi-label, not mutually exclusive)

    Args:
        num_classes:    Number of output classes (default: 14 for CheXpert).
        pretrained:     Load ImageNet weights (default: True).
        dropout_rate:   Dropout probability before the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Keep all feature layers, discard original classifier
        self.features = backbone.features
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(1024, num_classes)

        self.num_classes = num_classes
        self.label_names = CHEXPERT_LABELS[:num_classes]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — normalised chest X-ray tensor.

        Returns:
            logits: (B, num_classes) raw logits.
        """
        features = self.features(x)
        out = self.relu(features)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid-activated probabilities."""
        return torch.sigmoid(self.forward(x))

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "num_classes": self.num_classes}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DenseNetChexpert":
        checkpoint = torch.load(path, map_location=device)
        model = cls(num_classes=checkpoint["num_classes"], pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = DenseNetChexpert(pretrained=False)
    dummy = torch.randn(2, 3, 320, 320)
    logits = model(dummy)
    probs = model.predict_proba(dummy)
    print(f"Logits shape : {logits.shape}")   # (2, 14)
    print(f"Probs  shape : {probs.shape}")    # (2, 14)
    print(f"Label names  : {model.label_names}")
