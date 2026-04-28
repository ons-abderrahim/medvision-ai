"""
EfficientNet-B4 fine-tuning for ISIC skin lesion classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights


ISIC_LABELS = [
    "Melanoma",
    "Melanocytic Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
    "Squamous Cell Carcinoma",
]


class EfficientNetISIC(nn.Module):
    """
    EfficientNet-B4 for ISIC multi-class skin lesion classification.

    Args:
        num_classes:  Number of lesion categories (default: 8).
        pretrained:   Load ImageNet weights (default: True).
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b4(weights=weights)

        # Replace the default classifier
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone
        self.num_classes = num_classes
        self.label_names = ISIC_LABELS[:num_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "num_classes": self.num_classes}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "EfficientNetISIC":
        checkpoint = torch.load(path, map_location=device)
        model = cls(num_classes=checkpoint["num_classes"], pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


if __name__ == "__main__":
    model = EfficientNetISIC(pretrained=False)
    dummy = torch.randn(2, 3, 380, 380)
    logits = model(dummy)
    probs = model.predict_proba(dummy)
    print(f"Logits : {logits.shape}")   # (2, 8)
    print(f"Probs  : {probs.shape}")    # (2, 8)
