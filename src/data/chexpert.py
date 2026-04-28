"""
CheXpert dataset loader.

Dataset homepage:
    https://stanfordmlgroup.github.io/competitions/chexpert

Expected directory layout:
    data/chexpert/
    ├── train.csv
    ├── valid.csv
    └── train/
        └── patient00001/
            └── study1/
                └── view1_frontal.jpg

Label policy:
    Uncertain labels (-1) are mapped to 0 (negative) by default.
    Use `uncertainty_policy="ones"` to map to 1 (positive).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import get_xray_transforms

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


class CheXpertDataset(Dataset):
    """
    Multi-label chest X-ray dataset from CheXpert.

    Args:
        root:               Path to dataset root containing CSVs.
        split:              'train' or 'valid'.
        image_size:         Spatial resolution to resize to.
        uncertainty_policy: How to handle -1 labels: 'zeros' | 'ones'.
        frontal_only:       If True, only load frontal-view images.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 320,
        uncertainty_policy: str = "zeros",
        frontal_only: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.uncertainty_policy = uncertainty_policy
        self.transform = get_xray_transforms(image_size, split)

        csv_path = self.root / f"{split}.csv"
        self.df = pd.read_csv(csv_path)

        if frontal_only and "Frontal/Lateral" in self.df.columns:
            self.df = self.df[self.df["Frontal/Lateral"] == "Frontal"]

        self.df = self.df.reset_index(drop=True)
        self._process_labels()

    def _process_labels(self) -> None:
        """Apply uncertainty policy and fill NaN with 0."""
        label_cols = [c for c in CHEXPERT_LABELS if c in self.df.columns]
        self.label_cols = label_cols

        fill_val = 1.0 if self.uncertainty_policy == "ones" else 0.0
        self.df[label_cols] = (
            self.df[label_cols]
            .fillna(0)
            .replace(-1, fill_val)
            .astype(float)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Build absolute path from the relative path in CSV
        img_path = self.root.parent / row["Path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        return image, labels
