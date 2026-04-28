"""
ISIC 2020 skin lesion classification dataset loader.

Dataset homepage:
    https://isic-archive.com

Expected directory layout:
    data/isic/
    ├── train.csv          # columns: image_name, diagnosis
    ├── valid.csv
    └── images/
        ├── ISIC_0000000.jpg
        └── ...
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import get_train_transforms, get_val_transforms

ISIC_LABELS = [
    "melanoma",
    "nevus",
    "basal cell carcinoma",
    "actinic keratosis",
    "benign keratosis",
    "dermatofibroma",
    "vascular lesion",
    "squamous cell carcinoma",
]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(ISIC_LABELS)}


class ISICDataset(Dataset):
    """
    Multi-class skin lesion dataset from ISIC.

    Args:
        root:       Path to dataset root containing CSVs and images/ dir.
        split:      'train' or 'valid'.
        image_size: Spatial resolution.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 380,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / "images"
        self.transform = (
            get_train_transforms(image_size) if split == "train"
            else get_val_transforms(image_size)
        )

        csv_path = self.root / f"{split}.csv"
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

        # Map string labels → integer indices
        self.df["label_idx"] = self.df["diagnosis"].str.lower().map(LABEL_TO_IDX)
        # Drop rows with unknown diagnoses
        self.df = self.df.dropna(subset=["label_idx"]).reset_index(drop=True)
        self.df["label_idx"] = self.df["label_idx"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_dir / f"{row['image_name']}.jpg"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label_idx"], dtype=torch.long)
        return image, label
