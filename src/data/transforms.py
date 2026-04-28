"""
Shared image preprocessing pipelines for medical imaging.

Medical images require careful normalisation:
- Chest X-rays: single-channel (greyscale) → replicated to 3 channels
- Dermoscopy: standard RGB
- MRI: slice normalisation + windowing
"""

from __future__ import annotations

from torchvision import transforms


# ImageNet statistics (used for pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 320) -> transforms.Compose:
    """
    Strong augmentation pipeline for training.

    Includes:
        - Random resized crop (scale 80–100 %)
        - Horizontal flip
        - Random rotation (±10°)
        - Colour jitter (brightness + contrast)
        - Gaussian blur
        - ImageNet normalisation
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 320) -> transforms.Compose:
    """
    Deterministic pipeline for validation / inference.

    Includes:
        - Resize to image_size
        - Centre crop
        - ImageNet normalisation
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_xray_transforms(image_size: int = 320, split: str = "train") -> transforms.Compose:
    """
    X-ray specific pipeline (greyscale → 3-channel).

    Args:
        image_size: Target spatial resolution.
        split:      'train' for augmented, anything else for val.
    """
    base = get_train_transforms(image_size) if split == "train" else get_val_transforms(image_size)
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        *base.transforms,
    ])
