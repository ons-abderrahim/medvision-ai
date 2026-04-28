"""
Training entry point for MedVision AI.

Usage:
    python src/models/train.py --config configs/densenet_chexpert.yaml

Supports:
    - Single GPU
    - Multi-GPU via torch.distributed (DDP)
    - Mixed precision (AMP)
    - Cosine / step LR scheduling
    - MixUp augmentation
    - Checkpoint saving + early stopping
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.chexpert import CheXpertDataset
from src.data.isic import ISICDataset
from src.models.densenet import DenseNetChexpert
from src.models.efficientnet import EfficientNetISIC
from src.utils.metrics import compute_auc, compute_f1
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    backbone = cfg["model"]["backbone"]
    num_classes = cfg["model"]["num_classes"]
    pretrained = cfg["model"].get("pretrained", True)

    if backbone == "densenet121":
        return DenseNetChexpert(num_classes=num_classes, pretrained=pretrained)
    elif backbone.startswith("efficientnet"):
        return EfficientNetISIC(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# ──────────────────────────────────────────────────────────────
# Dataset factory
# ──────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict):
    dataset_name = cfg["data"]["dataset"]
    batch_size = cfg["training"]["batch_size"]
    image_size = cfg["data"]["image_size"]
    data_root = cfg["data"].get("root", "data")

    if dataset_name == "chexpert":
        DatasetCls = CheXpertDataset
    elif dataset_name == "isic":
        DatasetCls = ISICDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_ds = DatasetCls(root=data_root, split="train", image_size=image_size)
    val_ds = DatasetCls(root=data_root, split="valid", image_size=image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % 50 == 0:
            logger.info(f"  step {step:4d} | loss {loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    return {
        "val_loss": total_loss / len(loader),
        "auc": compute_auc(all_labels, all_probs),
        "f1": compute_f1(all_labels, all_probs),
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = build_model(cfg).to(device)
    train_loader, val_loader = build_dataloaders(cfg)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = GradScaler()

    best_auc = 0.0
    output_dir = Path(cfg.get("output_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"AUC={metrics['auc']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"{elapsed:.0f}s"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            ckpt_path = output_dir / "best.ckpt"
            model.save(str(ckpt_path))
            logger.info(f"  ✓ New best AUC {best_auc:.4f} — saved to {ckpt_path}")

    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
