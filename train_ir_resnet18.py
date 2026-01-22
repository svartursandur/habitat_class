#!/usr/bin/env python3
"""
CPU-only training script for single-channel IR classification with pretrained ResNet18.
Assumes X_train, y_train, X_test are NumPy arrays already loaded in memory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models


@dataclass
class Config:
    num_classes: int = 2
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    seed: int = 42
    resize_hw: Tuple[int, int] = (96, 96)
    submission_path: str = "submission.csv"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def ir_to_3ch_resize(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    x: (1, H, W) float tensor in range [0, 1] or [0, 255].
    Returns: (3, H, W) resized and normalized to ImageNet stats.
    """
    if x.dtype != torch.float32:
        x = x.float()
    if x.max() > 1.0:
        x = x / 255.0

    x = x.unsqueeze(0)  # (1, 1, H, W)
    x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    x = x.squeeze(0)  # (1, H, W)
    x = x.repeat(3, 1, 1)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
    x = (x - mean) / std
    return x


class IRDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None, size_hw: Tuple[int, int]):
        self.x = x
        self.y = y
        self.size_hw = size_hw

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        ir = self.x[idx]
        if ir.ndim == 2:
            ir = ir[None, ...]
        elif ir.ndim == 3 and ir.shape[0] != 1:
            # If channels are last, move to (1, H, W)
            if ir.shape[-1] == 1:
                ir = np.transpose(ir, (2, 0, 1))
            else:
                raise ValueError("Expected single-channel IR input.")

        ir_t = torch.from_numpy(ir)
        x = ir_to_3ch_resize(ir_t, self.size_hw)

        if self.y is None:
            return x
        return x, int(self.y[idx])


def build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
) -> Tuple[DataLoader, DataLoader]:
    dataset = IRDataset(x_train, y_train, cfg.resize_hw)
    val_len = int(math.floor(len(dataset) * cfg.val_split))
    train_len = len(dataset) - val_len
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model: nn.Module, x_test: np.ndarray, cfg: Config) -> np.ndarray:
    model.eval()
    test_ds = IRDataset(x_test, None, cfg.resize_hw)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    preds = []
    for x in test_loader:
        logits = model(x)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def save_submission(preds: np.ndarray, path: str) -> None:
    df = pd.DataFrame({"id": np.arange(len(preds)), "label": preds})
    df.to_csv(path, index=False)


def main() -> None:
    cfg = Config(
        num_classes=2,
        batch_size=32,
        epochs=5,
        lr=1e-3,
        weight_decay=1e-4,
        val_split=0.2,
        resize_hw=(96, 96),
        submission_path="submission.csv",
    )

    # TODO: Replace these placeholders with actual loading logic.
    # Example:
    # X_train = np.load("X_train.npy")
    # y_train = np.load("y_train.npy")
    # X_test = np.load("X_test.npy")
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray

    set_seed(cfg.seed)
    device = torch.device("cpu")

    model = build_model(cfg.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader, val_loader = make_loaders(X_train, y_train, cfg)

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = eval_one_epoch(model, val_loader)
        print(f"epoch {epoch + 1}/{cfg.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    preds = predict(model, X_test, cfg)
    save_submission(preds, cfg.submission_path)
    print(f"saved predictions to {cfg.submission_path}")


if __name__ == "__main__":
    main()
