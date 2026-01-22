#!/usr/bin/env python3
"""
Inference + submission writer for habitat classification.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


def aspect_to_sin_cos(aspect_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(aspect_deg)
    return np.sin(theta), np.cos(theta)


def preprocess_patch(patch: np.ndarray, aspect_mode: str) -> Tuple[np.ndarray, np.ndarray | None]:
    base = patch[:14]
    aspect = patch[14]
    sin_map, cos_map = aspect_to_sin_cos(aspect)
    if aspect_mode == "map":
        stacked = np.concatenate([base, sin_map[None, ...], cos_map[None, ...]], axis=0)
        return stacked, None
    if aspect_mode == "scalar":
        sin_scalar = float(sin_map.mean())
        cos_scalar = float(cos_map.mean())
        return base, np.array([sin_scalar, cos_scalar], dtype=np.float32)
    raise ValueError(f"unknown aspect_mode: {aspect_mode}")


class HabitatTestDataset(Dataset):
    def __init__(
        self,
        patches: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        resize_hw: Tuple[int, int],
        aspect_mode: str,
    ):
        self.patches = patches
        self.mean = torch.from_numpy(mean).float().view(-1, 1, 1)
        self.std = torch.from_numpy(std).float().view(-1, 1, 1)
        self.resize_hw = resize_hw
        self.aspect_mode = aspect_mode

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]
        patch_proc, aspect_scalar = preprocess_patch(patch, self.aspect_mode)
        x = torch.from_numpy(patch_proc).float()
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=self.resize_hw, mode="bilinear", align_corners=False)
        x = x.squeeze(0)
        x = (x - self.mean) / self.std
        if aspect_scalar is None:
            aspect_tensor = torch.zeros(2, dtype=torch.float32)
        else:
            aspect_tensor = torch.from_numpy(aspect_scalar).float()
        return x, aspect_tensor


def adapt_first_conv(model: nn.Module, in_ch: int) -> None:
    if hasattr(model, "conv1"):
        conv1 = model.conv1
        if conv1.in_channels == in_ch:
            return
        new_conv = nn.Conv2d(
            in_ch,
            conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None,
        )
        with torch.no_grad():
            w = conv1.weight
            w_mean = w.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(w_mean.repeat(1, in_ch, 1, 1))
            if conv1.bias is not None:
                new_conv.bias.copy_(conv1.bias)
        model.conv1 = new_conv
        return
    if hasattr(model, "features"):
        first = model.features[0][0]
        if first.in_channels == in_ch:
            return
        new_conv = nn.Conv2d(
            in_ch,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=first.bias is not None,
        )
        with torch.no_grad():
            w = first.weight
            w_mean = w.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(w_mean.repeat(1, in_ch, 1, 1))
            if first.bias is not None:
                new_conv.bias.copy_(first.bias)
        model.features[0][0] = new_conv
        return
    raise ValueError("unknown model architecture for conv adaptation")


def build_backbone(name: str, in_channels: int) -> Tuple[nn.Module, int]:
    def safe_load(factory, weights):
        try:
            return factory(weights=weights)
        except Exception:
            return factory(weights=None)

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = safe_load(models.resnet18, weights)
        adapt_first_conv(model, in_channels)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim
    if name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT
        model = safe_load(models.resnet34, weights)
        adapt_first_conv(model, in_channels)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = safe_load(models.efficientnet_b0, weights)
        adapt_first_conv(model, in_channels)
        feat_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, feat_dim
    raise ValueError(f"unknown backbone: {name}")


class MultiHeadModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        aspect_mode: str,
        coarse_to_fine: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.aspect_mode = aspect_mode
        self.coarse_to_fine = coarse_to_fine
        in_dim = feat_dim + (2 if aspect_mode == "scalar" else 0)
        self.head_vistlendi = nn.Linear(in_dim, 13)
        fine_in = in_dim + (13 if coarse_to_fine else 0)
        self.head_vistgerd = nn.Sequential(
            nn.Linear(fine_in, fine_in),
            nn.ReLU(),
            nn.Linear(fine_in, 71),
        )

    def forward(self, x: torch.Tensor, aspect_scalar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        if self.aspect_mode == "scalar":
            feats = torch.cat([feats, aspect_scalar], dim=1)
        logits_coarse = self.head_vistlendi(feats)
        fine_feats = feats
        if self.coarse_to_fine:
            coarse_probs = torch.softmax(logits_coarse, dim=1)
            fine_feats = torch.cat([feats, coarse_probs], dim=1)
        logits_fine = self.head_vistgerd(fine_feats)
        return logits_fine, logits_coarse


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict habitat labels")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-patches", type=str, required=True)
    parser.add_argument("--out", type=str, default="submission.csv")
    parser.add_argument("--output", choices=["vistgerd", "vistlendi"], default="vistgerd")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    mean = ckpt["mean"]
    std = ckpt["std"]

    aspect_mode = cfg.get("aspect_mode", "map")
    backbone_name = cfg.get("backbone", "resnet18")
    resize_hw = tuple(cfg.get("resize_hw", (48, 48)))
    coarse_to_fine = cfg.get("coarse_to_fine", False)

    x_test = np.load(args.test_patches)
    sample_patch, _ = preprocess_patch(x_test[0], aspect_mode)
    in_channels = sample_patch.shape[0]

    backbone, feat_dim = build_backbone(backbone_name, in_channels)
    model = MultiHeadModel(backbone, feat_dim, aspect_mode, coarse_to_fine)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    test_ds = HabitatTestDataset(x_test, mean, std, resize_hw, aspect_mode)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for x, aspect_scalar in test_loader:
            logits_f, logits_c = model(x, aspect_scalar)
            if args.output == "vistlendi":
                pred = torch.argmax(logits_c, dim=1)
            else:
                pred = torch.argmax(logits_f, dim=1)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    df = pd.DataFrame({"id": np.arange(len(preds)), "label": preds})
    df.to_csv(args.out, index=False)
    print(f"saved predictions to {args.out}")


if __name__ == "__main__":
    main()
