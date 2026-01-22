"""
Model for habitat classification.

This file contains the predict() function that will be called by the API.
Replace baseline_model() with your own model implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


def predict(patch: np.ndarray) -> int:
    """
    Predict habitat class for a single patch.

    Args:
        patch: numpy array of shape (15, 35, 35)
               - Channels 0-11: Sentinel-2 spectral bands
               - Channel 12: Elevation
               - Channel 13: Slope
               - Channel 14: Aspect

    Returns:
        Predicted class index (0-70)
    """
    model, cfg, mean, std = load_model()
    x, aspect_scalar = preprocess_patch(patch, cfg["aspect_mode"])
    x_t = torch.from_numpy(x).float().unsqueeze(0)
    x_t = F.interpolate(
        x_t, size=tuple(cfg["resize_hw"]), mode="bilinear", align_corners=False
    )
    mean_t = torch.from_numpy(mean).float().view(1, -1, 1, 1)
    std_t = torch.from_numpy(std).float().view(1, -1, 1, 1)
    x_t = (x_t - mean_t) / std_t
    aspect_t = torch.from_numpy(aspect_scalar).float().unsqueeze(0)
    with torch.no_grad():
        logits_fine, _ = model(x_t, aspect_t)
        pred = torch.argmax(logits_fine, dim=1).item()
    return int(pred)


def baseline_model(patch: np.ndarray) -> int:
    """
    Baseline model: Stratified random sampling based on training distribution.

    This achieves ~4% weighted F1 score. You should beat this!
    """
    # Load training labels to get class distribution
    raise RuntimeError("baseline_model is disabled; load a trained checkpoint instead")


def aspect_to_sin_cos(aspect_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(aspect_deg)
    return np.sin(theta), np.cos(theta)


def preprocess_patch(patch: np.ndarray, aspect_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    base = patch[:14]
    aspect = patch[14]
    sin_map, cos_map = aspect_to_sin_cos(aspect)
    if aspect_mode == "map":
        stacked = np.concatenate([base, sin_map[None, ...], cos_map[None, ...]], axis=0)
        return stacked, np.zeros(2, dtype=np.float32)
    if aspect_mode == "scalar":
        sin_scalar = float(sin_map.mean())
        cos_scalar = float(cos_map.mean())
        return base, np.array([sin_scalar, cos_scalar], dtype=np.float32)
    raise ValueError(f"unknown aspect_mode: {aspect_mode}")


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
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = safe_load(models.resnet50, weights)
        adapt_first_conv(model, in_channels)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim
    if name == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT
        model = safe_load(models.resnet101, weights)
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
    def __init__(self, backbone: nn.Module, feat_dim: int, aspect_mode: str):
        super().__init__()
        self.backbone = backbone
        self.aspect_mode = aspect_mode
        in_dim = feat_dim + (2 if aspect_mode == "scalar" else 0)
        self.head_vistlendi = nn.Linear(in_dim, 13)
        self.head_vistgerd = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 71),
        )

    def forward(self, x: torch.Tensor, aspect_scalar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        if self.aspect_mode == "scalar":
            feats = torch.cat([feats, aspect_scalar], dim=1)
        logits_coarse = self.head_vistlendi(feats)
        logits_fine = self.head_vistgerd(feats)
        return logits_fine, logits_coarse


_CACHED_MODEL = None
_CACHED_CFG = None
_CACHED_MEAN = None
_CACHED_STD = None


def load_model():
    global _CACHED_MODEL, _CACHED_CFG, _CACHED_MEAN, _CACHED_STD
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_CFG, _CACHED_MEAN, _CACHED_STD

    ckpt_path = Path(__file__).parent / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    mean = ckpt["mean"].cpu().numpy()
    std = ckpt["std"].cpu().numpy()

    sample_channels = 16 if cfg.get("aspect_mode", "map") == "map" else 14
    backbone, feat_dim = build_backbone(cfg.get("backbone", "resnet18"), sample_channels)
    model = MultiHeadModel(backbone, feat_dim, cfg.get("aspect_mode", "map"))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_CFG = cfg
    _CACHED_MEAN = mean
    _CACHED_STD = std
    return model, cfg, mean, std
