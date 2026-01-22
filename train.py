#!/usr/bin/env python3
"""
CPU-friendly training pipeline for habitat classification.
Supports vistgerd (71), vistlendi (13), and multitask training.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from utils import load_training_data


@dataclass
class Config:
    task: str = "multitask"  # vistgerd | vistlendi | multitask
    backbone: str = "resnet18"  # resnet18 | resnet34 | efficientnet_b0
    aspect_mode: str = "map"  # map | scalar
    batch_size: int = 64
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_split: float = 0.2
    seed: int = 42
    resize_hw: Tuple[int, int] = (48, 48)
    num_workers: int = 4
    crop_pad: int = 2
    use_class_weights: bool = True
    use_focal: bool = False
    focal_gamma: float = 2.0
    scheduler: str = "plateau"  # plateau | cosine | none
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.001
    checkpoint_path: str = "checkpoints/best.pt"
    coarse_to_fine: bool = False


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_train_patches_from_parts() -> np.ndarray | None:
    data_dir = Path(__file__).parent / "data" / "train"
    parts = sorted(data_dir.glob("patches_part*.npy"))
    if not parts:
        return None
    arrays = [np.load(p) for p in parts]
    return np.concatenate(arrays, axis=0)


def compute_channel_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 2, 3))
    std = x.std(axis=(0, 2, 3))
    std = np.where(std == 0, 1.0, std)
    return mean, std


def compute_channel_stats_stream(patches: np.ndarray, aspect_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    sum_c = None
    sumsq_c = None
    total = 0
    for patch in patches:
        proc, _ = preprocess_patch(patch, aspect_mode)
        if sum_c is None:
            sum_c = proc.sum(axis=(1, 2))
            sumsq_c = (proc ** 2).sum(axis=(1, 2))
        else:
            sum_c += proc.sum(axis=(1, 2))
            sumsq_c += (proc ** 2).sum(axis=(1, 2))
        total += proc.shape[1] * proc.shape[2]
    mean = sum_c / total
    var = sumsq_c / total - mean ** 2
    var = np.where(var <= 0, 1.0, var)
    std = np.sqrt(var)
    return mean, std


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


def random_flip_rot(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        patch = patch[:, :, ::-1]
    if rng.random() < 0.5:
        patch = patch[:, ::-1, :]
    k = rng.integers(0, 4)
    if k:
        patch = np.rot90(patch, k, axes=(1, 2))
    return patch.copy()


def random_shift_crop(patch: np.ndarray, pad: int, rng: np.random.Generator) -> np.ndarray:
    if pad <= 0:
        return patch
    c, h, w = patch.shape
    padded = np.pad(patch, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    top = rng.integers(0, 2 * pad + 1)
    left = rng.integers(0, 2 * pad + 1)
    return padded[:, top:top + h, left:left + w]


class HabitatDataset(Dataset):
    def __init__(
        self,
        patches: np.ndarray,
        labels_vistgerd: np.ndarray,
        labels_vistlendi: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        resize_hw: Tuple[int, int],
        aspect_mode: str,
        train: bool,
        crop_pad: int,
        seed: int,
    ):
        self.patches = patches
        self.labels_vistgerd = labels_vistgerd
        self.labels_vistlendi = labels_vistlendi
        self.mean = torch.from_numpy(mean).float().view(-1, 1, 1)
        self.std = torch.from_numpy(std).float().view(-1, 1, 1)
        self.resize_hw = resize_hw
        self.aspect_mode = aspect_mode
        self.train = train
        self.crop_pad = crop_pad
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]
        if self.train:
            patch = random_shift_crop(patch, self.crop_pad, self.rng)
            patch = random_flip_rot(patch, self.rng)
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
        return (
            x,
            aspect_tensor,
            int(self.labels_vistgerd[idx]),
            int(self.labels_vistlendi[idx]),
        )


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if hasattr(dataset, "rng"):
        seed = int(dataset.rng.integers(0, 2**32 - 1)) + worker_id
        dataset.rng = np.random.default_rng(seed)


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


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None, gamma: float) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    p = torch.exp(logp)
    loss = F.nll_loss((1 - p) ** gamma * logp, targets, weight=weight)
    return loss


def loss_for_task(
    logits_fine: torch.Tensor,
    logits_coarse: torch.Tensor,
    y_fine: torch.Tensor,
    y_coarse: torch.Tensor,
    cfg: Config,
    w_fine: torch.Tensor | None,
    w_coarse: torch.Tensor | None,
) -> torch.Tensor:
    if cfg.use_focal:
        loss_fine = focal_loss(logits_fine, y_fine, w_fine, cfg.focal_gamma)
        loss_coarse = focal_loss(logits_coarse, y_coarse, w_coarse, cfg.focal_gamma)
    else:
        loss_fine = F.cross_entropy(logits_fine, y_fine, weight=w_fine)
        loss_coarse = F.cross_entropy(logits_coarse, y_coarse, weight=w_coarse)

    if cfg.task == "vistgerd":
        return loss_fine + 0.3 * loss_coarse
    if cfg.task == "vistlendi":
        return loss_coarse + 0.3 * loss_fine
    return loss_fine + loss_coarse


def print_class_summary(labels: np.ndarray, num_classes: int, name: str) -> None:
    counts = np.bincount(labels, minlength=num_classes)
    print(f"{name} class count min/max: {counts.min()} / {counts.max()}")
    rare = np.where(counts < 2)[0]
    if len(rare) > 0:
        print(f"{name} rare classes (<2 samples): {rare.tolist()}")


def stratified_split_with_rare(
    labels: np.ndarray, val_split: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(labels, return_counts=True)
    rare_classes = classes[counts < 2]
    if len(rare_classes) == 0:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_split, random_state=seed
        )
        idx_train, idx_val = next(splitter.split(np.zeros(len(labels)), labels))
        return idx_train, idx_val

    rare_mask = np.isin(labels, rare_classes)
    rare_idx = np.where(rare_mask)[0]
    keep_idx = np.where(~rare_mask)[0]

    if len(keep_idx) < 2:
        raise ValueError("not enough samples to create a validation split")

    desired_val = max(1, int(round(val_split * len(labels))))
    max_val = len(keep_idx) - 1
    desired_val = min(desired_val, max_val)
    test_size = desired_val / len(keep_idx)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    keep_train, keep_val = next(
        splitter.split(np.zeros(len(keep_idx)), labels[keep_idx])
    )
    idx_train = np.concatenate([rare_idx, keep_idx[keep_train]])
    idx_val = keep_idx[keep_val]
    return idx_train, idx_val


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
) -> Dict[str, float]:
    model.eval()
    y_true_fine = []
    y_pred_fine = []
    y_true_coarse = []
    y_pred_coarse = []
    total_loss = 0.0
    with torch.no_grad():
        for x, aspect_scalar, y_fine, y_coarse in loader:
            logits_fine, logits_coarse = model(x, aspect_scalar)
            loss = loss_for_task(
                logits_fine,
                logits_coarse,
                y_fine,
                y_coarse,
                cfg,
                None,
                None,
            )
            total_loss += loss.item() * x.size(0)
            y_true_fine.append(y_fine.numpy())
            y_true_coarse.append(y_coarse.numpy())
            y_pred_fine.append(torch.argmax(logits_fine, dim=1).numpy())
            y_pred_coarse.append(torch.argmax(logits_coarse, dim=1).numpy())

    y_true_fine = np.concatenate(y_true_fine, axis=0)
    y_pred_fine = np.concatenate(y_pred_fine, axis=0)
    y_true_coarse = np.concatenate(y_true_coarse, axis=0)
    y_pred_coarse = np.concatenate(y_pred_coarse, axis=0)

    combined_weighted = 0.5 * (metrics_fine_weighted := f1_score(y_true_fine, y_pred_fine, average="weighted")) + 0.5 * (
        metrics_coarse_weighted := f1_score(y_true_coarse, y_pred_coarse, average="weighted")
    )
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "acc_fine": accuracy_score(y_true_fine, y_pred_fine),
        "f1_fine": f1_score(y_true_fine, y_pred_fine, average="macro"),
        "f1_fine_weighted": metrics_fine_weighted,
        "acc_coarse": accuracy_score(y_true_coarse, y_pred_coarse),
        "f1_coarse": f1_score(y_true_coarse, y_pred_coarse, average="macro"),
        "f1_coarse_weighted": metrics_coarse_weighted,
        "f1_combined_weighted": combined_weighted,
    }
    return metrics


def save_checkpoint(path: str, model: nn.Module, cfg: Config, mean: np.ndarray, std: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "mean": mean,
        "std": std,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train habitat classification model")
    parser.add_argument("--task", choices=["vistgerd", "vistlendi", "multitask"], default=None)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34", "efficientnet_b0"], default=None)
    parser.add_argument("--aspect-mode", choices=["map", "scalar"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default=None)
    parser.add_argument("--use-focal", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--coarse-to-fine", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.task:
        cfg.task = args.task
    if args.backbone:
        cfg.backbone = args.backbone
    if args.aspect_mode:
        cfg.aspect_mode = args.aspect_mode
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.val_split is not None:
        cfg.val_split = args.val_split
    if args.resize is not None:
        cfg.resize_hw = (args.resize[0], args.resize[1])
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.scheduler is not None:
        cfg.scheduler = args.scheduler
    if args.use_focal:
        cfg.use_focal = True
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    if args.coarse_to_fine:
        cfg.coarse_to_fine = True

    set_seed(cfg.seed)

    try:
        x_train, _ = load_training_data()
    except FileNotFoundError:
        x_train = load_train_patches_from_parts()
        if x_train is None:
            raise FileNotFoundError("no training patches found in data/train")

    labels_df = pd.read_csv(Path(__file__).parent / "data" / "train.csv")
    y_fine = labels_df["vistgerd_idx"].values
    y_coarse = labels_df["vistlendi_idx"].values

    print_class_summary(y_fine, 71, "vistgerd")
    print_class_summary(y_coarse, 13, "vistlendi")

    if cfg.task == "vistlendi":
        split_labels = y_coarse
    else:
        split_labels = y_fine

    idx_train, idx_val = stratified_split_with_rare(
        split_labels, cfg.val_split, cfg.seed
    )

    train_patches = x_train[idx_train]
    val_patches = x_train[idx_val]
    y_fine_train = y_fine[idx_train]
    y_coarse_train = y_coarse[idx_train]
    y_fine_val = y_fine[idx_val]
    y_coarse_val = y_coarse[idx_val]

    sample_patch, _ = preprocess_patch(train_patches[0], cfg.aspect_mode)
    mean, std = compute_channel_stats_stream(train_patches, cfg.aspect_mode)

    train_ds = HabitatDataset(
        train_patches,
        y_fine_train,
        y_coarse_train,
        mean,
        std,
        cfg.resize_hw,
        cfg.aspect_mode,
        train=True,
        crop_pad=cfg.crop_pad,
        seed=cfg.seed,
    )
    val_ds = HabitatDataset(
        val_patches,
        y_fine_val,
        y_coarse_val,
        mean,
        std,
        cfg.resize_hw,
        cfg.aspect_mode,
        train=False,
        crop_pad=0,
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )

    in_channels = sample_patch.shape[0]
    backbone, feat_dim = build_backbone(cfg.backbone, in_channels)
    model = MultiHeadModel(backbone, feat_dim, cfg.aspect_mode, cfg.coarse_to_fine)

    w_fine = compute_class_weights(y_fine_train, 71) if cfg.use_class_weights else None
    w_coarse = compute_class_weights(y_coarse_train, 13) if cfg.use_class_weights else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    elif cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = None

    best_score = -1.0
    best_state = None
    no_improve = 0
    t_start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for x, aspect_scalar, y_f, y_c in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits_f, logits_c = model(x, aspect_scalar)
            loss = loss_for_task(logits_f, logits_c, y_f, y_c, cfg, w_fine, w_coarse)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        train_loss = epoch_loss / len(train_loader.dataset)
        metrics = evaluate(model, val_loader, cfg)
        epoch_time = time.time() - t_start

        if cfg.task == "vistlendi":
            score = metrics["f1_coarse_weighted"]
        elif cfg.task == "multitask":
            score = metrics["f1_combined_weighted"]
        else:
            score = metrics["f1_fine_weighted"]

        if cfg.task == "vistlendi":
            f1_msg = f"val_f1_weighted={metrics['f1_coarse_weighted']:.4f}"
        elif cfg.task == "multitask":
            f1_msg = f"val_f1_weighted={metrics['f1_combined_weighted']:.4f}"
        else:
            f1_msg = f"val_f1_weighted={metrics['f1_fine_weighted']:.4f}"
        print(
            "epoch "
            f"{epoch + 1}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"{f1_msg} "
            f"time_s={epoch_time:.1f}"
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

        if score > best_score + cfg.early_stop_min_delta:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"early stopping at epoch {epoch + 1} (best_score={best_score:.4f})")
                break

    total_time = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate(model, val_loader, cfg)
    if cfg.task == "vistlendi":
        final_msg = f"val_f1_weighted={metrics['f1_coarse_weighted']:.4f}"
    elif cfg.task == "multitask":
        final_msg = f"val_f1_weighted={metrics['f1_combined_weighted']:.4f}"
    else:
        final_msg = f"val_f1_weighted={metrics['f1_fine_weighted']:.4f}"
    print(
        f"final val_loss={metrics['loss']:.4f} "
        f"{final_msg} "
        f"total_time_s={total_time:.1f}"
    )

    save_checkpoint(cfg.checkpoint_path, model, cfg, mean, std)
    print(f"saved checkpoint to {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
