#!/usr/bin/env python3
"""
ResNet18 training script with mixup, class-balanced loss, and OneCycleLR.
Supports 15-channel inputs with aspect converted to sin/cos (16 channels).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models

from utils import load_training_data
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import warnings


@dataclass
class Config:
    num_classes: int = 71
    batch_size: int = 64
    epochs: int = 30
    lr: float = 3e-4
    max_lr: float = 2e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    seed: int = 42
    resize_hw: Tuple[int, int] = (48, 48)
    submission_path: str = "submission.csv"
    test_patches_path: str | None = None
    debug_overfit_samples: int = 0
    label_order: str = "sample_id"  # or "original_idx"
    loss_plot_path: str = "loss_curve.png"
    loss_csv_path: str = "loss_curve.csv"
    use_balanced_sampler: bool = False
    use_class_weights: bool = True
    class_weight_beta: float = 0.999
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.001
    diagnose_only: bool = False
    mixup_alpha: float = 0.4
    label_smoothing: float = 0.05
    grad_clip_norm: float = 1.0
    num_workers: int = 2


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_channel_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 2, 3))
    std = x.std(axis=(0, 2, 3))
    std = np.where(std == 0, 1.0, std)
    return mean, std


def preprocess_aspect(patches: np.ndarray) -> np.ndarray:
    """
    Convert aspect channel to sin/cos and append as two channels.
    Input: (N, 15, H, W). Output: (N, 16, H, W).
    """
    aspect = patches[:, 14, :, :]
    aspect_rad = np.deg2rad(aspect)
    sin_a = np.sin(aspect_rad)
    cos_a = np.cos(aspect_rad)
    return np.concatenate(
        [patches[:, :14, :, :], sin_a[:, None, :, :], cos_a[:, None, :, :]], axis=1
    )


def run_diagnostics(x: np.ndarray, y: np.ndarray) -> None:
    print("diagnostics:")
    print(f"  shape: {x.shape}")
    print(f"  dtype: {x.dtype}")
    print(f"  min/max: {x.min():.4f} / {x.max():.4f}")
    if np.isnan(x).any():
        print("  warning: NaNs found in inputs")
    mean = x.mean(axis=(0, 2, 3))
    std = x.std(axis=(0, 2, 3))
    print(f"  per-channel mean (first 5): {np.round(mean[:5], 4)}")
    print(f"  per-channel std  (first 5): {np.round(std[:5], 4)}")
    zero_std = np.where(std == 0)[0]
    if len(zero_std) > 0:
        print(f"  warning: zero-std channels at indices {zero_std.tolist()}")
    counts = np.bincount(y, minlength=Config.num_classes)
    print(f"  class count min/max: {counts.min()} / {counts.max()}")
    rare = np.where(counts < 2)[0]
    if len(rare) > 0:
        print(f"  rare classes (<2 samples): {rare.tolist()}")


def resize_and_normalize(
    x: torch.Tensor,
    size_hw: Tuple[int, int],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    x = x.squeeze(0)
    x = (x - mean) / std
    return x


class HabitatDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray | None,
        size_hw: Tuple[int, int],
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.x = x
        self.y = y
        self.size_hw = size_hw
        self.mean = torch.from_numpy(mean).float().view(-1, 1, 1)
        self.std = torch.from_numpy(std).float().view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        patch = self.x[idx]
        patch_t = torch.from_numpy(patch)
        x = resize_and_normalize(patch_t, self.size_hw, self.mean, self.std)
        if self.y is None:
            return x
        return x, int(self.y[idx])


def adapt_first_conv(model: nn.Module, in_ch: int) -> None:
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
        new_w = w_mean.repeat(1, in_ch, 1, 1)
        new_conv.weight.copy_(new_w)
        if conv1.bias is not None:
            new_conv.bias.copy_(conv1.bias)
    model.conv1 = new_conv


def build_model(num_classes: int, in_channels: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    adapt_first_conv(model, in_channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_class_weights(y: np.ndarray, num_classes: int, beta: float) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes)
    eff_num = 1.0 - np.power(beta, counts)
    eff_num = np.where(eff_num == 0, 1.0, eff_num)
    weights = (1.0 - beta) / eff_num
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float()


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    dataset = HabitatDataset(x_train, y_train, cfg.resize_hw, mean, std)
    if cfg.debug_overfit_samples > 0:
        rng = np.random.default_rng(cfg.seed)
        n = min(cfg.debug_overfit_samples, len(dataset))
        idx = rng.choice(len(dataset), size=n, replace=False)
        subset = Subset(dataset, idx)
        loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=True)
        return loader, loader, np.asarray(y_train)[idx]

    y_arr = np.asarray(y_train)
    classes, counts = np.unique(y_arr, return_counts=True)
    rare_classes = classes[counts < 2]

    if len(rare_classes) == 0:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=cfg.val_split, random_state=cfg.seed
        )
        idx_train, idx_val = next(splitter.split(np.zeros(len(y_arr)), y_arr))
    else:
        warnings.warn(
            f"classes with <2 samples found {rare_classes.tolist()}, "
            "placing them in train split only"
        )
        rare_mask = np.isin(y_arr, rare_classes)
        rare_idx = np.where(rare_mask)[0]
        keep_idx = np.where(~rare_mask)[0]

        if len(keep_idx) < 2:
            raise ValueError("not enough samples to create a validation split")

        desired_val = max(1, int(round(cfg.val_split * len(y_arr))))
        max_val = len(keep_idx) - 1
        desired_val = min(desired_val, max_val)
        test_size = desired_val / len(keep_idx)

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=cfg.seed
        )
        keep_train, keep_val = next(
            splitter.split(np.zeros(len(keep_idx)), y_arr[keep_idx])
        )
        idx_train = np.concatenate([rare_idx, keep_idx[keep_train]])
        idx_val = keep_idx[keep_val]

    train_ds = Subset(dataset, idx_train)
    val_ds = Subset(dataset, idx_val)
    if cfg.use_balanced_sampler:
        train_labels = y_arr[idx_train]
        class_counts = np.bincount(train_labels, minlength=cfg.num_classes)
        class_counts = np.where(class_counts == 0, 1, class_counts)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(train_labels),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, y_arr[idx_train]


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], float(lam)


def loss_fn(
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    loss_a = F.cross_entropy(
        logits, y_a, weight=class_weights, label_smoothing=label_smoothing
    )
    loss_b = F.cross_entropy(
        logits, y_b, weight=class_weights, label_smoothing=label_smoothing
    )
    return lam * loss_a + (1.0 - lam) * loss_b


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    class_weights: torch.Tensor | None,
    cfg: Config,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x, y_a, y_b, lam = mixup_batch(x, y, cfg.mixup_alpha)

        optim.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y_a, y_b, lam, class_weights, cfg.label_smoothing)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optim)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y_a, y_b, lam, class_weights, cfg.label_smoothing)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optim.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)
    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average="weighted")
    cm = confusion_matrix(y_true_np, y_pred_np, labels=np.arange(0, 71))
    return total_loss / len(loader.dataset), acc, f1, cm


@torch.no_grad()
def predict(model: nn.Module, x_test: np.ndarray, mean: np.ndarray, std: np.ndarray, cfg: Config, device: torch.device) -> np.ndarray:
    model.eval()
    x_test = preprocess_aspect(x_test)
    test_ds = HabitatDataset(x_test, None, cfg.resize_hw, mean, std)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    preds = []
    for x in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def save_submission(preds: np.ndarray, path: str) -> None:
    df = pd.DataFrame({"id": np.arange(len(preds)), "label": preds})
    df.to_csv(path, index=False)


def load_test_patches(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"test patches not found at {path}")
    return np.load(p)


def load_train_patches_from_parts() -> np.ndarray | None:
    data_dir = Path(__file__).parent / "data" / "train"
    parts = sorted(data_dir.glob("patches_part*.npy"))
    if not parts:
        return None
    arrays = [np.load(p) for p in parts]
    return np.concatenate(arrays, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 on habitat patches")
    parser.add_argument("--debug-overfit", type=int, default=0, help="overfit N samples")
    parser.add_argument(
        "--label-order",
        choices=["sample_id", "original_idx"],
        default="sample_id",
        help="label ordering to align with patches",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="resize height and width",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size")
    parser.add_argument("--val-split", type=float, default=None, help="override val split")
    parser.add_argument("--test-patches", type=str, default=None, help="path to test patches")
    parser.add_argument("--diagnose", action="store_true", help="print data diagnostics and exit")
    args = parser.parse_args()

    cfg = Config(
        debug_overfit_samples=args.debug_overfit,
        label_order=args.label_order,
        test_patches_path=args.test_patches,
    )
    if args.resize is not None:
        cfg.resize_hw = (args.resize[0], args.resize[1])
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.val_split is not None:
        cfg.val_split = args.val_split
    cfg.diagnose_only = args.diagnose

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    try:
        x_train, _ = load_training_data()
    except FileNotFoundError:
        x_train = load_train_patches_from_parts()
        if x_train is None:
            raise FileNotFoundError("no training patches found in data/train")

    labels_df = pd.read_csv(Path(__file__).parent / "data" / "train.csv")
    if cfg.label_order == "original_idx":
        labels_df = labels_df.sort_values("original_idx")
    y_train = labels_df["vistgerd_idx"].values
    x_train = preprocess_aspect(x_train)

    mean, std = compute_channel_stats(x_train)

    if cfg.diagnose_only:
        run_diagnostics(x_train, y_train)
        return

    model = build_model(cfg.num_classes, in_channels=x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader, val_loader, train_labels = make_loaders(x_train, y_train, mean, std, cfg)

    class_weights = None
    if cfg.use_class_weights:
        class_weights = compute_class_weights(train_labels, cfg.num_classes, cfg.class_weight_beta)
        class_weights = class_weights.to(device)

    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = []
    best_f1 = -1.0
    best_state = None
    no_improve = 0
    t_start = time.time()
    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, class_weights, cfg, scaler, device
        )
        val_loss, val_acc, val_f1, _ = eval_one_epoch(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        history.append((epoch + 1, train_loss, val_loss, val_acc, val_f1, epoch_time))
        print(
            "epoch "
            f"{epoch + 1}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f} "
            f"time_s={epoch_time:.1f}"
        )
        if val_f1 > best_f1 + cfg.early_stop_min_delta:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"early stopping at epoch {epoch + 1} (best_f1={best_f1:.4f})")
                break

    total_time = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc, val_f1, cm = eval_one_epoch(model, val_loader, device)
    if cfg.debug_overfit_samples > 0:
        print(
            f"final train_loss={val_loss:.4f} train_acc={val_acc:.4f} train_f1={val_f1:.4f}"
        )
    else:
        print(f"final val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        print("confusion_matrix:")
        print(cm)
        print(f"total_train_time_s={total_time:.1f}")

    hist_df = pd.DataFrame(
        history,
        columns=["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "time_s"],
    )
    hist_df.to_csv(cfg.loss_csv_path, index=False)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
        plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.loss_plot_path)
        plt.close()
        print(f"saved loss plot to {cfg.loss_plot_path}")
    except Exception as exc:
        print(f"could not plot loss curve: {exc}")

    x_test = load_test_patches(cfg.test_patches_path)
    if x_test is None:
        print("no test patches provided; skipping submission.csv")
        return

    preds = predict(model, x_test, mean, std, cfg, device)
    save_submission(preds, cfg.submission_path)
    print(f"saved predictions to {cfg.submission_path}")


if __name__ == "__main__":
    main()
