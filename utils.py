"""
Utility functions for habitat classification.
"""

import numpy as np
import pandas as pd
import base64
from pathlib import Path


def decode_patch(encoded: str) -> np.ndarray:
    """
    Decode base64 string to numpy array.

    Args:
        encoded: base64-encoded string representing a (15, 35, 35) float32 array

    Returns:
        numpy array of shape (15, 35, 35)
    """
    patch_bytes = base64.b64decode(encoded)
    patch = np.frombuffer(patch_bytes, dtype=np.float32)
    return patch.reshape(15, 35, 35)


def encode_patch(patch: np.ndarray) -> str:
    """
    Encode numpy array to base64 string.

    Args:
        patch: numpy array of shape (15, 35, 35)

    Returns:
        base64-encoded string
    """
    patch_float32 = patch.astype(np.float32)
    return base64.b64encode(patch_float32.tobytes()).decode("utf-8")


def load_training_data():
    """
    Load training patches and labels.

    Returns:
        patches: numpy array of shape (N, 15, 35, 35)
        labels: numpy array of shape (N,) with class indices 0-70
    """
    data_dir = Path(__file__).parent / "data"
    patches_path = data_dir / "train" / "patches.npy"
    if patches_path.exists():
        patches = np.load(patches_path)
    else:
        parts = sorted((data_dir / "train").glob("patches_part*.npy"))
        if not parts:
            raise FileNotFoundError(
                f"no training patches found at {patches_path} or patches_part*.npy"
            )
        arrays = [np.load(p) for p in parts]
        patches = np.concatenate(arrays, axis=0)
    labels_df = pd.read_csv(data_dir / "train.csv")
    return patches, labels_df["vistgerd_idx"].values


def load_class_names():
    """
    Load class name mappings.

    Returns:
        dict with 'vistgerd' and 'vistlendi' mappings
    """
    import json
    data_dir = Path(__file__).parent / "data"
    with open(data_dir / "class_names.json", "r") as f:
        return json.load(f)


def load_hierarchy():
    """
    Load hierarchy mapping (vistgerd -> vistlendi).

    Returns:
        dict mapping vistgerd index to vistlendi index
    """
    import json
    data_dir = Path(__file__).parent / "data"
    with open(data_dir / "hierarchy.json", "r") as f:
        return json.load(f)


def extract_features(patch: np.ndarray) -> np.ndarray:
    """
    Extract simple features from a patch.

    Args:
        patch: numpy array of shape (15, 35, 35)

    Returns:
        feature vector (30 values: 15 means + 15 stds)
    """
    means = patch.mean(axis=(1, 2))
    stds = patch.std(axis=(1, 2))
    return np.concatenate([means, stds])


def _normalized_diff_pairs(band_means: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    diffs = []
    for i in range(len(band_means) - 1):
        a = band_means[i]
        for j in range(i + 1, len(band_means)):
            b = band_means[j]
            diffs.append((a - b) / (abs(a) + abs(b) + eps))
    return np.array(diffs, dtype=np.float32)


def extract_features_extended(patch: np.ndarray) -> np.ndarray:
    """
    Extract richer per-band statistics to improve tree models.

    Returns:
        feature vector (15 bands * 7 stats = 105 values)
    """
    means = patch.mean(axis=(1, 2))
    stds = patch.std(axis=(1, 2))
    mins = patch.min(axis=(1, 2))
    maxs = patch.max(axis=(1, 2))
    medians = np.median(patch, axis=(1, 2))
    p25 = np.percentile(patch, 25, axis=(1, 2))
    p75 = np.percentile(patch, 75, axis=(1, 2))
    spectral_means = means[:12]
    nd_pairs = _normalized_diff_pairs(spectral_means)
    return np.concatenate([means, stds, mins, maxs, medians, p25, p75, nd_pairs])


def extract_features_batch_extended(patches: np.ndarray) -> np.ndarray:
    means = patches.mean(axis=(2, 3))
    stds = patches.std(axis=(2, 3))
    mins = patches.min(axis=(2, 3))
    maxs = patches.max(axis=(2, 3))
    medians = np.median(patches, axis=(2, 3))
    p25 = np.percentile(patches, 25, axis=(2, 3))
    p75 = np.percentile(patches, 75, axis=(2, 3))
    spectral_means = means[:, :12]
    nd_pairs = []
    for i in range(spectral_means.shape[1] - 1):
        a = spectral_means[:, i][:, None]
        for j in range(i + 1, spectral_means.shape[1]):
            b = spectral_means[:, j][:, None]
            nd_pairs.append((a - b) / (np.abs(a) + np.abs(b) + 1e-6))
    nd_pairs = np.concatenate(nd_pairs, axis=1)
    return np.concatenate([means, stds, mins, maxs, medians, p25, p75, nd_pairs], axis=1)
