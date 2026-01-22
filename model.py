"""
Model for habitat classification.

This file contains the predict() function that will be called by the API.
Replace baseline_model() with your own model implementation.
"""

import numpy as np
import pandas as pd
from pathlib import Path


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
    # TODO: Replace with your own model
    return baseline_model(patch)


def baseline_model(patch: np.ndarray) -> int:
    """
    Baseline model: Stratified random sampling based on training distribution.

    This achieves ~4% weighted F1 score. You should beat this!
    """
    # Load training labels to get class distribution
    train_df = pd.read_csv(Path(__file__).parent / "data" / "train.csv")
    class_counts = train_df["vistgerd_idx"].value_counts(normalize=True).sort_index()

    # Sample according to training distribution
    prediction = np.random.choice(
        class_counts.index.values,
        p=class_counts.values
    )
    return int(prediction)
