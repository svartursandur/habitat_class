"""
Model for habitat classification using an XGBoost classifier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from utils import extract_features_extended

_CACHED_MODEL = None


def predict(patch: np.ndarray) -> int:
    """
    Predict habitat class for a single patch.

    Args:
        patch: numpy array of shape (15, 35, 35)

    Returns:
        Predicted class index (0-70)
    """
    model, classes = load_model()
    features = extract_features_extended(patch).reshape(1, -1)
    pred_enc = model.predict(features)[0]
    pred = classes[int(pred_enc)]
    return int(pred)


def load_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    model_path = Path(__file__).parent / "checkpoints" / "xgboost.json"
    classes_path = model_path.with_suffix(".classes.npy")
    if not model_path.exists():
        raise FileNotFoundError(
            f"model not found at {model_path}; run train_xgboost.py first"
        )
    if not classes_path.exists():
        raise FileNotFoundError(
            f"classes not found at {classes_path}; run train_xgboost.py first"
        )

    model = XGBClassifier()
    model.load_model(str(model_path))
    classes = np.load(classes_path)
    _CACHED_MODEL = (model, classes)
    return _CACHED_MODEL
