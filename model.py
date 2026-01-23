"""
Model for habitat classification using an XGBoost classifier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import json
from xgboost import Booster, DMatrix

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
    booster, classes, best_iteration = load_model()
    features = extract_features_extended(patch).reshape(1, -1)
    dmat = DMatrix(features)
    if best_iteration is None:
        proba = booster.predict(dmat)
    else:
        proba = booster.predict(dmat, iteration_range=(0, best_iteration + 1))
    pred = classes[int(np.argmax(proba, axis=1)[0])]
    return int(pred)


def load_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    model_path = Path(__file__).parent / "checkpoints" / "xgboost.json"
    classes_path = model_path.with_suffix(".classes.npy")
    meta_path = model_path.with_suffix(".meta.json")
    if not model_path.exists():
        raise FileNotFoundError(
            f"model not found at {model_path}; run train_xgboost.py first"
        )
    if not classes_path.exists():
        raise FileNotFoundError(
            f"classes not found at {classes_path}; run train_xgboost.py first"
        )

    booster = Booster()
    booster.load_model(str(model_path))
    classes = np.load(classes_path)
    best_iteration = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("best_iteration", -1) >= 0:
            best_iteration = int(meta["best_iteration"])
    _CACHED_MODEL = (booster, classes, best_iteration)
    return _CACHED_MODEL
