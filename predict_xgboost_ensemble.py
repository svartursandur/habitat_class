#!/usr/bin/env python3
"""
Predict habitat classes using an ensemble of XGBoost models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix

from utils import extract_features_batch_extended


def load_meta(path: Path) -> int | None:
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    best = meta.get("best_iteration", -1)
    return int(best) if best >= 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with XGBoost ensemble")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--test-patches", type=str, required=True)
    parser.add_argument("--out", type=str, default="submission.csv")
    args = parser.parse_args()

    model_paths = sorted(Path().glob(args.models))
    if not model_paths:
        raise FileNotFoundError(f"no models found for pattern: {args.models}")

    classes_path = model_paths[0].with_suffix(".classes.npy")
    if not classes_path.exists():
        raise FileNotFoundError(f"classes not found at {classes_path}")
    classes = np.load(classes_path)

    patches = np.load(args.test_patches)
    features = extract_features_batch_extended(patches)
    dmat = DMatrix(features)

    proba_sum = None
    for model_path in model_paths:
        booster = Booster()
        booster.load_model(str(model_path))
        best_iteration = load_meta(model_path)
        if best_iteration is None:
            proba = booster.predict(dmat)
        else:
            proba = booster.predict(dmat, iteration_range=(0, best_iteration + 1))
        proba_sum = proba if proba_sum is None else proba_sum + proba

    proba_avg = proba_sum / len(model_paths)
    preds = classes[np.argmax(proba_avg, axis=1).astype(int)]

    df = pd.DataFrame({"id": np.arange(len(preds)), "label": preds})
    df.to_csv(args.out, index=False)
    print(f"saved predictions to {args.out}")


if __name__ == "__main__":
    main()
