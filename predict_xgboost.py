#!/usr/bin/env python3
"""
Predict habitat classes using a trained XGBoost model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix

from utils import extract_features_batch_extended


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with XGBoost model")
    parser.add_argument("--model", type=str, default="checkpoints/xgboost.json")
    parser.add_argument("--test-patches", type=str, required=True)
    parser.add_argument("--out", type=str, default="submission.csv")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found at {model_path}")
    classes_path = model_path.with_suffix(".classes.npy")
    if not classes_path.exists():
        raise FileNotFoundError(f"classes not found at {classes_path}")
    meta_path = model_path.with_suffix(".meta.json")

    patches = np.load(args.test_patches)
    features = extract_features_batch_extended(patches)

    classes = np.load(classes_path)
    best_iteration = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("best_iteration", -1) >= 0:
            best_iteration = int(meta["best_iteration"])

    booster = Booster()
    booster.load_model(str(model_path))
    dmat = DMatrix(features)
    if best_iteration is None:
        proba = booster.predict(dmat)
    else:
        proba = booster.predict(dmat, iteration_range=(0, best_iteration + 1))

    preds = classes[np.argmax(proba, axis=1).astype(int)]

    df = pd.DataFrame({"id": np.arange(len(preds)), "label": preds})
    df.to_csv(args.out, index=False)
    print(f"saved predictions to {args.out}")


if __name__ == "__main__":
    main()
