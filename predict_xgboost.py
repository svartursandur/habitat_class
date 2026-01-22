#!/usr/bin/env python3
"""
Predict habitat classes using a trained XGBoost model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

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

    patches = np.load(args.test_patches)
    features = extract_features_batch_extended(patches)

    model = XGBClassifier()
    model.load_model(str(model_path))
    classes = np.load(classes_path)
    preds_enc = model.predict(features)
    preds = classes[preds_enc.astype(int)]

    df = pd.DataFrame({"id": np.arange(len(preds)), "label": preds})
    df.to_csv(args.out, index=False)
    print(f"saved predictions to {args.out}")


if __name__ == "__main__":
    main()
