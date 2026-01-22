#!/usr/bin/env python3
"""
Predict habitat classes using a trained RandomForestClassifier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from utils import extract_features_batch_extended


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with RandomForest model")
    parser.add_argument("--model", type=str, default="checkpoints/random_forest.joblib")
    parser.add_argument("--test-patches", type=str, required=True)
    parser.add_argument("--out", type=str, default="submission.csv")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found at {model_path}")

    patches = np.load(args.test_patches)
    features = extract_features_batch_extended(patches)

    model = load(model_path)
    preds = model.predict(features)

    df = pd.DataFrame({"id": np.arange(len(preds)), "label": preds})
    df.to_csv(args.out, index=False)
    print(f"saved predictions to {args.out}")


if __name__ == "__main__":
    main()
