#!/usr/bin/env python3
"""
Train a RandomForestClassifier on simple band statistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from utils import extract_features_batch_extended, load_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ExtraTrees for habitat classification")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--class-weight", type=str, default="balanced_subsample")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", type=str, default="checkpoints/random_forest.joblib")
    args = parser.parse_args()

    patches, labels = load_training_data()
    features = extract_features_batch_extended(patches)

    unique, counts = np.unique(labels, return_counts=True)
    min_count = counts.min() if len(counts) else 0
    if min_count < 2:
        print("warning: some classes have <2 samples; using non-stratified split")
        splitter = ShuffleSplit(
            n_splits=1, test_size=args.val_split, random_state=args.seed
        )
        train_idx, val_idx = next(splitter.split(features))
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_split, random_state=args.seed
        )
        train_idx, val_idx = next(splitter.split(features, labels))
    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]

    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    acc = accuracy_score(y_val, val_pred)
    f1_w = f1_score(y_val, val_pred, average="weighted")
    f1_m = f1_score(y_val, val_pred, average="macro")
    print(f"val accuracy: {acc:.4f}")
    print(f"val f1_weighted: {f1_w:.4f}")
    print(f"val f1_macro: {f1_m:.4f}")
    print("val classification report (vistgerd):")
    print(classification_report(y_val, val_pred, digits=4, zero_division=0))
    print("val confusion matrix (vistgerd):")
    print(confusion_matrix(y_val, val_pred))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out_path)
    print(f"saved model to {out_path}")


if __name__ == "__main__":
    main()
