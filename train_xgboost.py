#!/usr/bin/env python3
"""
Train an XGBoost model on engineered band statistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from xgboost import XGBClassifier

from utils import extract_features_batch_extended, load_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost for habitat classification")
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", type=str, default="checkpoints/xgboost.json")
    args = parser.parse_args()

    patches, labels = load_training_data()
    features = extract_features_batch_extended(patches)

    unique, counts = np.unique(labels, return_counts=True)
    rare_classes = unique[counts < 2]
    rare_mask = np.isin(labels, rare_classes)
    rare_idx = np.where(rare_mask)[0]
    common_idx = np.where(~rare_mask)[0]
    labels_common = labels[common_idx]

    if len(rare_classes) > 0:
        print("warning: some classes have <2 samples; keeping all rare classes in train")

    if len(common_idx) == 0:
        raise ValueError("not enough non-rare samples to create a validation split")

    if len(np.unique(labels_common)) > 1:
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=args.val_split, random_state=args.seed
            )
            train_common, val_common = next(
                splitter.split(features[common_idx], labels_common)
            )
        except ValueError:
            splitter = ShuffleSplit(
                n_splits=1, test_size=args.val_split, random_state=args.seed
            )
            train_common, val_common = next(splitter.split(features[common_idx]))
    else:
        splitter = ShuffleSplit(
            n_splits=1, test_size=args.val_split, random_state=args.seed
        )
        train_common, val_common = next(splitter.split(features[common_idx]))

    train_idx = np.concatenate([rare_idx, common_idx[train_common]])
    val_idx = common_idx[val_common]

    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]

    classes = np.unique(labels)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_train_enc = np.array([class_to_idx[c] for c in y_train], dtype=np.int32)
    y_val_enc = np.array([class_to_idx[c] for c in y_val], dtype=np.int32)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        eval_metric="mlogloss",
    )
    model.fit(
        x_train,
        y_train_enc,
        eval_set=[(x_val, y_val_enc)],
        verbose=True,
    )

    val_pred_enc = model.predict(x_val)
    val_pred = classes[val_pred_enc.astype(int)]
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

    model.save_model(args.out)
    classes_path = Path(args.out).with_suffix(".classes.npy")
    np.save(classes_path, classes)
    print(f"saved model to {args.out}")
    print(f"saved classes to {classes_path}")


if __name__ == "__main__":
    main()
