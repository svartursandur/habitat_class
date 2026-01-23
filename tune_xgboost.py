#!/usr/bin/env python3
"""
Lightweight XGBoost tuner for weighted F1 on a fixed validation split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from xgboost import DMatrix
import xgboost

from utils import extract_features_batch_extended, load_training_data


def make_split(features: np.ndarray, labels: np.ndarray, val_split: float, seed: int):
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
                n_splits=1, test_size=val_split, random_state=seed
            )
            train_common, val_common = next(
                splitter.split(features[common_idx], labels_common)
            )
        except ValueError:
            splitter = ShuffleSplit(
                n_splits=1, test_size=val_split, random_state=seed
            )
            train_common, val_common = next(splitter.split(features[common_idx]))
    else:
        splitter = ShuffleSplit(
            n_splits=1, test_size=val_split, random_state=seed
        )
        train_common, val_common = next(splitter.split(features[common_idx]))

    train_idx = np.concatenate([rare_idx, common_idx[train_common]])
    val_idx = common_idx[val_common]
    return train_idx, val_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune XGBoost for weighted F1")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--out", type=str, default="checkpoints/xgboost_tuned.json")
    args = parser.parse_args()

    patches, labels = load_training_data()
    features = extract_features_batch_extended(patches)

    train_idx, val_idx = make_split(features, labels, args.val_split, args.seed)
    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]

    classes = np.unique(labels)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_train_enc = np.array([class_to_idx[c] for c in y_train], dtype=np.int32)
    y_val_enc = np.array([class_to_idx[c] for c in y_val], dtype=np.int32)

    counts_train = np.bincount(y_train_enc, minlength=len(classes))
    class_weights = (len(y_train_enc) / (len(classes) * counts_train)).astype(np.float32)
    sample_weight = class_weights[y_train_enc]

    dtrain = DMatrix(x_train, label=y_train_enc, weight=sample_weight)
    dval = DMatrix(x_val, label=y_val_enc)

    # Small, high-ROI grid around common good settings.
    configs = [
        {"max_depth": 6, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85, "eta": 0.05, "lambda": 1.0},
        {"max_depth": 6, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85, "eta": 0.05, "lambda": 2.0},
        {"max_depth": 5, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9, "eta": 0.05, "lambda": 2.0},
        {"max_depth": 7, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "eta": 0.05, "lambda": 1.0},
    ]

    best = {"f1_weighted": -1.0}
    for i, cfg in enumerate(configs, 1):
        params = {
            "objective": "multi:softprob",
            "num_class": len(classes),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "seed": args.seed,
            "nthread": -1,
            **cfg,
        }
        print(f"trial {i}/{len(configs)} params={cfg}")
        booster = xgboost.train(
            params,
            dtrain,
            num_boost_round=args.n_estimators,
            evals=[(dval, "validation")],
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=False,
        )
        best_iteration = booster.best_iteration if booster.best_iteration is not None else -1
        proba = booster.predict(
            dval,
            iteration_range=(0, best_iteration + 1) if best_iteration >= 0 else None,
        )
        preds = classes[np.argmax(proba, axis=1).astype(int)]
        f1_w = f1_score(y_val, preds, average="weighted")
        print(f"trial {i} f1_weighted={f1_w:.4f} best_iter={best_iteration}")

        if f1_w > best["f1_weighted"]:
            best = {
                "f1_weighted": float(f1_w),
                "best_iteration": int(best_iteration),
                "params": cfg,
            }
            out_path = Path(args.out)
            booster.save_model(out_path)
            np.save(out_path.with_suffix(".classes.npy"), classes)
            out_path.with_suffix(".meta.json").write_text(
                json.dumps({"best_iteration": int(best_iteration), "seed": args.seed})
            )

    print("best:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
