#!/usr/bin/env python3
"""
Train an XGBoost model on engineered band statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import xgboost
from xgboost import DMatrix

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
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="")
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

    classes = np.unique(labels)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    if len(rare_classes) > 0:
        print("warning: some classes have <2 samples; keeping all rare classes in train")

    if len(common_idx) == 0:
        raise ValueError("not enough non-rare samples to create a validation split")

    seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [args.seed]
    )
    out_base = Path(args.out)

    for seed in seeds:
        if args.val_split <= 0:
            train_idx = np.concatenate([rare_idx, common_idx])
            val_idx = None
        else:
            if len(np.unique(labels_common)) > 1:
                try:
                    splitter = StratifiedShuffleSplit(
                        n_splits=1, test_size=args.val_split, random_state=seed
                    )
                    train_common, val_common = next(
                        splitter.split(features[common_idx], labels_common)
                    )
                except ValueError:
                    splitter = ShuffleSplit(
                        n_splits=1, test_size=args.val_split, random_state=seed
                    )
                    train_common, val_common = next(splitter.split(features[common_idx]))
            else:
                splitter = ShuffleSplit(
                    n_splits=1, test_size=args.val_split, random_state=seed
                )
                train_common, val_common = next(splitter.split(features[common_idx]))

            train_idx = np.concatenate([rare_idx, common_idx[train_common]])
            val_idx = common_idx[val_common]

        x_train, y_train = features[train_idx], labels[train_idx]
        if val_idx is not None:
            x_val, y_val = features[val_idx], labels[val_idx]
        y_train_enc = np.array([class_to_idx[c] for c in y_train], dtype=np.int32)
        if val_idx is not None:
            y_val_enc = np.array([class_to_idx[c] for c in y_val], dtype=np.int32)

        counts_train = np.bincount(y_train_enc, minlength=len(classes))
        class_weights = (len(y_train_enc) / (len(classes) * counts_train)).astype(np.float32)
        sample_weight = class_weights[y_train_enc]

        params = {
            "objective": "multi:softprob",
            "num_class": len(classes),
            "max_depth": args.max_depth,
            "eta": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "lambda": args.reg_lambda,
            "tree_method": "hist",
            "nthread": args.n_jobs,
            "seed": seed,
            "eval_metric": "mlogloss",
        }
        dtrain = DMatrix(x_train, label=y_train_enc, weight=sample_weight)
        if val_idx is not None:
            dval = DMatrix(x_val, label=y_val_enc)
            booster = xgboost.train(
                params,
                dtrain,
                num_boost_round=args.n_estimators,
                evals=[(dval, "validation")],
                early_stopping_rounds=args.early_stopping_rounds,
                verbose_eval=True,
            )

            best_iteration = booster.best_iteration if booster.best_iteration is not None else -1
            val_proba = booster.predict(
                dval,
                iteration_range=(0, best_iteration + 1) if best_iteration >= 0 else None,
            )
            val_pred = classes[np.argmax(val_proba, axis=1).astype(int)]
            acc = accuracy_score(y_val, val_pred)
            f1_w = f1_score(y_val, val_pred, average="weighted")
            f1_m = f1_score(y_val, val_pred, average="macro")
            print(f"seed {seed} val accuracy: {acc:.4f}")
            print(f"seed {seed} val f1_weighted: {f1_w:.4f}")
            print(f"seed {seed} val f1_macro: {f1_m:.4f}")
            print("val classification report (vistgerd):")
            print(classification_report(y_val, val_pred, digits=4, zero_division=0))
            print("val confusion matrix (vistgerd):")
            print(confusion_matrix(y_val, val_pred))
        else:
            booster = xgboost.train(
                params,
                dtrain,
                num_boost_round=args.n_estimators,
                evals=[],
                verbose_eval=False,
            )
            best_iteration = -1

        out_path = (
            out_base
            if len(seeds) == 1
            else out_base.with_name(f"{out_base.stem}_seed{seed}{out_base.suffix}")
        )
        booster.save_model(out_path)
        classes_path = out_path.with_suffix(".classes.npy")
        np.save(classes_path, classes)
        meta_path = out_path.with_suffix(".meta.json")
        meta = {
            "best_iteration": int(best_iteration),
            "seed": seed,
            "val_split": float(args.val_split),
        }
        meta_path.write_text(json.dumps(meta))
        print(f"saved model to {out_path}")
        print(f"saved classes to {classes_path}")
        print(f"saved meta to {meta_path}")


if __name__ == "__main__":
    main()
