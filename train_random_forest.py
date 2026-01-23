#!/usr/bin/env python3
"""
Train RandomForest models on all labeled data using multiple seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from utils import extract_features_batch_extended, load_training_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RandomForest models on all labeled data (no validation split)"
    )
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--class-weight", type=str, default="balanced_subsample")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated list of seeds to train (default: 0,1,2,3,4)",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", type=str, default="checkpoints/random_forest.joblib")
    args = parser.parse_args()

    patches, labels = load_training_data()
    features = extract_features_batch_extended(patches)

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seed_list:
        raise ValueError("no seeds provided; use --seeds with at least one integer")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for seed in seed_list:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight=None if args.class_weight == "none" else args.class_weight,
            random_state=seed,
            n_jobs=args.n_jobs,
        )
        model.fit(features, labels)

        if "{seed}" in args.out:
            save_path = Path(args.out.format(seed=seed))
        elif len(seed_list) == 1:
            save_path = out_path
        else:
            save_path = out_path.with_name(f"{out_path.stem}_seed{seed}{out_path.suffix}")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        dump(model, save_path)
        print(f"saved model (seed={seed}) to {save_path}")


if __name__ == "__main__":
    main()
