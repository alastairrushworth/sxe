"""Head-to-head: SXE vs LightGBM on the three reference tasks.

Unlike the original script this one writes its results next to itself, takes
the thread count and dataset selection from the command line, and reports the
same numbers for both libraries so the CSV is the artefact, not the console.

    python benchmarks/bench_final.py                  # all datasets, 1 thread
    python benchmarks/bench_final.py --n-jobs -1      # all cores
    python benchmarks/bench_final.py -d friedman1_noise

California and Covertype download via scikit-learn on first use.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import time
import warnings

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_covtype, make_friedman1
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from sxe import SXEClassifier, SXERegressor

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).parent
RS = 0

# Per-dataset configurations, tuned by hand. Defaults do not reproduce these
# numbers -- see the tuning guidance in TECHNICAL_REPORT.md section 5.
CONFIGS = {
    "california": dict(
        max_leaf_nodes=63,
        max_depth_choices=(0,),
        learning_rate=0.05,
        batch_size=4,
        max_iter=1200,
        n_iter_no_change=25,
        min_samples_leaf_range=(10, 25),
        subsample_range=(0.9, 1.0),
        max_features_range=(0.85, 1.0),
        validation_fraction=0.15,
        prune_strength_range=(1e-3, 0.05),
        n_alphas=15,
    ),
    "friedman1_noise": dict(
        learning_rate=0.3,
        batch_size=16,
        max_iter=45,
        n_iter_no_change=2,
        max_depth_choices=(4, 5, 6),
        min_samples_leaf_range=(20, 70),
        subsample_range=(0.2, 0.35),
        max_features_range=(0.35, 0.65),
        validation_fraction=0.10,
        n_alphas=12,
    ),
    "covtype_binary": dict(
        bundle_binary_features=True,
        max_depth_choices=(10, 12, 14),
        min_samples_leaf_range=(2, 15),
        learning_rate=0.25,
        batch_size=12,
        max_iter=50,
        subsample_range=(0.5, 0.8),
        n_iter_no_change=5,
        validation_fraction=0.15,
        n_alphas=15,
        splitter="random",
        n_thresholds=3,
    ),
}


def datasets(names):
    if "california" in names:
        d = fetch_california_housing()
        yield "california", d.data, d.target, "regression"
    if "friedman1_noise" in names:
        X, y = make_friedman1(n_samples=25_000, n_features=25, noise=1.0, random_state=RS)
        yield "friedman1_noise", X, y, "regression"
    if "covtype_binary" in names:
        d = fetch_covtype()
        rng = np.random.default_rng(RS)
        idx = rng.choice(len(d.target), 30_000, replace=False)
        yield "covtype_binary", d.data[idx], (d.target[idx] == 2).astype(int), "classification"


def score(task, y_true, model, X):
    if task == "regression":
        return "rmse", mean_squared_error(y_true, model.predict(X)) ** 0.5
    return "auc", roc_auc_score(y_true, model.predict_proba(X)[:, 1])


def run_sxe(name, task, Xtr, ytr, Xte, yte, n_jobs):
    est = SXEClassifier if task == "classification" else SXERegressor
    model = est(random_state=RS, n_jobs=n_jobs, **CONFIGS[name])
    t = time.perf_counter()
    model.fit(Xtr, ytr)
    train_s = time.perf_counter() - t
    metric, value = score(task, yte, model, Xte)
    s = model.summary()
    return dict(
        model="SXE",
        metric=metric,
        value=value,
        train_s=train_s,
        size_mb=s["model_size_mb"],
        n_leaves=s["n_leaves"],
        n_trees=s["n_trees_kept"],
        note="compact-format estimate",
    )


def run_lightgbm(task, Xtr, ytr, Xte, yte, n_jobs):
    import lightgbm as lgb

    LG = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
    X2, Xes, y2, yes_ = train_test_split(
        Xtr,
        ytr,
        test_size=0.15,
        random_state=RS,
        stratify=ytr if task == "classification" else None,
    )
    g = LG(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RS,
        n_jobs=n_jobs,
        verbose=-1,
    )
    t = time.perf_counter()
    g.fit(X2, y2, eval_set=[(Xes, yes_)], callbacks=[lgb.early_stopping(50, verbose=False)])
    train_s = time.perf_counter() - t
    metric, value = score(task, yte, g, Xte)
    leaves = sum(tree["num_leaves"] for tree in g.booster_.dump_model()["tree_info"])
    return dict(
        model="LightGBM-ES",
        metric=metric,
        value=value,
        train_s=train_s,
        size_mb=len(pickle.dumps(g)) / 1e6,
        n_leaves=leaves,
        n_trees=int(g.best_iteration_ or g.n_estimators),
        note="pickled size",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-d",
        "--datasets",
        nargs="*",
        default=list(CONFIGS),
        choices=list(CONFIGS),
        help="datasets to run",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="threads for both libraries (default 1, as in the report)",
    )
    ap.add_argument("--no-lightgbm", action="store_true", help="skip the baseline")
    ap.add_argument("-o", "--out", default=str(HERE / "results.csv"))
    args = ap.parse_args()

    rows = []
    for name, X, y, task in datasets(set(args.datasets)):
        strat = y if task == "classification" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=strat)
        print(f"\n=== {name} ({task}, n={len(X)}, p={X.shape[1]}) ===", flush=True)
        runs = [run_sxe(name, task, Xtr, ytr, Xte, yte, args.n_jobs)]
        if not args.no_lightgbm:
            try:
                runs.append(run_lightgbm(task, Xtr, ytr, Xte, yte, args.n_jobs))
            except ImportError:
                print("  lightgbm not installed, skipping baseline", flush=True)
        for r in runs:
            r.update(dataset=name, task=task, n_jobs=args.n_jobs)
            print(
                f"  {r['model']:<12} {r['metric']} {r['value']:.4f}  "
                f"{r['train_s']:6.2f}s  {r['size_mb']:7.2f} MB  "
                f"{r['n_leaves']:>7,} leaves  ({r['note']})",
                flush=True,
            )
        rows += runs

    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    except ImportError:
        import csv

        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
