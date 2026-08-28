"""Head-to-head: SXE vs LightGBM on the three reference tasks.

Every number both libraries are compared on is computed the same way on both
sides. Two things this script deliberately does not do, because earlier
versions did and the results were misleading:

  * it does not compare an SXE format estimate against a LightGBM pickle --
    model size uses one formula, applied to both (see ``model_bytes``);
  * it does not run both libraries at ``n_jobs=-1`` and call the difference a
    result. LightGBM slows down monotonically with threads on datasets this
    small, so the honest timing comparison is each library at its own best
    thread count. Pass several values to --n-jobs and the summary reports the
    best per library.

Results are averaged over seeds; the seed controls the split, the model, and
(for Friedman) the data itself.

    python benchmarks/bench_final.py                     # 5 seeds, 1 and all cores
    python benchmarks/bench_final.py --seeds 0           # single seed, quick
    python benchmarks/bench_final.py --n-jobs 1          # single-thread only
    python benchmarks/bench_final.py -d friedman1_noise

California and Covertype download via scikit-learn on first use.

Note on reproducibility: SXE draws its per-tree randomness through
``std::uniform_*_distribution``, whose algorithms are implementation-defined,
so figures reproduce exactly only on the same C++ standard library. Results
are invariant to ``n_jobs`` on any one platform.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import statistics
import time
import warnings

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_covtype, make_friedman1
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from sxe import SXEClassifier, SXERegressor

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).parent

# Compact-model accounting, applied identically to both libraries. An internal
# node is (feature int32, threshold uint8, left int32, right int32) and a leaf
# is one double. A binned encoding needs the bin table to map raw features to
# bin ids at predict time, so it is counted too -- for both sides, over the
# feature count each model actually operates on (SXE's is post-bundling).
#
# This flatters neither library. LightGBM's real format stores float64
# thresholds and therefore needs no bin table; charging it one here is the
# assumption that makes the tree structures directly comparable.
BYTES_PER_INTERNAL = 13
BYTES_PER_LEAF = 8
BYTES_PER_BIN_EDGE = 8
BINS_PER_FEATURE = 255

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

# LightGBM's early-stopping fold. SXE carves its own fold out of the training
# set (validation_fraction), so this is set per dataset to leave both
# libraries the same number of rows to actually fit on.
LGBM_ES_FRACTION = {
    "california": 0.15,
    "friedman1_noise": 0.10,
    "covtype_binary": 0.15,
}


def datasets(names, seed):
    """Yield (name, X, y, task). The seed varies the data where it can."""
    if "california" in names:
        d = fetch_california_housing()
        yield "california", d.data, d.target, "regression"
    if "friedman1_noise" in names:
        X, y = make_friedman1(n_samples=25_000, n_features=25, noise=1.0, random_state=seed)
        yield "friedman1_noise", X, y, "regression"
    if "covtype_binary" in names:
        d = fetch_covtype()
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(d.target), 30_000, replace=False)
        yield "covtype_binary", d.data[idx], (d.target[idx] == 2).astype(int), "classification"


def model_bytes(n_leaves, n_trees, n_features):
    """Compact serialized size under one formula, for either library.

    Trees are full binary, so internal nodes = leaves - trees.
    """
    n_internal = n_leaves - n_trees
    struct = n_internal * BYTES_PER_INTERNAL + n_leaves * BYTES_PER_LEAF
    bins = n_features * BINS_PER_FEATURE * BYTES_PER_BIN_EDGE
    return struct, bins, struct + bins


def score(task, y_true, model, X):
    if task == "regression":
        return "rmse", mean_squared_error(y_true, model.predict(X)) ** 0.5
    return "auc", roc_auc_score(y_true, model.predict_proba(X)[:, 1])


def run_sxe(name, task, Xtr, ytr, Xte, yte, n_jobs, seed):
    est = SXEClassifier if task == "classification" else SXERegressor
    model = est(random_state=seed, n_jobs=n_jobs, **CONFIGS[name])
    t = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t
    metric, value = score(task, yte, model, Xte)

    # the feature count the trees actually index, i.e. after bundling
    n_feat = model.bundler_.n_features_out_ if model.bundler_ is not None else Xtr.shape[1]
    struct, bins, compact = model_bytes(model.n_leaves_, model.n_trees_kept_, n_feat)
    return dict(
        model="SXE",
        metric=metric,
        value=value,
        fit_s=fit_s,
        n_trees=model.n_trees_kept_,
        n_leaves=model.n_leaves_,
        n_features_model=n_feat,
        struct_bytes=struct,
        bins_bytes=bins,
        compact_bytes=compact,
        pickle_bytes=len(pickle.dumps(model)),
    )


def run_lightgbm(name, task, Xtr, ytr, Xte, yte, n_jobs, seed):
    import lightgbm as lgb

    LG = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
    X2, Xes, y2, yes_ = train_test_split(
        Xtr,
        ytr,
        test_size=LGBM_ES_FRACTION[name],
        random_state=seed,
        stratify=ytr if task == "classification" else None,
    )
    g = LG(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )
    t = time.perf_counter()
    g.fit(X2, y2, eval_set=[(Xes, yes_)], callbacks=[lgb.early_stopping(50, verbose=False)])
    fit_s = time.perf_counter() - t
    metric, value = score(task, yte, g, Xte)

    info = g.booster_.dump_model()["tree_info"]
    n_leaves = sum(tree["num_leaves"] for tree in info)
    n_trees = len(info)
    struct, bins, compact = model_bytes(n_leaves, n_trees, Xtr.shape[1])
    return dict(
        model="LightGBM-ES",
        metric=metric,
        value=value,
        fit_s=fit_s,
        n_trees=n_trees,
        n_leaves=n_leaves,
        n_features_model=Xtr.shape[1],
        struct_bytes=struct,
        bins_bytes=bins,
        compact_bytes=compact,
        pickle_bytes=len(pickle.dumps(g)),
    )


def summarise(rows, datasets_run):
    """Print the table the README quotes: mean +- sd over seeds, best-thread time."""

    def cell(vals):
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return m, s

    for name in datasets_run:
        sub = [r for r in rows if r["dataset"] == name]
        if not sub:
            continue
        metric = sub[0]["metric"]
        print(f"\n=== {name} ({metric}, {len(set(r['seed'] for r in sub))} seeds) ===")
        line = (
            f"  {'model':<13}{'score':<20}{'leaves':<12}"
            f"{'size MB':<11}{'best fit s':<13}{'@threads'}"
        )
        print(line)
        print("  " + "-" * (len(line) - 2))
        for model in ("LightGBM-ES", "SXE"):
            ms = [r for r in sub if r["model"] == model]
            if not ms:
                continue
            # accuracy/structure are thread-invariant: take the single-thread runs
            base = [r for r in ms if r["n_jobs"] == min(r2["n_jobs"] for r2 in ms)] or ms
            vm, vs = cell([r["value"] for r in base])
            lm, _ = cell([r["n_leaves"] for r in base])
            sm, _ = cell([r["compact_bytes"] for r in base])
            # timing: each library at whichever thread count suits it best
            by_thread = {}
            for r in ms:
                by_thread.setdefault(r["n_jobs"], []).append(r["fit_s"])
            best_nj = min(by_thread, key=lambda k: statistics.mean(by_thread[k]))
            tm, ts = cell(by_thread[best_nj])
            print(
                f"  {model:<13}{vm:.4f} +- {vs:.4f}   {lm:>9,.0f}   "
                f"{sm / 1e6:>8.3f}   {tm:>6.2f} +- {ts:.2f}   {best_nj:>5}"
            )
        sxe = [r for r in sub if r["model"] == "SXE"]
        lgb_ = [r for r in sub if r["model"] == "LightGBM-ES"]
        if sxe and lgb_:
            rs = statistics.mean([r["compact_bytes"] for r in sxe])
            rl = statistics.mean([r["compact_bytes"] for r in lgb_])
            print(f"  size ratio LightGBM/SXE: {rl / rs:.2f}x  (same formula both sides)")


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
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3, 4],
        help="seeds to average over (default 0..4)",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        nargs="*",
        default=[1, -1],
        help="thread counts to time each library at; the summary reports each "
        "library's best (default: 1 and all cores)",
    )
    ap.add_argument("--no-lightgbm", action="store_true", help="skip the baseline")
    ap.add_argument("-o", "--out", default=str(HERE / "results.csv"))
    args = ap.parse_args()

    wanted = set(args.datasets)
    rows = []
    for seed in args.seeds:
        for name, X, y, task in datasets(wanted, seed):
            strat = y if task == "classification" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=strat
            )
            for n_jobs in args.n_jobs:
                print(
                    f"\n=== {name} ({task}, n={len(X)}, p={X.shape[1]}) "
                    f"seed={seed} n_jobs={n_jobs} ===",
                    flush=True,
                )
                runs = [run_sxe(name, task, Xtr, ytr, Xte, yte, n_jobs, seed)]
                if not args.no_lightgbm:
                    try:
                        runs.append(run_lightgbm(name, task, Xtr, ytr, Xte, yte, n_jobs, seed))
                    except ImportError:
                        print("  lightgbm not installed, skipping baseline", flush=True)
                for r in runs:
                    r.update(dataset=name, task=task, seed=seed, n_jobs=n_jobs)
                    print(
                        f"  {r['model']:<12} {r['metric']} {r['value']:.4f}  "
                        f"{r['fit_s']:6.2f}s  {r['compact_bytes'] / 1e6:7.3f} MB compact  "
                        f"{r['n_leaves']:>7,} leaves",
                        flush=True,
                    )
                rows += runs

    summarise(rows, [d for d in CONFIGS if d in wanted])

    fieldnames = list(rows[0])
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.out, index=False)
    except ImportError:
        import csv

        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
