"""Check the packaged core against the original ctypes prototype.

The prototype in this directory is the implementation that produced the
v1/v2 numbers in paper/sxe.pdf. The package is a port -- pybind11 instead of
ctypes, a std::thread pool instead of OpenMP -- so it needs an oracle, and
this is it: same seed, same splits, same configurations, and the predictions
must come out bit-for-bit identical.

Run it after any change to the growth, pruning or prediction code:

    cd reference
    g++ -O3 -march=native -fopenmp -shared -fPIC sxe.cpp -o libsxe.so
    python parity_check.py

Exits non-zero if any dataset diverges. California and Covertype download via
scikit-learn on first use.
"""

from __future__ import annotations

import pathlib
import sys
import time
import warnings

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_covtype, make_friedman1
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from cpp_sxe import CppSXEBoost  # noqa: E402  (the prototype, alongside this file)

from sxe import SXEClassifier, SXERegressor  # noqa: E402

warnings.filterwarnings("ignore")
RS = 0

# The same configuration expressed in both parameter vocabularies.
CONFIGS = {
    "california": (
        dict(
            max_leaves=63, depth_choices=(0,), shrink=0.05, batch=4, max_batches=1200,
            patience=25, leaf_range=(10, 25), subsample_range=(0.9, 1.0),
            feat_range=(0.85, 1.0), val_fraction=0.15, lam_range=(1e-3, 0.05), n_alphas=15,
        ),
        dict(
            max_leaf_nodes=63, max_depth_choices=(0,), learning_rate=0.05, batch_size=4,
            max_iter=1200, n_iter_no_change=25, min_samples_leaf_range=(10, 25),
            subsample_range=(0.9, 1.0), max_features_range=(0.85, 1.0),
            validation_fraction=0.15, prune_strength_range=(1e-3, 0.05), n_alphas=15,
        ),
    ),
    "friedman1_noise": (
        dict(
            shrink=0.3, batch=16, max_batches=45, patience=2, depth_choices=(4, 5, 6),
            leaf_range=(20, 70), subsample_range=(0.2, 0.35), feat_range=(0.35, 0.65),
            val_fraction=0.10, n_alphas=12,
        ),
        dict(
            learning_rate=0.3, batch_size=16, max_iter=45, n_iter_no_change=2,
            max_depth_choices=(4, 5, 6), min_samples_leaf_range=(20, 70),
            subsample_range=(0.2, 0.35), max_features_range=(0.35, 0.65),
            validation_fraction=0.10, n_alphas=12,
        ),
    ),
    "covtype_binary": (
        dict(
            bundle_binary=True, depth_choices=(10, 12, 14), leaf_range=(2, 15), shrink=0.25,
            batch=12, max_batches=50, subsample_range=(0.5, 0.8), patience=5,
            val_fraction=0.15, n_alphas=15, argmax_bins=0, n_thresh=3,
        ),
        dict(
            bundle_binary_features=True, max_depth_choices=(10, 12, 14),
            min_samples_leaf_range=(2, 15), learning_rate=0.25, batch_size=12, max_iter=50,
            subsample_range=(0.5, 0.8), n_iter_no_change=5, validation_fraction=0.15,
            n_alphas=15, splitter="random", n_thresholds=3,
        ),
    ),
}


def datasets():
    d = fetch_california_housing()
    yield "california", d.data, d.target, "regression"
    X, y = make_friedman1(n_samples=25_000, n_features=25, noise=1.0, random_state=RS)
    yield "friedman1_noise", X, y, "regression"
    d = fetch_covtype()
    rng = np.random.default_rng(RS)
    idx = rng.choice(len(d.target), 30_000, replace=False)
    yield "covtype_binary", d.data[idx], (d.target[idx] == 2).astype(int), "classification"


def main() -> int:
    ok = True
    for name, X, y, task in datasets():
        old_cfg, new_cfg = CONFIGS[name]
        strat = y if task == "classification" else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=RS, stratify=strat
        )

        t = time.perf_counter()
        old = CppSXEBoost(task=task, random_state=RS, **old_cfg).fit(Xtr, ytr)
        t_old = time.perf_counter() - t

        est = SXEClassifier if task == "classification" else SXERegressor
        t = time.perf_counter()
        new = est(random_state=RS, **new_cfg).fit(Xtr, ytr)
        t_new = time.perf_counter() - t

        if task == "regression":
            p_old, p_new = old.predict(Xte), new.predict(Xte)
            label = "RMSE"
            m_old = mean_squared_error(yte, p_old) ** 0.5
            m_new = mean_squared_error(yte, p_new) ** 0.5
        else:
            p_old = old.predict_proba(Xte)[:, 1]
            p_new = new.predict_proba(Xte)[:, 1]
            label = "AUC"
            m_old = roc_auc_score(yte, p_old)
            m_new = roc_auc_score(yte, p_new)

        so, sn = old.stats(), new.summary()
        identical = np.array_equal(p_old, p_new)
        same_structure = (
            so["members_selected"] == sn["n_trees_kept"]
            and so["members_grown"] == sn["n_trees_grown"]
            and so["total_leaves"] == sn["n_leaves"]
        )
        ok &= identical and same_structure

        print(f"\n=== {name} ===")
        print(f"  {label:4}    prototype {m_old:.6f}   package {m_new:.6f}")
        print(f"  trees     {so['members_selected']}/{so['members_grown']}"
              f"   vs {sn['n_trees_kept']}/{sn['n_trees_grown']}")
        print(f"  leaves    {so['total_leaves']} vs {sn['n_leaves']}")
        print(f"  time s    {t_old:.2f} vs {t_new:.2f}")
        print(f"  identical predictions: {identical}   identical structure: {same_structure}")

    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
