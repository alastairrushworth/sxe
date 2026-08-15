"""Final head-to-head: SXE v3 (CppSXEBoost) vs LightGBM, same splits as before."""
import time
import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_covtype, make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from cpp_sxe import CppSXEBoost

CONFIGS = {
    "california": dict(max_leaves=63, depth_choices=(0,), shrink=0.05, batch=4, max_batches=1200,
                       patience=25, leaf_range=(10,25), subsample_range=(0.9,1.0),
                       feat_range=(0.85,1.0), val_fraction=0.15, lam_range=(1e-3,0.05), n_alphas=15),
    "friedman1_noise": dict(shrink=0.3, batch=16, max_batches=45, patience=2,
                            depth_choices=(4,5,6), leaf_range=(20,70), subsample_range=(0.2,0.35),
                            feat_range=(0.35,0.65), val_fraction=0.10, n_alphas=12),
    "covtype_binary": dict(bundle_binary=True, depth_choices=(10,12,14), leaf_range=(2,15),
                           shrink=0.25, batch=12, max_batches=50, subsample_range=(0.5,0.8),
                           patience=5, val_fraction=0.15, n_alphas=15, argmax_bins=0, n_thresh=3),
}

def datasets():
    d = fetch_california_housing()
    yield "california", d.data, d.target, "regression"
    X, y = make_friedman1(n_samples=25_000, n_features=25, noise=1.0, random_state=0)
    yield "friedman1_noise", X, y, "regression"
    d = fetch_covtype(); rng = np.random.default_rng(0)
    idx = rng.choice(len(d.target), 30_000, replace=False)
    yield "covtype_binary", d.data[idx], (d.target[idx] == 2).astype(int), "classification"

for name, X, y, task in datasets():
    strat = y if task == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=strat)
    m = CppSXEBoost(task=task, random_state=0, **CONFIGS[name])
    t = time.perf_counter(); m.fit(Xtr, ytr); ts = time.perf_counter() - t
    if task == "regression":
        v, lab = mean_squared_error(yte, m.predict(Xte)) ** 0.5, "RMSE"
    else:
        v, lab = roc_auc_score(yte, m.predict_proba(Xte)[:, 1]), "AUC"
    st = m.stats()
    print(f"{name}: {lab} {v:.4f} | {ts:.2f}s | {st['size_mb']}MB | kept {st['members_selected']}/{st['members_grown']}")
