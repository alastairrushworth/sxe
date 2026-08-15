"""Benchmark SparseExtraEnsemble (SXE) vs RandomForest vs LightGBM.

Datasets:
  1. California housing   (regression, real, n~20.6k, p=8)
  2. Friedman1 + noise    (regression, synthetic, n=25k, p=25 of which 20 pure noise)
  3. Covertype (binary)   (classification, real, subsampled to 30k, p=54)

Metrics: test error (RMSE/R2 or AUC/accuracy), train time, predict time,
pickled model size, total leaves, effective number of trees.
"""
import json
import pickle
import time
import warnings

import pathlib

import numpy as np

_OUT = pathlib.Path(__file__).parent
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_covtype, make_friedman1, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from sparse_extra import SparseExtraEnsemble

warnings.filterwarnings("ignore")
RS = 0


def get_datasets():
    ds = []
    try:
        d = fetch_california_housing()
        ds.append(("california", d.data, d.target, "regression"))
        print("loaded california", d.data.shape, flush=True)
    except Exception as e:
        print("california failed:", e, flush=True)
    X, y = make_friedman1(n_samples=25_000, n_features=25, noise=1.0, random_state=RS)
    ds.append(("friedman1_noise", X, y, "regression"))
    print("built friedman1", X.shape, flush=True)
    try:
        d = fetch_covtype()
        rng = np.random.default_rng(RS)
        idx = rng.choice(len(d.target), 30_000, replace=False)
        X, y = d.data[idx], (d.target[idx] == 2).astype(int)
        ds.append(("covtype_binary", X, y, "classification"))
        print("loaded covtype", X.shape, "pos rate", y.mean().round(3), flush=True)
    except Exception as e:
        print("covtype failed, using synthetic:", e, flush=True)
        X, y = make_classification(n_samples=30_000, n_features=40, n_informative=12,
                                   n_redundant=8, flip_y=0.05, random_state=RS)
        ds.append(("synthetic_clf", X, y, "classification"))
    return ds


def evaluate(name, task, model, Xte, yte, train_s, extra=None):
    t = time.perf_counter()
    if task == "regression":
        pred = model.predict(Xte)
        pred_s = time.perf_counter() - t
        m1 = mean_squared_error(yte, pred) ** 0.5
        m2 = r2_score(yte, pred)
        mnames = ("rmse", "r2")
    else:
        proba = model.predict_proba(Xte)[:, 1]
        pred_s = time.perf_counter() - t
        m1 = roc_auc_score(yte, proba)
        m2 = accuracy_score(yte, (proba >= 0.5).astype(int))
        mnames = ("auc", "accuracy")
    size_mb = len(pickle.dumps(model)) / 1e6
    row = dict(model=name, metric1=mnames[0], value1=m1, metric2=mnames[1], value2=m2,
               train_s=train_s, pred_s=pred_s, size_mb=size_mb)
    row.update(extra or {})
    return row


def leaves_rf(rf):
    return int(sum(e.tree_.n_leaves for e in rf.estimators_))


def leaves_lgbm(model):
    info = model.booster_.dump_model()["tree_info"]
    return int(sum(t["num_leaves"] for t in info))


def run_dataset(dname, X, y, task):
    print(f"\n=== {dname} ({task}) ===", flush=True)
    strat = y if task == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=strat)
    rows = []

    # ---- Random Forest ----
    RF = RandomForestClassifier if task == "classification" else RandomForestRegressor
    rf = RF(n_estimators=300, n_jobs=-1, random_state=RS)
    t = time.perf_counter(); rf.fit(Xtr, ytr); ts = time.perf_counter() - t
    rows.append(evaluate("RandomForest-300", task, rf, Xte, yte, ts,
                         dict(total_leaves=leaves_rf(rf), n_trees=300)))
    print("RF done", round(ts, 1), "s", flush=True)

    # ---- LightGBM (early stopping) ----
    LG = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
    X2, Xes, y2, yes_ = train_test_split(Xtr, ytr, test_size=0.15, random_state=RS,
                                         stratify=ytr if task == "classification" else None)
    g = LG(n_estimators=2000, learning_rate=0.05, num_leaves=31,
           random_state=RS, n_jobs=-1, verbose=-1)
    t = time.perf_counter()
    g.fit(X2, y2, eval_set=[(Xes, yes_)], callbacks=[lgb.early_stopping(50, verbose=False)])
    ts = time.perf_counter() - t
    rows.append(evaluate("LightGBM-ES", task, g, Xte, yte, ts,
                         dict(total_leaves=leaves_lgbm(g), n_trees=int(g.best_iteration_ or g.n_estimators))))
    print("LGBM done", round(ts, 1), "s best_iter", g.best_iteration_, flush=True)

    # ---- SXE single round ----
    sxe = SparseExtraEnsemble(task=task, n_trees=300, random_state=RS)
    t = time.perf_counter(); sxe.fit(Xtr, ytr); ts = time.perf_counter() - t
    st = sxe.stats()
    rows.append(evaluate("SXE-300", task, sxe, Xte, yte, ts,
                         dict(total_leaves=st["total_leaves"], n_trees=st["members_selected"],
                              members_grown=st["members_grown"])))
    print("SXE done", round(ts, 1), "s", st, flush=True)

    # ---- SXE boosted (3 residual rounds) ----
    sxb = SparseExtraEnsemble(task=task, n_rounds=3, trees_per_round=120, random_state=RS)
    t = time.perf_counter(); sxb.fit(Xtr, ytr); ts = time.perf_counter() - t
    st = sxb.stats()
    rows.append(evaluate("SXE-boost-3x120", task, sxb, Xte, yte, ts,
                         dict(total_leaves=st["total_leaves"], n_trees=st["members_selected"],
                              members_grown=st["members_grown"])))
    print("SXE-boost done", round(ts, 1), "s", st, flush=True)

    for r in rows:
        r["dataset"] = dname
        r["task"] = task
    return rows


def main():
    all_rows = []
    for dname, X, y, task in get_datasets():
        all_rows += run_dataset(dname, X, y, task)
        pd.DataFrame(all_rows).to_csv(str(_OUT / "results.csv"), index=False)
    df = pd.DataFrame(all_rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n", df.round(4).to_string(index=False), flush=True)
    print("\nsaved results.csv", flush=True)


if __name__ == "__main__":
    main()
