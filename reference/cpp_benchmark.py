"""Benchmark the C++ SXE core on the same datasets/splits as benchmark.py."""
import pickle
import time
import warnings

import pathlib

import numpy as np

_OUT = pathlib.Path(__file__).parent
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_covtype, make_friedman1
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from cpp_sxe import CppSXE

warnings.filterwarnings("ignore")
RS = 0


def datasets():
    d = fetch_california_housing()
    yield "california", d.data, d.target, "regression"
    X, y = make_friedman1(n_samples=25_000, n_features=25, noise=1.0, random_state=RS)
    yield "friedman1_noise", X, y, "regression"
    d = fetch_covtype()
    rng = np.random.default_rng(RS)
    idx = rng.choice(len(d.target), 30_000, replace=False)
    yield "covtype_binary", d.data[idx], (d.target[idx] == 2).astype(int), "classification"


rows = []
for dname, X, y, task in datasets():
    strat = y if task == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=strat)
    for label, kw in [
        ("CppSXE-300", dict(n_trees=300)),
        ("CppSXE-boost-3x120", dict(n_rounds=3, trees_per_round=120)),
        ("CppSXE-boost-5x100", dict(n_rounds=5, trees_per_round=100)),
    ]:
        m = CppSXE(task=task, random_state=RS, **kw)
        t = time.perf_counter(); m.fit(Xtr, ytr); ts = time.perf_counter() - t
        tp = time.perf_counter()
        if task == "regression":
            pred = m.predict(Xte); pred_s = time.perf_counter() - tp
            v1, v2, m1, m2n = mean_squared_error(yte, pred) ** 0.5, r2_score(yte, pred), "rmse", "r2"
        else:
            proba = m.predict_proba(Xte)[:, 1]; pred_s = time.perf_counter() - tp
            v1, v2, m1, m2n = roc_auc_score(yte, proba), accuracy_score(yte, (proba >= .5).astype(int)), "auc", "accuracy"
        st = m.stats()
        rows.append(dict(dataset=dname, model=label, metric1=m1, value1=v1, metric2=m2n, value2=v2,
                         train_s=ts, grow_s=st["grow_s"], weight_fit_s=st["weight_fit_s"],
                         pred_s=pred_s, size_mb=st["size_mb"], total_leaves=st["total_leaves"],
                         n_trees=st["members_selected"], members_grown=st["members_grown"]))
        print(dname, label, m1, round(v1, 4), "train", round(ts, 2),
              f"(grow {st['grow_s']}, wfit {st['weight_fit_s']})", "size", st["size_mb"], flush=True)

pd.DataFrame(rows).to_csv(str(_OUT / "results_cpp.csv"), index=False)
print("saved results_cpp.csv", flush=True)
