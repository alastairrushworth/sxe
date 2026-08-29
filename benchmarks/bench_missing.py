"""Native NaN handling against the median imputation it replaces.

The main suite cannot answer this. Its categorical columns are ordinal-coded
with missingness as its own level, so after encoding none of the nine datasets
has a NaN left for either library to handle. This script puts the holes back:
it punches missing values into the encoded matrix at a given rate and fits SXE
twice, once on the holes and once on a median-imputed copy.

The claim it exists to support is a modest one -- that native handling is a
convenience rather than an accuracy win. If that stops being true in either
direction, this is where it shows.

    python benchmarks/bench_missing.py
    python benchmarks/bench_missing.py -d adult --rates 0.2
"""

from __future__ import annotations

import argparse
import statistics
import warnings

import numpy as np
from bench_suite import DATASETS, encode
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from sxe import SXEClassifier, SXERegressor

warnings.filterwarnings("ignore")


def median_impute(Xtr, Xte):
    """The workaround a user would have written before NaN was accepted."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    out = []
    for A in (Xtr.copy(), Xte.copy()):
        bad = ~np.isfinite(A)
        if bad.any():
            A[bad] = np.take(med, np.nonzero(bad)[1])
        out.append(A)
    return out


def loss(task, model, X, y):
    """Lower is better, for either task."""
    if task == "regression":
        return mean_squared_error(y, model.predict(X)) ** 0.5
    return -roc_auc_score(y, model.predict_proba(X)[:, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-d",
        "--datasets",
        nargs="*",
        default=["california", "superconduct", "covtype_binary", "adult"],
        choices=[k for k, v in DATASETS.items() if k != "letter"],
    )
    ap.add_argument("--rates", type=float, nargs="*", default=[0.10, 0.30])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    print(f"{'dataset':<16}{'missing':>9}{'native':>12}{'imputed':>12}   better")
    for name in args.datasets:
        spec = DATASETS[name]
        for rate in args.rates:
            native, imputed = [], []
            for seed in args.seeds:
                X_raw, y, task, cat_idx = spec["loader"](seed)
                X = encode(X_raw, cat_idx)
                rng = np.random.default_rng(seed)
                X[rng.random(X.shape) < rate] = np.nan
                strat = y if task != "regression" else None
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, random_state=seed, stratify=strat
                )
                est = SXERegressor if task == "regression" else SXEClassifier
                Xi_tr, Xi_te = median_impute(Xtr, Xte)
                for (A, B), store in (((Xtr, Xte), native), ((Xi_tr, Xi_te), imputed)):
                    model = est(random_state=seed, n_jobs=-1).fit(A, ytr)
                    store.append(loss(task, model, B, yte))
            n, i = statistics.mean(native), statistics.mean(imputed)
            print(
                f"{name:<16}{f'{rate:.0%}':>9}{n:>12.4f}{i:>12.4f}   "
                f"{'native' if n < i else 'imputed'}",
                flush=True,
            )


if __name__ == "__main__":
    main()
