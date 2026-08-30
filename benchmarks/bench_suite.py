"""SXE against LightGBM across nine datasets, both tuned, both at defaults.

Every dataset produces four rows, which answer two different questions:

  * **``SXE`` vs ``LightGBM``** -- both tuned by random search over a fixed
    space with the same number of draws, so this is the head-to-head with each
    library working as intended. The search runs once per dataset, on seed 0,
    and the winning configuration is cached to ``tuned_configs.json`` and
    reused across seeds. LightGBM's early-stopping fold doubles as its
    selection fold, which is mildly generous to it.
  * **``SXE-default`` vs ``LightGBM-default``** -- both at package defaults.
    The design goal is that SXE should not need tuning the way LightGBM does;
    the gap between each library's tuned and default rows is what measures
    that, rather than asserting it. Cheap to collect, and it is the number
    that tells a user what they get before they have done any work.

Both libraries get the same preprocessed matrix, so the comparison is model
against model rather than preprocessing against preprocessing. Where a dataset
has categorical columns or missing values, a fifth row ``LightGBM-native``
reports LightGBM on raw NaNs and declared categoricals: the gap between it and
``LightGBM`` is the price of SXE's missing features, measured rather than
asserted.

Size uses one formula on both sides (13 bytes per internal node, 8 per leaf,
plus the bin table a binned model needs to predict at all). Time is each
library at its own best thread count -- LightGBM slows monotonically with
threads on small data, so timing both at ``n_jobs=-1`` would report an SXE win
that is an artefact of misconfiguring the baseline.

    python benchmarks/bench_suite.py                     # everything
    python benchmarks/bench_suite.py -d adult letter     # a subset
    python benchmarks/bench_suite.py --seeds 0           # single seed, quick
    python benchmarks/bench_suite.py --retune            # redo both searches

Datasets download via scikit-learn and OpenML on first use, then cache.

Reproducibility: SXE draws per-tree randomness through
``std::uniform_*_distribution``, whose algorithms are implementation-defined,
so figures reproduce exactly only on the same C++ standard library. Results
are invariant to ``n_jobs`` and to ``hybrid_min`` on any one platform.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import time
import warnings

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml, make_friedman1
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from sxe import SXEClassifier, SXERegressor

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).parent

# Two size columns, because they answer different questions.
#
# `portable_bytes` is one formula applied identically to both libraries: 13
# bytes per internal node (feature int32, threshold uint8, two child int32), 8
# per leaf, plus a full 255-edge float64 bin table per input feature. It
# compares *tree structure* on equal terms and nothing else. It is a poor guide
# to actual size: it charges LightGBM for a bin table its format does not need,
# and it charges both for edges that deduplicate away on low-cardinality data
# or are never indexed at all on wide data.
#
# `real_bytes` is what each library would actually have to ship. For SXE it is
# measured -- `model_size_bytes_`, the bytes serialize() writes plus the
# quantile edges of the columns the trees index. For LightGBM it is the
# smallest sensible encoding of what its trees need: a float64 threshold rather
# than a bin id, an explicit left child, and no bin table. That is generous to
# LightGBM, whose real files also carry split gains and node counts.
BYTES_PER_INTERNAL = 13
BYTES_PER_LEAF = 8
BYTES_PER_BIN_EDGE = 8
BINS_PER_FEATURE = 255
LGBM_BYTES_PER_INTERNAL = 20  # feature int32 + threshold float64 + 2 x child int32


def _use_certifi():
    """Point OpenSSL at certifi's CA bundle if the interpreter has no trust store.

    python.org builds on macOS ship without one unless the user has run
    ``Install Certificates.command``, and every OpenML fetch then fails on
    CERTIFICATE_VERIFY_FAILED. Only fills in a value that is not already set.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:  # pragma: no cover - certifi ships with most installs
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
# Each loader returns (X, y, task, cat_idx). X may be an object array carrying
# NaN and string categories; cat_idx names the categorical columns. The suite
# is chosen so that every entry stresses a different property, and several
# stress properties SXE does not have.


def _openml(data_id):
    _use_certifi()
    return fetch_openml(data_id=data_id, as_frame=True, parser="auto")


def _frame(bunch, y=None):
    """Split a dataframe into a float/object matrix plus categorical indices."""
    df = bunch.data
    cat_idx = [i for i, c in enumerate(df.columns) if str(df[c].dtype) in ("category", "object")]
    X = df.to_numpy(dtype=object)
    target = bunch.target if y is None else y
    return X, np.asarray(target), cat_idx


def load_california(seed):
    d = fetch_california_housing()
    return d.data, d.target, "regression", []


def load_friedman1(seed):
    X, y = make_friedman1(n_samples=25_000, n_features=25, noise=1.0, random_state=seed)
    return X, y, "regression", []


def load_superconduct(seed):
    b = _openml(43174)
    X, y, cat = _frame(b)
    return X.astype(np.float64), y.astype(np.float64), "regression", cat


def load_covtype_binary(seed):
    d = fetch_covtype()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d.target), 30_000, replace=False)
    return d.data[idx], (d.target[idx] == 2).astype(int), "classification", []


def load_covtype_full(seed):
    d = fetch_covtype()
    return d.data, (d.target == 2).astype(int), "classification", []


def load_adult(seed):
    b = _openml(1590)
    X, y, cat = _frame(b)
    return X, (np.asarray(y) == ">50K").astype(int), "classification", cat


def load_bank_marketing(seed):
    b = _openml(1461)
    X, y, cat = _frame(b)
    return X, (np.asarray(y).astype(str) == "2").astype(int), "classification", cat


def load_letter(seed):
    b = _openml(6)
    X, y, cat = _frame(b)
    return X.astype(np.float64), np.asarray(y), "multiclass", cat


def load_madelon(seed):
    b = _openml(1485)
    X, y, cat = _frame(b)
    return X.astype(np.float64), np.asarray(y).astype(int), "classification", cat


DATASETS = {
    # loader, seeds averaged over, tuning draws per library, SXE max_iter cap,
    # thread counts to time at, and what the dataset is here to stress.
    #
    # threads=(1, -1) times each library at its own best thread count, which
    # matters on small data where LightGBM's within-tree parallelism costs it.
    # Where a single fit runs into minutes, only all-cores is timed -- that is
    # the setting anyone would actually use at that size, and the sweep would
    # otherwise dominate the suite's runtime.
    "california": dict(
        loader=load_california,
        seeds=5,
        budget=24,
        cap=1500,
        threads=(1, -1),
        note="smooth low-dimensional signal, small n",
    ),
    "friedman1_noise": dict(
        loader=load_friedman1,
        seeds=5,
        budget=24,
        cap=600,
        threads=(1, -1),
        note="known signal buried in 20 noise features",
    ),
    "superconduct": dict(
        loader=load_superconduct,
        seeds=5,
        budget=24,
        cap=600,
        threads=(1, -1),
        note="real regression, 81 correlated features",
    ),
    "covtype_binary": dict(
        loader=load_covtype_binary,
        seeds=5,
        budget=24,
        cap=600,
        threads=(1, -1),
        note="the original reference point, 30k subsample",
    ),
    "covtype_full": dict(
        loader=load_covtype_full,
        seeds=3,
        budget=10,
        cap=600,
        threads=(-1,),
        note="large n: 581k rows, where gradient-boosting scaling shows",
    ),
    "adult": dict(
        loader=load_adult,
        seeds=5,
        budget=24,
        cap=600,
        threads=(1, -1),
        note="mixed-cardinality categoricals (2-41), missing values in them",
    ),
    "bank_marketing": dict(
        loader=load_bank_marketing,
        seeds=5,
        budget=24,
        cap=600,
        threads=(1, -1),
        note="categoricals (2-12), no missing values",
    ),
    "letter": dict(
        loader=load_letter,
        seeds=3,
        budget=16,
        cap=300,
        threads=(-1,),
        note="26 classes: the cost of the one-vs-rest reduction",
    ),
    "madelon": dict(
        loader=load_madelon,
        seeds=5,
        budget=24,
        cap=600,
        threads=(1, -1),
        note="p >> n: 500 features, 2600 rows, mostly noise",
    ),
}

# The hand-tuned configurations from the original report. They are entered
# into the SXE search as warm starts so the random search cannot come out
# behind the hand tuning it replaces.
TUNED_SXE = {
    "california": dict(
        max_leaf_nodes=63,
        max_depth_choices=(0,),
        learning_rate=0.05,
        batch_size=4,
        max_iter=1200,
        n_iter_no_change=25,
        min_samples_leaf_range=(10, 25),
        holdout="random",
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
        holdout="random",
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
        holdout="random",
        subsample_range=(0.5, 0.8),
        n_iter_no_change=5,
        validation_fraction=0.15,
        n_alphas=15,
        splitter="random",
        n_thresholds=3,
    ),
}


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def encode(X, cat_idx):
    """Ordinal-code categorical columns; leave everything else numeric.

    Codes come from the whole column rather than the training split. No target
    is involved, so this leaks nothing about y -- it only spares the benchmark
    an unseen-category branch that would otherwise differ between libraries.
    Missing categories get their own code, which is the honest encoding of
    "not recorded" and is what LightGBM does natively.
    """
    X = np.asarray(X, dtype=object)
    out = np.empty(X.shape, dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        if j in cat_idx:
            vals = np.array([("" if v is None or v != v else str(v)) for v in col])
            _, codes = np.unique(vals, return_inverse=True)
            out[:, j] = codes.astype(np.float64)
        else:
            out[:, j] = np.array([np.nan if v is None else float(v) for v in col])
    return out


def impute(Xtr, Xte):
    """Median-impute numeric NaNs from the training split only.

    SXE rejects non-finite input outright, so without this it cannot fit half
    the suite. That is the point: the imputation is a workaround for a missing
    feature, and LightGBM-native measures what the workaround costs.
    """
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for A in (Xtr, Xte):
        bad = ~np.isfinite(A)
        if bad.any():
            A[bad] = np.take(med, np.nonzero(bad)[1])
    return Xtr, Xte


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def portable_bytes(n_leaves, n_trees, n_features):
    """The equal-terms structural formula. Trees are full binary, so
    internal nodes = leaves - trees."""
    n_internal = max(0, n_leaves - n_trees)
    struct = n_internal * BYTES_PER_INTERNAL + n_leaves * BYTES_PER_LEAF
    bins = n_features * BINS_PER_FEATURE * BYTES_PER_BIN_EDGE
    return struct, bins, struct + bins


def lightgbm_real_bytes(n_leaves, n_trees):
    """What LightGBM's trees would cost in the smallest sensible encoding."""
    return max(0, n_leaves - n_trees) * LGBM_BYTES_PER_INTERNAL + n_leaves * BYTES_PER_LEAF


def score(task, model, X, y):
    """(primary metric, value, secondary metric, value)."""
    if task == "regression":
        pred = model.predict(X)
        return ("rmse", mean_squared_error(y, pred) ** 0.5, "mae", mean_absolute_error(y, pred))
    proba = model.predict_proba(X)
    if task == "classification":
        return (
            "auc",
            roc_auc_score(y, proba[:, 1]),
            "logloss",
            log_loss(y, np.clip(proba[:, 1], 1e-15, 1 - 1e-15)),
        )
    return (
        "accuracy",
        accuracy_score(y, model.predict(X)),
        "logloss",
        log_loss(y, proba, labels=list(model.classes_)),
    )


def result(model_name, task, model, X, y, fit_s, n_trees, n_leaves, n_features, real_bytes):
    m1, v1, m2, v2 = score(task, model, X, y)
    struct, bins, portable = portable_bytes(n_leaves, n_trees, n_features)
    return dict(
        model=model_name,
        metric=m1,
        value=v1,
        metric2=m2,
        value2=v2,
        fit_s=fit_s,
        n_trees=n_trees,
        n_leaves=n_leaves,
        n_features_model=n_features,
        struct_bytes=struct,
        bins_bytes=bins,
        portable_bytes=portable,
        real_bytes=real_bytes,
    )


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------
def run_sxe(task, Xtr, ytr, Xte, yte, n_jobs, seed, cfg=None, name="SXE"):
    est = SXERegressor if task == "regression" else SXEClassifier
    model = est(random_state=seed, n_jobs=n_jobs, **(cfg or {}))
    t = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t
    # the feature count the trees actually index, i.e. after any bundling
    n_feat = model.bundler_.n_features_out_ if model.bundler_ is not None else Xtr.shape[1]
    return result(name, task, model, Xte, yte, fit_s, model.n_trees_kept_, model.n_leaves_, n_feat)


# --- search spaces ---------------------------------------------------------
# Both spaces are the same size in spirit: roughly a dozen knobs, ranges wide
# enough to contain each library's own documented tuning advice, and the same
# number of draws from each.

LGBM_SPACE = dict(
    learning_rate=(0.01, 0.3),  # log-uniform
    num_leaves=(8, 512),  # log-uniform, integer
    min_child_samples=(5, 100),  # log-uniform, integer
    feature_fraction=(0.4, 1.0),
    bagging_fraction=(0.5, 1.0),
    lambda_l2=(1e-3, 10.0),  # log-uniform
)

SXE_DEPTHS = [(4, 5, 6), (6, 7, 8), (8, 10, 12), (10, 12, 14), (0,)]
SXE_LEAVES = [(2, 15), (3, 20), (5, 30), (10, 60), (20, 70)]
# holdout design: out-of-batch folds (each tree sees 3/4 or 7/8 of the rows)
# or the older per-tree random subsample at one of these rates
SXE_HOLDOUT = [("batch", 8), ("batch", 8), ("batch", 4), ("random", (0.4, 0.75)),
               ("random", (0.5, 0.8)), ("random", (0.9, 1.0))]
SXE_FEATURES = [(0.35, 0.65), (0.5, 1.0), (0.85, 1.0)]
SXE_PRUNE = [(1e-3, 0.05), (1e-3, 1.0), (1e-2, 2.0)]
SXE_SPLITTERS = [("best", 2), ("random", 2), ("random", 3)]
SXE_LEAFWISE = [None, None, 31, 63, 127, 255, 511]  # None twice: depth-wise is the default shape


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def sample_lgbm(rng, cap, n_binary):
    return dict(
        learning_rate=_logu(rng, *LGBM_SPACE["learning_rate"]),
        num_leaves=int(round(_logu(rng, *LGBM_SPACE["num_leaves"]))),
        min_child_samples=int(round(_logu(rng, *LGBM_SPACE["min_child_samples"]))),
        feature_fraction=float(rng.uniform(*LGBM_SPACE["feature_fraction"])),
        bagging_fraction=float(rng.uniform(*LGBM_SPACE["bagging_fraction"])),
        bagging_freq=1,
        lambda_l2=_logu(rng, *LGBM_SPACE["lambda_l2"]),
    )


def sample_sxe(rng, cap, n_binary):
    """One draw from the SXE space.

    ``max_iter`` is pinned to the per-dataset cap rather than sampled: batches
    are early-stopped, so the trajectory length is set by ``n_iter_no_change``
    and the learning rate, and sampling both would just waste draws on runs
    that stop at the same place.
    """
    params = dict(
        learning_rate=_logu(rng, 0.01, 0.4),
        batch_size=int(_pick(rng, [4, 8, 12, 16])),
        max_iter=int(cap),
        n_iter_no_change=int(_pick(rng, [3, 8, 20, 40])),
        min_samples_leaf_range=_pick(rng, SXE_LEAVES),
        max_features_range=_pick(rng, SXE_FEATURES),
        prune_strength_range=_pick(rng, SXE_PRUNE),
        validation_fraction=float(_pick(rng, [0.10, 0.15, 0.18])),
    )
    params["splitter"], params["n_thresholds"] = _pick(rng, SXE_SPLITTERS)
    mode, arg = _pick(rng, SXE_HOLDOUT)
    if mode == "batch":
        params["holdout"], params["holdout_folds"] = "batch", int(arg)
    else:
        params["holdout"], params["subsample_range"] = "random", tuple(arg)
    leafwise = _pick(rng, SXE_LEAFWISE)
    if leafwise is None:
        params["max_depth_choices"] = _pick(rng, SXE_DEPTHS)
    else:
        params["max_leaf_nodes"] = int(leafwise)
        params["max_depth_choices"] = (0,)
    if n_binary >= 8:
        # only worth a draw where there is something to bundle
        params["bundle_binary_features"] = bool(rng.integers(2))
    return params


# --- fitting ---------------------------------------------------------------
def _lgbm_fit(task, params, Xa, ya, Xb, yb, seed, n_jobs, cat_idx=None):
    import lightgbm as lgb

    LG = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    g = LG(n_estimators=3000, random_state=seed, n_jobs=n_jobs, verbose=-1, **params)
    kw = {"categorical_feature": list(cat_idx)} if cat_idx else {}
    t = time.perf_counter()
    g.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(100, verbose=False)], **kw)
    return g, time.perf_counter() - t


def _sxe_fit(task, params, Xa, ya, Xb, yb, seed, n_jobs, cat_idx=None):
    """SXE carves its own validation fold, so (Xb, yb) is unused here."""
    est = SXERegressor if task == "regression" else SXEClassifier
    m = est(random_state=seed, n_jobs=n_jobs, **params)
    t = time.perf_counter()
    m.fit(Xa, ya)
    return m, time.perf_counter() - t


FITTERS = {"SXE": _sxe_fit, "LightGBM": _lgbm_fit}
SAMPLERS = {"SXE": sample_sxe, "LightGBM": sample_lgbm}


def selection_loss(task, model, X, y):
    """Lower is better, for every task."""
    if task == "regression":
        return mean_squared_error(y, model.predict(X)) ** 0.5
    if task == "classification":
        return -roc_auc_score(y, model.predict_proba(X)[:, 1])
    return -accuracy_score(y, model.predict(X))


def fitted_size(library, model):
    """Bytes the fitted model would ship, in each library's own format."""
    if library == "SXE":
        return int(model.model_size_bytes_)
    info = model.booster_.dump_model()["tree_info"]
    return lightgbm_real_bytes(sum(t["num_leaves"] for t in info), len(info))


def tune(library, task, Xtr, ytr, seed, budget, cap, n_binary, warm=(), record=None):
    """Random search on a held-out fold, identical protocol for both libraries.

    LightGBM early-stops on the same fold it is selected on, which is mildly
    generous to it; SXE early-stops on a fold carved from the training half
    and so is selected on data it has not seen at all. Correcting that would
    mean a third fold and less training data for LightGBM, which would be a
    different kind of unfair.

    Every draw's (loss, size) goes into ``record``. Selection optimises
    accuracy alone, so the winner says nothing about what either library can do
    under a size budget -- the draws do, and they cost nothing extra to keep.
    """
    strat = ytr if task != "regression" else None
    Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.15, random_state=seed, stratify=strat)
    rng = np.random.default_rng(seed)
    sampler, fitter = SAMPLERS[library], FITTERS[library]
    draws = list(warm) + [sampler(rng, cap, n_binary) for _ in range(budget)]
    best, best_loss = None, np.inf
    for i, params in enumerate(draws):
        try:
            model, fit_s = fitter(task, params, Xa, ya, Xb, yb, seed, -1)
            loss = selection_loss(task, model, Xb, yb)
            size = fitted_size(library, model)
        except Exception as exc:  # a pathological draw, e.g. num_leaves > rows
            print(f"    {library} [{i + 1}/{len(draws)}] skipped: {type(exc).__name__}", flush=True)
            continue
        if record is not None:
            record.append(dict(library=library, draw=i, loss=loss, real_bytes=size, fit_s=fit_s))
        if loss < best_loss:
            best, best_loss = params, loss
            print(
                f"    {library} [{i + 1}/{len(draws)}] best {loss:+.5f} "
                f"({fit_s:.1f}s, {size / 1e6:.2f} MB) {params}",
                flush=True,
            )
    return best, float(best_loss)


def run_model(library, task, Xtr, ytr, Xte, yte, params, n_jobs, seed, cat_idx=None, tag=None):
    """Fit one configuration on the full training split and score it."""
    tag = tag or library
    if library == "SXE":
        m, fit_s = _sxe_fit(task, params, Xtr, ytr, None, None, seed, n_jobs)
        n_feat = m.bundler_.n_features_out_ if m.bundler_ is not None else Xtr.shape[1]
        return result(
            tag,
            task,
            m,
            Xte,
            yte,
            fit_s,
            m.n_trees_kept_,
            m.n_leaves_,
            n_feat,
            m.model_size_bytes_,
        )
    strat = ytr if task != "regression" else None
    Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.15, random_state=seed, stratify=strat)
    g, fit_s = _lgbm_fit(task, params, Xa, ya, Xb, yb, seed, n_jobs, cat_idx=cat_idx)
    info = g.booster_.dump_model()["tree_info"]
    n_leaves = sum(t["num_leaves"] for t in info)
    return result(
        tag,
        task,
        g,
        Xte,
        yte,
        fit_s,
        len(info),
        n_leaves,
        Xtr.shape[1],
        lightgbm_real_bytes(n_leaves, len(info)),
    )


def load_tuned(path):
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def summarise(rows, names):
    def cell(vals):
        m = statistics.mean(vals)
        return m, (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    for name in names:
        sub = [r for r in rows if r["dataset"] == name]
        if not sub:
            continue
        m1, m2 = sub[0]["metric"], sub[0]["metric2"]
        seeds = len({r["seed"] for r in sub})
        print(f"\n=== {name} ({m1} / {m2}, {seeds} seeds) ===")
        head = (
            f"  {'model':<18}{m1:<20}{m2:<18}{'leaves':<11}{'real MB':<10}"
            f"{'portable':<11}{'best fit s':<14}{'@threads'}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for model in sorted({r["model"] for r in sub}):
            ms = [r for r in sub if r["model"] == model]
            # accuracy and structure are thread-invariant: read them off the
            # lowest thread count so repeated threads do not double-count
            base = [r for r in ms if r["n_jobs"] == min(x["n_jobs"] for x in ms)]
            v1m, v1s = cell([r["value"] for r in base])
            v2m, _ = cell([r["value2"] for r in base])
            lm, _ = cell([r["n_leaves"] for r in base])
            sm, _ = cell([r["real_bytes"] for r in base])
            pm, _ = cell([r["portable_bytes"] for r in base])
            by_thread = {}
            for r in ms:
                by_thread.setdefault(r["n_jobs"], []).append(r["fit_s"])
            best_nj = min(by_thread, key=lambda k: statistics.mean(by_thread[k]))
            tm, ts = cell(by_thread[best_nj])
            print(
                f"  {model:<18}{v1m:.4f} +- {v1s:.4f}   {v2m:>9.4f}         "
                f"{lm:>9,.0f}  {sm / 1e6:>8.3f}  {pm / 1e6:>9.3f}  "
                f"{tm:>6.2f} +- {ts:<5.2f}  {best_nj:>5}"
            )


def size_budgets(draws, names):
    """Best selection loss each library reached under a size ceiling.

    This is the question the size claim is actually about -- what accuracy is
    available if the model has to fit in N megabytes -- and it is the one a
    single accuracy-tuned point per library cannot answer.
    """
    for name in names:
        sub = [d for d in draws if d["dataset"] == name]
        if not sub:
            continue
        libs = sorted({d["library"] for d in sub})
        ceilings = [0.1e6, 0.5e6, 2e6, 10e6, float("inf")]
        print(f"\n=== {name}: best selection loss under a size ceiling ===")
        print("  ceiling   " + "".join(f"{lib:>14}" for lib in libs))
        for c in ceilings:
            label = "any" if c == float("inf") else f"{c / 1e6:g} MB"
            cells = []
            for lib in libs:
                ok = [d["loss"] for d in sub if d["library"] == lib and d["real_bytes"] <= c]
                cells.append(f"{min(ok):>14.5f}" if ok else f"{'-':>14}")
            print(f"  {label:<10}" + "".join(cells))


def write_csv(rows, out):
    fieldnames = list(rows[0])
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(out, index=False)
    except ImportError:
        import csv

        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {out}")


def prepare(loader, seed):
    """One dataset at one seed: split, encode, impute, and describe."""
    X_raw, y, task, cat_idx = loader(seed)
    X_enc = encode(X_raw, cat_idx)
    strat = y if task != "regression" else None
    Xtr_r, Xte_r, ytr, yte = train_test_split(
        X_enc, y, test_size=0.2, random_state=seed, stratify=strat
    )
    Xtr, Xte = impute(Xtr_r, Xte_r)
    return dict(
        task=task,
        cat_idx=cat_idx,
        Xtr=Xtr,
        Xte=Xte,
        Xtr_raw=Xtr_r,
        Xte_raw=Xte_r,
        ytr=ytr,
        yte=yte,
        n=len(X_enc),
        p=X_enc.shape[1],
        n_missing=int((~np.isfinite(Xtr_r)).sum()),
        n_binary=int(sum(len(np.unique(Xtr[:, j])) == 2 for j in range(Xtr.shape[1]))),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", "--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="override per-dataset seeds")
    ap.add_argument(
        "--n-jobs", type=int, nargs="*", default=None, help="override the per-dataset thread counts"
    )
    ap.add_argument("--retune", action="store_true", help="redo both random searches")
    ap.add_argument("--budget", type=int, default=None, help="override tuning draws per library")
    ap.add_argument("--libraries", nargs="*", default=["SXE", "LightGBM"])
    ap.add_argument("-o", "--out", default=str(HERE / "results_suite.csv"))
    ap.add_argument("--tuned-file", default=str(HERE / "tuned_configs.json"))
    args = ap.parse_args()

    tuned_path = pathlib.Path(args.tuned_file)
    tuned = load_tuned(tuned_path)
    rows, draws = [], []

    for name in args.datasets:
        spec = DATASETS[name]
        loader, cap, note = spec["loader"], spec["cap"], spec["note"]
        seeds = args.seeds if args.seeds is not None else list(range(spec["seeds"]))
        budget = args.budget if args.budget is not None else spec["budget"]
        threads = args.n_jobs if args.n_jobs is not None else spec["threads"]
        splits = {seeds[0]: prepare(loader, seeds[0])}

        # The search runs ONCE per dataset, on the first seed's split, and the
        # winner is reused for every seed. Searching per seed would tune each
        # model on the very split it is then scored on.
        d0 = splits[seeds[0]]
        for library in args.libraries:
            key = f"{name}/{library}"
            if not (args.retune or key not in tuned):
                continue
            print(f"\n### tuning {library} on {name} ({budget} draws) ###", flush=True)
            # draw zero is always the library's own defaults, so a "tuned" row
            # can never be worse than its "default" row on the selection fold;
            # the hand-tuned configs from the original report join it where
            # they exist
            warm = [{}]
            if library == "SXE" and name in TUNED_SXE:
                warm.append(TUNED_SXE[name])
            first = len(draws)
            params, loss = tune(
                library,
                d0["task"],
                d0["Xtr"],
                d0["ytr"],
                seeds[0],
                budget,
                cap,
                d0["n_binary"],
                warm,
                draws,
            )
            for d in draws[first:]:
                d["dataset"] = name
            tuned[key] = {"params": params, "search_loss": loss}
            tuned_path.write_text(json.dumps(tuned, indent=2, sort_keys=True, default=list))

        for seed in seeds:
            d = splits.pop(seed, None) or prepare(loader, seed)
            for n_jobs in threads:
                print(
                    f"\n=== {name} ({d['task']}, n={d['n']}, p={d['p']}, "
                    f"cats={len(d['cat_idx'])}, binary={d['n_binary']}, "
                    f"missing={d['n_missing']}) seed={seed} n_jobs={n_jobs} ===\n    {note}",
                    flush=True,
                )
                runs = []
                for library in args.libraries:
                    params = tuned[f"{name}/{library}"]["params"]
                    runs.append(
                        run_model(
                            library,
                            d["task"],
                            d["Xtr"],
                            d["ytr"],
                            d["Xte"],
                            d["yte"],
                            params,
                            n_jobs,
                            seed,
                        )
                    )
                    runs.append(
                        run_model(
                            library,
                            d["task"],
                            d["Xtr"],
                            d["ytr"],
                            d["Xte"],
                            d["yte"],
                            {},
                            n_jobs,
                            seed,
                            tag=f"{library}-default",
                        )
                    )
                    if library == "LightGBM" and (d["cat_idx"] or d["n_missing"]):
                        # LightGBM as it would actually be used on this data:
                        # declared categoricals and raw NaNs, no imputation.
                        # The gap to the row above prices SXE's preprocessing.
                        runs.append(
                            run_model(
                                library,
                                d["task"],
                                d["Xtr_raw"],
                                d["ytr"],
                                d["Xte_raw"],
                                d["yte"],
                                params,
                                n_jobs,
                                seed,
                                cat_idx=d["cat_idx"],
                                tag="LightGBM-native",
                            )
                        )
                for r in runs:
                    r.update(dataset=name, task=d["task"], seed=seed, n_jobs=n_jobs)
                    print(
                        f"  {r['model']:<18} {r['metric']} {r['value']:.4f}  "
                        f"{r['metric2']} {r['value2']:.4f}  {r['fit_s']:7.2f}s  "
                        f"{r['real_bytes'] / 1e6:8.3f} MB  {r['n_leaves']:>8,} leaves",
                        flush=True,
                    )
                rows += runs

    summarise(rows, args.datasets)
    if draws:
        size_budgets(draws, args.datasets)
        write_csv(draws, str(HERE / "tuning_draws.csv"))
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
