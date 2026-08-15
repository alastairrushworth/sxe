"""ctypes wrapper around the C++ SXE core (libsxe.so).

Same algorithm as sparse_extra.SparseExtraEnsemble, with the hot path
(tree growth, holdout pruning variants, prediction) in C++:

  Python:  quantile binning, train/val split, non-negative elastic-net
           weight fit, residual-round orchestration
  C++:     histogram ExtraTrees growth on random subsamples, holdout
           reduced-error pruning at randomized strengths (raw / lambda=0 /
           random-lambda nested variants), member & weighted prediction

Pruning here is the *holdout* scheme from the original discussion (route
each tree's out-of-subsample rows once, collapse subtrees bottom-up unless
they beat the collapsed leaf by lambda*var per extra leaf), rather than the
CCP-path approximation used in the pure-Python prototype.
"""
from __future__ import annotations

import ctypes as ct
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV

from sparse_extra import QuantileBinner

_LIB = ct.CDLL(str(Path(__file__).parent / "libsxe.so"))
_LIB.sxe_create.restype = ct.c_void_p
_LIB.sxe_free.argtypes = [ct.c_void_p]
_LIB.sxe_total_members.argtypes = [ct.c_void_p]
_LIB.sxe_total_members.restype = ct.c_int
_LIB.sxe_grow.restype = ct.c_int
_LIB.sxe_grow.argtypes = [
    ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_int,
    ct.c_int, ct.c_int, ct.c_double, ct.c_double,
    ct.c_void_p, ct.c_int, ct.c_int, ct.c_int,
    ct.c_double, ct.c_double, ct.c_double, ct.c_double,
    ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_uint64,
]
_LIB.sxe_predict.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_int,
                             ct.c_void_p, ct.c_int, ct.c_void_p]
_LIB.sxe_predict_weighted.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_int,
                                      ct.c_void_p, ct.c_void_p, ct.c_int,
                                      ct.c_double, ct.c_void_p]
_LIB.sxe_member_info.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
_LIB.sxe_boost.restype = ct.c_int
_LIB.sxe_boost.argtypes = [
    ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_int,
    ct.c_void_p, ct.c_int64, ct.c_void_p,
    ct.c_int, ct.c_int, ct.c_double, ct.c_int, ct.c_double,
    ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double,
    ct.c_void_p, ct.c_int, ct.c_int, ct.c_int,
    ct.c_double, ct.c_double, ct.c_double, ct.c_double,
    ct.c_int, ct.c_uint64,
]


def _ptr(a):
    return a.ctypes.data_as(ct.c_void_p)


class CppSXE:
    """C++-backed Sparse eXtra Ensemble (drop-in analogue of SparseExtraEnsemble)."""

    def __init__(self, task="regression", n_trees=300, n_rounds=1, trees_per_round=None,
                 n_thresh=2, val_fraction=0.2, n_bins=256,
                 subsample_range=(0.55, 0.85), depth_choices=(8, 10, 12, 14, 0),
                 leaf_range=(2, 40), feat_range=(0.4, 1.0),
                 lam_range=(0.02, 2.0), emit=(1, 1, 1), hybrid_min=600, bundle_binary=False,
                 l1_ratios=(1.0,), n_alphas=20, cv=3,
                 random_state=0):
        self.task = task
        self.n_trees = n_trees
        self.n_rounds = n_rounds
        self.trees_per_round = trees_per_round
        self.n_thresh = n_thresh
        self.val_fraction = val_fraction
        self.n_bins = n_bins
        self.subsample_range = subsample_range
        self.depth_choices = depth_choices
        self.leaf_range = leaf_range
        self.feat_range = feat_range
        self.lam_range = lam_range
        self.emit = emit
        self.hybrid_min = hybrid_min
        self.bundle_binary = bundle_binary
        self.l1_ratios = l1_ratios
        self.n_alphas = n_alphas
        self.cv = cv
        self.random_state = random_state
        self._h = None

    # -- helpers -------------------------------------------------------------
    def _bin_F(self, X):
        """binned uint8, column-major."""
        if getattr(self, "bundler_", None) is not None:
            X = self.bundler_.transform(X)
        Xb = self.binner_.transform(X).astype(np.uint8)
        return np.asfortranarray(Xb)

    def _predict_members(self, XbF, mem_idx):
        n = XbF.shape[0]
        m = len(mem_idx)
        mem = np.ascontiguousarray(mem_idx, dtype=np.int32)
        out = np.empty((m, n), dtype=np.float64)  # row j = member j (col-major n x m)
        _LIB.sxe_predict(self._h, _ptr(XbF), n, XbF.shape[1], _ptr(mem), m, _ptr(out))
        return out.T  # view: n x m

    def _predict_weighted(self, XbF):
        n = XbF.shape[0]
        mem = np.ascontiguousarray(self.sel_idx_, dtype=np.int32)
        w = np.ascontiguousarray(self.weights_, dtype=np.float64)
        out = np.empty(n, dtype=np.float64)
        _LIB.sxe_predict_weighted(self._h, _ptr(XbF), n, XbF.shape[1],
                                  _ptr(mem), _ptr(w), len(mem),
                                  ct.c_double(self.intercept_), _ptr(out))
        return out

    def _fit_weights(self, P, y_val, full_cv=True):
        """Non-negative lasso weight fit. Full cross-validated alpha search on
        the first and final rounds; intermediate rounds reuse the last chosen
        alpha (slightly relaxed, since the dictionary grew) with a single
        warm coordinate-descent fit -- selection there only needs to be good
        enough to define the next residual target."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if full_cv or self._alpha_ is None:
                grid = (dict(alphas=self._alpha_ * np.logspace(0.5, -1.0, 10))
                        if self._alpha_ is not None else dict(n_alphas=self.n_alphas))
                enet = ElasticNetCV(l1_ratio=list(self.l1_ratios), positive=True, precompute=True,
                                    cv=self.cv, max_iter=4000, random_state=0, **grid)
                enet.fit(P, y_val)
                self._alpha_ = float(enet.alpha_)
                w, b = enet.coef_.copy(), float(enet.intercept_)
            else:
                en = ElasticNet(alpha=self._alpha_ * 0.7, l1_ratio=1.0,
                                positive=True, precompute=True, max_iter=4000)
                en.fit(P, y_val)
                w, b = en.coef_.copy(), float(en.intercept_)
            if not np.any(w > 1e-10):
                en = ElasticNet(alpha=(self._alpha_ or 1e-3) * 1e-3, l1_ratio=1.0,
                                positive=True, max_iter=6000)
                en.fit(P, y_val)
                w, b = en.coef_.copy(), float(en.intercept_)
        return w, b

    # -- API -----------------------------------------------------------------
    def fit(self, X, y):
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y, dtype=np.float64)
        if self.task == "classification":
            self.classes_ = np.unique(y)
            y = (y == self.classes_[1]).astype(np.float64)

        if self.bundle_binary:
            self.bundler_ = BinaryBundler().fit(X)
            X = self.bundler_.transform(X)
        else:
            self.bundler_ = None
        self.binner_ = QuantileBinner(self.n_bins, random_state=self.random_state).fit(X)
        Xb = self._bin_F_fit(X)

        n = len(y)
        perm = rng.permutation(n)
        n_val = int(round(self.val_fraction * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        XtrF = np.asfortranarray(Xb[tr_idx])
        XvalF = np.asfortranarray(Xb[val_idx])
        ytr = np.ascontiguousarray(y[tr_idx])
        y_val = y[val_idx]

        if self._h:
            _LIB.sxe_free(self._h)
        self._h = _LIB.sxe_create()

        depth = np.ascontiguousarray(self.depth_choices, dtype=np.int32)
        per_round = self.trees_per_round or self.n_trees
        rounds = max(1, self.n_rounds)
        target = ytr.copy()
        self.round_selected_ = []
        self.grow_time_ = 0.0
        self.weight_time_ = 0.0
        P_val = None
        self._alpha_ = None

        for r in range(rounds):
            n_base = per_round if rounds > 1 else self.n_trees
            tg = time.perf_counter()
            total = _LIB.sxe_grow(
                self._h, _ptr(XtrF), _ptr(np.ascontiguousarray(target)),
                XtrF.shape[0], XtrF.shape[1],
                n_base, self.n_thresh,
                *map(float, self.subsample_range),
                _ptr(depth), len(depth),
                int(self.leaf_range[0]), int(self.leaf_range[1]),
                *map(float, self.feat_range),
                *map(float, self.lam_range),
                *self.emit,
                int(self.hybrid_min),
                int(rng.integers(2**63 - 1)),
            )
            self.grow_time_ += time.perf_counter() - tg

            new_cols = self._predict_members(XvalF, np.arange(0 if P_val is None else P_val.shape[1], total))
            P_val = new_cols if P_val is None else np.hstack([P_val, new_cols])

            tw = time.perf_counter()
            w, b = self._fit_weights(P_val, y_val, full_cv=(r == 0 or r == rounds - 1))
            self.weight_time_ += time.perf_counter() - tw
            sel = np.flatnonzero(w > 1e-10)
            self.round_selected_.append(len(sel))

            self.sel_idx_ = sel.astype(np.int32)
            self.weights_ = w[sel]
            self.intercept_ = b
            if r < rounds - 1:
                target = ytr - self._predict_weighted(XtrF)

        info = np.empty(4, dtype=np.int32)
        nodes = leaves = 0
        for i in self.sel_idx_:
            _LIB.sxe_member_info(self._h, int(i), _ptr(info))
            nodes += int(info[0]); leaves += int(info[1])
        self.n_members_grown_ = int(_LIB.sxe_total_members(self._h))
        self.n_selected_ = len(self.sel_idx_)
        self.total_nodes_ = nodes
        self.total_leaves_ = leaves
        # compact serialized-size estimate: internals 13B (feat,thr,l,r), leaves 8B (value)
        self.size_mb_ = ((nodes - leaves) * 13 + leaves * 8) / 1e6
        self.fit_time_ = time.perf_counter() - t0
        return self

    def _bin_F_fit(self, X):
        return self.binner_.transform(X).astype(np.uint8)

    def _raw(self, X):
        return self._predict_weighted(self._bin_F(X))

    def predict(self, X):
        r = self._raw(X)
        if self.task == "classification":
            return self.classes_[(r >= 0.5).astype(int)]
        return r

    def predict_proba(self, X):
        p = np.clip(self._raw(X), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def stats(self):
        return dict(members_grown=self.n_members_grown_,
                    members_selected=self.n_selected_,
                    total_leaves=self.total_leaves_,
                    size_mb=round(self.size_mb_, 2),
                    grow_s=round(self.grow_time_, 2),
                    weight_fit_s=round(self.weight_time_, 2),
                    fit_s=round(self.fit_time_, 2),
                    selected_per_round=self.round_selected_)

    def __del__(self):
        try:
            if self._h:
                _LIB.sxe_free(self._h)
        except Exception:
            pass


class BinaryBundler:
    """Exclusive Feature Bundling (LightGBM-style) restricted to binary
    columns: mutually exclusive 0/1 features (e.g. one-hot blocks) are
    merged into single small-integer columns (0 = none active, k = k-th
    bundled feature active). Conflict tolerance is zero, so the transform
    is lossless up to the ordinal encoding of the bundle."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        is_bin = [j for j in range(p)
                  if np.all((X[:, j] == 0) | (X[:, j] == 1)) and np.unique(X[:, j]).size <= 2]
        self.passthrough_ = [j for j in range(p) if j not in set(is_bin)]
        nz = X[:, is_bin] != 0
        order = np.argsort(-nz.sum(axis=0))
        bundles = []  # (col_list, used_rows)
        for k in order:
            col = nz[:, k]
            placed = False
            for cols, used in bundles:
                if len(cols) < 250 and not np.any(used & col):
                    cols.append(is_bin[k]); used |= col; placed = True; break
            if not placed:
                bundles.append(([is_bin[k]], col.copy()))
        self.bundles_ = [cols for cols, _ in bundles if len(cols) >= 2]
        bundled = {c for cols in self.bundles_ for c in cols}
        self.passthrough_ += [j for j in is_bin if j not in bundled]
        self.passthrough_.sort()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = [X[:, self.passthrough_]]
        for cols in self.bundles_:
            sub = X[:, cols] != 0
            out.append(np.where(sub.any(axis=1), sub.argmax(axis=1) + 1, 0)[:, None].astype(np.float64))
        return np.hstack(out)


class CppSXEBoost(CppSXE):
    """v3: batch-boosted dictionary generation entirely in C++ + one final
    sparse non-negative weight fit. Each tree emits a single member, pruned
    on its own holdout at a per-tree random strength (nu-shrunk residual
    fitting, early stopping on validation MSE)."""

    def __init__(self, task="regression", batch=10, max_batches=120, shrink=0.15,
                 patience=8, n_thresh=2, argmax_bins=1, max_leaves=0, val_fraction=0.18, n_bins=256,
                 subsample_range=(0.4, 0.75), depth_choices=(4, 5, 6, 7, 8),
                 leaf_range=(10, 60), feat_range=(0.5, 1.0), lam_range=(1e-3, 1.0),
                 hybrid_min=600, bundle_binary=False,
                 l1_ratios=(1.0,), n_alphas=20, cv=3, random_state=0):
        super().__init__(task=task, n_thresh=n_thresh, val_fraction=val_fraction,
                         n_bins=n_bins, subsample_range=subsample_range,
                         depth_choices=depth_choices, leaf_range=leaf_range,
                         feat_range=feat_range, lam_range=lam_range,
                         hybrid_min=hybrid_min, bundle_binary=bundle_binary,
                         l1_ratios=l1_ratios, n_alphas=n_alphas, cv=cv,
                         random_state=random_state)
        self.batch = batch
        self.max_batches = max_batches
        self.group_by_batch = True
        self.shrink = shrink
        self.patience = patience
        self.argmax_bins = argmax_bins
        self.max_leaves = max_leaves

    def fit(self, X, y):
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y, dtype=np.float64)
        if self.task == "classification":
            self.classes_ = np.unique(y)
            y = (y == self.classes_[1]).astype(np.float64)

        if self.bundle_binary:
            self.bundler_ = BinaryBundler().fit(X)
            X = self.bundler_.transform(X)
        else:
            self.bundler_ = None
        self.binner_ = QuantileBinner(self.n_bins, random_state=self.random_state).fit(X)
        Xb = self.binner_.transform(X).astype(np.uint8)

        n = len(y)
        perm = rng.permutation(n)
        n_val = int(round(self.val_fraction * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        XtrF = np.asfortranarray(Xb[tr_idx]); ytr = np.ascontiguousarray(y[tr_idx])
        XvalF = np.asfortranarray(Xb[val_idx]); y_val = np.ascontiguousarray(y[val_idx])

        if self._h:
            _LIB.sxe_free(self._h)
        self._h = _LIB.sxe_create()
        depth = np.ascontiguousarray(self.depth_choices, dtype=np.int32)

        tg = time.perf_counter()
        total = _LIB.sxe_boost(
            self._h, _ptr(XtrF), _ptr(ytr), XtrF.shape[0], XtrF.shape[1],
            _ptr(XvalF), XvalF.shape[0], _ptr(y_val),
            int(self.batch), int(self.max_batches), float(self.shrink),
            int(self.patience), float(1e-4 * np.var(y_val)),
            int(self.n_thresh), int(self.argmax_bins), int(self.max_leaves),
            *map(float, self.subsample_range),
            _ptr(depth), len(depth),
            int(self.leaf_range[0]), int(self.leaf_range[1]),
            *map(float, self.feat_range),
            *map(float, self.lam_range),
            int(self.hybrid_min), int(rng.integers(2**63 - 1)),
        )
        self.grow_time_ = time.perf_counter() - tg

        P_val = self._predict_members(XvalF, np.arange(total))
        tw = time.perf_counter()
        self._alpha_ = None
        if self.group_by_batch:
            nb = total // self.batch
            Pb = P_val[:, :nb * self.batch].reshape(len(y_val), nb, self.batch).mean(axis=2)
            wb, b = self._fit_weights(np.ascontiguousarray(Pb), y_val, full_cv=True)
            selb = np.flatnonzero(wb > 1e-10)
            sel = (selb[:, None] * self.batch + np.arange(self.batch)[None, :]).ravel()
            w_mem = np.repeat(wb[selb] / self.batch, self.batch)
        else:
            w, b = self._fit_weights(P_val, y_val, full_cv=True)
            sel = np.flatnonzero(w > 1e-10)
            w_mem = w[sel]
        self.weight_time_ = time.perf_counter() - tw
        self.sel_idx_ = np.ascontiguousarray(sel, dtype=np.int32)
        self.weights_ = np.ascontiguousarray(w_mem, dtype=np.float64)
        self.intercept_ = b
        self.round_selected_ = [len(sel)]

        info = np.empty(4, dtype=np.int32)
        nodes = leaves = 0
        for i in self.sel_idx_:
            _LIB.sxe_member_info(self._h, int(i), _ptr(info))
            nodes += int(info[0]); leaves += int(info[1])
        self.n_members_grown_ = total
        self.n_selected_ = len(sel)
        self.total_nodes_ = nodes
        self.total_leaves_ = leaves
        self.size_mb_ = ((nodes - leaves) * 13 + leaves * 8) / 1e6
        self.fit_time_ = time.perf_counter() - t0
        return self
