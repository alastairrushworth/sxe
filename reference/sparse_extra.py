"""
Sparse eXtra Ensemble (SXE)
===========================

Implements the method designed in conversation:

1. Global histogram binning (quantile bins, <=256) -- bin once, every
   downstream stage works on the same small integer matrix.
2. A dictionary of extremely randomized trees, each grown on a random
   subsample (without replacement) with randomized depth / max_features /
   min_samples_leaf, i.e. members deliberately span the bias-variance
   spectrum.
3. Per-tree pruning variants: for each grown tree we also emit one or more
   cost-complexity-pruned versions of the *same* tree (same random_state =>
   nested subtrees), at pruning strengths sampled from an empirical alpha
   distribution estimated from a few pilot pruning paths.  This is the
   "several dictionary members per tree at different bias-variance points"
   idea.
4. A convex, sparse, non-negative weight fit (elastic net with positive
   coefficients + intercept) on a held-out validation fold.  Trees with zero
   weight are physically dropped from the model -- this is the pruning of the
   *ensemble* and the source of the model-size reduction.
5. Optional multi-round residual fitting ("parallel boosting"): grow a batch
   of trees in parallel against the residuals of the current weighted
   ensemble, then refit the weights over ALL members so far.  Boosting-style
   bias reduction with forest-level parallelism inside each round.

Classification is handled as regression on {0,1} targets (Brier-style,
Breiman's stacked-classifier approach): leaf means are probability
estimates, the weight fit is a least-squares fit of probabilities to
outcomes, and predictions are clipped to [0,1].
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.tree import ExtraTreeRegressor

__all__ = ["QuantileBinner", "SparseExtraEnsemble"]


# ---------------------------------------------------------------------------
# Histogram binning
# ---------------------------------------------------------------------------
class QuantileBinner:
    """Bin each feature into <= n_bins quantile bins (LightGBM-style).

    transform() returns float32 bin indices so sklearn trees can consume the
    matrix directly without a per-fit dtype conversion.  Random thresholds
    drawn by ExtraTrees on this matrix are exactly random bin cut-points.
    """

    def __init__(self, n_bins: int = 256, max_sample: int = 50_000, random_state: int = 0):
        self.n_bins = n_bins
        self.max_sample = max_sample
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        if n > self.max_sample:
            X = X[rng.choice(n, self.max_sample, replace=False)]
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape, dtype=np.float32)
        for j, e in enumerate(self.edges_):
            out[:, j] = np.searchsorted(e, X[:, j], side="right")
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
@dataclass
class _Member:
    tree: ExtraTreeRegressor
    round: int
    ccp_alpha: float
    seed: int

    @property
    def n_leaves(self) -> int:
        return int(self.tree.tree_.n_leaves)


class SparseExtraEnsemble:
    """Pruned, sparsely re-weighted extremely randomized trees.

    Parameters
    ----------
    task : 'regression' | 'classification' (binary)
    n_trees : base trees grown in a single-round model
    n_rounds : residual-fitting rounds (1 = plain dictionary + weight fit)
    trees_per_round : base trees per round when n_rounds > 1
    prune_variants : if True, each base tree also contributes a
        cost-complexity-pruned nested subtree at a randomized strength
    val_fraction : fraction of training data held out for the weight fit
    subsample_range : per-tree row subsample rate range (without replacement)
    depth_choices : per-tree max_depth sampled from this tuple
    leaf_range : per-tree min_samples_leaf ~ log-uniform over this range
    feat_range : per-tree max_features fraction ~ uniform over this range
    l1_ratios / n_alphas / cv : ElasticNetCV grid for the weight fit
    """

    def __init__(
        self,
        task: str = "regression",
        n_trees: int = 300,
        n_rounds: int = 1,
        trees_per_round: int | None = None,
        prune_variants: bool = True,
        val_fraction: float = 0.2,
        n_bins: int = 256,
        subsample_range=(0.55, 0.85),
        depth_choices=(8, 10, 12, 14, None),
        leaf_range=(2, 40),
        feat_range=(0.4, 1.0),
        l1_ratios=(0.85, 1.0),
        n_alphas: int = 60,
        cv: int = 5,
        random_state: int = 0,
        n_jobs: int = -1,
    ):
        self.task = task
        self.n_trees = n_trees
        self.n_rounds = n_rounds
        self.trees_per_round = trees_per_round
        self.prune_variants = prune_variants
        self.val_fraction = val_fraction
        self.n_bins = n_bins
        self.subsample_range = subsample_range
        self.depth_choices = depth_choices
        self.leaf_range = leaf_range
        self.feat_range = feat_range
        self.l1_ratios = l1_ratios
        self.n_alphas = n_alphas
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    # -- internals ----------------------------------------------------------
    def _sample_params(self, rng):
        lo, hi = self.leaf_range
        return dict(
            max_depth=self.depth_choices[rng.integers(len(self.depth_choices))],
            max_features=float(rng.uniform(*self.feat_range)),
            min_samples_leaf=int(round(np.exp(rng.uniform(np.log(lo), np.log(hi))))),
        )

    def _pilot_alphas(self, Xtr, target, rng, n_pilot: int = 3):
        """Estimate the empirical distribution of cost-complexity alphas by
        computing full weakest-link pruning paths on a few pilot trees."""
        pool = []
        m = int(0.7 * len(target))
        for _ in range(n_pilot):
            idx = rng.choice(len(target), m, replace=False)
            est = ExtraTreeRegressor(
                max_depth=12, max_features=0.7, min_samples_leaf=5,
                random_state=int(rng.integers(2**31 - 1)),
            )
            path = est.cost_complexity_pruning_path(Xtr[idx], target[idx])
            a = np.asarray(path.ccp_alphas)
            a = a[a > 0]
            if a.size:
                pool.append(a)
        return np.concatenate(pool) if pool else np.array([])

    def _fit_weights(self, P, y_val):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enet = ElasticNetCV(
                l1_ratio=list(self.l1_ratios), n_alphas=self.n_alphas,
                positive=True, cv=self.cv, max_iter=4000,
                n_jobs=self.n_jobs, random_state=0,
            )
            enet.fit(P, y_val)
            w, b = enet.coef_.copy(), float(enet.intercept_)
            if not np.any(w > 1e-10):  # over-regularized fallback
                en = ElasticNet(alpha=float(enet.alpha_) * 1e-3, l1_ratio=1.0,
                                positive=True, max_iter=6000)
                en.fit(P, y_val)
                w, b = en.coef_.copy(), float(en.intercept_)
        return w, b

    # -- API ----------------------------------------------------------------
    def fit(self, X, y):
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y, dtype=np.float64)
        if self.task == "classification":
            self.classes_ = np.unique(y)
            if len(self.classes_) != 2:
                raise ValueError("binary classification only")
            y = (y == self.classes_[1]).astype(np.float64)

        self.binner_ = QuantileBinner(self.n_bins, random_state=self.random_state).fit(X)
        Xb = self.binner_.transform(X)

        n = len(y)
        perm = rng.permutation(n)
        n_val = int(round(self.val_fraction * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xtr, ytr = Xb[tr_idx], y[tr_idx]
        Xval, y_val = Xb[val_idx], y[val_idx]
        n_tr = len(ytr)

        members: list[_Member] = []
        val_cols: list[np.ndarray] = []
        target = ytr.copy()
        per_round = self.trees_per_round or self.n_trees
        rounds = max(1, self.n_rounds)
        self.round_selected_ = []

        for r in range(rounds):
            n_base = per_round if rounds > 1 else self.n_trees
            alpha_pool = (
                self._pilot_alphas(Xtr, target, rng) if self.prune_variants else np.array([])
            )
            for _ in range(n_base):
                params = self._sample_params(rng)
                sub = rng.uniform(*self.subsample_range)
                idx = rng.choice(n_tr, int(round(sub * n_tr)), replace=False)
                seed = int(rng.integers(2**31 - 1))
                alphas = [0.0]
                if self.prune_variants and alpha_pool.size:
                    alphas.append(float(np.quantile(alpha_pool, rng.uniform(0.5, 0.95))))
                for a in alphas:
                    est = ExtraTreeRegressor(ccp_alpha=a, random_state=seed, **params)
                    est.fit(Xtr[idx], target[idx])
                    members.append(_Member(est, r, a, seed))
                    val_cols.append(est.predict(Xval).astype(np.float32))

            P = np.column_stack(val_cols)
            w, b = self._fit_weights(P, y_val)
            sel = np.flatnonzero(w > 1e-10)
            self.round_selected_.append(len(sel))

            if r < rounds - 1:  # residual target for the next round
                pred_tr = np.full(n_tr, b)
                for i in sel:
                    pred_tr += w[i] * members[i].tree.predict(Xtr)
                target = ytr - pred_tr

        # finalize: physically drop zero-weight members
        self.intercept_ = b
        self.weights_ = w[sel].astype(np.float64)
        self.members_ = [members[i] for i in sel]
        self.n_members_grown_ = len(members)
        self.n_selected_ = len(self.members_)
        self.total_leaves_ = int(sum(m.n_leaves for m in self.members_))
        self.fit_time_ = time.perf_counter() - t0
        return self

    def _raw(self, X):
        Xb = self.binner_.transform(X)
        out = np.full(len(Xb), self.intercept_, dtype=np.float64)
        for wgt, m in zip(self.weights_, self.members_):
            out += wgt * m.tree.predict(Xb)
        return out

    def predict(self, X):
        r = self._raw(X)
        if self.task == "classification":
            return self.classes_[(r >= 0.5).astype(int)]
        return r

    def predict_proba(self, X):
        p = np.clip(self._raw(X), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def stats(self) -> dict:
        return dict(
            members_grown=self.n_members_grown_,
            members_selected=self.n_selected_,
            total_leaves=self.total_leaves_,
            fit_time_s=round(self.fit_time_, 2),
            selected_per_round=self.round_selected_,
        )
