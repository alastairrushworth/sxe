"""scikit-learn estimators for Sparse eXtra Ensembles."""

from __future__ import annotations

import time
import warnings

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression, enet_path
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from . import _core
from .binning import MAX_BINS, QuantileBinner
from .bundling import BinaryBundler

__all__ = ["SXERegressor", "SXEClassifier"]

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data as _sk_validate_data
except ImportError:  # pragma: no cover - older scikit-learn
    _sk_validate_data = None


# scikit-learn's "y was not supplied at all" sentinel. It must be distinct
# from None, which means "y was supplied and is invalid".
_NO_VALIDATION = "no_validation"


def _validate(est, X, y=_NO_VALIDATION, *, reset=True, **kw):
    kw.setdefault(_FINITE_KW, "allow-nan")
    if _sk_validate_data is not None:
        return _sk_validate_data(est, X, y, reset=reset, **kw)
    return est._validate_data(X, y, reset=reset, **kw)  # pragma: no cover


def _sklearn_at_least(major, minor):
    try:
        parts = sklearn.__version__.split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (AttributeError, IndexError, ValueError):  # pragma: no cover
        return True


# scikit-learn 1.7 folded ``n_alphas`` into ``alphas`` (which now accepts an
# int) and deprecated it; 1.9 removed it outright. Same regularisation path
# either way -- only the keyword moved.
_ENET_USES_ALPHAS_INT = _sklearn_at_least(1, 7)

# scikit-learn 1.6 renamed ``force_all_finite`` to ``ensure_all_finite``. Both
# spellings mean the same thing; "allow-nan" admits NaN and still rejects
# infinity, which is what SXE wants -- NaN is missing, infinity is a number
# the binning cannot place.
_FINITE_KW = "ensure_all_finite" if _sklearn_at_least(1, 6) else "force_all_finite"


def _enet_path_size(n_alphas):
    key = "alphas" if _ENET_USES_ALPHAS_INT else "n_alphas"
    return {key: int(n_alphas)}


def _fit_platt(scores, y):
    """Platt scaling of a 1-D score against 0/1 targets: (a, b) or None.

    Fits ``p = sigmoid(a * score + b)`` by maximum likelihood against Platt's
    smoothed targets -- (N+ + 1)/(N+ + 2) and 1/(N- + 2) rather than 1 and 0 --
    which keeps the optimum finite when the fold happens to be separable. The
    smoothing is applied by giving each observation both labels with weights
    ``t`` and ``1 - t``, so an ordinary logistic solver optimises the smoothed
    objective exactly.

    Returns None when the fold cannot identify a calibration (one class only,
    a constant score, or a fit that came out non-increasing) -- the caller then
    falls back to clipping rather than shipping a map that reorders the scores.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y = (np.asarray(y).ravel() > 0.5).astype(np.float64)
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0 or not np.isfinite(scores).all():
        return None
    if np.ptp(scores) <= 0:
        return None

    t = np.where(y > 0.5, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    Xd = np.repeat(scores, 2).reshape(-1, 1)
    yd = np.tile([1, 0], len(scores))
    wd = np.empty(2 * len(scores))
    wd[0::2], wd[1::2] = t, 1.0 - t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # C is large because the smoothed targets, not a penalty, are what
        # keep the optimum finite; this is maximum likelihood, not ridge.
        lr = LogisticRegression(C=1e10, max_iter=2000).fit(Xd, yd, sample_weight=wd)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0.0:
        return None
    return a, b


def _apply_platt(ab, scores):
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(scores, dtype=np.float64) + b)))


def _make_rng(random_state):
    """Generator from an sklearn-style random_state.

    Integers and None go straight to ``default_rng`` so that a given seed
    reproduces the reference implementation exactly.
    """
    if random_state is None or isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(random_state)
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state.randint(0, 2**32 - 1))
    return np.random.default_rng(random_state)


class _BaseSXE(BaseEstimator):
    """Shared implementation for the regressor and the classifier.

    Parameters
    ----------
    learning_rate : float, default=0.07
        Shrinkage applied to each batch-average correction (nu).
    max_iter : int, default=300
        Maximum number of boosting batches.
    batch_size : int, default=16
        Trees grown in parallel per batch. batch_size=1 recovers ordinary
        stochastic gradient boosting; a single large batch recovers a forest.
    n_iter_no_change : int, default=15
        Batches without validation improvement before early stopping.
    tol : float, default=1e-4
        Early-stopping tolerance, in units of the validation target variance.
    validation_fraction : float, default=0.18
        Fraction of the training data held out for early stopping and the
        weight fit. Note this fold is used for both.
    max_bins : int, default=255
        Quantile bins per feature for non-missing values; at most 255. Missing
        values occupy a further, reserved bin, as in scikit-learn's
        ``HistGradientBoosting*``.
    max_depth_choices : tuple of int, default=(6, 8, 10, 12)
        Per-tree max depth, sampled uniformly. 0 means unlimited (capped 64).
    max_leaf_nodes : int or None, default=None
        If set, grow leaf-wise with this leaf budget instead of depth-wise.
    min_samples_leaf_range : tuple of int, default=(10, 60)
        Per-tree min_samples_leaf, sampled log-uniformly.
    max_features_range : tuple of float, default=(0.5, 1.0)
        Per-tree candidate-feature fraction, sampled uniformly.
    subsample_range : tuple of float, default=(0.4, 0.75)
        Per-tree row subsample rate. The complement is that tree's pruning
        holdout, so rates near 1.0 leave little to prune against.
    prune_strength_range : tuple of float, default=(1e-3, 1.0)
        Per-tree holdout pruning penalty lambda, sampled log-uniformly.
    leaf_shrinkage : float or None, default=None
        Holdout-fitted step size per leaf. After pruning, each leaf's training
        mean is multiplied by the factor that minimises that tree's holdout
        error, ``c = (sum_h(r)/mu + leaf_shrinkage) / (n_h + leaf_shrinkage)``
        clipped to [0, 1] -- so a leaf the holdout contradicts is damped and
        one it confirms is kept whole. The value is the strength of the pull
        toward 1 (in holdout rows); 0 is the raw holdout fit. None disables.
    hist_bits : int, default=0
        0 grows on exact residual histograms. A value in [2, 15] rounds each
        row's residual stochastically to that many bits per tree and scores
        splits from integer histograms (quantized training, Shi et al. 2022);
        leaf values are still the exact means. Integer sums are cheaper to
        build and order-independent. Experimental: measure before relying on
        it.
    splitter : {"best", "random"}, default="best"
        "best" scans all bins for the optimal cut per candidate feature;
        "random" draws ``n_thresholds`` random cut-points instead, trading
        split quality for member diversity.
    n_thresholds : int, default=2
        Random cut-points per candidate feature when splitter="random".
    bundle_binary_features : bool, default=False
        Enable exclusive feature bundling for binary columns.
    multiclass : {"vector", "ovr"}, default="vector"
        Classifier only; ignored for binary targets. "vector" grows one
        ensemble with K-vector leaves (structure shared across classes);
        "ovr" fits one binary ensemble per class.
    calibrate : {"sigmoid", None}, default="sigmoid"
        Classifier only; ignored by ``SXERegressor``. The ensemble fits
        squared error against 0/1 targets, so its raw output is a regression
        score that routinely falls outside [0, 1] and is not a probability.
        "sigmoid" fits Platt scaling on the validation fold, turning it into
        one. Ranking is untouched -- the map is strictly increasing -- so AUC
        is unchanged while log-loss and Brier score improve substantially.
        None restores the previous behaviour of clipping the raw score into
        [0, 1], which is cruder *and* perturbs AUC by collapsing both tails
        into ties.
    hybrid_min : int, default=600
        Node size above which the histogram-subtraction grower is used.
        Both growers recurse smaller-child-first and pick splits by the same
        rule, so this selects an implementation, not a model: the fitted trees
        are unchanged. They do sum histograms differently (subtraction versus
        direct accumulation), so equality is to floating-point rounding rather
        than by construction.
    n_alphas : int, default=20
        Alphas in the weight-fit regularisation path.
    weight_cv : int, default=3
        Folds for choosing the weight-fit alpha.
    l1_ratio : float or list of float, default=1.0
        Elastic-net mixing for the weight fit. 1.0 is a pure lasso.
    weight_refit : {"relaxed", None}, default="relaxed"
        "relaxed" fits a relaxed lasso (Meinshausen, 2007): the penalty
        chooses which batches survive, and a non-negative least-squares refit
        on those batches undoes the shrinkage the penalty put on their
        weights. The penalty strength and the blend between the two solutions
        are chosen together by cross-validation on the validation fold, so a
        stronger penalty -- fewer batches, a smaller model -- can win when the
        refit repairs what it cost. None keeps the plain lasso solution.
        Measured across the benchmark suite, "relaxed" improved every
        regression task and most classification tasks at equal or smaller
        size, which is why it is the default.
    n_jobs : int or None, default=None
        Threads used for growth and prediction. ``None`` and ``-1`` both mean
        all available cores. Note this differs from the scikit-learn default
        of one core, and matches other gradient-boosting libraries.
    random_state : int, RandomState, Generator or None
        Seeds binning, the validation split, and per-tree randomisation.
        Results are independent of ``n_jobs`` and reproduce bit-for-bit across
        runs on one machine. They are *not* portable across platforms: the C++
        standard library's random distributions are implementation-defined, so
        libstdc++, libc++ and MSVC grow different trees from the same seed.
    verbose : int, default=0
        If non-zero, print a one-line fit summary.
    """

    def __init__(
        self,
        *,
        learning_rate=0.07,
        max_iter=300,
        batch_size=16,
        n_iter_no_change=15,
        tol=1e-4,
        validation_fraction=0.18,
        max_bins=255,
        max_depth_choices=(6, 8, 10, 12),
        max_leaf_nodes=None,
        min_samples_leaf_range=(10, 60),
        max_features_range=(0.5, 1.0),
        subsample_range=(0.4, 0.75),
        prune_strength_range=(1e-3, 1.0),
        leaf_shrinkage=None,
        hist_bits=0,
        splitter="best",
        n_thresholds=2,
        bundle_binary_features=False,
        calibrate="sigmoid",
        multiclass="vector",
        hybrid_min=600,
        n_alphas=20,
        weight_cv=3,
        l1_ratio=1.0,
        weight_refit="relaxed",
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.max_bins = max_bins
        self.max_depth_choices = max_depth_choices
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features_range = max_features_range
        self.subsample_range = subsample_range
        self.prune_strength_range = prune_strength_range
        self.leaf_shrinkage = leaf_shrinkage
        self.hist_bits = hist_bits
        self.splitter = splitter
        self.n_thresholds = n_thresholds
        self.bundle_binary_features = bundle_binary_features
        self.calibrate = calibrate
        self.multiclass = multiclass
        self.hybrid_min = hybrid_min
        self.n_alphas = n_alphas
        self.weight_cv = weight_cv
        self.l1_ratio = l1_ratio
        self.weight_refit = weight_refit
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    # -- parameter checking --------------------------------------------------
    def _check_params(self):
        def _pair(name, lo_min=None, ordered=True):
            v = getattr(self, name)
            try:
                lo, hi = (float(v[0]), float(v[1]))
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(f"{name} must be a pair (low, high), got {v!r}") from exc
            if ordered and lo > hi:
                raise ValueError(f"{name} must satisfy low <= high, got {v!r}")
            if lo_min is not None and lo < lo_min:
                raise ValueError(f"{name} lower bound must be >= {lo_min}, got {v!r}")
            return lo, hi

        if self.calibrate not in (None, "sigmoid"):
            raise ValueError(f"calibrate must be 'sigmoid' or None, got {self.calibrate!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.n_iter_no_change < 1:
            raise ValueError(f"n_iter_no_change must be >= 1, got {self.n_iter_no_change}")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {self.validation_fraction}"
            )
        if not 2 <= self.max_bins <= MAX_BINS:
            raise ValueError(f"max_bins must be in [2, {MAX_BINS}], got {self.max_bins}")
        if self.splitter not in ("best", "random"):
            raise ValueError(f"splitter must be 'best' or 'random', got {self.splitter!r}")
        if self.n_thresholds < 1:
            raise ValueError(f"n_thresholds must be >= 1, got {self.n_thresholds}")
        if self.max_leaf_nodes is not None and self.max_leaf_nodes < 2:
            raise ValueError(f"max_leaf_nodes must be >= 2 or None, got {self.max_leaf_nodes}")
        if self.hybrid_min < 1:
            raise ValueError(f"hybrid_min must be >= 1, got {self.hybrid_min}")
        if not (self.hist_bits == 0 or 2 <= int(self.hist_bits) <= 15):
            raise ValueError(f"hist_bits must be 0 or in [2, 15], got {self.hist_bits}")
        if self.weight_refit not in ("relaxed", None):
            raise ValueError(f"weight_refit must be 'relaxed' or None, got {self.weight_refit!r}")
        if self.leaf_shrinkage is not None and not self.leaf_shrinkage >= 0:
            raise ValueError(f"leaf_shrinkage must be >= 0 or None, got {self.leaf_shrinkage}")

        depths = list(self.max_depth_choices)
        if not depths:
            raise ValueError("max_depth_choices must be non-empty")
        if any(int(d) < 0 for d in depths):
            raise ValueError(f"max_depth_choices entries must be >= 0, got {depths}")

        leaf = _pair("min_samples_leaf_range", lo_min=1)
        feat = _pair("max_features_range", lo_min=0.0)
        sub = _pair("subsample_range", lo_min=0.0)
        lam = _pair("prune_strength_range")
        if lam[0] <= 0:
            raise ValueError(
                f"prune_strength_range lower bound must be > 0 (it is sampled "
                f"log-uniformly), got {self.prune_strength_range!r}"
            )
        if not 0.0 < sub[1] <= 1.0:
            raise ValueError(f"subsample_range must lie in (0, 1], got {self.subsample_range!r}")
        if not 0.0 < feat[1] <= 1.0:
            raise ValueError(
                f"max_features_range must lie in (0, 1], got {self.max_features_range!r}"
            )
        return depths, leaf, feat, sub, lam

    def _n_jobs(self):
        return -1 if self.n_jobs is None else self.n_jobs

    # -- weight fit ----------------------------------------------------------
    def _fit_weights(self, P, y_val):
        """Non-negative (positive) lasso over the batch-average columns.

        Selection happens on held-out data on purpose: fitting the weights
        in-sample lets the deepest members absorb everything.
        """
        l1 = self.l1_ratio
        l1_list = [float(l1)] if np.isscalar(l1) else [float(v) for v in l1]
        n_val = P.shape[0]
        cv = int(min(self.weight_cv, n_val))

        if self.weight_refit == "relaxed" and cv >= 2 and n_val >= 4 and len(l1_list) == 1:
            return self._fit_weights_relaxed(P, y_val, l1_list[0], cv)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cv >= 2 and n_val >= 4:
                enet = ElasticNetCV(
                    l1_ratio=l1_list,
                    positive=True,
                    precompute=True,
                    cv=cv,
                    max_iter=4000,
                    random_state=0,
                    **_enet_path_size(self.n_alphas),
                )
                enet.fit(P, y_val)
                w, b, alpha = enet.coef_.copy(), float(enet.intercept_), float(enet.alpha_)
            else:
                # too few validation rows to cross-validate; barely regularise
                en = ElasticNet(alpha=1e-6, l1_ratio=1.0, positive=True, max_iter=6000)
                en.fit(P, y_val)
                w, b, alpha = en.coef_.copy(), float(en.intercept_), 1e-6
            if not np.any(w > 1e-10):
                # the CV solution collapsed to the intercept; back off hard so
                # the model is still an ensemble rather than a constant
                en = ElasticNet(alpha=alpha * 1e-3, l1_ratio=1.0, positive=True, max_iter=6000)
                en.fit(P, y_val)
                w, b = en.coef_.copy(), float(en.intercept_)
        return w, b

    @staticmethod
    def _nnls_refit(P, y, support):
        """Non-negative least squares with a free intercept on the given columns."""
        w = np.zeros(P.shape[1])
        if support.size == 0:
            return w, float(y.mean())
        Ps = P[:, support]
        pm, ym = Ps.mean(axis=0), float(y.mean())
        coef, _ = nnls(Ps - pm, y - ym)
        w[support] = coef
        return w, ym - float(pm @ coef)

    def _fit_weights_relaxed(self, P, y_val, l1, cv):
        """Relaxed positive lasso: choose (alpha, blend) by CV over the fold.

        For every alpha on the path the lasso picks a support; the refit is
        NNLS on that support; the candidate solutions are the blends
        ``phi * lasso + (1 - phi) * refit``. The pair minimising CV error is
        refitted on the whole fold.
        """
        phis = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        n_alphas = int(self.n_alphas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # enet_path works on centred data (no intercept); the alpha grid
            # it derives from the whole fold is reused on every CV split.
            alphas, _, _ = enet_path(
                P - P.mean(axis=0), y_val - y_val.mean(), l1_ratio=l1,
                n_alphas=n_alphas, positive=True,
            )
            folds = list(KFold(cv, shuffle=False).split(P))
            err = np.zeros((len(alphas), len(phis)))
            for tr, te in folds:
                Ptr, ytr = P[tr], y_val[tr]
                pm, ym = Ptr.mean(axis=0), ytr.mean()
                _, coefs, _ = enet_path(
                    Ptr - pm, ytr - ym, l1_ratio=l1, alphas=alphas, positive=True
                )
                for a in range(len(alphas)):
                    w_l = coefs[:, a]
                    b_l = ym - pm @ w_l
                    w_r, b_r = self._nnls_refit(Ptr, ytr, np.flatnonzero(w_l > 1e-10))
                    for k, phi in enumerate(phis):
                        w = phi * w_l + (1.0 - phi) * w_r
                        b = phi * b_l + (1.0 - phi) * b_r
                        res = y_val[te] - (P[te] @ w + b)
                        err[a, k] += float(res @ res)
            a_best, k_best = np.unravel_index(np.argmin(err), err.shape)
            alpha, phi = float(alphas[a_best]), float(phis[k_best])
            en = ElasticNet(alpha=alpha, l1_ratio=l1, positive=True, max_iter=6000).fit(P, y_val)
            w_l, b_l = en.coef_.copy(), float(en.intercept_)
            w_r, b_r = self._nnls_refit(P, y_val, np.flatnonzero(w_l > 1e-10))
            w = phi * w_l + (1.0 - phi) * w_r
            b = phi * b_l + (1.0 - phi) * b_r
            if not np.any(w > 1e-10):
                en = ElasticNet(alpha=alpha * 1e-3, l1_ratio=1.0, positive=True, max_iter=6000)
                en.fit(P, y_val)
                w, b = en.coef_.copy(), float(en.intercept_)
        return w, b

    # -- fit -----------------------------------------------------------------
    def _fit(self, X, y):
        t0 = time.perf_counter()
        depths, leaf, feat, sub, lam = self._check_params()
        rng = _make_rng(self.random_state)
        n_jobs = self._n_jobs()

        if self.bundle_binary_features:
            self.bundler_ = BinaryBundler().fit(X)
            X = self.bundler_.transform(X)
        else:
            self.bundler_ = None
        n_model_features = X.shape[1]

        self.binner_ = QuantileBinner(self.max_bins, random_state=self.random_state).fit(X)
        Xb = self.binner_.transform(X)

        n = len(y)
        if n < 2:
            raise ValueError(
                f"SXE holds out a validation fold for early stopping and the weight "
                f"fit, so it needs at least 2 samples, got n_samples={n}"
            )
        perm = rng.permutation(n)
        n_val = int(round(self.validation_fraction * n))
        n_val = max(1, min(n_val, n - 1))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xtr = np.asfortranarray(Xb[tr_idx])
        Xval = np.asfortranarray(Xb[val_idx])
        ytr = np.ascontiguousarray(y[tr_idx])
        y_val = np.ascontiguousarray(y[val_idx])

        batch = int(self.batch_size)
        # a (n, K) target early-stops on the MSE summed over its K columns
        var_val = float(np.var(y_val, axis=0).sum()) if y_val.ndim == 2 else float(np.var(y_val))
        ens = _core.Ensemble()
        t_grow = time.perf_counter()
        n_members, n_batches = ens.boost(
            X=Xtr,
            y=ytr,
            X_val=Xval,
            y_val=y_val,
            batch=batch,
            max_batches=int(self.max_iter),
            shrink=float(self.learning_rate),
            patience=int(self.n_iter_no_change),
            tol=float(self.tol) * var_val,
            n_thresh=int(self.n_thresholds),
            argmax_bins=1 if self.splitter == "best" else 0,
            max_leaves=0 if self.max_leaf_nodes is None else int(self.max_leaf_nodes),
            sub_lo=sub[0],
            sub_hi=sub[1],
            depth_choices=[int(d) for d in depths],
            leaf_lo=int(leaf[0]),
            leaf_hi=int(leaf[1]),
            feat_lo=feat[0],
            feat_hi=feat[1],
            lam_lo=lam[0],
            lam_hi=lam[1],
            leaf_shrink=-1.0 if self.leaf_shrinkage is None else float(self.leaf_shrinkage),
            hist_bits=int(self.hist_bits),
            hybrid_min=int(self.hybrid_min),
            seed=int(rng.integers(2**63 - 1)),
            n_jobs=n_jobs,
        )
        self.grow_time_ = time.perf_counter() - t_grow

        # One column per batch rather than per tree: the weight fit's cost
        # becomes independent of the number of trees, and it can drop whole
        # steps of the boosting trajectory.
        t_w = time.perf_counter()
        # For a K-column target the design matrix stacks the K outputs of
        # every validation row -- one weight per batch, shared by all K.
        Pb = ens.predict_batch_means(Xval, batch, n_jobs)
        w_batch, intercept = self._fit_weights(Pb, np.ascontiguousarray(y_val).ravel())
        sel_b = np.flatnonzero(w_batch > 1e-10)
        if sel_b.size == 0:  # degenerate: keep the first batch at zero weight
            sel_b = np.array([0], dtype=np.int64)
            w_batch = np.zeros(Pb.shape[1])
        sel = (sel_b[:, None] * batch + np.arange(batch)[None, :]).ravel()
        weights = np.repeat(w_batch[sel_b] / batch, batch)
        self.weight_fit_time_ = time.perf_counter() - t_w

        # Physically drop the members that carry no weight. After this the
        # model object holds exactly the trees that contribute.
        self.ensemble_ = ens.subset(np.ascontiguousarray(sel, dtype=np.int32))
        # Round leaves to the 16-bit codes the blob stores, before anything
        # (the calibrator, the reported size) sees them: the model in memory
        # is the model on disk.
        self.ensemble_.quantize_leaves()
        self._members = np.arange(self.ensemble_.n_members, dtype=np.int32)
        self.weights_ = np.ascontiguousarray(weights, dtype=np.float64)
        self.intercept_ = float(intercept)
        self.calibrator_ = self._fit_calibrator(Xval, y_val, n_jobs)

        nodes, leaves = self.ensemble_.structure(self._members)
        self.n_iter_ = int(n_batches)
        self.n_trees_ = int(n_members)
        self.n_trees_kept_ = int(self.ensemble_.n_members)
        self.n_leaves_ = int(leaves)
        self.n_nodes_ = int(nodes)

        # Accumulated once, here, rather than on every feature_importances_
        # call: the members and weights are fixed from now on, and caching the
        # p-vector is what lets the serialized trees drop their split gains.
        self.feature_gains_ = self.ensemble_.feature_gains(
            self._members, self.weights_, n_model_features
        )
        self.used_features_ = np.asarray(
            self.ensemble_.used_features(self._members, n_model_features), dtype=bool
        )
        # Compaction is deliberately not done here. restrict() can only remove
        # edges, so a one-vs-rest classifier whose sub-models compacted
        # individually could never build the shared table its union needs --
        # every public fit() calls _compact once, with the right mask.
        self.fit_time_ = time.perf_counter() - t0

        return self

    def _report(self):
        if self.verbose:
            print(
                f"SXE: {self.n_iter_} batches, kept {self.n_trees_kept_}/{self.n_trees_} trees, "
                f"{self.n_leaves_} leaves, {self.model_size_bytes_ / 1e6:.2f} MB, "
                f"{self.fit_time_:.2f}s"
            )
        return self

    def _compact(self, used):
        """Drop dead bin edges and record what the fitted model actually costs.

        ``model_size_bytes_`` is the artefact: the bytes ``serialize()`` writes
        plus the quantile edges that have to travel with it. Nothing here is an
        estimate of a format SXE does not implement.
        """
        subs = getattr(self, "estimators_", None)
        trees = subs if subs else [self]
        # Ship only the edges behind cuts some tree makes, and renumber the
        # trees' thresholds to the compacted table. The union over sub-models
        # matters: they share one table.
        cuts = np.logical_or.reduce(
            [
                np.asarray(m.ensemble_.used_thresholds(m._members, len(used)), dtype=bool)
                for m in trees
            ]
        )
        cuts[~used] = False
        table = self.binner_.restrict_thresholds(cuts)
        for m in trees:
            m.ensemble_.remap_thresholds(table)
        self.tree_bytes_ = int(sum(m.ensemble_.serialized_size() for m in trees))
        self.bin_table_bytes_ = int(sum(e.nbytes for e in self.binner_.edges_))
        self.model_size_bytes_ = self.tree_bytes_ + self.bin_table_bytes_

    def _fit_calibrator(self, Xval, y_val, n_jobs):
        """Regression output is already on the target's scale."""
        return None

    # -- prediction ----------------------------------------------------------
    def _score_validated(self, X):
        """Raw ensemble output for an already-validated float64 matrix."""
        if self.bundler_ is not None:
            X = self.bundler_.transform(X)
        Xb = np.asfortranarray(self.binner_.transform(X))
        return self.ensemble_.predict_weighted(
            Xb, self._members, self.weights_, self.intercept_, self._n_jobs()
        )

    def _raw_predict(self, X):
        check_is_fitted(self)
        X = _validate(self, X, reset=False, dtype=np.float64, ensure_2d=True)
        return self._score_validated(X)

    # -- introspection -------------------------------------------------------
    @property
    def feature_importances_(self):
        """Split-gain importances over the input columns, summing to one.

        Weighted by each member's ensemble weight. With
        ``bundle_binary_features=True`` the trees split on bundled columns, so
        each bundle's gain is divided equally among its members -- a threshold
        on a bundle separates a group of them, not one. Treat those entries as
        a per-bundle figure spread over its columns rather than a per-column
        measurement.
        """
        check_is_fitted(self)
        g = self.feature_gains_
        if self.bundler_ is not None:
            g = self.bundler_.unbundle(g)
        total = g.sum()
        return g / total if total > 0 else g

    def summary(self) -> dict:
        """Fit diagnostics: trees grown and kept, structure, size, timings.

        ``model_size_mb`` is everything that has to ship to make a prediction,
        measured rather than estimated: the bytes ``serialize()`` writes for
        the trees plus the quantile edges of the columns they index, broken out
        as ``tree_size_mb`` and ``bin_table_mb``. A pickle of this object is
        somewhat larger -- it carries the estimator's parameters and Python
        object overhead on top.
        """
        check_is_fitted(self)
        return {
            "n_iter": self.n_iter_,
            "n_trees_grown": self.n_trees_,
            "n_trees_kept": self.n_trees_kept_,
            "n_leaves": self.n_leaves_,
            "model_size_mb": round(self.model_size_bytes_ / 1e6, 3),
            "tree_size_mb": round(self.tree_bytes_ / 1e6, 3),
            "bin_table_mb": round(self.bin_table_bytes_ / 1e6, 3),
            "grow_time_s": round(self.grow_time_, 3),
            "weight_fit_time_s": round(self.weight_fit_time_, 3),
            "fit_time_s": round(self.fit_time_, 3),
        }

    # -- tags ----------------------------------------------------------------
    def __sklearn_tags__(self):  # scikit-learn >= 1.6
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = False
        return tags

    def _more_tags(self):  # pragma: no cover - scikit-learn < 1.6
        return {"allow_nan": True, "requires_y": True}


class SXERegressor(RegressorMixin, _BaseSXE):
    """Sparse eXtra Ensemble for regression.

    Trees are generated by batch boosting -- small batches of randomized,
    subsampled trees grown in parallel against the current residual, added
    with shrinkage and early-stopped on a validation fold. Each tree is
    pruned on its own out-of-subsample holdout at a randomized strength. A
    single non-negative lasso then re-weights the boosting trajectory and
    deletes the batches that contribute nothing.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sxe import SXERegressor
    >>> X, y = make_regression(n_samples=500, n_features=8, random_state=0)
    >>> model = SXERegressor(random_state=0).fit(X, y)
    >>> preds = model.predict(X)
    """

    def fit(self, X, y):
        X, y = _validate(self, X, y, dtype=np.float64, y_numeric=True, ensure_2d=True)
        y = np.asarray(y, dtype=np.float64)
        self._fit(X, y)
        self._compact(self.used_features_)
        return self._report()

    def predict(self, X):
        return self._raw_predict(X)


class _CalibratesBinary:
    """Turns one binary sub-model's regression score into a probability.

    Mixed into both the public classifier's own binary path and the
    one-vs-rest sub-models, so a class score means the same thing either way.
    """

    def _fit_calibrator(self, Xval, y_val, n_jobs):
        if self.calibrate is None:
            return None
        # The fold is the one already carved for early stopping and the weight
        # fit, so the calibration is mildly optimistic on it. Carving a fourth,
        # dedicated fold costs more accuracy (less data to grow on) than the
        # reuse costs calibration -- measured, not assumed.
        s = self.ensemble_.predict_weighted(
            Xval, self._members, self.weights_, self.intercept_, n_jobs
        )
        if s.ndim == 2:
            # one sigmoid per class column; all or none, so the columns stay
            # on one scale before they are normalised to a distribution
            cals = [_fit_platt(s[:, k], y_val[:, k]) for k in range(s.shape[1])]
            return cals if all(c is not None for c in cals) else None
        return _fit_platt(s, y_val)

    def _probability(self, scores):
        """Map raw ensemble scores onto [0, 1], column by column if 2-D."""
        cal = getattr(self, "calibrator_", None)
        if cal is not None:
            if scores.ndim == 2:
                return np.column_stack([_apply_platt(c, scores[:, k]) for k, c in enumerate(cal)])
            return _apply_platt(cal, scores)
        # Uncalibrated fallback. Clipping is crude: it collapses both tails
        # into ties, which perturbs ranking metrics as well as log-loss.
        return np.clip(scores, 1e-6, 1 - 1e-6)


class _BinaryCore(_CalibratesBinary, _BaseSXE):
    """One binary sub-model of a one-vs-rest classifier.

    Carries the same parameters as the public estimators but is never
    exposed: validation and class bookkeeping happen in SXEClassifier.
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self._fit(X, np.asarray(y, dtype=np.float64))
        # SXEClassifier drives the multiclass path through _fit and compacts
        # once over the union; this is here so a sub-model fitted on its own is
        # not left without a size or a pruned bin table.
        self._compact(self.used_features_)
        return self


class SXEClassifier(ClassifierMixin, _CalibratesBinary, _BaseSXE):
    """Sparse eXtra Ensemble for classification.

    Binary targets are handled as regression on {0, 1} (the Brier-score /
    stacked-classifier reduction): leaf means are probability estimates, and
    MSE splits on 0/1 targets are equivalent to Gini up to constants.

    Targets with more than two classes are fitted by default as one ensemble
    whose trees carry a K-vector leaf (``multiclass="vector"``): every split
    is scored on the gain summed over the K one-hot residual columns, so the
    tree structure is shared across classes and only the leaves are per
    class. ``multiclass="ovr"`` keeps the older one-vs-rest reduction -- one
    full binary ensemble per class -- which costs K times the structure.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sxe import SXEClassifier
    >>> X, y = make_classification(n_samples=500, n_features=8, random_state=0)
    >>> model = SXEClassifier(random_state=0).fit(X, y)
    >>> proba = model.predict_proba(X)[:, 1]
    """

    def fit(self, X, y):
        X, y = _validate(self, X, y, dtype=np.float64, ensure_2d=True)
        check_classification_targets(y)
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError(f"SXEClassifier needs at least 2 classes, got 1 class: {classes[0]!r}")

        if len(classes) == 2:
            # cleared first: _compact sums over estimators_ when there are any,
            # and refitting a multiclass model on binary data would otherwise
            # size the new model from the previous fit's sub-models
            self.estimators_ = None
            self._fit(X, (y == classes[1]).astype(np.float64))
            self._compact(self.used_features_)
        elif self.multiclass == "vector":
            self.estimators_ = None
            Y = np.ascontiguousarray((y[:, None] == classes[None, :]).astype(np.float64))
            self._fit(X, Y)
            self._compact(self.used_features_)
        elif self.multiclass == "ovr":
            params = self.get_params()
            estimators = []
            for c in classes:
                sub = _BinaryCore(**params)
                sub._fit(X, (y == c).astype(np.float64))
                sub.n_features_in_ = X.shape[1]
                estimators.append(sub)
            self.estimators_ = estimators
            self._aggregate_multiclass()
        else:
            raise ValueError(f"multiclass must be 'vector' or 'ovr', got {self.multiclass!r}")
        self.classes_ = classes
        return self._report()

    def _aggregate_multiclass(self):
        """Roll the per-class sub-models up into the usual fitted attributes."""
        subs = self.estimators_
        self.n_iter_ = max(s.n_iter_ for s in subs)
        self.n_trees_ = sum(s.n_trees_ for s in subs)
        self.n_trees_kept_ = sum(s.n_trees_kept_ for s in subs)
        self.n_leaves_ = sum(s.n_leaves_ for s in subs)
        self.n_nodes_ = sum(s.n_nodes_ for s in subs)
        self.feature_gains_ = np.mean([s.feature_gains_ for s in subs], axis=0)
        self.bundler_ = subs[0].bundler_

        # Every sub-model binned the same X with the same seed, so their edges
        # are identical and one table can serve all of them -- but it has to
        # carry every column ANY of them splits on, which is why no sub-model
        # compacted its own copy first: restrict() only removes.
        self.used_features_ = np.logical_or.reduce([s.used_features_ for s in subs])
        self.binner_ = subs[0].binner_
        for sub in subs:
            sub.binner_ = self.binner_
        self._compact(self.used_features_)

        # All classes calibrated or none. A rare class can leave its validation
        # fold with no positives, and Platt scaling has nothing to fit; mixing
        # a sigmoid-mapped column with a raw clipped one and then normalising
        # the row would put the classes on different scales.
        self.calibrator_ = None
        if any(s.calibrator_ is None for s in subs):
            for sub in subs:
                sub.calibrator_ = None
        self.grow_time_ = sum(s.grow_time_ for s in subs)
        self.weight_fit_time_ = sum(s.weight_fit_time_ for s in subs)
        self.fit_time_ = sum(s.fit_time_ for s in subs)

    def _class_scores(self, X):
        check_is_fitted(self)
        X = _validate(self, X, reset=False, dtype=np.float64, ensure_2d=True)
        if self.estimators_ is None:
            p = self._probability(self._score_validated(X))
            if p.ndim == 2:  # vector-leaf multiclass: K calibrated columns
                total = p.sum(axis=1, keepdims=True)
                return np.divide(
                    p, total, out=np.full_like(p, 1.0 / p.shape[1]), where=total > 0
                )
            return np.column_stack([1.0 - p, p])
        # one-vs-rest: each sub-model maps its own score onto [0, 1] with its
        # own calibration, then the columns are normalised to a distribution
        scores = np.column_stack([s._probability(s._score_validated(X)) for s in self.estimators_])
        total = scores.sum(axis=1, keepdims=True)
        return np.divide(
            scores, total, out=np.full_like(scores, 1.0 / scores.shape[1]), where=total > 0
        )

    def predict_proba(self, X):
        return self._class_scores(X)

    def predict(self, X):
        # Thresholding the probability rather than the raw score keeps predict
        # and predict_proba consistent: calibration moves the raw score that
        # corresponds to p = 0.5 away from 0.5.
        p = self._class_scores(X)
        if len(self.classes_) == 2:
            # threshold rather than argmax, so p == 0.5 resolves to the
            # positive class as it does elsewhere in the binary reduction
            return self.classes_[(p[:, 1] >= 0.5).astype(int)]
        return self.classes_[np.argmax(p, axis=1)]

    @property
    def feature_importances_(self):
        check_is_fitted(self)
        if self.estimators_ is None:
            return super().feature_importances_
        imp = np.mean([s.feature_importances_ for s in self.estimators_], axis=0)
        total = imp.sum()
        return imp / total if total > 0 else imp
