"""Global quantile binning.

Every downstream stage -- growth, holdout routing, pruning, member
prediction, the weight-fit design matrix -- operates on the single uint8
matrix produced here. A uniform random threshold drawn by an extremely
randomized split on binned data is exactly a random bin cut-point, so
binning and ExtraTrees-style splitting compose without approximation.

Missing values get the top bin. Finite values occupy 0..254 and NaN occupies
255, which needs no support at all in the tree code: a split routes bin 255
like any other, so missing always travels with the largest values *unless* a
split at 254 separates it, and the exact-cut scan will find that split when it
pays. Nothing is imputed and nothing is guessed.

Edges are float32, rounded *before* the training data is binned, so the table
that ships is the table the trees were grown against. After a fit the table is
compacted to the edges some tree actually cuts at (``restrict_thresholds``);
the trees' bin thresholds are renumbered to match.
"""

from __future__ import annotations

import numpy as np

__all__ = ["QuantileBinner"]

# Bins for non-finite values, matching scikit-learn's HistGradientBoosting
# vocabulary: ``max_bins`` counts the bins available to real values, and the
# 256th is reserved for missing.
MAX_BINS = 255
MISSING_BIN = 255
N_THRESHOLDS = 256  # a split threshold is any uint8 bin id


class QuantileBinner:
    """Bin each feature into at most ``n_bins`` quantile bins.

    Edges are estimated from a subsample of at most ``subsample`` rows.
    ``transform`` returns ``uint8`` bin ids directly -- the C++ core consumes
    that dtype, so there is no intermediate float matrix.

    Parameters
    ----------
    n_bins : int, default=255
        Bins per feature for non-missing values; at most 255. Missing values
        occupy a further, reserved bin.
    subsample : int, default=50_000
        Rows sampled to estimate the quantile edges.
    random_state : int, RandomState, Generator or None
        Controls the subsample used to estimate edges.
    """

    def __init__(self, n_bins: int = MAX_BINS, subsample: int = 50_000, random_state=None):
        self.n_bins = n_bins
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X) -> QuantileBinner:
        if not 2 <= self.n_bins <= MAX_BINS:
            raise ValueError(f"n_bins must be in [2, {MAX_BINS}], got {self.n_bins}")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        if n > self.subsample:
            X = X[rng.choice(n, self.subsample, replace=False)]
        # n_bins - 1 interior points give n_bins bins numbered 0..n_bins-1, so
        # with n_bins <= 255 the largest is 254 and MISSING_BIN stays free.
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self.edges_ = [self._column_edges(X[:, j], qs) for j in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        return self

    @staticmethod
    def _column_edges(col, qs):
        col = col[np.isfinite(col)]
        if col.size == 0:  # an all-missing column: every row lands in MISSING_BIN
            return np.empty(0, dtype=np.float32)
        # Rounded to float32 here, not at save time: binning the training data
        # with edges the model does not ship would make the trees on disk
        # subtly different from the trees that were grown.
        return np.unique(np.quantile(col, qs).astype(np.float32))

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "edges_"):
            raise ValueError("QuantileBinner is not fitted yet; call fit() first")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but QuantileBinner was fitted with "
                f"{self.n_features_in_} features"
            )
        out = np.empty(X.shape, dtype=np.uint8)
        for j, e in enumerate(self.edges_):
            col = X[:, j]
            # searchsorted yields bin ids in [0, len(e)] and len(e) <= 254, so
            # MISSING_BIN cannot collide with a finite bin. NaN sorts above
            # every edge, which would silently put it in the top finite bin --
            # hence the explicit overwrite rather than relying on the sort.
            b = np.searchsorted(e, col, side="right").astype(np.uint8)
            missing = np.isnan(col)
            if missing.any():
                b[missing] = MISSING_BIN
            out[:, j] = b
        return out

    def restrict(self, used) -> QuantileBinner:
        """Drop the edges of features nothing downstream will index.

        A fitted ensemble usually splits on a fraction of the columns it was
        offered, but the edges of the rest ship with the model anyway -- on
        wide data that is most of the model. Dropping them leaves an empty
        edge array, and ``searchsorted`` on empty returns 0, so those columns
        transform to bin 0 instead of their true bin. That is safe precisely
        because nothing reads them; call this only with a mask taken from the
        consumer that will use the result.
        """
        if not hasattr(self, "edges_"):
            raise ValueError("QuantileBinner is not fitted yet; call fit() first")
        used = np.asarray(used, dtype=bool)
        if used.shape != (self.n_features_in_,):
            raise ValueError(
                f"used must be a boolean mask over {self.n_features_in_} features, "
                f"got shape {used.shape}"
            )
        empty = np.empty(0, dtype=np.float32)
        self.edges_ = [e if u else empty for e, u in zip(self.edges_, used)]
        return self

    def restrict_thresholds(self, used) -> np.ndarray:
        """Keep only the edges behind cuts a model uses; return the bin remap.

        ``used`` is a boolean ``(n_features, 256)`` mask: ``used[f, t]`` says
        some tree splits feature ``f`` at ``bin <= t``. Because ``transform``
        assigns ``bin = #(edges <= x)``, that cut is ``x < edges[t]`` and needs
        exactly one edge. A cut at ``t >= len(edges)`` separates missing (bin
        255) from every finite value and needs none.

        Returns a ``(n_features, 256)`` uint8 table mapping each old threshold
        to its id under the compacted edges, for the trees to apply. Columns
        with no used cut keep no edges and transform to bin 0, which is safe
        for the same reason as in ``restrict``: nothing reads them.
        """
        if not hasattr(self, "edges_"):
            raise ValueError("QuantileBinner is not fitted yet; call fit() first")
        used = np.asarray(used, dtype=bool)
        if used.shape != (self.n_features_in_, N_THRESHOLDS):
            raise ValueError(
                f"used must be a boolean mask of shape {(self.n_features_in_, N_THRESHOLDS)}, "
                f"got shape {used.shape}"
            )
        table = np.zeros((self.n_features_in_, N_THRESHOLDS), dtype=np.uint8)
        new_edges = []
        for f, e in enumerate(self.edges_):
            k = len(e)
            keep = np.flatnonzero(used[f, :k])
            new_edges.append(e[keep])
            # old threshold t < k -> its rank among the kept edges; t >= k -> the
            # count of kept edges (finite-vs-missing cut). Unused t < k also get
            # a consistent id, though no tree will ask for one.
            ranks = np.searchsorted(keep, np.arange(k), side="left")
            table[f, :k] = ranks
            table[f, k:] = len(keep)
        self.edges_ = new_edges
        return table

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
