"""Global quantile binning.

Every downstream stage -- growth, holdout routing, pruning, member
prediction, the weight-fit design matrix -- operates on the single uint8
matrix produced here. A uniform random threshold drawn by an extremely
randomized split on binned data is exactly a random bin cut-point, so
binning and ExtraTrees-style splitting compose without approximation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["QuantileBinner"]

MAX_BINS = 256


class QuantileBinner:
    """Bin each feature into at most ``n_bins`` quantile bins.

    Edges are estimated from a subsample of at most ``subsample`` rows.
    ``transform`` returns ``uint8`` bin ids directly -- the C++ core consumes
    that dtype, so there is no intermediate float matrix.

    Parameters
    ----------
    n_bins : int, default=256
        Maximum number of bins per feature. Must be in [2, 256].
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
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        return self

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
            # searchsorted yields bin ids in [0, len(e)]; len(e) <= n_bins - 1 <= 255
            out[:, j] = np.searchsorted(e, X[:, j], side="right")
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
        empty = np.empty(0, dtype=np.float64)
        self.edges_ = [e if u else empty for e, u in zip(self.edges_, used)]
        return self

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
