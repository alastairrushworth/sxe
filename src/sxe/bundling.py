"""Exclusive Feature Bundling (EFB) restricted to binary columns.

LightGBM-style bundling: mutually exclusive 0/1 features (one-hot blocks,
indicator matrices) are merged into single small-integer columns -- 0 when no
member is active, k when the k-th bundled feature is active. Conflict
tolerance is zero, so the transform is lossless up to the ordinal encoding of
the bundle: a threshold split on a bundle selects a contiguous group of
categories rather than one-vs-rest, but trees compose splits.

On Covertype this compresses 54 features to 12 and cuts histogram work ~4.5x.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BinaryBundler"]

MAX_BUNDLE = 250


class BinaryBundler:
    """Bundle mutually exclusive binary columns into small-integer columns."""

    def fit(self, X) -> BinaryBundler:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        n, p = X.shape
        self.n_features_in_ = p

        # Detect binary columns one at a time rather than casting the whole
        # matrix: on full Covertype a float64 copy of X is ~250 MB.
        is_bin = []
        nz_cols = []
        for j in range(p):
            col = np.asarray(X[:, j], dtype=np.float64)
            nz = col != 0.0
            if np.all(nz | (col == 0.0)) and np.all(col[nz] == 1.0):
                is_bin.append(j)
                nz_cols.append(nz)

        binset = set(is_bin)
        self.passthrough_ = [j for j in range(p) if j not in binset]

        bundles: list[tuple[list[int], np.ndarray]] = []
        if is_bin:
            nz = np.column_stack(nz_cols)
            order = np.argsort(-nz.sum(axis=0))
            for k in order:
                col = nz[:, k]
                placed = False
                for cols, used in bundles:
                    if len(cols) < MAX_BUNDLE and not np.any(used & col):
                        cols.append(is_bin[k])
                        used |= col
                        placed = True
                        break
                if not placed:
                    bundles.append(([is_bin[k]], col.copy()))

        self.bundles_ = [cols for cols, _ in bundles if len(cols) >= 2]
        bundled = {c for cols in self.bundles_ for c in cols}
        self.passthrough_ += [j for j in is_bin if j not in bundled]
        self.passthrough_.sort()
        self.n_features_out_ = len(self.passthrough_) + len(self.bundles_)
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "bundles_"):
            raise ValueError("BinaryBundler is not fitted yet; call fit() first")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but BinaryBundler was fitted with "
                f"{self.n_features_in_} features"
            )
        out = np.empty((X.shape[0], self.n_features_out_), dtype=np.float64)
        k = len(self.passthrough_)
        if k:
            out[:, :k] = np.asarray(X[:, self.passthrough_], dtype=np.float64)
        for i, cols in enumerate(self.bundles_):
            raw = np.asarray(X[:, cols], dtype=np.float64)
            # A bundled column was strictly 0/1 during fit but can still arrive
            # missing later. NaN is not an active member -- it is an unknown
            # one -- so the bundle as a whole becomes missing and the binner
            # sends it to its own bin.
            unknown = np.isnan(raw)
            sub = (raw != 0.0) & ~unknown

            # Exclusivity is verified on the training data and assumed here.
            # Silently encoding a conflicting row as its first active member
            # would make it indistinguishable from a row where only that
            # member is set, so check rather than assume: it is one extra pass
            # over a boolean array that was needed anyway.
            active = sub.sum(axis=1)
            bad = np.flatnonzero(active > 1)
            if bad.size:
                r = int(bad[0])
                members = [cols[j] for j in np.flatnonzero(sub[r])]
                raise ValueError(
                    f"row {r} sets columns {members} at once, but they were "
                    f"mutually exclusive in the training data and were bundled "
                    f"together. Refit with bundle_binary_features=False, or drop "
                    f"the conflicting rows."
                )
            col = np.where(active > 0, sub.argmax(axis=1) + 1, 0).astype(np.float64)
            col[unknown.any(axis=1)] = np.nan
            out[:, k + i] = col
        return out

    def unbundle(self, values):
        """Spread per-bundled-column quantities back over the input columns.

        A threshold split on a bundle separates a contiguous group of its
        members from the rest, so the split's gain belongs to the group rather
        than to any one member. Dividing it equally is the least arbitrary
        choice available without tracking which members each threshold cut.
        """
        out = np.zeros(self.n_features_in_, dtype=np.float64)
        for i, j in enumerate(self.passthrough_):
            out[j] = values[i]
        k = len(self.passthrough_)
        for i, cols in enumerate(self.bundles_):
            out[list(cols)] += values[k + i] / len(cols)
        return out

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
