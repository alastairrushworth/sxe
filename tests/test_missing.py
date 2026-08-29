"""Missing values are a bin, not an error.

Finite values occupy bins 0-254 and NaN occupies 255, so the tree code needs
no notion of missingness at all: a split routes bin 255 like any other. That
means missing travels with the largest values by default, and a split at bin
254 -- which the exact-cut scan will find when it pays -- separates it.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import roc_auc_score

from sxe import SXEClassifier, SXERegressor
from sxe.binning import MISSING_BIN, QuantileBinner

CFG = dict(max_iter=20, batch_size=6)


@pytest.fixture(scope="module")
def data():
    return make_regression(n_samples=1500, n_features=8, noise=1.0, random_state=0)


def punch(X, rng, rate=0.15):
    X = X.copy()
    X[rng.random(X.shape) < rate] = np.nan
    return X


def test_binner_reserves_the_top_bin(data):
    X, _ = data
    b = QuantileBinner(random_state=0).fit(X)
    assert max(len(e) for e in b.edges_) <= MISSING_BIN - 1
    Xb = b.transform(punch(X, np.random.default_rng(0)))
    assert (Xb == MISSING_BIN).any()


def test_all_missing_column_is_handled(data):
    X, _ = data
    X = X.copy()
    X[:, 3] = np.nan
    b = QuantileBinner(random_state=0).fit(X)
    assert b.edges_[3].size == 0
    assert (b.transform(X)[:, 3] == MISSING_BIN).all()


def test_fit_and_predict_with_missing(data):
    X, y = data
    rng = np.random.default_rng(0)
    Xm = punch(X, rng)
    model = SXERegressor(random_state=0, **CFG).fit(Xm, y)
    pred = model.predict(Xm)
    assert np.isfinite(pred).all()
    assert model.score(Xm, y) > 0.3


def test_missing_at_predict_when_training_had_none(data):
    """A column that was complete in training must still take NaN later."""
    X, y = data
    model = SXERegressor(random_state=0, **CFG).fit(X, y)
    Xm = punch(X, np.random.default_rng(1))
    assert np.isfinite(model.predict(Xm)).all()


def test_a_split_can_isolate_missingness():
    """Where missingness carries signal, the model must be able to use it."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 4))
    informative = rng.random(n) < 0.35
    y = informative.astype(int)
    # the signal is *which rows are missing*, not the values themselves
    X[informative, 0] = np.nan
    model = SXEClassifier(random_state=0, **CFG).fit(X, y)
    assert roc_auc_score(y, model.predict_proba(X)[:, 1]) > 0.95


def test_deterministic_with_missing(data):
    X, y = data
    Xm = punch(X, np.random.default_rng(2))
    a = SXERegressor(random_state=0, n_jobs=1, **CFG).fit(Xm, y).predict(Xm)
    b = SXERegressor(random_state=0, n_jobs=4, **CFG).fit(Xm, y).predict(Xm)
    assert np.array_equal(a, b)


def test_classifier_takes_missing():
    X, y = make_classification(n_samples=1200, n_features=10, random_state=0)
    Xm = punch(X, np.random.default_rng(3))
    model = SXEClassifier(random_state=0, **CFG).fit(Xm, y)
    p = model.predict_proba(Xm)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert model.score(Xm, y) > 0.6


def test_bundling_propagates_missingness():
    """A bundled column can arrive missing even though it was 0/1 at fit.

    NaN is not an active member, so it must not be encoded as one, and two
    missing members must not look like an exclusivity violation.
    """
    from sxe.bundling import BinaryBundler

    train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    b = BinaryBundler().fit(train)
    out = b.transform(
        np.array([[1, 0, 0], [np.nan, 0, 0], [np.nan, np.nan, 0], [0, 0, 0]], dtype=float)
    ).ravel()
    assert out[0] == 1
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert out[3] == 0
    with pytest.raises(ValueError, match="mutually exclusive"):
        b.transform(np.array([[1, 1, 0]], dtype=float))


def test_bundled_model_fits_with_missing():
    rng = np.random.default_rng(0)
    n = 1200
    onehot = np.zeros((n, 6))
    onehot[np.arange(n), rng.integers(0, 6, n)] = 1.0
    X = np.column_stack([onehot, rng.normal(size=(n, 3))])
    y = onehot[:, 0] + onehot[:, 3] * 2 + rng.normal(scale=0.3, size=n)
    X[rng.random(X.shape) < 0.05] = np.nan
    model = SXERegressor(random_state=0, bundle_binary_features=True, **CFG).fit(X, y)
    assert np.isfinite(model.predict(X)).all()
