"""A fixed seed must give bit-identical results regardless of thread count.

Per-tree RNG seeds derive from splitmix64(seed + c*tree_index), never from
scheduling order, so n_jobs is a pure performance knob. This is the property
that lets someone reproduce a result on a different machine, so it is worth
testing directly rather than trusting by construction.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from sxe import SXEClassifier, SXERegressor

CFG = dict(max_iter=8, batch_size=6, max_depth_choices=(4, 5), min_samples_leaf_range=(5, 20))


@pytest.fixture(scope="module")
def reg_data():
    return make_regression(n_samples=1200, n_features=10, noise=1.0, random_state=0)


@pytest.fixture(scope="module")
def clf_data():
    return make_classification(n_samples=1200, n_features=10, random_state=0)


@pytest.mark.parametrize("n_jobs", [1, 2, 4, -1])
def test_predictions_independent_of_n_jobs(reg_data, n_jobs):
    X, y = reg_data
    ref = SXERegressor(random_state=0, n_jobs=1, **CFG).fit(X, y).predict(X)
    got = SXERegressor(random_state=0, n_jobs=n_jobs, **CFG).fit(X, y).predict(X)
    assert np.array_equal(ref, got)


def test_structure_independent_of_n_jobs(reg_data):
    X, y = reg_data
    a = SXERegressor(random_state=0, n_jobs=1, **CFG).fit(X, y)
    b = SXERegressor(random_state=0, n_jobs=4, **CFG).fit(X, y)
    # timings legitimately differ; everything structural must not
    for key in ("n_iter", "n_trees_grown", "n_trees_kept", "n_leaves", "model_size_mb"):
        assert a.summary()[key] == b.summary()[key], key
    assert np.array_equal(a.weights_, b.weights_)
    assert a.intercept_ == b.intercept_


def test_repeated_fits_identical(reg_data):
    X, y = reg_data
    a = SXERegressor(random_state=7, **CFG).fit(X, y).predict(X)
    b = SXERegressor(random_state=7, **CFG).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_different_seeds_differ(reg_data):
    X, y = reg_data
    a = SXERegressor(random_state=0, **CFG).fit(X, y).predict(X)
    b = SXERegressor(random_state=1, **CFG).fit(X, y).predict(X)
    assert not np.array_equal(a, b)


def test_classifier_deterministic(clf_data):
    X, y = clf_data
    a = SXEClassifier(random_state=0, n_jobs=1, **CFG).fit(X, y).predict_proba(X)
    b = SXEClassifier(random_state=0, n_jobs=4, **CFG).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)


def test_random_state_none_varies(reg_data):
    X, y = reg_data
    a = SXERegressor(random_state=None, **CFG).fit(X, y).predict(X)
    b = SXERegressor(random_state=None, **CFG).fit(X, y).predict(X)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("hybrid_min", [1, 50, 600, 10_000_000])
def test_predictions_independent_of_hybrid_min(reg_data, hybrid_min):
    """hybrid_min selects a grower, and must not select a model.

    The histogram-subtraction grower and the direct-accumulation one pick
    splits by the same rule but recurse smaller-child-first and left-first
    respectively. Since both draw from one per-tree RNG stream, mismatched
    recursion orders would shift which node each draw landed on -- so this
    performance knob would quietly change the fitted trees. Both now recurse
    smaller-first. The two paths still sum histograms differently (subtraction
    versus direct accumulation), so exact equality is not guaranteed by
    construction; it holds on everything tested and a failure here means a
    near-tied split flipped on the last bits, which is worth knowing about.
    """
    X, y = reg_data
    ref = SXERegressor(random_state=0, hybrid_min=1, **CFG).fit(X, y).predict(X)
    got = SXERegressor(random_state=0, hybrid_min=hybrid_min, **CFG).fit(X, y).predict(X)
    assert np.array_equal(ref, got)


def test_hybrid_min_invariance_holds_for_random_splitter(reg_data):
    """The random splitter draws thresholds too, so it consumes more RNG."""
    X, y = reg_data
    cfg = dict(CFG, splitter="random", n_thresholds=3)
    ref = SXERegressor(random_state=0, hybrid_min=1, **cfg).fit(X, y).predict(X)
    got = SXERegressor(random_state=0, hybrid_min=10_000_000, **cfg).fit(X, y).predict(X)
    assert np.array_equal(ref, got)
