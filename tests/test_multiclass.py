"""Multiclass targets are fitted one-vs-rest over the binary core.

The C++ core only knows the {0, 1} MSE reduction, so more than two classes
means one binary ensemble per class with the scores normalised afterwards.
These tests pin the contract that reduction has to honour.
"""

import pickle

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from sxe import SXEClassifier

CFG = dict(max_iter=6, batch_size=4, max_depth_choices=(4,), min_samples_leaf_range=(5, 20))
OVR = dict(multiclass="ovr", **CFG)


@pytest.fixture(scope="module")
def multiclass_data():
    return make_classification(
        n_samples=900,
        n_features=10,
        n_informative=6,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=0,
    )


def test_fits_and_predicts(multiclass_data):
    X, y = multiclass_data
    model = SXEClassifier(random_state=0, **CFG).fit(X, y)
    assert list(model.classes_) == [0, 1, 2, 3]
    assert model.estimators_ is None  # one ensemble, K-vector leaves
    assert model.ensemble_.n_out == 4
    pred = model.predict(X)
    assert set(np.unique(pred)) <= set(model.classes_)
    assert model.score(X, y) > 0.5


def test_proba_is_a_distribution(multiclass_data):
    X, y = multiclass_data
    model = SXEClassifier(random_state=0, **CFG).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 4)
    assert (proba >= 0).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_agrees_with_proba(multiclass_data):
    X, y = multiclass_data
    model = SXEClassifier(random_state=0, **CFG).fit(X, y)
    expected = model.classes_[np.argmax(model.predict_proba(X), axis=1)]
    assert np.array_equal(model.predict(X), expected)


def test_string_labels(multiclass_data):
    X, y = multiclass_data
    labels = np.array(["ant", "bee", "cat", "dog"])[y]
    model = SXEClassifier(random_state=0, **CFG).fit(X, labels)
    assert list(model.classes_) == ["ant", "bee", "cat", "dog"]
    assert set(np.unique(model.predict(X))) <= set(model.classes_)


def test_binary_path_still_uses_one_model():
    X, y = make_classification(n_samples=600, n_features=8, random_state=0)
    model = SXEClassifier(random_state=0, **CFG).fit(X, y)
    assert model.estimators_ is None


def test_deterministic(multiclass_data):
    X, y = multiclass_data
    a = SXEClassifier(random_state=0, n_jobs=1, **CFG).fit(X, y).predict_proba(X)
    b = SXEClassifier(random_state=0, n_jobs=4, **CFG).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)


def test_pickle_roundtrip(multiclass_data):
    X, y = multiclass_data
    model = SXEClassifier(random_state=0, **CFG).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(model.predict_proba(X), restored.predict_proba(X))


def test_one_vs_rest_is_still_available(multiclass_data):
    X, y = multiclass_data
    model = SXEClassifier(random_state=0, **OVR).fit(X, y)
    assert len(model.estimators_) == 4
    assert model.score(X, y) > 0.5
    p = model.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_vector_leaves_share_structure(multiclass_data):
    """K-vector leaves: the same structure serves every class, so the model
    carries far fewer nodes than one ensemble per class."""
    X, y = multiclass_data
    vec = SXEClassifier(random_state=0, **CFG).fit(X, y)
    ovr = SXEClassifier(random_state=0, **OVR).fit(X, y)
    assert vec.n_nodes_ < ovr.n_nodes_
    assert vec.score(X, y) > 0.5


def test_summary_aggregates_over_classes(multiclass_data):
    X, y = multiclass_data
    model = SXEClassifier(random_state=0, **OVR).fit(X, y)
    s = model.summary()
    assert s["n_trees_kept"] == sum(m.n_trees_kept_ for m in model.estimators_)
    assert s["n_leaves"] == sum(m.n_leaves_ for m in model.estimators_)


def test_feature_importances_averaged(multiclass_data):
    X, y = multiclass_data
    model = SXEClassifier(random_state=0, **CFG).fit(X, y)
    imp = model.feature_importances_
    assert imp.shape == (X.shape[1],)
    assert imp.sum() == pytest.approx(1.0)


def test_cross_val_score(multiclass_data):
    X, y = multiclass_data
    scores = cross_val_score(SXEClassifier(random_state=0, **CFG), X, y, cv=3)
    assert np.isfinite(scores).all()


def test_refit_from_multiclass_to_binary_clears_state(multiclass_data):
    """estimators_ from a previous fit must not survive into the next one."""
    X, y = multiclass_data
    model = SXEClassifier(random_state=0, **OVR).fit(X, y)
    assert model.estimators_ is not None
    multi_size = model.model_size_bytes_

    model.fit(X, (y == y[0]).astype(int))
    assert model.estimators_ is None
    assert len(model.classes_) == 2
    assert model.model_size_bytes_ < multi_size
    assert model.tree_bytes_ == len(model.ensemble_.serialize())
