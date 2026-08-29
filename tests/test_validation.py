"""Bad input must produce a clear Python exception, never a crash or a
silently wrong answer."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from sxe import SXEClassifier, SXERegressor

CFG = dict(max_iter=5, batch_size=4, max_depth_choices=(4,), min_samples_leaf_range=(5, 20))


@pytest.fixture(scope="module")
def reg_data():
    return make_regression(n_samples=600, n_features=8, noise=1.0, random_state=0)


def test_predict_with_wrong_n_features_raises(reg_data):
    X, y = reg_data
    model = SXERegressor(random_state=0, **CFG).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(X[:, :5])
    with pytest.raises(ValueError, match="features"):
        model.predict(np.hstack([X, X[:, :2]]))


def test_predict_before_fit_raises(reg_data):
    X, _ = reg_data
    with pytest.raises(Exception, match="not fitted|NotFitted"):
        SXERegressor().predict(X)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinity_still_raises(reg_data, bad):
    """NaN means missing; infinity is a number the binning cannot place."""
    X, y = reg_data
    Xb = X.copy()
    Xb[0, 0] = bad
    with pytest.raises(ValueError, match="[Ii]nfinity|NaN|nan"):
        SXERegressor(random_state=0, **CFG).fit(Xb, y)
    model = SXERegressor(random_state=0, **CFG).fit(X, y)
    with pytest.raises(ValueError, match="[Ii]nfinity|NaN|nan"):
        model.predict(Xb)


def test_multiclass_target_is_supported():
    """Multiclass is handled one-vs-rest rather than rejected; see
    tests/test_multiclass.py for the behavioural contract."""
    X, y = make_classification(
        n_samples=600, n_features=8, n_informative=5, n_classes=3, random_state=0
    )
    model = SXEClassifier(random_state=0, **CFG).fit(X, y)
    assert len(model.estimators_) == 3
    assert set(np.unique(model.predict(X))) <= {0, 1, 2}


def test_single_class_target_raises(reg_data):
    X, _ = reg_data
    y = np.zeros(len(X))
    with pytest.raises(ValueError, match="at least 2 classes"):
        SXEClassifier(random_state=0, **CFG).fit(X, y)


def test_classifier_preserves_class_labels():
    X, y = make_classification(n_samples=600, n_features=8, random_state=0)
    labels = np.array(["no", "yes"])[y]
    model = SXEClassifier(random_state=0, **CFG).fit(X, labels)
    assert set(np.unique(model.predict(X))) <= {"no", "yes"}
    assert list(model.classes_) == ["no", "yes"]
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "params, match",
    [
        (dict(learning_rate=0), "learning_rate"),
        (dict(max_iter=0), "max_iter"),
        (dict(batch_size=0), "batch_size"),
        (dict(n_iter_no_change=0), "n_iter_no_change"),
        (dict(validation_fraction=0.0), "validation_fraction"),
        (dict(validation_fraction=1.0), "validation_fraction"),
        (dict(max_bins=1), "max_bins"),
        (dict(max_bins=512), "max_bins"),
        (dict(splitter="fastest"), "splitter"),
        (dict(n_thresholds=0), "n_thresholds"),
        (dict(max_leaf_nodes=1), "max_leaf_nodes"),
        (dict(max_depth_choices=()), "max_depth_choices"),
        (dict(min_samples_leaf_range=(0, 10)), "min_samples_leaf_range"),
        (dict(min_samples_leaf_range=(50, 10)), "min_samples_leaf_range"),
        (dict(subsample_range=(0.5, 2.0)), "subsample_range"),
        (dict(max_features_range=(0.5, 3.0)), "max_features_range"),
        (dict(prune_strength_range=(0.0, 1.0)), "prune_strength_range"),
        (dict(hybrid_min=0), "hybrid_min"),
    ],
)
def test_invalid_params_raise(reg_data, params, match):
    X, y = reg_data
    with pytest.raises(ValueError, match=match):
        SXERegressor(random_state=0, **{**CFG, **params}).fit(X, y)


def test_core_rejects_out_of_range_member_index(reg_data):
    X, y = reg_data
    model = SXERegressor(random_state=0, **CFG).fit(X, y)
    Xb = np.asfortranarray(model.binner_.transform(X[:10]))
    bad = np.array([model.ensemble_.n_members + 5], dtype=np.int32)
    with pytest.raises(Exception, match="out of range"):
        model.ensemble_.predict_members(Xb, bad, 1)


def test_core_rejects_wrong_feature_count(reg_data):
    X, y = reg_data
    model = SXERegressor(random_state=0, **CFG).fit(X, y)
    Xb = np.asfortranarray(model.binner_.transform(X[:10])[:, :4])
    with pytest.raises(Exception, match="features"):
        model.ensemble_.predict_members(Xb, np.array([0], dtype=np.int32), 1)


def test_constant_target_is_handled(reg_data):
    X, _ = reg_data
    y = np.full(len(X), 3.5)
    model = SXERegressor(random_state=0, **CFG).fit(X, y)
    assert np.allclose(model.predict(X), 3.5, atol=1e-6)


def test_small_dataset_fits(reg_data):
    X, y = make_regression(n_samples=40, n_features=4, noise=1.0, random_state=0)
    model = SXERegressor(random_state=0, max_iter=3, batch_size=2).fit(X, y)
    assert model.predict(X).shape == (40,)


def test_bundler_rejects_predict_time_conflicts():
    """A bundle assumes exclusivity; a row that breaks it must not be guessed at.

    Encoding a two-active row as its first active member makes it
    indistinguishable from a row where only that member is set, so the
    prediction would be silently wrong rather than refused.
    """
    from sxe.bundling import BinaryBundler

    train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    b = BinaryBundler().fit(train)
    assert b.bundles_ == [[0, 1, 2]]
    assert np.array_equal(b.transform(train).ravel(), [1, 2, 3, 1])
    with pytest.raises(ValueError, match="mutually exclusive"):
        b.transform(np.array([[1, 1, 0]], dtype=float))
