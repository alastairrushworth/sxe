"""Bad input must produce a clear Python exception, never a crash or a
silently wrong answer."""

import pickle

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
    assert model.ensemble_.n_out == 3
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


def test_leaf_shrinkage_and_weight_refit_are_validated():
    from sklearn.datasets import make_regression

    from sxe import SXERegressor

    X, y = make_regression(n_samples=300, n_features=6, random_state=0)
    with pytest.raises(ValueError, match="leaf_shrinkage"):
        SXERegressor(leaf_shrinkage=-1.0).fit(X, y)
    with pytest.raises(ValueError, match="weight_refit"):
        SXERegressor(weight_refit="nnls").fit(X, y)


def test_leaf_shrinkage_never_amplifies_a_leaf():
    """Every leaf is the training mean times a factor in [0, 1], so the
    shrunk ensemble's member predictions are bounded by the unshrunk one's."""
    from sklearn.datasets import make_regression

    from sxe import SXERegressor

    X, y = make_regression(n_samples=1500, n_features=6, noise=5.0, random_state=0)
    # one batch: the trees are grown on the same residual either way, so
    # they differ only in their leaf values
    cfg = dict(random_state=0, max_iter=1, batch_size=8)
    raw = SXERegressor(**cfg).fit(X, y)
    shrunk = SXERegressor(leaf_shrinkage=0.0, **cfg).fit(X, y)
    Xb = np.asfortranarray(raw.binner_.transform(X))
    Xs = np.asfortranarray(shrunk.binner_.transform(X))
    members = np.arange(raw.ensemble_.n_members, dtype=np.int32)
    a = np.abs(raw.ensemble_.predict_members(Xb, members))
    b = np.abs(shrunk.ensemble_.predict_members(Xs, members))
    # tolerance: one 16-bit leaf-code step on this target's range
    step = np.ptp(y) / 65535.0
    assert (b <= a + step).all()
    assert (b < a - step).any()


def test_structural_defaults_follow_the_training_size():
    from sklearn.datasets import make_regression

    from sxe import SXERegressor

    X, y = make_regression(n_samples=1200, n_features=6, random_state=0)
    m = SXERegressor(random_state=0, max_iter=3, batch_size=4).fit(X, y)
    assert m.max_depth_choices_ == (6, 8, 10, 12)
    assert m.min_samples_leaf_range_ == (10, 60)
    assert m.max_iter_ == 3  # an explicit max_iter is kept
    small, mid, big = (SXERegressor._structural_defaults(n) for n in (10_000, 50_000, 200_000))
    assert small[0] == (6, 8, 10, 12) and mid[0] == (8, 10, 12, 14) and big[0] == (10, 12, 14)
    assert small[1][0] > mid[1][0] > big[1][0]
    assert small[2] == mid[2] == 600 and big[2] == 300


def test_holdout_modes_and_new_parameters_are_validated():
    from sklearn.datasets import make_regression

    from sxe import SXERegressor

    X, y = make_regression(n_samples=400, n_features=6, random_state=0)
    with pytest.raises(ValueError, match="holdout"):
        SXERegressor(holdout="oob").fit(X, y)
    with pytest.raises(ValueError, match="holdout_folds"):
        SXERegressor(holdout_folds=1).fit(X, y)
    with pytest.raises(ValueError, match="weight_tol"):
        SXERegressor(weight_tol=-1.0).fit(X, y)
    with pytest.raises(ValueError, match="linear_leaves"):
        SXERegressor(linear_leaves=300).fit(X, y)
    with pytest.raises(ValueError, match="linear_lambda"):
        SXERegressor(linear_leaves=2, linear_lambda=0.0).fit(X, y)
    with pytest.raises(ValueError, match="loss"):
        SXEClassifier(loss="hinge").fit(X, (y > 0).astype(int))
    # folds beyond the batch width are clamped, not rejected
    SXERegressor(random_state=0, max_iter=2, batch_size=4, holdout_folds=16).fit(X, y)


def test_linear_leaves_round_trip_and_change_predictions():
    from sklearn.datasets import make_regression

    from sxe import SXERegressor

    X, y = make_regression(n_samples=1500, n_features=5, noise=1.0, random_state=0)
    cfg = dict(random_state=0, max_iter=4, batch_size=4, max_depth_choices=(3,))
    const = SXERegressor(**cfg).fit(X, y)
    lin = SXERegressor(linear_leaves=3, **cfg).fit(X, y)
    assert not np.array_equal(const.predict(X), lin.predict(X))
    restored = pickle.loads(pickle.dumps(lin))
    assert np.array_equal(lin.predict(X), restored.predict(X))
    assert lin.tree_bytes_ == len(lin.ensemble_.serialize())
    # a linear leaf reads its features' bin ids, so their edges must ship
    assert lin.bin_table_bytes_ > 0
