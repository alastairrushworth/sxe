"""The point of the bolt-on: the estimators must behave like scikit-learn
estimators inside scikit-learn machinery."""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from sxe import SXEClassifier, SXERegressor

CFG = dict(max_iter=5, batch_size=4, max_depth_choices=(4,), min_samples_leaf_range=(5, 20))


@pytest.fixture(scope="module")
def reg_data():
    return make_regression(n_samples=500, n_features=8, noise=1.0, random_state=0)


@pytest.fixture(scope="module")
def clf_data():
    return make_classification(n_samples=500, n_features=8, random_state=0)


def test_estimator_types():
    assert is_regressor(SXERegressor())
    assert is_classifier(SXEClassifier())


def test_get_set_params_roundtrip():
    model = SXERegressor(learning_rate=0.3, batch_size=7)
    params = model.get_params()
    assert params["learning_rate"] == 0.3
    assert params["batch_size"] == 7
    model.set_params(learning_rate=0.05)
    assert model.get_params()["learning_rate"] == 0.05
    assert clone(model).get_params() == model.get_params()


def test_repr_does_not_raise():
    assert "SXERegressor" in repr(SXERegressor())


def test_cross_val_score(reg_data):
    X, y = reg_data
    scores = cross_val_score(SXERegressor(random_state=0, **CFG), X, y, cv=3)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_cross_val_score_parallel(reg_data):
    """n_jobs>1 pickles the estimator to workers and the results back."""
    X, y = reg_data
    scores = cross_val_score(SXERegressor(random_state=0, n_jobs=1, **CFG), X, y, cv=3, n_jobs=2)
    assert np.isfinite(scores).all()


def test_pipeline(reg_data):
    X, y = reg_data
    pipe = make_pipeline(StandardScaler(), SXERegressor(random_state=0, **CFG))
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


def test_grid_search(reg_data):
    X, y = reg_data
    gs = GridSearchCV(
        SXERegressor(random_state=0, **CFG),
        {"learning_rate": [0.1, 0.3]},
        cv=2,
        n_jobs=2,
    )
    gs.fit(X, y)
    assert gs.best_params_["learning_rate"] in (0.1, 0.3)
    assert gs.predict(X).shape == (len(X),)


def test_score_methods(reg_data, clf_data):
    X, y = reg_data
    assert np.isfinite(SXERegressor(random_state=0, **CFG).fit(X, y).score(X, y))
    Xc, yc = clf_data
    assert 0.0 <= SXEClassifier(random_state=0, **CFG).fit(Xc, yc).score(Xc, yc) <= 1.0


def test_n_features_in_and_feature_names_in_(reg_data):
    pd = pytest.importorskip("pandas")
    X, y = reg_data
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model = SXERegressor(random_state=0, **CFG).fit(df, y)
    assert model.n_features_in_ == X.shape[1]
    assert list(model.feature_names_in_) == list(df.columns)
    assert model.predict(df).shape == (len(X),)


def test_feature_importances(reg_data):
    X, y = reg_data
    model = SXERegressor(random_state=0, **CFG).fit(X, y)
    imp = model.feature_importances_
    assert imp.shape == (X.shape[1],)
    assert (imp >= 0).all()
    assert imp.sum() == pytest.approx(1.0)


def test_feature_importances_unavailable_when_bundling():
    X, y = make_classification(n_samples=400, n_features=10, random_state=0)
    Xb = (X > 0).astype(float)
    model = SXEClassifier(random_state=0, bundle_binary_features=True, **CFG).fit(Xb, y)
    with pytest.raises(AttributeError, match="bundle_binary_features"):
        _ = model.feature_importances_


def test_check_estimator_regressor():
    check_estimator(SXERegressor(max_iter=3, batch_size=2, max_depth_choices=(3,)))


def test_check_estimator_classifier():
    check_estimator(SXEClassifier(max_iter=3, batch_size=2, max_depth_choices=(3,)))
