"""Guards against silent accuracy regressions.

Exact golden values would be brittle across platforms -- libm differences of
one ulp in the per-tree log-uniform draws change which trees get grown -- so
these assert generous bands that a genuine regression would still break.
All data is synthetic so the suite never needs a network.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from sxe import SXEClassifier, SXERegressor

# The configuration used for the Friedman1 benchmark.
FRIEDMAN_CFG = dict(
    learning_rate=0.3,
    batch_size=16,
    max_iter=45,
    n_iter_no_change=2,
    max_depth_choices=(4, 5, 6),
    min_samples_leaf_range=(20, 70),
    subsample_range=(0.2, 0.35),
    max_features_range=(0.35, 0.65),
    validation_fraction=0.10,
    n_alphas=12,
)


@pytest.fixture(scope="module")
def friedman():
    X, y = make_friedman1(n_samples=25_000, n_features=25, noise=1.0, random_state=0)
    return train_test_split(X, y, test_size=0.2, random_state=0)


def test_friedman_rmse(friedman):
    """Noise sigma = 1, so the achievable RMSE floor is about 1.0."""
    Xtr, Xte, ytr, yte = friedman
    model = SXERegressor(random_state=0, **FRIEDMAN_CFG).fit(Xtr, ytr)
    rmse = mean_squared_error(yte, model.predict(Xte)) ** 0.5
    assert rmse < 1.15, f"Friedman1 RMSE regressed to {rmse:.4f} (expected ~1.09)"


def test_friedman_model_stays_compact(friedman):
    """Small models are the headline claim; a blow-up is a regression."""
    Xtr, _, ytr, _ = friedman
    model = SXERegressor(random_state=0, **FRIEDMAN_CFG).fit(Xtr, ytr)
    s = model.summary()
    assert s["model_size_mb"] < 0.5, f"model grew to {s['model_size_mb']} MB (expected ~0.12)"
    assert s["n_trees_kept"] < s["n_trees_grown"], "the weight fit dropped nothing"


def test_weight_fit_prunes_the_trajectory(friedman):
    """The lasso is supposed to compress the dictionary substantially."""
    Xtr, _, ytr, _ = friedman
    model = SXERegressor(random_state=0, **FRIEDMAN_CFG).fit(Xtr, ytr)
    kept_fraction = model.n_trees_kept_ / model.n_trees_
    assert 0.05 < kept_fraction < 0.95, f"kept {kept_fraction:.1%} of members"


def test_early_stopping_triggers():
    """max_iter should not be reached when the validation loss plateaus."""
    X, y = make_friedman1(n_samples=8_000, n_features=10, noise=1.0, random_state=0)
    model = SXERegressor(random_state=0, max_iter=500, n_iter_no_change=3, batch_size=8).fit(X, y)
    assert model.n_iter_ < 500


def test_beats_the_mean_baseline():
    X, y = make_friedman1(n_samples=4_000, n_features=10, noise=1.0, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    model = SXERegressor(random_state=0).fit(Xtr, ytr)
    rmse = mean_squared_error(yte, model.predict(Xte)) ** 0.5
    baseline = mean_squared_error(yte, np.full_like(yte, ytr.mean())) ** 0.5
    assert rmse < 0.5 * baseline


def test_classifier_auc():
    X, y = make_classification(
        n_samples=10_000, n_features=20, n_informative=10, flip_y=0.02, random_state=0
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = SXEClassifier(random_state=0).fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    assert auc > 0.92, f"AUC regressed to {auc:.4f}"


def test_binary_bundling_matches_unbundled_accuracy():
    """EFB is meant to be a speed optimisation, not an accuracy trade."""
    X, y = make_classification(n_samples=6_000, n_features=30, n_informative=10, random_state=0)
    Xb = (X > 0).astype(float)  # all-binary matrix: the case EFB targets
    Xtr, Xte, ytr, yte = train_test_split(Xb, y, test_size=0.25, random_state=0, stratify=y)
    cfg = dict(random_state=0, max_iter=20, batch_size=8)
    plain = SXEClassifier(**cfg).fit(Xtr, ytr)
    bundled = SXEClassifier(bundle_binary_features=True, **cfg).fit(Xtr, ytr)
    auc_plain = roc_auc_score(yte, plain.predict_proba(Xte)[:, 1])
    auc_bundled = roc_auc_score(yte, bundled.predict_proba(Xte)[:, 1])
    assert auc_bundled > auc_plain - 0.03, (
        f"EFB cost accuracy: {auc_plain:.4f} -> {auc_bundled:.4f}"
    )


def test_leafwise_growth_runs(friedman):
    Xtr, Xte, ytr, yte = friedman
    model = SXERegressor(
        random_state=0, max_leaf_nodes=31, max_depth_choices=(0,), max_iter=30, batch_size=8
    ).fit(Xtr, ytr)
    rmse = mean_squared_error(yte, model.predict(Xte)) ** 0.5
    assert rmse < 1.5


def test_random_splitter_runs(friedman):
    Xtr, Xte, ytr, yte = friedman
    model = SXERegressor(
        random_state=0, splitter="random", n_thresholds=3, max_iter=30, batch_size=8
    ).fit(Xtr, ytr)
    rmse = mean_squared_error(yte, model.predict(Xte)) ** 0.5
    assert rmse < 1.5
