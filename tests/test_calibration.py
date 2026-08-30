"""predict_proba must return probabilities, not clipped regression scores.

The ensemble fits squared error against 0/1 targets, so its raw output is a
score that routinely falls outside [0, 1] -- on the benchmark tasks, for a
quarter to two fifths of rows. Platt scaling on the validation fold turns it
into a probability without touching the ranking.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from sxe import SXEClassifier

CFG = dict(max_iter=40, batch_size=8, random_state=0)


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=6000, n_features=20, n_informative=8, weights=[0.9, 0.1], random_state=0
    )
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


def test_raw_scores_really_do_leave_the_unit_interval(data):
    """The premise of the whole fix. If this stops holding, revisit it."""
    Xtr, Xte, ytr, _ = data
    m = SXEClassifier(**CFG).fit(Xtr, ytr)
    raw = m._score_validated(np.asarray(Xte, dtype=np.float64))
    assert ((raw < 0) | (raw > 1)).mean() > 0.05


def test_calibration_is_on_by_default(data):
    Xtr, _, ytr, _ = data
    assert SXEClassifier(**CFG).fit(Xtr, ytr).calibrator_ is not None


def test_proba_is_a_distribution(data):
    Xtr, Xte, ytr, _ = data
    for n_classes, (X, y) in (
        (2, (Xtr, ytr)),
        (
            3,
            make_classification(
                n_samples=1500,
                n_features=12,
                n_informative=6,
                n_classes=3,
                n_clusters_per_class=1,
                random_state=0,
            ),
        ),
    ):
        m = SXEClassifier(**CFG).fit(X, y)
        p = m.predict_proba(Xte if n_classes == 2 else X)
        assert p.shape[1] == n_classes
        assert (p >= 0).all() and (p <= 1).all()
        assert np.allclose(p.sum(axis=1), 1.0)


def test_calibration_improves_log_loss_and_brier(data):
    Xtr, Xte, ytr, yte = data
    on = SXEClassifier(calibrate="sigmoid", **CFG).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    off = SXEClassifier(calibrate=None, **CFG).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    assert log_loss(yte, on) < log_loss(yte, off)
    assert brier_score_loss(yte, on) < brier_score_loss(yte, off)


def test_calibration_preserves_ranking_exactly(data):
    """Platt is strictly increasing, so AUC is the raw model's, to the bit.

    The clipping fallback is not: it collapses both tails into ties, which
    moves AUC in whichever direction the ties happen to fall.
    """
    Xtr, Xte, ytr, yte = data
    m = SXEClassifier(calibrate="sigmoid", **CFG).fit(Xtr, ytr)
    raw = m._score_validated(np.asarray(Xte, dtype=np.float64))
    assert roc_auc_score(yte, m.predict_proba(Xte)[:, 1]) == roc_auc_score(yte, raw)


def test_predict_agrees_with_predict_proba(data):
    """Calibration moves the raw score at which p crosses 0.5 away from 0.5."""
    Xtr, Xte, ytr, _ = data
    for calibrate in ("sigmoid", None):
        m = SXEClassifier(calibrate=calibrate, **CFG).fit(Xtr, ytr)
        expected = m.classes_[(m.predict_proba(Xte)[:, 1] >= 0.5).astype(int)]
        assert np.array_equal(m.predict(Xte), expected)


def test_calibrate_none_restores_the_clip(data):
    Xtr, Xte, ytr, _ = data
    m = SXEClassifier(calibrate=None, **CFG).fit(Xtr, ytr)
    assert m.calibrator_ is None
    raw = m._score_validated(np.asarray(Xte, dtype=np.float64))
    assert np.array_equal(m.predict_proba(Xte)[:, 1], np.clip(raw, 1e-6, 1 - 1e-6))


def test_unknown_calibrate_value_is_rejected(data):
    Xtr, _, ytr, _ = data
    with pytest.raises(ValueError, match="calibrate"):
        SXEClassifier(calibrate="isotonic", **CFG).fit(Xtr, ytr)


def test_calibrator_survives_a_round_trip(data):
    import pickle

    Xtr, Xte, ytr, _ = data
    m = SXEClassifier(**CFG).fit(Xtr, ytr)
    back = pickle.loads(pickle.dumps(m))
    assert back.calibrator_ == m.calibrator_
    assert np.array_equal(back.predict_proba(Xte), m.predict_proba(Xte))


def test_degenerate_fold_falls_back_rather_than_reordering():
    """A fold that cannot identify a calibration must not produce a bad one."""
    from sxe.estimator import _fit_platt

    s = np.linspace(0, 1, 50)
    assert _fit_platt(s, np.ones(50)) is None  # one class
    assert _fit_platt(np.zeros(50), np.r_[np.zeros(25), np.ones(25)]) is None  # constant score
    a, _ = _fit_platt(s, (s > 0.5).astype(float))
    assert a > 0  # and a usable fold gives an increasing map


def test_multiclass_calibration_is_all_or_nothing():
    """A rare class can leave its fold with no positives and nothing to fit.

    Mixing a sigmoid-mapped column with a raw clipped one puts the classes on
    different scales before the row is normalised, so the model calibrates
    every class or none of them.
    """
    X, y = make_classification(
        n_samples=900,
        n_features=8,
        n_informative=5,
        n_classes=4,
        n_clusters_per_class=1,
        weights=[0.94, 0.03, 0.02, 0.01],
        random_state=3,
    )
    m = SXEClassifier(random_state=0, max_iter=10, batch_size=4, multiclass="ovr").fit(X, y)
    fitted = [s.calibrator_ is not None for s in m.estimators_]
    assert all(fitted) or not any(fitted)
    p = m.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)

    # the vector-leaf path calibrates its K columns on the same rule
    v = SXEClassifier(random_state=0, max_iter=10, batch_size=4).fit(X, y)
    assert v.calibrator_ is None or len(v.calibrator_) == 4
    p = v.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)
