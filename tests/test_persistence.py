"""A fitted model must survive pickling -- including into a fresh process.

The reference ctypes implementation stored the trees behind a raw pointer
that pickled as a plain integer, so a round-trip silently produced a model
that aliased freed memory: predictions looked plausible in-process and the
interpreter segfaulted when the model was loaded anywhere else. These tests
exist to keep that from coming back.
"""

import pickle
import subprocess
import sys
import textwrap
from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from sxe import SXEClassifier, SXERegressor

CFG = dict(max_iter=6, batch_size=4, max_depth_choices=(4,), min_samples_leaf_range=(5, 20))


@pytest.fixture(scope="module")
def fitted_reg():
    X, y = make_regression(n_samples=800, n_features=8, noise=1.0, random_state=0)
    return SXERegressor(random_state=0, **CFG).fit(X, y), X


def test_pickle_roundtrip_predictions_identical(fitted_reg):
    model, X = fitted_reg
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(model.predict(X), restored.predict(X))


def test_pickle_preserves_structure(fitted_reg):
    model, _ = fitted_reg
    restored = pickle.loads(pickle.dumps(model))
    assert restored.summary()["n_trees_kept"] == model.summary()["n_trees_kept"]
    assert restored.summary()["n_leaves"] == model.summary()["n_leaves"]
    assert np.array_equal(restored.weights_, model.weights_)
    assert restored.intercept_ == model.intercept_
    assert restored.n_features_in_ == model.n_features_in_


def test_deepcopy_is_independent(fitted_reg):
    """A copy must own its own trees, not alias the original's."""
    model, X = fitted_reg
    clone = deepcopy(model)
    assert np.array_equal(model.predict(X), clone.predict(X))
    del clone  # must not free memory the original still uses
    assert np.isfinite(model.predict(X)).all()


def test_pickle_size_is_the_compact_model(fitted_reg):
    """The blob should be the trees, not a handle: it must scale with them."""
    model, _ = fitted_reg
    blob = pickle.dumps(model)
    assert len(blob) > model.n_leaves_ * 8, "pickle is too small to contain the trees"


def test_classifier_pickle_roundtrip():
    X, y = make_classification(n_samples=800, n_features=8, random_state=0)
    model = SXEClassifier(random_state=0, **CFG).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(model.predict_proba(X), restored.predict_proba(X))
    assert np.array_equal(restored.classes_, model.classes_)


def test_pickle_survives_a_fresh_process(tmp_path, fitted_reg):
    """The regression test that matters: load in an interpreter that never
    saw the fitting process, and check the predictions still match."""
    model, X = fitted_reg
    model_path = tmp_path / "model.pkl"
    ref_path = tmp_path / "ref.npy"
    model_path.write_bytes(pickle.dumps(model))
    np.save(ref_path, model.predict(X[:50]))
    np.save(tmp_path / "X.npy", X[:50])

    script = textwrap.dedent(
        f"""
        import pickle, numpy as np
        m = pickle.loads(open({str(model_path)!r}, "rb").read())
        X = np.load({str(tmp_path / "X.npy")!r})
        got = m.predict(X)
        ref = np.load({str(ref_path)!r})
        assert np.array_equal(got, ref), "predictions differ after reload"
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, (
        f"fresh-process load failed (returncode {proc.returncode}, "
        f"a negative value means the interpreter crashed)\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK" in proc.stdout


def test_corrupt_blob_raises_rather_than_crashing():
    from sxe._core import Ensemble

    with pytest.raises(Exception, match="magic"):
        Ensemble.deserialize(b"not a model at all")
    with pytest.raises(Exception, match="truncated|magic"):
        Ensemble.deserialize(b"SXEB")


def test_reported_size_is_the_bytes_actually_written(fitted_reg):
    """summary() must describe the artefact, not a format SXE does not write.

    The old estimate assumed 13 bytes per internal node and 8 per leaf while
    serialize() wrote 29 bytes for every node, so the advertised size was off
    by nearly 3x. model_size_bytes_ is now measured.
    """
    model, _ = fitted_reg
    blob = model.ensemble_.serialize()
    assert model.tree_bytes_ == len(blob)
    assert model.bin_table_bytes_ == sum(e.nbytes for e in model.binner_.edges_)
    assert model.model_size_bytes_ == model.tree_bytes_ + model.bin_table_bytes_


def test_blob_is_smaller_than_the_portable_layout(fitted_reg):
    """The pre-order invariant makes left children implicit, so the real blob
    beats the 13-bytes-per-internal-node layout used to compare libraries."""
    model, _ = fitted_reg
    internal = model.n_nodes_ - model.n_leaves_
    portable = internal * 13 + model.n_leaves_ * 8
    assert model.tree_bytes_ < portable


def test_importances_survive_a_round_trip(fitted_reg):
    """Split gains are dropped from the blob, so they must be cached at fit."""
    model, _ = fitted_reg
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(restored.feature_importances_, model.feature_importances_)


def test_pruned_bin_table_does_not_change_predictions(fitted_reg):
    """Compaction drops unused columns *and* unused edges, renumbering the
    trees' thresholds to match. It must be invisible to predict()."""
    from sxe.binning import QuantileBinner
    from sxe.estimator import _BaseSXE

    model, X = fitted_reg
    full = sum(e.nbytes for e in QuantileBinner(model.max_bins, random_state=0).fit(X).edges_)
    assert model.bin_table_bytes_ < full

    _, y = make_regression(n_samples=800, n_features=8, noise=1.0, random_state=0)
    raw = SXERegressor(**model.get_params())
    compact = _BaseSXE._compact
    _BaseSXE._compact = lambda self, used: None  # fit without compaction
    try:
        raw.fit(X, y)
    finally:
        _BaseSXE._compact = compact
    before = raw.predict(X)
    compact(raw, raw.used_features_)
    assert np.array_equal(before, raw.predict(X))
    assert raw.bin_table_bytes_ == sum(e.nbytes for e in raw.binner_.edges_)


def test_multiclass_shares_one_bin_table():
    """One table for all classes, covering the union of the columns they use.

    The sub-models split on different features, so a table pruned to any one
    class silently zeroes columns the others rely on. Wide data with a small
    candidate-feature fraction is what makes the sets diverge -- with p=10 they
    all use everything and the bug hides.
    """
    from sxe.binning import QuantileBinner

    X, y = make_classification(
        n_samples=1500,
        n_features=120,
        n_informative=40,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=0,
    )
    cfg = dict(
        random_state=0,
        max_iter=6,
        batch_size=4,
        max_depth_choices=(3,),
        max_features_range=(0.1, 0.2),
        multiclass="ovr",
    )
    model = SXEClassifier(**cfg).fit(X, y)
    per_class = [s.used_features_ for s in model.estimators_]
    assert len({u.sum() for u in per_class}) > 1, "classes must differ for this to test anything"

    assert all(sub.binner_ is model.binner_ for sub in model.estimators_)
    carried = np.array([e.size > 0 for e in model.binner_.edges_])
    union = np.logical_or.reduce(per_class)
    assert not (union & ~carried).any(), "a class splits on a column with no edges"
    assert model.bin_table_bytes_ == sum(e.nbytes for e in model.binner_.edges_)

    from sxe.estimator import _BaseSXE

    compact = _BaseSXE._compact
    _BaseSXE._compact = lambda self, used: None  # fit without compaction
    try:
        reference = SXEClassifier(**cfg).fit(X, y)
    finally:
        _BaseSXE._compact = compact
    assert np.array_equal(model.predict_proba(X), reference.predict_proba(X))


def test_restrict_thresholds_renumbers_cuts_consistently():
    """A cut ``bin <= t`` is ``x < edges[t]``; after compaction the same edge
    sits at a new index and the remap table must point at it. Cuts at or
    beyond the last edge separate missing from finite and need no edge."""
    from sxe.binning import MISSING_BIN, QuantileBinner

    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 2))
    X[rng.random(2000) < 0.1, 1] = np.nan
    b = QuantileBinner(64, random_state=0).fit(X)
    k0, k1 = len(b.edges_[0]), len(b.edges_[1])
    old = b.transform(X)

    used = np.zeros((2, 256), dtype=bool)
    used[0, [3, 17, k0 - 1]] = True
    used[1, [5, k1]] = True  # k1: the finite-vs-missing cut
    table = b.restrict_thresholds(used)
    new = b.transform(X)

    for f, ts in ((0, [3, 17, k0 - 1]), (1, [5, k1])):
        for t in ts:
            assert np.array_equal(old[:, f] <= t, new[:, f] <= table[f, t])
    assert len(b.edges_[0]) == 3 and len(b.edges_[1]) == 1
    assert (new[np.isnan(X[:, 1]), 1] == MISSING_BIN).all()


def test_leaf_codes_are_applied_in_memory(fitted_reg):
    """Every leaf must be exactly representable by the 16-bit code the blob
    stores, so pickling changes nothing -- and the rounding is tiny."""
    model, X = fitted_reg
    from sxe._core import Ensemble

    restored = Ensemble.deserialize(model.ensemble_.serialize())
    members = np.arange(model.ensemble_.n_members, dtype=np.int32)
    a = model.ensemble_.predict_members(np.asfortranarray(model.binner_.transform(X)), members)
    b = restored.predict_members(np.asfortranarray(model.binner_.transform(X)), members)
    assert np.array_equal(a, b)
