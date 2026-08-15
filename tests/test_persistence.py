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
