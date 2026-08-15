"""SXE -- Sparse eXtra Ensembles.

Batch-boosted, holdout-pruned, sparsely re-weighted histogram trees, with a
scikit-learn compatible API.

    from sxe import SXERegressor
    model = SXERegressor(random_state=0).fit(X_train, y_train)
    preds = model.predict(X_test)
    print(model.summary())
"""

from ._core import n_threads_available
from .binning import QuantileBinner
from .bundling import BinaryBundler
from .estimator import SXEClassifier, SXERegressor

__version__ = "0.1.0.dev0"

__all__ = [
    "SXEClassifier",
    "SXERegressor",
    "QuantileBinner",
    "BinaryBundler",
    "n_threads_available",
    "__version__",
]
