import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    from cuml.svm import SVR
    _backend = 'cuml'
except ImportError:
    from sklearn.svm import SVR
    _backend = 'sklearn'


class SVRModel:
    """Support Vector Regression — expects flattened (n, lookback*n_feat) input."""

    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVR(kernel=kernel, C=C)
        self._backend = _backend

    def fit(self, X_flat, y):
        self.model.fit(X_flat, y)

    def predict_step(self, window_3d):
        """window_3d: (1, lookback, n_feat) → scalar."""
        return float(self.model.predict(window_3d.reshape(1, -1)))

    def predict_batch(self, windows_3d):
        """windows_3d: (n, lookback, n_feat) → (n,)."""
        n = windows_3d.shape[0]
        return self.model.predict(windows_3d.reshape(n, -1))
