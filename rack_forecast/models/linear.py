import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    from cuml.linear_model import Ridge
    _backend = 'cuml'
except ImportError:
    from sklearn.linear_model import Ridge
    _backend = 'sklearn'


class LinearModel:
    """Ridge regression — expects flattened (n, lookback*n_feat) input."""

    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self._backend = _backend

    def fit(self, X_flat, y):
        self.model.fit(X_flat, y)

    def predict_step(self, window_3d):
        """window_3d: (1, lookback, n_feat) → (n_feat,) predicted next-step vector."""
        return self.model.predict(window_3d.reshape(1, -1))[0]

    def predict_batch(self, windows_3d):
        """windows_3d: (n, lookback, n_feat) → (n, n_feat)."""
        n = windows_3d.shape[0]
        return self.model.predict(windows_3d.reshape(n, -1))
