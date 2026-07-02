import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    from cuml.ensemble import RandomForestRegressor
    _backend = 'cuml'
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    _backend = 'sklearn'


class RFModel:
    """Random Forest — expects flattened (n, lookback*n_feat) input."""

    def __init__(self, n_estimators=100, random_state=42):
        kwargs = {'n_estimators': n_estimators, 'random_state': random_state}
        if _backend == 'sklearn':
            kwargs['n_jobs'] = -1
        self.model = RandomForestRegressor(**kwargs)
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
