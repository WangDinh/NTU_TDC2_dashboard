import torch
from xgboost import XGBRegressor

_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class XGBoostModel:
    """XGBoost — expects flattened (n, lookback*n_feat) input."""

    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=6):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            device=_device,
            random_state=42,
        )

    def fit(self, X_flat, y, xgb_model=None):
        """xgb_model: an existing Booster to continue boosting from (used for
        rack-by-rack fine-tuning — see pipeline._train_rack_by_rack)."""
        self.model.fit(X_flat, y, xgb_model=xgb_model)

    def predict_step(self, window_3d):
        """window_3d: (1, lookback, n_feat) → (n_feat,) predicted next-step vector."""
        return self.model.predict(window_3d.reshape(1, -1))[0]

    def predict_batch(self, windows_3d):
        """windows_3d: (n, lookback, n_feat) → (n, n_feat)."""
        n = windows_3d.shape[0]
        return self.model.predict(windows_3d.reshape(n, -1))
