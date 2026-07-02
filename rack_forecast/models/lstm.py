import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM → linear head. Input: (batch, lookback, n_feat)."""

    def __init__(self, n_feat, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=0.1)
        self.fc   = nn.Linear(hidden, n_feat)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def predict_step(self, window_3d):
        """window_3d: numpy (1, lookback, n_feat) → (n_feat,) predicted vector."""
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(window_3d, dtype=torch.float32)).numpy()[0]

    def predict_batch(self, windows_3d):
        """windows_3d: (n, lookback, n_feat) → (n, n_feat) numpy array."""
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(windows_3d, dtype=torch.float32)).numpy()
