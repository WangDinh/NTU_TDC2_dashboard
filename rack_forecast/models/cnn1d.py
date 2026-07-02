import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """Two Conv1D layers → flatten → linear head. Input: (batch, lookback, n_feat)."""

    def __init__(self, n_feat, lookback, filters=64, kernel=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(filters * lookback, n_feat)

    def forward(self, x):          # x: (B, L, F)
        x = x.permute(0, 2, 1)    # (B, F, L)
        x = self.conv(x).flatten(1)
        return self.fc(x)

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
