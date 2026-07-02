import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """TransformerEncoder → linear head. Input: (batch, lookback, n_feat)."""

    def __init__(self, n_feat, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc  = nn.Linear(d_model, n_feat)

    def forward(self, x):
        x = self.proj(x)
        x = self.enc(x)
        return self.fc(x[:, -1, :])

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
