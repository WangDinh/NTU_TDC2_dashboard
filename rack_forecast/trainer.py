"""Model training: the PyTorch loop and the name→trained-model dispatcher.

GPU is used automatically when available (DL models). The sklearn/XGBoost
models train on whatever backend their wrapper resolves to (see rack_forecast/
models/*.py).
"""

from .models.transformer import TransformerModel
from .models.cnn1d import CNN1DModel
from .models.lstm import LSTMModel
from .models import REGISTRY
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# Single global device — DL models train here, then move back to CPU for saving.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[rack_forecast] device: {DEVICE}')

# Names that map to PyTorch models (everything else is sklearn/XGBoost).
DL_MODELS = {'lstm', 'cnn1d', 'transformer'}


# ── PyTorch training loop ──────────────────────────────────────────────────

def train_dl(model, X_3d, y, epochs=30, lr=1e-3, batch=256, val_frac=0.1, patience=5):
    """Train a PyTorch model with early stopping; returns the best-val-loss model.

    The last `val_frac` of the (time-ordered) data is held out as validation.
    A tqdm bar reports live val_loss. The returned model is on CPU.

    Data stays on CPU; only one mini-batch at a time is moved to `DEVICE`, so
    memory use doesn't scale with the full dataset size (train OR validation).
    """
    model = model.to(DEVICE)

    # Time-ordered train/val split (no shuffle — this is a time series).
    split = int(len(X_3d) * (1 - val_frac))
    Xt = torch.tensor(X_3d[:split])
    yt = torch.tensor(y[:split])
    Xv = torch.tensor(X_3d[split:])
    yv = torch.tensor(y[split:])

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=False)
    val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=batch, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val, wait, best_state = float('inf'), 0, None

    pbar = tqdm(range(epochs), desc='  epochs', unit='ep', leave=False)
    for ep in pbar:
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # Validation loss drives early stopping — computed in mini-batches too.
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss_sum += loss_fn(model(xb), yb).item() * len(xb)
                val_n += len(xb)
        val_loss = val_loss_sum / val_n
        pbar.set_postfix(val_loss=f'{val_loss:.5f}', best=f'{best_val:.5f}')

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                pbar.write(
                    f'    early stop at epoch {ep + 1}  val_loss={val_loss:.5f}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model.cpu()


# ── dispatcher ─────────────────────────────────────────────────────────────

def build_and_train(name, X_3d, y, lookback, n_feat, dl_epochs=30):
    """Train model `name` and return the fitted object.

    Every returned model exposes .predict_step(window) and .predict_batch(windows)
    working in SCALED space (see rack_forecast/models/*.py).
    """
    if name in DL_MODELS:
        arch = {
            'lstm': lambda: LSTMModel(n_feat),
            'cnn1d': lambda: CNN1DModel(n_feat, lookback),
            'transformer': lambda: TransformerModel(n_feat),
        }[name]()
        return train_dl(arch, X_3d, y, epochs=dl_epochs)

    if name in REGISTRY:
        # sklearn / XGBoost models want flattened (n, lookback*n_feat) input.
        m = REGISTRY[name]()
        backend = getattr(m, '_backend', 'cuda' if name ==
                          'xgboost' else 'cpu')
        print(f'    backend: {backend}')
        m.fit(X_3d.reshape(len(X_3d), -1), y)
        return m

    raise ValueError(f'Unknown model name: {name!r}')
