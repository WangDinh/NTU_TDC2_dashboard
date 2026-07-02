from .linear import LinearModel
from .rf import RFModel
from .xgboost import XGBoostModel
from .lstm import LSTMModel
from .cnn1d import CNN1DModel
from .transformer import TransformerModel

REGISTRY = {
    'linear':      LinearModel,
    'rf':          RFModel,
    'xgboost':     XGBoostModel,
    'lstm':        LSTMModel,
    'cnn1d':       CNN1DModel,
    'transformer': TransformerModel,
}
