"""
ML Models Package
=================
Provides machine learning models for trading signal generation:

- Gradient Boosting: XGBoost, LightGBM, CatBoost
- Neural Networks: LSTM, GRU (PyTorch)
- Transformers: Time-series transformer (PyTorch)
- Reinforcement Learning: PPO, DQN (stable-baselines3)
"""

from .base_predictor import BasePredictor
from .feature_engineering import FeatureEngineer

# Conditional imports - these may not be available until dependencies installed
try:
    from .gradient_boosting import GradientBoostingPredictor
except ImportError:
    GradientBoostingPredictor = None

try:
    from .lstm_predictor import LSTMPredictor, GRUPredictor
except ImportError:
    LSTMPredictor = None
    GRUPredictor = None

try:
    from .transformer_predictor import TransformerPredictor
except ImportError:
    TransformerPredictor = None

try:
    from .rl_trader import RLTrader
    from .trading_env import TradingEnvironment
except ImportError:
    RLTrader = None
    TradingEnvironment = None


__all__ = [
    'BasePredictor',
    'FeatureEngineer',
    'GradientBoostingPredictor',
    'LSTMPredictor',
    'GRUPredictor',
    'TransformerPredictor',
    'RLTrader',
    'TradingEnvironment',
]
