"""
Feature Engineering for ML Models
==================================
Creates features from OHLCV data for ML model training.
Reuses existing indicators and adds ML-specific features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    include_returns: bool = True
    include_volatility: bool = True
    include_volume: bool = True
    include_time: bool = True
    include_ta_indicators: bool = True
    return_periods: List[int] = None
    volatility_periods: List[int] = None
    normalize: bool = True
    scaler_type: str = 'standard'  # 'standard' or 'minmax'

    def __post_init__(self):
        if self.return_periods is None:
            self.return_periods = [1, 5, 10, 20]
        if self.volatility_periods is None:
            self.volatility_periods = [5, 10, 20]


class FeatureEngineer:
    """
    Feature engineering for ML models.

    Creates a comprehensive feature set from OHLCV data including:
    - Price returns over multiple periods
    - Volatility features (rolling std, ATR ratios)
    - Volume profile features
    - Time features (hour, day_of_week)
    - Technical indicators (from existing modules)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer.

        Args:
            config: Feature configuration options
        """
        self.config = config or FeatureConfig()
        self.scaler = None
        self.feature_names: List[str] = []
        self.fitted = False

    def create_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create all features from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            fit: Whether to fit the scaler (True for training, False for inference)

        Returns:
            DataFrame with all features
        """
        df = df.copy()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # Try lowercase
                col_lower = col.lower()
                if col_lower in df.columns:
                    df[col] = df[col_lower]
                else:
                    raise ValueError(f"Missing required column: {col}")

        features = pd.DataFrame(index=df.index)

        # Price returns
        if self.config.include_returns:
            features = pd.concat([features, self._create_return_features(df)], axis=1)

        # Volatility features
        if self.config.include_volatility:
            features = pd.concat([features, self._create_volatility_features(df)], axis=1)

        # Volume features
        if self.config.include_volume:
            features = pd.concat([features, self._create_volume_features(df)], axis=1)

        # Time features
        if self.config.include_time:
            features = pd.concat([features, self._create_time_features(df)], axis=1)

        # Technical indicators
        if self.config.include_ta_indicators:
            features = pd.concat([features, self._create_ta_features(df)], axis=1)

        # Drop NaN rows (from rolling calculations)
        features = features.dropna()

        # Store feature names
        self.feature_names = features.columns.tolist()

        # Normalize features
        if self.config.normalize:
            features = self._normalize_features(features, fit=fit)

        return features

    def create_target(self, df: pd.DataFrame, target_type: str = 'direction',
                      horizon: int = 1, threshold: float = 0.0) -> pd.Series:
        """
        Create target variable for supervised learning.

        Args:
            df: DataFrame with close prices
            target_type: 'direction' (classification) or 'return' (regression)
            horizon: How many candles ahead to predict
            threshold: For ternary classification, minimum move % for signal

        Returns:
            Series with target values
        """
        close = df['close'] if 'close' in df.columns else df['Close']

        if target_type == 'direction':
            # Binary: 1 if price goes up, 0 if down
            future_return = close.shift(-horizon) / close - 1

            if threshold > 0:
                # Ternary: 1 (long), -1 (short), 0 (hold)
                target = pd.Series(0, index=df.index)
                target[future_return > threshold] = 1
                target[future_return < -threshold] = -1
            else:
                # Binary: 1 (long), 0 (short)
                target = (future_return > 0).astype(int)

        elif target_type == 'return':
            # Regression: actual return
            target = close.shift(-horizon) / close - 1

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        return target

    def create_sequences(self, features: pd.DataFrame, target: pd.Series,
                        sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/Transformer training.

        Args:
            features: Feature DataFrame
            target: Target Series
            sequence_length: Number of timesteps in each sequence

        Returns:
            Tuple of (X, y) where X has shape (samples, sequence_length, features)
        """
        # Align features and target
        common_idx = features.index.intersection(target.dropna().index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]

        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features.iloc[i:i + sequence_length].values)
            y.append(target.iloc[i + sequence_length - 1])

        return np.array(X), np.array(y)

    def _create_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price return features over multiple periods."""
        features = pd.DataFrame(index=df.index)
        close = df['close']

        for period in self.config.return_periods:
            features[f'return_{period}'] = close.pct_change(period)

        # Log returns
        features['log_return_1'] = np.log(close / close.shift(1))

        # Cumulative returns
        features['cum_return_10'] = close.pct_change(10).rolling(10).sum()

        return features

    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']

        for period in self.config.volatility_periods:
            # Rolling standard deviation of returns
            features[f'volatility_{period}'] = close.pct_change().rolling(period).std()

            # Average True Range ratio
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            features[f'atr_ratio_{period}'] = atr / close

        # Bollinger Band width
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['bb_width'] = (4 * std_20) / sma_20

        # Parkinson volatility (high-low based)
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(high / low) ** 2).rolling(20).mean()
        )

        return features

    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume features."""
        features = pd.DataFrame(index=df.index)
        volume = df['volume']
        close = df['close']

        # Volume ratios
        features['volume_sma_ratio'] = volume / volume.rolling(20).mean()

        # Volume price trend
        features['vpt'] = (volume * close.pct_change()).cumsum()
        features['vpt_norm'] = features['vpt'] / features['vpt'].rolling(50).std()

        # On-balance volume change rate
        obv = (np.sign(close.diff()) * volume).cumsum()
        features['obv_roc'] = obv.pct_change(10)

        # Volume-weighted price change
        features['vwpc'] = (close.pct_change() * volume).rolling(10).sum() / volume.rolling(10).sum()

        return features

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        features = pd.DataFrame(index=df.index)

        # Convert index to datetime if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(df.index)
            except Exception:
                # If conversion fails, skip time features
                return features
        else:
            idx = df.index

        # Hour of day (normalized 0-1)
        features['hour'] = idx.hour / 24

        # Day of week (normalized 0-1)
        features['day_of_week'] = idx.dayofweek / 6

        # Is weekend
        features['is_weekend'] = (idx.dayofweek >= 5).astype(int)

        # Time since midnight (for intraday patterns)
        features['time_of_day'] = (idx.hour * 60 + idx.minute) / (24 * 60)

        return features

    def _create_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_norm'] = features['rsi'] / 100  # Normalized 0-1

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        features['macd_norm'] = features['macd'] / close  # Normalized

        # Bollinger Bands position
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['bb_position'] = (close - sma_20) / (2 * std_20)  # -1 to 1

        # Stochastic
        lowest_14 = low.rolling(14).min()
        highest_14 = high.rolling(14).max()
        features['stoch_k'] = 100 * (close - lowest_14) / (highest_14 - lowest_14)
        features['stoch_k_norm'] = features['stoch_k'] / 100

        # ADX
        plus_dm = high.diff().where(lambda x: (x > 0) & (x > -low.diff()), 0)
        minus_dm = (-low.diff()).where(lambda x: (x > 0) & (x > high.diff()), 0)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        features['adx'] = dx.rolling(14).mean()
        features['adx_norm'] = features['adx'] / 100

        # Price position in range
        features['price_position'] = (close - lowest_14) / (highest_14 - lowest_14)

        # Moving average crossover features
        sma_9 = close.rolling(9).mean()
        sma_21 = close.rolling(21).mean()
        features['ma_cross'] = (sma_9 - sma_21) / close

        # Rate of change
        features['roc_10'] = close.pct_change(10)

        # Momentum
        features['momentum_10'] = close - close.shift(10)
        features['momentum_norm'] = features['momentum_10'] / close.shift(10)

        return features

    def _normalize_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features using StandardScaler or MinMaxScaler."""
        if self.config.scaler_type == 'standard':
            scaler_class = StandardScaler
        else:
            scaler_class = MinMaxScaler

        if fit:
            self.scaler = scaler_class()
            normalized = self.scaler.fit_transform(features)
            self.fitted = True
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            normalized = self.scaler.transform(features)

        return pd.DataFrame(normalized, index=features.index, columns=features.columns)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()

    def save_scaler(self, path: str) -> None:
        """Save fitted scaler to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, path: str) -> None:
        """Load scaler from disk."""
        import pickle
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.fitted = True
