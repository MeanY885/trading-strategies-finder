"""
Multi-Engine Indicator Calculator

Provides three calculation methods for technical indicators:
1. TradingView Default - Matches TradingView's built-in ta.* functions
2. pandas_ta - Current pandas_ta library implementation
3. mihakralj - Mathematically rigorous implementations from QuanTAlib

This allows comparison between methods to identify discrepancies and
choose the most consistent approach for backtesting vs TradingView execution.

Reference: https://github.com/mihakralj/pinescript
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Try to import pandas_ta, but make it optional for comparison
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False


class IndicatorEngine(Enum):
    """Available indicator calculation engines"""
    TRADINGVIEW = "tradingview"   # Match TradingView's built-in ta.* functions
    PANDAS_TA = "pandas_ta"       # pandas_ta library
    MIHAKRALJ = "mihakralj"       # QuanTAlib mathematically rigorous


@dataclass
class IndicatorResult:
    """Result from an indicator calculation with metadata"""
    values: pd.Series
    engine: IndicatorEngine
    indicator_name: str
    parameters: Dict


class MultiEngineCalculator:
    """
    Calculate indicators using multiple engines for comparison.

    Each engine aims to produce the same result, but may differ due to:
    - Smoothing method differences (RMA vs EMA vs SMA)
    - Warmup period handling
    - Edge case handling
    - Floating point precision
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV dataframe.

        Args:
            df: DataFrame with columns: open, high, low, close, volume (optional)
        """
        self.df = df.copy()
        self._validate_columns()

    def _validate_columns(self):
        """Ensure required columns exist"""
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # =========================================================================
    # RSI - Relative Strength Index
    # =========================================================================

    def rsi_tradingview(self, length: int = 14, source: str = 'close') -> pd.Series:
        """
        RSI using TradingView's method (RMA/Wilder's smoothing).

        TradingView formula:
        - change = close - close[1]
        - gain = max(change, 0)
        - loss = max(-change, 0)
        - avg_gain = RMA(gain, length)
        - avg_loss = RMA(loss, length)
        - rs = avg_gain / avg_loss
        - rsi = 100 - (100 / (1 + rs))

        RMA (Wilder's smoothing): alpha = 1/length
        """
        src = self.df[source]
        change = src.diff()

        gain = change.clip(lower=0)
        loss = (-change).clip(lower=0)

        # RMA (Wilder's smoothing) - alpha = 1/length
        alpha = 1.0 / length

        avg_gain = self._rma(gain, length, alpha)
        avg_loss = self._rma(loss, length, alpha)

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100)  # When avg_loss is 0, RSI = 100

        return rsi

    def rsi_pandas_ta(self, length: int = 14, source: str = 'close') -> pd.Series:
        """RSI using pandas_ta library"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.rsi(self.df[source], length=length)

    def rsi_mihakralj(self, length: int = 14, source: str = 'close') -> pd.Series:
        """
        RSI using mihakralj's implementation (with warmup compensation).

        From mihakralj/pinescript:
        - Uses Wilder's smoothing with warmup compensation
        - Produces valid output from bar 1
        """
        src = self.df[source].values
        n = len(src)

        alpha = 1.0 / length
        rsi_values = np.full(n, np.nan)

        smooth_up = 0.0
        smooth_down = 0.0

        for i in range(1, n):
            u = max(src[i] - src[i-1], 0)
            d = max(src[i-1] - src[i], 0)

            if i < length:
                # Warmup period - simple accumulation
                smooth_up = u
                smooth_down = d
            else:
                # Standard RMA
                smooth_up = smooth_up * (1 - alpha) + u * alpha
                smooth_down = smooth_down * (1 - alpha) + d * alpha

            if smooth_down == 0:
                rsi_values[i] = 100 if smooth_up > 0 else 50
            else:
                rs = smooth_up / smooth_down
                rsi_values[i] = 100 - (100 / (1 + rs))

        return pd.Series(rsi_values, index=self.df.index, name='RSI')

    # =========================================================================
    # SMA - Simple Moving Average
    # =========================================================================

    def sma_tradingview(self, length: int = 20, source: str = 'close') -> pd.Series:
        """SMA - same across all engines (simple mean)"""
        return self.df[source].rolling(window=length).mean()

    def sma_pandas_ta(self, length: int = 20, source: str = 'close') -> pd.Series:
        """SMA using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.sma(self.df[source], length=length)

    def sma_mihakralj(self, length: int = 20, source: str = 'close') -> pd.Series:
        """
        SMA using mihakralj's O(1) circular buffer implementation.
        Produces valid output from bar 1 (partial window).
        """
        src = self.df[source].values
        n = len(src)
        sma_values = np.full(n, np.nan)

        buffer = np.zeros(length)
        head = 0
        running_sum = 0.0
        count = 0

        for i in range(n):
            # Remove oldest value
            if count >= length:
                running_sum -= buffer[head]
            else:
                count += 1

            # Add new value
            current = src[i] if not np.isnan(src[i]) else 0.0
            running_sum += current
            buffer[head] = current
            head = (head + 1) % length

            sma_values[i] = running_sum / count

        return pd.Series(sma_values, index=self.df.index, name='SMA')

    # =========================================================================
    # EMA - Exponential Moving Average
    # =========================================================================

    def ema_tradingview(self, length: int = 20, source: str = 'close') -> pd.Series:
        """
        EMA using TradingView's method.
        alpha = 2 / (length + 1)
        EMA = alpha * close + (1 - alpha) * EMA[1]
        First value = SMA
        """
        src = self.df[source]
        alpha = 2.0 / (length + 1)

        # First value is SMA
        ema = src.ewm(span=length, adjust=False).mean()
        return ema

    def ema_pandas_ta(self, length: int = 20, source: str = 'close') -> pd.Series:
        """EMA using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.ema(self.df[source], length=length)

    def ema_mihakralj(self, length: int = 20, source: str = 'close') -> pd.Series:
        """
        EMA using mihakralj's implementation with warmup compensation.
        Produces valid output from bar 1.
        """
        src = self.df[source].values
        n = len(src)

        alpha = 2.0 / (length + 1)
        beta = 1.0 - alpha

        ema_values = np.full(n, np.nan)
        ema = 0.0
        e = 1.0  # Warmup compensation factor
        warmup = True
        EPSILON = 1e-10

        for i in range(n):
            if np.isnan(src[i]):
                continue

            ema = alpha * (src[i] - ema) + ema

            if warmup:
                e *= beta
                c = 1.0 / (1.0 - e)
                ema_values[i] = c * ema
                warmup = e > EPSILON
            else:
                ema_values[i] = ema

        return pd.Series(ema_values, index=self.df.index, name='EMA')

    # =========================================================================
    # MACD - Moving Average Convergence Divergence
    # =========================================================================

    def macd_tradingview(self, fast: int = 12, slow: int = 26, signal: int = 9,
                         source: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD using TradingView's method.
        Returns: (macd_line, signal_line, histogram)
        """
        src = self.df[source]

        ema_fast = src.ewm(span=fast, adjust=False).mean()
        ema_slow = src.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def macd_pandas_ta(self, fast: int = 12, slow: int = 26, signal: int = 9,
                       source: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")

        macd_df = ta.macd(self.df[source], fast=fast, slow=slow, signal=signal)

        # Find column names (vary by version)
        macd_col = [c for c in macd_df.columns if c.startswith('MACD_') and
                    not c.startswith('MACDs') and not c.startswith('MACDh')][0]
        signal_col = [c for c in macd_df.columns if c.startswith('MACDs_')][0]
        hist_col = [c for c in macd_df.columns if c.startswith('MACDh_')][0]

        return macd_df[macd_col], macd_df[signal_col], macd_df[hist_col]

    def macd_mihakralj(self, fast: int = 12, slow: int = 26, signal: int = 9,
                       source: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD using mihakralj's implementation with warmup compensation.
        """
        src = self.df[source].values
        n = len(src)

        alpha_fast = 2.0 / (fast + 1)
        alpha_slow = 2.0 / (slow + 1)
        alpha_signal = 2.0 / (signal + 1)

        beta_fast = 1.0 - alpha_fast
        beta_slow = 1.0 - alpha_slow
        beta_signal = 1.0 - alpha_signal

        macd_values = np.full(n, np.nan)
        signal_values = np.full(n, np.nan)
        hist_values = np.full(n, np.nan)

        ema_fast = 0.0
        ema_slow = 0.0
        ema_signal = 0.0

        e_fast = 1.0
        e_slow = 1.0
        e_signal = 1.0

        EPSILON = 1e-10
        warmup = True

        for i in range(n):
            if np.isnan(src[i]):
                continue

            ema_fast = alpha_fast * (src[i] - ema_fast) + ema_fast
            ema_slow = alpha_slow * (src[i] - ema_slow) + ema_slow

            if warmup:
                e_fast *= beta_fast
                e_slow *= beta_slow
                e_signal *= beta_signal

                c_fast = 1.0 / (1.0 - e_fast) if e_fast < 1 - EPSILON else 1.0
                c_slow = 1.0 / (1.0 - e_slow) if e_slow < 1 - EPSILON else 1.0
                c_signal = 1.0 / (1.0 - e_signal) if e_signal < 1 - EPSILON else 1.0

                result_fast = c_fast * ema_fast
                result_slow = c_slow * ema_slow
                macd_line = result_fast - result_slow

                ema_signal = alpha_signal * (macd_line - ema_signal) + ema_signal
                result_signal = c_signal * ema_signal

                warmup = e_fast > EPSILON or e_slow > EPSILON or e_signal > EPSILON
            else:
                macd_line = ema_fast - ema_slow
                ema_signal = alpha_signal * (macd_line - ema_signal) + ema_signal
                result_signal = ema_signal

            macd_values[i] = macd_line if not warmup else result_fast - result_slow
            signal_values[i] = result_signal if warmup else ema_signal
            hist_values[i] = macd_values[i] - signal_values[i]

        return (pd.Series(macd_values, index=self.df.index, name='MACD'),
                pd.Series(signal_values, index=self.df.index, name='Signal'),
                pd.Series(hist_values, index=self.df.index, name='Histogram'))

    # =========================================================================
    # Bollinger Bands
    # =========================================================================

    def bbands_tradingview(self, length: int = 20, mult: float = 2.0,
                           source: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands using TradingView's method.
        Returns: (middle, upper, lower)
        """
        src = self.df[source]

        basis = src.rolling(window=length).mean()
        # TradingView uses population std (ddof=0)
        std = src.rolling(window=length).std(ddof=0)

        upper = basis + mult * std
        lower = basis - mult * std

        return basis, upper, lower

    def bbands_pandas_ta(self, length: int = 20, mult: float = 2.0,
                         source: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")

        bb = ta.bbands(self.df[source], length=length, std=mult)

        upper_col = [c for c in bb.columns if c.startswith('BBU_')][0]
        mid_col = [c for c in bb.columns if c.startswith('BBM_')][0]
        lower_col = [c for c in bb.columns if c.startswith('BBL_')][0]

        return bb[mid_col], bb[upper_col], bb[lower_col]

    def bbands_mihakralj(self, length: int = 20, mult: float = 2.0,
                         source: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands using mihakralj's O(1) circular buffer implementation.
        """
        src = self.df[source].values
        n = len(src)

        basis_values = np.full(n, np.nan)
        upper_values = np.full(n, np.nan)
        lower_values = np.full(n, np.nan)

        buffer = np.zeros(length)
        head = 0
        running_sum = 0.0
        running_sum_sq = 0.0
        count = 0

        for i in range(n):
            if count >= length:
                oldest = buffer[head]
                running_sum -= oldest
                running_sum_sq -= oldest * oldest
            else:
                count += 1

            current = src[i] if not np.isnan(src[i]) else 0.0
            running_sum += current
            running_sum_sq += current * current
            buffer[head] = current
            head = (head + 1) % length

            basis = running_sum / count
            variance = max(0.0, running_sum_sq / count - basis * basis)
            std = np.sqrt(variance)

            basis_values[i] = basis
            upper_values[i] = basis + mult * std
            lower_values[i] = basis - mult * std

        return (pd.Series(basis_values, index=self.df.index, name='BB_Mid'),
                pd.Series(upper_values, index=self.df.index, name='BB_Upper'),
                pd.Series(lower_values, index=self.df.index, name='BB_Lower'))

    # =========================================================================
    # Stochastic Oscillator
    # =========================================================================

    def stoch_tradingview(self, k_period: int = 14, d_period: int = 3,
                          smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic using TradingView's method.
        Returns: (%K smoothed, %D)

        TradingView formula:
        - raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        - k = SMA(raw_k, smooth_k)
        - d = SMA(k, d_period)
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        raw_k = raw_k.replace([np.inf, -np.inf], np.nan)

        # Smooth %K
        k = raw_k.rolling(window=smooth_k).mean()
        # %D is SMA of smoothed %K
        d = k.rolling(window=d_period).mean()

        return k, d

    def stoch_pandas_ta(self, k_period: int = 14, d_period: int = 3,
                        smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")

        stoch = ta.stoch(self.df['high'], self.df['low'], self.df['close'],
                        k=k_period, d=d_period, smooth_k=smooth_k)

        k_col = [c for c in stoch.columns if c.startswith('STOCHk_')][0]
        d_col = [c for c in stoch.columns if c.startswith('STOCHd_')][0]

        return stoch[k_col], stoch[d_col]

    def stoch_mihakralj(self, k_period: int = 14, d_period: int = 3,
                        smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic using mihakralj's deque-based implementation.
        """
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        n = len(close)

        k_values = np.full(n, np.nan)
        d_values = np.full(n, np.nan)

        # Circular buffers for min/max
        low_buffer = np.full(k_period, np.inf)
        high_buffer = np.full(k_period, -np.inf)

        # Circular buffer for %K smoothing
        k_buffer = np.zeros(smooth_k)
        k_sum = 0.0
        k_count = 0
        k_head = 0

        # Circular buffer for %D
        d_buffer = np.zeros(d_period)
        d_sum = 0.0
        d_count = 0
        d_head = 0

        for i in range(n):
            idx = i % k_period
            low_buffer[idx] = low[i]
            high_buffer[idx] = high[i]

            if i >= k_period - 1:
                lowest = np.min(low_buffer)
                highest = np.max(high_buffer)

                range_val = highest - lowest
                raw_k = 100 * (close[i] - lowest) / range_val if range_val > 0 else 0.0

                # Smooth %K
                if k_count >= smooth_k:
                    k_sum -= k_buffer[k_head]
                else:
                    k_count += 1
                k_sum += raw_k
                k_buffer[k_head] = raw_k
                k_head = (k_head + 1) % smooth_k

                k_val = k_sum / k_count
                k_values[i] = k_val

                # Calculate %D
                if d_count >= d_period:
                    d_sum -= d_buffer[d_head]
                else:
                    d_count += 1
                d_sum += k_val
                d_buffer[d_head] = k_val
                d_head = (d_head + 1) % d_period

                d_values[i] = d_sum / d_count

        return (pd.Series(k_values, index=self.df.index, name='Stoch_K'),
                pd.Series(d_values, index=self.df.index, name='Stoch_D'))

    # =========================================================================
    # ATR - Average True Range
    # =========================================================================

    def atr_tradingview(self, length: int = 14) -> pd.Series:
        """
        ATR using TradingView's method (RMA smoothing).
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # RMA smoothing
        alpha = 1.0 / length
        atr = self._rma(tr, length, alpha)

        return atr

    def atr_pandas_ta(self, length: int = 14) -> pd.Series:
        """ATR using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.atr(self.df['high'], self.df['low'], self.df['close'], length=length)

    def atr_mihakralj(self, length: int = 14) -> pd.Series:
        """
        ATR using mihakralj's implementation with warmup compensation.
        """
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        n = len(close)

        atr_values = np.full(n, np.nan)

        alpha = 1.0 / length
        beta = 1.0 - alpha
        EPSILON = 1e-10

        raw_rma = 0.0
        e = 1.0
        prev_close = close[0]

        for i in range(n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - prev_close)
            tr3 = abs(low[i] - prev_close)
            true_range = max(tr1, tr2, tr3)

            prev_close = close[i]

            if not np.isnan(true_range):
                raw_rma = (raw_rma * (length - 1) + true_range) / length
                e *= beta

                if e > EPSILON:
                    atr_values[i] = raw_rma / (1.0 - e)
                else:
                    atr_values[i] = raw_rma

        return pd.Series(atr_values, index=self.df.index, name='ATR')

    # =========================================================================
    # Williams %R
    # =========================================================================

    def willr_tradingview(self, length: int = 14) -> pd.Series:
        """
        Williams %R using TradingView's method.
        Formula: 100 * (src - highest) / (highest - lowest)
        Range: -100 to 0 (unlike RSI which is 0 to 100)
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        highest = high.rolling(window=length).max()
        lowest = low.rolling(window=length).min()

        willr = 100 * (close - highest) / (highest - lowest)
        willr = willr.replace([np.inf, -np.inf], np.nan)

        return willr

    def willr_pandas_ta(self, length: int = 14) -> pd.Series:
        """Williams %R using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.willr(self.df['high'], self.df['low'], self.df['close'], length=length)

    # =========================================================================
    # Momentum (MOM)
    # =========================================================================

    def mom_tradingview(self, length: int = 10, source: str = 'close') -> pd.Series:
        """
        Momentum using TradingView's method.
        Formula: src - src[length]
        """
        src = self.df[source]
        return src - src.shift(length)

    def mom_pandas_ta(self, length: int = 10, source: str = 'close') -> pd.Series:
        """Momentum using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.mom(self.df[source], length=length)

    # =========================================================================
    # Money Flow Index (MFI)
    # =========================================================================

    def mfi_tradingview(self, length: int = 14) -> pd.Series:
        """
        Money Flow Index using TradingView's method.
        Uses ta.mfi(hlc3, length) internally.
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))

        # Typical price (HLC3)
        tp = (high + low + close) / 3
        raw_money_flow = tp * volume

        # Calculate positive and negative money flow
        tp_change = tp.diff()
        pos_flow = (raw_money_flow * (tp_change > 0)).rolling(window=length).sum()
        neg_flow = (raw_money_flow * (tp_change < 0)).rolling(window=length).sum()

        mfi = 100 - 100 / (1 + pos_flow / neg_flow.replace(0, np.nan))
        mfi = mfi.fillna(100)

        return mfi

    def mfi_pandas_ta(self, length: int = 14) -> pd.Series:
        """MFI using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))
        return ta.mfi(self.df['high'], self.df['low'], self.df['close'], volume, length=length)

    # =========================================================================
    # Keltner Channels
    # =========================================================================

    def keltner_tradingview(self, length: int = 20, mult: float = 2.0,
                            atr_length: int = 10) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels using TradingView's method.
        Returns: (basis, upper, lower)

        TradingView formula:
        - basis = EMA(close, length)
        - band = mult * ATR(atr_length)
        - upper = basis + band
        - lower = basis - band
        """
        close = self.df['close']

        # EMA for basis
        basis = close.ewm(span=length, adjust=False).mean()

        # ATR using RMA
        atr = self.atr_tradingview(atr_length)

        upper = basis + mult * atr
        lower = basis - mult * atr

        return basis, upper, lower

    def keltner_pandas_ta(self, length: int = 20, mult: float = 2.0,
                          atr_length: int = 10) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")

        kc = ta.kc(self.df['high'], self.df['low'], self.df['close'],
                   length=length, scalar=mult, mamode='ema')

        upper_col = [c for c in kc.columns if c.startswith('KCU')][0]
        mid_col = [c for c in kc.columns if c.startswith('KCB')][0]
        lower_col = [c for c in kc.columns if c.startswith('KCL')][0]

        return kc[mid_col], kc[upper_col], kc[lower_col]

    # =========================================================================
    # Donchian Channels
    # =========================================================================

    def donchian_tradingview(self, length: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels using TradingView's method.
        Returns: (basis, upper, lower)

        TradingView formula:
        - upper = highest(high, length)
        - lower = lowest(low, length)
        - basis = (upper + lower) / 2
        """
        high = self.df['high']
        low = self.df['low']

        upper = high.rolling(window=length).max()
        lower = low.rolling(window=length).min()
        basis = (upper + lower) / 2

        return basis, upper, lower

    def donchian_pandas_ta(self, length: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")

        dc = ta.donchian(self.df['high'], self.df['low'], lower_length=length, upper_length=length)

        upper_col = [c for c in dc.columns if 'DCU' in c][0]
        mid_col = [c for c in dc.columns if 'DCM' in c][0]
        lower_col = [c for c in dc.columns if 'DCL' in c][0]

        return dc[mid_col], dc[upper_col], dc[lower_col]

    # =========================================================================
    # Ichimoku Cloud
    # =========================================================================

    def ichimoku_tradingview(self, conversion: int = 9, base: int = 26,
                             span_b: int = 52) -> Dict[str, pd.Series]:
        """
        Ichimoku Cloud using TradingView's method.
        Returns: dict with 'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'

        TradingView formula:
        - donchian(len) = (highest(len) + lowest(len)) / 2
        - tenkan (conversion line) = donchian(conversion)
        - kijun (base line) = donchian(base)
        - senkou_a (leading span A) = (tenkan + kijun) / 2, displaced forward
        - senkou_b (leading span B) = donchian(span_b), displaced forward
        - chikou (lagging span) = close, displaced backward
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        def donchian(length):
            return (high.rolling(window=length).max() + low.rolling(window=length).min()) / 2

        tenkan = donchian(conversion)
        kijun = donchian(base)

        # Senkou spans are displaced forward by 'base' periods in TradingView display
        senkou_a = (tenkan + kijun) / 2
        senkou_b = donchian(span_b)

        # Chikou is close displaced backward by 'base' periods
        chikou = close.shift(-base)

        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }

    def ichimoku_pandas_ta(self, conversion: int = 9, base: int = 26,
                           span_b: int = 52) -> Dict[str, pd.Series]:
        """Ichimoku Cloud using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")

        ichi = ta.ichimoku(self.df['high'], self.df['low'], self.df['close'],
                           tenkan=conversion, kijun=base, senkou=span_b)

        # First dataframe contains lines, second contains spans
        if isinstance(ichi, tuple):
            lines, spans = ichi
            return {
                'tenkan': lines.iloc[:, 0],
                'kijun': lines.iloc[:, 1],
                'senkou_a': spans.iloc[:, 0] if len(spans.columns) > 0 else pd.Series(),
                'senkou_b': spans.iloc[:, 1] if len(spans.columns) > 1 else pd.Series(),
                'chikou': lines.iloc[:, 2] if len(lines.columns) > 2 else pd.Series()
            }
        return {}

    # =========================================================================
    # Aroon
    # =========================================================================

    def aroon_tradingview(self, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Aroon indicator using TradingView's method.
        Returns: (aroon_up, aroon_down, aroon_osc)

        TradingView formula:
        - aroon_up = 100 * (highestbars(high, length+1) + length) / length
        - aroon_down = 100 * (lowestbars(low, length+1) + length) / length
        - aroon_osc = aroon_up - aroon_down
        """
        high = self.df['high']
        low = self.df['low']
        n = len(self.df)

        aroon_up = np.full(n, np.nan)
        aroon_down = np.full(n, np.nan)

        for i in range(length, n):
            # Get window of data
            high_window = high.iloc[i-length:i+1].values
            low_window = low.iloc[i-length:i+1].values

            # Find bars since highest/lowest
            bars_since_high = length - np.argmax(high_window[::-1])
            bars_since_low = length - np.argmin(low_window[::-1])

            aroon_up[i] = 100 * (length - bars_since_high + 1) / length
            aroon_down[i] = 100 * (length - bars_since_low + 1) / length

        aroon_up_series = pd.Series(aroon_up, index=self.df.index, name='Aroon_Up')
        aroon_down_series = pd.Series(aroon_down, index=self.df.index, name='Aroon_Down')
        aroon_osc = aroon_up_series - aroon_down_series

        return aroon_up_series, aroon_down_series, aroon_osc

    def aroon_pandas_ta(self, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Aroon indicator using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")

        aroon = ta.aroon(self.df['high'], self.df['low'], length=length)

        up_col = [c for c in aroon.columns if 'AROONU' in c][0]
        down_col = [c for c in aroon.columns if 'AROOND' in c][0]
        osc_col = [c for c in aroon.columns if 'AROONOSC' in c][0]

        return aroon[up_col], aroon[down_col], aroon[osc_col]

    # =========================================================================
    # ADX - Average Directional Index
    # =========================================================================

    def adx_tradingview(self, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        ADX using TradingView's method.
        Returns: (adx, plus_di, minus_di)

        Uses RMA (Wilder's smoothing) for all smoothing operations.
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DM and -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0),
                           index=self.df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0),
                            index=self.df.index)

        # RMA smoothing
        alpha = 1.0 / length
        tr_rma = self._rma(tr, length, alpha)
        plus_dm_rma = self._rma(plus_dm, length, alpha)
        minus_dm_rma = self._rma(minus_dm, length, alpha)

        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_rma / tr_rma.replace(0, np.nan)
        minus_di = 100 * minus_dm_rma / tr_rma.replace(0, np.nan)

        # Calculate DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = self._rma(dx.fillna(0), length, alpha)

        return adx, plus_di.fillna(0), minus_di.fillna(0)

    def adx_pandas_ta(self, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")

        adx_df = ta.adx(self.df['high'], self.df['low'], self.df['close'], length=length)

        adx_col = [c for c in adx_df.columns if c.startswith('ADX_')][0]
        plus_col = [c for c in adx_df.columns if c.startswith('DMP_')][0]
        minus_col = [c for c in adx_df.columns if c.startswith('DMN_')][0]

        return adx_df[adx_col], adx_df[plus_col], adx_df[minus_col]

    # =========================================================================
    # CCI - Commodity Channel Index
    # =========================================================================

    def cci_tradingview(self, length: int = 20) -> pd.Series:
        """
        CCI using TradingView's method.
        Formula: (tp - SMA(tp)) / (0.015 * MAD(tp))

        Where MAD = Mean Absolute Deviation
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=length).mean()

        # Calculate Mean Absolute Deviation
        mad = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

        cci = (tp - sma_tp) / (0.015 * mad)
        return cci.replace([np.inf, -np.inf], np.nan)

    def cci_pandas_ta(self, length: int = 20) -> pd.Series:
        """CCI using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.cci(self.df['high'], self.df['low'], self.df['close'], length=length)

    # =========================================================================
    # ROC - Rate of Change
    # =========================================================================

    def roc_tradingview(self, length: int = 9, source: str = 'close') -> pd.Series:
        """
        ROC using TradingView's method.
        Formula: 100 * (close - close[length]) / close[length]
        """
        src = self.df[source]
        return 100 * (src - src.shift(length)) / src.shift(length)

    def roc_pandas_ta(self, length: int = 9, source: str = 'close') -> pd.Series:
        """ROC using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.roc(self.df[source], length=length)

    # =========================================================================
    # VWAP - Volume Weighted Average Price
    # =========================================================================

    def vwap_tradingview(self) -> pd.Series:
        """
        VWAP using TradingView's method.
        Formula: cumsum(tp * volume) / cumsum(volume)

        Note: TradingView resets VWAP at session start. This is cumulative.
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))

        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum()

    def vwap_pandas_ta(self) -> pd.Series:
        """VWAP using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))
        return ta.vwap(self.df['high'], self.df['low'], self.df['close'], volume)

    # =========================================================================
    # CMF - Chaikin Money Flow
    # =========================================================================

    def cmf_tradingview(self, length: int = 20) -> pd.Series:
        """
        CMF using TradingView's method.
        Formula: sum(((close-low)-(high-close))/(high-low)*volume, length) / sum(volume, length)
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))

        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        mfm = mfm.fillna(0)

        # Money Flow Volume
        mfv = mfm * volume

        cmf = mfv.rolling(window=length).sum() / volume.rolling(window=length).sum()
        return cmf

    def cmf_pandas_ta(self, length: int = 20) -> pd.Series:
        """CMF using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))
        return ta.cmf(self.df['high'], self.df['low'], self.df['close'], volume, length=length)

    # =========================================================================
    # OBV - On Balance Volume
    # =========================================================================

    def obv_tradingview(self) -> pd.Series:
        """
        OBV using TradingView's method.
        Cumulative sum of volume where volume is signed by price change.
        """
        close = self.df['close']
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))

        price_change = close.diff()
        signed_volume = np.where(price_change > 0, volume,
                                np.where(price_change < 0, -volume, 0))

        return pd.Series(signed_volume, index=self.df.index).cumsum()

    def obv_pandas_ta(self) -> pd.Series:
        """OBV using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))
        return ta.obv(self.df['close'], volume)

    # =========================================================================
    # Supertrend
    # =========================================================================

    def supertrend_tradingview(self, factor: float = 3.0, atr_length: int = 10) -> Tuple[pd.Series, pd.Series]:
        """
        Supertrend using TradingView's method.
        Returns: (supertrend_value, direction)

        Direction: 1 = bearish (price below), -1 = bullish (price above)
        This matches TradingView's convention.
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        n = len(self.df)

        # Calculate ATR
        atr = self.atr_tradingview(atr_length)

        # Calculate basic bands
        hl2 = (high + low) / 2
        basic_upper = hl2 + factor * atr
        basic_lower = hl2 - factor * atr

        # Initialize arrays
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        supertrend = np.full(n, np.nan)
        direction = np.full(n, np.nan)

        for i in range(1, n):
            if np.isnan(basic_upper.iloc[i]) or np.isnan(basic_lower.iloc[i]):
                continue

            # Upper band: min(basic_upper, upper[1]) if close[1] < upper[1]
            if np.isnan(upper[i-1]) or close.iloc[i-1] > upper[i-1]:
                upper[i] = basic_upper.iloc[i]
            else:
                upper[i] = min(basic_upper.iloc[i], upper[i-1])

            # Lower band: max(basic_lower, lower[1]) if close[1] > lower[1]
            if np.isnan(lower[i-1]) or close.iloc[i-1] < lower[i-1]:
                lower[i] = basic_lower.iloc[i]
            else:
                lower[i] = max(basic_lower.iloc[i], lower[i-1])

            # Determine direction and supertrend value
            if np.isnan(supertrend[i-1]):
                direction[i] = -1  # Start bullish
                supertrend[i] = lower[i]
            elif supertrend[i-1] == upper[i-1]:
                # Was bearish
                if close.iloc[i] > upper[i]:
                    direction[i] = -1  # Turn bullish
                    supertrend[i] = lower[i]
                else:
                    direction[i] = 1  # Stay bearish
                    supertrend[i] = upper[i]
            else:
                # Was bullish
                if close.iloc[i] < lower[i]:
                    direction[i] = 1  # Turn bearish
                    supertrend[i] = upper[i]
                else:
                    direction[i] = -1  # Stay bullish
                    supertrend[i] = lower[i]

        return (pd.Series(supertrend, index=self.df.index, name='Supertrend'),
                pd.Series(direction, index=self.df.index, name='Direction'))

    def supertrend_pandas_ta(self, factor: float = 3.0, atr_length: int = 10) -> Tuple[pd.Series, pd.Series]:
        """Supertrend using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")

        st = ta.supertrend(self.df['high'], self.df['low'], self.df['close'],
                          length=atr_length, multiplier=factor)

        st_col = [c for c in st.columns if c.startswith('SUPERT_')][0]
        dir_col = [c for c in st.columns if c.startswith('SUPERTd_')][0]

        return st[st_col], st[dir_col]

    # =========================================================================
    # Parabolic SAR
    # =========================================================================

    def psar_tradingview(self, start: float = 0.02, increment: float = 0.02,
                         maximum: float = 0.2) -> pd.Series:
        """
        Parabolic SAR using TradingView's method.
        """
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        n = len(close)

        psar = np.full(n, np.nan)
        af = start
        ep = high[0]
        is_uptrend = True
        psar[0] = low[0]

        for i in range(1, n):
            if is_uptrend:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                psar[i] = min(psar[i], low[i-1], low[i-2] if i > 1 else low[i-1])

                if high[i] > ep:
                    ep = high[i]
                    af = min(af + increment, maximum)

                if low[i] < psar[i]:
                    is_uptrend = False
                    psar[i] = ep
                    ep = low[i]
                    af = start
            else:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                psar[i] = max(psar[i], high[i-1], high[i-2] if i > 1 else high[i-1])

                if low[i] < ep:
                    ep = low[i]
                    af = min(af + increment, maximum)

                if high[i] > psar[i]:
                    is_uptrend = True
                    psar[i] = ep
                    ep = high[i]
                    af = start

        return pd.Series(psar, index=self.df.index, name='PSAR')

    def psar_pandas_ta(self, start: float = 0.02, increment: float = 0.02,
                       maximum: float = 0.2) -> pd.Series:
        """Parabolic SAR using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")

        psar_df = ta.psar(self.df['high'], self.df['low'], self.df['close'],
                         af0=start, af=increment, max_af=maximum)

        # Get long or short SAR value
        long_col = [c for c in psar_df.columns if 'PSARl' in c]
        short_col = [c for c in psar_df.columns if 'PSARs' in c]

        if long_col and short_col:
            psar = psar_df[long_col[0]].fillna(psar_df[short_col[0]])
            return psar

        return psar_df.iloc[:, 0]

    # =========================================================================
    # Pivot Points
    # =========================================================================

    def pivot_tradingview(self) -> Dict[str, pd.Series]:
        """
        Traditional Pivot Points using TradingView's method.
        Returns dict with: 'pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3'

        Uses previous bar's OHLC for calculation.
        """
        high = self.df['high'].shift(1)
        low = self.df['low'].shift(1)
        close = self.df['close'].shift(1)

        pivot = (high + low + close) / 3

        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = pivot + 2 * (high - low)
        s3 = pivot - 2 * (high - low)

        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }

    # =========================================================================
    # TRIX
    # =========================================================================

    def trix_tradingview(self, length: int = 18, source: str = 'close') -> pd.Series:
        """
        TRIX using TradingView's method.
        Triple exponential smoothed rate of change.
        """
        src = self.df[source]

        # Triple EMA
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()

        # Rate of change of triple EMA
        trix = 10000 * (ema3 - ema3.shift(1)) / ema3.shift(1)

        return trix

    def trix_pandas_ta(self, length: int = 18, source: str = 'close') -> pd.Series:
        """TRIX using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.trix(self.df[source], length=length)

    # =========================================================================
    # Ultimate Oscillator
    # =========================================================================

    def uo_tradingview(self, fast: int = 7, medium: int = 14, slow: int = 28) -> pd.Series:
        """
        Ultimate Oscillator using TradingView's method.
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        prev_close = close.shift(1)

        # True Low
        tl = pd.concat([low, prev_close], axis=1).min(axis=1)

        # Buying Pressure
        bp = close - tl

        # True Range
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Sum of BP and TR for each period
        bp_fast = bp.rolling(window=fast).sum()
        tr_fast = tr.rolling(window=fast).sum()

        bp_medium = bp.rolling(window=medium).sum()
        tr_medium = tr.rolling(window=medium).sum()

        bp_slow = bp.rolling(window=slow).sum()
        tr_slow = tr.rolling(window=slow).sum()

        # Average ratios with weights 4:2:1
        avg1 = bp_fast / tr_fast.replace(0, np.nan)
        avg2 = bp_medium / tr_medium.replace(0, np.nan)
        avg3 = bp_slow / tr_slow.replace(0, np.nan)

        uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

        return uo

    def uo_pandas_ta(self, fast: int = 7, medium: int = 14, slow: int = 28) -> pd.Series:
        """Ultimate Oscillator using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.uo(self.df['high'], self.df['low'], self.df['close'],
                     fast=fast, medium=medium, slow=slow)

    # =========================================================================
    # Choppiness Index
    # =========================================================================

    def chop_tradingview(self, length: int = 14) -> pd.Series:
        """
        Choppiness Index using TradingView's method.
        Higher values = choppier/ranging, Lower values = trending
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        # True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Sum of TR
        atr_sum = tr.rolling(window=length).sum()

        # High-Low range
        highest = high.rolling(window=length).max()
        lowest = low.rolling(window=length).min()
        hl_range = highest - lowest

        # Choppiness Index
        chop = 100 * np.log10(atr_sum / hl_range.replace(0, np.nan)) / np.log10(length)

        return chop

    def chop_pandas_ta(self, length: int = 14) -> pd.Series:
        """Choppiness Index using pandas_ta"""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta not available")
        return ta.chop(self.df['high'], self.df['low'], self.df['close'], length=length)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _rma(self, series: pd.Series, length: int, alpha: float) -> pd.Series:
        """
        RMA (Wilder's smoothing) - used by TradingView for RSI, ATR, etc.
        alpha = 1/length (different from EMA's 2/(length+1))
        """
        result = series.copy()

        # Initialize with SMA for first 'length' values
        result.iloc[:length] = series.iloc[:length].expanding().mean()

        # Apply RMA formula
        for i in range(length, len(series)):
            result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]

        return result

    # =========================================================================
    # Comparison Methods
    # =========================================================================

    def compare_indicator(self, indicator: str, **params) -> pd.DataFrame:
        """
        Compare an indicator across all three engines.

        Args:
            indicator: One of 'rsi', 'sma', 'ema', 'macd', 'bbands', 'stoch', 'atr'
            **params: Indicator parameters (e.g., length=14)

        Returns:
            DataFrame with columns for each engine and difference stats
        """
        methods = {
            'rsi': (self.rsi_tradingview, self.rsi_pandas_ta, self.rsi_mihakralj),
            'sma': (self.sma_tradingview, self.sma_pandas_ta, self.sma_mihakralj),
            'ema': (self.ema_tradingview, self.ema_pandas_ta, self.ema_mihakralj),
            'atr': (self.atr_tradingview, self.atr_pandas_ta, self.atr_mihakralj),
        }

        if indicator not in methods:
            raise ValueError(f"Unknown indicator: {indicator}. Available: {list(methods.keys())}")

        tv_method, pta_method, mih_method = methods[indicator]

        results = pd.DataFrame()
        results['tradingview'] = tv_method(**params)

        try:
            results['pandas_ta'] = pta_method(**params)
        except ImportError:
            results['pandas_ta'] = np.nan

        results['mihakralj'] = mih_method(**params)

        # Calculate differences
        results['tv_vs_pta_diff'] = results['tradingview'] - results['pandas_ta']
        results['tv_vs_mih_diff'] = results['tradingview'] - results['mihakralj']
        results['pta_vs_mih_diff'] = results['pandas_ta'] - results['mihakralj']

        # Calculate percentage differences
        results['tv_vs_pta_pct'] = (results['tv_vs_pta_diff'] / results['tradingview'].abs()) * 100
        results['tv_vs_mih_pct'] = (results['tv_vs_mih_diff'] / results['tradingview'].abs()) * 100

        return results

    def compare_all_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Compare all main indicators across engines.

        Returns:
            Dict of indicator name -> comparison DataFrame
        """
        comparisons = {}

        # RSI
        comparisons['rsi'] = self.compare_indicator('rsi', length=14)

        # SMA
        comparisons['sma'] = self.compare_indicator('sma', length=20)

        # EMA
        comparisons['ema'] = self.compare_indicator('ema', length=20)

        # ATR
        comparisons['atr'] = self.compare_indicator('atr', length=14)

        return comparisons

    def get_comparison_summary(self) -> Dict:
        """
        Get a summary of differences between engines.

        Returns:
            Dict with mean absolute differences for each indicator
        """
        comparisons = self.compare_all_indicators()

        summary = {}
        for indicator, df in comparisons.items():
            summary[indicator] = {
                'tv_vs_pandas_ta_mae': df['tv_vs_pta_diff'].abs().mean(),
                'tv_vs_mihakralj_mae': df['tv_vs_mih_diff'].abs().mean(),
                'pandas_ta_vs_mihakralj_mae': df['pta_vs_mih_diff'].abs().mean(),
                'tv_vs_pandas_ta_max_pct': df['tv_vs_pta_pct'].abs().max(),
                'tv_vs_mihakralj_max_pct': df['tv_vs_mih_pct'].abs().max(),
            }

        return summary


def calculate_indicators(df: pd.DataFrame, engine: IndicatorEngine = IndicatorEngine.MIHAKRALJ) -> pd.DataFrame:
    """
    Calculate all indicators using the specified engine.

    Args:
        df: OHLCV DataFrame
        engine: Which calculation engine to use

    Returns:
        DataFrame with indicator columns added
    """
    calc = MultiEngineCalculator(df)
    result = df.copy()

    if engine == IndicatorEngine.TRADINGVIEW:
        # Core indicators
        result['rsi'] = calc.rsi_tradingview()
        result['sma_20'] = calc.sma_tradingview(20)
        result['sma_50'] = calc.sma_tradingview(50)
        result['ema_9'] = calc.ema_tradingview(9)
        result['ema_21'] = calc.ema_tradingview(21)
        result['atr'] = calc.atr_tradingview()

        macd, signal, hist = calc.macd_tradingview()
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist

        bb_mid, bb_upper, bb_lower = calc.bbands_tradingview()
        result['bb_mid'] = bb_mid
        result['bb_upper'] = bb_upper
        result['bb_lower'] = bb_lower

        stoch_k, stoch_d = calc.stoch_tradingview()
        result['stoch_k'] = stoch_k
        result['stoch_d'] = stoch_d

        # Additional TradingView-specific indicators
        result['willr'] = calc.willr_tradingview()
        result['mom'] = calc.mom_tradingview()
        result['mfi'] = calc.mfi_tradingview()
        result['cci'] = calc.cci_tradingview()
        result['roc'] = calc.roc_tradingview()

        # ADX
        adx, plus_di, minus_di = calc.adx_tradingview()
        result['adx'] = adx
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di

        # Aroon
        aroon_up, aroon_down, aroon_osc = calc.aroon_tradingview()
        result['aroon_up'] = aroon_up
        result['aroon_down'] = aroon_down
        result['aroon_osc'] = aroon_osc

        # Keltner Channels
        kc_mid, kc_upper, kc_lower = calc.keltner_tradingview()
        result['kc_mid'] = kc_mid
        result['kc_upper'] = kc_upper
        result['kc_lower'] = kc_lower

        # Donchian Channels
        dc_mid, dc_upper, dc_lower = calc.donchian_tradingview()
        result['dc_mid'] = dc_mid
        result['dc_upper'] = dc_upper
        result['dc_lower'] = dc_lower

        # Ichimoku Cloud
        ichimoku = calc.ichimoku_tradingview()
        result['tenkan'] = ichimoku['tenkan']
        result['kijun'] = ichimoku['kijun']
        result['senkou_a'] = ichimoku['senkou_a']
        result['senkou_b'] = ichimoku['senkou_b']
        result['chikou'] = ichimoku['chikou']

        # Supertrend
        supertrend, supertrend_dir = calc.supertrend_tradingview()
        result['supertrend'] = supertrend
        result['supertrend_dir'] = supertrend_dir

        # Volume indicators (if volume available)
        if 'volume' in df.columns:
            result['vwap'] = calc.vwap_tradingview()
            result['cmf'] = calc.cmf_tradingview()
            result['obv'] = calc.obv_tradingview()

        # Other oscillators
        result['trix'] = calc.trix_tradingview()
        result['uo'] = calc.uo_tradingview()
        result['chop'] = calc.chop_tradingview()
        result['psar'] = calc.psar_tradingview()

        # Pivot Points
        pivots = calc.pivot_tradingview()
        result['pivot'] = pivots['pivot']
        result['pivot_r1'] = pivots['r1']
        result['pivot_r2'] = pivots['r2']
        result['pivot_r3'] = pivots['r3']
        result['pivot_s1'] = pivots['s1']
        result['pivot_s2'] = pivots['s2']
        result['pivot_s3'] = pivots['s3']

    elif engine == IndicatorEngine.PANDAS_TA:
        result['rsi'] = calc.rsi_pandas_ta()
        result['sma_20'] = calc.sma_pandas_ta(20)
        result['sma_50'] = calc.sma_pandas_ta(50)
        result['ema_9'] = calc.ema_pandas_ta(9)
        result['ema_21'] = calc.ema_pandas_ta(21)
        result['atr'] = calc.atr_pandas_ta()

        macd, signal, hist = calc.macd_pandas_ta()
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist

        bb_mid, bb_upper, bb_lower = calc.bbands_pandas_ta()
        result['bb_mid'] = bb_mid
        result['bb_upper'] = bb_upper
        result['bb_lower'] = bb_lower

        stoch_k, stoch_d = calc.stoch_pandas_ta()
        result['stoch_k'] = stoch_k
        result['stoch_d'] = stoch_d

    elif engine == IndicatorEngine.MIHAKRALJ:
        result['rsi'] = calc.rsi_mihakralj()
        result['sma_20'] = calc.sma_mihakralj(20)
        result['sma_50'] = calc.sma_mihakralj(50)
        result['ema_9'] = calc.ema_mihakralj(9)
        result['ema_21'] = calc.ema_mihakralj(21)
        result['atr'] = calc.atr_mihakralj()

        macd, signal, hist = calc.macd_mihakralj()
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist

        bb_mid, bb_upper, bb_lower = calc.bbands_mihakralj()
        result['bb_mid'] = bb_mid
        result['bb_upper'] = bb_upper
        result['bb_lower'] = bb_lower

        stoch_k, stoch_d = calc.stoch_mihakralj()
        result['stoch_k'] = stoch_k
        result['stoch_d'] = stoch_d

    return result
