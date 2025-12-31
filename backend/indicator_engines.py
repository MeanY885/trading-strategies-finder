"""
Dual-Engine Indicator Calculator

Provides two calculation methods for technical indicators:
1. TradingView - Matches TradingView's built-in ta.* functions (for Pine Script export)
2. Native - Uses TA-Lib for fast execution + candlestick pattern recognition

Use TradingView engine when exporting strategies to Pine Script.
Use Native engine for in-app execution with pattern-based signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from enum import Enum

# Import TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


class IndicatorEngine(Enum):
    """Available indicator calculation engines"""
    TRADINGVIEW = "tradingview"  # Match TradingView's built-in ta.* functions
    NATIVE = "native"            # TA-Lib based (fast + patterns)


# =============================================================================
# CANDLESTICK PATTERNS - Available in Native engine only
# =============================================================================

CANDLESTICK_PATTERNS = {
    # Bullish Patterns
    'CDL_HAMMER': {'name': 'Hammer', 'type': 'bullish', 'func': 'CDLHAMMER'},
    'CDL_INVERTED_HAMMER': {'name': 'Inverted Hammer', 'type': 'bullish', 'func': 'CDLINVERTEDHAMMER'},
    'CDL_ENGULFING_BULL': {'name': 'Bullish Engulfing', 'type': 'bullish', 'func': 'CDLENGULFING'},
    'CDL_PIERCING': {'name': 'Piercing Line', 'type': 'bullish', 'func': 'CDLPIERCING'},
    'CDL_MORNING_STAR': {'name': 'Morning Star', 'type': 'bullish', 'func': 'CDLMORNINGSTAR'},
    'CDL_THREE_WHITE_SOLDIERS': {'name': 'Three White Soldiers', 'type': 'bullish', 'func': 'CDL3WHITESOLDIERS'},
    'CDL_BULLISH_HARAMI': {'name': 'Bullish Harami', 'type': 'bullish', 'func': 'CDLHARAMI'},
    'CDL_DRAGONFLY_DOJI': {'name': 'Dragonfly Doji', 'type': 'bullish', 'func': 'CDLDRAGONFLYDOJI'},
    'CDL_MORNING_DOJI_STAR': {'name': 'Morning Doji Star', 'type': 'bullish', 'func': 'CDLMORNINGDOJISTAR'},

    # Bearish Patterns
    'CDL_HANGING_MAN': {'name': 'Hanging Man', 'type': 'bearish', 'func': 'CDLHANGINGMAN'},
    'CDL_SHOOTING_STAR': {'name': 'Shooting Star', 'type': 'bearish', 'func': 'CDLSHOOTINGSTAR'},
    'CDL_ENGULFING_BEAR': {'name': 'Bearish Engulfing', 'type': 'bearish', 'func': 'CDLENGULFING'},
    'CDL_DARK_CLOUD': {'name': 'Dark Cloud Cover', 'type': 'bearish', 'func': 'CDLDARKCLOUDCOVER'},
    'CDL_EVENING_STAR': {'name': 'Evening Star', 'type': 'bearish', 'func': 'CDLEVENINGSTAR'},
    'CDL_THREE_BLACK_CROWS': {'name': 'Three Black Crows', 'type': 'bearish', 'func': 'CDL3BLACKCROWS'},
    'CDL_BEARISH_HARAMI': {'name': 'Bearish Harami', 'type': 'bearish', 'func': 'CDLHARAMI'},
    'CDL_GRAVESTONE_DOJI': {'name': 'Gravestone Doji', 'type': 'bearish', 'func': 'CDLGRAVESTONEDOJI'},
    'CDL_EVENING_DOJI_STAR': {'name': 'Evening Doji Star', 'type': 'bearish', 'func': 'CDLEVENINGDOJISTAR'},

    # Neutral/Reversal Patterns
    'CDL_DOJI': {'name': 'Doji', 'type': 'neutral', 'func': 'CDLDOJI'},
    'CDL_SPINNING_TOP': {'name': 'Spinning Top', 'type': 'neutral', 'func': 'CDLSPINNINGTOP'},
    'CDL_MARUBOZU': {'name': 'Marubozu', 'type': 'neutral', 'func': 'CDLMARUBOZU'},
    'CDL_LONG_LINE': {'name': 'Long Line', 'type': 'neutral', 'func': 'CDLLONGLINE'},
    'CDL_BELT_HOLD': {'name': 'Belt Hold', 'type': 'neutral', 'func': 'CDLBELTHOLD'},
    'CDL_KICKING': {'name': 'Kicking', 'type': 'neutral', 'func': 'CDLKICKING'},
    'CDL_TASUKI_GAP': {'name': 'Tasuki Gap', 'type': 'neutral', 'func': 'CDLTASUKIGAP'},
    'CDL_GAP_THREE_METHODS': {'name': 'Gap Three Methods', 'type': 'neutral', 'func': 'CDLGAPSIDESIDEWHITE'},
}


class MultiEngineCalculator:
    """
    Calculate indicators using TradingView or Native (TA-Lib) engine.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV dataframe.

        Args:
            df: DataFrame with columns: open, high, low, close, volume (optional)
        """
        self.df = df.copy()
        self._validate_columns()

        # Prepare numpy arrays for TA-Lib (requires float64)
        self.open = self.df['open'].values.astype(np.float64)
        self.high = self.df['high'].values.astype(np.float64)
        self.low = self.df['low'].values.astype(np.float64)
        self.close = self.df['close'].values.astype(np.float64)
        self.volume = self.df['volume'].values.astype(np.float64) if 'volume' in self.df.columns else None

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
        """
        src = self.df[source]
        change = src.diff()

        gain = change.clip(lower=0)
        loss = (-change).clip(lower=0)

        alpha = 1.0 / length
        avg_gain = self._rma(gain, length, alpha)
        avg_loss = self._rma(loss, length, alpha)

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100)

        return rsi

    def rsi_native(self, length: int = 14, source: str = 'close') -> pd.Series:
        """RSI using TA-Lib"""
        if TALIB_AVAILABLE:
            src = self.df[source].values.astype(np.float64)
            result = talib.RSI(src, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='RSI')
        else:
            return self.rsi_tradingview(length, source)

    # =========================================================================
    # SMA - Simple Moving Average
    # =========================================================================

    def sma_tradingview(self, length: int = 20, source: str = 'close') -> pd.Series:
        """SMA using TradingView's method"""
        return self.df[source].rolling(window=length).mean()

    def sma_native(self, length: int = 20, source: str = 'close') -> pd.Series:
        """SMA using TA-Lib"""
        if TALIB_AVAILABLE:
            src = self.df[source].values.astype(np.float64)
            result = talib.SMA(src, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='SMA')
        else:
            return self.sma_tradingview(length, source)

    # =========================================================================
    # EMA - Exponential Moving Average
    # =========================================================================

    def ema_tradingview(self, length: int = 20, source: str = 'close') -> pd.Series:
        """EMA using TradingView's method"""
        src = self.df[source]
        return src.ewm(span=length, adjust=False).mean()

    def ema_native(self, length: int = 20, source: str = 'close') -> pd.Series:
        """EMA using TA-Lib"""
        if TALIB_AVAILABLE:
            src = self.df[source].values.astype(np.float64)
            result = talib.EMA(src, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='EMA')
        else:
            return self.ema_tradingview(length, source)

    # =========================================================================
    # MACD - Moving Average Convergence Divergence
    # =========================================================================

    def macd_tradingview(self, fast: int = 12, slow: int = 26, signal: int = 9,
                         source: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD using TradingView's method"""
        src = self.df[source]

        ema_fast = src.ewm(span=fast, adjust=False).mean()
        ema_slow = src.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def macd_native(self, fast: int = 12, slow: int = 26, signal: int = 9,
                    source: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD using TA-Lib"""
        if TALIB_AVAILABLE:
            src = self.df[source].values.astype(np.float64)
            macd, signal_line, hist = talib.MACD(src, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return (pd.Series(macd, index=self.df.index, name='MACD'),
                    pd.Series(signal_line, index=self.df.index, name='Signal'),
                    pd.Series(hist, index=self.df.index, name='Histogram'))
        else:
            return self.macd_tradingview(fast, slow, signal, source)

    # =========================================================================
    # Bollinger Bands
    # =========================================================================

    def bbands_tradingview(self, length: int = 20, mult: float = 2.0,
                           source: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands using TradingView's method"""
        src = self.df[source]

        basis = src.rolling(window=length).mean()
        std = src.rolling(window=length).std(ddof=0)

        upper = basis + mult * std
        lower = basis - mult * std

        return basis, upper, lower

    def bbands_native(self, length: int = 20, mult: float = 2.0,
                      source: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands using TA-Lib"""
        if TALIB_AVAILABLE:
            src = self.df[source].values.astype(np.float64)
            upper, middle, lower = talib.BBANDS(src, timeperiod=length, nbdevup=mult, nbdevdn=mult)
            return (pd.Series(middle, index=self.df.index, name='BB_Mid'),
                    pd.Series(upper, index=self.df.index, name='BB_Upper'),
                    pd.Series(lower, index=self.df.index, name='BB_Lower'))
        else:
            return self.bbands_tradingview(length, mult, source)

    # =========================================================================
    # Stochastic Oscillator
    # =========================================================================

    def stoch_tradingview(self, k_period: int = 14, d_period: int = 3,
                          smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        raw_k = raw_k.replace([np.inf, -np.inf], np.nan)

        k = raw_k.rolling(window=smooth_k).mean()
        d = k.rolling(window=d_period).mean()

        return k, d

    def stoch_native(self, k_period: int = 14, d_period: int = 3,
                     smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic using TA-Lib"""
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(self.high, self.low, self.close,
                                        fastk_period=k_period, slowk_period=smooth_k, slowd_period=d_period)
            return (pd.Series(slowk, index=self.df.index, name='Stoch_K'),
                    pd.Series(slowd, index=self.df.index, name='Stoch_D'))
        else:
            return self.stoch_tradingview(k_period, d_period, smooth_k)

    # =========================================================================
    # ATR - Average True Range
    # =========================================================================

    def atr_tradingview(self, length: int = 14) -> pd.Series:
        """ATR using TradingView's method (RMA smoothing)"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        alpha = 1.0 / length
        atr = self._rma(tr, length, alpha)

        return atr

    def atr_native(self, length: int = 14) -> pd.Series:
        """ATR using TA-Lib"""
        if TALIB_AVAILABLE:
            result = talib.ATR(self.high, self.low, self.close, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='ATR')
        else:
            return self.atr_tradingview(length)

    # =========================================================================
    # ADX - Average Directional Index
    # =========================================================================

    def adx_tradingview(self, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0),
                           index=self.df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0),
                            index=self.df.index)

        alpha = 1.0 / length
        tr_rma = self._rma(tr, length, alpha)
        plus_dm_rma = self._rma(plus_dm, length, alpha)
        minus_dm_rma = self._rma(minus_dm, length, alpha)

        plus_di = 100 * plus_dm_rma / tr_rma.replace(0, np.nan)
        minus_di = 100 * minus_dm_rma / tr_rma.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = self._rma(dx.fillna(0), length, alpha)

        return adx, plus_di.fillna(0), minus_di.fillna(0)

    def adx_native(self, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX using TA-Lib"""
        if TALIB_AVAILABLE:
            adx = talib.ADX(self.high, self.low, self.close, timeperiod=length)
            plus_di = talib.PLUS_DI(self.high, self.low, self.close, timeperiod=length)
            minus_di = talib.MINUS_DI(self.high, self.low, self.close, timeperiod=length)
            return (pd.Series(adx, index=self.df.index, name='ADX'),
                    pd.Series(plus_di, index=self.df.index, name='Plus_DI'),
                    pd.Series(minus_di, index=self.df.index, name='Minus_DI'))
        else:
            return self.adx_tradingview(length)

    # =========================================================================
    # Williams %R
    # =========================================================================

    def willr_tradingview(self, length: int = 14) -> pd.Series:
        """Williams %R using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        highest = high.rolling(window=length).max()
        lowest = low.rolling(window=length).min()

        willr = 100 * (close - highest) / (highest - lowest)
        return willr.replace([np.inf, -np.inf], np.nan)

    def willr_native(self, length: int = 14) -> pd.Series:
        """Williams %R using TA-Lib"""
        if TALIB_AVAILABLE:
            result = talib.WILLR(self.high, self.low, self.close, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='WILLR')
        else:
            return self.willr_tradingview(length)

    # =========================================================================
    # CCI - Commodity Channel Index
    # =========================================================================

    def cci_tradingview(self, length: int = 20) -> pd.Series:
        """CCI using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=length).mean()
        mad = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

        cci = (tp - sma_tp) / (0.015 * mad)
        return cci.replace([np.inf, -np.inf], np.nan)

    def cci_native(self, length: int = 20) -> pd.Series:
        """CCI using TA-Lib"""
        if TALIB_AVAILABLE:
            result = talib.CCI(self.high, self.low, self.close, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='CCI')
        else:
            return self.cci_tradingview(length)

    # =========================================================================
    # MFI - Money Flow Index
    # =========================================================================

    def mfi_tradingview(self, length: int = 14) -> pd.Series:
        """MFI using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))

        tp = (high + low + close) / 3
        raw_money_flow = tp * volume

        tp_change = tp.diff()
        pos_flow = (raw_money_flow * (tp_change > 0)).rolling(window=length).sum()
        neg_flow = (raw_money_flow * (tp_change < 0)).rolling(window=length).sum()

        mfi = 100 - 100 / (1 + pos_flow / neg_flow.replace(0, np.nan))
        return mfi.fillna(100)

    def mfi_native(self, length: int = 14) -> pd.Series:
        """MFI using TA-Lib"""
        if TALIB_AVAILABLE and self.volume is not None:
            result = talib.MFI(self.high, self.low, self.close, self.volume, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='MFI')
        else:
            return self.mfi_tradingview(length)

    # =========================================================================
    # ROC - Rate of Change
    # =========================================================================

    def roc_tradingview(self, length: int = 9, source: str = 'close') -> pd.Series:
        """ROC using TradingView's method"""
        src = self.df[source]
        return 100 * (src - src.shift(length)) / src.shift(length)

    def roc_native(self, length: int = 9, source: str = 'close') -> pd.Series:
        """ROC using TA-Lib"""
        if TALIB_AVAILABLE:
            src = self.df[source].values.astype(np.float64)
            result = talib.ROC(src, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='ROC')
        else:
            return self.roc_tradingview(length, source)

    # =========================================================================
    # Momentum
    # =========================================================================

    def mom_tradingview(self, length: int = 10, source: str = 'close') -> pd.Series:
        """Momentum using TradingView's method"""
        src = self.df[source]
        return src - src.shift(length)

    def mom_native(self, length: int = 10, source: str = 'close') -> pd.Series:
        """Momentum using TA-Lib"""
        if TALIB_AVAILABLE:
            src = self.df[source].values.astype(np.float64)
            result = talib.MOM(src, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='MOM')
        else:
            return self.mom_tradingview(length, source)

    # =========================================================================
    # Supertrend
    # =========================================================================

    def supertrend_tradingview(self, factor: float = 3.0, atr_length: int = 10) -> Tuple[pd.Series, pd.Series]:
        """Supertrend using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        n = len(self.df)

        atr = self.atr_tradingview(atr_length)

        hl2 = (high + low) / 2
        basic_upper = hl2 + factor * atr
        basic_lower = hl2 - factor * atr

        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        supertrend = np.full(n, np.nan)
        direction = np.full(n, np.nan)

        for i in range(1, n):
            if np.isnan(basic_upper.iloc[i]) or np.isnan(basic_lower.iloc[i]):
                continue

            if np.isnan(upper[i-1]) or close.iloc[i-1] > upper[i-1]:
                upper[i] = basic_upper.iloc[i]
            else:
                upper[i] = min(basic_upper.iloc[i], upper[i-1])

            if np.isnan(lower[i-1]) or close.iloc[i-1] < lower[i-1]:
                lower[i] = basic_lower.iloc[i]
            else:
                lower[i] = max(basic_lower.iloc[i], lower[i-1])

            if np.isnan(supertrend[i-1]):
                direction[i] = -1
                supertrend[i] = lower[i]
            elif supertrend[i-1] == upper[i-1]:
                if close.iloc[i] > upper[i]:
                    direction[i] = -1
                    supertrend[i] = lower[i]
                else:
                    direction[i] = 1
                    supertrend[i] = upper[i]
            else:
                if close.iloc[i] < lower[i]:
                    direction[i] = 1
                    supertrend[i] = upper[i]
                else:
                    direction[i] = -1
                    supertrend[i] = lower[i]

        return (pd.Series(supertrend, index=self.df.index, name='Supertrend'),
                pd.Series(direction, index=self.df.index, name='Direction'))

    def supertrend_native(self, factor: float = 3.0, atr_length: int = 10) -> Tuple[pd.Series, pd.Series]:
        """Supertrend - no TA-Lib equivalent, use TradingView implementation"""
        return self.supertrend_tradingview(factor, atr_length)

    # =========================================================================
    # McGinley Dynamic
    # =========================================================================

    def mcginley_dynamic(self, length: int = 14, k: float = 0.6) -> pd.Series:
        """
        McGinley Dynamic - Self-adjusting moving average that tracks price more closely.

        The McGinley Dynamic automatically adjusts its speed based on market velocity,
        providing smoother trend following with less lag than traditional MAs.

        Formula: MD = MD[1] + (Close - MD[1]) / (k * N * (Close/MD[1])^4)

        Where:
        - MD = McGinley Dynamic value
        - N = Period length
        - k = Constant (0.6 in TradingView)

        Args:
            length: Lookback period (default 14)
            k: Speed adjustment constant (default 0.6, TradingView uses 0.6)

        Returns:
            pd.Series: McGinley Dynamic values
        """
        close = self.df['close'].values
        n = len(close)

        md = np.full(n, np.nan)
        md[0] = close[0]

        for i in range(1, n):
            if md[i-1] == 0 or np.isnan(md[i-1]):
                md[i] = close[i]
            else:
                # McGinley Dynamic formula
                ratio = close[i] / md[i-1]
                # Prevent extreme ratios from causing numerical issues
                ratio = max(0.5, min(ratio, 2.0))
                divisor = k * length * (ratio ** 4)
                # Ensure divisor is not zero
                divisor = max(divisor, 0.001)
                md[i] = md[i-1] + (close[i] - md[i-1]) / divisor

        return pd.Series(md, index=self.df.index, name=f'McGinley_{length}')

    def mcginley_tradingview(self, length: int = 14) -> pd.Series:
        """McGinley Dynamic using TradingView-compatible calculation"""
        return self.mcginley_dynamic(length, k=0.6)

    def mcginley_native(self, length: int = 14) -> pd.Series:
        """McGinley Dynamic - same implementation for both engines"""
        return self.mcginley_dynamic(length, k=0.6)

    def mcginley_direction(self, length: int = 14) -> pd.Series:
        """
        Get McGinley Dynamic trend direction.

        Returns:
            pd.Series: 1 for rising (bullish), -1 for falling (bearish), 0 for flat
        """
        md = self.mcginley_dynamic(length)
        direction = np.zeros(len(md))

        for i in range(1, len(md)):
            if pd.isna(md.iloc[i]) or pd.isna(md.iloc[i-1]):
                direction[i] = 0
            elif md.iloc[i] > md.iloc[i-1]:
                direction[i] = 1  # Rising - bullish
            elif md.iloc[i] < md.iloc[i-1]:
                direction[i] = -1  # Falling - bearish
            else:
                direction[i] = direction[i-1]  # Maintain previous

        return pd.Series(direction, index=self.df.index, name='McGinley_Direction')

    # =========================================================================
    # Aroon
    # =========================================================================

    def aroon_tradingview(self, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Aroon indicator using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        n = len(self.df)

        aroon_up = np.full(n, np.nan)
        aroon_down = np.full(n, np.nan)

        for i in range(length, n):
            high_window = high.iloc[i-length:i+1].values
            low_window = low.iloc[i-length:i+1].values

            bars_since_high = length - np.argmax(high_window[::-1])
            bars_since_low = length - np.argmin(low_window[::-1])

            aroon_up[i] = 100 * (length - bars_since_high + 1) / length
            aroon_down[i] = 100 * (length - bars_since_low + 1) / length

        aroon_up_series = pd.Series(aroon_up, index=self.df.index, name='Aroon_Up')
        aroon_down_series = pd.Series(aroon_down, index=self.df.index, name='Aroon_Down')
        aroon_osc = aroon_up_series - aroon_down_series

        return aroon_up_series, aroon_down_series, aroon_osc

    # =========================================================================
    # Parabolic SAR
    # =========================================================================

    def psar_tradingview(self, start: float = 0.02, increment: float = 0.02,
                         maximum: float = 0.2) -> pd.Series:
        """Parabolic SAR using TradingView's method"""
        high = self.df['high'].values
        low = self.df['low'].values
        n = len(high)

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

    # =========================================================================
    # Keltner Channels
    # =========================================================================

    def keltner_tradingview(self, length: int = 20, mult: float = 2.0,
                            atr_length: int = 10) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels using TradingView's method"""
        close = self.df['close']
        basis = close.ewm(span=length, adjust=False).mean()
        atr = self.atr_tradingview(atr_length)
        upper = basis + mult * atr
        lower = basis - mult * atr
        return basis, upper, lower

    # =========================================================================
    # Donchian Channels
    # =========================================================================

    def donchian_tradingview(self, length: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        upper = high.rolling(window=length).max()
        lower = low.rolling(window=length).min()
        basis = (upper + lower) / 2
        return basis, upper, lower

    # =========================================================================
    # Ichimoku Cloud
    # =========================================================================

    def ichimoku_tradingview(self, conversion: int = 9, base: int = 26,
                             span_b: int = 52) -> Dict[str, pd.Series]:
        """Ichimoku Cloud using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        def donchian(length):
            return (high.rolling(window=length).max() + low.rolling(window=length).min()) / 2

        tenkan = donchian(conversion)
        kijun = donchian(base)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = donchian(span_b)
        chikou = close.shift(-base)

        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }

    # =========================================================================
    # Ultimate Oscillator
    # =========================================================================

    def uo_tradingview(self, fast: int = 7, medium: int = 14, slow: int = 28) -> pd.Series:
        """Ultimate Oscillator using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        prev_close = close.shift(1)

        tl = pd.concat([low, prev_close], axis=1).min(axis=1)
        bp = close - tl

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        bp_fast = bp.rolling(window=fast).sum()
        tr_fast = tr.rolling(window=fast).sum()
        bp_medium = bp.rolling(window=medium).sum()
        tr_medium = tr.rolling(window=medium).sum()
        bp_slow = bp.rolling(window=slow).sum()
        tr_slow = tr.rolling(window=slow).sum()

        avg1 = bp_fast / tr_fast.replace(0, np.nan)
        avg2 = bp_medium / tr_medium.replace(0, np.nan)
        avg3 = bp_slow / tr_slow.replace(0, np.nan)

        return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

    # =========================================================================
    # Choppiness Index
    # =========================================================================

    def chop_tradingview(self, length: int = 14) -> pd.Series:
        """Choppiness Index using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_sum = tr.rolling(window=length).sum()
        highest = high.rolling(window=length).max()
        lowest = low.rolling(window=length).min()
        hl_range = highest - lowest

        return 100 * np.log10(atr_sum / hl_range.replace(0, np.nan)) / np.log10(length)

    # =========================================================================
    # VWAP
    # =========================================================================

    def vwap_tradingview(self) -> pd.Series:
        """VWAP using TradingView's method"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))

        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum()

    # =========================================================================
    # VWMA (Volume Weighted Moving Average)
    # =========================================================================

    def vwma_tradingview(self, length: int = 20) -> pd.Series:
        """
        Volume Weighted Moving Average - matches ta.vwma()
        VWMA = SMA(close * volume, length) / SMA(volume, length)

        Weights each bar's price by its volume - high volume bars have more influence.
        """
        close = self.df['close']
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))

        # VWMA formula: SMA(close * volume) / SMA(volume)
        return (close * volume).rolling(window=length).mean() / volume.rolling(window=length).mean()

    # =========================================================================
    # OBV (On Balance Volume)
    # =========================================================================

    def obv_tradingview(self) -> pd.Series:
        """
        On Balance Volume - matches ta.obv()
        OBV adds volume on up days and subtracts on down days.
        """
        close = self.df['close']
        volume = self.df.get('volume', pd.Series(1, index=self.df.index))

        # Calculate OBV: +volume if close > prev, -volume if close < prev, 0 if unchanged
        close_diff = close.diff()
        obv_change = np.where(close_diff > 0, volume,
                     np.where(close_diff < 0, -volume, 0))
        return pd.Series(obv_change, index=self.df.index).cumsum()

    def obv_native(self) -> pd.Series:
        """OBV using TA-Lib if available, otherwise pandas calculation"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.OBV(self.df['close'].values, self.df['volume'].values),
                index=self.df.index
            )
        return self.obv_tradingview()

    # =========================================================================
    # CANDLESTICK PATTERNS - Native engine only
    # =========================================================================

    def detect_pattern(self, pattern_key: str) -> pd.Series:
        """
        Detect a specific candlestick pattern.

        Args:
            pattern_key: Key from CANDLESTICK_PATTERNS dict

        Returns:
            Series with values: 100 (bullish signal), -100 (bearish signal), 0 (no pattern)
        """
        if not TALIB_AVAILABLE:
            return pd.Series(0, index=self.df.index, name=pattern_key)

        if pattern_key not in CANDLESTICK_PATTERNS:
            raise ValueError(f"Unknown pattern: {pattern_key}")

        pattern_info = CANDLESTICK_PATTERNS[pattern_key]
        func_name = pattern_info['func']

        # Get the TA-Lib function
        func = getattr(talib, func_name, None)
        if func is None:
            return pd.Series(0, index=self.df.index, name=pattern_key)

        # Call the pattern function
        result = func(self.open, self.high, self.low, self.close)

        # For engulfing patterns, filter by type
        if 'ENGULFING' in pattern_key:
            if 'BULL' in pattern_key:
                result = np.where(result > 0, result, 0)
            elif 'BEAR' in pattern_key:
                result = np.where(result < 0, result, 0)

        # For harami patterns, filter by type
        if 'HARAMI' in pattern_key:
            if 'BULLISH' in pattern_key:
                result = np.where(result > 0, result, 0)
            elif 'BEARISH' in pattern_key:
                result = np.where(result < 0, result, 0)

        return pd.Series(result, index=self.df.index, name=pattern_key)

    def detect_all_patterns(self) -> pd.DataFrame:
        """
        Detect all candlestick patterns.

        Returns:
            DataFrame with columns for each pattern
        """
        if not TALIB_AVAILABLE:
            return pd.DataFrame(index=self.df.index)

        patterns = {}
        for pattern_key in CANDLESTICK_PATTERNS:
            patterns[pattern_key] = self.detect_pattern(pattern_key)

        return pd.DataFrame(patterns)

    def get_bullish_patterns(self) -> List[str]:
        """Get list of bullish pattern keys"""
        return [k for k, v in CANDLESTICK_PATTERNS.items() if v['type'] == 'bullish']

    def get_bearish_patterns(self) -> List[str]:
        """Get list of bearish pattern keys"""
        return [k for k, v in CANDLESTICK_PATTERNS.items() if v['type'] == 'bearish']

    # =========================================================================
    # HILBERT TRANSFORM - Native engine only
    # =========================================================================

    def hilbert_dominant_cycle(self) -> pd.Series:
        """
        Hilbert Transform - Dominant Cycle Period.
        Identifies the dominant cycle in the price data.
        """
        if not TALIB_AVAILABLE:
            return pd.Series(np.nan, index=self.df.index, name='HT_DCPERIOD')

        result = talib.HT_DCPERIOD(self.close)
        return pd.Series(result, index=self.df.index, name='HT_DCPERIOD')

    def hilbert_dominant_phase(self) -> pd.Series:
        """Hilbert Transform - Dominant Cycle Phase"""
        if not TALIB_AVAILABLE:
            return pd.Series(np.nan, index=self.df.index, name='HT_DCPHASE')

        result = talib.HT_DCPHASE(self.close)
        return pd.Series(result, index=self.df.index, name='HT_DCPHASE')

    def hilbert_phasor(self) -> Tuple[pd.Series, pd.Series]:
        """Hilbert Transform - Phasor Components (inphase, quadrature)"""
        if not TALIB_AVAILABLE:
            return (pd.Series(np.nan, index=self.df.index, name='HT_INPHASE'),
                    pd.Series(np.nan, index=self.df.index, name='HT_QUADRATURE'))

        inphase, quadrature = talib.HT_PHASOR(self.close)
        return (pd.Series(inphase, index=self.df.index, name='HT_INPHASE'),
                pd.Series(quadrature, index=self.df.index, name='HT_QUADRATURE'))

    def hilbert_sine(self) -> Tuple[pd.Series, pd.Series]:
        """Hilbert Transform - SineWave (sine, leadsine)"""
        if not TALIB_AVAILABLE:
            return (pd.Series(np.nan, index=self.df.index, name='HT_SINE'),
                    pd.Series(np.nan, index=self.df.index, name='HT_LEADSINE'))

        sine, leadsine = talib.HT_SINE(self.close)
        return (pd.Series(sine, index=self.df.index, name='HT_SINE'),
                pd.Series(leadsine, index=self.df.index, name='HT_LEADSINE'))

    def hilbert_trendmode(self) -> pd.Series:
        """Hilbert Transform - Trend vs Cycle Mode (1 = trend, 0 = cycle)"""
        if not TALIB_AVAILABLE:
            return pd.Series(np.nan, index=self.df.index, name='HT_TRENDMODE')

        result = talib.HT_TRENDMODE(self.close)
        return pd.Series(result, index=self.df.index, name='HT_TRENDMODE')

    # =========================================================================
    # Additional Native-only indicators
    # =========================================================================

    def kama_native(self, length: int = 30) -> pd.Series:
        """Kaufman Adaptive Moving Average - TA-Lib only"""
        if TALIB_AVAILABLE:
            result = talib.KAMA(self.close, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='KAMA')
        else:
            return self.ema_tradingview(length)

    def t3_native(self, length: int = 5, vfactor: float = 0.7) -> pd.Series:
        """Triple Exponential Moving Average (T3) - TA-Lib only"""
        if TALIB_AVAILABLE:
            result = talib.T3(self.close, timeperiod=length, vfactor=vfactor)
            return pd.Series(result, index=self.df.index, name='T3')
        else:
            return self.ema_tradingview(length)

    def linearreg_native(self, length: int = 14) -> pd.Series:
        """Linear Regression - TA-Lib only"""
        if TALIB_AVAILABLE:
            result = talib.LINEARREG(self.close, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='LINEARREG')
        else:
            return self.sma_tradingview(length)

    def linearreg_slope_native(self, length: int = 14) -> pd.Series:
        """Linear Regression Slope - TA-Lib only"""
        if TALIB_AVAILABLE:
            result = talib.LINEARREG_SLOPE(self.close, timeperiod=length)
            return pd.Series(result, index=self.df.index, name='LINEARREG_SLOPE')
        else:
            return pd.Series(0, index=self.df.index, name='LINEARREG_SLOPE')

    # =========================================================================
    # Kalman Filter
    # =========================================================================

    def kalman_filter(self, source: str = 'close', gain: float = 0.7) -> pd.Series:
        """
        Kalman Filter - Adaptive smoothing filter with velocity tracking.

        The Kalman Filter is a recursive algorithm that estimates the true value
        of a noisy signal. For trading, it provides:
        - Smoother price tracking than moving averages
        - Less lag due to velocity prediction
        - Adaptive response to price changes

        Args:
            source: Price column to filter ('close', 'high', 'low', 'open')
            gain: Kalman gain (0.0-1.0), higher = more responsive, lower = smoother

        Returns:
            pd.Series: Kalman filtered values
        """
        src = self.df[source].values
        n = len(src)
        kf = np.zeros(n)
        kf[0] = src[0]

        velocity = 0.0
        for i in range(1, n):
            # Predict step
            prediction = kf[i-1] + velocity
            # Update step
            error = src[i] - prediction
            kf[i] = prediction + gain * error
            velocity = velocity + gain * error

        return pd.Series(kf, index=self.df.index, name='kalman')

    def kalman_tradingview(self, gain: float = 0.7) -> pd.Series:
        """Kalman Filter - TradingView compatible"""
        return self.kalman_filter('close', gain)

    def kalman_native(self, gain: float = 0.7) -> pd.Series:
        """Kalman Filter - Native implementation (same algorithm)"""
        return self.kalman_filter('close', gain)

    def kalman_bands(self, length: int = 20, mult: float = 2.0, gain: float = 0.7) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Kalman-based Bollinger Bands.

        Uses Kalman filter as the center line instead of SMA,
        with standard deviation bands.

        Args:
            length: Period for standard deviation calculation
            mult: Band multiplier (default 2.0)
            gain: Kalman gain for center line

        Returns:
            Tuple of (basis, upper_band, lower_band)
        """
        basis = self.kalman_filter('close', gain)
        std = self.df['close'].rolling(length).std()
        upper = basis + std * mult
        lower = basis - std * mult

        upper.name = 'kalman_upper'
        lower.name = 'kalman_lower'

        return basis, upper, lower

    def kalman_bands_tradingview(self, length: int = 20, mult: float = 2.0, gain: float = 0.7) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Kalman Bands - TradingView compatible"""
        return self.kalman_bands(length, mult, gain)

    def kalman_bands_native(self, length: int = 20, mult: float = 2.0, gain: float = 0.7) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Kalman Bands - Native implementation"""
        return self.kalman_bands(length, mult, gain)

    def kalman_smooth(self, series: pd.Series, gain: float = 0.5) -> pd.Series:
        """
        Apply Kalman smoothing to any indicator series.

        This is a simplified Kalman filter (without velocity) that can be
        applied to any oscillator or indicator to reduce noise.

        Args:
            series: Input indicator series (e.g., RSI, MFI, ADX)
            gain: Smoothing factor (0.0-1.0), lower = smoother

        Returns:
            pd.Series: Smoothed indicator values
        """
        src = series.values
        n = len(src)
        kf = np.zeros(n)

        # Initialize with first valid value
        first_valid_idx = 0
        for i in range(n):
            if not np.isnan(src[i]):
                kf[i] = src[i]
                first_valid_idx = i
                break

        # Apply Kalman smoothing
        for i in range(first_valid_idx + 1, n):
            if np.isnan(src[i]):
                kf[i] = kf[i-1]  # Hold previous value if NaN
            else:
                kf[i] = kf[i-1] + gain * (src[i] - kf[i-1])

        return pd.Series(kf, index=series.index, name=f'kalman_{series.name}')

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _rma(self, series: pd.Series, length: int, alpha: float) -> pd.Series:
        """RMA (Wilder's smoothing)"""
        result = series.copy()
        result.iloc[:length] = series.iloc[:length].expanding().mean()

        for i in range(length, len(series)):
            result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]

        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_indicators(df: pd.DataFrame, engine: IndicatorEngine = IndicatorEngine.NATIVE) -> pd.DataFrame:
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
        # Core indicators - TradingView compatible
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

        result['willr'] = calc.willr_tradingview()
        result['mom'] = calc.mom_tradingview()
        result['mfi'] = calc.mfi_tradingview()
        result['cci'] = calc.cci_tradingview()
        result['roc'] = calc.roc_tradingview()

        adx, plus_di, minus_di = calc.adx_tradingview()
        result['adx'] = adx
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di

        supertrend, supertrend_dir = calc.supertrend_tradingview()
        result['supertrend'] = supertrend
        result['supertrend_dir'] = supertrend_dir

        # McGinley Dynamic
        result['mcginley'] = calc.mcginley_tradingview()
        result['mcginley_direction'] = calc.mcginley_direction()

        # Kalman Filter
        result['kalman'] = calc.kalman_tradingview()
        kalman_basis, kalman_upper, kalman_lower = calc.kalman_bands_tradingview()
        result['kalman_upper'] = kalman_upper
        result['kalman_lower'] = kalman_lower

        # Kalman-smoothed indicators
        result['kalman_rsi'] = calc.kalman_smooth(result['rsi'])
        result['kalman_mfi'] = calc.kalman_smooth(result['mfi'])
        result['kalman_adx'] = calc.kalman_smooth(result['adx'])
        result['kalman_macd'] = calc.kalman_smooth(result['macd'])
        result['kalman_macd_signal'] = calc.kalman_smooth(result['macd_signal'])

    elif engine == IndicatorEngine.NATIVE:
        # Core indicators - TA-Lib (fast)
        result['rsi'] = calc.rsi_native()
        result['sma_20'] = calc.sma_native(20)
        result['sma_50'] = calc.sma_native(50)
        result['ema_9'] = calc.ema_native(9)
        result['ema_21'] = calc.ema_native(21)
        result['atr'] = calc.atr_native()

        macd, signal, hist = calc.macd_native()
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist

        bb_mid, bb_upper, bb_lower = calc.bbands_native()
        result['bb_mid'] = bb_mid
        result['bb_upper'] = bb_upper
        result['bb_lower'] = bb_lower

        stoch_k, stoch_d = calc.stoch_native()
        result['stoch_k'] = stoch_k
        result['stoch_d'] = stoch_d

        result['willr'] = calc.willr_native()
        result['mom'] = calc.mom_native()
        result['mfi'] = calc.mfi_native()
        result['cci'] = calc.cci_native()
        result['roc'] = calc.roc_native()

        adx, plus_di, minus_di = calc.adx_native()
        result['adx'] = adx
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di

        supertrend, supertrend_dir = calc.supertrend_native()
        result['supertrend'] = supertrend
        result['supertrend_dir'] = supertrend_dir

        # McGinley Dynamic
        result['mcginley'] = calc.mcginley_native()
        result['mcginley_direction'] = calc.mcginley_direction()

        # Kalman Filter
        result['kalman'] = calc.kalman_native()
        kalman_basis, kalman_upper, kalman_lower = calc.kalman_bands_native()
        result['kalman_upper'] = kalman_upper
        result['kalman_lower'] = kalman_lower

        # Kalman-smoothed indicators
        result['kalman_rsi'] = calc.kalman_smooth(result['rsi'])
        result['kalman_mfi'] = calc.kalman_smooth(result['mfi'])
        result['kalman_adx'] = calc.kalman_smooth(result['adx'])
        result['kalman_macd'] = calc.kalman_smooth(result['macd'])
        result['kalman_macd_signal'] = calc.kalman_smooth(result['macd_signal'])

        # Native-only: Advanced indicators
        result['kama'] = calc.kama_native()
        result['t3'] = calc.t3_native()
        result['linearreg'] = calc.linearreg_native()
        result['linearreg_slope'] = calc.linearreg_slope_native()

        # Native-only: Hilbert Transform
        result['ht_dcperiod'] = calc.hilbert_dominant_cycle()
        result['ht_trendmode'] = calc.hilbert_trendmode()
        sine, leadsine = calc.hilbert_sine()
        result['ht_sine'] = sine
        result['ht_leadsine'] = leadsine

        # Native-only: Candlestick patterns
        patterns = calc.detect_all_patterns()
        for col in patterns.columns:
            result[col.lower()] = patterns[col]

    return result


def get_available_patterns() -> Dict[str, Dict]:
    """Get all available candlestick patterns with metadata"""
    return CANDLESTICK_PATTERNS.copy()


def is_native_available() -> bool:
    """Check if TA-Lib is available for native execution"""
    return TALIB_AVAILABLE
