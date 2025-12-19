"""
Multi-Timeframe (MTF) Analysis Module

This module calculates trend alignment across multiple timeframes
to provide confluence scoring for trading decisions.

Based on the user's prompt specifications:
- 80%+ alignment: +2 to score
- 60-80% alignment: +1 to score
- <40% alignment: -1 to score (low confluence)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Try to import TA-Lib, fall back to manual calculation if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


@dataclass
class TimeframeAnalysis:
    """Analysis results for a single timeframe."""
    timeframe: str
    trend: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: int  # 0-10
    rsi: float
    price_vs_ema20: str  # 'above', 'below', 'at'
    price_vs_ema50: str
    price_vs_ema200: str
    ema_stack: str  # 'bullish', 'bearish', 'mixed'
    macd_signal: str  # 'bullish', 'bearish', 'neutral'
    adx: float
    volume_trend: str  # 'increasing', 'decreasing', 'neutral'


@dataclass
class MTFConfluence:
    """Multi-timeframe confluence results."""
    alignment_percent: float
    direction: str  # 'bullish', 'bearish', 'mixed'
    score_adjustment: int
    timeframe_analyses: Dict[str, TimeframeAnalysis]
    strongest_timeframe: str
    weakest_timeframe: str
    recommendation: str


class MTFAnalyzer:
    """
    Multi-Timeframe Analyzer for trend confluence.

    Analyzes trend alignment across multiple timeframes to determine
    overall market direction and confluence strength.
    """

    # Timeframe weights (higher timeframes have more weight)
    TIMEFRAME_WEIGHTS = {
        '1m': 0.5,
        '5m': 1.0,
        '15m': 1.5,
        '30m': 2.0,
        '1h': 2.5,
        '4h': 3.0,
        '1d': 3.5,
        '1w': 4.0,
    }

    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Initialize with multiple timeframe DataFrames.

        Args:
            dataframes: Dict mapping timeframe strings to OHLCV DataFrames
                       e.g., {'5m': df_5m, '1h': df_1h, '4h': df_4h}
        """
        self.dataframes = dataframes
        self.analyses = {}

    def analyze_timeframe(self, tf: str, df: pd.DataFrame) -> TimeframeAnalysis:
        """
        Analyze a single timeframe.

        Args:
            tf: Timeframe string (e.g., '5m', '1h')
            df: OHLCV DataFrame for that timeframe

        Returns:
            TimeframeAnalysis object with all indicators
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Calculate indicators
        if TALIB_AVAILABLE and len(close) >= 200:
            ema20 = talib.EMA(close, timeperiod=20)
            ema50 = talib.EMA(close, timeperiod=50)
            ema200 = talib.EMA(close, timeperiod=200)
            rsi = talib.RSI(close, timeperiod=14)
            adx = talib.ADX(high, low, close, timeperiod=14)
            macd, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        else:
            # Fallback to pandas calculations
            ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
            ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
            ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().values if len(close) >= 200 else ema50
            rsi = self._calculate_rsi(close, 14)
            adx = self._calculate_adx_simple(high, low, close, 14)
            macd, signal = self._calculate_macd(close)

        current_close = close[-1]
        current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
        current_adx = adx[-1] if not np.isnan(adx[-1]) else 20
        current_ema20 = ema20[-1] if not np.isnan(ema20[-1]) else current_close
        current_ema50 = ema50[-1] if not np.isnan(ema50[-1]) else current_close
        current_ema200 = ema200[-1] if not np.isnan(ema200[-1]) else current_close

        # Determine price vs EMA positions
        price_vs_ema20 = 'above' if current_close > current_ema20 else ('below' if current_close < current_ema20 else 'at')
        price_vs_ema50 = 'above' if current_close > current_ema50 else ('below' if current_close < current_ema50 else 'at')
        price_vs_ema200 = 'above' if current_close > current_ema200 else ('below' if current_close < current_ema200 else 'at')

        # Determine EMA stack
        if current_ema20 > current_ema50 > current_ema200:
            ema_stack = 'bullish'
        elif current_ema20 < current_ema50 < current_ema200:
            ema_stack = 'bearish'
        else:
            ema_stack = 'mixed'

        # MACD signal
        current_macd = macd[-1] if not np.isnan(macd[-1]) else 0
        current_signal = signal[-1] if not np.isnan(signal[-1]) else 0
        if current_macd > current_signal:
            macd_sig = 'bullish'
        elif current_macd < current_signal:
            macd_sig = 'bearish'
        else:
            macd_sig = 'neutral'

        # Volume trend
        if 'volume' in df.columns and len(df) >= 20:
            vol = df['volume'].values
            vol_ma = pd.Series(vol).rolling(20).mean().values
            current_vol_ma = vol_ma[-1] if not np.isnan(vol_ma[-1]) else vol[-1]
            if vol[-1] > current_vol_ma * 1.2:
                vol_trend = 'increasing'
            elif vol[-1] < current_vol_ma * 0.8:
                vol_trend = 'decreasing'
            else:
                vol_trend = 'neutral'
        else:
            vol_trend = 'neutral'

        # Overall trend determination
        bullish_signals = 0
        bearish_signals = 0

        if price_vs_ema20 == 'above':
            bullish_signals += 1
        elif price_vs_ema20 == 'below':
            bearish_signals += 1

        if price_vs_ema50 == 'above':
            bullish_signals += 1
        elif price_vs_ema50 == 'below':
            bearish_signals += 1

        if ema_stack == 'bullish':
            bullish_signals += 2
        elif ema_stack == 'bearish':
            bearish_signals += 2

        if macd_sig == 'bullish':
            bullish_signals += 1
        elif macd_sig == 'bearish':
            bearish_signals += 1

        if current_rsi > 50:
            bullish_signals += 1
        elif current_rsi < 50:
            bearish_signals += 1

        if bullish_signals > bearish_signals + 2:
            trend = 'bullish'
        elif bearish_signals > bullish_signals + 2:
            trend = 'bearish'
        else:
            trend = 'neutral'

        # Trend strength (0-10)
        trend_strength = min(10, int(current_adx / 5)) if current_adx else 5

        return TimeframeAnalysis(
            timeframe=tf,
            trend=trend,
            trend_strength=trend_strength,
            rsi=round(current_rsi, 2),
            price_vs_ema20=price_vs_ema20,
            price_vs_ema50=price_vs_ema50,
            price_vs_ema200=price_vs_ema200,
            ema_stack=ema_stack,
            macd_signal=macd_sig,
            adx=round(current_adx, 2),
            volume_trend=vol_trend
        )

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI using pandas."""
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values

    def _calculate_adx_simple(self, high: np.ndarray, low: np.ndarray,
                              close: np.ndarray, period: int = 14) -> np.ndarray:
        """Simplified ADX calculation."""
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
        atr = pd.Series(tr).rolling(period).mean().values
        # Simplified - return ATR as percentage of price as proxy for trend strength
        return (atr / close) * 100 * 10  # Scale to roughly 0-50 range

    def _calculate_macd(self, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD using pandas."""
        exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd_line = (exp1 - exp2).values
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        return macd_line, signal_line

    def calculate_confluence(self) -> MTFConfluence:
        """
        Calculate multi-timeframe confluence.

        Returns:
            MTFConfluence object with alignment and recommendations
        """
        # Analyze each timeframe
        for tf, df in self.dataframes.items():
            if len(df) >= 50:  # Need minimum data
                self.analyses[tf] = self.analyze_timeframe(tf, df)

        if not self.analyses:
            return MTFConfluence(
                alignment_percent=0,
                direction='neutral',
                score_adjustment=0,
                timeframe_analyses={},
                strongest_timeframe='',
                weakest_timeframe='',
                recommendation='Insufficient data for MTF analysis'
            )

        # Calculate weighted alignment
        total_weight = 0
        bullish_weight = 0
        bearish_weight = 0

        strongest_tf = None
        strongest_strength = 0
        weakest_tf = None
        weakest_strength = 11

        for tf, analysis in self.analyses.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 1.0)
            total_weight += weight

            if analysis.trend == 'bullish':
                bullish_weight += weight
            elif analysis.trend == 'bearish':
                bearish_weight += weight

            if analysis.trend_strength > strongest_strength:
                strongest_strength = analysis.trend_strength
                strongest_tf = tf

            if analysis.trend_strength < weakest_strength:
                weakest_strength = analysis.trend_strength
                weakest_tf = tf

        # Calculate alignment percentage
        if total_weight > 0:
            dominant_weight = max(bullish_weight, bearish_weight)
            alignment = (dominant_weight / total_weight) * 100
        else:
            alignment = 0

        # Determine direction
        if bullish_weight > bearish_weight * 1.2:
            direction = 'bullish'
        elif bearish_weight > bullish_weight * 1.2:
            direction = 'bearish'
        else:
            direction = 'mixed'

        # Calculate score adjustment based on alignment
        # Per user's prompt:
        # - 80%+ alignment: +2 to score
        # - 60-80% alignment: +1 to score
        # - <40% alignment: -1 to score
        if alignment >= 80:
            score_adjustment = 2
        elif alignment >= 60:
            score_adjustment = 1
        elif alignment < 40:
            score_adjustment = -1
        else:
            score_adjustment = 0

        # Generate recommendation
        if alignment >= 80:
            rec = f"Strong {direction} confluence ({alignment:.0f}%). High confidence in direction."
        elif alignment >= 60:
            rec = f"Moderate {direction} confluence ({alignment:.0f}%). Proceed with caution."
        elif alignment >= 40:
            rec = f"Mixed signals ({alignment:.0f}%). No clear directional bias."
        else:
            rec = f"Low confluence ({alignment:.0f}%). Avoid directional trades."

        return MTFConfluence(
            alignment_percent=round(alignment, 1),
            direction=direction,
            score_adjustment=score_adjustment,
            timeframe_analyses=self.analyses,
            strongest_timeframe=strongest_tf or '',
            weakest_timeframe=weakest_tf or '',
            recommendation=rec
        )


def resample_to_timeframe(df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
    """
    Resample OHLCV data from source timeframe to target timeframe.

    Args:
        df: Source DataFrame with time, open, high, low, close, volume
        source_tf: Source timeframe (e.g., '5m')
        target_tf: Target timeframe (e.g., '1h')

    Returns:
        Resampled DataFrame
    """
    # Map timeframe strings to pandas offset aliases
    tf_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '4h': '4h',
        '1d': '1D',
        '1w': '1W',
    }

    target_offset = tf_map.get(target_tf, target_tf)

    # Ensure datetime index
    if 'time' in df.columns:
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    # Resample
    resampled = df.resample(target_offset).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum' if 'volume' in df.columns else 'first'
    }).dropna()

    resampled.reset_index(inplace=True)

    return resampled


def create_mtf_dataframes(base_df: pd.DataFrame, base_tf: str = '5m',
                          target_tfs: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Create multi-timeframe DataFrames from a base DataFrame.

    Args:
        base_df: Base OHLCV DataFrame
        base_tf: Base timeframe of the data
        target_tfs: List of target timeframes to create

    Returns:
        Dictionary of timeframe -> DataFrame
    """
    if target_tfs is None:
        target_tfs = ['15m', '1h', '4h']

    # Timeframe ordering (smallest to largest)
    tf_order = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']

    # Get index of base timeframe
    try:
        base_idx = tf_order.index(base_tf)
    except ValueError:
        base_idx = 1  # Default to 5m if unknown

    dataframes = {base_tf: base_df.copy()}

    for tf in target_tfs:
        try:
            tf_idx = tf_order.index(tf)
        except ValueError:
            continue

        # Only resample to larger timeframes
        if tf_idx > base_idx:
            resampled = resample_to_timeframe(base_df, base_tf, tf)
            if len(resampled) >= 50:
                dataframes[tf] = resampled

    return dataframes


# Convenience function for strategy use
def calculate_mtf_score_adjustment(base_df: pd.DataFrame, base_tf: str = '5m') -> Tuple[int, str]:
    """
    Quick calculation of MTF score adjustment.

    Args:
        base_df: Base OHLCV DataFrame
        base_tf: Base timeframe

    Returns:
        Tuple of (score_adjustment, direction)
    """
    try:
        dataframes = create_mtf_dataframes(base_df, base_tf, ['15m', '1h', '4h'])
        analyzer = MTFAnalyzer(dataframes)
        confluence = analyzer.calculate_confluence()
        return confluence.score_adjustment, confluence.direction
    except Exception as e:
        print(f"MTF analysis error: {e}")
        return 0, 'neutral'
