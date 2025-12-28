"""
ENTRY SIGNALS
=============
Strategy entry signal generation using a registry pattern.
Replaces the giant if/elif chain in strategy_engine.py.

Each strategy is registered with a decorator and can be looked up by name.
"""
import pandas as pd
import numpy as np
from typing import Callable, Dict

# Signal registry - maps strategy names to signal functions
SIGNAL_REGISTRY: Dict[str, Callable] = {}


def register_signal(name: str):
    """Decorator to register a signal function."""
    def decorator(func: Callable):
        SIGNAL_REGISTRY[name] = func
        return func
    return decorator


def safe_bool(series: pd.Series) -> pd.Series:
    """Ensure boolean series with no NaN values."""
    return series.fillna(False).astype(bool)


def safe_col(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Safely get a column with NaN instead of None."""
    if col_name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col_name], errors='coerce')


# =============================================================================
# MOMENTUM STRATEGIES
# =============================================================================

@register_signal('always')
def always_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Always signal - enters on every bar."""
    return pd.Series(True, index=df.index)


@register_signal('rsi_extreme')
def rsi_extreme_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """
    RSI Strategy: crossover/crossunder through overbought/oversold levels.
    Long: RSI crosses UP through oversold (30)
    Short: RSI crosses DOWN through overbought (70)
    """
    if direction == 'long':
        return safe_bool((df['rsi'] > 30) & (df['rsi'].shift(1) <= 30))
    else:
        return safe_bool((df['rsi'] < 70) & (df['rsi'].shift(1) >= 70))


@register_signal('rsi_cross_50')
def rsi_cross_50_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """RSI crosses 50 level."""
    if direction == 'long':
        return safe_bool((df['rsi'] > 50) & (df['rsi'].shift(1) <= 50))
    else:
        return safe_bool((df['rsi'] < 50) & (df['rsi'].shift(1) >= 50))


@register_signal('stoch_extreme')
def stoch_extreme_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """
    Stochastic Slow Strategy (TradingView pattern).
    Long: %K crosses OVER %D AND current K < 20
    Short: %K crosses UNDER %D AND current K > 80
    """
    if direction == 'long':
        k_cross_d_over = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        return safe_bool(k_cross_d_over & (df['stoch_k'] < 20))
    else:
        k_cross_d_under = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        return safe_bool(k_cross_d_under & (df['stoch_k'] > 80))


# =============================================================================
# VOLATILITY STRATEGIES
# =============================================================================

@register_signal('bb_touch')
def bb_touch_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """
    Bollinger Bands Strategy: price crosses through bands.
    Long: Price crosses UP through lower band
    Short: Price crosses DOWN through upper band
    """
    if direction == 'long':
        return safe_bool((df['close'] > df['bb_lower']) & (df['close'].shift(1) <= df['bb_lower'].shift(1)))
    else:
        return safe_bool((df['close'] < df['bb_upper']) & (df['close'].shift(1) >= df['bb_upper'].shift(1)))


@register_signal('bb_squeeze_breakout')
def bb_squeeze_breakout_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Bollinger Band squeeze breakout."""
    squeeze = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8
    expanding = df['bb_width'] > df['bb_width'].shift(1)
    if direction == 'long':
        return safe_bool(squeeze.shift(1) & expanding & (df['close'] > df['bb_mid']))
    else:
        return safe_bool(squeeze.shift(1) & expanding & (df['close'] < df['bb_mid']))


@register_signal('keltner_breakout')
def keltner_breakout_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Keltner Channel breakout."""
    if direction == 'long':
        return safe_bool((df['close'] > df['kc_upper']) & (df['close'].shift(1) <= df['kc_upper'].shift(1)))
    else:
        return safe_bool((df['close'] < df['kc_lower']) & (df['close'].shift(1) >= df['kc_lower'].shift(1)))


@register_signal('donchian_breakout')
def donchian_breakout_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Donchian Channel breakout."""
    if direction == 'long':
        return safe_bool((df['close'] > df['dc_upper']) & (df['close'].shift(1) <= df['dc_upper'].shift(1)))
    else:
        return safe_bool((df['close'] < df['dc_lower']) & (df['close'].shift(1) >= df['dc_lower'].shift(1)))


# =============================================================================
# TREND STRATEGIES
# =============================================================================

@register_signal('price_vs_sma')
def price_vs_sma_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Price vs SMA deviation."""
    if direction == 'long':
        return safe_bool(df['close'] < df['sma_20'] * 0.99)
    else:
        return safe_bool(df['close'] > df['sma_20'] * 1.01)


@register_signal('price_above_sma')
def price_above_sma_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Price crosses above/below SMA."""
    if direction == 'long':
        return safe_bool((df['close'] > df['sma_20']) & (df['close'].shift(1) <= df['sma_20'].shift(1)))
    else:
        return safe_bool((df['close'] < df['sma_20']) & (df['close'].shift(1) >= df['sma_20'].shift(1)))


@register_signal('ema_cross')
def ema_cross_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """EMA crossover (fast crosses slow)."""
    if direction == 'long':
        return safe_bool((df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1)))
    else:
        return safe_bool((df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1)))


@register_signal('sma_cross')
def sma_cross_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """SMA crossover (fast crosses slow)."""
    if 'sma_fast' in df.columns and 'sma_slow' in df.columns:
        sma_fast = safe_col(df, 'sma_fast')
        sma_slow = safe_col(df, 'sma_slow')
    else:
        sma_fast = df['close'].rolling(9).mean()
        sma_slow = df['close'].rolling(18).mean()

    if direction == 'long':
        return safe_bool((sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1)))
    else:
        return safe_bool((sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1)))


@register_signal('macd_cross')
def macd_cross_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """
    MACD Strategy: histogram crosses zero.
    Long: histogram crosses OVER zero
    Short: histogram crosses UNDER zero
    """
    histogram = df['macd'] - df['macd_signal']
    if direction == 'long':
        return safe_bool((histogram > 0) & (histogram.shift(1) <= 0))
    else:
        return safe_bool((histogram < 0) & (histogram.shift(1) >= 0))


@register_signal('supertrend_flip')
def supertrend_flip_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """
    Supertrend direction flip.
    TradingView: 1 = bearish (price below), -1 = bullish (price above)
    """
    if direction == 'long':
        return safe_bool((df['supertrend_dir'] == -1) & (df['supertrend_dir'].shift(1) == 1))
    else:
        return safe_bool((df['supertrend_dir'] == 1) & (df['supertrend_dir'].shift(1) == -1))


@register_signal('psar_flip')
def psar_flip_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Parabolic SAR flip."""
    if direction == 'long':
        return safe_bool((df['close'] > df['psar']) & (df['close'].shift(1) <= df['psar'].shift(1)))
    else:
        return safe_bool((df['close'] < df['psar']) & (df['close'].shift(1) >= df['psar'].shift(1)))


@register_signal('mcginley_cross')
def mcginley_cross_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """McGinley Dynamic crossover."""
    if 'mcginley' not in df.columns:
        return pd.Series(False, index=df.index)
    if direction == 'long':
        return safe_bool((df['close'] > df['mcginley']) & (df['close'].shift(1) <= df['mcginley'].shift(1)))
    else:
        return safe_bool((df['close'] < df['mcginley']) & (df['close'].shift(1) >= df['mcginley'].shift(1)))


@register_signal('mcginley_trend')
def mcginley_trend_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """McGinley Dynamic trend following."""
    if 'mcginley' not in df.columns:
        return pd.Series(False, index=df.index)
    mcg_rising = df['mcginley'] > df['mcginley'].shift(1)
    mcg_falling = df['mcginley'] < df['mcginley'].shift(1)
    if direction == 'long':
        return safe_bool((df['close'] > df['mcginley']) & mcg_rising)
    else:
        return safe_bool((df['close'] < df['mcginley']) & mcg_falling)


# =============================================================================
# ADX / TREND STRENGTH STRATEGIES
# =============================================================================

@register_signal('adx_strong_trend')
def adx_strong_trend_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """ADX strong trend with DI confirmation."""
    strong_trend = df['adx'] > 25
    if direction == 'long':
        return safe_bool(strong_trend & (df['di_plus'] > df['di_minus']))
    else:
        return safe_bool(strong_trend & (df['di_minus'] > df['di_plus']))


@register_signal('adx_breakout')
def adx_breakout_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """ADX crossing 20 with DI confirmation."""
    adx_cross_20 = (df['adx'] > 20) & (df['adx'].shift(1) <= 20)
    if direction == 'long':
        return safe_bool(adx_cross_20 & (df['di_plus'] > df['di_minus']))
    else:
        return safe_bool(adx_cross_20 & (df['di_minus'] > df['di_plus']))


# =============================================================================
# AROON STRATEGIES
# =============================================================================

@register_signal('aroon_cross')
def aroon_cross_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Aroon Up/Down crossover."""
    if direction == 'long':
        return safe_bool((df['aroon_up'] > df['aroon_down']) & (df['aroon_up'].shift(1) <= df['aroon_down'].shift(1)))
    else:
        return safe_bool((df['aroon_down'] > df['aroon_up']) & (df['aroon_down'].shift(1) <= df['aroon_up'].shift(1)))


@register_signal('aroon_extreme')
def aroon_extreme_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Aroon extreme levels."""
    if direction == 'long':
        return safe_bool((df['aroon_up'] > 70) & (df['aroon_down'] < 30))
    else:
        return safe_bool((df['aroon_down'] > 70) & (df['aroon_up'] < 30))


# =============================================================================
# ICHIMOKU STRATEGIES
# =============================================================================

@register_signal('ichimoku_cloud_breakout')
def ichimoku_cloud_breakout_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Price breaks above/below Ichimoku cloud."""
    cloud_top = df[['senkou_a', 'senkou_b']].max(axis=1)
    cloud_bottom = df[['senkou_a', 'senkou_b']].min(axis=1)
    if direction == 'long':
        return safe_bool((df['close'] > cloud_top) & (df['close'].shift(1) <= cloud_top.shift(1)))
    else:
        return safe_bool((df['close'] < cloud_bottom) & (df['close'].shift(1) >= cloud_bottom.shift(1)))


@register_signal('ichimoku_tk_cross')
def ichimoku_tk_cross_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Tenkan-Kijun crossover."""
    if direction == 'long':
        return safe_bool((df['tenkan'] > df['kijun']) & (df['tenkan'].shift(1) <= df['kijun'].shift(1)))
    else:
        return safe_bool((df['tenkan'] < df['kijun']) & (df['tenkan'].shift(1) >= df['kijun'].shift(1)))


# =============================================================================
# OSCILLATOR STRATEGIES
# =============================================================================

@register_signal('cci_extreme')
def cci_extreme_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """CCI crosses through extreme levels."""
    if direction == 'long':
        return safe_bool((df['cci'] > -100) & (df['cci'].shift(1) <= -100))
    else:
        return safe_bool((df['cci'] < 100) & (df['cci'].shift(1) >= 100))


@register_signal('willr_extreme')
def willr_extreme_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Williams %R crosses through extreme levels."""
    if direction == 'long':
        return safe_bool((df['willr'] > -80) & (df['willr'].shift(1) <= -80))
    else:
        return safe_bool((df['willr'] < -20) & (df['willr'].shift(1) >= -20))


@register_signal('uo_extreme')
def uo_extreme_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Ultimate Oscillator extreme levels."""
    if direction == 'long':
        return safe_bool((df['uo'] > 30) & (df['uo'].shift(1) <= 30))
    else:
        return safe_bool((df['uo'] < 70) & (df['uo'].shift(1) >= 70))


@register_signal('mom_cross_zero')
def mom_cross_zero_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Momentum crosses zero."""
    if direction == 'long':
        return safe_bool((df['mom'] > 0) & (df['mom'].shift(1) <= 0))
    else:
        return safe_bool((df['mom'] < 0) & (df['mom'].shift(1) >= 0))


@register_signal('roc_cross_zero')
def roc_cross_zero_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Rate of Change crosses zero."""
    if direction == 'long':
        return safe_bool((df['roc'] > 0) & (df['roc'].shift(1) <= 0))
    else:
        return safe_bool((df['roc'] < 0) & (df['roc'].shift(1) >= 0))


# =============================================================================
# CANDLE PATTERN STRATEGIES
# =============================================================================

@register_signal('doji_reversal')
def doji_reversal_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Doji candle reversal pattern."""
    doji = df['body'] < df['range'] * 0.1
    if direction == 'long':
        return safe_bool(doji & (df['close'].shift(1) < df['open'].shift(1)))
    else:
        return safe_bool(doji & (df['close'].shift(1) > df['open'].shift(1)))


@register_signal('engulfing')
def engulfing_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Engulfing candle pattern."""
    if direction == 'long':
        return safe_bool(
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        )
    else:
        return safe_bool(
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        )


@register_signal('hammer')
def hammer_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Hammer/Shooting Star pattern."""
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)

    if direction == 'long':
        # Hammer: small body, long lower shadow
        return safe_bool((body < total_range * 0.3) & (lower_shadow > body * 2))
    else:
        # Shooting star: small body, long upper shadow
        return safe_bool((body < total_range * 0.3) & (upper_shadow > body * 2))


@register_signal('consecutive_bars')
def consecutive_bars_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Consecutive green/red bars."""
    n_bars = int(df.get('consecutive_bars', pd.Series(3)).iloc[0]) if 'consecutive_bars' in df.columns else 3

    if direction == 'long':
        consecutive_red = (df['close'] < df['open']).rolling(n_bars).sum() == n_bars
        return safe_bool(consecutive_red.shift(1) & (df['close'] > df['open']))
    else:
        consecutive_green = (df['close'] > df['open']).rolling(n_bars).sum() == n_bars
        return safe_bool(consecutive_green.shift(1) & (df['close'] < df['open']))


# =============================================================================
# VOLUME STRATEGIES
# =============================================================================

@register_signal('obv_divergence')
def obv_divergence_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """OBV divergence from price."""
    if 'obv' not in df.columns:
        return pd.Series(False, index=df.index)

    lookback = int(df.get('obv_lookback', pd.Series(5)).iloc[0]) if 'obv_lookback' in df.columns else 5

    price_higher = df['close'] > df['close'].shift(lookback)
    price_lower = df['close'] < df['close'].shift(lookback)
    obv_higher = df['obv'] > df['obv'].shift(lookback)
    obv_lower = df['obv'] < df['obv'].shift(lookback)

    if direction == 'long':
        # Bullish divergence: price lower but OBV higher
        return safe_bool(price_lower & obv_higher)
    else:
        # Bearish divergence: price higher but OBV lower
        return safe_bool(price_higher & obv_lower)


@register_signal('vwap_cross')
def vwap_cross_signal(df: pd.DataFrame, direction: str) -> pd.Series:
    """Price crosses VWAP."""
    if 'vwap' not in df.columns:
        return pd.Series(False, index=df.index)

    if direction == 'long':
        return safe_bool((df['close'] > df['vwap']) & (df['close'].shift(1) <= df['vwap'].shift(1)))
    else:
        return safe_bool((df['close'] < df['vwap']) & (df['close'].shift(1) >= df['vwap'].shift(1)))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def get_signals(df: pd.DataFrame, strategy: str, direction: str) -> pd.Series:
    """
    Get entry signals for a strategy.

    Args:
        df: DataFrame with OHLCV and indicator columns
        strategy: Strategy name (e.g., 'rsi_extreme', 'macd_cross')
        direction: 'long' or 'short'

    Returns:
        Boolean Series with True where entry signals occur
    """
    if strategy not in SIGNAL_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(SIGNAL_REGISTRY.keys())}")

    return SIGNAL_REGISTRY[strategy](df, direction)


def get_available_strategies() -> list:
    """Get list of all available strategy names."""
    return list(SIGNAL_REGISTRY.keys())


def get_strategy_count() -> int:
    """Get count of registered strategies."""
    return len(SIGNAL_REGISTRY)
