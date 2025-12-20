"""
TradingView Built-in Indicators Reference
==========================================
This file contains the EXACT formulas and default settings from TradingView's
built-in indicators. Use this as the authoritative reference for indicator calculations.

Source: TradingView's official indicator templates (December 2025)
Version: Pine Script v6

CRITICAL NOTES FOR MATCHING:
1. TradingView uses ta.rma() (Wilder's smoothing) for RSI, ATR, ADX - NOT SMA or EMA
2. Default parameters may differ between indicators and strategies
3. Stochastic INDICATOR uses smoothK=1, but Stochastic STRATEGY uses smoothK=3
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class IndicatorConfig:
    """Configuration for a TradingView indicator"""
    name: str
    pine_function: str
    default_params: Dict[str, Any]
    formula_notes: str
    python_equivalent: str


# =============================================================================
# CORE INDICATOR CONFIGURATIONS
# =============================================================================

INDICATORS = {
    # =========================================================================
    # RSI - Relative Strength Index
    # =========================================================================
    "rsi": IndicatorConfig(
        name="Relative Strength Index",
        pine_function="ta.rsi(source, length)",
        default_params={
            "length": 14,
            "source": "close",
            "overbought": 70,
            "oversold": 30,
        },
        formula_notes="""
TradingView RSI Formula (lines 3335-3338):
    change = ta.change(rsiSourceInput)
    up = ta.rma(math.max(change, 0), rsiLengthInput)
    down = ta.rma(-math.min(change, 0), rsiLengthInput)
    rsi = down == 0 ? 100 : up == 0 ? 0 : 100 - (100 / (1 + up / down))

KEY: Uses ta.rma() (Wilder's smoothing / RMA), NOT SMA or EMA!
- ta.rma() is equivalent to: alpha = 1/length, result = alpha*current + (1-alpha)*previous
- First value is initialized with SMA of first 'length' values
        """,
        python_equivalent="""
def rsi_tradingview(close, length=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    # RMA (Wilder's smoothing) = exponential moving average with alpha = 1/length
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
        """
    ),

    # =========================================================================
    # BOLLINGER BANDS
    # =========================================================================
    "bollinger_bands": IndicatorConfig(
        name="Bollinger Bands",
        pine_function="ta.bb(source, length, mult)",
        default_params={
            "length": 20,
            "source": "close",
            "mult": 2.0,
            "ma_type": "SMA",  # Options: SMA, EMA, RMA, WMA, VWMA
        },
        formula_notes="""
TradingView BB Formula (lines 860-877):
    basis = ta.sma(src, length)  // or other MA type
    dev = mult * ta.stdev(src, length)
    upper = basis + dev
    lower = basis - dev

KEY: ta.stdev() uses POPULATION std dev (ddof=0), NOT sample std dev (ddof=1)!
        """,
        python_equivalent="""
def bollinger_bands_tradingview(close, length=20, mult=2.0):
    basis = close.rolling(length).mean()
    dev = mult * close.rolling(length).std(ddof=0)  # POPULATION std!
    upper = basis + dev
    lower = basis - dev
    return basis, upper, lower
        """
    ),

    # =========================================================================
    # MACD
    # =========================================================================
    "macd": IndicatorConfig(
        name="Moving Average Convergence Divergence",
        pine_function="ta.macd(source, fast, slow, signal)",
        default_params={
            "fast_length": 12,
            "slow_length": 26,
            "signal_length": 9,
            "source": "close",
            "osc_ma_type": "EMA",  # Options: EMA, SMA
            "sig_ma_type": "EMA",  # Options: EMA, SMA
        },
        formula_notes="""
TradingView MACD Formula (lines 2470-2491):
    maFast = ta.ema(sourceInput, fastLenInput)
    maSlow = ta.ema(sourceInput, slowLenInput)
    macd = maFast - maSlow
    signal = ta.ema(macd, sigLenInput)
    hist = macd - signal

KEY: Both oscillator and signal line use EMA by default (not SMA)
        """,
        python_equivalent="""
def macd_tradingview(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
        """
    ),

    # =========================================================================
    # STOCHASTIC (Indicator version - smoothK=1)
    # =========================================================================
    "stochastic": IndicatorConfig(
        name="Stochastic",
        pine_function="ta.stoch(close, high, low, length)",
        default_params={
            "length": 14,
            "smooth_k": 1,  # INDICATOR default is 1!
            "smooth_d": 3,
        },
        formula_notes="""
TradingView Stochastic INDICATOR Formula (lines 4080-4085):
    periodK = input.int(14, title="%K Length", minval=1)
    smoothK = input.int(1, title="%K Smoothing", minval=1)  // NOTE: Default is 1!
    periodD = input.int(3, title="%D Smoothing", minval=1)
    k = ta.sma(ta.stoch(close, high, low, periodK), smoothK)
    d = ta.sma(k, periodD)

CRITICAL: The INDICATOR uses smoothK=1 by default!
The STRATEGY (Stochastic Slow) uses smoothK=3 (hardcoded, not an input)!
        """,
        python_equivalent="""
def stochastic_tradingview(high, low, close, length=14, smooth_k=1, smooth_d=3):
    lowest_low = low.rolling(length).min()
    highest_high = high.rolling(length).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d
        """
    ),

    # =========================================================================
    # STOCHASTIC SLOW (Strategy version - smoothK=3)
    # =========================================================================
    "stochastic_slow": IndicatorConfig(
        name="Stochastic Slow (Strategy)",
        pine_function="ta.stoch with double smoothing",
        default_params={
            "length": 14,
            "smooth_k": 3,  # STRATEGY uses 3 (hardcoded)!
            "smooth_d": 3,
            "overbought": 80,
            "oversold": 20,
        },
        formula_notes="""
TradingView Stochastic Slow STRATEGY Formula (from strategy file):
    smoothK = 3  // HARDCODED - not an input!
    smoothD = 3
    k = ta.sma(ta.stoch(close, high, low, length), smoothK)
    d = ta.sma(k, smoothD)

Entry conditions:
    Long: ta.crossover(k, d) AND k < OverSold (20)
    Short: ta.crossunder(k, d) AND k > OverBought (80)
        """,
        python_equivalent="""
def stochastic_slow_tradingview(high, low, close, length=14, smooth_k=3, smooth_d=3):
    lowest_low = low.rolling(length).min()
    highest_high = high.rolling(length).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d
        """
    ),

    # =========================================================================
    # ATR - Average True Range
    # =========================================================================
    "atr": IndicatorConfig(
        name="Average True Range",
        pine_function="ta.atr(length)",
        default_params={
            "length": 14,
            "smoothing": "RMA",  # Options: RMA (default), SMA, EMA, WMA
        },
        formula_notes="""
TradingView ATR Formula (lines 838-847):
    ma_function(source, length) =>
        switch smoothing
            "RMA" => ta.rma(source, length)  // DEFAULT
            "SMA" => ta.sma(source, length)
            "EMA" => ta.ema(source, length)
            => ta.wma(source, length)
    plot(ma_function(ta.tr(true), length))

KEY: Default smoothing is RMA (Wilder's smoothing), NOT SMA!
ta.tr(true) = True Range considering gaps
        """,
        python_equivalent="""
def atr_tradingview(high, low, close, length=14):
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # RMA (Wilder's smoothing)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr
        """
    ),

    # =========================================================================
    # ADX - Average Directional Index
    # =========================================================================
    "adx": IndicatorConfig(
        name="Average Directional Index",
        pine_function="ta.dmi(di_length, adx_smoothing)",
        default_params={
            "adx_smoothing": 14,
            "di_length": 14,
        },
        formula_notes="""
TradingView ADX Formula (lines 818-834):
    dirmov(len) =>
        up = ta.change(high)
        down = -ta.change(low)
        plusDM = na(up) ? na : (up > down and up > 0 ? up : 0)
        minusDM = na(down) ? na : (down > up and down > 0 ? down : 0)
        truerange = ta.rma(ta.tr, len)
        plus = fixnan(100 * ta.rma(plusDM, len) / truerange)
        minus = fixnan(100 * ta.rma(minusDM, len) / truerange)
        [plus, minus]

    adx(dilen, adxlen) =>
        [plus, minus] = dirmov(dilen)
        sum = plus + minus
        adx = 100 * ta.rma(math.abs(plus - minus) / (sum == 0 ? 1 : sum), adxlen)

KEY: Uses ta.rma() for all smoothing operations
        """,
        python_equivalent="""
def adx_tradingview(high, low, close, length=14):
    # +DM and -DM
    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    # True Range with RMA smoothing
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()

    # +DI and -DI
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / atr

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()

    return plus_di, minus_di, adx
        """
    ),

    # =========================================================================
    # EMA - Exponential Moving Average
    # =========================================================================
    "ema": IndicatorConfig(
        name="Exponential Moving Average",
        pine_function="ta.ema(source, length)",
        default_params={
            "length": 9,
            "source": "close",
        },
        formula_notes="""
TradingView EMA Formula:
    multiplier = 2 / (length + 1)
    ema = (close - ema[1]) * multiplier + ema[1]

First value is typically initialized with SMA of first 'length' values
        """,
        python_equivalent="""
def ema_tradingview(close, length=9):
    return close.ewm(span=length, adjust=False).mean()
        """
    ),

    # =========================================================================
    # SMA - Simple Moving Average
    # =========================================================================
    "sma": IndicatorConfig(
        name="Simple Moving Average",
        pine_function="ta.sma(source, length)",
        default_params={
            "length": 20,
            "source": "close",
        },
        formula_notes="""
TradingView SMA Formula:
    sma = sum(source, length) / length
        """,
        python_equivalent="""
def sma_tradingview(close, length=20):
    return close.rolling(length).mean()
        """
    ),

    # =========================================================================
    # SUPERTREND
    # =========================================================================
    "supertrend": IndicatorConfig(
        name="Supertrend",
        pine_function="ta.supertrend(factor, atr_period)",
        default_params={
            "factor": 3.0,
            "atr_period": 10,
        },
        formula_notes="""
TradingView Supertrend Strategy (lines 470-482):
    atrPeriod = input(10, "ATR Length")
    factor = input.float(3.0, "Factor", step = 0.01)
    [_, direction] = ta.supertrend(factor, atrPeriod)

    if ta.change(direction) < 0  // Long entry
    if ta.change(direction) > 0  // Short entry

Direction values:
    1 = bearish (price below supertrend)
    -1 = bullish (price above supertrend)
        """,
        python_equivalent="""
# See pandas_ta.supertrend implementation
# NOTE: pandas_ta uses OPPOSITE direction convention!
# In pandas_ta: 1 = bullish, -1 = bearish
# In TradingView: 1 = bearish, -1 = bullish
        """
    ),

    # =========================================================================
    # WILLIAMS %R
    # =========================================================================
    "williams_r": IndicatorConfig(
        name="Williams %R",
        pine_function="ta.wpr(length)",
        default_params={
            "length": 14,
        },
        formula_notes="""
TradingView Williams %R Formula:
    %R = -100 * (highest_high - close) / (highest_high - lowest_low)

Range: -100 to 0
    > -20 = overbought
    < -80 = oversold
        """,
        python_equivalent="""
def williams_r_tradingview(high, low, close, length=14):
    highest_high = high.rolling(length).max()
    lowest_low = low.rolling(length).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr
        """
    ),

    # =========================================================================
    # CCI - Commodity Channel Index
    # =========================================================================
    "cci": IndicatorConfig(
        name="Commodity Channel Index",
        pine_function="ta.cci(high, low, close, length)",
        default_params={
            "length": 20,
        },
        formula_notes="""
TradingView CCI Formula:
    typical_price = (high + low + close) / 3
    sma_tp = ta.sma(typical_price, length)
    mean_deviation = ta.dev(typical_price, length)  # Mean absolute deviation
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        """,
        python_equivalent="""
def cci_tradingview(high, low, close, length=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(length).mean()
    mad = (tp - sma_tp).abs().rolling(length).mean()
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci
        """
    ),
}


# =============================================================================
# STRATEGY DEFAULT SETTINGS
# These are the exact settings from TradingView's built-in strategies
# =============================================================================

STRATEGY_DEFAULTS = {
    "default_qty_type": "strategy.percent_of_equity",
    "default_qty_value": 10,  # Most strategies use 10% by default
    "commission_percent": 0.1,  # Our default, TV doesn't specify
    "slippage": 0,

    # Strategy-specific overrides
    "strategies": {
        "BarUpDn Strategy": {"default_qty_value": 10, "max_intraday_loss_pct": 1.0},
        "Supertrend Strategy": {"default_qty_value": 15},
        "Technical Ratings Strategy": {"default_qty_value": 5},
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_indicator_config(indicator_name: str) -> IndicatorConfig:
    """Get the configuration for a specific indicator."""
    return INDICATORS.get(indicator_name.lower())


def get_indicator_defaults(indicator_name: str) -> Dict[str, Any]:
    """Get the default parameters for a specific indicator."""
    config = get_indicator_config(indicator_name)
    return config.default_params if config else {}


def list_all_indicators() -> list:
    """List all available indicator names."""
    return list(INDICATORS.keys())


# =============================================================================
# IMPORTANT DIFFERENCES SUMMARY
# =============================================================================
"""
KEY DIFFERENCES THAT CAUSE MISMATCHES:

1. RSI SMOOTHING:
   - TradingView: Uses RMA (ta.rma) = Wilder's smoothing (alpha = 1/length)
   - pandas_ta: Uses RMA by default ✓
   - Some Python libs: Use SMA or EMA ✗

2. STOCHASTIC SMOOTHING:
   - TradingView INDICATOR: smoothK = 1 (default)
   - TradingView STRATEGY (Stochastic Slow): smoothK = 3 (hardcoded!)
   - Make sure to use smoothK = 3 for strategy matching!

3. BOLLINGER BANDS STD DEV:
   - TradingView: Uses POPULATION std dev (ddof=0)
   - Some Python libs: Use SAMPLE std dev (ddof=1) ✗

4. ATR SMOOTHING:
   - TradingView: Uses RMA (Wilder's smoothing) by default
   - Some Python libs: Use SMA ✗

5. ADX SMOOTHING:
   - TradingView: Uses RMA for all components
   - Some Python libs: Use different smoothing ✗

6. SUPERTREND DIRECTION:
   - TradingView: 1 = bearish, -1 = bullish
   - pandas_ta: 1 = bullish, -1 = bearish (OPPOSITE!)
   - Account for this inversion in signal logic!

7. EMA INITIALIZATION:
   - TradingView: First EMA value is initialized with SMA
   - Some Python libs: Different initialization ✗
"""
