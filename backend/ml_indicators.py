"""
ML-Based Advanced Indicators
=============================
Cutting-edge indicators inspired by TradingView Editor's Picks and community research:
- Nadaraya-Watson Kernel Regression (adaptive S/R envelopes)
- 3-Wave Divergence Detection (Vdubus style momentum exhaustion)
- Cumulative Volume Delta (CVD) - tick-approximated order flow
- Smart Money Concepts (Order Blocks, Fair Value Gaps)
- Squeeze Momentum (LazyBear style BB/KC squeeze)
- Connors RSI (composite momentum indicator)
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands, KeltnerChannel, AverageTrueRange
from ta.trend import MACD


# =============================================================================
# KERNEL-BASED SMOOTHING (Nadaraya-Watson)
# =============================================================================

def gaussian_kernel(distance: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian (RBF) kernel function"""
    return np.exp(-0.5 * (distance / bandwidth) ** 2)


def nadaraya_watson_estimator(series: pd.Series, bandwidth: int = 8) -> pd.Series:
    """
    Nadaraya-Watson Kernel Regression Estimator
    
    A non-parametric regression technique that uses kernel-weighted averaging
    to estimate the underlying trend. Creates smooth, adaptive moving average.
    
    Args:
        series: Price series to smooth
        bandwidth: Kernel bandwidth (higher = smoother)
    
    Returns:
        Smoothed series using kernel regression
    """
    n = len(series)
    result = np.zeros(n)
    values = series.values
    
    for i in range(n):
        # Calculate weights for all points relative to current point
        distances = np.abs(np.arange(n) - i)
        weights = gaussian_kernel(distances, bandwidth)
        
        # Weighted average
        result[i] = np.sum(weights * values) / np.sum(weights)
    
    return pd.Series(result, index=series.index)


def nadaraya_watson_envelope(df: pd.DataFrame, bandwidth: int = 8, 
                              mult: float = 3.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Nadaraya-Watson Kernel Regression Envelope
    
    Creates adaptive support/resistance bands using kernel regression
    with ATR-based envelope width.
    
    Returns: (center_line, upper_band, lower_band)
    """
    close = df['close']
    
    # Calculate kernel regression center line
    center = nadaraya_watson_estimator(close, bandwidth)
    
    # Calculate ATR for envelope width
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    atr_values = atr.average_true_range()
    
    # Create bands
    upper = center + (atr_values * mult)
    lower = center - (atr_values * mult)
    
    return center, upper, lower


def kernel_ma(series: pd.Series, length: int = 20, kernel_type: str = 'gaussian') -> pd.Series:
    """
    Kernel-weighted Moving Average
    
    Alternative to traditional MAs using kernel functions for weighting.
    """
    result = np.zeros(len(series))
    values = series.values
    
    for i in range(length - 1, len(series)):
        window = values[i - length + 1:i + 1]
        
        if kernel_type == 'gaussian':
            # Distance from most recent point
            distances = np.arange(length)[::-1]
            weights = gaussian_kernel(distances, length / 3)
        else:
            # Linear weights (similar to WMA)
            weights = np.arange(1, length + 1)
        
        result[i] = np.sum(weights * window) / np.sum(weights)
    
    return pd.Series(result, index=series.index)


# =============================================================================
# DIVERGENCE DETECTION (Vdubus 3-Wave Theory)
# =============================================================================

def find_swing_points(series: pd.Series, lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Find swing highs and swing lows in a series.
    
    Returns: (swing_highs, swing_lows) as boolean series
    """
    highs = pd.Series(False, index=series.index)
    lows = pd.Series(False, index=series.index)
    
    for i in range(lookback, len(series) - lookback):
        # Swing high: higher than lookback bars on both sides
        if series.iloc[i] == series.iloc[i - lookback:i + lookback + 1].max():
            highs.iloc[i] = True
        # Swing low: lower than lookback bars on both sides
        if series.iloc[i] == series.iloc[i - lookback:i + lookback + 1].min():
            lows.iloc[i] = True
    
    return highs, lows


def detect_divergence_3wave(df: pd.DataFrame, macd_fast: int = 8, 
                            macd_slow: int = 21, macd_signal: int = 5,
                            lookback: int = 50) -> Tuple[pd.Series, pd.Series]:
    """
    3-Wave Divergence Detection (Vdubus Style)
    
    Detects momentum exhaustion patterns using 3 consecutive peaks/troughs:
    - Wave 1→2: Initial divergence (warning)
    - Wave 2→3: Confirmation divergence (entry signal)
    
    Based on Vdubus Divergence Wave Pattern Generator theory:
    - Standard Reversal: Momentum decays across 3 waves
    - Climax Reversal: Strong momentum followed by sudden failure
    
    Returns: (bullish_divergence, bearish_divergence) signals
    """
    close = df['close']
    
    # Calculate MACD for momentum
    macd = MACD(close, window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
    macd_hist = macd.macd_diff()
    
    # Find swing points in price and momentum
    price_highs, price_lows = find_swing_points(close, 5)
    mom_highs, mom_lows = find_swing_points(macd_hist, 5)
    
    bullish_div = pd.Series(False, index=df.index)
    bearish_div = pd.Series(False, index=df.index)
    
    # Collect swing point indices
    price_high_idx = price_highs[price_highs].index.tolist()
    price_low_idx = price_lows[price_lows].index.tolist()
    
    for i in range(lookback, len(df)):
        current_idx = df.index[i]
        
        # Look for 3 recent price lows (for bullish divergence)
        recent_lows = [idx for idx in price_low_idx if idx <= current_idx and idx >= df.index[max(0, i - lookback)]]
        
        if len(recent_lows) >= 3:
            # Get last 3 lows
            l1, l2, l3 = recent_lows[-3], recent_lows[-2], recent_lows[-1]
            
            # Check for lower lows in price
            if close[l3] < close[l2] < close[l1]:
                # Check for higher lows in momentum (bullish divergence)
                if macd_hist[l3] > macd_hist[l2] or macd_hist[l2] > macd_hist[l1]:
                    bullish_div.iloc[i] = True
        
        # Look for 3 recent price highs (for bearish divergence)
        recent_highs = [idx for idx in price_high_idx if idx <= current_idx and idx >= df.index[max(0, i - lookback)]]
        
        if len(recent_highs) >= 3:
            h1, h2, h3 = recent_highs[-3], recent_highs[-2], recent_highs[-1]
            
            # Check for higher highs in price
            if close[h3] > close[h2] > close[h1]:
                # Check for lower highs in momentum (bearish divergence)
                if macd_hist[h3] < macd_hist[h2] or macd_hist[h2] < macd_hist[h1]:
                    bearish_div.iloc[i] = True
    
    return bullish_div, bearish_div


def detect_rsi_divergence(df: pd.DataFrame, rsi_length: int = 14,
                          lookback: int = 30) -> Tuple[pd.Series, pd.Series]:
    """
    RSI Divergence Detection
    
    Detects regular bullish/bearish divergence between price and RSI.
    
    Returns: (bullish_divergence, bearish_divergence)
    """
    close = df['close']
    rsi = RSIIndicator(close, window=rsi_length).rsi()
    
    price_highs, price_lows = find_swing_points(close, 5)
    rsi_highs, rsi_lows = find_swing_points(rsi, 5)
    
    bullish_div = pd.Series(False, index=df.index)
    bearish_div = pd.Series(False, index=df.index)
    
    price_low_idx = price_lows[price_lows].index.tolist()
    price_high_idx = price_highs[price_highs].index.tolist()
    
    for i in range(lookback, len(df)):
        current_idx = df.index[i]
        start_idx = df.index[max(0, i - lookback)]
        
        # Bullish: Lower low in price, higher low in RSI
        recent_lows = [idx for idx in price_low_idx if start_idx <= idx <= current_idx]
        if len(recent_lows) >= 2:
            l1, l2 = recent_lows[-2], recent_lows[-1]
            if close[l2] < close[l1] and rsi[l2] > rsi[l1]:
                bullish_div.iloc[i] = True
        
        # Bearish: Higher high in price, lower high in RSI
        recent_highs = [idx for idx in price_high_idx if start_idx <= idx <= current_idx]
        if len(recent_highs) >= 2:
            h1, h2 = recent_highs[-2], recent_highs[-1]
            if close[h2] > close[h1] and rsi[h2] < rsi[h1]:
                bearish_div.iloc[i] = True
    
    return bullish_div, bearish_div


# =============================================================================
# CUMULATIVE VOLUME DELTA (CVD) - Order Flow
# =============================================================================

def calculate_cvd(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative Volume Delta (CVD)
    
    Approximates buying vs selling pressure from OHLC data.
    Uses the close position within the bar to estimate delta:
    - Close near high = buying pressure
    - Close near low = selling pressure
    
    Returns: Cumulative volume delta series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
    
    # Calculate close position within bar (0 to 1)
    bar_range = high - low
    bar_range = bar_range.replace(0, np.nan)  # Avoid division by zero
    
    close_position = (close - low) / bar_range
    close_position = close_position.fillna(0.5)  # Default to neutral
    
    # Convert to delta: -1 (selling) to +1 (buying)
    delta_ratio = (close_position * 2) - 1
    
    # Volume delta for each bar
    volume_delta = delta_ratio * volume
    
    # Cumulative sum
    cvd = volume_delta.cumsum()
    
    return cvd


def calculate_cvd_normalized(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Normalized CVD - CVD as percentage change from rolling mean
    
    Useful for detecting divergences and extreme readings.
    """
    cvd = calculate_cvd(df)
    cvd_ma = cvd.rolling(window=period).mean()
    cvd_std = cvd.rolling(window=period).std()
    
    # Z-score normalization
    cvd_norm = (cvd - cvd_ma) / cvd_std.replace(0, np.nan)
    
    return cvd_norm.fillna(0)


def cvd_divergence(df: pd.DataFrame, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    CVD-Price Divergence
    
    Detects when price and volume delta disagree:
    - Bullish: Price making lower lows, CVD making higher lows
    - Bearish: Price making higher highs, CVD making lower highs
    
    Returns: (bullish_divergence, bearish_divergence)
    """
    close = df['close']
    cvd = calculate_cvd(df)
    
    price_highs, price_lows = find_swing_points(close, 5)
    cvd_highs, cvd_lows = find_swing_points(cvd, 5)
    
    bullish_div = pd.Series(False, index=df.index)
    bearish_div = pd.Series(False, index=df.index)
    
    price_low_idx = price_lows[price_lows].index.tolist()
    price_high_idx = price_highs[price_highs].index.tolist()
    
    for i in range(lookback, len(df)):
        current_idx = df.index[i]
        start_idx = df.index[max(0, i - lookback)]
        
        # Bullish divergence
        recent_lows = [idx for idx in price_low_idx if start_idx <= idx <= current_idx]
        if len(recent_lows) >= 2:
            l1, l2 = recent_lows[-2], recent_lows[-1]
            if close[l2] < close[l1] and cvd[l2] > cvd[l1]:
                bullish_div.iloc[i] = True
        
        # Bearish divergence
        recent_highs = [idx for idx in price_high_idx if start_idx <= idx <= current_idx]
        if len(recent_highs) >= 2:
            h1, h2 = recent_highs[-2], recent_highs[-1]
            if close[h2] > close[h1] and cvd[h2] < cvd[h1]:
                bearish_div.iloc[i] = True
    
    return bullish_div, bearish_div


# =============================================================================
# SMART MONEY CONCEPTS (SMC)
# =============================================================================

def detect_order_blocks(df: pd.DataFrame, lookback: int = 10,
                        min_move: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Order Block Detection
    
    Identifies institutional entry zones where large orders were placed.
    Order blocks are the last opposing candle before a significant move.
    
    Args:
        lookback: Bars to look back for OB detection
        min_move: Minimum ATR multiplier for "significant move"
    
    Returns: (bullish_ob_zone, bearish_ob_zone, bullish_ob_top, bullish_ob_bottom)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    
    # Calculate ATR for move significance
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    
    bullish_ob = pd.Series(False, index=df.index)
    bearish_ob = pd.Series(False, index=df.index)
    bullish_ob_top = pd.Series(np.nan, index=df.index)
    bullish_ob_bottom = pd.Series(np.nan, index=df.index)
    
    for i in range(lookback + 2, len(df)):
        # Check for bullish order block
        # Look for bearish candle followed by strong bullish move
        for j in range(1, min(lookback, i)):
            prev_idx = i - j
            
            # Is previous candle bearish?
            if close.iloc[prev_idx] < open_price.iloc[prev_idx]:
                # Check if there was a strong bullish move after
                move = close.iloc[i] - low.iloc[prev_idx]
                if move > atr.iloc[i] * min_move:
                    bullish_ob.iloc[i] = True
                    bullish_ob_top.iloc[i] = high.iloc[prev_idx]
                    bullish_ob_bottom.iloc[i] = low.iloc[prev_idx]
                    break
        
        # Check for bearish order block
        for j in range(1, min(lookback, i)):
            prev_idx = i - j
            
            # Is previous candle bullish?
            if close.iloc[prev_idx] > open_price.iloc[prev_idx]:
                # Check if there was a strong bearish move after
                move = high.iloc[prev_idx] - close.iloc[i]
                if move > atr.iloc[i] * min_move:
                    bearish_ob.iloc[i] = True
                    break
    
    return bullish_ob, bearish_ob, bullish_ob_top, bullish_ob_bottom


def detect_fair_value_gaps(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Fair Value Gap (FVG) Detection
    
    FVGs are price imbalances where the market moved so fast that
    no trades occurred in a price range. Price often returns to fill these gaps.
    
    Bullish FVG: Current low > 2-bars-ago high
    Bearish FVG: Current high < 2-bars-ago low
    
    Returns: (bullish_fvg, bearish_fvg, fvg_top, fvg_bottom)
    """
    high = df['high']
    low = df['low']
    
    bullish_fvg = pd.Series(False, index=df.index)
    bearish_fvg = pd.Series(False, index=df.index)
    fvg_top = pd.Series(np.nan, index=df.index)
    fvg_bottom = pd.Series(np.nan, index=df.index)
    
    for i in range(2, len(df)):
        # Bullish FVG: Gap up (current low > 2-bars-ago high)
        if low.iloc[i] > high.iloc[i - 2]:
            bullish_fvg.iloc[i] = True
            fvg_top.iloc[i] = low.iloc[i]
            fvg_bottom.iloc[i] = high.iloc[i - 2]
        
        # Bearish FVG: Gap down (current high < 2-bars-ago low)
        if high.iloc[i] < low.iloc[i - 2]:
            bearish_fvg.iloc[i] = True
            fvg_top.iloc[i] = low.iloc[i - 2]
            fvg_bottom.iloc[i] = high.iloc[i]
    
    return bullish_fvg, bearish_fvg, fvg_top, fvg_bottom


def detect_liquidity_zones(df: pd.DataFrame, lookback: int = 20,
                           touch_threshold: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Liquidity Zone Detection
    
    Identifies price levels where stops are likely accumulated
    (multiple touches of similar highs/lows).
    
    Returns: (liquidity_above, liquidity_below) - proximity to liquidity zones
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # ATR for zone width tolerance
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    
    liquidity_above = pd.Series(0.0, index=df.index)
    liquidity_below = pd.Series(0.0, index=df.index)
    
    for i in range(lookback, len(df)):
        # Get recent highs and lows
        recent_highs = high.iloc[i - lookback:i]
        recent_lows = low.iloc[i - lookback:i]
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else close.iloc[i] * 0.02
        
        # Find clusters of similar highs (potential stop zones above)
        high_levels = []
        for h in recent_highs:
            similar_count = sum(abs(recent_highs - h) < current_atr * 0.5)
            if similar_count >= touch_threshold:
                high_levels.append(h)
        
        if high_levels:
            nearest_high = min(high_levels, key=lambda x: abs(x - close.iloc[i]))
            if nearest_high > close.iloc[i]:
                distance = (nearest_high - close.iloc[i]) / current_atr
                liquidity_above.iloc[i] = max(0, 1 - distance / 5)  # Normalize to 0-1
        
        # Find clusters of similar lows (potential stop zones below)
        low_levels = []
        for l in recent_lows:
            similar_count = sum(abs(recent_lows - l) < current_atr * 0.5)
            if similar_count >= touch_threshold:
                low_levels.append(l)
        
        if low_levels:
            nearest_low = min(low_levels, key=lambda x: abs(x - close.iloc[i]))
            if nearest_low < close.iloc[i]:
                distance = (close.iloc[i] - nearest_low) / current_atr
                liquidity_below.iloc[i] = max(0, 1 - distance / 5)
    
    return liquidity_above, liquidity_below


# =============================================================================
# SQUEEZE MOMENTUM (LazyBear Style)
# =============================================================================

def squeeze_momentum(df: pd.DataFrame, bb_length: int = 20, bb_mult: float = 2.0,
                     kc_length: int = 20, kc_mult: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Squeeze Momentum Indicator (LazyBear Style)
    
    Detects when Bollinger Bands are inside Keltner Channels (squeeze),
    indicating low volatility that often precedes explosive moves.
    
    Returns: (squeeze_on, momentum, momentum_direction)
        - squeeze_on: Boolean, True when in squeeze
        - momentum: Momentum histogram value
        - momentum_direction: 1 = increasing, -1 = decreasing
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Bollinger Bands
    bb = BollingerBands(close, window=bb_length, window_dev=bb_mult)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    
    # Keltner Channels
    kc = KeltnerChannel(high, low, close, window=kc_length, window_atr=kc_length)
    kc_upper = kc.keltner_channel_hband() * kc_mult / 2  # Adjust for multiplier
    kc_lower = kc.keltner_channel_lband() * kc_mult / 2
    
    # Recalculate KC with proper multiplier
    ema_mid = close.ewm(span=kc_length, adjust=False).mean()
    atr = AverageTrueRange(high, low, close, window=kc_length).average_true_range()
    kc_upper = ema_mid + (atr * kc_mult)
    kc_lower = ema_mid - (atr * kc_mult)
    
    # Squeeze detection: BB inside KC
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    # Momentum calculation using linear regression
    # Simplified: using price deviation from midline
    midline = (high.rolling(kc_length).max() + low.rolling(kc_length).min()) / 2
    momentum = close - midline
    
    # Smooth momentum
    momentum = momentum.rolling(window=3).mean()
    
    # Direction: momentum increasing or decreasing
    momentum_dir = np.sign(momentum - momentum.shift(1))
    
    return squeeze_on, momentum, pd.Series(momentum_dir, index=df.index)


# =============================================================================
# CONNORS RSI (Composite Momentum)
# =============================================================================

def calculate_streak(series: pd.Series) -> pd.Series:
    """
    Calculate consecutive up/down streak.
    Positive = consecutive up days, Negative = consecutive down days
    """
    diff = series.diff()
    streak = pd.Series(0, index=series.index)
    
    current_streak = 0
    for i in range(1, len(series)):
        if diff.iloc[i] > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
        elif diff.iloc[i] < 0:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
        else:
            current_streak = 0
        
        streak.iloc[i] = current_streak
    
    return streak


def connors_rsi(df: pd.DataFrame, rsi_length: int = 3, streak_length: int = 2,
                roc_length: int = 100) -> pd.Series:
    """
    Connors RSI - Composite Momentum Indicator
    
    Combines three components:
    1. Short-term RSI
    2. Streak RSI (RSI of up/down streak)
    3. Percent Rank of ROC
    
    Returns: Composite RSI value (0-100)
    """
    close = df['close']
    
    # Component 1: Short-term RSI
    rsi = RSIIndicator(close, window=rsi_length).rsi()
    
    # Component 2: Streak RSI
    streak = calculate_streak(close)
    streak_rsi = RSIIndicator(streak, window=streak_length).rsi()
    
    # Component 3: Percent Rank of ROC
    roc = ROCIndicator(close, window=1).roc()
    
    # Calculate percent rank (what % of values are below current)
    def percent_rank(series, length):
        result = pd.Series(np.nan, index=series.index)
        for i in range(length, len(series)):
            window = series.iloc[i - length:i]
            current = series.iloc[i]
            rank = (window < current).sum() / length * 100
            result.iloc[i] = rank
        return result
    
    roc_rank = percent_rank(roc, roc_length)
    
    # Combine components (equal weight)
    crsi = (rsi + streak_rsi + roc_rank) / 3
    
    return crsi.fillna(50)


# =============================================================================
# MASTER CALCULATION FUNCTION
# =============================================================================

def calculate_ml_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all ML-based indicators and add to dataframe.
    
    This is the main entry point for the ml_indicators module.
    """
    result = df.copy()
    
    if len(df) < 50:
        print("  Warning: Not enough data for ML indicators (need 50+ bars)")
        return result
    
    print("  Calculating ML-based indicators...")
    
    try:
        # === Nadaraya-Watson Kernel Regression ===
        for bandwidth in [8, 12]:
            for mult in [2.0, 3.0]:
                center, upper, lower = nadaraya_watson_envelope(df, bandwidth, mult)
                result[f'nw_center_{bandwidth}_{mult}'] = center
                result[f'nw_upper_{bandwidth}_{mult}'] = upper
                result[f'nw_lower_{bandwidth}_{mult}'] = lower
        
        # Default NW for quick access
        result['nw_center'] = result['nw_center_8_3.0']
        result['nw_upper'] = result['nw_upper_8_3.0']
        result['nw_lower'] = result['nw_lower_8_3.0']
        
        # Kernel MA
        result['kernel_ma_20'] = kernel_ma(df['close'], 20)
        
    except Exception as e:
        print(f"    Warning: Nadaraya-Watson failed: {e}")
    
    try:
        # === Divergence Detection ===
        # 3-Wave MACD divergence
        bull_div_3w, bear_div_3w = detect_divergence_3wave(df, 8, 21, 5, 50)
        result['bullish_div_3wave'] = bull_div_3w
        result['bearish_div_3wave'] = bear_div_3w
        
        # RSI divergence
        bull_div_rsi, bear_div_rsi = detect_rsi_divergence(df, 14, 30)
        result['bullish_div_rsi'] = bull_div_rsi
        result['bearish_div_rsi'] = bear_div_rsi
        
    except Exception as e:
        print(f"    Warning: Divergence detection failed: {e}")
    
    try:
        # === CVD (Volume Delta) ===
        if 'volume' in df.columns and df['volume'].sum() > 0:
            result['cvd'] = calculate_cvd(df)
            result['cvd_norm'] = calculate_cvd_normalized(df, 20)
            
            # CVD divergence
            bull_cvd_div, bear_cvd_div = cvd_divergence(df, 20)
            result['bullish_cvd_div'] = bull_cvd_div
            result['bearish_cvd_div'] = bear_cvd_div
        
    except Exception as e:
        print(f"    Warning: CVD calculation failed: {e}")
    
    try:
        # === Smart Money Concepts ===
        # Order Blocks
        bull_ob, bear_ob, ob_top, ob_bottom = detect_order_blocks(df, 10, 1.5)
        result['bullish_ob'] = bull_ob
        result['bearish_ob'] = bear_ob
        result['ob_top'] = ob_top
        result['ob_bottom'] = ob_bottom
        
        # Fair Value Gaps
        bull_fvg, bear_fvg, fvg_top, fvg_bottom = detect_fair_value_gaps(df)
        result['bullish_fvg'] = bull_fvg
        result['bearish_fvg'] = bear_fvg
        result['fvg_top'] = fvg_top
        result['fvg_bottom'] = fvg_bottom
        
        # Liquidity zones
        liq_above, liq_below = detect_liquidity_zones(df, 20, 3)
        result['liquidity_above'] = liq_above
        result['liquidity_below'] = liq_below
        
    except Exception as e:
        print(f"    Warning: SMC indicators failed: {e}")
    
    try:
        # === Squeeze Momentum ===
        for kc_mult in [1.5, 2.0]:
            squeeze_on, momentum, mom_dir = squeeze_momentum(df, 20, 2.0, 20, kc_mult)
            result[f'squeeze_on_{kc_mult}'] = squeeze_on
            result[f'squeeze_mom_{kc_mult}'] = momentum
            result[f'squeeze_dir_{kc_mult}'] = mom_dir
        
        # Default squeeze for quick access
        result['squeeze_on'] = result['squeeze_on_1.5']
        result['squeeze_mom'] = result['squeeze_mom_1.5']
        result['squeeze_dir'] = result['squeeze_dir_1.5']
        
    except Exception as e:
        print(f"    Warning: Squeeze momentum failed: {e}")
    
    try:
        # === Connors RSI ===
        result['connors_rsi'] = connors_rsi(df, 3, 2, 100)
        result['connors_rsi_slow'] = connors_rsi(df, 5, 3, 100)
        
    except Exception as e:
        print(f"    Warning: Connors RSI failed: {e}")
    
    print("  ML-based indicators complete.")
    return result



