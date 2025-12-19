"""
DaviddTech-Style Advanced Indicators
=====================================
Implements sophisticated indicators used in professional trading strategies:
- JMA (Jurik Moving Average) - Ultra-smooth, low lag
- Stiffness Indicator - Trend strength measurement
- TDFI (Trend Direction Force Index) - Trend direction
- McGinley Dynamic - Adaptive moving average
- Volatility Quality - Volatility filtering
- Trendilo/Trendilio - ALMA-based trend detection
- Range Filter - Volatility-based trend filtering
- T3 (Tillson T3) - Triple smoothed EMA
- LWPI (Larry Williams Proxy Index)
- Flat Market Detector - Choppy market identification
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=length).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, length + 1)
    return series.rolling(window=length).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)


def hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average - Low lag MA"""
    half_length = max(int(length / 2), 1)
    sqrt_length = max(int(np.sqrt(length)), 1)
    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    return wma(2 * wma_half - wma_full, sqrt_length)


def dema(series: pd.Series, length: int) -> pd.Series:
    """Double Exponential Moving Average"""
    ema1 = ema(series, length)
    ema2 = ema(ema1, length)
    return 2 * ema1 - ema2


def tema(series: pd.Series, length: int) -> pd.Series:
    """Triple Exponential Moving Average"""
    ema1 = ema(series, length)
    ema2 = ema(ema1, length)
    ema3 = ema(ema2, length)
    return 3 * ema1 - 3 * ema2 + ema3


def alma(series: pd.Series, length: int = 9, offset: float = 0.85, sigma: float = 6) -> pd.Series:
    """
    Arnaud Legoux Moving Average
    Low-lag, smooth moving average used in Trendilo
    """
    m = offset * (length - 1)
    s = length / sigma
    weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * s * s))
    weights = weights / weights.sum()
    
    def alma_calc(x):
        return np.sum(weights * x)
    
    return series.rolling(window=length).apply(alma_calc, raw=True)


def t3(series: pd.Series, length: int = 5, vfactor: float = 0.7) -> pd.Series:
    """
    Tillson T3 Moving Average
    Triple smoothed EMA with volume factor for reduced lag
    Used in DaviddTech strategies
    """
    c1 = -vfactor ** 3
    c2 = 3 * vfactor ** 2 + 3 * vfactor ** 3
    c3 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3
    c4 = 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2
    
    e1 = ema(series, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def zlema(series: pd.Series, length: int) -> pd.Series:
    """Zero-Lag EMA - Reduces lag by adjusting for it"""
    lag = (length - 1) // 2
    data = 2 * series - series.shift(lag)
    return ema(data, length)


def jma(series: pd.Series, length: int = 7, phase: int = 50, power: int = 2) -> pd.Series:
    """
    Jurik Moving Average (Approximation)
    Ultra-smooth, low-lag moving average
    Used extensively in Stiff Surge strategy
    
    Phase: -100 to 100 (higher = more smoothing)
    Power: typically 2
    """
    # JMA is proprietary, this is an approximation using adaptive EMA
    phase_ratio = (phase / 100) + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = pow(beta, power)
    
    result = series.copy()
    e0 = series.iloc[0] if len(series) > 0 else 0
    e1 = 0
    e2 = 0
    
    jma_values = []
    for i, price in enumerate(series):
        if pd.isna(price):
            jma_values.append(np.nan)
            continue
            
        e0 = (1 - alpha) * price + alpha * e0
        e1 = (price - e0) * (1 - beta) + beta * e1
        e2 = (e0 + phase_ratio * e1 - result.iloc[i-1] if i > 0 else e0) * pow(1 - alpha, 2) + pow(alpha, 2) * e2
        jma_value = e2 + (result.iloc[i-1] if i > 0 else e0)
        jma_values.append(jma_value)
    
    return pd.Series(jma_values, index=series.index)


def mcginley_dynamic(series: pd.Series, length: int = 14, k: float = 0.6) -> pd.Series:
    """
    McGinley Dynamic
    Self-adjusting moving average that tracks price more closely
    Used in McGinley Trend Followers strategy
    
    K constant: 0.6 is standard, lower = faster
    """
    md = series.copy()
    md.iloc[0] = series.iloc[0]
    
    for i in range(1, len(series)):
        if pd.isna(series.iloc[i]) or pd.isna(md.iloc[i-1]):
            md.iloc[i] = md.iloc[i-1] if not pd.isna(md.iloc[i-1]) else series.iloc[i]
            continue
        
        prev_md = md.iloc[i-1]
        price = series.iloc[i]
        
        if prev_md != 0:
            ratio = price / prev_md
            adjustment = length * pow(ratio, 4) * k
            if adjustment > 0:
                md.iloc[i] = prev_md + (price - prev_md) / max(adjustment, 1)
            else:
                md.iloc[i] = prev_md
        else:
            md.iloc[i] = price
    
    return md


def stiffness(df: pd.DataFrame, stiff_length: int = 60, ma_length: int = 100, 
              smooth_length: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stiffness Indicator
    Measures how often price stays above/below a moving average
    Key component of Stiff Surge strategy
    
    Returns: (stiffness_value, stiffness_ma)
    """
    close = df['close']
    ma = sma(close, ma_length)
    
    # Count bars above MA in lookback period
    above_ma = (close > ma).astype(int)
    stiff = above_ma.rolling(window=stiff_length).sum() / stiff_length * 100
    
    # Smooth the stiffness
    stiff_smooth = sma(stiff, smooth_length)
    
    return stiff, stiff_smooth


def tdfi(df: pd.DataFrame, lookback: int = 13, mma_length: int = 13, 
         smma_length: int = 13, n_length: int = 3) -> pd.Series:
    """
    Trend Direction Force Index (TDFI)
    Measures trend direction and strength
    Used in Stiff Surge strategy
    """
    close = df['close']
    
    # Calculate price momentum
    momentum = close - close.shift(lookback)
    
    # Smooth with multiple MAs
    mma = ema(momentum, mma_length)
    smma = ema(mma, smma_length)
    
    # Normalize
    n_val = sma(smma.abs(), n_length)
    tdfi_val = smma / n_val.replace(0, np.nan)
    
    return tdfi_val.fillna(0)


def volatility_quality(df: pd.DataFrame, length: int = 14, 
                       smooth_length: int = 5) -> pd.Series:
    """
    Volatility Quality Indicator
    Measures quality of price movement relative to volatility
    Used in Stiff Surge strategy
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # True Range
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    
    # Average True Range
    atr = sma(tr, length)
    
    # Price change
    change = close - close.shift(length)
    
    # Volatility Quality = Movement efficiency
    vq = change / (atr * np.sqrt(length) + 1e-10)
    
    return sma(vq, smooth_length)


def trendilo(df: pd.DataFrame, lookback: int = 50, smooth: int = 3,
             band_mult: float = 1.0, alma_offset: float = 0.85, 
             alma_sigma: float = 6) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Trendilo/Trendilio Indicator
    ALMA-based trend detection used in Trendhoo strategy
    
    Returns: (trendilo_value, upper_band, lower_band)
    """
    source = (df['high'] + df['low']) / 2
    
    # Calculate ALMA
    alma_val = alma(source, lookback, alma_offset, alma_sigma)
    
    # Smooth the ALMA
    trend = sma(alma_val, smooth)
    
    # Calculate bands using ATR
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': (df['high'] - df['close'].shift(1)).abs(),
        'lc': (df['low'] - df['close'].shift(1)).abs()
    }).max(axis=1)
    atr = sma(tr, lookback)
    
    upper = trend + atr * band_mult
    lower = trend - atr * band_mult
    
    return trend, upper, lower


def range_filter(df: pd.DataFrame, period: int = 100, mult: float = 3.0) -> Tuple[pd.Series, pd.Series, int]:
    """
    Range Filter
    Volatility-based trend filter used in many DaviddTech strategies
    
    Returns: (range_filter_value, direction, -)
    """
    close = df['close']
    
    # Calculate range
    wper = period * 2 - 1
    avrng = ema(pd.DataFrame({
        'hl': df['high'] - df['low']
    })['hl'], period)
    smoothrng = ema(avrng, wper) * mult
    
    # Range filter calculation
    filt = close.copy()
    upward = pd.Series(0, index=close.index)
    downward = pd.Series(0, index=close.index)
    
    for i in range(1, len(close)):
        prev_filt = filt.iloc[i-1]
        if pd.isna(prev_filt):
            prev_filt = close.iloc[i]
        
        rng = smoothrng.iloc[i] if not pd.isna(smoothrng.iloc[i]) else 0
        
        if close.iloc[i] > prev_filt:
            if close.iloc[i] - rng < prev_filt:
                filt.iloc[i] = prev_filt
            else:
                filt.iloc[i] = close.iloc[i] - rng
        else:
            if close.iloc[i] + rng > prev_filt:
                filt.iloc[i] = prev_filt
            else:
                filt.iloc[i] = close.iloc[i] + rng
        
        upward.iloc[i] = 1 if filt.iloc[i] > filt.iloc[i-1] else (upward.iloc[i-1] + 1 if filt.iloc[i] < filt.iloc[i-1] else 0)
        downward.iloc[i] = 1 if filt.iloc[i] < filt.iloc[i-1] else (downward.iloc[i-1] + 1 if filt.iloc[i] > filt.iloc[i-1] else 0)
    
    # Direction: 1 = uptrend, -1 = downtrend
    direction = pd.Series(np.where(upward > 0, 1, np.where(downward > 0, -1, 0)), index=close.index)
    
    return filt, direction


def flat_market_detector(df: pd.DataFrame, ma_length: int = 75, 
                         threshold: float = 30, ma_type: str = 'vwma') -> pd.Series:
    """
    Flat Market Detector
    Identifies choppy/ranging markets to avoid
    Returns True when market is flat (avoid trading)
    
    Uses ADX-style calculation but with MA-based detection
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Calculate directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=close.index)
    
    # ATR
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    atr = sma(tr, 14)
    
    # Directional indicators
    plus_di = 100 * sma(plus_dm, 14) / atr.replace(0, np.nan)
    minus_di = 100 * sma(minus_dm, 14) / atr.replace(0, np.nan)
    
    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = sma(dx, 14)
    
    # Flat market when ADX is below threshold
    is_flat = adx < threshold
    
    return is_flat


def lwpi(df: pd.DataFrame, length: int = 13, smooth: bool = True, 
         smooth_period: int = 3) -> pd.Series:
    """
    Larry Williams Proxy Index
    Used in McGinley Trend Followers strategy
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Highest high and lowest low
    hh = high.rolling(window=length).max()
    ll = low.rolling(window=length).min()
    
    # LWPI calculation
    lwpi_val = (hh - close) / (hh - ll + 1e-10) * -100
    
    if smooth:
        lwpi_val = sma(lwpi_val, smooth_period)
    
    return lwpi_val


def normalized_volume(df: pd.DataFrame, length: int = 55, 
                      high_threshold: float = 110, low_threshold: float = 130) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Normalized Volume Indicator
    Used in Precision Trend Mastery
    
    Returns: (norm_vol, is_high_vol, is_low_vol)
    """
    volume = df['volume']
    
    # Normalize volume to percentage of average
    avg_vol = sma(volume, length)
    norm_vol = (volume / avg_vol.replace(0, np.nan)) * 100
    
    is_high = norm_vol > high_threshold
    is_low = norm_vol < (100 - (100 - low_threshold))  # Invert for low detection
    
    return norm_vol, is_high, is_low


def supertrend(df: pd.DataFrame, length: int = 10, mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend Indicator
    Trend following indicator based on ATR
    
    Returns: (supertrend_value, direction)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # ATR
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    atr = sma(tr, length)
    
    # Basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + mult * atr
    lower_band = hl2 - mult * atr
    
    # Supertrend calculation
    supertrend = close.copy()
    direction = pd.Series(1, index=close.index)
    
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    
    for i in range(1, len(close)):
        # Update bands
        if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
            
        if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        
        # Direction
        if pd.isna(supertrend.iloc[i-1]):
            direction.iloc[i] = 1
        elif supertrend.iloc[i-1] == final_upper.iloc[i-1]:
            direction.iloc[i] = -1 if close.iloc[i] > final_upper.iloc[i] else -1
        else:
            direction.iloc[i] = 1 if close.iloc[i] < final_lower.iloc[i] else 1
        
        if close.iloc[i] > final_upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        
        supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]
    
    return supertrend, direction


def calculate_daviddtech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all DaviddTech-style indicators
    Returns dataframe with all indicators added
    """
    result = df.copy()
    
    # === JMA ===
    for length in [4, 7, 14, 43]:
        for phase in [50, 84, 97]:
            result[f'jma_{length}_{phase}'] = jma(df['close'], length, phase, 2)
    
    # === McGinley Dynamic ===
    for length in [14, 130, 194]:
        for k in [0.6, 0.7]:
            result[f'mcginley_{length}_{k}'] = mcginley_dynamic(df['close'], length, k)
    
    # === Stiffness ===
    for stiff_len in [39, 60, 213]:
        for ma_len in [50, 100, 140]:
            stiff, stiff_smooth = stiffness(df, stiff_len, ma_len, 6)
            result[f'stiff_{stiff_len}_{ma_len}'] = stiff
            result[f'stiff_smooth_{stiff_len}_{ma_len}'] = stiff_smooth
    
    # === TDFI ===
    for lookback in [10, 13, 15, 23]:
        result[f'tdfi_{lookback}'] = tdfi(df, lookback)
    
    # === Volatility Quality ===
    for length in [14, 29, 35, 199]:
        result[f'vq_{length}'] = volatility_quality(df, length, 5)
    
    # === T3 ===
    for length in [5, 10, 20]:
        result[f't3_{length}'] = t3(df['close'], length, 0.7)
    
    # === ZLEMA ===
    for length in [10, 20, 50]:
        result[f'zlema_{length}'] = zlema(df['close'], length)
    
    # === HMA ===
    for length in [65, 85, 100]:
        result[f'hma_{length}'] = hma(df['close'], length)
    
    # === Trendilo ===
    for lookback in [45, 52]:
        trend, upper, lower = trendilo(df, lookback, 3, 1.8)
        result[f'trendilo_{lookback}'] = trend
        result[f'trendilo_upper_{lookback}'] = upper
        result[f'trendilo_lower_{lookback}'] = lower
    
    # === Range Filter ===
    for period in [100, 164, 200]:
        for mult in [2.2, 3.0, 4.5]:
            filt, direction = range_filter(df, period, mult)
            result[f'rf_{period}_{mult}'] = filt
            result[f'rf_dir_{period}_{mult}'] = direction
    
    # === Flat Market Detector ===
    for threshold in [15, 30, 40]:
        result[f'is_flat_{threshold}'] = flat_market_detector(df, 75, threshold)
    
    # === LWPI ===
    for length in [13, 130]:
        result[f'lwpi_{length}'] = lwpi(df, length, True, 12)
    
    # === Normalized Volume ===
    norm_vol, high_vol, low_vol = normalized_volume(df, 55)
    result['norm_vol'] = norm_vol
    result['high_vol'] = high_vol
    result['low_vol'] = low_vol
    
    # === Supertrend ===
    for length in [10, 14]:
        for mult in [2.5, 3.0]:
            st, st_dir = supertrend(df, length, mult)
            result[f'supertrend_{length}_{mult}'] = st
            result[f'st_dir_{length}_{mult}'] = st_dir
    
    # === ALMA ===
    for length in [9, 11]:
        for offset in [0.425, 0.85, 1.511]:
            result[f'alma_{length}_{offset}'] = alma(df['close'], length, offset, 6)
    
    return result



