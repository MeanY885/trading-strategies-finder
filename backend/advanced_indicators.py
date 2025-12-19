"""
Advanced ML-Oriented Indicators for BTCGBP Trading
Based on professional quant recommendations
"""
import pandas as pd
import numpy as np
from typing import Tuple


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP - Volume Weighted Average Price
    Feature: Distance from VWAP as percentage
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap


def calculate_vwap_distance(df: pd.DataFrame) -> pd.Series:
    """Distance from VWAP as percentage: (Price - VWAP) / Price"""
    vwap = calculate_vwap(df)
    return (df['close'] - vwap) / df['close'] * 100


def calculate_stoch_rsi(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    StochRSI - Stochastic RSI
    Returns: (stoch_rsi_k, stoch_rsi_d)
    Zone feature: 1 if >90, -1 if <10, else 0
    """
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic of RSI
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    stoch_rsi_k = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
    stoch_rsi_d = stoch_rsi_k.rolling(window=3).mean()
    
    return stoch_rsi_k, stoch_rsi_d


def calculate_stoch_rsi_zones(df: pd.DataFrame) -> pd.Series:
    """StochRSI Zone: 1 if >90, -1 if <10, else 0"""
    stoch_k, _ = calculate_stoch_rsi(df)
    zones = pd.Series(0, index=df.index)
    zones[stoch_k > 90] = 1
    zones[stoch_k < 10] = -1
    return zones


def calculate_fisher_transform(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Fisher Transform - Identifies turning points
    Sharp spikes indicate potential reversals
    """
    hl2 = (df['high'] + df['low']) / 2
    min_low = hl2.rolling(window=period).min()
    max_high = hl2.rolling(window=period).max()
    
    # Normalize price to -1 to 1 range
    value = 2 * ((hl2 - min_low) / (max_high - min_low) - 0.5)
    value = value.clip(-0.999, 0.999)  # Prevent infinity in log
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    fisher = fisher.ewm(span=period, adjust=False).mean()
    
    return fisher


def calculate_awesome_oscillator(df: pd.DataFrame) -> pd.Series:
    """
    Awesome Oscillator (AO)
    Measures market momentum using 5 and 34 period SMAs of median price
    """
    median_price = (df['high'] + df['low']) / 2
    ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
    return ao


def calculate_ao_zero_cross(df: pd.DataFrame) -> pd.Series:
    """AO Zero Cross: 1 if positive, -1 if negative"""
    ao = calculate_awesome_oscillator(df)
    return np.sign(ao)


def calculate_force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """
    Force Index - Trend strength using price change * volume
    Normalized by rolling standard deviation
    """
    force = df['close'].diff() * df['volume']
    force_ema = force.ewm(span=period, adjust=False).mean()
    
    # Normalize
    force_std = force_ema.rolling(window=period * 2).std()
    force_normalized = force_ema / force_std
    
    return force_normalized.fillna(0)


def calculate_bollinger_percent_b(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """
    Bollinger %B - Position within Bollinger Bands
    >1 = Breakout above upper band
    <0 = Oversold below lower band
    0.5 = At middle band
    """
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    percent_b = (df['close'] - lower) / (upper - lower)
    return percent_b


def calculate_bollinger_bandwidth(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """
    Bollinger Bandwidth - Normalized band width
    Low values indicate squeeze (potential explosion)
    """
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    bandwidth = (upper - lower) / sma * 100
    
    # Normalize to detect squeezes
    bw_min = bandwidth.rolling(window=period * 5).min()
    bw_max = bandwidth.rolling(window=period * 5).max()
    bw_normalized = (bandwidth - bw_min) / (bw_max - bw_min)
    
    return bw_normalized.fillna(0.5)


def calculate_atr_ratio(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR Ratio - Current range vs ATR
    High values indicate volatility spike/anomaly
    """
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    current_range = df['high'] - df['low']
    
    atr_ratio = current_range / atr
    return atr_ratio.fillna(1)


def calculate_donchian_proximity(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Donchian Channel Proximity
    1 = At period high, 0 = At period low
    """
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    
    proximity = (df['close'] - low_min) / (high_max - low_min)
    return proximity.fillna(0.5)


def calculate_hull_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Hull Moving Average (HMA) - Lag-free trend indicator
    """
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = df['close'].rolling(window=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    wma_full = df['close'].rolling(window=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.rolling(window=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    
    return hma


def calculate_hma_slope(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """HMA Slope: Current HMA - Previous HMA"""
    hma = calculate_hull_ma(df, period)
    return hma - hma.shift(1)


def calculate_zlema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Zero Lag EMA (ZLEMA) - Reduced lag EMA
    """
    lag = int((period - 1) / 2)
    ema_data = df['close'] + (df['close'] - df['close'].shift(lag))
    zlema = ema_data.ewm(span=period, adjust=False).mean()
    return zlema


def calculate_zlema_crossover(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """ZLEMA Crossover: Price - ZLEMA"""
    zlema = calculate_zlema(df, period)
    return df['close'] - zlema


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend Indicator
    Returns: (supertrend_line, trend_direction: 1=bullish, -1=bearish)
    """
    hl2 = (df['high'] + df['low']) / 2
    
    # Calculate ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate basic bands
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
    
    return supertrend, direction


def calculate_linear_regression_slope(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Linear Regression Slope - Direction of micro-trend
    """
    def linreg_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        y = series.values
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    slope = df['close'].rolling(window=period).apply(linreg_slope, raw=False)
    return slope


def calculate_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF)
    Positive = Accumulation, Negative = Distribution
    """
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    return cmf.fillna(0)


def calculate_z_score(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Z-Score - Standard deviations from mean
    """
    mean = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    z_score = (df['close'] - mean) / std
    return z_score.fillna(0)


def calculate_hurst_exponent_approx(df: pd.DataFrame, period: int = 100) -> pd.Series:
    """
    Approximate Hurst Exponent using R/S analysis
    H < 0.5 = Mean reverting
    H = 0.5 = Random walk
    H > 0.5 = Trending
    """
    def rs_analysis(series):
        if len(series) < 20:
            return 0.5
        
        # Mean-centered series
        mean = np.mean(series)
        y = series - mean
        
        # Cumulative deviation
        z = np.cumsum(y)
        
        # Range
        r = np.max(z) - np.min(z)
        
        # Standard deviation
        s = np.std(series, ddof=1)
        
        if s == 0:
            return 0.5
        
        # Simplified Hurst approximation
        rs = r / s
        n = len(series)
        h = np.log(rs) / np.log(n) if rs > 0 and n > 1 else 0.5
        
        return np.clip(h, 0, 1)
    
    hurst = df['close'].rolling(window=period).apply(rs_analysis, raw=True)
    return hurst.fillna(0.5)


def calculate_all_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all advanced indicators and add to dataframe"""
    result = df.copy()
    
    # Only calculate if we have enough data
    if len(df) < 50:
        return result
    
    print("  Calculating advanced indicators...")
    
    # VWAP
    if df['volume'].sum() > 0:
        result['vwap'] = calculate_vwap(df)
        result['vwap_distance'] = calculate_vwap_distance(df)
    
    # StochRSI
    stoch_k, stoch_d = calculate_stoch_rsi(df)
    result['stoch_rsi_k'] = stoch_k
    result['stoch_rsi_d'] = stoch_d
    result['stoch_rsi_zone'] = calculate_stoch_rsi_zones(df)
    
    # Fisher Transform
    result['fisher'] = calculate_fisher_transform(df)
    
    # Awesome Oscillator
    result['ao'] = calculate_awesome_oscillator(df)
    result['ao_cross'] = calculate_ao_zero_cross(df)
    
    # Force Index
    if df['volume'].sum() > 0:
        result['force_index'] = calculate_force_index(df)
    
    # Bollinger derivatives
    result['bb_percent_b'] = calculate_bollinger_percent_b(df)
    result['bb_bandwidth'] = calculate_bollinger_bandwidth(df)
    
    # ATR Ratio
    result['atr_ratio'] = calculate_atr_ratio(df)
    
    # Donchian Proximity
    result['donchian_prox'] = calculate_donchian_proximity(df)
    
    # Hull MA
    result['hma'] = calculate_hull_ma(df)
    result['hma_slope'] = calculate_hma_slope(df)
    
    # ZLEMA
    result['zlema'] = calculate_zlema(df)
    result['zlema_cross'] = calculate_zlema_crossover(df)
    
    # Supertrend
    st_line, st_dir = calculate_supertrend(df)
    result['supertrend'] = st_line
    result['supertrend_dir'] = st_dir
    
    # Linear Regression
    result['linreg_slope'] = calculate_linear_regression_slope(df)
    
    # CMF
    if df['volume'].sum() > 0:
        result['cmf'] = calculate_chaikin_money_flow(df)
    
    # Z-Score
    result['z_score'] = calculate_z_score(df)
    
    # Hurst (computationally expensive, use larger period)
    if len(df) >= 100:
        result['hurst'] = calculate_hurst_exponent_approx(df)
    
    print("  Advanced indicators complete.")
    return result



