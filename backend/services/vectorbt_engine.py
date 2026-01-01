"""
VECTORBT BACKTESTING ENGINE
============================
High-performance vectorized backtesting using VectorBT.
Provides 100x+ speedup over iterative bar-by-bar approach.

Usage:
    from services.vectorbt_engine import VectorBTEngine

    engine = VectorBTEngine(df, config)
    results = engine.run_optimization(strategies, tp_range, sl_range, ...)
"""
import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# Suppress VectorBT warnings during import
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import vectorbt as vbt
    from numba import njit
    VECTORBT_AVAILABLE = True

    # ============================================
    # VECTORBT PERFORMANCE CONFIGURATION
    # ============================================
    # Enable Numba JIT for portfolio simulation
    vbt.settings.portfolio['use_numba'] = True

    # Disable type checking for faster Numba compilation
    vbt.settings.numba['check_func_type'] = False
    vbt.settings.numba['check_func_suffix'] = False

    # Silence warnings for cleaner output
    vbt.settings.array_wrapper['silence_warnings'] = True

except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    njit = lambda f: f  # No-op decorator if numba not available

import os
import psutil
from logging_config import log
from config import DEFAULT_TRADING_COSTS

# VectorBT logging configuration
VBT_LOG_LEVEL = os.getenv('VBT_LOG_LEVEL', 'INFO')  # DEBUG, INFO, WARNING
VBT_VERBOSE = VBT_LOG_LEVEL == 'DEBUG'

def vbt_log(message: str, level: str = 'INFO'):
    """Log VectorBT messages based on configured level."""
    levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
    msg_level = levels.get(level, 1)
    config_level = levels.get(VBT_LOG_LEVEL, 1)

    if msg_level >= config_level:
        log(message, level=level)


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


@dataclass
class VectorBTResult:
    """Result container compatible with StrategyResult."""
    strategy_name: str
    strategy_category: str
    direction: str
    tp_percent: float
    sl_percent: float
    entry_rule: str

    # Core metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    avg_trade: float = 0.0
    avg_trade_percent: float = 0.0

    # Buy & Hold comparison
    buy_hold_return: float = 0.0
    vs_buy_hold: float = 0.0
    beats_buy_hold: bool = False

    # Composite score
    composite_score: float = 0.0

    # Data for detailed analysis
    params: Dict = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)
    trades_list: List[Dict] = field(default_factory=list)

    # Open position tracking
    has_open_position: bool = False
    open_position: Dict = None

    # Period metrics
    period_metrics: Dict = None
    consistency_score: float = 0.0

    # GBP conversion
    total_pnl_gbp: float = 0.0
    max_drawdown_gbp: float = 0.0

    # Data range (for period display in Strategy History)
    data_start: str = None
    data_end: str = None
    avg_trade_gbp: float = 0.0
    equity_curve_gbp: List[float] = field(default_factory=list)
    source_currency: str = "USD"
    display_currencies: List[str] = field(default_factory=lambda: ["USD", "GBP"])

    # Bidirectional fields
    long_trades: int = 0
    long_wins: int = 0
    long_pnl: float = 0.0
    short_trades: int = 0
    short_wins: int = 0
    short_pnl: float = 0.0
    flip_count: int = 0


class VectorBTEngine:
    """
    High-performance backtesting engine using VectorBT.

    Key features:
    - Vectorized operations (100x faster than iterative)
    - Broadcasting for parameter sweeps
    - Compatible with existing StrategyEngine output format
    """

    # Entry strategies - must match StrategyEngine.ENTRY_STRATEGIES keys
    ENTRY_STRATEGIES = {
        'rsi_extreme': {'category': 'Momentum', 'description': 'RSI crosses overbought/oversold'},
        'rsi_cross_50': {'category': 'Momentum', 'description': 'RSI crosses 50'},
        'stoch_extreme': {'category': 'Momentum', 'description': 'Stochastic extreme levels'},
        'bb_touch': {'category': 'Volatility', 'description': 'Price touches Bollinger Band'},
        'bb_squeeze_breakout': {'category': 'Volatility', 'description': 'BB squeeze breakout'},
        'ema_cross': {'category': 'Trend', 'description': 'EMA 9/21 crossover'},
        'sma_cross': {'category': 'Trend', 'description': 'SMA fast/slow crossover'},
        'macd_cross': {'category': 'Momentum', 'description': 'MACD histogram crosses zero'},
        'supertrend': {'category': 'Trend', 'description': 'Supertrend direction change'},
        'consecutive_candles': {'category': 'Price Action', 'description': 'N consecutive up/down closes'},
        'engulfing': {'category': 'Pattern', 'description': 'Engulfing candle pattern'},
        'inside_bar': {'category': 'Pattern', 'description': 'Inside bar pattern'},
        'outside_bar': {'category': 'Pattern', 'description': 'Outside bar pattern'},
        'atr_breakout': {'category': 'Volatility', 'description': 'ATR-based breakout'},
        # Missing strategies from StrategyEngine (8 core strategies)
        'always': {'category': 'Baseline', 'description': 'Enter on every bar - tests pure TP/SL effectiveness'},
        'price_vs_sma': {'category': 'Mean Reversion', 'description': 'Price 1%+ away from SMA20'},
        'price_above_sma': {'category': 'Trend', 'description': 'Price crosses SMA20'},
        'big_candle': {'category': 'Pattern', 'description': 'Large candle 2x ATR in opposite direction'},
        'doji_reversal': {'category': 'Pattern', 'description': 'Doji candle after trend'},
        'low_volatility_breakout': {'category': 'Volatility', 'description': 'Breakout after low volatility period'},
        'higher_low': {'category': 'Price Action', 'description': 'Higher low (long) or lower high (short)'},
        'support_resistance': {'category': 'Price Action', 'description': 'Price at recent support/resistance level'},
        # Additional strategies from StrategyEngine
        'williams_r': {'category': 'Momentum', 'description': 'Williams %R < -80 (long) or > -20 (short)'},
        'cci_extreme': {'category': 'Momentum', 'description': 'CCI < -100 (long) or > 100 (short)'},
        'adx_strong_trend': {'category': 'Trend', 'description': 'ADX > 25 with DI+ or DI- dominance'},
        'psar_reversal': {'category': 'Trend', 'description': 'Price crosses Parabolic SAR'},
        'vwap_bounce': {'category': 'Mean Reversion', 'description': 'Price bounces off VWAP'},
        'rsi_divergence': {'category': 'Momentum', 'description': 'Price makes new low but RSI doesnt (bullish divergence)'},
        'keltner_breakout': {'category': 'Volatility', 'description': 'Price breaks above/below Keltner Channel'},
        'donchian_breakout': {'category': 'Trend', 'description': 'Price breaks above/below Donchian Channel (Turtle Trading)'},
        # Combo strategies
        'bb_rsi_combo': {'category': 'Mean Reversion', 'description': 'Bollinger Band touch + RSI extreme'},
        'supertrend_adx_combo': {'category': 'Trend', 'description': 'Supertrend signal + ADX > 25 filter'},
        'ema_rsi_combo': {'category': 'Trend', 'description': 'EMA cross + RSI confirmation'},
        'macd_stoch_combo': {'category': 'Momentum', 'description': 'MACD cross + Stochastic confirmation'},
        # VWMA strategies
        'vwma_cross': {'category': 'Trend Following', 'description': 'Price crosses VWMA'},
        'vwma_trend': {'category': 'Trend Following', 'description': 'VWMA direction change'},
        # Pivot Points
        'pivot_bounce': {'category': 'Price Action', 'description': 'Price bounces off pivot point levels'},
        # Linear Regression
        'linreg_channel': {'category': 'Trend', 'description': 'Price touches/breaks linear regression channel'},
        # Awesome Oscillator
        'ao_zero_cross': {'category': 'Momentum', 'description': 'Awesome Oscillator crosses zero'},
        'ao_twin_peaks': {'category': 'Momentum', 'description': 'Awesome Oscillator twin peaks pattern'},
        # Elder Ray
        'elder_ray': {'category': 'Momentum', 'description': 'Bull/Bear power with EMA trend filter'},
        # RSI + MACD Combo
        'rsi_macd_combo': {'category': 'Momentum', 'description': 'RSI extreme + MACD confirmation'},
        # Advanced trend strategies
        'triple_ema': {'category': 'Trend', 'description': 'EMA 9 > EMA 21 > EMA 50 alignment'},
        'mcginley_cross': {'category': 'Trend', 'description': 'Price crosses McGinley Dynamic'},
        'mcginley_trend': {'category': 'Trend', 'description': 'McGinley Dynamic changes slope'},
        'hull_ma_cross': {'category': 'Trend', 'description': 'Price crosses Hull Moving Average'},
        'hull_ma_turn': {'category': 'Trend', 'description': 'Hull MA changes direction'},
        'zlema_cross': {'category': 'Trend', 'description': 'Price crosses Zero-Lag EMA'},
        'chandelier_entry': {'category': 'Volatility', 'description': 'Chandelier Exit signal'},
        'tsi_cross': {'category': 'Momentum', 'description': 'TSI crosses signal line'},
        # New momentum and volatility strategies
        'tsi_zero': {'category': 'Momentum', 'description': 'TSI crosses zero line'},
        'cmf_cross': {'category': 'Momentum', 'description': 'Chaikin Money Flow crosses zero'},
        'obv_trend': {'category': 'Momentum', 'description': 'OBV makes new high/low with price'},
        'mfi_extreme': {'category': 'Momentum', 'description': 'MFI < 20 (long) or > 80 (short)'},
        'ppo_cross': {'category': 'Momentum', 'description': 'PPO crosses signal line'},
        'fisher_cross': {'category': 'Momentum', 'description': 'Fisher Transform crosses signal line'},
        'squeeze_momentum': {'category': 'Volatility', 'description': 'BB inside Keltner + momentum direction'},
        'vwap_cross': {'category': 'Mean Reversion', 'description': 'Price crosses VWAP'},
        # Ichimoku strategies
        'ichimoku_cross': {'category': 'Trend', 'description': 'Tenkan-sen crosses Kijun-sen (Ichimoku Cloud)'},
        'ichimoku_cloud': {'category': 'Trend', 'description': 'Price breaks above/below Ichimoku Cloud'},
        # Trend strategies
        'aroon_cross': {'category': 'Trend', 'description': 'Aroon Up crosses Aroon Down'},
        'double_ema_cross': {'category': 'Trend', 'description': 'EMA 12/26 Cross (same periods as MACD)'},
        # Momentum strategies
        'momentum_zero': {'category': 'Momentum', 'description': 'Momentum crosses above/below zero line'},
        'roc_extreme': {'category': 'Momentum', 'description': 'Rate of Change at extreme levels (5th/95th percentile)'},
        'uo_extreme': {'category': 'Momentum', 'description': 'Ultimate Oscillator < 30 (long) or > 70 (short)'},
        # Volatility strategies
        'chop_trend': {'category': 'Volatility', 'description': 'Choppiness Index < 38.2 indicates trending market'},
        # Kalman Filter strategies
        'kalman_trend': {'category': 'Trend', 'description': 'Price crosses Kalman filter line'},
        'kalman_bb': {'category': 'Mean Reversion', 'description': 'Price touches Kalman-based bands'},
        'kalman_rsi': {'category': 'Momentum', 'description': 'Kalman-smoothed RSI crosses 30/70'},
        'kalman_mfi': {'category': 'Momentum', 'description': 'Kalman-smoothed MFI crosses 20/80'},
        'kalman_adx': {'category': 'Trend', 'description': 'Kalman ADX > 25 with DI dominance'},
        'kalman_psar': {'category': 'Trend', 'description': 'Price crosses Kalman-smoothed Parabolic SAR'},
        'kalman_macd': {'category': 'Momentum', 'description': 'Kalman-smoothed MACD signal cross'},
    }

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 1000.0,
        position_size_pct: float = 100.0,
        commission_pct: float = None,
        spread_pct: float = None,
        slippage_pct: float = None,
    ):
        """
        Initialize VectorBT engine with OHLCV data.

        Args:
            df: DataFrame with columns: time, open, high, low, close, volume
            initial_capital: Starting capital
            position_size_pct: Position size as % of equity
            commission_pct: Trading commission (default from config)
            spread_pct: Bid-ask spread (default from config)
            slippage_pct: Slippage estimate (default from config)
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is not installed. Run: pip install vectorbt numba")

        self.df = df.copy()
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct

        # Trading costs
        costs = DEFAULT_TRADING_COSTS
        self.commission_pct = commission_pct or costs["commission_pct"]
        self.spread_pct = spread_pct or costs["spread_pct"]
        self.slippage_pct = slippage_pct or costs["slippage_pct"]

        # Combined fees for VectorBT (as decimal)
        # Round-trip: 2 * commission + spread + slippage
        self.total_fees = (2 * self.commission_pct + self.spread_pct + self.slippage_pct) / 100

        # Ensure we have required columns
        self._prepare_dataframe()

        # Pre-calculate indicators
        self._calculate_indicators()

        # LRU signal cache with thread safety for concurrent access
        self._signal_cache: OrderedDict[str, pd.Series] = OrderedDict()
        self._signal_cache_max_size = 50  # Keep last 50 strategy/direction combinations
        self._signal_cache_lock = threading.RLock()

        # Detect data frequency for accurate annualized metrics
        self.data_freq = self._detect_frequency()

        vbt_log(f"[VectorBT] Engine initialized: {len(df)} bars, capital=${initial_capital:,.0f}, position_size={position_size_pct}%", level='DEBUG')
        vbt_log(f"[VectorBT] Data frequency: {self.data_freq}, Trading costs: commission={self.commission_pct}%, spread={self.spread_pct}%, slippage={self.slippage_pct}%", level='DEBUG')

    def _detect_frequency(self) -> str:
        """Detect data frequency from DataFrame index for accurate annualized metrics."""
        if len(self.df) < 2:
            return '1D'

        # Calculate average time delta between bars
        time_diff = pd.Series(self.df.index).diff().mean()

        if time_diff <= pd.Timedelta(minutes=5):
            return '5T'  # 5 minutes
        elif time_diff <= pd.Timedelta(minutes=15):
            return '15T'
        elif time_diff <= pd.Timedelta(hours=1):
            return '1H'
        elif time_diff <= pd.Timedelta(hours=4):
            return '4H'
        elif time_diff <= pd.Timedelta(days=1):
            return '1D'
        else:
            return '1W'

    def _prepare_dataframe(self):
        """Prepare and validate DataFrame for VectorBT backtesting."""

        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Validate data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                log(f"[VectorBT] Converting {col} to numeric", level='WARNING')
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Check for NaN values
        nan_counts = self.df[required].isna().sum()
        if nan_counts.any():
            log(f"[VectorBT] NaN values detected: {nan_counts[nan_counts > 0].to_dict()}", level='WARNING')
            # Forward fill NaN values
            self.df[required] = self.df[required].ffill()

        # Validate data integrity
        if len(self.df) < 50:
            raise ValueError(f"Insufficient data: {len(self.df)} rows (minimum 50 required)")

        # Check for price anomalies
        price_range = self.df['close'].max() / self.df['close'].min()
        if price_range > 100:
            log(f"[VectorBT] Large price range detected ({price_range:.1f}x) - verify data quality", level='WARNING')

        # Ensure time index for VectorBT
        if 'time' in self.df.columns:
            self.df.set_index('time', inplace=True)

        # Add helper columns
        self.df['range'] = self.df['high'] - self.df['low']
        self.df['body'] = abs(self.df['close'] - self.df['open'])
        self.df['green'] = self.df['close'] > self.df['open']
        self.df['red'] = self.df['close'] < self.df['open']
        # Doji detection: body less than 10% of range (matches StrategyEngine)
        self.df['doji'] = self.df['body'] < self.df['range'] * 0.1

    def _calculate_indicators(self):
        """Calculate technical indicators using VectorBT where available."""
        df = self.df
        close = df['close']
        high = df['high']
        low = df['low']

        # RSI
        rsi = vbt.RSI.run(close, window=14)
        df['rsi'] = rsi.rsi.values

        # Stochastic
        stoch = vbt.STOCH.run(high, low, close, k_window=14, d_window=3)
        df['stoch_k'] = stoch.percent_k.values
        df['stoch_d'] = stoch.percent_d.values

        # Bollinger Bands
        bb = vbt.BBANDS.run(close, window=20, alpha=2)
        df['bb_upper'] = bb.upper.values
        df['bb_mid'] = bb.middle.values
        df['bb_lower'] = bb.lower.values
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        # Moving Averages
        df['sma_20'] = vbt.MA.run(close, window=20).ma.values
        df['ema_9'] = vbt.MA.run(close, window=9, ewm=True).ma.values
        df['ema_21'] = vbt.MA.run(close, window=21, ewm=True).ma.values

        # MACD
        macd = vbt.MACD.run(close, fast_window=12, slow_window=26, signal_window=9)
        df['macd'] = macd.macd.values
        df['macd_signal'] = macd.signal.values
        df['macd_hist'] = macd.hist.values

        # ATR
        atr = vbt.ATR.run(high, low, close, window=14)
        df['atr'] = atr.atr.values

        # Supertrend (using built-in if available, else calculate)
        try:
            st = vbt.SUPERTREND.run(high, low, close, period=10, multiplier=3.0)
            df['supertrend'] = st.supert.values
            df['supertrend_dir'] = st.superd.values
        except AttributeError:
            # Fallback: calculate manually
            df['supertrend'] = df['close']  # Simplified
            df['supertrend_dir'] = 1

        # ADX (Average Directional Index) - manual calculation for combo strategies
        # Using Wilder's smoothing method
        period = 14
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high - high.shift(1)
        minus_dm = low.shift(1) - low
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Wilder's smoothing (exponential with alpha = 1/period)
        alpha = 1 / period
        atr_adx = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_adx)
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_adx)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=alpha, adjust=False).mean()
        df['di_plus'] = plus_di
        df['di_minus'] = minus_di

        # === Williams %R ===
        willr_period = 14
        highest_high = high.rolling(willr_period).max()
        lowest_low_willr = low.rolling(willr_period).min()
        df['willr'] = ((highest_high - close) / (highest_high - lowest_low_willr)) * -100

        # === CCI (Commodity Channel Index) ===
        cci_period = 20
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(cci_period).mean()
        mean_deviation = typical_price.rolling(cci_period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['cci'] = (typical_price - sma_tp) / (0.015 * mean_deviation)

        # === Parabolic SAR ===
        af_start, af_increment, af_max = 0.02, 0.02, 0.2
        psar = pd.Series(index=df.index, dtype=float)
        psar_af = pd.Series(index=df.index, dtype=float)
        psar_ep = pd.Series(index=df.index, dtype=float)
        psar_trend = pd.Series(index=df.index, dtype=float)
        psar.iloc[0], psar_af.iloc[0], psar_ep.iloc[0], psar_trend.iloc[0] = low.iloc[0], af_start, high.iloc[0], 1
        for i in range(1, len(df)):
            prev_psar, prev_af, prev_ep, prev_trend = psar.iloc[i-1], psar_af.iloc[i-1], psar_ep.iloc[i-1], psar_trend.iloc[i-1]
            if prev_trend == 1:
                new_psar = prev_psar + prev_af * (prev_ep - prev_psar)
                new_psar = min(new_psar, low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
                if low.iloc[i] < new_psar:
                    psar_trend.iloc[i], psar.iloc[i], psar_ep.iloc[i], psar_af.iloc[i] = -1, prev_ep, low.iloc[i], af_start
                else:
                    psar_trend.iloc[i], psar.iloc[i] = 1, new_psar
                    if high.iloc[i] > prev_ep:
                        psar_ep.iloc[i], psar_af.iloc[i] = high.iloc[i], min(prev_af + af_increment, af_max)
                    else:
                        psar_ep.iloc[i], psar_af.iloc[i] = prev_ep, prev_af
            else:
                new_psar = prev_psar + prev_af * (prev_ep - prev_psar)
                new_psar = max(new_psar, high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
                if high.iloc[i] > new_psar:
                    psar_trend.iloc[i], psar.iloc[i], psar_ep.iloc[i], psar_af.iloc[i] = 1, prev_ep, high.iloc[i], af_start
                else:
                    psar_trend.iloc[i], psar.iloc[i] = -1, new_psar
                    if low.iloc[i] < prev_ep:
                        psar_ep.iloc[i], psar_af.iloc[i] = low.iloc[i], min(prev_af + af_increment, af_max)
                    else:
                        psar_ep.iloc[i], psar_af.iloc[i] = prev_ep, prev_af
        df['psar'] = psar

        # === VWAP (Volume Weighted Average Price) ===
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        if volume.sum() > 0:
            cumulative_tp_vol = (typical_price * volume).cumsum()
            cumulative_vol = volume.cumsum()
            df['vwap'] = cumulative_tp_vol / cumulative_vol
        else:
            df['vwap'] = df['sma_20']

        # === Keltner Channels ===
        kc_length, kc_mult, kc_atr_length = 20, 2.0, 10
        df['kc_mid'] = close.ewm(span=kc_length, adjust=False).mean()
        kc_atr = vbt.ATR.run(high, low, close, window=kc_atr_length).atr.values
        df['kc_upper'] = df['kc_mid'] + kc_mult * kc_atr
        df['kc_lower'] = df['kc_mid'] - kc_mult * kc_atr

        # === Donchian Channels ===
        dc_length = 20
        df['dc_upper'] = high.rolling(dc_length).max()
        df['dc_lower'] = low.rolling(dc_length).min()
        df['dc_mid'] = (df['dc_upper'] + df['dc_lower']) / 2

        # === VWMA (Volume Weighted Moving Average) ===
        vwma_length = 20
        df['vwma'] = (df['close'] * df['volume']).rolling(vwma_length).sum() / df['volume'].rolling(vwma_length).sum()

        # === Pivot Points (Classic) ===
        # Using previous bar's HLC for pivot calculation
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['pivot_r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['pivot_s1'] = 2 * df['pivot'] - df['high'].shift(1)

        # === Awesome Oscillator ===
        hl2 = (df['high'] + df['low']) / 2
        ao_fast = 5
        ao_slow = 34
        df['ao'] = hl2.rolling(ao_fast).mean() - hl2.rolling(ao_slow).mean()

        # === Elder Ray (Bull/Bear Power) ===
        df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
        df['bull_power'] = df['high'] - df['ema_13']
        df['bear_power'] = df['low'] - df['ema_13']

        # === Linear Regression Channel ===
        linreg_period = 50

        def calc_linreg(data):
            """Calculate linear regression value for the last point."""
            if len(data) < linreg_period:
                return np.nan
            x = np.arange(linreg_period)
            y = data.values[-linreg_period:]
            if np.any(np.isnan(y)):
                return np.nan
            slope, intercept = np.polyfit(x, y, 1)
            return intercept + slope * (linreg_period - 1)

        def calc_linreg_std(data):
            """Calculate standard deviation from linear regression line."""
            if len(data) < linreg_period:
                return np.nan
            x = np.arange(linreg_period)
            y = data.values[-linreg_period:]
            if np.any(np.isnan(y)):
                return np.nan
            slope, intercept = np.polyfit(x, y, 1)
            pred = intercept + slope * x
            return np.std(y - pred)

        df['linreg'] = df['close'].rolling(linreg_period).apply(calc_linreg, raw=False)
        df['linreg_std'] = df['close'].rolling(linreg_period).apply(calc_linreg_std, raw=False)
        df['linreg_upper'] = df['linreg'] + 2 * df['linreg_std']
        df['linreg_lower'] = df['linreg'] - 2 * df['linreg_std']

        # === ICHIMOKU CLOUD ===
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        tenkan_period = 9
        df['tenkan'] = (df['high'].rolling(tenkan_period).max() + df['low'].rolling(tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        kijun_period = 26
        df['kijun'] = (df['high'].rolling(kijun_period).max() + df['low'].rolling(kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
        senkou_period = 52
        df['senkou_b'] = (df['high'].rolling(senkou_period).max() + df['low'].rolling(senkou_period).min()) / 2

        # === AROON OSCILLATOR ===
        aroon_period = 14
        df['aroon_up'] = 100 * (aroon_period - df['high'].rolling(aroon_period + 1).apply(lambda x: aroon_period - x.argmax(), raw=True)) / aroon_period
        df['aroon_down'] = 100 * (aroon_period - df['low'].rolling(aroon_period + 1).apply(lambda x: aroon_period - x.argmin(), raw=True)) / aroon_period

        # === MOMENTUM ===
        mom_period = 10
        df['mom'] = df['close'] - df['close'].shift(mom_period)

        # === RATE OF CHANGE (ROC) ===
        roc_period = 9
        df['roc'] = ((df['close'] - df['close'].shift(roc_period)) / df['close'].shift(roc_period)) * 100

        # === ULTIMATE OSCILLATOR ===
        uo_fast = 7
        uo_mid = 14
        uo_slow = 28
        
        # Buying Pressure = Close - Min(Low, Previous Close)
        prev_close = df['close'].shift(1)
        bp = df['close'] - pd.concat([df['low'], prev_close], axis=1).min(axis=1)
        
        # True Range = Max(High, Previous Close) - Min(Low, Previous Close)
        tr_uo = pd.concat([df['high'], prev_close], axis=1).max(axis=1) - pd.concat([df['low'], prev_close], axis=1).min(axis=1)
        
        # Average BP and TR for each period
        avg_bp_fast = bp.rolling(uo_fast).sum()
        avg_tr_fast = tr_uo.rolling(uo_fast).sum()
        avg_bp_mid = bp.rolling(uo_mid).sum()
        avg_tr_mid = tr_uo.rolling(uo_mid).sum()
        avg_bp_slow = bp.rolling(uo_slow).sum()
        avg_tr_slow = tr_uo.rolling(uo_slow).sum()
        
        # UO = 100 * ((4 * fast_avg) + (2 * mid_avg) + slow_avg) / 7
        df['uo'] = 100 * ((4 * avg_bp_fast / avg_tr_fast) + (2 * avg_bp_mid / avg_tr_mid) + (avg_bp_slow / avg_tr_slow)) / 7

        # === CHOPPINESS INDEX ===
        chop_period = 14
        atr_sum = df['atr'].rolling(chop_period).sum()
        high_max = df['high'].rolling(chop_period).max()
        low_min = df['low'].rolling(chop_period).min()
        df['chop'] = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(chop_period)

        # === EMA 12 and EMA 26 for double_ema_cross ===
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # === EMA 50 (for triple_ema strategy) ===
        df['ema_50'] = vbt.MA.run(close, window=50, ewm=True).ma.values

        # === McGinley Dynamic ===
        # Formula: MD = MD_prev + (Close - MD_prev) / (k * n * (Close/MD_prev)^4)
        mcg_length = 14
        mcg_k = 0.6
        close_vals = close.values
        n_bars = len(close_vals)
        md = np.full(n_bars, np.nan)
        md[0] = close_vals[0]
        for i in range(1, n_bars):
            if md[i-1] == 0 or np.isnan(md[i-1]):
                md[i] = close_vals[i]
            else:
                ratio = close_vals[i] / md[i-1]
                ratio = max(0.5, min(ratio, 2.0))  # Clamp ratio for stability
                divisor = mcg_k * mcg_length * (ratio ** 4)
                divisor = max(divisor, 0.001)  # Prevent division by zero
                md[i] = md[i-1] + (close_vals[i] - md[i-1]) / divisor
        df['mcginley'] = md

        # === Hull Moving Average ===
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        hull_period = 20
        half_period = hull_period // 2
        sqrt_period = int(np.sqrt(hull_period))
        wma1 = close.rolling(half_period).apply(lambda x: np.sum(x * np.arange(1, half_period+1)) / np.sum(np.arange(1, half_period+1)), raw=True)
        wma2 = close.rolling(hull_period).apply(lambda x: np.sum(x * np.arange(1, hull_period+1)) / np.sum(np.arange(1, hull_period+1)), raw=True)
        raw_hma = 2 * wma1 - wma2
        df['hull_ma'] = raw_hma.rolling(sqrt_period).apply(lambda x: np.sum(x * np.arange(1, sqrt_period+1)) / np.sum(np.arange(1, sqrt_period+1)), raw=True)

        # === ZLEMA (Zero-Lag EMA) ===
        zlema_period = 20
        zlema_lag = (zlema_period - 1) // 2
        ema_data = close + (close - close.shift(zlema_lag))
        df['zlema'] = ema_data.ewm(span=zlema_period, adjust=False).mean()

        # === Chandelier Exit ===
        chandelier_period = 22
        chandelier_mult = 3.0
        df['chandelier_long'] = high.rolling(chandelier_period).max() - df['atr'] * chandelier_mult
        df['chandelier_short'] = low.rolling(chandelier_period).min() + df['atr'] * chandelier_mult

        # === TSI (True Strength Index) ===
        close_diff = close.diff()
        double_smooth_pc = close_diff.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        double_smooth_apc = close_diff.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        df['tsi'] = 100 * (double_smooth_pc / double_smooth_apc.replace(0, np.nan))
        df['tsi_signal'] = df['tsi'].ewm(span=7, adjust=False).mean()

        # === CMF (Chaikin Money Flow) ===
        cmf_length = 20
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        mfv = mfm * df['volume']
        df['cmf'] = mfv.rolling(cmf_length).sum() / df['volume'].rolling(cmf_length).sum()

        # === OBV (On Balance Volume) ===
        obv_sign = np.sign(close.diff()).fillna(0)
        df['obv'] = (obv_sign * df['volume']).cumsum()

        # === MFI (Money Flow Index) ===
        mfi_length = 14
        typical_price_mfi = (high + low + close) / 3
        raw_mf = typical_price_mfi * df['volume']
        positive_mf = raw_mf.where(typical_price_mfi > typical_price_mfi.shift(1), 0).rolling(mfi_length).sum()
        negative_mf = raw_mf.where(typical_price_mfi < typical_price_mfi.shift(1), 0).rolling(mfi_length).sum()
        mf_ratio = positive_mf / negative_mf.replace(0, np.nan)
        df['mfi'] = 100 - (100 / (1 + mf_ratio))

        # === PPO (Percentage Price Oscillator) ===
        ema_fast_ppo = close.ewm(span=12, adjust=False).mean()
        ema_slow_ppo = close.ewm(span=26, adjust=False).mean()
        df['ppo'] = ((ema_fast_ppo - ema_slow_ppo) / ema_slow_ppo) * 100
        df['ppo_signal'] = df['ppo'].ewm(span=9, adjust=False).mean()

        # === Fisher Transform ===
        fisher_period = 10
        hl2_fisher = (high + low) / 2
        hl2_min = hl2_fisher.rolling(fisher_period).min()
        hl2_max = hl2_fisher.rolling(fisher_period).max()
        value = 2 * ((hl2_fisher - hl2_min) / (hl2_max - hl2_min).replace(0, np.nan)) - 1
        value = value.clip(-0.999, 0.999)
        df['fisher'] = 0.5 * np.log((1 + value) / (1 - value))
        df['fisher'] = df['fisher'].ewm(span=3, adjust=False).mean()
        df['fisher_signal'] = df['fisher'].shift(1)

        # === Keltner Channel (for squeeze_momentum) ===
        kc_length, kc_mult = 20, 1.5
        kc_mid = close.ewm(span=kc_length, adjust=False).mean()
        kc_range = df['atr'].rolling(kc_length).mean() if 'atr' in df.columns else df['range'].rolling(kc_length).mean()
        df['kc_upper'] = kc_mid + kc_mult * kc_range
        df['kc_lower'] = kc_mid - kc_mult * kc_range

        # === VWAP (Volume Weighted Average Price) ===
        typical_price_vwap = (high + low + close) / 3
        df['vwap'] = (typical_price_vwap * df['volume']).cumsum() / df['volume'].cumsum()

        # === KALMAN FILTER ===
        # Kalman Filter with velocity tracking for adaptive smoothing
        kalman_gain = 0.7
        close_vals = close.values
        n_bars = len(close_vals)
        kf = np.full(n_bars, np.nan)
        velocity = 0.0

        # Find first valid (non-NaN) value for initialization
        first_valid_idx = 0
        for idx in range(n_bars):
            if not np.isnan(close_vals[idx]):
                first_valid_idx = idx
                kf[idx] = close_vals[idx]
                break

        for i in range(first_valid_idx + 1, n_bars):
            # Handle NaN in current close value
            if np.isnan(close_vals[i]):
                kf[i] = kf[i-1]  # Hold previous value
                continue
            # Handle NaN in previous Kalman value (shouldn't happen after init, but safety)
            if np.isnan(kf[i-1]):
                kf[i] = close_vals[i]
                velocity = 0.0
                continue
            prediction = kf[i-1] + velocity
            error = close_vals[i] - prediction
            kf[i] = prediction + kalman_gain * error
            velocity = velocity + kalman_gain * error
        df['kalman'] = kf

        # Kalman Bands (Kalman filter as center with standard deviation bands)
        kalman_std = close.rolling(20).std()
        df['kalman_upper'] = df['kalman'] + kalman_std * 2.0
        df['kalman_lower'] = df['kalman'] - kalman_std * 2.0

        # Kalman-smoothed indicators (simplified Kalman smoothing)
        def kalman_smooth(series, gain=0.5):
            """Apply simplified Kalman smoothing to a series."""
            vals = series.values
            n = len(vals)
            result = np.full(n, np.nan)

            # Find first valid (non-NaN) value for initialization
            first_valid_idx = 0
            for idx in range(n):
                if not np.isnan(vals[idx]):
                    first_valid_idx = idx
                    result[idx] = vals[idx]
                    break

            for i in range(first_valid_idx + 1, n):
                if np.isnan(vals[i]):
                    result[i] = result[i-1]  # Hold previous value
                elif np.isnan(result[i-1]):
                    result[i] = vals[i]  # Reset from new value
                else:
                    result[i] = result[i-1] + gain * (vals[i] - result[i-1])
            return pd.Series(result, index=series.index)

        df['kalman_rsi'] = kalman_smooth(df['rsi'])
        df['kalman_mfi'] = kalman_smooth(df['mfi'])
        df['kalman_adx'] = kalman_smooth(df['adx'])
        df['kalman_macd'] = kalman_smooth(df['macd'])
        df['kalman_macd_signal'] = kalman_smooth(df['macd_signal'])
        df['kalman_psar'] = kalman_smooth(df['psar'])

        # Also need plus_di and minus_di for kalman_adx strategy
        df['plus_di'] = df['di_plus']
        df['minus_di'] = df['di_minus']

        vbt_log(f"[VectorBT] Calculated {len([c for c in df.columns if c not in ['open','high','low','close','volume']])} indicators", level='DEBUG')

    def _get_signals(self, strategy: str, direction: str) -> pd.Series:
        """
        Generate entry signals for a strategy.
        Mirrors StrategyEngine._get_signals() for compatibility.
        Uses thread-safe caching to avoid recalculating signals for the same strategy/direction.
        """
        cache_key = f"{strategy}_{direction}"

        # Thread-safe cache check
        with self._signal_cache_lock:
            if cache_key in self._signal_cache:
                self._signal_cache.move_to_end(cache_key)  # Mark as recently used
                return self._signal_cache[cache_key]

        df = self.df

        def safe_bool(series):
            return series.fillna(False).astype(bool)

        def cache_and_return(result: pd.Series) -> pd.Series:
            """Thread-safe cache storage with LRU eviction."""
            with self._signal_cache_lock:
                # LRU eviction if at capacity
                while len(self._signal_cache) >= self._signal_cache_max_size:
                    self._signal_cache.popitem(last=False)  # Remove oldest
                self._signal_cache[cache_key] = result
                self._signal_cache.move_to_end(cache_key)
            return result

        if strategy == 'always':
            return cache_and_return(pd.Series(True, index=df.index))

        elif strategy == 'rsi_extreme':
            if direction == 'long':
                return cache_and_return(safe_bool((df['rsi'] > 30) & (df['rsi'].shift(1) <= 30)))
            else:
                return cache_and_return(safe_bool((df['rsi'] < 70) & (df['rsi'].shift(1) >= 70)))

        elif strategy == 'rsi_cross_50':
            if direction == 'long':
                return cache_and_return(safe_bool((df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)))
            else:
                return cache_and_return(safe_bool((df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)))

        elif strategy == 'stoch_extreme':
            if direction == 'long':
                k_cross = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
                return cache_and_return(safe_bool(k_cross & (df['stoch_k'] < 20)))
            else:
                k_cross = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
                return cache_and_return(safe_bool(k_cross & (df['stoch_k'] > 80)))

        elif strategy == 'bb_touch':
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['bb_lower']) & (df['close'].shift(1) <= df['bb_lower'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['bb_upper']) & (df['close'].shift(1) >= df['bb_upper'].shift(1))))

        elif strategy == 'bb_squeeze_breakout':
            squeeze = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8
            expanding = df['bb_width'] > df['bb_width'].shift(1)
            if direction == 'long':
                return cache_and_return(safe_bool(squeeze.shift(1) & expanding & (df['close'] > df['bb_mid'])))
            else:
                return cache_and_return(safe_bool(squeeze.shift(1) & expanding & (df['close'] < df['bb_mid'])))

        elif strategy == 'ema_cross':
            if direction == 'long':
                return cache_and_return(safe_bool((df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1))))

        elif strategy == 'sma_cross':
            sma_fast = df['close'].rolling(9).mean()
            sma_slow = df['close'].rolling(18).mean()
            if direction == 'long':
                return cache_and_return(safe_bool((sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))))
            else:
                return cache_and_return(safe_bool((sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))))

        elif strategy == 'macd_cross':
            histogram = df['macd'] - df['macd_signal']
            if direction == 'long':
                return cache_and_return(safe_bool((histogram > 0) & (histogram.shift(1) <= 0)))
            else:
                return cache_and_return(safe_bool((histogram < 0) & (histogram.shift(1) >= 0)))

        elif strategy == 'supertrend':
            if direction == 'long':
                return cache_and_return(safe_bool((df['supertrend_dir'] == 1) & (df['supertrend_dir'].shift(1) == -1)))
            else:
                return cache_and_return(safe_bool((df['supertrend_dir'] == -1) & (df['supertrend_dir'].shift(1) == 1)))

        elif strategy == 'consecutive_candles':
            up_close = df['close'] > df['close'].shift(1)
            down_close = df['close'] < df['close'].shift(1)
            ups = up_close.astype(int).groupby((~up_close).cumsum()).cumsum()
            dns = down_close.astype(int).groupby((~down_close).cumsum()).cumsum()
            if direction == 'long':
                return cache_and_return(safe_bool(ups >= 3))
            else:
                return cache_and_return(safe_bool(dns >= 3))

        elif strategy == 'engulfing':
            if direction == 'long':
                return cache_and_return(safe_bool(df['green'] & df['red'].shift(1) &
                               (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))))
            else:
                return cache_and_return(safe_bool(df['red'] & df['green'].shift(1) &
                               (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))))

        elif strategy == 'inside_bar':
            inside = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
            if direction == 'long':
                return cache_and_return(safe_bool(inside & (df['close'] > df['open'])))
            else:
                return cache_and_return(safe_bool(inside & (df['close'] < df['open'])))

        elif strategy == 'outside_bar':
            outside = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
            if direction == 'long':
                return cache_and_return(safe_bool(outside & (df['close'] > df['open'])))
            else:
                return cache_and_return(safe_bool(outside & (df['close'] < df['open'])))

        elif strategy == 'atr_breakout':
            move = abs(df['close'] - df['close'].shift(1))
            if direction == 'long':
                return cache_and_return(safe_bool((move > df['atr'] * 1.5) & (df['close'] > df['close'].shift(1))))
            else:
                return cache_and_return(safe_bool((move > df['atr'] * 1.5) & (df['close'] < df['close'].shift(1))))

        # === MISSING STRATEGIES FROM STRATEGY_ENGINE ===

        elif strategy == 'price_vs_sma':
            # Mean Reversion: Price 1%+ away from SMA20
            sma = df['sma_20']
            deviation = (df['close'] - sma) / sma * 100
            if direction == 'long':
                # Long: price more than 1% below SMA (oversold)
                return cache_and_return(safe_bool(deviation < -1.0))
            else:
                # Short: price more than 1% above SMA (overbought)
                return cache_and_return(safe_bool(deviation > 1.0))

        elif strategy == 'price_above_sma':
            # Trend: Price crosses SMA20
            sma = df['sma_20']
            if direction == 'long':
                # Long: price crosses above SMA
                return cache_and_return(safe_bool((df['close'] > sma) & (df['close'].shift(1) <= sma.shift(1))))
            else:
                # Short: price crosses below SMA
                return cache_and_return(safe_bool((df['close'] < sma) & (df['close'].shift(1) >= sma.shift(1))))

        elif strategy == 'big_candle':
            # Pattern: Large candle 2x ATR in opposite direction
            candle_size = df['body']
            big_candle = candle_size > df['atr'] * 2
            if direction == 'long':
                # Long after big red candle (reversal)
                return cache_and_return(safe_bool(big_candle.shift(1) & df['red'].shift(1) & df['green']))
            else:
                # Short after big green candle (reversal)
                return cache_and_return(safe_bool(big_candle.shift(1) & df['green'].shift(1) & df['red']))

        elif strategy == 'doji_reversal':
            # Pattern: Doji candle after trend
            doji = df['doji']
            # Detect recent trend using 5-bar closes
            recent_up_trend = (df['close'].shift(2) > df['close'].shift(3)) & \
                              (df['close'].shift(3) > df['close'].shift(4)) & \
                              (df['close'].shift(4) > df['close'].shift(5))
            recent_down_trend = (df['close'].shift(2) < df['close'].shift(3)) & \
                                (df['close'].shift(3) < df['close'].shift(4)) & \
                                (df['close'].shift(4) < df['close'].shift(5))
            if direction == 'long':
                # Long: Doji after downtrend (potential bullish reversal)
                return cache_and_return(safe_bool(doji.shift(1) & recent_down_trend & df['green']))
            else:
                # Short: Doji after uptrend (potential bearish reversal)
                return cache_and_return(safe_bool(doji.shift(1) & recent_up_trend & df['red']))

        elif strategy == 'low_volatility_breakout':
            # Volatility: Breakout after low volatility period
            avg_range = df['range'].rolling(20).mean()
            current_range = df['range']
            low_vol = current_range.rolling(5).mean() < avg_range * 0.5  # Low volatility
            breakout = current_range > avg_range  # Current bar is a breakout
            if direction == 'long':
                return cache_and_return(safe_bool(low_vol.shift(1) & breakout & df['green']))
            else:
                return cache_and_return(safe_bool(low_vol.shift(1) & breakout & df['red']))

        elif strategy == 'higher_low':
            # Price Action: Higher low (long) or lower high (short)
            if direction == 'long':
                # Higher low: current low > previous swing low (simplified: last 5 bars)
                prev_low = df['low'].rolling(5).min().shift(1)
                curr_low = df['low']
                return cache_and_return(safe_bool((curr_low > prev_low) & (curr_low < curr_low.shift(1)) & df['green']))
            else:
                # Lower high: current high < previous swing high (simplified: last 5 bars)
                prev_high = df['high'].rolling(5).max().shift(1)
                curr_high = df['high']
                return cache_and_return(safe_bool((curr_high < prev_high) & (curr_high > curr_high.shift(1)) & df['red']))

        elif strategy == 'support_resistance':
            # Price Action: Price at recent support/resistance level
            lookback = 20
            support = df['low'].rolling(lookback).min()
            resistance = df['high'].rolling(lookback).max()
            # Define proximity as within 0.5% of level
            near_support = abs(df['close'] - support) / support < 0.005
            near_resistance = abs(df['close'] - resistance) / resistance < 0.005
            if direction == 'long':
                # Long: bounce from support
                return cache_and_return(safe_bool(near_support & df['green'] & (df['close'] > df['open'])))
            else:
                # Short: rejection from resistance
                return cache_and_return(safe_bool(near_resistance & df['red'] & (df['close'] < df['open'])))

        # === COMBO STRATEGIES ===

        elif strategy == 'bb_rsi_combo':
            # Bollinger Band touch + RSI extreme (Mean Reversion)
            if direction == 'long':
                bb_touch = df['close'] <= df['bb_lower']
                rsi_oversold = df['rsi'] < 35
                return cache_and_return(safe_bool(bb_touch & rsi_oversold))
            else:
                bb_touch = df['close'] >= df['bb_upper']
                rsi_overbought = df['rsi'] > 65
                return cache_and_return(safe_bool(bb_touch & rsi_overbought))

        elif strategy == 'supertrend_adx_combo':
            # Supertrend signal + ADX > 25 filter (Trend)
            direction_change = df['supertrend_dir'] - df['supertrend_dir'].shift(1)
            adx_strong = df['adx'] > 25
            if direction == 'long':
                return cache_and_return(safe_bool((direction_change > 0) & adx_strong))
            else:
                return cache_and_return(safe_bool((direction_change < 0) & adx_strong))

        elif strategy == 'ema_rsi_combo':
            # EMA cross + RSI confirmation (Trend)
            ema_cross_up = (df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))
            ema_cross_down = (df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1))
            if direction == 'long':
                return cache_and_return(safe_bool(ema_cross_up & (df['rsi'] > 50)))
            else:
                return cache_and_return(safe_bool(ema_cross_down & (df['rsi'] < 50)))

        elif strategy == 'macd_stoch_combo':
            # MACD cross + Stochastic confirmation (Momentum)
            macd_cross_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            if direction == 'long':
                return cache_and_return(safe_bool(macd_cross_up & (df['stoch_k'] < 50)))
            else:
                return cache_and_return(safe_bool(macd_cross_down & (df['stoch_k'] > 50)))

        # === NEW STRATEGIES ===

        elif strategy == 'ichimoku_cross':
            # Ichimoku Tenkan-Kijun cross
            if direction == 'long':
                # Long: Tenkan crosses above Kijun
                return cache_and_return(safe_bool((df['tenkan'] > df['kijun']) & (df['tenkan'].shift(1) <= df['kijun'].shift(1))))
            else:
                # Short: Tenkan crosses below Kijun
                return cache_and_return(safe_bool((df['tenkan'] < df['kijun']) & (df['tenkan'].shift(1) >= df['kijun'].shift(1))))

        elif strategy == 'ichimoku_cloud':
            # Ichimoku Cloud breakout
            cloud_top = df[['senkou_a', 'senkou_b']].max(axis=1)
            cloud_bottom = df[['senkou_a', 'senkou_b']].min(axis=1)
            if direction == 'long':
                # Long: price breaks above the cloud
                return cache_and_return(safe_bool((df['close'] > cloud_top) & (df['close'].shift(1) <= cloud_top.shift(1))))
            else:
                # Short: price breaks below the cloud
                return cache_and_return(safe_bool((df['close'] < cloud_bottom) & (df['close'].shift(1) >= cloud_bottom.shift(1))))

        elif strategy == 'aroon_cross':
            # Aroon oscillator cross
            if direction == 'long':
                # Long: Aroon Up crosses above Aroon Down
                return cache_and_return(safe_bool((df['aroon_up'] > df['aroon_down']) & (df['aroon_up'].shift(1) <= df['aroon_down'].shift(1))))
            else:
                # Short: Aroon Down crosses above Aroon Up
                return cache_and_return(safe_bool((df['aroon_down'] > df['aroon_up']) & (df['aroon_down'].shift(1) <= df['aroon_up'].shift(1))))

        elif strategy == 'momentum_zero':
            # Momentum crosses zero
            if direction == 'long':
                return cache_and_return(safe_bool((df['mom'] > 0) & (df['mom'].shift(1) <= 0)))
            else:
                return cache_and_return(safe_bool((df['mom'] < 0) & (df['mom'].shift(1) >= 0)))

        elif strategy == 'roc_extreme':
            # Rate of Change extreme values - ADAPTIVE to any pair/timeframe
            # Uses 5th/95th percentile to identify extremes relative to the data
            roc_lower = df['roc'].quantile(0.05)  # Bottom 5% = oversold
            roc_upper = df['roc'].quantile(0.95)  # Top 5% = overbought
            if direction == 'long':
                # Long: ROC in bottom 5th percentile (oversold)
                return cache_and_return(safe_bool(df['roc'] < roc_lower))
            else:
                # Short: ROC in top 95th percentile (overbought)
                return cache_and_return(safe_bool(df['roc'] > roc_upper))

        elif strategy == 'uo_extreme':
            # Ultimate Oscillator extreme values
            if direction == 'long':
                # Long: UO below 30 (oversold)
                return cache_and_return(safe_bool(df['uo'] < 30))
            else:
                # Short: UO above 70 (overbought)
                return cache_and_return(safe_bool(df['uo'] > 70))

        elif strategy == 'chop_trend':
            # Choppiness Index indicates trending market
            # Low choppiness (< 38.2) = trending, high (> 61.8) = ranging
            is_trending = df['chop'] < 38.2
            if direction == 'long':
                # Long: trending market with price above SMA
                return cache_and_return(safe_bool(is_trending & (df['close'] > df['sma_20'])))
            else:
                # Short: trending market with price below SMA
                return cache_and_return(safe_bool(is_trending & (df['close'] < df['sma_20'])))

        elif strategy == 'double_ema_cross':
            # EMA 12/26 cross (same as MACD periods)
            if direction == 'long':
                return cache_and_return(safe_bool((df['ema_12'] > df['ema_26']) & (df['ema_12'].shift(1) <= df['ema_26'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['ema_12'] < df['ema_26']) & (df['ema_12'].shift(1) >= df['ema_26'].shift(1))))

        # === TRIPLE EMA ALIGNMENT ===
        elif strategy == 'triple_ema':
            # Triple EMA alignment (9 > 21 > 50 for long, reverse for short)
            if direction == 'long':
                # All EMAs aligned bullishly and just crossed into alignment
                aligned = (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
                was_not_aligned = ~((df['ema_9'].shift(1) > df['ema_21'].shift(1)) & (df['ema_21'].shift(1) > df['ema_50'].shift(1)))
                return cache_and_return(safe_bool(aligned & was_not_aligned))
            else:
                # All EMAs aligned bearishly
                aligned = (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
                was_not_aligned = ~((df['ema_9'].shift(1) < df['ema_21'].shift(1)) & (df['ema_21'].shift(1) < df['ema_50'].shift(1)))
                return cache_and_return(safe_bool(aligned & was_not_aligned))

        # === McGINLEY DYNAMIC STRATEGIES ===
        elif strategy == 'mcginley_cross':
            # Price crosses McGinley Dynamic
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['mcginley']) & (df['close'].shift(1) <= df['mcginley'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['mcginley']) & (df['close'].shift(1) >= df['mcginley'].shift(1))))

        elif strategy == 'mcginley_trend':
            # McGinley changes direction (slope)
            mcg_slope = df['mcginley'] - df['mcginley'].shift(1)
            mcg_slope_prev = df['mcginley'].shift(1) - df['mcginley'].shift(2)
            if direction == 'long':
                # Slope turns positive
                return cache_and_return(safe_bool((mcg_slope > 0) & (mcg_slope_prev <= 0)))
            else:
                # Slope turns negative
                return cache_and_return(safe_bool((mcg_slope < 0) & (mcg_slope_prev >= 0)))

        # === HULL MOVING AVERAGE ===
        elif strategy == 'hull_ma_cross':
            # Price crosses Hull MA
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['hull_ma']) & (df['close'].shift(1) <= df['hull_ma'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['hull_ma']) & (df['close'].shift(1) >= df['hull_ma'].shift(1))))

        elif strategy == 'hull_ma_turn':
            # Hull MA changes direction
            hull_slope = df['hull_ma'] - df['hull_ma'].shift(1)
            hull_slope_prev = df['hull_ma'].shift(1) - df['hull_ma'].shift(2)
            if direction == 'long':
                return cache_and_return(safe_bool((hull_slope > 0) & (hull_slope_prev <= 0)))
            else:
                return cache_and_return(safe_bool((hull_slope < 0) & (hull_slope_prev >= 0)))

        # === ZLEMA (Zero-Lag EMA) ===
        elif strategy == 'zlema_cross':
            # Price crosses Zero-Lag EMA
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['zlema']) & (df['close'].shift(1) <= df['zlema'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['zlema']) & (df['close'].shift(1) >= df['zlema'].shift(1))))

        # === CHANDELIER EXIT ===
        elif strategy == 'chandelier_entry':
            # Chandelier Exit signal
            if direction == 'long':
                # Price crosses above chandelier long stop
                return cache_and_return(safe_bool((df['close'] > df['chandelier_long']) & (df['close'].shift(1) <= df['chandelier_long'].shift(1))))
            else:
                # Price crosses below chandelier short stop
                return cache_and_return(safe_bool((df['close'] < df['chandelier_short']) & (df['close'].shift(1) >= df['chandelier_short'].shift(1))))

        # === TSI (True Strength Index) ===
        elif strategy == 'tsi_cross':
            # TSI crosses signal line
            if direction == 'long':
                return cache_and_return(safe_bool((df['tsi'] > df['tsi_signal']) & (df['tsi'].shift(1) <= df['tsi_signal'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['tsi'] < df['tsi_signal']) & (df['tsi'].shift(1) >= df['tsi_signal'].shift(1))))

        elif strategy == 'tsi_zero':
            # TSI crosses zero line
            if direction == 'long':
                return cache_and_return(safe_bool((df['tsi'] > 0) & (df['tsi'].shift(1) <= 0)))
            else:
                return cache_and_return(safe_bool((df['tsi'] < 0) & (df['tsi'].shift(1) >= 0)))

        # === CMF (Chaikin Money Flow) ===
        elif strategy == 'cmf_cross':
            # CMF crosses zero
            if direction == 'long':
                return cache_and_return(safe_bool((df['cmf'] > 0) & (df['cmf'].shift(1) <= 0)))
            else:
                return cache_and_return(safe_bool((df['cmf'] < 0) & (df['cmf'].shift(1) >= 0)))

        # === OBV (On Balance Volume) ===
        elif strategy == 'obv_trend':
            # OBV makes new high/low with price
            lookback = 14
            obv_high = df['obv'].rolling(lookback).max()
            obv_low = df['obv'].rolling(lookback).min()
            price_high = df['close'].rolling(lookback).max()
            price_low = df['close'].rolling(lookback).min()
            if direction == 'long':
                return cache_and_return(safe_bool((df['obv'] == obv_high) & (df['close'] >= price_high * 0.98)))
            else:
                return cache_and_return(safe_bool((df['obv'] == obv_low) & (df['close'] <= price_low * 1.02)))

        # === MFI (Money Flow Index) ===
        elif strategy == 'mfi_extreme':
            if direction == 'long':
                return cache_and_return(safe_bool((df['mfi'] > 20) & (df['mfi'].shift(1) <= 20)))
            else:
                return cache_and_return(safe_bool((df['mfi'] < 80) & (df['mfi'].shift(1) >= 80)))

        # === PPO (Percentage Price Oscillator) ===
        elif strategy == 'ppo_cross':
            if direction == 'long':
                return cache_and_return(safe_bool((df['ppo'] > df['ppo_signal']) & (df['ppo'].shift(1) <= df['ppo_signal'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['ppo'] < df['ppo_signal']) & (df['ppo'].shift(1) >= df['ppo_signal'].shift(1))))

        # === Fisher Transform ===
        elif strategy == 'fisher_cross':
            if direction == 'long':
                return cache_and_return(safe_bool((df['fisher'] > df['fisher_signal']) & (df['fisher'].shift(1) <= df['fisher_signal'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['fisher'] < df['fisher_signal']) & (df['fisher'].shift(1) >= df['fisher_signal'].shift(1))))

        # === Squeeze Momentum ===
        elif strategy == 'squeeze_momentum':
            # BB inside Keltner + momentum direction
            squeeze = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
            squeeze_fired = squeeze.shift(1) & ~squeeze  # Squeeze released
            mom = df['close'] - df['close'].shift(20)  # Simple momentum
            if direction == 'long':
                return cache_and_return(safe_bool(squeeze_fired & (mom > 0)))
            else:
                return cache_and_return(safe_bool(squeeze_fired & (mom < 0)))

        # === VWAP Cross ===
        elif strategy == 'vwap_cross':
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['vwap']) & (df['close'].shift(1) <= df['vwap'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['vwap']) & (df['close'].shift(1) >= df['vwap'].shift(1))))

        # === VWMA CROSS ===
        elif strategy == 'vwma_cross':
            # Price crosses VWMA (Trend Following)
            vwma = df['vwma']
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > vwma) & (df['close'].shift(1) <= vwma.shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < vwma) & (df['close'].shift(1) >= vwma.shift(1))))

        # === VWMA TREND ===
        elif strategy == 'vwma_trend':
            # VWMA direction change (Trend Following)
            vwma = df['vwma']
            vwma_slope = vwma - vwma.shift(1)
            if direction == 'long':
                # Long when VWMA starts sloping up
                return cache_and_return(safe_bool((vwma_slope > 0) & (vwma_slope.shift(1) <= 0)))
            else:
                # Short when VWMA starts sloping down
                return cache_and_return(safe_bool((vwma_slope < 0) & (vwma_slope.shift(1) >= 0)))

        # === PIVOT BOUNCE ===
        elif strategy == 'pivot_bounce':
            # Price bounces off pivot point levels (Price Action)
            r1 = df['pivot_r1']
            s1 = df['pivot_s1']
            if direction == 'long':
                # Price bounces off S1
                near_s1 = (df['low'] <= s1 * 1.005) & (df['low'] >= s1 * 0.995)
                return cache_and_return(safe_bool(near_s1 & (df['close'] > df['open'])))
            else:
                # Price bounces off R1
                near_r1 = (df['high'] >= r1 * 0.995) & (df['high'] <= r1 * 1.005)
                return cache_and_return(safe_bool(near_r1 & (df['close'] < df['open'])))

        # === LINEAR REGRESSION CHANNEL ===
        elif strategy == 'linreg_channel':
            # Price touches/breaks linear regression channel (Trend)
            upper = df['linreg_upper']
            lower = df['linreg_lower']
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > lower) & (df['close'].shift(1) <= lower.shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < upper) & (df['close'].shift(1) >= upper.shift(1))))

        # === AWESOME OSCILLATOR ZERO CROSS ===
        elif strategy == 'ao_zero_cross':
            # Awesome Oscillator crosses zero (Momentum)
            ao = df['ao']
            if direction == 'long':
                return cache_and_return(safe_bool((ao > 0) & (ao.shift(1) <= 0)))
            else:
                return cache_and_return(safe_bool((ao < 0) & (ao.shift(1) >= 0)))

        # === AWESOME OSCILLATOR TWIN PEAKS ===
        elif strategy == 'ao_twin_peaks':
            # Awesome Oscillator twin peaks pattern (Momentum)
            ao = df['ao']
            lookback = 20
            if direction == 'long':
                # Twin peaks below zero: AO < 0, AO > ao_low, AO rising
                ao_low = ao.rolling(lookback).min()
                ao_rising = ao > ao.shift(1)
                return cache_and_return(safe_bool((ao < 0) & (ao > ao_low) & ao_rising))
            else:
                # Twin peaks above zero: AO > 0, AO < ao_high, AO falling
                ao_high = ao.rolling(lookback).max()
                ao_falling = ao < ao.shift(1)
                return cache_and_return(safe_bool((ao > 0) & (ao < ao_high) & ao_falling))

        # === ELDER RAY ===
        elif strategy == 'elder_ray':
            # Bull/Bear power with EMA trend filter (Momentum)
            ema_13 = df['ema_13']
            bull_power = df['bull_power']
            bear_power = df['bear_power']
            if direction == 'long':
                # EMA rising, bear power negative but rising
                ema_rising = ema_13 > ema_13.shift(1)
                bear_rising = bear_power > bear_power.shift(1)
                return cache_and_return(safe_bool(ema_rising & (bear_power < 0) & bear_rising))
            else:
                # EMA falling, bull power positive but falling
                ema_falling = ema_13 < ema_13.shift(1)
                bull_falling = bull_power < bull_power.shift(1)
                return cache_and_return(safe_bool(ema_falling & (bull_power > 0) & bull_falling))

        # === RSI + MACD COMBO ===
        elif strategy == 'rsi_macd_combo':
            # RSI extreme + MACD confirmation (Momentum)
            histogram = df['macd'] - df['macd_signal']
            if direction == 'long':
                rsi_oversold = df['rsi'] < 30
                macd_bullish = histogram > histogram.shift(1)
                return cache_and_return(safe_bool(rsi_oversold & macd_bullish))
            else:
                rsi_overbought = df['rsi'] > 70
                macd_bearish = histogram < histogram.shift(1)
                return cache_and_return(safe_bool(rsi_overbought & macd_bearish))

        # === WILLIAMS %R EXTREME ===
        elif strategy == 'williams_r':
            if direction == 'long':
                return cache_and_return(safe_bool(df['willr'] < -80))
            else:
                return cache_and_return(safe_bool(df['willr'] > -20))

        # === CCI EXTREME ===
        elif strategy == 'cci_extreme':
            if direction == 'long':
                return cache_and_return(safe_bool(df['cci'] < -100))
            else:
                return cache_and_return(safe_bool(df['cci'] > 100))

        # === ADX STRONG TREND ===
        elif strategy == 'adx_strong_trend':
            strong_trend = df['adx'] > 25
            if direction == 'long':
                return cache_and_return(safe_bool(strong_trend & (df['di_plus'] > df['di_minus'])))
            else:
                return cache_and_return(safe_bool(strong_trend & (df['di_minus'] > df['di_plus'])))

        # === PARABOLIC SAR REVERSAL ===
        elif strategy == 'psar_reversal':
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['psar']) & (df['close'].shift(1) <= df['psar'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['psar']) & (df['close'].shift(1) >= df['psar'].shift(1))))

        # === VWAP BOUNCE ===
        elif strategy == 'vwap_bounce':
            if 'vwap' not in df.columns or df['vwap'].isna().all():
                return cache_and_return(pd.Series(False, index=df.index))
            vwap = df['vwap'].ffill().fillna(df['close'])
            if direction == 'long':
                touched_below = df['low'] < vwap
                closed_above = df['close'] > vwap
                return cache_and_return(safe_bool(touched_below & closed_above))
            else:
                touched_above = df['high'] > vwap
                closed_below = df['close'] < vwap
                return cache_and_return(safe_bool(touched_above & closed_below))

        # === RSI DIVERGENCE ===
        elif strategy == 'rsi_divergence':
            lookback = 5
            if direction == 'long':
                price_lower_low = df['low'] < df['low'].rolling(lookback).min().shift(1)
                rsi_higher_low = df['rsi'] > df['rsi'].rolling(lookback).min().shift(1)
                return cache_and_return(safe_bool(price_lower_low & rsi_higher_low & (df['rsi'] < 40)))
            else:
                price_higher_high = df['high'] > df['high'].rolling(lookback).max().shift(1)
                rsi_lower_high = df['rsi'] < df['rsi'].rolling(lookback).max().shift(1)
                return cache_and_return(safe_bool(price_higher_high & rsi_lower_high & (df['rsi'] > 60)))

        # === KELTNER CHANNEL BREAKOUT ===
        elif strategy == 'keltner_breakout':
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['kc_upper']) & (df['close'].shift(1) <= df['kc_upper'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['kc_lower']) & (df['close'].shift(1) >= df['kc_lower'].shift(1))))

        # === DONCHIAN CHANNEL BREAKOUT (TURTLE TRADING) ===
        elif strategy == 'donchian_breakout':
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['dc_upper'].shift(1)) & (df['close'].shift(1) <= df['dc_upper'].shift(2))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['dc_lower'].shift(1)) & (df['close'].shift(1) >= df['dc_lower'].shift(2))))

        # === KALMAN FILTER STRATEGIES ===
        elif strategy == 'kalman_trend':
            # Price crosses Kalman filter line
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['kalman']) & (df['close'].shift(1) <= df['kalman'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['kalman']) & (df['close'].shift(1) >= df['kalman'].shift(1))))

        elif strategy == 'kalman_bb':
            # Price touches Kalman-based bands
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['kalman_lower']) & (df['close'].shift(1) <= df['kalman_lower'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['kalman_upper']) & (df['close'].shift(1) >= df['kalman_upper'].shift(1))))

        elif strategy == 'kalman_rsi':
            # Kalman-smoothed RSI crosses configurable thresholds (default 30/70)
            lower = df['kalman_rsi_lower'].iloc[0] if 'kalman_rsi_lower' in df.columns else 30
            upper = df['kalman_rsi_upper'].iloc[0] if 'kalman_rsi_upper' in df.columns else 70
            if direction == 'long':
                return cache_and_return(safe_bool((df['kalman_rsi'] > lower) & (df['kalman_rsi'].shift(1) <= lower)))
            else:
                return cache_and_return(safe_bool((df['kalman_rsi'] < upper) & (df['kalman_rsi'].shift(1) >= upper)))

        elif strategy == 'kalman_mfi':
            # Kalman-smoothed MFI crosses configurable thresholds (default 20/80)
            lower = df['kalman_mfi_lower'].iloc[0] if 'kalman_mfi_lower' in df.columns else 20
            upper = df['kalman_mfi_upper'].iloc[0] if 'kalman_mfi_upper' in df.columns else 80
            if direction == 'long':
                return cache_and_return(safe_bool((df['kalman_mfi'] > lower) & (df['kalman_mfi'].shift(1) <= lower)))
            else:
                return cache_and_return(safe_bool((df['kalman_mfi'] < upper) & (df['kalman_mfi'].shift(1) >= upper)))

        elif strategy == 'kalman_adx':
            # Kalman ADX > configurable threshold (default 25) with DI dominance
            threshold = df['kalman_adx_threshold'].iloc[0] if 'kalman_adx_threshold' in df.columns else 25
            if direction == 'long':
                return cache_and_return(safe_bool((df['kalman_adx'] > threshold) & (df['plus_di'] > df['minus_di'])))
            else:
                return cache_and_return(safe_bool((df['kalman_adx'] > threshold) & (df['minus_di'] > df['plus_di'])))

        elif strategy == 'kalman_psar':
            # Price crosses Kalman-smoothed Parabolic SAR
            if direction == 'long':
                return cache_and_return(safe_bool((df['close'] > df['kalman_psar']) & (df['close'].shift(1) <= df['kalman_psar'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['close'] < df['kalman_psar']) & (df['close'].shift(1) >= df['kalman_psar'].shift(1))))

        elif strategy == 'kalman_macd':
            # Kalman-smoothed MACD signal cross
            if direction == 'long':
                return cache_and_return(safe_bool((df['kalman_macd'] > df['kalman_macd_signal']) & (df['kalman_macd'].shift(1) <= df['kalman_macd_signal'].shift(1))))
            else:
                return cache_and_return(safe_bool((df['kalman_macd'] < df['kalman_macd_signal']) & (df['kalman_macd'].shift(1) >= df['kalman_macd_signal'].shift(1))))

        # Default: no signals (also cached)
        return cache_and_return(pd.Series(False, index=df.index))

    def run_single_backtest(
        self,
        strategy: str,
        direction: str,
        tp_percent: float,
        sl_percent: float,
    ) -> VectorBTResult:
        """
        Run a single backtest using VectorBT.

        Args:
            strategy: Strategy name
            direction: 'long' or 'short'
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage

        Returns:
            VectorBTResult with metrics
        """
        # Get entry signals
        entries = self._get_signals(strategy, direction)

        # Skip if no signals
        if not entries.any():
            return self._empty_result(strategy, direction, tp_percent, sl_percent)

        # Determine if short
        short = direction == 'short'

        # Run VectorBT portfolio simulation
        try:
            pf = vbt.Portfolio.from_signals(
                close=self.df['close'],
                entries=entries if not short else pd.Series(False, index=self.df.index),
                short_entries=entries if short else pd.Series(False, index=self.df.index),
                sl_stop=sl_percent / 100,
                tp_stop=tp_percent / 100,
                size=self.position_size_pct / 100,
                size_type='percent',
                fees=self.total_fees,
                init_cash=self.initial_capital,
                freq=self.data_freq,
            )

            # Extract metrics (include equity curve and trades list for single backtests)
            return self._extract_metrics(
                pf, strategy, direction, tp_percent, sl_percent,
                include_equity_curve=True, entries=entries
            )

        except Exception as e:
            log(f"[VectorBT] Backtest error for {strategy}/{direction}: {e}", level='WARNING')
            return self._empty_result(strategy, direction, tp_percent, sl_percent)

    def run_bidirectional_backtest(
        self,
        strategy: str,
        tp_percent: float,
        sl_percent: float,
    ) -> VectorBTResult:
        """
        Run a bidirectional backtest using VectorBT.

        Uses flip-style position handling: opposite signal closes current
        position and opens new one. Conflicting signals (both fire on same bar)
        are skipped to match strategy_engine.py behavior.

        Args:
            strategy: Strategy name
            tp_percent: Take profit percentage (same for both directions)
            sl_percent: Stop loss percentage (same for both directions)

        Returns:
            VectorBTResult with direction='both' and per-direction metrics
        """
        # Get BOTH long and short signals upfront
        long_entries = self._get_signals(strategy, 'long')
        short_entries = self._get_signals(strategy, 'short')

        # Skip if no signals in either direction
        if not long_entries.any() and not short_entries.any():
            return self._empty_result(strategy, 'both', tp_percent, sl_percent)

        # Pre-process signals: set both to False where they conflict
        # This matches strategy_engine.py backtest_bidirectional() behavior
        conflict_mask = long_entries & short_entries
        long_entries_clean = long_entries & ~conflict_mask
        short_entries_clean = short_entries & ~conflict_mask

        try:
            # Run VectorBT with bidirectional support
            # upon_opposite_entry='Reverse' = flip positions on opposite signal
            # NOTE: Must use size_type='value' (not 'percent') because VectorBT
            # doesn't support position reversal with percentage-based sizing
            position_value = self.initial_capital * (self.position_size_pct / 100)
            pf = vbt.Portfolio.from_signals(
                close=self.df['close'],
                entries=long_entries_clean,
                short_entries=short_entries_clean,
                sl_stop=sl_percent / 100,
                tp_stop=tp_percent / 100,
                size=position_value,
                size_type='value',
                fees=self.total_fees,
                init_cash=self.initial_capital,
                freq=self.data_freq,
                upon_opposite_entry='Reverse',  # Flip on opposite signal
            )

            return self._extract_bidirectional_metrics(
                pf, strategy, tp_percent, sl_percent,
                long_entries_clean, short_entries_clean,
                include_equity_curve=True
            )

        except Exception as e:
            log(f"[VectorBT] Bidirectional backtest error for {strategy}: {e}", level='WARNING')
            return self._empty_result(strategy, 'both', tp_percent, sl_percent)

    def _extract_bidirectional_metrics(
        self,
        pf,
        strategy: str,
        tp_percent: float,
        sl_percent: float,
        long_entries: pd.Series,
        short_entries: pd.Series,
        include_equity_curve: bool = False,
    ) -> VectorBTResult:
        """
        Extract metrics from bidirectional VectorBT portfolio.

        Includes per-direction metrics (long_trades, short_trades, etc.) and flip count.
        """
        try:
            trades = pf.trades
            total_trades = trades.count()

            if total_trades == 0:
                return self._empty_result(strategy, 'both', tp_percent, sl_percent)

            # Get trade records with direction info
            records = trades.records_readable

            # Per-direction metrics using VectorBT's Direction column
            # VectorBT uses 0 for Long, 1 for Short in the direction column
            if 'Direction' in records.columns:
                long_mask = records['Direction'] == 'Long'
                short_mask = records['Direction'] == 'Short'
            else:
                # Fallback if Direction column doesn't exist
                long_mask = pd.Series([False] * len(records))
                short_mask = pd.Series([False] * len(records))

            long_trades_count = int(long_mask.sum())
            short_trades_count = int(short_mask.sum())

            # Per-direction wins and PnL
            pnl_values = trades.pnl.values if hasattr(trades, 'pnl') else np.array([])

            if len(pnl_values) > 0 and len(records) > 0:
                long_pnl_values = pnl_values[long_mask.values] if long_trades_count > 0 else np.array([])
                short_pnl_values = pnl_values[short_mask.values] if short_trades_count > 0 else np.array([])

                long_wins = int((long_pnl_values > 0).sum()) if len(long_pnl_values) > 0 else 0
                short_wins = int((short_pnl_values > 0).sum()) if len(short_pnl_values) > 0 else 0

                long_pnl = float(long_pnl_values.sum()) if len(long_pnl_values) > 0 else 0.0
                short_pnl = float(short_pnl_values.sum()) if len(short_pnl_values) > 0 else 0.0
            else:
                long_wins = short_wins = 0
                long_pnl = short_pnl = 0.0

            # Flip count: count trades where direction changed from previous
            flip_count = self._count_flips(records)

            # Standard combined metrics
            wins = long_wins + short_wins
            losses = int(total_trades) - wins
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            total_return = float(pf.total_return()) * 100
            total_pnl = float(pf.final_value() - self.initial_capital)

            # Profit factor
            if len(pnl_values) > 0:
                gross_profits = float(pnl_values[pnl_values > 0].sum())
                gross_losses = abs(float(pnl_values[pnl_values < 0].sum()))
            else:
                gross_profits = gross_losses = 0
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else (999.0 if gross_profits > 0 else 0)

            # Drawdown
            max_dd_pct = float(pf.max_drawdown()) * 100
            max_dd = float(pf.max_drawdown() * self.initial_capital)

            # Average trade
            avg_trade = total_pnl / total_trades if total_trades > 0 else 0
            avg_trade_pct = total_return / total_trades if total_trades > 0 else 0

            # Buy & hold comparison
            buy_hold = float((self.df['close'].iloc[-1] / self.df['close'].iloc[0] - 1) * 100)
            vs_buy_hold = total_return - buy_hold

            # Equity curve
            equity = pf.value().values.tolist() if include_equity_curve else []

            # Trades list for bidirectional
            trades_list = []
            if include_equity_curve:
                trades_list = self._build_bidirectional_trades_list(
                    pf, tp_percent, sl_percent, long_entries, short_entries
                )

            # Calculate composite score
            composite = self._calculate_composite_score(
                win_rate, profit_factor, total_return, max_dd_pct, int(total_trades)
            )

            # Data range for period display
            data_start_str = str(self.df.index[0]) if len(self.df) > 0 else None
            data_end_str = str(self.df.index[-1]) if len(self.df) > 0 else None

            return VectorBTResult(
                strategy_name=f"{strategy}_both",
                strategy_category=self.ENTRY_STRATEGIES.get(strategy, {}).get('category', 'Unknown'),
                direction='both',
                tp_percent=tp_percent,
                sl_percent=sl_percent,
                entry_rule=strategy,
                total_trades=int(total_trades),
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_pnl_percent=total_return,
                profit_factor=min(profit_factor, 999.0),
                max_drawdown=max_dd,
                max_drawdown_percent=max_dd_pct,
                avg_trade=avg_trade,
                avg_trade_percent=avg_trade_pct,
                buy_hold_return=buy_hold,
                vs_buy_hold=vs_buy_hold,
                beats_buy_hold=vs_buy_hold > 0,
                composite_score=composite,
                equity_curve=equity,
                trades_list=trades_list,
                params={'tp_percent': tp_percent, 'sl_percent': sl_percent, 'direction': 'both'},
                data_start=data_start_str,
                data_end=data_end_str,
                # Bidirectional-specific fields
                long_trades=long_trades_count,
                long_wins=long_wins,
                long_pnl=long_pnl,
                short_trades=short_trades_count,
                short_wins=short_wins,
                short_pnl=short_pnl,
                flip_count=flip_count,
            )

        except Exception as e:
            log(f"[VectorBT] Error extracting bidirectional metrics: {e}", level='WARNING')
            return self._empty_result(strategy, 'both', tp_percent, sl_percent)

    def _count_flips(self, records: pd.DataFrame) -> int:
        """Count position flips (consecutive trades with opposite directions)."""
        if len(records) < 2:
            return 0

        flip_count = 0
        prev_direction = None

        for idx, trade in records.iterrows():
            curr_direction = trade.get('Direction', None)
            if prev_direction is not None and curr_direction != prev_direction:
                flip_count += 1
            prev_direction = curr_direction

        return flip_count

    def _build_bidirectional_trades_list(
        self,
        pf,
        tp_percent: float,
        sl_percent: float,
        long_entries: pd.Series,
        short_entries: pd.Series,
    ) -> List[Dict]:
        """Build trades list for bidirectional strategies with per-trade direction."""
        trades_list = []

        try:
            trades = pf.trades
            if trades.count() == 0:
                return trades_list

            records = trades.records_readable

            for idx, trade in records.iterrows():
                try:
                    # Get direction from VectorBT trade record
                    vbt_direction = trade.get('Direction', 'Long')
                    direction = 'long' if vbt_direction == 'Long' else 'short'

                    entry_bar_time = trade.get('Entry Timestamp', None)
                    exit_bar_time = trade.get('Exit Timestamp', None)

                    if hasattr(entry_bar_time, 'strftime'):
                        entry_bar_time = str(entry_bar_time)
                    if hasattr(exit_bar_time, 'strftime'):
                        exit_bar_time = str(exit_bar_time)

                    entry_price = float(trade.get('Avg Entry Price', 0) or 0)
                    exit_price = float(trade.get('Avg Exit Price', 0) or 0)
                    pnl = float(trade.get('PnL', 0) or 0)
                    pnl_pct = float(trade.get('Return', 0) or 0) * 100

                    # Calculate TP/SL prices based on direction
                    if direction == 'long':
                        tp_price = entry_price * (1 + tp_percent / 100)
                        sl_price = entry_price * (1 - sl_percent / 100)
                    else:
                        tp_price = entry_price * (1 - tp_percent / 100)
                        sl_price = entry_price * (1 + sl_percent / 100)

                    # Determine exit type
                    exit_idx = 0
                    if exit_bar_time:
                        try:
                            exit_ts = pd.Timestamp(exit_bar_time)
                            if exit_ts in self.df.index:
                                exit_idx = self.df.index.get_loc(exit_ts)
                        except:
                            pass

                    exit_type = self._determine_exit_type(
                        exit_price, entry_price, tp_price, sl_price, direction, exit_idx, len(self.df)
                    )

                    trade_record = {
                        'trade_num': idx + 1,
                        'direction': direction,
                        'entry_time': str(entry_bar_time) if entry_bar_time else None,
                        'exit_time': str(exit_bar_time) if exit_bar_time else None,
                        'entry': round(entry_price, 2),
                        'exit': round(exit_price, 2),
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'result': 'WIN' if pnl > 0 else 'LOSS',
                        'tp_price': round(tp_price, 2),
                        'sl_price': round(sl_price, 2),
                        'exit_type': exit_type,
                    }

                    trades_list.append(trade_record)

                except Exception as e:
                    log(f"[VectorBT] Error processing bidirectional trade {idx}: {e}", level='DEBUG')
                    continue

        except Exception as e:
            log(f"[VectorBT] Error building bidirectional trades list: {e}", level='WARNING')

        return trades_list

    # Maximum results to keep to prevent OOM on large optimizations
    MAX_RESULTS = 10000

    def run_optimization(
        self,
        strategies: List[str] = None,
        directions: List[str] = None,
        tp_range: np.ndarray = None,
        sl_range: np.ndarray = None,
        mode: str = 'all',
        progress_callback: Callable = None,
        max_results: int = None,
        checkpoint_callback: Callable = None,
    ) -> List[VectorBTResult]:
        """
        Run optimization across all parameter combinations using VectorBT broadcasting.

        This is the main entry point for strategy optimization.

        Args:
            strategies: List of strategy names (default: all)
            directions: List of directions (default: ['long', 'short'])
            tp_range: Array of TP percentages
            sl_range: Array of SL percentages
            mode: 'separate', 'bidirectional', or 'all'
            progress_callback: Function to call with progress updates
            max_results: Maximum results to return (default: MAX_RESULTS=10000)
            checkpoint_callback: Function to call with checkpoint data for crash recovery

        Returns:
            List of VectorBTResult sorted by composite score (limited to max_results)
        """
        if strategies is None:
            strategies = list(self.ENTRY_STRATEGIES.keys())

        if directions is None:
            directions = ['long', 'short']

        if tp_range is None:
            tp_range = np.arange(0.5, 5.1, 0.5)

        if sl_range is None:
            sl_range = np.arange(0.5, 5.1, 0.5)

        # Clear signal cache at start of optimization to prevent memory leak
        self._signal_cache.clear()

        results = []

        # Calculate total combinations based on mode
        separate_combos = len(strategies) * len(directions) * len(tp_range) * len(sl_range) if mode in ['separate', 'all'] else 0
        bidir_combos = len(strategies) * len(tp_range) * len(sl_range) if mode in ['bidirectional', 'all'] else 0
        total_combos = separate_combos + bidir_combos
        completed = 0

        import time
        start_time = time.time()

        start_memory = get_memory_mb()
        vbt_log(f"[VectorBT] Starting optimization - Memory: {start_memory:.0f} MB", level='DEBUG')

        # Extract data range for period display in Strategy History
        data_start_str = str(self.df.index[0]) if len(self.df) > 0 else None
        data_end_str = str(self.df.index[-1]) if len(self.df) > 0 else None
        vbt_log(f"[VectorBT] Data range: {data_start_str} to {data_end_str}", level='DEBUG')

        mode_desc = "bidirectional only" if mode == "bidirectional" else ("separate + bidirectional" if mode == "all" else "separate (long/short)")
        vbt_log(f"[VectorBT] Starting VECTORIZED optimization (mode: {mode_desc})", level='DEBUG')
        vbt_log(f"[VectorBT] Parameters: {len(strategies)} strategies x {len(directions)} directions x {len(tp_range)} TPs x {len(sl_range)} SLs", level='DEBUG')
        vbt_log(f"[VectorBT] Total combinations: {total_combos:,} (separate: {separate_combos:,}, bidirectional: {bidir_combos:,})", level='DEBUG')

        # ========== PHASE 1: SEPARATE LONG/SHORT STRATEGIES ==========
        if mode in ['separate', 'all']:
            # Use VectorBT broadcasting for massive speedup
            for strategy in strategies:
                for direction in directions:
                    # Get signals once per strategy/direction
                    entries = self._get_signals(strategy, direction)

                    if not entries.any():
                        completed += len(tp_range) * len(sl_range)
                        continue

                    short = direction == 'short'

                    # VectorBT broadcasting: tile close and entries to create multi-column DataFrames
                    # Each column represents a different TP/SL combination
                    n_tp = len(tp_range)
                    n_sl = len(sl_range)
                    n_combos = n_tp * n_sl

                    try:
                        # Create column labels for TP/SL combinations
                        # Format: MultiIndex with (tp_value, sl_value)
                        col_tuples = [(tp, sl) for tp in tp_range for sl in sl_range]
                        columns = pd.MultiIndex.from_tuples(col_tuples, names=['tp', 'sl'])

                        # Tile close prices to have n_combos columns
                        close_tiled = pd.DataFrame(
                            np.tile(self.df['close'].values.reshape(-1, 1), (1, n_combos)),
                            index=self.df.index,
                            columns=columns
                        )

                        # Tile entries to match
                        entries_arr = entries.values.reshape(-1, 1)
                        entries_tiled = pd.DataFrame(
                            np.tile(entries_arr, (1, n_combos)),
                            index=self.df.index,
                            columns=columns
                        )

                        # Create TP/SL arrays matching the column structure
                        # Each column gets the same TP/SL value for all rows
                        tp_arr = np.array([tp / 100 for tp, sl in col_tuples])  # Shape: (n_combos,)
                        sl_arr = np.array([sl / 100 for tp, sl in col_tuples])  # Shape: (n_combos,)

                        # Run vectorized portfolio simulation across all combinations
                        pf = vbt.Portfolio.from_signals(
                            close=close_tiled,
                            entries=entries_tiled if not short else pd.DataFrame(False, index=self.df.index, columns=columns),
                            short_entries=entries_tiled if short else pd.DataFrame(False, index=self.df.index, columns=columns),
                            sl_stop=sl_arr,  # Per-column stop loss
                            tp_stop=tp_arr,  # Per-column take profit
                            size=self.position_size_pct / 100,
                            size_type='percent',
                            fees=self.total_fees,
                            init_cash=self.initial_capital,
                            freq=self.data_freq,
                        )

                        # VECTORIZED METRIC EXTRACTION - get ALL metrics at once (massive speedup)
                        # Instead of 52,800 individual calls, we make ~10 vectorized calls
                        try:
                            all_returns = pf.total_return() * 100  # Series indexed by (tp, sl)
                            all_final_values = pf.final_value()
                            all_max_dd = pf.max_drawdown() * 100

                            # Trade stats - more complex but still vectorized
                            trades = pf.trades
                            trade_counts = trades.count()

                            # Get win/loss counts per column
                            if hasattr(trades, 'pnl') and len(trades.pnl) > 0:
                                # Group PnL by column and count wins/losses
                                trade_pnl = trades.pnl.values
                                trade_cols = trades.col.values

                                # Pre-compute wins/losses/profits per column
                                n_cols = len(col_tuples)
                                wins_arr = np.zeros(n_cols)
                                gross_profit_arr = np.zeros(n_cols)
                                gross_loss_arr = np.zeros(n_cols)

                                for i in range(len(trade_pnl)):
                                    col_idx = trade_cols[i]
                                    pnl = trade_pnl[i]
                                    if pnl > 0:
                                        wins_arr[col_idx] += 1
                                        gross_profit_arr[col_idx] += pnl
                                    else:
                                        gross_loss_arr[col_idx] += abs(pnl)
                            else:
                                wins_arr = np.zeros(len(col_tuples))
                                gross_profit_arr = np.zeros(len(col_tuples))
                                gross_loss_arr = np.zeros(len(col_tuples))

                            # Buy & hold (same for all combinations)
                            buy_hold = float((self.df['close'].iloc[-1] / self.df['close'].iloc[0] - 1) * 100)

                            # Build results from vectorized data
                            for idx, (tp, sl) in enumerate(col_tuples):
                                total_trades = int(trade_counts.iloc[idx]) if idx < len(trade_counts) else 0

                                if total_trades > 0:
                                    total_return = float(all_returns.iloc[idx])
                                    total_pnl = float(all_final_values.iloc[idx]) - self.initial_capital
                                    max_dd_pct = float(all_max_dd.iloc[idx])
                                    wins = int(wins_arr[idx])
                                    losses = total_trades - wins
                                    win_rate = (wins / total_trades * 100)
                                    gross_profit = gross_profit_arr[idx]
                                    gross_loss = gross_loss_arr[idx]
                                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0)

                                    avg_trade = total_pnl / total_trades
                                    avg_trade_pct = total_return / total_trades
                                    vs_buy_hold = total_return - buy_hold

                                    composite = self._calculate_composite_score(
                                        win_rate, profit_factor, total_return, max_dd_pct, total_trades
                                    )

                                    results.append(VectorBTResult(
                                        strategy_name=strategy,
                                        strategy_category=self.ENTRY_STRATEGIES.get(strategy, {}).get('category', 'Unknown'),
                                        direction=direction,
                                        tp_percent=tp,
                                        sl_percent=sl,
                                        entry_rule=strategy,  # Use strategy KEY, not description
                                        total_trades=total_trades,
                                        wins=wins,
                                        losses=losses,
                                        win_rate=win_rate,
                                        total_pnl=total_pnl,
                                        total_pnl_percent=total_return,
                                        profit_factor=min(profit_factor, 999.0),
                                        max_drawdown=max_dd_pct * self.initial_capital / 100,
                                        max_drawdown_percent=max_dd_pct,
                                        avg_trade=avg_trade,
                                        avg_trade_percent=avg_trade_pct,
                                        buy_hold_return=buy_hold,
                                        vs_buy_hold=vs_buy_hold,
                                        beats_buy_hold=vs_buy_hold > 0,
                                        composite_score=composite,
                                        equity_curve=[],  # Skip for performance
                                        params={'tp_percent': tp, 'sl_percent': sl, 'direction': direction},
                                        data_start=data_start_str,
                                        data_end=data_end_str,
                                    ))

                                completed += 1

                                # Progress callback every 50 combinations for better watchdog responsiveness
                                if progress_callback and completed % 50 == 0:
                                    progress_callback(completed, total_combos)

                                # Checkpoint every 1000 combinations for crash recovery
                                if checkpoint_callback and completed % 1000 == 0:
                                    checkpoint_data = {
                                        'completed': completed,
                                        'total': total_combos,
                                        'current_strategy': strategy,
                                        'current_direction': direction,
                                        'results_count': len(results),
                                    }
                                    try:
                                        checkpoint_callback(checkpoint_data)
                                    except Exception as e:
                                        log(f"[VectorBT] Checkpoint save failed: {e}", level='DEBUG')

                        except Exception as extract_err:
                            log(f"[VectorBT] Vectorized extraction failed: {extract_err}, falling back", level='WARNING')
                            # Fallback to individual extraction
                            for tp, sl in col_tuples:
                                try:
                                    sub_pf = pf[(tp, sl)]
                                    result = self._extract_metrics(sub_pf, strategy, direction, tp, sl)
                                    if result.total_trades > 0:
                                        results.append(result)
                                except:
                                    pass
                                completed += 1
                                if progress_callback and completed % 50 == 0:
                                    progress_callback(completed, total_combos)

                        # CRITICAL: Explicitly free large objects to prevent memory leak
                        del pf
                        del close_tiled
                        del entries_tiled
                        del tp_arr
                        del sl_arr
                        import gc
                        gc.collect()

                        current_memory = get_memory_mb()
                        if current_memory > start_memory * 1.5:  # Memory grew by 50%+
                            log(f"[VectorBT] Memory warning: {current_memory:.0f} MB (started at {start_memory:.0f} MB)", level='WARNING')

                    except Exception as e:
                        log(f"[VectorBT] Broadcast error for {strategy}/{direction}: {e}", level='WARNING')
                        # Fall back to iterative approach
                        for tp in tp_range:
                            for sl in sl_range:
                                try:
                                    result = self.run_single_backtest(strategy, direction, tp, sl)
                                    if result.total_trades > 0:
                                        results.append(result)
                                except Exception as inner_e:
                                    log(f"[VectorBT] Fallback error: {inner_e}", level='DEBUG')

                                completed += 1

                                if progress_callback and completed % 50 == 0:
                                    progress_callback(completed, total_combos)

        # ========== PHASE 2: BIDIRECTIONAL STRATEGIES ==========
        if mode in ['bidirectional', 'all']:
            vbt_log(f"[VectorBT] Starting bidirectional optimization for {len(strategies)} strategies", level='DEBUG')

            for strategy in strategies:
                # Get both long and short signals
                long_entries = self._get_signals(strategy, 'long')
                short_entries = self._get_signals(strategy, 'short')

                # Skip if no signals in either direction
                if not long_entries.any() and not short_entries.any():
                    completed += len(tp_range) * len(sl_range)
                    continue

                # Pre-filter conflicting signals (where both fire on same bar)
                conflict_mask = long_entries & short_entries
                long_entries_clean = long_entries & ~conflict_mask
                short_entries_clean = short_entries & ~conflict_mask

                n_tp = len(tp_range)
                n_sl = len(sl_range)
                n_combos = n_tp * n_sl

                try:
                    # Create column labels for TP/SL combinations
                    col_tuples = [(tp, sl) for tp in tp_range for sl in sl_range]
                    columns = pd.MultiIndex.from_tuples(col_tuples, names=['tp', 'sl'])

                    # Tile close prices
                    close_tiled = pd.DataFrame(
                        np.tile(self.df['close'].values.reshape(-1, 1), (1, n_combos)),
                        index=self.df.index,
                        columns=columns
                    )

                    # Tile long entries
                    long_entries_tiled = pd.DataFrame(
                        np.tile(long_entries_clean.values.reshape(-1, 1), (1, n_combos)),
                        index=self.df.index,
                        columns=columns
                    )

                    # Tile short entries
                    short_entries_tiled = pd.DataFrame(
                        np.tile(short_entries_clean.values.reshape(-1, 1), (1, n_combos)),
                        index=self.df.index,
                        columns=columns
                    )

                    # TP/SL arrays
                    tp_arr = np.array([tp / 100 for tp, sl in col_tuples])
                    sl_arr = np.array([sl / 100 for tp, sl in col_tuples])

                    # Run bidirectional portfolio simulation
                    # NOTE: Must use size_type='value' (not 'percent') because VectorBT
                    # doesn't support position reversal with percentage-based sizing
                    position_value = self.initial_capital * (self.position_size_pct / 100)
                    pf = vbt.Portfolio.from_signals(
                        close=close_tiled,
                        entries=long_entries_tiled,
                        short_entries=short_entries_tiled,
                        sl_stop=sl_arr,
                        tp_stop=tp_arr,
                        size=position_value,
                        size_type='value',
                        fees=self.total_fees,
                        init_cash=self.initial_capital,
                        freq=self.data_freq,
                        upon_opposite_entry='Reverse',  # Flip on opposite signal
                    )

                    # Extract metrics for each TP/SL combination
                    try:
                        all_returns = pf.total_return() * 100
                        all_final_values = pf.final_value()
                        all_max_dd = pf.max_drawdown() * 100

                        trades = pf.trades
                        trade_counts = trades.count()

                        # Get win/loss counts per column
                        if hasattr(trades, 'pnl') and len(trades.pnl) > 0:
                            trade_pnl = trades.pnl.values
                            trade_cols = trades.col.values

                            n_cols = len(col_tuples)
                            wins_arr = np.zeros(n_cols)
                            gross_profit_arr = np.zeros(n_cols)
                            gross_loss_arr = np.zeros(n_cols)

                            for i in range(len(trade_pnl)):
                                col_idx = trade_cols[i]
                                pnl = trade_pnl[i]
                                if pnl > 0:
                                    wins_arr[col_idx] += 1
                                    gross_profit_arr[col_idx] += pnl
                                else:
                                    gross_loss_arr[col_idx] += abs(pnl)
                        else:
                            wins_arr = np.zeros(len(col_tuples))
                            gross_profit_arr = np.zeros(len(col_tuples))
                            gross_loss_arr = np.zeros(len(col_tuples))

                        buy_hold = float((self.df['close'].iloc[-1] / self.df['close'].iloc[0] - 1) * 100)

                        # Build results
                        for idx, (tp, sl) in enumerate(col_tuples):
                            total_trades = int(trade_counts.iloc[idx]) if idx < len(trade_counts) else 0

                            if total_trades > 0:
                                total_return = float(all_returns.iloc[idx])
                                total_pnl = float(all_final_values.iloc[idx]) - self.initial_capital
                                max_dd_pct = float(all_max_dd.iloc[idx])
                                wins = int(wins_arr[idx])
                                losses = total_trades - wins
                                win_rate = (wins / total_trades * 100)
                                gross_profit = gross_profit_arr[idx]
                                gross_loss = gross_loss_arr[idx]
                                profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0)

                                avg_trade = total_pnl / total_trades
                                avg_trade_pct = total_return / total_trades
                                vs_buy_hold = total_return - buy_hold

                                composite = self._calculate_composite_score(
                                    win_rate, profit_factor, total_return, max_dd_pct, total_trades
                                )

                                results.append(VectorBTResult(
                                    strategy_name=f"{strategy}_both",
                                    strategy_category=self.ENTRY_STRATEGIES.get(strategy, {}).get('category', 'Unknown'),
                                    direction='both',
                                    tp_percent=tp,
                                    sl_percent=sl,
                                    entry_rule=strategy,
                                    total_trades=total_trades,
                                    wins=wins,
                                    losses=losses,
                                    win_rate=win_rate,
                                    total_pnl=total_pnl,
                                    total_pnl_percent=total_return,
                                    profit_factor=min(profit_factor, 999.0),
                                    max_drawdown=max_dd_pct * self.initial_capital / 100,
                                    max_drawdown_percent=max_dd_pct,
                                    avg_trade=avg_trade,
                                    avg_trade_percent=avg_trade_pct,
                                    buy_hold_return=buy_hold,
                                    vs_buy_hold=vs_buy_hold,
                                    beats_buy_hold=vs_buy_hold > 0,
                                    composite_score=composite,
                                    equity_curve=[],
                                    params={'tp_percent': tp, 'sl_percent': sl, 'direction': 'both'},
                                    data_start=data_start_str,
                                    data_end=data_end_str,
                                ))

                            completed += 1

                            if progress_callback and completed % 50 == 0:
                                progress_callback(completed, total_combos)

                    except Exception as extract_err:
                        log(f"[VectorBT] Bidirectional extraction failed for {strategy}: {extract_err}", level='WARNING')
                        # Fallback to individual backtests
                        for tp, sl in col_tuples:
                            try:
                                result = self.run_bidirectional_backtest(strategy, tp, sl)
                                if result.total_trades > 0:
                                    results.append(result)
                            except:
                                pass
                            completed += 1
                            if progress_callback and completed % 50 == 0:
                                progress_callback(completed, total_combos)

                    # Memory cleanup
                    del pf
                    del close_tiled
                    del long_entries_tiled
                    del short_entries_tiled
                    del tp_arr
                    del sl_arr
                    import gc
                    gc.collect()

                except Exception as e:
                    log(f"[VectorBT] Bidirectional broadcast error for {strategy}: {e}", level='WARNING')
                    # Fallback to iterative
                    for tp in tp_range:
                        for sl in sl_range:
                            try:
                                result = self.run_bidirectional_backtest(strategy, tp, sl)
                                if result.total_trades > 0:
                                    results.append(result)
                            except Exception as inner_e:
                                log(f"[VectorBT] Bidirectional fallback error: {inner_e}", level='DEBUG')

                            completed += 1

                            if progress_callback and completed % 50 == 0:
                                progress_callback(completed, total_combos)

        # Sort by composite score and limit results to prevent OOM
        results.sort(key=lambda r: r.composite_score, reverse=True)

        # Apply max_results limit
        result_limit = max_results if max_results is not None else self.MAX_RESULTS
        if len(results) > result_limit:
            vbt_log(f"[VectorBT] Limiting results from {len(results)} to {result_limit} (max_results)", level='DEBUG')
            results = results[:result_limit]

        elapsed = time.time() - start_time
        combos_per_sec = total_combos / elapsed if elapsed > 0 else 0

        log(f"[VectorBT] Optimization complete: {len(results)} results from {total_combos:,} combinations in {elapsed:.1f}s ({combos_per_sec:,.0f}/sec)")

        # Compare to estimated iterative time
        estimated_iterative = total_combos * 0.1  # ~100ms per combo in iterative mode
        speedup = estimated_iterative / elapsed if elapsed > 0 else 0
        if speedup > 1:
            vbt_log(f"[VectorBT] Speedup: ~{speedup:.0f}x faster than iterative engine (est. {estimated_iterative/60:.1f} min iterative)", level='DEBUG')

        # MEMORY CLEANUP: Clear signal cache and force garbage collection at end of optimization
        self._signal_cache.clear()
        import gc
        gc.collect()
        vbt_log(f"[VectorBT] Memory cleanup completed - signal cache cleared", level='DEBUG')

        # Log result statistics summary
        if results:
            returns = [r.total_pnl_percent for r in results if r.total_pnl_percent is not None]
            win_rates = [r.win_rate for r in results if r.win_rate is not None]
            scores = [r.composite_score for r in results if r.composite_score is not None]

            stats = {
                'total_results': len(results),
                'profitable': sum(1 for r in returns if r > 0),
                'avg_return': f"{sum(returns)/len(returns):.2f}%" if returns else 'N/A',
                'max_return': f"{max(returns):.2f}%" if returns else 'N/A',
                'avg_win_rate': f"{sum(win_rates)/len(win_rates):.1f}%" if win_rates else 'N/A',
                'top_score': f"{max(scores):.1f}" if scores else 'N/A',
            }
            vbt_log(f"[VectorBT] Results summary: {stats}", level='DEBUG')

        end_memory = get_memory_mb()
        vbt_log(f"[VectorBT] Optimization complete - Memory: {end_memory:.0f} MB (delta: {end_memory - start_memory:+.0f} MB)", level='DEBUG')

        return results

    def _extract_metrics(
        self,
        pf,
        strategy: str,
        direction: str,
        tp_percent: float,
        sl_percent: float,
        include_equity_curve: bool = False,
        entries: pd.Series = None,
    ) -> VectorBTResult:
        """Extract metrics from VectorBT portfolio to VectorBTResult.

        Args:
            include_equity_curve: If False (default), skip expensive equity curve extraction.
                                  Only set True for final top results display.
            entries: Entry signals series. If provided with include_equity_curve=True,
                     trades_list with debug fields will be populated.
        """
        try:
            trades = pf.trades
            total_trades = trades.count()

            if total_trades == 0:
                return self._empty_result(strategy, direction, tp_percent, sl_percent)

            # Core metrics
            wins = int((trades.pnl.values > 0).sum()) if hasattr(trades, 'pnl') else 0
            losses = total_trades - wins
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            total_return = float(pf.total_return()) * 100
            total_pnl = float(pf.final_value() - self.initial_capital)

            # Profit factor
            gross_profits = float(trades.pnl.values[trades.pnl.values > 0].sum()) if hasattr(trades, 'pnl') else 0
            gross_losses = abs(float(trades.pnl.values[trades.pnl.values < 0].sum())) if hasattr(trades, 'pnl') else 0
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else (999.0 if gross_profits > 0 else 0)

            # Drawdown
            max_dd_pct = float(pf.max_drawdown()) * 100
            max_dd = float(pf.max_drawdown() * self.initial_capital)

            # Average trade
            avg_trade = total_pnl / total_trades if total_trades > 0 else 0
            avg_trade_pct = total_return / total_trades if total_trades > 0 else 0

            # Buy & hold comparison
            buy_hold = float((self.df['close'].iloc[-1] / self.df['close'].iloc[0] - 1) * 100)
            vs_buy_hold = total_return - buy_hold

            # Equity curve - SKIP during optimization (expensive!)
            # Only extract for top results when displaying to user
            equity = pf.value().values.tolist() if include_equity_curve else []

            # Trades list with debug fields - SKIP during optimization (expensive!)
            # Only extract for top results when displaying to user
            trades_list = []
            if include_equity_curve and entries is not None:
                trades_list = self._build_trades_list(pf, direction, tp_percent, sl_percent, entries)

            # Calculate composite score
            composite = self._calculate_composite_score(
                win_rate, profit_factor, total_return, max_dd_pct, total_trades
            )

            return VectorBTResult(
                strategy_name=strategy,
                strategy_category=self.ENTRY_STRATEGIES.get(strategy, {}).get('category', 'Unknown'),
                direction=direction,
                tp_percent=tp_percent,
                sl_percent=sl_percent,
                entry_rule=strategy,  # Use strategy KEY, not description
                total_trades=int(total_trades),
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_pnl_percent=total_return,
                profit_factor=min(profit_factor, 999.0),
                max_drawdown=max_dd,
                max_drawdown_percent=max_dd_pct,
                avg_trade=avg_trade,
                avg_trade_percent=avg_trade_pct,
                buy_hold_return=buy_hold,
                vs_buy_hold=vs_buy_hold,
                beats_buy_hold=vs_buy_hold > 0,
                composite_score=composite,
                equity_curve=equity,
                trades_list=trades_list,
                params={
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'direction': direction,
                    'entry_rule': strategy,  # Use strategy KEY, not description (fixes Pine Script generation)
                }
            )

        except Exception as e:
            log(f"[VectorBT] Metric extraction error: {e}", level='WARNING')
            return self._empty_result(strategy, direction, tp_percent, sl_percent)

    def _build_trades_list(
        self,
        pf,
        direction: str,
        tp_percent: float,
        sl_percent: float,
        entries: pd.Series,
    ) -> List[Dict]:
        """
        Build detailed trades list with debug fields for discrepancy debugging.

        Args:
            pf: VectorBT portfolio object
            direction: 'long' or 'short'
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage
            entries: Entry signals series

        Returns:
            List of trade dictionaries with debug fields
        """
        trades_list = []

        try:
            trades = pf.trades
            if trades.count() == 0:
                return trades_list

            # Get trade records as DataFrame
            records = trades.records_readable

            for idx, trade in records.iterrows():
                try:
                    # VectorBT records_readable provides these columns directly:
                    # 'Entry Timestamp', 'Exit Timestamp', 'Avg Entry Price', 'Avg Exit Price', 'PnL', 'Return'

                    # Get timestamps directly from VectorBT (not from indices)
                    entry_bar_time = trade.get('Entry Timestamp', None)
                    exit_bar_time = trade.get('Exit Timestamp', None)

                    # Convert to string if timestamp object
                    if hasattr(entry_bar_time, 'strftime'):
                        entry_bar_time = str(entry_bar_time)
                    if hasattr(exit_bar_time, 'strftime'):
                        exit_bar_time = str(exit_bar_time)

                    # Get entry/exit indices from timestamps (needed for indicator lookup and exit type)
                    entry_idx = 0
                    exit_idx = 0
                    signal_bar_time = None

                    if entry_bar_time:
                        try:
                            entry_ts = pd.Timestamp(entry_bar_time)
                            if entry_ts in self.df.index:
                                entry_idx = self.df.index.get_loc(entry_ts)
                                if entry_idx > 0:
                                    signal_bar_time = str(self.df.index[entry_idx - 1])
                        except:
                            pass

                    if exit_bar_time:
                        try:
                            exit_ts = pd.Timestamp(exit_bar_time)
                            if exit_ts in self.df.index:
                                exit_idx = self.df.index.get_loc(exit_ts)
                        except:
                            pass

                    # Get prices directly from VectorBT
                    entry_price = float(trade.get('Avg Entry Price', 0) or 0)
                    exit_price = float(trade.get('Avg Exit Price', 0) or 0)
                    pnl = float(trade.get('PnL', 0) or 0)
                    pnl_pct = float(trade.get('Return', 0) or 0) * 100

                    # Calculate TP/SL prices based on direction
                    if direction == 'long':
                        tp_price = entry_price * (1 + tp_percent / 100)
                        sl_price = entry_price * (1 - sl_percent / 100)
                    else:  # short
                        tp_price = entry_price * (1 - tp_percent / 100)
                        sl_price = entry_price * (1 + sl_percent / 100)

                    # Determine exit type by comparing exit price to TP/SL
                    exit_type = self._determine_exit_type(
                        exit_price, entry_price, tp_price, sl_price, direction, exit_idx, len(self.df)
                    )

                    # Get indicator values at entry for debugging
                    entry_indicators = self._get_entry_indicators(entry_idx)

                    # Build trade record with debug fields
                    trade_record = {
                        # Standard fields
                        'trade_num': idx + 1,
                        'direction': direction,
                        'entry_time': str(entry_bar_time) if entry_bar_time else None,
                        'exit_time': str(exit_bar_time) if exit_bar_time else None,
                        'entry': round(entry_price, 2),
                        'exit': round(exit_price, 2),
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'result': 'WIN' if pnl > 0 else 'LOSS',

                        # NEW DEBUG FIELDS
                        'signal_bar_time': str(signal_bar_time) if signal_bar_time else None,
                        'entry_bar_time': str(entry_bar_time) if entry_bar_time else None,
                        'exit_bar_time': str(exit_bar_time) if exit_bar_time else None,
                        'tp_price': round(tp_price, 2),
                        'sl_price': round(sl_price, 2),
                        'exit_type': exit_type,
                        'entry_indicators': entry_indicators,
                    }

                    trades_list.append(trade_record)

                except Exception as e:
                    log(f"[VectorBT] Error processing trade {idx}: {e}", level='DEBUG')
                    continue

        except Exception as e:
            log(f"[VectorBT] Error building trades list: {e}", level='WARNING')

        return trades_list

    def _determine_exit_type(
        self,
        exit_price: float,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        direction: str,
        exit_idx: int,
        data_length: int,
    ) -> str:
        """
        Determine how the trade exited based on exit price vs TP/SL levels.

        Returns:
            "TP_HIT", "SL_HIT", "EMERGENCY", or "SIGNAL"
        """
        # Small tolerance for price comparison (0.01%)
        tolerance = 0.0001

        if direction == 'long':
            # For long: TP is above entry, SL is below entry
            if exit_price >= tp_price * (1 - tolerance):
                return "TP_HIT"
            elif exit_price <= sl_price * (1 + tolerance):
                return "SL_HIT"
        else:  # short
            # For short: TP is below entry, SL is above entry
            if exit_price <= tp_price * (1 + tolerance):
                return "TP_HIT"
            elif exit_price >= sl_price * (1 - tolerance):
                return "SL_HIT"

        # If exited at end of data, it's an emergency close
        if exit_idx >= data_length - 1:
            return "EMERGENCY"

        # Otherwise, exited due to a new signal (position flip)
        return "SIGNAL"

    def _get_entry_indicators(self, entry_idx: int) -> Dict[str, float]:
        """
        Get indicator values at entry bar for debugging.

        Args:
            entry_idx: Index of entry bar in DataFrame

        Returns:
            Dictionary with RSI, ATR, ADX, and BB position values
        """
        try:
            if entry_idx < 0 or entry_idx >= len(self.df):
                return {'rsi': None, 'atr': None, 'adx': None, 'bb_position': None}

            row = self.df.iloc[entry_idx]

            # RSI value
            rsi = float(row.get('rsi', 0)) if pd.notna(row.get('rsi')) else None

            # ATR value
            atr = float(row.get('atr', 0)) if pd.notna(row.get('atr')) else None

            # ADX value
            adx = float(row.get('adx', 0)) if pd.notna(row.get('adx')) else None

            # BB position: 0 = at lower band, 0.5 = middle, 1 = upper band
            bb_upper = row.get('bb_upper')
            bb_lower = row.get('bb_lower')
            close = row.get('close')

            if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(close):
                bb_range = bb_upper - bb_lower
                if bb_range > 0:
                    bb_position = float((close - bb_lower) / bb_range)
                    bb_position = max(0.0, min(1.0, bb_position))  # Clamp to 0-1
                else:
                    bb_position = 0.5
            else:
                bb_position = None

            return {
                'rsi': round(rsi, 2) if rsi is not None else None,
                'atr': round(atr, 4) if atr is not None else None,
                'adx': round(adx, 2) if adx is not None else None,
                'bb_position': round(bb_position, 4) if bb_position is not None else None,
            }

        except Exception as e:
            log(f"[VectorBT] Error getting entry indicators at idx {entry_idx}: {e}", level='DEBUG')
            return {'rsi': None, 'atr': None, 'adx': None, 'bb_position': None}

    def _calculate_composite_score(
        self,
        win_rate: float,
        profit_factor: float,
        total_return: float,
        max_drawdown: float,
        total_trades: int,
    ) -> float:
        """
        Calculate composite score for ranking strategies.
        Matches StrategyEngine._calculate_composite_score() logic.
        """
        if total_trades < 1:
            return 0.0

        # Win rate component (0-40 points)
        wr_score = min(win_rate / 100 * 40, 40)

        # Profit factor component (0-30 points)
        pf_capped = min(profit_factor, 5.0)  # Cap at 5
        pf_score = (pf_capped / 5.0) * 30

        # Return component (0-20 points)
        ret_score = min(max(total_return / 100, 0) * 20, 20)

        # Drawdown penalty (-10 to 0 points)
        dd_penalty = min(max_drawdown / 50, 1.0) * 10  # -10 at 50%+ DD

        # Trade count bonus (0-5 points for statistical significance)
        trade_bonus = min(total_trades / 20, 1.0) * 5

        composite = wr_score + pf_score + ret_score - dd_penalty + trade_bonus

        return max(0, composite)

    def _empty_result(
        self,
        strategy: str,
        direction: str,
        tp_percent: float,
        sl_percent: float,
    ) -> VectorBTResult:
        """Return an empty result for no-trade scenarios."""
        return VectorBTResult(
            strategy_name=strategy,
            strategy_category=self.ENTRY_STRATEGIES.get(strategy, {}).get('category', 'Unknown'),
            direction=direction,
            tp_percent=tp_percent,
            sl_percent=sl_percent,
            entry_rule=strategy,  # Use strategy KEY, not description
        )


def is_vectorbt_available() -> bool:
    """Check if VectorBT is available."""
    return VECTORBT_AVAILABLE
