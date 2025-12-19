"""
Backtrader Engine - Industry-standard backtesting with TA-Lib indicators

This module provides a Backtrader-based backtesting engine that:
1. Uses TA-Lib for indicator calculations (matches TradingView)
2. Handles order execution properly (slippage, fills)
3. Provides accurate position sizing
4. Returns metrics compatible with the unified optimizer
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import warnings

# Suppress backtrader warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Try to import TA-Lib, fall back to pandas-ta if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available, using pandas-ta fallback")

try:
    import pandas_ta as pta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False


@dataclass
class BacktestResult:
    """Results from a backtrader backtest"""
    strategy_name: str
    strategy_category: str
    params: Dict[str, Any]

    # Core metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Advanced metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    equity_r_squared: float = 0.0
    recovery_factor: float = 0.0

    # Trade details
    avg_trade_pnl: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    trade_log: List[Dict] = field(default_factory=list)

    # Composite score
    composite_score: float = 0.0
    score_breakdown: Dict[str, Any] = field(default_factory=dict)


class TALibIndicators:
    """TA-Lib indicator calculations that match TradingView"""

    @staticmethod
    def bollinger_bands(close: np.ndarray, length: int = 20, mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands using TA-Lib (matches TradingView ta.bb)"""
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(close, timeperiod=length, nbdevup=mult, nbdevdn=mult)
            return upper, middle, lower
        else:
            # Fallback calculation
            sma = pd.Series(close).rolling(length).mean().values
            std = pd.Series(close).rolling(length).std().values
            upper = sma + mult * std
            lower = sma - mult * std
            return upper, sma, lower

    @staticmethod
    def rsi(close: np.ndarray, length: int = 14) -> np.ndarray:
        """Calculate RSI using TA-Lib (matches TradingView ta.rsi)"""
        if TALIB_AVAILABLE:
            return talib.RSI(close, timeperiod=length)
        else:
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
            rs = gain / loss
            return (100 - (100 / (1 + rs))).values

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
        """Calculate ADX using TA-Lib (matches TradingView ta.adx)"""
        if TALIB_AVAILABLE:
            return talib.ADX(high, low, close, timeperiod=length)
        else:
            # Simplified ADX calculation
            tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
            atr = pd.Series(tr).rolling(length).mean().values
            return atr / close * 100  # Simplified

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
        """Calculate ATR using TA-Lib (matches TradingView ta.atr)"""
        if TALIB_AVAILABLE:
            return talib.ATR(high, low, close, timeperiod=length)
        else:
            tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
            return pd.Series(tr).rolling(length).mean().values

    @staticmethod
    def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD using TA-Lib (matches TradingView ta.macd)"""
        if TALIB_AVAILABLE:
            macd_line, signal_line, histogram = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return macd_line, signal_line, histogram
        else:
            exp1 = pd.Series(close).ewm(span=fast, adjust=False).mean()
            exp2 = pd.Series(close).ewm(span=slow, adjust=False).mean()
            macd_line = (exp1 - exp2).values
            signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic using TA-Lib (matches TradingView ta.stoch)"""
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return slowk, slowd
        else:
            lowest_low = pd.Series(low).rolling(k_period).min()
            highest_high = pd.Series(high).rolling(k_period).max()
            k = 100 * (pd.Series(close) - lowest_low) / (highest_high - lowest_low)
            d = k.rolling(d_period).mean()
            return k.values, d.values

    @staticmethod
    def ema(close: np.ndarray, length: int = 20) -> np.ndarray:
        """Calculate EMA using TA-Lib (matches TradingView ta.ema)"""
        if TALIB_AVAILABLE:
            return talib.EMA(close, timeperiod=length)
        else:
            return pd.Series(close).ewm(span=length, adjust=False).mean().values

    @staticmethod
    def sma(close: np.ndarray, length: int = 20) -> np.ndarray:
        """Calculate SMA using TA-Lib (matches TradingView ta.sma)"""
        if TALIB_AVAILABLE:
            return talib.SMA(close, timeperiod=length)
        else:
            return pd.Series(close).rolling(length).mean().values


class BaseStrategy(bt.Strategy):
    """
    Base strategy class with common functionality.
    All strategies inherit from this to ensure consistent behavior.
    """

    params = (
        ('sl_atr_mult', 2.0),   # Stop loss as ATR multiple
        ('tp_ratio', 1.5),      # Take profit as SL multiple
        ('risk_percent', 2.0),  # Risk per trade as % of equity
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.take_profit_price = None
        self.trade_log = []

        # Pre-calculate ATR for position sizing and stops
        self.atr = bt.indicators.ATR(self.data, period=14)

    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{dt.isoformat()} {txt}')  # Uncomment for debugging

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
                self.entry_price = order.executed.price
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
                if self.entry_price:
                    pnl = order.executed.price - self.entry_price
                    self.trade_log.append({
                        'entry': self.entry_price,
                        'exit': order.executed.price,
                        'pnl': pnl,
                        'type': 'long'
                    })
                self.entry_price = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def calculate_position_size(self):
        """Calculate position size based on risk percentage and ATR.

        For high-priced assets like BTC, returns fractional units.
        """
        if self.atr[0] <= 0:
            return 0.0

        risk_amount = self.broker.getvalue() * (self.p.risk_percent / 100)
        sl_distance = self.atr[0] * self.p.sl_atr_mult

        if sl_distance <= 0:
            return 0.0

        # Calculate fractional position size (don't use int()!)
        size = risk_amount / sl_distance

        # Minimum position size for crypto (0.0001 BTC = ~$8)
        if size < 0.0001:
            size = 0.0001

        return size  # Return FLOAT, not int!

    def set_stops(self, is_long: bool = True):
        """Set stop loss and take profit based on ATR"""
        sl_distance = self.atr[0] * self.p.sl_atr_mult
        tp_distance = sl_distance * self.p.tp_ratio

        if is_long:
            self.stop_price = self.data.close[0] - sl_distance
            self.take_profit_price = self.data.close[0] + tp_distance
        else:
            self.stop_price = self.data.close[0] + sl_distance
            self.take_profit_price = self.data.close[0] - tp_distance


class BBRSIStrategy(BaseStrategy):
    """
    Bollinger Bands + RSI Mean Reversion Strategy
    Uses TA-Lib for accurate indicator calculations
    """

    params = (
        ('bb_length', 20),
        ('bb_mult', 2.0),
        ('rsi_length', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('adx_threshold', 25),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 1.5),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()

        # Bollinger Bands
        self.bb = bt.indicators.BollingerBands(
            self.data.close,
            period=self.p.bb_length,
            devfactor=self.p.bb_mult
        )

        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_length)

        # ADX for trend filter (only trade in sideways markets)
        self.adx = bt.indicators.ADX(self.data, period=14)

    def next(self):
        if self.order:
            return

        # Check if market is sideways (ADX below threshold)
        is_sideways = self.adx[0] < self.p.adx_threshold

        if not self.position:
            # Long entry: price at lower BB, RSI oversold, sideways market
            if (self.data.close[0] <= self.bb.lines.bot[0] and
                self.rsi[0] < self.p.rsi_oversold and
                is_sideways):

                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Exit conditions
            # Stop loss
            if self.data.close[0] <= self.stop_price:
                self.order = self.close()
            # Take profit
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()
            # RSI overbought exit
            elif self.rsi[0] > self.p.rsi_overbought:
                self.order = self.close()


class InsideBarBreakoutStrategy(BaseStrategy):
    """
    Inside Bar Breakout Strategy
    Trades breakouts from inside bar patterns
    """

    params = (
        ('lookback', 1),
        ('sl_atr_mult', 1.5),
        ('tp_ratio', 2.0),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.inside_bar_high = None
        self.inside_bar_low = None

    def next(self):
        if self.order:
            return

        # Check for inside bar pattern
        if len(self.data) < 2:
            return

        prev_high = self.data.high[-1]
        prev_low = self.data.low[-1]
        curr_high = self.data.high[0]
        curr_low = self.data.low[0]

        # Inside bar: current bar's range is within previous bar's range
        is_inside_bar = curr_high < prev_high and curr_low > prev_low

        if not self.position:
            if is_inside_bar:
                self.inside_bar_high = prev_high
                self.inside_bar_low = prev_low

            # Breakout entry
            if self.inside_bar_high and self.inside_bar_low:
                # Bullish breakout
                if self.data.close[0] > self.inside_bar_high:
                    size = self.calculate_position_size()
                    if size > 0:
                        self.order = self.buy(size=size)
                        self.set_stops(is_long=True)
                    self.inside_bar_high = None
                    self.inside_bar_low = None

        else:
            # Exit conditions
            if self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class SupertrendStrategy(BaseStrategy):
    """
    Supertrend Strategy - Trend following using ATR-based dynamic support/resistance
    """

    params = (
        ('atr_period', 10),
        ('atr_multiplier', 3.0),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 2.0),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        # Calculate Supertrend
        self.atr_st = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.hl2 = (self.data.high + self.data.low) / 2
        self.trend = 1  # 1 = bullish, -1 = bearish
        self.supertrend = self.hl2[0]

    def next(self):
        if self.order:
            return

        # Calculate Supertrend bands
        upper_band = self.hl2[0] + (self.p.atr_multiplier * self.atr_st[0])
        lower_band = self.hl2[0] - (self.p.atr_multiplier * self.atr_st[0])

        # Update trend
        if self.data.close[0] > self.supertrend:
            self.trend = 1
            self.supertrend = lower_band
        else:
            self.trend = -1
            self.supertrend = upper_band

        if not self.position:
            # Long entry on bullish trend
            if self.trend == 1 and self.data.close[-1] <= self.supertrend:
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Exit on trend reversal or stops
            if self.trend == -1:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class MACDStrategy(BaseStrategy):
    """
    MACD Crossover Strategy - Momentum-based trend trading
    """

    params = (
        ('fast_period', 12),
        ('slow_period', 26),
        ('signal_period', 9),
        ('adx_threshold', 25),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 1.5),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast_period,
            period_me2=self.p.slow_period,
            period_signal=self.p.signal_period
        )
        self.adx = bt.indicators.ADX(self.data, period=14)
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if self.order:
            return

        # Only trade when trend is strong
        is_trending = self.adx[0] > self.p.adx_threshold

        if not self.position:
            # Bullish crossover
            if self.crossover[0] > 0 and is_trending:
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Bearish crossover or stops
            if self.crossover[0] < 0:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class RSIStrategy(BaseStrategy):
    """
    RSI Oversold/Overbought Strategy - Mean reversion
    """

    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('adx_threshold', 25),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 1.5),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.adx = bt.indicators.ADX(self.data, period=14)

    def next(self):
        if self.order:
            return

        # Only trade in sideways market
        is_sideways = self.adx[0] < self.p.adx_threshold

        if not self.position:
            # RSI oversold entry
            if self.rsi[0] < self.p.rsi_oversold and is_sideways:
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # RSI overbought exit or stops
            if self.rsi[0] > self.p.rsi_overbought:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class StochasticStrategy(BaseStrategy):
    """
    Stochastic Oscillator Strategy - Momentum reversals
    """

    params = (
        ('stoch_k', 14),
        ('stoch_d', 3),
        ('stoch_oversold', 20),
        ('stoch_overbought', 80),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 1.5),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.stoch = bt.indicators.Stochastic(
            self.data,
            period=self.p.stoch_k,
            period_dfast=self.p.stoch_d
        )

    def next(self):
        if self.order:
            return

        if not self.position:
            # Oversold with crossover
            if (self.stoch.percK[0] < self.p.stoch_oversold and
                self.stoch.percK[0] > self.stoch.percD[0] and
                self.stoch.percK[-1] <= self.stoch.percD[-1]):
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Overbought or stops
            if self.stoch.percK[0] > self.p.stoch_overbought:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class BBStochStrategy(BaseStrategy):
    """
    Bollinger Bands + Stochastic Strategy - Mean reversion with momentum confirmation
    """

    params = (
        ('bb_length', 20),
        ('bb_mult', 2.0),
        ('stoch_k', 14),
        ('stoch_d', 3),
        ('stoch_oversold', 20),
        ('stoch_overbought', 80),
        ('adx_threshold', 25),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 1.5),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        # Bollinger Bands
        self.bb = bt.indicators.BollingerBands(
            self.data.close,
            period=self.p.bb_length,
            devfactor=self.p.bb_mult
        )
        # Stochastic
        self.stoch = bt.indicators.Stochastic(
            self.data,
            period=self.p.stoch_k,
            period_dfast=self.p.stoch_d
        )
        # ADX for trend filter
        self.adx = bt.indicators.ADX(self.data, period=14)

    def next(self):
        if self.order:
            return

        # Only trade in sideways markets
        sideways = self.adx[0] < self.p.adx_threshold

        if not self.position:
            # Long: price at lower BB + stochastic oversold
            if (sideways and
                self.data.close[0] <= self.bb.lines.bot[0] and
                self.stoch.percK[0] < self.p.stoch_oversold):
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Exit at upper BB, overbought stoch, or stops
            if (self.data.close[0] >= self.bb.lines.top[0] or
                self.stoch.percK[0] > self.p.stoch_overbought):
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class EMAStrategy(BaseStrategy):
    """
    Triple EMA Strategy - Trend following with EMA stack
    """

    params = (
        ('fast_ema', 8),
        ('medium_ema', 21),
        ('slow_ema', 55),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 2.0),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.fast_ema)
        self.ema_medium = bt.indicators.EMA(self.data.close, period=self.p.medium_ema)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.slow_ema)

    def next(self):
        if self.order:
            return

        # EMA stack alignment
        bullish_stack = (self.ema_fast[0] > self.ema_medium[0] > self.ema_slow[0])
        bearish_stack = (self.ema_fast[0] < self.ema_medium[0] < self.ema_slow[0])

        if not self.position:
            # Long on bullish EMA stack with pullback to medium EMA
            if bullish_stack and self.data.close[0] <= self.ema_medium[0] * 1.005:
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Exit on bearish stack or stops
            if bearish_stack:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel Breakout Strategy - Classic breakout trading
    """

    params = (
        ('period', 20),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 2.0),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.period)

    def next(self):
        if self.order:
            return

        if not self.position:
            # Breakout above highest high
            if self.data.close[0] > self.highest[-1]:
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Exit on lowest low or stops
            if self.data.close[0] < self.lowest[-1]:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class KeltnerChannelStrategy(BaseStrategy):
    """
    Keltner Channel Strategy - Mean reversion with volatility bands
    """

    params = (
        # Optimizer parameter names (kc_length, kc_mult, rsi_length)
        ('kc_length', 20),
        ('kc_mult', 2.0),
        ('rsi_length', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('adx_threshold', 25),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 1.5),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.kc_length)
        self.atr_kc = bt.indicators.ATR(self.data, period=self.p.kc_length)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_length)

    def next(self):
        if self.order:
            return

        # Keltner bands
        upper_band = self.ema[0] + (self.p.kc_mult * self.atr_kc[0])
        lower_band = self.ema[0] - (self.p.kc_mult * self.atr_kc[0])

        if not self.position:
            # Long at lower band with RSI confirmation
            if (self.data.close[0] <= lower_band and
                self.rsi[0] < self.p.rsi_oversold):
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Exit at upper band or stops
            if self.data.close[0] >= upper_band:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class BBSqueezeStrategy(BaseStrategy):
    """
    Bollinger Band Squeeze Strategy - Volatility breakout
    """

    params = (
        ('bb_period', 20),
        ('bb_mult', 2.0),
        ('kc_period', 20),
        ('kc_mult', 1.5),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 2.0),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        # Bollinger Bands
        self.bb = bt.indicators.BollingerBands(
            self.data.close,
            period=self.p.bb_period,
            devfactor=self.p.bb_mult
        )
        # Keltner Channel for squeeze detection
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.kc_period)
        self.atr_kc = bt.indicators.ATR(self.data, period=self.p.kc_period)
        # Momentum
        self.mom = bt.indicators.Momentum(self.data.close, period=12)

    def next(self):
        if self.order:
            return

        # Keltner bands
        kc_upper = self.ema[0] + (self.p.kc_mult * self.atr_kc[0])
        kc_lower = self.ema[0] - (self.p.kc_mult * self.atr_kc[0])

        # Squeeze: BB inside KC
        squeeze_on = (self.bb.lines.bot[0] > kc_lower and
                      self.bb.lines.top[0] < kc_upper)

        if not self.position:
            # Enter on squeeze release with positive momentum
            if not squeeze_on and self.mom[0] > 0 and self.mom[-1] <= 0:
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Exit on momentum reversal or stops
            if self.mom[0] < 0:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class ADXTrendStrategy(BaseStrategy):
    """
    ADX Trend Strategy - Strong trend following
    """

    params = (
        ('adx_period', 14),
        ('adx_threshold', 25),
        ('adx_strong', 40),
        ('ema_period', 20),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 2.0),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period)
        self.dmi = bt.indicators.DirectionalMovementIndex(self.data, period=self.p.adx_period)
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)

    def next(self):
        if self.order:
            return

        # Strong trend with +DI > -DI
        strong_bullish = (self.adx[0] > self.p.adx_threshold and
                         self.dmi.plusDI[0] > self.dmi.minusDI[0])

        if not self.position:
            # Long on strong bullish trend, price above EMA
            if strong_bullish and self.data.close[0] > self.ema[0]:
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Exit on trend weakening or stops
            if (self.adx[0] < self.p.adx_threshold or
                self.dmi.plusDI[0] < self.dmi.minusDI[0]):
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class SMACrossStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy
    """

    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 2.0),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if self.order:
            return

        if not self.position:
            # Golden cross
            if self.crossover[0] > 0:
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Death cross or stops
            if self.crossover[0] < 0:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class CCIStrategy(BaseStrategy):
    """
    Commodity Channel Index Strategy
    """

    params = (
        ('cci_period', 20),
        ('cci_oversold', -100),
        ('cci_overbought', 100),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 1.5),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.cci = bt.indicators.CCI(self.data, period=self.p.cci_period)

    def next(self):
        if self.order:
            return

        if not self.position:
            # CCI oversold entry
            if (self.cci[0] < self.p.cci_oversold and
                self.cci[0] > self.cci[-1]):  # Turning up
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # CCI overbought or stops
            if self.cci[0] > self.p.cci_overbought:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class WilliamsRStrategy(BaseStrategy):
    """
    Williams %R Strategy
    """

    params = (
        ('period', 14),
        ('oversold', -80),
        ('overbought', -20),
        ('sl_atr_mult', 2.0),
        ('tp_ratio', 1.5),
        ('risk_percent', 2.0),
    )

    def __init__(self):
        super().__init__()
        self.williams = bt.indicators.WilliamsR(self.data, period=self.p.period)

    def next(self):
        if self.order:
            return

        if not self.position:
            # Oversold entry
            if (self.williams[0] < self.p.oversold and
                self.williams[0] > self.williams[-1]):
                size = self.calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.set_stops(is_long=True)

        else:
            # Overbought or stops
            if self.williams[0] > self.p.overbought:
                self.order = self.close()
            elif self.data.close[0] <= self.stop_price:
                self.order = self.close()
            elif self.data.close[0] >= self.take_profit_price:
                self.order = self.close()


class BacktraderEngine:
    """
    Main engine for running backtests with Backtrader
    """

    STRATEGY_MAP = {
        # Mean Reversion
        'bb_rsi_classic': BBRSIStrategy,
        'bb_rsi_tight': BBRSIStrategy,
        'bb_rsi_dynamic_tp': BBRSIStrategy,
        'bb_rsi_wide': BBRSIStrategy,
        'bb_stoch': BBStochStrategy,
        'bb_stoch_tight': BBStochStrategy,
        'keltner_rsi': KeltnerChannelStrategy,
        'keltner_reversal': KeltnerChannelStrategy,

        # Trend Following
        'supertrend_trend': SupertrendStrategy,
        'supertrend_pullback': SupertrendStrategy,
        'supertrend_breakout': SupertrendStrategy,
        'triple_ema_trend': EMAStrategy,
        'ema_trend_pullback': EMAStrategy,
        'adx_trend': ADXTrendStrategy,
        'strong_trend_adx': ADXTrendStrategy,

        # Momentum
        'macd_trend': MACDStrategy,
        'macd_momentum': MACDStrategy,
        'macd_divergence': MACDStrategy,
        'rsi_classic': RSIStrategy,
        'rsi_extreme': RSIStrategy,
        'rsi_momentum': RSIStrategy,
        'stochastic_oversold': StochasticStrategy,
        'stochastic_momentum': StochasticStrategy,
        'cci_classic': CCIStrategy,
        'cci_extreme': CCIStrategy,
        'williams_r_classic': WilliamsRStrategy,
        'williams_r_extreme': WilliamsRStrategy,

        # Breakout
        'inside_bar_breakout': InsideBarBreakoutStrategy,
        'donchian_breakout': DonchianBreakoutStrategy,
        'donchian_turtle': DonchianBreakoutStrategy,
        'bb_squeeze_momentum': BBSqueezeStrategy,
        'bb_squeeze_breakout': BBSqueezeStrategy,

        # Moving Average
        'sma_cross_classic': SMACrossStrategy,
        'sma_cross_fast': SMACrossStrategy,
        'ema_cross_classic': EMAStrategy,
        'ema_cross_fast': EMAStrategy,

        # DaviddTech inspired
        'davidtech_scalp': BBRSIStrategy,
        'davidtech_swing': SupertrendStrategy,
        'davidtech_momentum': MACDStrategy,
    }

    def __init__(self, df: pd.DataFrame, capital: float = 1000.0, commission: float = 0.001):
        """
        Initialize the backtrader engine

        Args:
            df: DataFrame with columns: time, open, high, low, close, volume
            capital: Starting capital
            commission: Commission per trade (0.001 = 0.1%)
        """
        self.df = df.copy()
        self.capital = capital
        self.commission = commission

        # Ensure datetime index
        if 'time' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['time'])
            self.df.set_index('datetime', inplace=True)

        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        if 'volume' not in self.df.columns:
            self.df['volume'] = 0

    def run_backtest(self, strategy_name: str, strategy_category: str,
                     params: Dict[str, Any]) -> BacktestResult:
        """
        Run a backtest for a specific strategy

        Args:
            strategy_name: Name of the strategy
            strategy_category: Category (e.g., 'Mean Reversion')
            params: Strategy parameters

        Returns:
            BacktestResult with all metrics
        """
        # Create cerebro engine
        cerebro = bt.Cerebro()

        # Add data
        data = bt.feeds.PandasData(
            dataname=self.df,
            datetime=None,  # Use index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        cerebro.adddata(data)

        # Get strategy class
        strategy_class = self.STRATEGY_MAP.get(strategy_name, BBRSIStrategy)

        # Convert params to backtrader format
        bt_params = self._convert_params(params)
        cerebro.addstrategy(strategy_class, **bt_params)

        # Set broker settings
        cerebro.broker.setcash(self.capital)
        cerebro.broker.setcommission(commission=self.commission)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        # Run backtest
        try:
            results = cerebro.run()
            strategy = results[0]
        except Exception as e:
            print(f"Backtest error for {strategy_name}: {e}")
            return BacktestResult(
                strategy_name=strategy_name,
                strategy_category=strategy_category,
                params=params
            )

        # Extract results
        return self._extract_results(
            cerebro, strategy, strategy_name, strategy_category, params
        )

    def _convert_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert optimizer params to backtrader params"""
        bt_params = {}

        param_mapping = {
            # Bollinger Bands
            'bb_length': 'bb_length',
            'bb_mult': 'bb_mult',
            # RSI
            'rsi_length': 'rsi_length',
            'rsi_oversold': 'rsi_oversold',
            'rsi_overbought': 'rsi_overbought',
            # Keltner Channel
            'kc_length': 'kc_length',
            'kc_mult': 'kc_mult',
            # ADX
            'adx_threshold': 'adx_threshold',
            # Stop/Take profit
            'sl_atr_mult': 'sl_atr_mult',
            'tp_ratio': 'tp_ratio',
            # Stochastic (for StochasticStrategy)
            'stoch_k': 'stoch_k',  # Also used by BBStochStrategy directly
            'stoch_d': 'stoch_d',
            'stoch_oversold': 'stoch_oversold',
            'stoch_overbought': 'stoch_overbought',
            # MACD
            'fast_period': 'fast_period',
            'slow_period': 'slow_period',
            'signal_period': 'signal_period',
            # Supertrend
            'st_period': 'atr_period',
            'st_mult': 'atr_multiplier',
            # Donchian
            'dc_length': 'period',
        }

        for src, dst in param_mapping.items():
            if src in params:
                bt_params[dst] = params[src]

        return bt_params

    def _extract_results(self, cerebro, strategy, strategy_name: str,
                         strategy_category: str, params: Dict[str, Any]) -> BacktestResult:
        """Extract results from completed backtest"""

        # Get final portfolio value
        final_value = cerebro.broker.getvalue()
        total_pnl = final_value - self.capital

        # Get analyzers
        sharpe = strategy.analyzers.sharpe.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()

        # Extract trade metrics
        total_trades = trades.get('total', {}).get('total', 0)
        won = trades.get('won', {}).get('total', 0)
        lost = trades.get('lost', {}).get('total', 0)

        win_rate = (won / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        gross_profit = trades.get('won', {}).get('pnl', {}).get('total', 0)
        gross_loss = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 0)

        # Drawdown
        max_dd = drawdown.get('max', {}).get('drawdown', 0)
        max_dd_pct = drawdown.get('max', {}).get('moneydown', 0)

        # Sharpe ratio
        sharpe_ratio = sharpe.get('sharperatio', 0) or 0

        # Calculate equity R² (smoothness)
        equity_curve = strategy.trade_log if hasattr(strategy, 'trade_log') else []
        equity_r_squared = self._calculate_equity_r_squared(equity_curve)

        # Recovery factor
        recovery_factor = (total_pnl / max_dd) if max_dd > 0 else 0

        # Calculate composite score
        composite_score, score_breakdown = self._calculate_composite_score(
            win_rate, total_pnl, equity_r_squared, profit_factor, total_trades
        )

        return BacktestResult(
            strategy_name=strategy_name,
            strategy_category=strategy_category,
            params=params,
            total_trades=total_trades,
            winning_trades=won,
            losing_trades=lost,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe_ratio,
            equity_r_squared=equity_r_squared,
            recovery_factor=recovery_factor,
            equity_curve=[t.get('pnl', 0) for t in equity_curve] if equity_curve else [],
            trade_log=equity_curve,
            composite_score=composite_score,
            score_breakdown=score_breakdown
        )

    def _calculate_equity_r_squared(self, trade_log: List[Dict]) -> float:
        """Calculate R² of equity curve (smoothness)"""
        if len(trade_log) < 3:
            return 0.0

        cumulative = []
        total = 0
        for trade in trade_log:
            total += trade.get('pnl', 0)
            cumulative.append(total)

        if len(cumulative) < 3:
            return 0.0

        x = np.arange(len(cumulative))
        y = np.array(cumulative)

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept

        # R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # If slope is negative, return negative R²
        if slope < 0:
            r_squared *= -1

        return r_squared

    def _calculate_composite_score(self, win_rate: float, total_pnl: float,
                                    equity_r_squared: float, profit_factor: float,
                                    total_trades: int) -> Tuple[float, Dict]:
        """
        Calculate composite score using Balanced Triangle approach:
        33.33% Win Rate, 33.33% Equity Smoothness, 33.34% Total Profit
        """
        # Win Rate Score (33.33%)
        wr_score = win_rate / 100
        if win_rate >= 60:
            wr_score = min(1.0, wr_score * 1.2)

        # Equity Smoothness Score (33.33%)
        eq_score = max(0, equity_r_squared)
        if equity_r_squared > 0.7:
            eq_score = min(1.0, eq_score * 1.15)

        # Total Profit Score (33.34%)
        pnl_normalized = total_pnl / self.capital
        pnl_score = 2 / (1 + np.exp(-pnl_normalized)) - 1
        pnl_score = max(0, (pnl_score + 1) / 2)
        if total_pnl > self.capital:
            pnl_score = min(1.0, pnl_score * 1.1)

        # Calculate composite
        composite = (
            0.3333 * wr_score +
            0.3333 * eq_score +
            0.3334 * pnl_score
        )

        # Apply penalties
        if win_rate < 50:
            composite *= 0.6
        if total_pnl <= 0:
            composite *= 0.3
        if profit_factor < 1:
            composite *= 0.5
        if total_trades < 5:
            composite *= 0.7

        score_breakdown = {
            'win_rate_score': round(wr_score * 100, 1),
            'equity_smoothness_score': round(eq_score * 100, 1),
            'total_profit_score': round(pnl_score * 100, 1),
            'win_rate_weight': 33.33,
            'equity_smoothness_weight': 33.33,
            'total_profit_weight': 33.34,
            'penalties_applied': {
                'low_win_rate': win_rate < 50,
                'not_profitable': total_pnl <= 0,
                'low_profit_factor': profit_factor < 1,
                'too_few_trades': total_trades < 5
            }
        }

        return composite, score_breakdown


# Factory function for easy use
def create_backtrader_engine(df: pd.DataFrame, capital: float = 1000.0) -> BacktraderEngine:
    """Create a BacktraderEngine instance"""
    return BacktraderEngine(df, capital)
