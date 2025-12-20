"""
STRATEGY ENGINE
===============
Simple, powerful strategy finder that actually works.

Philosophy:
- Simple beats complex
- Save what works, build on it
- Creative entries, optimized exits
- No over-engineering

Uses pandas-ta for 130+ technical indicators.
Auto-scales based on available CPU cores.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Import database
try:
    from strategy_database import get_strategy_db
    HAS_DB = True
except ImportError:
    HAS_DB = False

# Import multi-engine calculator for engine selection
try:
    from indicator_engines import MultiEngineCalculator
    HAS_MULTI_ENGINE = True
except ImportError:
    HAS_MULTI_ENGINE = False


@dataclass
class TradeResult:
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float  # Percentage gain/loss
    exit_reason: str    # 'tp' or 'sl'
    # Enhanced fields for TradingView-style reporting
    trade_num: int = 0
    entry_time: str = None
    exit_time: str = None
    position_size: float = 0.0  # £ value
    position_qty: float = 0.0   # BTC quantity
    run_up: float = 0.0         # Max favorable excursion £
    run_up_pct: float = 0.0     # Max favorable excursion %
    drawdown: float = 0.0       # Max adverse excursion £
    drawdown_pct: float = 0.0   # Max adverse excursion %
    cumulative_pnl: float = 0.0 # Running total P&L £


# Auto-scaling configuration
def get_system_resources() -> Dict:
    """Detect system resources (CPU, memory) for auto-scaling."""
    resources = {
        'cpu_cores': os.cpu_count() or 4,
        'memory_gb': 4.0,  # Default fallback
        'memory_available_gb': 4.0,
        'container_memory_limit_gb': None,
        'is_container': False
    }

    # Try to get memory info
    try:
        # Check if running in Docker (cgroup memory limit)
        cgroup_limit_path = '/sys/fs/cgroup/memory/memory.limit_in_bytes'
        cgroup_v2_path = '/sys/fs/cgroup/memory.max'

        if os.path.exists(cgroup_limit_path):
            with open(cgroup_limit_path, 'r') as f:
                limit_bytes = int(f.read().strip())
                # If limit is very high (>100TB), it's effectively unlimited
                if limit_bytes < 100 * 1024**4:
                    resources['container_memory_limit_gb'] = limit_bytes / (1024**3)
                    resources['is_container'] = True
        elif os.path.exists(cgroup_v2_path):
            with open(cgroup_v2_path, 'r') as f:
                content = f.read().strip()
                if content != 'max':
                    limit_bytes = int(content)
                    resources['container_memory_limit_gb'] = limit_bytes / (1024**3)
                    resources['is_container'] = True

        # Get system memory from /proc/meminfo (Linux)
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        mem_kb = int(line.split()[1])
                        resources['memory_gb'] = mem_kb / (1024**2)
                    elif line.startswith('MemAvailable:'):
                        mem_kb = int(line.split()[1])
                        resources['memory_available_gb'] = mem_kb / (1024**2)
    except Exception as e:
        print(f"Warning: Could not detect memory: {e}")

    return resources


def get_optimal_workers() -> Tuple[int, Dict]:
    """
    Determine optimal number of workers based on CPU and memory.

    Rules:
    - Each worker needs ~500MB RAM for backtesting
    - Use 75% of CPU cores
    - Cap at available memory / 500MB
    - Minimum 2 workers, maximum 16
    """
    resources = get_system_resources()

    # CPU-based limit (75% of cores)
    cpu_workers = max(2, int(resources['cpu_cores'] * 0.75))

    # Memory-based limit (~500MB per worker)
    mem_per_worker_gb = 0.5

    # Use container limit if available, otherwise system memory
    if resources['container_memory_limit_gb']:
        available_mem = resources['container_memory_limit_gb']
    else:
        available_mem = resources['memory_available_gb']

    # Reserve 1GB for OS/system
    usable_mem = max(0.5, available_mem - 1.0)
    mem_workers = max(2, int(usable_mem / mem_per_worker_gb))

    # Take the minimum of CPU and memory limits
    optimal = min(cpu_workers, mem_workers)

    # Cap between 2 and 16
    optimal = max(2, min(16, optimal))

    resources['cpu_based_workers'] = cpu_workers
    resources['memory_based_workers'] = mem_workers
    resources['optimal_workers'] = optimal

    return optimal, resources


OPTIMAL_WORKERS, SYSTEM_RESOURCES = get_optimal_workers()

# Log auto-scaling decision
print(f"=== Auto-Scaling Configuration ===")
print(f"  CPU Cores: {SYSTEM_RESOURCES['cpu_cores']}")
print(f"  Memory: {SYSTEM_RESOURCES['memory_gb']:.1f} GB total, {SYSTEM_RESOURCES['memory_available_gb']:.1f} GB available")
if SYSTEM_RESOURCES['container_memory_limit_gb']:
    print(f"  Container Limit: {SYSTEM_RESOURCES['container_memory_limit_gb']:.1f} GB")
print(f"  Workers (CPU-based): {SYSTEM_RESOURCES['cpu_based_workers']}")
print(f"  Workers (Memory-based): {SYSTEM_RESOURCES['memory_based_workers']}")
print(f"  USING: {OPTIMAL_WORKERS} workers")
print(f"=================================")


@dataclass
class StrategyResult:
    """Result of a strategy backtest."""
    strategy_name: str
    strategy_category: str
    direction: str
    tp_percent: float
    sl_percent: float
    entry_rule: str

    # Metrics
    total_trades: int
    wins: int          # TP hits
    losses: int        # SL hits
    win_rate: float
    total_pnl: float
    total_pnl_percent: float  # Total percentage return
    profit_factor: float
    max_drawdown: float
    max_drawdown_percent: float
    avg_trade: float
    avg_trade_percent: float

    # Buy & Hold comparison
    buy_hold_return: float = 0.0      # Buy & hold return %
    vs_buy_hold: float = 0.0          # Strategy return - B&H return (+ = outperform)
    beats_buy_hold: bool = False      # Quick flag

    # Composite score (balanced ranking)
    composite_score: float = 0.0

    # For database
    params: Dict = None
    equity_curve: List[float] = None
    trades_list: List[Dict] = None  # For detailed analysis

    # Open position tracking (for UI warning)
    has_open_position: bool = False
    open_position: Dict = None  # {direction, entry_price, entry_time, current_price, unrealized_pnl, unrealized_pnl_pct}

    def __post_init__(self):
        if self.params is None:
            self.params = {
                'tp_percent': self.tp_percent,
                'sl_percent': self.sl_percent,
                'direction': self.direction,
                'entry_rule': self.entry_rule
            }
        if self.equity_curve is None:
            self.equity_curve = []
        if self.trades_list is None:
            self.trades_list = []

        # Calculate composite score
        self._calculate_composite_score()

    def _calculate_composite_score(self):
        """
        Calculate a balanced composite score.

        Weights:
        - Win Rate: 35% (high win rate = reliable)
        - Profit Factor: 25% (risk-adjusted returns)
        - Total Return %: 25% (actual profitability)
        - Trade Count: 15% (statistical significance)
        """
        # Win Rate Score (0-100): Target 60%+, penalize below 40%
        if self.win_rate >= 60:
            wr_score = 100
        elif self.win_rate >= 50:
            wr_score = 60 + (self.win_rate - 50) * 4
        elif self.win_rate >= 40:
            wr_score = 40 + (self.win_rate - 40) * 2
        else:
            wr_score = max(0, self.win_rate)

        # Profit Factor Score (0-100): Target PF > 1.5
        if self.profit_factor >= 2.0:
            pf_score = 100
        elif self.profit_factor >= 1.5:
            pf_score = 70 + (self.profit_factor - 1.5) * 60
        elif self.profit_factor >= 1.0:
            pf_score = 30 + (self.profit_factor - 1.0) * 80
        else:
            pf_score = max(0, self.profit_factor * 30)

        # PnL Score (0-100): Based on percentage return
        pnl_pct = self.total_pnl_percent if self.total_pnl_percent else 0
        if pnl_pct >= 50:
            pnl_score = 100
        elif pnl_pct >= 20:
            pnl_score = 60 + (pnl_pct - 20) * (40/30)
        elif pnl_pct >= 0:
            pnl_score = pnl_pct * 3
        else:
            pnl_score = 0

        # Trade Count Score (0-100): Sweet spot 20-50 trades
        if 20 <= self.total_trades <= 50:
            trades_score = 100
        elif self.total_trades >= 10:
            trades_score = 50 + min(50, (self.total_trades - 10) * 5)
        elif self.total_trades >= 5:
            trades_score = (self.total_trades - 5) * 10
        else:
            trades_score = 0

        # Weighted composite
        self.composite_score = (
            wr_score * 0.35 +
            pf_score * 0.25 +
            pnl_score * 0.25 +
            trades_score * 0.15
        )


class StrategyEngine:
    """
    The main strategy finding engine.
    Simple, effective, saves results.
    """

    # Entry strategies - simple rules that work
    ENTRY_STRATEGIES = {
        # === ALWAYS (pure TP/SL optimization) ===
        'always': {
            'name': 'Always Enter',
            'category': 'Baseline',
            'description': 'Enter on every bar - tests pure TP/SL effectiveness'
        },

        # === MOMENTUM ===
        'rsi_extreme': {
            'name': 'RSI Extreme',
            'category': 'Momentum',
            'description': 'RSI < 25 (long) or > 75 (short)'
        },
        'rsi_cross_50': {
            'name': 'RSI Cross 50',
            'category': 'Momentum',
            'description': 'RSI crosses above/below 50'
        },
        'stoch_extreme': {
            'name': 'Stochastic Extreme',
            'category': 'Momentum',
            'description': 'Stoch K < 20 (long) or > 80 (short)'
        },

        # === MEAN REVERSION ===
        'bb_touch': {
            'name': 'Bollinger Band Touch',
            'category': 'Mean Reversion',
            'description': 'Price touches lower/upper BB'
        },
        'bb_squeeze_breakout': {
            'name': 'BB Squeeze Breakout',
            'category': 'Mean Reversion',
            'description': 'BB width contracts then expands'
        },
        'price_vs_sma': {
            'name': 'Price vs SMA',
            'category': 'Mean Reversion',
            'description': 'Price 1%+ away from SMA20'
        },

        # === TREND ===
        'ema_cross': {
            'name': 'EMA 9/21 Cross',
            'category': 'Trend',
            'description': 'Fast EMA crosses slow EMA'
        },
        'sma_cross': {
            'name': 'SMA 9/18 Cross',
            'category': 'Trend',
            'description': 'TradingView MovingAvg2Line Cross - Fast SMA(9) crosses Slow SMA(18)'
        },
        'macd_cross': {
            'name': 'MACD Cross',
            'category': 'Trend',
            'description': 'MACD line crosses signal line'
        },
        'price_above_sma': {
            'name': 'Price Above/Below SMA',
            'category': 'Trend',
            'description': 'Price crosses SMA20'
        },

        # === PATTERN ===
        'consecutive_candles': {
            'name': 'Consecutive Candles',
            'category': 'Pattern',
            'description': '3 consecutive red (long) or green (short)'
        },
        'big_candle': {
            'name': 'Big Candle Reversal',
            'category': 'Pattern',
            'description': 'Large candle (2x ATR) in opposite direction'
        },
        'doji_reversal': {
            'name': 'Doji Reversal',
            'category': 'Pattern',
            'description': 'Doji candle after trend'
        },
        'engulfing': {
            'name': 'Engulfing Pattern',
            'category': 'Pattern',
            'description': 'Bullish/bearish engulfing candle'
        },
        'inside_bar': {
            'name': 'Inside Bar',
            'category': 'Pattern',
            'description': 'TradingView InSide Bar Strategy - bar range inside previous bar'
        },
        'outside_bar': {
            'name': 'Outside Bar',
            'category': 'Pattern',
            'description': 'TradingView OutSide Bar Strategy - bar range engulfs previous bar'
        },

        # === VOLATILITY ===
        'atr_breakout': {
            'name': 'ATR Breakout',
            'category': 'Volatility',
            'description': 'Price moves more than 1.5x ATR'
        },
        'low_volatility_breakout': {
            'name': 'Low Vol Breakout',
            'category': 'Volatility',
            'description': 'Breakout after low volatility period'
        },

        # === SIMPLE PRICE ACTION ===
        'higher_low': {
            'name': 'Higher Low',
            'category': 'Price Action',
            'description': 'Higher low formed (long) or lower high (short)'
        },
        'support_resistance': {
            'name': 'Support/Resistance',
            'category': 'Price Action',
            'description': 'Price at recent support/resistance level'
        },

        # === NEW STRATEGIES (pandas-ta) ===
        'williams_r': {
            'name': 'Williams %R Extreme',
            'category': 'Momentum',
            'description': 'Williams %R < -80 (long) or > -20 (short)'
        },
        'cci_extreme': {
            'name': 'CCI Extreme',
            'category': 'Momentum',
            'description': 'CCI < -100 (long) or > 100 (short)'
        },
        'supertrend': {
            'name': 'Supertrend Signal',
            'category': 'Trend',
            'description': 'Supertrend direction change'
        },
        'adx_strong_trend': {
            'name': 'ADX Strong Trend',
            'category': 'Trend',
            'description': 'ADX > 25 with DI+ or DI- dominance'
        },
        'psar_reversal': {
            'name': 'Parabolic SAR Reversal',
            'category': 'Trend',
            'description': 'Price crosses Parabolic SAR'
        },
        'vwap_bounce': {
            'name': 'VWAP Bounce',
            'category': 'Mean Reversion',
            'description': 'Price bounces off VWAP'
        },
        'rsi_divergence': {
            'name': 'RSI Divergence',
            'category': 'Momentum',
            'description': 'Price makes new low but RSI doesnt (bullish divergence)'
        },

        # === ADDITIONAL TRADINGVIEW STRATEGIES ===
        'keltner_breakout': {
            'name': 'Keltner Channel Breakout',
            'category': 'Volatility',
            'description': 'Price breaks above/below Keltner Channel'
        },
        'donchian_breakout': {
            'name': 'Donchian Channel Breakout',
            'category': 'Trend',
            'description': 'Price breaks above/below Donchian Channel (Turtle Trading)'
        },
        'ichimoku_cross': {
            'name': 'Ichimoku TK Cross',
            'category': 'Trend',
            'description': 'Tenkan-sen crosses Kijun-sen (Ichimoku Cloud)'
        },
        'ichimoku_cloud': {
            'name': 'Ichimoku Cloud Breakout',
            'category': 'Trend',
            'description': 'Price breaks above/below the Ichimoku Cloud'
        },
        'aroon_cross': {
            'name': 'Aroon Cross',
            'category': 'Trend',
            'description': 'Aroon Up crosses Aroon Down'
        },
        'momentum_zero': {
            'name': 'Momentum Zero Cross',
            'category': 'Momentum',
            'description': 'Momentum crosses above/below zero line'
        },
        'roc_extreme': {
            'name': 'Rate of Change Extreme',
            'category': 'Momentum',
            'description': 'ROC reaches extreme levels (oversold/overbought)'
        },
        'uo_extreme': {
            'name': 'Ultimate Oscillator Extreme',
            'category': 'Momentum',
            'description': 'Ultimate Oscillator < 30 (long) or > 70 (short)'
        },
        'chop_trend': {
            'name': 'Choppiness Trend',
            'category': 'Volatility',
            'description': 'Choppiness Index indicates trending market (< 38.2)'
        },
        'double_ema_cross': {
            'name': 'Double EMA Cross',
            'category': 'Trend',
            'description': 'EMA 12/26 Cross (same periods as MACD)'
        },
        'triple_ema': {
            'name': 'Triple EMA Alignment',
            'category': 'Trend',
            'description': 'EMA 9 > EMA 21 > EMA 50 alignment'
        },
    }

    def __init__(self, df: pd.DataFrame, status_callback: Dict = None,
                 streaming_callback: Callable = None,
                 capital: float = 1000.0,
                 position_size_pct: float = 75.0,
                 calc_engine: str = "tradingview",
                 progress_min: int = 0,
                 progress_max: int = 100):
        self.df = df.copy()
        self.status = status_callback or {}
        self.streaming_callback = streaming_callback
        self.db = get_strategy_db() if HAS_DB else None

        # Store trading parameters from UI
        self.capital = capital
        self.position_size_pct = position_size_pct

        # Store calculation engine for indicator calculations
        self.calc_engine = calc_engine

        # Progress range for multi-engine mode (e.g., 0-33, 33-66, 66-100)
        self.progress_min = progress_min
        self.progress_max = progress_max

        # Calculate Buy & Hold benchmark
        self.buy_hold_return = self._calculate_buy_hold()

        self._calculate_indicators()

    def _calculate_buy_hold(self) -> float:
        """
        Calculate buy & hold return for the dataset.
        This is the benchmark all strategies must beat.
        """
        df = self.df
        if len(df) < 2:
            return 0.0

        # Use same start point as backtests (bar 50)
        start_idx = min(50, len(df) - 1)
        start_price = df.iloc[start_idx]['close']
        end_price = df.iloc[-1]['close']

        buy_hold_pct = ((end_price - start_price) / start_price) * 100
        print(f"Buy & Hold benchmark: {buy_hold_pct:.2f}% (from £{start_price:.2f} to £{end_price:.2f})")
        return round(buy_hold_pct, 2)

    def _update_status(self, message: str, progress: int):
        if self.status:
            self.status['message'] = message
            # Scale progress to the assigned range (e.g., 0-100 maps to 33-66)
            scaled_progress = self.progress_min + int((progress / 100) * (self.progress_max - self.progress_min))
            self.status['progress'] = scaled_progress

    def _publish_result(self, result: StrategyResult):
        """Stream result to frontend."""
        if self.streaming_callback:
            try:
                self.streaming_callback({
                    'type': 'strategy_result',
                    'strategy_name': result.strategy_name,
                    'strategy_category': result.strategy_category,
                    'entry_rule': result.entry_rule,
                    'composite_score': result.composite_score,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'total_pnl': result.total_pnl,
                    'total_pnl_percent': result.total_pnl_percent,
                    'total_trades': result.total_trades,
                    'wins': result.wins,
                    'losses': result.losses,
                    'max_drawdown': result.max_drawdown,
                    'max_drawdown_percent': result.max_drawdown_percent,
                    'params': result.params,
                    'equity_curve': result.equity_curve[-30:] if result.equity_curve else [],  # Last 30 points
                    # Buy & Hold comparison
                    'buy_hold_return': result.buy_hold_return,
                    'vs_buy_hold': result.vs_buy_hold,
                    'beats_buy_hold': result.beats_buy_hold,
                    # Open position warning
                    'has_open_position': result.has_open_position,
                    'open_position': result.open_position,
                })
            except Exception as e:
                print(f"Streaming error: {e}")

    def _calculate_indicators(self):
        """
        Calculate indicators using the selected calculation engine.

        Engines:
        - tradingview: Uses TradingView-compatible formulas (for Pine Script export)
        - native: Uses TA-Lib for fast execution + candlestick patterns
        """
        df = self.df
        engine = self.calc_engine

        # Use MultiEngineCalculator for main indicators if available
        if HAS_MULTI_ENGINE and engine in ['tradingview', 'native']:
            calc = MultiEngineCalculator(df)

            # === MOMENTUM INDICATORS ===
            if engine == 'tradingview':
                df['rsi'] = calc.rsi_tradingview(14)
                stoch_k, stoch_d = calc.stoch_tradingview(14, 3, 3)
            else:  # native (TA-Lib)
                df['rsi'] = calc.rsi_native(14)
                stoch_k, stoch_d = calc.stoch_native(14, 3, 3)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d

            # === VOLATILITY INDICATORS ===
            if engine == 'tradingview':
                bb_mid, bb_upper, bb_lower = calc.bbands_tradingview(20, 2.0)
                df['atr'] = calc.atr_tradingview(14)
            else:  # native (TA-Lib)
                bb_mid, bb_upper, bb_lower = calc.bbands_native(20, 2.0)
                df['atr'] = calc.atr_native(14)
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['bb_mid'] = bb_mid
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

            # === TREND INDICATORS ===
            if engine == 'tradingview':
                df['sma_20'] = calc.sma_tradingview(20)
                df['sma_50'] = calc.sma_tradingview(50)
                df['ema_9'] = calc.ema_tradingview(9)
                df['ema_21'] = calc.ema_tradingview(21)
                macd_line, signal_line, histogram = calc.macd_tradingview(12, 26, 9)
            else:  # native (TA-Lib)
                df['sma_20'] = calc.sma_native(20)
                df['sma_50'] = calc.sma_native(50)
                df['ema_9'] = calc.ema_native(9)
                df['ema_21'] = calc.ema_native(21)
                macd_line, signal_line, histogram = calc.macd_native(12, 26, 9)
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_hist'] = histogram

        else:
            # Fall back to pandas_ta for all indicators
            # === MOMENTUM INDICATORS ===
            df['rsi'] = ta.rsi(df['close'], length=14)

            # Stochastic with smoothing to match TradingView
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
            stoch_k_col = [c for c in stoch.columns if c.startswith('STOCHk_')][0]
            stoch_d_col = [c for c in stoch.columns if c.startswith('STOCHd_')][0]
            df['stoch_k'] = stoch[stoch_k_col]
            df['stoch_d'] = stoch[stoch_d_col]

            # === VOLATILITY INDICATORS ===
            bb = ta.bbands(df['close'], length=20, std=2)
            bb_upper_col = [c for c in bb.columns if c.startswith('BBU_')][0]
            bb_lower_col = [c for c in bb.columns if c.startswith('BBL_')][0]
            bb_mid_col = [c for c in bb.columns if c.startswith('BBM_')][0]
            df['bb_upper'] = bb[bb_upper_col]
            df['bb_lower'] = bb[bb_lower_col]
            df['bb_mid'] = bb[bb_mid_col]
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            # === TREND INDICATORS ===
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)

            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            macd_col = [c for c in macd.columns if c.startswith('MACD_') and not c.startswith('MACDs') and not c.startswith('MACDh')][0]
            macd_signal_col = [c for c in macd.columns if c.startswith('MACDs_')][0]
            macd_hist_col = [c for c in macd.columns if c.startswith('MACDh_')][0]
            df['macd'] = macd[macd_col]
            df['macd_signal'] = macd[macd_signal_col]
            df['macd_hist'] = macd[macd_hist_col]

        # === ADDITIONAL INDICATORS ===
        # Use MultiEngineCalculator if available, otherwise pandas_ta

        if HAS_MULTI_ENGINE and engine == 'tradingview':
            # Use TradingView-specific implementations
            df['willr'] = calc.willr_tradingview(14)
            df['cci'] = calc.cci_tradingview(20)
            df['mom'] = calc.mom_tradingview(10)
            df['roc'] = calc.roc_tradingview(9)

            # ADX
            adx, di_plus, di_minus = calc.adx_tradingview(14)
            df['adx'] = adx
            df['di_plus'] = di_plus
            df['di_minus'] = di_minus

            # Aroon
            aroon_up, aroon_down, aroon_osc = calc.aroon_tradingview(14)
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down
            df['aroon_osc'] = aroon_osc

            # Supertrend (TradingView direction: 1=bearish, -1=bullish)
            supertrend, supertrend_dir = calc.supertrend_tradingview(3.0, 10)
            df['supertrend'] = supertrend
            df['supertrend_dir'] = supertrend_dir

            # Parabolic SAR
            df['psar'] = calc.psar_tradingview()

            # Keltner Channels
            kc_mid, kc_upper, kc_lower = calc.keltner_tradingview(20, 2.0, 10)
            df['kc_mid'] = kc_mid
            df['kc_upper'] = kc_upper
            df['kc_lower'] = kc_lower

            # Donchian Channels
            dc_mid, dc_upper, dc_lower = calc.donchian_tradingview(20)
            df['dc_mid'] = dc_mid
            df['dc_upper'] = dc_upper
            df['dc_lower'] = dc_lower

            # Ichimoku
            ichimoku = calc.ichimoku_tradingview(9, 26, 52)
            df['tenkan'] = ichimoku['tenkan']
            df['kijun'] = ichimoku['kijun']
            df['senkou_a'] = ichimoku['senkou_a']
            df['senkou_b'] = ichimoku['senkou_b']

            # Ultimate Oscillator
            df['uo'] = calc.uo_tradingview(7, 14, 28)

            # Choppiness Index
            df['chop'] = calc.chop_tradingview(14)

            # VWAP - only works with volume data
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['vwap'] = calc.vwap_tradingview()
            else:
                df['vwap'] = df['sma_20']  # Fallback to SMA if no volume

        elif HAS_MULTI_ENGINE and engine == 'native':
            # Use Native (TA-Lib) implementations
            df['willr'] = calc.willr_native(14)
            df['cci'] = calc.cci_native(20)
            df['mom'] = calc.mom_native(10)
            df['roc'] = calc.roc_native(9)

            # ADX
            adx, di_plus, di_minus = calc.adx_native(14)
            df['adx'] = adx
            df['di_plus'] = di_plus
            df['di_minus'] = di_minus

            # Supertrend (uses TradingView impl as no TA-Lib equivalent)
            supertrend, supertrend_dir = calc.supertrend_native(3.0, 10)
            df['supertrend'] = supertrend
            df['supertrend_dir'] = supertrend_dir

            # Native-only: Candlestick patterns
            patterns = calc.detect_all_patterns()
            for col in patterns.columns:
                df[col.lower()] = patterns[col]

            # Native-only: Hilbert Transform
            df['ht_trendmode'] = calc.hilbert_trendmode()
            df['ht_dcperiod'] = calc.hilbert_dominant_cycle()

            # Fall through to pandas_ta for remaining indicators
            # Aroon (no native equivalent)
            aroon = ta.aroon(df['high'], df['low'], length=14)
            aroon_up_col = [c for c in aroon.columns if 'AROONU' in c][0]
            aroon_down_col = [c for c in aroon.columns if 'AROOND' in c][0]
            aroon_osc_col = [c for c in aroon.columns if 'AROONOSC' in c][0]
            df['aroon_up'] = aroon[aroon_up_col]
            df['aroon_down'] = aroon[aroon_down_col]
            df['aroon_osc'] = aroon[aroon_osc_col]

            # PSAR (using pandas_ta as fallback)
            psar = ta.psar(df['high'], df['low'], df['close'])
            psar_l = [c for c in psar.columns if 'PSARl' in c]
            psar_s = [c for c in psar.columns if 'PSARs' in c]
            if psar_l and psar_s:
                df['psar'] = psar[psar_l[0]].fillna(psar[psar_s[0]])
            else:
                df['psar'] = psar.iloc[:, 0]

            # Use pandas_ta for remaining indicators (no TA-Lib equivalent)
            # Keltner Channels
            kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2.0)
            kc_upper_col = [c for c in kc.columns if 'KCU' in c][0]
            kc_mid_col = [c for c in kc.columns if 'KCB' in c][0]
            kc_lower_col = [c for c in kc.columns if 'KCL' in c][0]
            df['kc_mid'] = kc[kc_mid_col]
            df['kc_upper'] = kc[kc_upper_col]
            df['kc_lower'] = kc[kc_lower_col]

            # Donchian Channels
            dc = ta.donchian(df['high'], df['low'], lower_length=20, upper_length=20)
            dc_upper_col = [c for c in dc.columns if 'DCU' in c][0]
            dc_mid_col = [c for c in dc.columns if 'DCM' in c][0]
            dc_lower_col = [c for c in dc.columns if 'DCL' in c][0]
            df['dc_mid'] = dc[dc_mid_col]
            df['dc_upper'] = dc[dc_upper_col]
            df['dc_lower'] = dc[dc_lower_col]

            # Ichimoku
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52)
            if isinstance(ichimoku, tuple):
                lines = ichimoku[0]
                df['tenkan'] = lines.iloc[:, 0]
                df['kijun'] = lines.iloc[:, 1]
                df['senkou_a'] = lines.iloc[:, 2] if lines.shape[1] > 2 else np.nan
                df['senkou_b'] = lines.iloc[:, 3] if lines.shape[1] > 3 else np.nan

            # Ultimate Oscillator
            df['uo'] = ta.uo(df['high'], df['low'], df['close'], fast=7, medium=14, slow=28)

            # Choppiness Index
            df['chop'] = ta.chop(df['high'], df['low'], df['close'], length=14)

            # VWAP
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            else:
                df['vwap'] = df['sma_20']

        else:
            # Use pandas_ta for all additional indicators
            # Williams %R
            df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)

            # CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)

            # Momentum
            df['mom'] = ta.mom(df['close'], length=10)

            # Rate of Change
            df['roc'] = ta.roc(df['close'], length=9)

            # ADX with DI+ and DI-
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            adx_col = [c for c in adx.columns if c.startswith('ADX_')][0]
            dmp_col = [c for c in adx.columns if c.startswith('DMP_')][0]
            dmn_col = [c for c in adx.columns if c.startswith('DMN_')][0]
            df['adx'] = adx[adx_col]
            df['di_plus'] = adx[dmp_col]
            df['di_minus'] = adx[dmn_col]

            # Aroon
            aroon = ta.aroon(df['high'], df['low'], length=14)
            aroon_up_col = [c for c in aroon.columns if 'AROONU' in c][0]
            aroon_down_col = [c for c in aroon.columns if 'AROOND' in c][0]
            aroon_osc_col = [c for c in aroon.columns if 'AROONOSC' in c][0]
            df['aroon_up'] = aroon[aroon_up_col]
            df['aroon_down'] = aroon[aroon_down_col]
            df['aroon_osc'] = aroon[aroon_osc_col]

            # Supertrend
            supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
            supert_col = [c for c in supertrend.columns if c.startswith('SUPERT_') and not c.startswith('SUPERTd_') and not c.startswith('SUPERTl_') and not c.startswith('SUPERTs_')][0]
            supertd_col = [c for c in supertrend.columns if c.startswith('SUPERTd_')][0]
            df['supertrend'] = supertrend[supert_col]
            df['supertrend_dir'] = supertrend[supertd_col]  # 1 = bullish, -1 = bearish

            # Parabolic SAR
            psar = ta.psar(df['high'], df['low'], df['close'])
            psar_long_col = [c for c in psar.columns if c.startswith('PSARl_')][0]
            psar_short_col = [c for c in psar.columns if c.startswith('PSARs_')][0]
            df['psar'] = psar[psar_long_col].fillna(psar[psar_short_col])

            # Keltner Channels
            kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2.0, mamode='ema')
            kc_upper_col = [c for c in kc.columns if c.startswith('KCU')][0]
            kc_mid_col = [c for c in kc.columns if c.startswith('KCB')][0]
            kc_lower_col = [c for c in kc.columns if c.startswith('KCL')][0]
            df['kc_upper'] = kc[kc_upper_col]
            df['kc_mid'] = kc[kc_mid_col]
            df['kc_lower'] = kc[kc_lower_col]

            # Donchian Channels
            dc = ta.donchian(df['high'], df['low'], lower_length=20, upper_length=20)
            dc_upper_col = [c for c in dc.columns if 'DCU' in c][0]
            dc_mid_col = [c for c in dc.columns if 'DCM' in c][0]
            dc_lower_col = [c for c in dc.columns if 'DCL' in c][0]
            df['dc_upper'] = dc[dc_upper_col]
            df['dc_mid'] = dc[dc_mid_col]
            df['dc_lower'] = dc[dc_lower_col]

            # Ichimoku
            try:
                ichi = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52)
                if isinstance(ichi, tuple):
                    lines, spans = ichi
                    df['tenkan'] = lines.iloc[:, 0]
                    df['kijun'] = lines.iloc[:, 1]
                    if len(spans.columns) > 0:
                        df['senkou_a'] = spans.iloc[:, 0]
                    if len(spans.columns) > 1:
                        df['senkou_b'] = spans.iloc[:, 1]
            except Exception:
                df['tenkan'] = df['close'].rolling(9).mean()
                df['kijun'] = df['close'].rolling(26).mean()
                df['senkou_a'] = (df['tenkan'] + df['kijun']) / 2
                df['senkou_b'] = df['close'].rolling(52).mean()

            # Ultimate Oscillator
            df['uo'] = ta.uo(df['high'], df['low'], df['close'], fast=7, medium=14, slow=28)

            # Choppiness Index
            df['chop'] = ta.chop(df['high'], df['low'], df['close'], length=14)

            # VWAP - only works with volume data
            if 'volume' in df.columns and df['volume'].sum() > 0:
                try:
                    vwap_result = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                    if vwap_result is not None and not vwap_result.isna().all():
                        df['vwap'] = vwap_result.fillna(df['sma_20'])
                    else:
                        df['vwap'] = df['sma_20']  # Fallback to SMA
                except Exception:
                    df['vwap'] = df['sma_20']  # Fallback to SMA on error
            else:
                df['vwap'] = df['sma_20']  # Fallback to SMA if no volume

        # === CANDLE PROPERTIES ===
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['green'] = df['close'] > df['open']
        df['red'] = df['close'] < df['open']
        df['doji'] = df['body'] < df['range'] * 0.1

        # Price changes
        df['pct_change'] = df['close'].pct_change() * 100

        print(f"Indicators ready (pandas-ta). {len(df)} bars, RSI: {df['rsi'].min():.1f}-{df['rsi'].max():.1f}")

    def _get_signals(self, strategy: str, direction: str) -> pd.Series:
        """Get entry signals for a strategy. Simple and direct."""
        df = self.df

        # Helper to ensure boolean series with no NaN values
        def safe_bool(series):
            return series.fillna(False).astype(bool)

        if strategy == 'always':
            return pd.Series(True, index=df.index)

        elif strategy == 'rsi_extreme':
            # TradingView RSI Strategy: crossover/crossunder through overbought/oversold levels
            if direction == 'long':
                # Long: RSI crosses UP through oversold level (30)
                return safe_bool((df['rsi'] > 30) & (df['rsi'].shift(1) <= 30))
            else:
                # Short: RSI crosses DOWN through overbought level (70)
                return safe_bool((df['rsi'] < 70) & (df['rsi'].shift(1) >= 70))

        elif strategy == 'rsi_cross_50':
            if direction == 'long':
                return safe_bool((df['rsi'] > 50) & (df['rsi'].shift(1) <= 50))
            else:
                return safe_bool((df['rsi'] < 50) & (df['rsi'].shift(1) >= 50))

        elif strategy == 'stoch_extreme':
            # TradingView Stochastic Slow Strategy: EXACT MATCH
            # if (co and k < OverSold) - check CURRENT k value after crossover
            # if (cu and k > OverBought) - check CURRENT k value after crossunder
            if direction == 'long':
                # Long: %K crosses OVER %D AND current K < 20 (oversold)
                # co = ta.crossover(k, d) AND k < OverSold
                k_cross_d_over = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
                return safe_bool(k_cross_d_over & (df['stoch_k'] < 20))
            else:
                # Short: %K crosses UNDER %D AND current K > 80 (overbought)
                # cu = ta.crossunder(k, d) AND k > OverBought
                k_cross_d_under = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
                return safe_bool(k_cross_d_under & (df['stoch_k'] > 80))

        elif strategy == 'bb_touch':
            # TradingView Bollinger Bands Strategy: price crosses through bands
            if direction == 'long':
                # Long: Price crosses UP through lower band (was below, now above)
                return safe_bool((df['close'] > df['bb_lower']) & (df['close'].shift(1) <= df['bb_lower'].shift(1)))
            else:
                # Short: Price crosses DOWN through upper band (was above, now below)
                return safe_bool((df['close'] < df['bb_upper']) & (df['close'].shift(1) >= df['bb_upper'].shift(1)))

        elif strategy == 'bb_squeeze_breakout':
            squeeze = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8
            expanding = df['bb_width'] > df['bb_width'].shift(1)
            if direction == 'long':
                return safe_bool(squeeze.shift(1) & expanding & (df['close'] > df['bb_mid']))
            else:
                return safe_bool(squeeze.shift(1) & expanding & (df['close'] < df['bb_mid']))

        elif strategy == 'price_vs_sma':
            if direction == 'long':
                return safe_bool(df['close'] < df['sma_20'] * 0.99)
            else:
                return safe_bool(df['close'] > df['sma_20'] * 1.01)

        elif strategy == 'ema_cross':
            if direction == 'long':
                return safe_bool((df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1)))
            else:
                return safe_bool((df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1)))

        elif strategy == 'sma_cross':
            # TradingView MovingAvg2Line Cross: SMA(9) crosses SMA(18)
            # mafast = ta.sma(price, fastLength)  // 9
            # maslow = ta.sma(price, slowLength)  // 18
            # Need to calculate these SMAs if not already available
            sma_9 = df['close'].rolling(9).mean()
            sma_18 = df['close'].rolling(18).mean()
            if direction == 'long':
                return safe_bool((sma_9 > sma_18) & (sma_9.shift(1) <= sma_18.shift(1)))
            else:
                return safe_bool((sma_9 < sma_18) & (sma_9.shift(1) >= sma_18.shift(1)))

        elif strategy == 'macd_cross':
            # TradingView MACD Strategy: histogram (delta) crosses zero, NOT macd crosses signal!
            # delta = MACD - Signal (histogram)
            histogram = df['macd'] - df['macd_signal']
            if direction == 'long':
                # Long: histogram crosses OVER zero
                return safe_bool((histogram > 0) & (histogram.shift(1) <= 0))
            else:
                # Short: histogram crosses UNDER zero
                return safe_bool((histogram < 0) & (histogram.shift(1) >= 0))

        elif strategy == 'price_above_sma':
            if direction == 'long':
                return safe_bool((df['close'] > df['sma_20']) & (df['close'].shift(1) <= df['sma_20'].shift(1)))
            else:
                return safe_bool((df['close'] < df['sma_20']) & (df['close'].shift(1) >= df['sma_20'].shift(1)))

        elif strategy == 'consecutive_candles':
            # TradingView Consecutive Up/Down Strategy: EXACT MATCH
            # ups := price > price[1] ? nz(ups[1]) + 1 : 0  (counts consecutive UP closes)
            # dns := price < price[1] ? nz(dns[1]) + 1 : 0  (counts consecutive DOWN closes)
            # if (ups >= consecutiveBarsUp) - long entry after 3 UP closes
            # if (dns >= consecutiveBarsDown) - short entry after 3 DOWN closes
            # Note: This is consecutive CLOSES moving up/down, NOT green/red candles
            up_close = df['close'] > df['close'].shift(1)
            down_close = df['close'] < df['close'].shift(1)

            # Count consecutive occurrences
            ups = up_close.astype(int).groupby((~up_close).cumsum()).cumsum()
            dns = down_close.astype(int).groupby((~down_close).cumsum()).cumsum()

            if direction == 'long':
                # Long after 3+ consecutive up closes
                return safe_bool(ups >= 3)
            else:
                # Short after 3+ consecutive down closes
                return safe_bool(dns >= 3)

        elif strategy == 'big_candle':
            big = df['range'] > df['atr'] * 2
            if direction == 'long':
                return safe_bool(big & df['red'])  # Big red candle = potential long reversal
            else:
                return safe_bool(big & df['green'])  # Big green candle = potential short reversal

        elif strategy == 'doji_reversal':
            if direction == 'long':
                return safe_bool(df['doji'] & (df['close'].shift(1) < df['open'].shift(1)))  # Doji after red
            else:
                return safe_bool(df['doji'] & (df['close'].shift(1) > df['open'].shift(1)))  # Doji after green

        elif strategy == 'engulfing':
            if direction == 'long':
                return safe_bool((df['green']) & (df['red'].shift(1)) & \
                       (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)))
            else:
                return safe_bool((df['red']) & (df['green'].shift(1)) & \
                       (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1)))

        elif strategy == 'inside_bar':
            # TradingView InSide Bar Strategy: EXACT MATCH
            # if (high < high[1] and low > low[1])
            #     if (close > open) -> long
            #     if (close < open) -> short
            inside = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
            if direction == 'long':
                return safe_bool(inside & (df['close'] > df['open']))
            else:
                return safe_bool(inside & (df['close'] < df['open']))

        elif strategy == 'outside_bar':
            # TradingView OutSide Bar Strategy: EXACT MATCH
            # if (high > high[1] and low < low[1])
            #     if (close > open) -> long
            #     if (close < open) -> short
            outside = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
            if direction == 'long':
                return safe_bool(outside & (df['close'] > df['open']))
            else:
                return safe_bool(outside & (df['close'] < df['open']))

        elif strategy == 'atr_breakout':
            move = abs(df['close'] - df['close'].shift(1))
            if direction == 'long':
                return safe_bool((move > df['atr'] * 1.5) & (df['close'] > df['close'].shift(1)))
            else:
                return safe_bool((move > df['atr'] * 1.5) & (df['close'] < df['close'].shift(1)))

        elif strategy == 'low_volatility_breakout':
            low_vol = df['atr'] < df['atr'].rolling(20).mean() * 0.7
            if direction == 'long':
                return safe_bool(low_vol.shift(1) & (df['close'] > df['high'].shift(1)))
            else:
                return safe_bool(low_vol.shift(1) & (df['close'] < df['low'].shift(1)))

        elif strategy == 'higher_low':
            if direction == 'long':
                return safe_bool((df['low'] > df['low'].shift(1)) & (df['low'].shift(1) > df['low'].shift(2)))
            else:
                return safe_bool((df['high'] < df['high'].shift(1)) & (df['high'].shift(1) < df['high'].shift(2)))

        elif strategy == 'support_resistance':
            recent_low = df['low'].rolling(20).min()
            recent_high = df['high'].rolling(20).max()
            if direction == 'long':
                return safe_bool(df['close'] <= recent_low * 1.005)
            else:
                return safe_bool(df['close'] >= recent_high * 0.995)

        # === NEW STRATEGIES (pandas-ta) ===

        elif strategy == 'williams_r':
            # Williams %R: < -80 oversold (long), > -20 overbought (short)
            if direction == 'long':
                return safe_bool(df['willr'] < -80)
            else:
                return safe_bool(df['willr'] > -20)

        elif strategy == 'cci_extreme':
            # CCI: < -100 oversold (long), > 100 overbought (short)
            if direction == 'long':
                return safe_bool(df['cci'] < -100)
            else:
                return safe_bool(df['cci'] > 100)

        elif strategy == 'supertrend':
            # TradingView Supertrend Strategy: EXACT MATCH
            # [_, direction] = ta.supertrend(factor, atrPeriod)
            # if ta.change(direction) < 0  -> long entry
            # if ta.change(direction) > 0  -> short entry
            #
            # In TradingView: direction = 1 (bearish, price below), -1 (bullish, price above)
            # Change from 1 to -1: becomes bullish = long entry (change = -2 < 0)
            # Change from -1 to 1: becomes bearish = short entry (change = 2 > 0)
            #
            # In pandas_ta: SUPERTd column: 1 = bullish, -1 = bearish (OPPOSITE!)
            # So for pandas_ta: direction changes from -1 to 1 = long, from 1 to -1 = short
            direction_change = df['supertrend_dir'] - df['supertrend_dir'].shift(1)
            if direction == 'long':
                # Long when direction changes to bullish (pandas_ta: -1 to 1, change = 2)
                # Or in TradingView terms: change < 0
                # Since pandas_ta is inverted, we check change > 0 for long
                return safe_bool(direction_change > 0)
            else:
                # Short when direction changes to bearish (pandas_ta: 1 to -1, change = -2)
                # Since pandas_ta is inverted, we check change < 0 for short
                return safe_bool(direction_change < 0)

        elif strategy == 'adx_strong_trend':
            # ADX > 25 indicates strong trend, use DI+/DI- for direction
            strong_trend = df['adx'] > 25
            if direction == 'long':
                return safe_bool(strong_trend & (df['di_plus'] > df['di_minus']))
            else:
                return safe_bool(strong_trend & (df['di_minus'] > df['di_plus']))

        elif strategy == 'psar_reversal':
            # Parabolic SAR reversal
            if direction == 'long':
                # Price crosses above PSAR (was below, now above)
                return safe_bool((df['close'] > df['psar']) & (df['close'].shift(1) <= df['psar'].shift(1)))
            else:
                # Price crosses below PSAR (was above, now below)
                return safe_bool((df['close'] < df['psar']) & (df['close'].shift(1) >= df['psar'].shift(1)))

        elif strategy == 'vwap_bounce':
            # Price bounces off VWAP
            if direction == 'long':
                # Price touched below VWAP and bounced back above
                touched_below = df['low'] < df['vwap']
                closed_above = df['close'] > df['vwap']
                return safe_bool(touched_below & closed_above)
            else:
                # Price touched above VWAP and rejected
                touched_above = df['high'] > df['vwap']
                closed_below = df['close'] < df['vwap']
                return safe_bool(touched_above & closed_below)

        elif strategy == 'rsi_divergence':
            # Simple divergence detection
            # Bullish: price makes lower low but RSI makes higher low
            # Bearish: price makes higher high but RSI makes lower high
            lookback = 5
            if direction == 'long':
                price_lower_low = df['low'] < df['low'].rolling(lookback).min().shift(1)
                rsi_higher_low = df['rsi'] > df['rsi'].rolling(lookback).min().shift(1)
                return safe_bool(price_lower_low & rsi_higher_low & (df['rsi'] < 40))
            else:
                price_higher_high = df['high'] > df['high'].rolling(lookback).max().shift(1)
                rsi_lower_high = df['rsi'] < df['rsi'].rolling(lookback).max().shift(1)
                return safe_bool(price_higher_high & rsi_lower_high & (df['rsi'] > 60))

        # === ADDITIONAL TRADINGVIEW STRATEGIES ===

        elif strategy == 'keltner_breakout':
            # Keltner Channel breakout
            if direction == 'long':
                # Long: price breaks above upper Keltner band
                return safe_bool((df['close'] > df['kc_upper']) & (df['close'].shift(1) <= df['kc_upper'].shift(1)))
            else:
                # Short: price breaks below lower Keltner band
                return safe_bool((df['close'] < df['kc_lower']) & (df['close'].shift(1) >= df['kc_lower'].shift(1)))

        elif strategy == 'donchian_breakout':
            # Donchian Channel breakout (Turtle Trading style)
            if direction == 'long':
                # Long: price breaks above upper Donchian channel
                return safe_bool((df['close'] > df['dc_upper'].shift(1)))
            else:
                # Short: price breaks below lower Donchian channel
                return safe_bool((df['close'] < df['dc_lower'].shift(1)))

        elif strategy == 'ichimoku_cross':
            # Ichimoku Tenkan-Kijun cross
            if direction == 'long':
                # Long: Tenkan crosses above Kijun
                return safe_bool((df['tenkan'] > df['kijun']) & (df['tenkan'].shift(1) <= df['kijun'].shift(1)))
            else:
                # Short: Tenkan crosses below Kijun
                return safe_bool((df['tenkan'] < df['kijun']) & (df['tenkan'].shift(1) >= df['kijun'].shift(1)))

        elif strategy == 'ichimoku_cloud':
            # Ichimoku Cloud breakout
            cloud_top = df[['senkou_a', 'senkou_b']].max(axis=1)
            cloud_bottom = df[['senkou_a', 'senkou_b']].min(axis=1)
            if direction == 'long':
                # Long: price breaks above the cloud
                return safe_bool((df['close'] > cloud_top) & (df['close'].shift(1) <= cloud_top.shift(1)))
            else:
                # Short: price breaks below the cloud
                return safe_bool((df['close'] < cloud_bottom) & (df['close'].shift(1) >= cloud_bottom.shift(1)))

        elif strategy == 'aroon_cross':
            # Aroon oscillator cross
            if direction == 'long':
                # Long: Aroon Up crosses above Aroon Down
                return safe_bool((df['aroon_up'] > df['aroon_down']) & (df['aroon_up'].shift(1) <= df['aroon_down'].shift(1)))
            else:
                # Short: Aroon Down crosses above Aroon Up
                return safe_bool((df['aroon_down'] > df['aroon_up']) & (df['aroon_down'].shift(1) <= df['aroon_up'].shift(1)))

        elif strategy == 'momentum_zero':
            # Momentum crosses zero
            if direction == 'long':
                return safe_bool((df['mom'] > 0) & (df['mom'].shift(1) <= 0))
            else:
                return safe_bool((df['mom'] < 0) & (df['mom'].shift(1) >= 0))

        elif strategy == 'roc_extreme':
            # Rate of Change extreme values
            if direction == 'long':
                # Long: ROC below -5% (oversold)
                return safe_bool(df['roc'] < -5)
            else:
                # Short: ROC above 5% (overbought)
                return safe_bool(df['roc'] > 5)

        elif strategy == 'uo_extreme':
            # Ultimate Oscillator extreme values
            if direction == 'long':
                # Long: UO below 30 (oversold)
                return safe_bool(df['uo'] < 30)
            else:
                # Short: UO above 70 (overbought)
                return safe_bool(df['uo'] > 70)

        elif strategy == 'chop_trend':
            # Choppiness Index indicates trending market
            # Low choppiness (< 38.2) = trending, high (> 61.8) = ranging
            is_trending = df['chop'] < 38.2
            if direction == 'long':
                # Long: trending market with price above SMA
                return safe_bool(is_trending & (df['close'] > df['sma_20']))
            else:
                # Short: trending market with price below SMA
                return safe_bool(is_trending & (df['close'] < df['sma_20']))

        elif strategy == 'double_ema_cross':
            # EMA 12/26 cross (same as MACD periods)
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            if direction == 'long':
                return safe_bool((ema_12 > ema_26) & (ema_12.shift(1) <= ema_26.shift(1)))
            else:
                return safe_bool((ema_12 < ema_26) & (ema_12.shift(1) >= ema_26.shift(1)))

        elif strategy == 'triple_ema':
            # Triple EMA alignment (9 > 21 > 50 for long, reverse for short)
            ema_50 = df['close'].ewm(span=50, adjust=False).mean()
            if direction == 'long':
                # All EMAs aligned bullishly and just crossed into alignment
                aligned = (df['ema_9'] > df['ema_21']) & (df['ema_21'] > ema_50)
                was_not_aligned = ~((df['ema_9'].shift(1) > df['ema_21'].shift(1)) & (df['ema_21'].shift(1) > ema_50.shift(1)))
                return safe_bool(aligned & was_not_aligned)
            else:
                # All EMAs aligned bearishly
                aligned = (df['ema_9'] < df['ema_21']) & (df['ema_21'] < ema_50)
                was_not_aligned = ~((df['ema_9'].shift(1) < df['ema_21'].shift(1)) & (df['ema_21'].shift(1) < ema_50.shift(1)))
                return safe_bool(aligned & was_not_aligned)

        return pd.Series(False, index=df.index)

    def backtest(self, strategy: str, direction: str,
                 tp_percent: float, sl_percent: float,
                 initial_capital: float = 1000.0,
                 position_size_pct: float = 75.0,
                 commission_pct: float = 0.1) -> StrategyResult:
        """
        Run backtest with PROPER position sizing to match TradingView.

        IMPORTANT: This now uses percentage-based position sizing with compounding,
        exactly like TradingView's "% of equity" setting.

        Args:
            strategy: Entry strategy name
            direction: 'long' or 'short'
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage
            initial_capital: Starting capital (default £1000)
            position_size_pct: Position size as % of equity (default 75%)
            commission_pct: Commission per trade (default 0.1%)
        """
        df = self.df
        signals = self._get_signals(strategy, direction)

        trades = []
        position = None
        equity = initial_capital
        equity_curve = [initial_capital]
        cumulative_pnl = 0.0
        trade_num = 0

        for i in range(50, len(df)):
            row = df.iloc[i]

            # Track run-up/drawdown while in position
            if position is not None:
                entry = position['entry_price']
                pos_size = position['position_size']  # £ value of position

                # Calculate current unrealized P&L for run-up/drawdown tracking
                if position['direction'] == 'long':
                    # For longs: high is best, low is worst
                    best_price = max(position.get('best_price', entry), row['high'])
                    worst_price = min(position.get('worst_price', entry), row['low'])
                    position['best_price'] = best_price
                    position['worst_price'] = worst_price
                else:
                    # For shorts: low is best (price going down), high is worst
                    best_price = min(position.get('best_price', entry), row['low'])
                    worst_price = max(position.get('worst_price', entry), row['high'])
                    position['best_price'] = best_price
                    position['worst_price'] = worst_price

            # Check exits
            if position is not None:
                entry = position['entry_price']
                pos_size = position['position_size']  # £ value of position
                entry_time = position['entry_time']
                entry_bar = position['entry_bar']
                pos_qty = position['position_qty']

                if position['direction'] == 'long':
                    tp_price = entry * (1 + tp_percent / 100)
                    sl_price = entry * (1 - sl_percent / 100)

                    # Calculate run-up and drawdown
                    run_up_pct = ((position['best_price'] - entry) / entry) * 100
                    run_up = pos_size * (run_up_pct / 100)
                    dd_pct = ((entry - position['worst_price']) / entry) * 100
                    dd = pos_size * (dd_pct / 100)

                    # Check SL first (more conservative - assume SL hit on same bar as TP)
                    if row['low'] <= sl_price:
                        loss_pct = -sl_percent
                        pnl = pos_size * (loss_pct / 100)
                        pnl -= pos_size * (commission_pct / 100) * 2  # Entry + Exit commission (TradingView applies per side)
                        equity += pnl
                        cumulative_pnl += pnl
                        trade_num += 1
                        exit_time = str(row['time']) if 'time' in row else f"bar_{i}"
                        trades.append(TradeResult(
                            'long', entry, sl_price, pnl, loss_pct, 'sl',
                            trade_num=trade_num, entry_time=entry_time, exit_time=exit_time,
                            position_size=pos_size, position_qty=pos_qty,
                            run_up=run_up, run_up_pct=run_up_pct,
                            drawdown=dd, drawdown_pct=dd_pct,
                            cumulative_pnl=cumulative_pnl
                        ))
                        equity_curve.append(equity)
                        position = None
                    elif row['high'] >= tp_price:
                        gain_pct = tp_percent
                        pnl = pos_size * (gain_pct / 100)
                        pnl -= pos_size * (commission_pct / 100) * 2  # Entry + Exit commission (TradingView applies per side)
                        equity += pnl
                        cumulative_pnl += pnl
                        trade_num += 1
                        exit_time = str(row['time']) if 'time' in row else f"bar_{i}"
                        trades.append(TradeResult(
                            'long', entry, tp_price, pnl, gain_pct, 'tp',
                            trade_num=trade_num, entry_time=entry_time, exit_time=exit_time,
                            position_size=pos_size, position_qty=pos_qty,
                            run_up=run_up, run_up_pct=run_up_pct,
                            drawdown=dd, drawdown_pct=dd_pct,
                            cumulative_pnl=cumulative_pnl
                        ))
                        equity_curve.append(equity)
                        position = None

                else:  # Short
                    tp_price = entry * (1 - tp_percent / 100)
                    sl_price = entry * (1 + sl_percent / 100)

                    # Calculate run-up and drawdown for shorts (inverted)
                    run_up_pct = ((entry - position['best_price']) / entry) * 100
                    run_up = pos_size * (run_up_pct / 100)
                    dd_pct = ((position['worst_price'] - entry) / entry) * 100
                    dd = pos_size * (dd_pct / 100)

                    if row['high'] >= sl_price:
                        loss_pct = -sl_percent
                        pnl = pos_size * (loss_pct / 100)
                        pnl -= pos_size * (commission_pct / 100) * 2  # Entry + Exit commission (TradingView applies per side)
                        equity += pnl
                        cumulative_pnl += pnl
                        trade_num += 1
                        exit_time = str(row['time']) if 'time' in row else f"bar_{i}"
                        trades.append(TradeResult(
                            'short', entry, sl_price, pnl, loss_pct, 'sl',
                            trade_num=trade_num, entry_time=entry_time, exit_time=exit_time,
                            position_size=pos_size, position_qty=pos_qty,
                            run_up=run_up, run_up_pct=run_up_pct,
                            drawdown=dd, drawdown_pct=dd_pct,
                            cumulative_pnl=cumulative_pnl
                        ))
                        equity_curve.append(equity)
                        position = None
                    elif row['low'] <= tp_price:
                        gain_pct = tp_percent
                        pnl = pos_size * (gain_pct / 100)
                        pnl -= pos_size * (commission_pct / 100) * 2  # Entry + Exit commission (TradingView applies per side)
                        equity += pnl
                        cumulative_pnl += pnl
                        trade_num += 1
                        exit_time = str(row['time']) if 'time' in row else f"bar_{i}"
                        trades.append(TradeResult(
                            'short', entry, tp_price, pnl, gain_pct, 'tp',
                            trade_num=trade_num, entry_time=entry_time, exit_time=exit_time,
                            position_size=pos_size, position_qty=pos_qty,
                            run_up=run_up, run_up_pct=run_up_pct,
                            drawdown=dd, drawdown_pct=dd_pct,
                            cumulative_pnl=cumulative_pnl
                        ))
                        equity_curve.append(equity)
                        position = None

            # Check entries (only if no position and equity > 0)
            if position is None and signals.iloc[i] and equity > 0:
                # Position size = % of current equity (compounding)
                pos_size = equity * (position_size_pct / 100)
                entry_price = row['close']
                pos_qty = pos_size / entry_price  # BTC quantity
                entry_time = str(row['time']) if 'time' in row else f"bar_{i}"
                position = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'position_size': pos_size,
                    'position_qty': pos_qty,
                    'entry_bar': i,
                    'entry_time': entry_time,
                    'best_price': entry_price,
                    'worst_price': entry_price
                }

        # Calculate metrics
        if not trades:
            return StrategyResult(
                strategy_name=f"{strategy}_{direction}",
                strategy_category=self.ENTRY_STRATEGIES.get(strategy, {}).get('category', 'Unknown'),
                direction=direction,
                tp_percent=tp_percent,
                sl_percent=sl_percent,
                entry_rule=strategy,
                total_trades=0, wins=0, losses=0,
                win_rate=0, total_pnl=0, total_pnl_percent=0,
                profit_factor=0, max_drawdown=0, max_drawdown_percent=0,
                avg_trade=0, avg_trade_percent=0,
                buy_hold_return=self.buy_hold_return,
                vs_buy_hold=-self.buy_hold_return,
                beats_buy_hold=False
            )

        wins = [t for t in trades if t.exit_reason == 'tp']
        losses = [t for t in trades if t.exit_reason == 'sl']
        total_pnl = sum(t.pnl for t in trades)
        total_pnl_percent = ((equity - initial_capital) / initial_capital) * 100

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.001
        pf = gross_profit / gross_loss if gross_loss > 0.001 else (10 if gross_profit > 0 else 0)

        # Max drawdown calculation
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = peak - equity_arr
        max_dd = drawdown.max()
        max_dd_pct = (max_dd / peak[np.argmax(drawdown)]) * 100 if peak[np.argmax(drawdown)] > 0 else 0

        # Build trades list for detailed analysis (TradingView-style)
        trades_list = [{
            'trade_num': t.trade_num,
            'direction': t.direction,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry': round(t.entry_price, 2),
            'exit': round(t.exit_price, 2),
            'position_size': round(t.position_size, 2),
            'position_qty': round(t.position_qty, 5),
            'pnl': round(t.pnl, 2),
            'pnl_pct': round(t.pnl_percent, 2),
            'run_up': round(t.run_up, 2),
            'run_up_pct': round(t.run_up_pct, 2),
            'drawdown': round(t.drawdown, 2),
            'drawdown_pct': round(t.drawdown_pct, 2),
            'cumulative_pnl': round(t.cumulative_pnl, 2),
            'result': 'WIN' if t.exit_reason == 'tp' else 'LOSS'
        } for t in trades]

        # Track open position at end of backtest period (for UI warning)
        open_position_data = None
        has_open = False
        if position is not None:
            has_open = True
            last_row = self.df.iloc[-1]
            current_price = last_row['close']
            entry_price = position['entry_price']
            pos_size = position['position_size']

            # Calculate unrealized P&L
            if position['direction'] == 'long':
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # short
                unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100

            unrealized_pnl = pos_size * (unrealized_pnl_pct / 100)

            open_position_data = {
                'direction': position['direction'],
                'entry_price': round(entry_price, 2),
                'entry_time': position['entry_time'],
                'current_price': round(current_price, 2),
                'position_size': round(pos_size, 2),
                'unrealized_pnl': round(unrealized_pnl, 2),
                'unrealized_pnl_pct': round(unrealized_pnl_pct, 2)
            }

        # Calculate vs Buy & Hold
        vs_bh = round(total_pnl_percent - self.buy_hold_return, 2)

        return StrategyResult(
            strategy_name=f"{strategy}_{direction}",
            strategy_category=self.ENTRY_STRATEGIES.get(strategy, {}).get('category', 'Unknown'),
            direction=direction,
            tp_percent=tp_percent,
            sl_percent=sl_percent,
            entry_rule=strategy,
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(trades) * 100 if trades else 0,
            total_pnl=round(total_pnl, 2),
            total_pnl_percent=round(total_pnl_percent, 2),
            profit_factor=round(pf, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_percent=round(max_dd_pct, 2),
            avg_trade=round(total_pnl / len(trades), 2) if trades else 0,
            avg_trade_percent=round(total_pnl_percent / len(trades), 2) if trades else 0,
            buy_hold_return=float(self.buy_hold_return),
            vs_buy_hold=float(vs_bh),
            beats_buy_hold=bool(total_pnl_percent > self.buy_hold_return),  # Convert to Python bool
            equity_curve=equity_curve,
            trades_list=trades_list,
            has_open_position=has_open,
            open_position=open_position_data
        )

    def find_strategies(self, min_trades: int = 3,
                        min_win_rate: float = 0,
                        save_to_db: bool = True,
                        symbol: str = None,
                        timeframe: str = None,
                        n_trials: int = 300) -> List[StrategyResult]:
        """
        Find all profitable strategies.
        Saves winners to database for future reference.

        n_trials controls granularity of TP/SL testing:
        - 100: 1.0% increments (fast)
        - 225: 0.67% increments
        - 400: 0.5% increments (thorough)
        - 625: 0.4% increments
        - 10000: 0.1% increments (exhaustive)
        """
        strategies = list(self.ENTRY_STRATEGIES.keys())
        directions = ['long', 'short']

        # TP/SL range: 0.1% to 10%
        # Granularity based on n_trials: more trials = finer increments
        # n_trials roughly equals TP_steps × SL_steps per strategy/direction
        steps = max(10, int(n_trials ** 0.5))  # sqrt of trials
        increment = 10.0 / steps  # 10% range divided by steps

        # Generate TP and SL ranges from 0.1% to 10%
        tp_range = [round(0.1 + i * increment, 2) for i in range(steps) if 0.1 + i * increment <= 10.0]
        sl_range = [round(0.1 + i * increment, 2) for i in range(steps) if 0.1 + i * increment <= 10.0]

        # Ensure we have at least some values
        if not tp_range:
            tp_range = [0.5, 1.0, 2.0, 3.0, 5.0]
        if not sl_range:
            sl_range = [1.0, 2.0, 3.0, 5.0, 7.0]

        results = []
        num_strategies = len(strategies)
        num_directions = len(directions)
        num_tp = len(tp_range)
        num_sl = len(sl_range)
        total = num_strategies * num_directions * num_tp * num_sl
        tested = 0
        profitable_count = 0

        # Progress phases:
        # 0-2%: Initialization (already done)
        # 2-90%: Testing combinations (main work)
        # 90-95%: Sorting/filtering
        # 95-100%: Saving to DB

        self._update_status(f"Testing {total:,} combinations ({num_strategies} strategies × 2 directions × {num_tp}×{num_sl} TP/SL @ {increment:.2f}% steps)...", 2)

        # Start database run if available
        db_run_id = None
        if self.db and save_to_db:
            db_run_id = self.db.start_optimization_run(
                symbol=symbol,
                timeframe=timeframe,
                data_rows=len(self.df)
            )

        # Calculate update frequency - update at least every 1% of progress or every 25 tests
        update_interval = max(1, min(25, total // 100))

        for strat_idx, strategy in enumerate(strategies):
            for dir_idx, direction in enumerate(directions):
                # Update at start of each strategy/direction combination
                combo_num = strat_idx * num_directions + dir_idx + 1
                combo_total = num_strategies * num_directions

                for tp in tp_range:
                    for sl in sl_range:
                        result = self.backtest(strategy, direction, tp, sl,
                                               initial_capital=self.capital,
                                               position_size_pct=self.position_size_pct)
                        tested += 1

                        # Update progress more frequently
                        if tested % update_interval == 0 or tested == total:
                            # Progress from 2% to 90% during testing phase
                            progress = int(2 + (tested / total) * 88)
                            pct_complete = (tested / total) * 100
                            self._update_status(
                                f"[{combo_num}/{combo_total}] {strategy} {direction.upper()} | {tested:,}/{total:,} ({pct_complete:.1f}%) | Found: {profitable_count}",
                                progress
                            )

                        if result.total_trades >= 1 and result.win_rate >= min_win_rate:
                            results.append(result)

                            # Stream profitable ones
                            if result.total_pnl > 0:
                                profitable_count += 1
                                self._publish_result(result)

        # Phase: Sorting results (90-95%)
        self._update_status(f"Sorting {len(results):,} results by composite score...", 90)

        # Sort by COMPOSITE SCORE (not just PnL)
        # This ensures high win rate + good PF strategies rank higher
        results.sort(key=lambda x: x.composite_score, reverse=True)

        self._update_status(f"Filtering profitable strategies...", 92)

        # Save profitable strategies to database
        profitable = [r for r in results if r.total_pnl > 0]

        # Phase: Saving to DB (95-100%)
        if self.db and save_to_db and profitable:
            self._update_status(f"Saving top {min(50, len(profitable))} strategies to database...", 95)

            for i, result in enumerate(profitable[:50]):  # Save top 50
                self.db.save_strategy(
                    result,
                    run_id=db_run_id,
                    symbol=symbol,
                    timeframe=timeframe
                )
                # Update progress during save
                if i % 10 == 0:
                    save_progress = 95 + int((i / min(50, len(profitable))) * 4)
                    self._update_status(f"Saving strategies... {i+1}/{min(50, len(profitable))}", save_progress)

            self.db.complete_optimization_run(
                db_run_id,
                strategies_tested=tested,
                profitable_found=len(profitable)
            )

            print(f"Saved {min(50, len(profitable))} strategies to database")

        self._update_status(
            f"Complete! Tested {tested:,} | Found {len(profitable)} profitable strategies",
            100
        )

        return results

    def get_saved_winners(self, symbol: str = None,
                          min_win_rate: float = 60,
                          limit: int = 20) -> List[Dict]:
        """Load previous winning strategies from database."""
        if not self.db:
            return []

        return self.db.get_top_strategies(
            limit=limit,
            symbol=symbol,
            min_win_rate=min_win_rate
        )

    def optimize_from_winners(self, winners: List[Dict],
                              fine_tune_range: float = 0.5) -> List[StrategyResult]:
        """
        Take previous winners and fine-tune them on current data.
        This builds on past success.
        """
        results = []

        for winner in winners:
            base_tp = winner.get('tp_percent', 1.0)
            base_sl = winner.get('sl_percent', 2.0)
            direction = winner.get('params', {}).get('direction', 'long')
            entry_rule = winner.get('params', {}).get('entry_rule', 'always')

            # Test variations around the winning parameters
            for tp_adj in [-fine_tune_range, 0, fine_tune_range]:
                for sl_adj in [-fine_tune_range, 0, fine_tune_range]:
                    tp = max(0.1, base_tp + tp_adj)
                    sl = max(0.1, base_sl + sl_adj)

                    result = self.backtest(entry_rule, direction, tp, sl)
                    if result.total_trades >= 3:
                        results.append(result)

        results.sort(key=lambda x: x.total_pnl, reverse=True)
        return results


def generate_pinescript(result: StrategyResult) -> str:
    """
    Generate EXACT-MATCH Pine Script with proper entry logic.

    CRITICAL: This generates the actual indicator calculations and entry conditions
    to match what Python tests.
    """
    is_long = result.direction == 'long'
    entry_rule = result.entry_rule

    # Generate entry condition code based on strategy type
    entry_conditions = {
        'always': 'entrySignal = true  // Enter on every bar',

        'rsi_extreme': f'''// RSI Strategy (TradingView built-in pattern)
// Long: RSI crosses OVER oversold (30), Short: RSI crosses UNDER overbought (70)
rsiValue = ta.rsi(close, 14)
entrySignal = {"ta.crossover(rsiValue, 30)" if is_long else "ta.crossunder(rsiValue, 70)"}''',

        'rsi_cross_50': f'''// RSI Cross 50 Entry
rsiValue = ta.rsi(close, 14)
entrySignal = {"ta.crossover(rsiValue, 50)" if is_long else "ta.crossunder(rsiValue, 50)"}''',

        'stoch_extreme': f'''// Stochastic Slow Strategy (TradingView built-in pattern)
// Long: K crosses OVER D while K < 20, Short: K crosses UNDER D while K > 80
k = ta.sma(ta.stoch(close, high, low, 14), 3)
d = ta.sma(k, 3)
entrySignal = {"ta.crossover(k, d) and k < 20" if is_long else "ta.crossunder(k, d) and k > 80"}''',

        'bb_touch': f'''// Bollinger Bands Strategy (TradingView built-in pattern)
// Long: price crosses OVER lower band, Short: price crosses UNDER upper band
[bbMid, bbUpper, bbLower] = ta.bb(close, 20, 2)
entrySignal = {"ta.crossover(close, bbLower)" if is_long else "ta.crossunder(close, bbUpper)"}''',

        'bb_squeeze_breakout': f'''// Bollinger Band Squeeze Breakout Entry
[bbMid, bbUpper, bbLower] = ta.bb(close, 20, 2)
bbWidth = (bbUpper - bbLower) / bbMid
avgWidth = ta.sma(bbWidth, 20)
squeeze = bbWidth < avgWidth * 0.8
expanding = bbWidth > bbWidth[1]
entrySignal = squeeze[1] and expanding and {"close > bbMid" if is_long else "close < bbMid"}''',

        'price_vs_sma': f'''// Price vs SMA Entry
sma20 = ta.sma(close, 20)
entrySignal = {"close < sma20 * 0.99" if is_long else "close > sma20 * 1.01"}''',

        'ema_cross': f'''// EMA 9/21 Cross Entry
ema9 = ta.ema(close, 9)
ema21 = ta.ema(close, 21)
entrySignal = {"ta.crossover(ema9, ema21)" if is_long else "ta.crossunder(ema9, ema21)"}''',

        'macd_cross': f'''// MACD Strategy (TradingView built-in pattern)
// Long: histogram crosses OVER zero, Short: histogram crosses UNDER zero
[macdLine, signalLine, histLine] = ta.macd(close, 12, 26, 9)
delta = macdLine - signalLine  // histogram
entrySignal = {"ta.crossover(delta, 0)" if is_long else "ta.crossunder(delta, 0)"}''',

        'price_above_sma': f'''// Price Above/Below SMA Entry
sma20 = ta.sma(close, 20)
entrySignal = {"ta.crossover(close, sma20)" if is_long else "ta.crossunder(close, sma20)"}''',

        'consecutive_candles': f'''// Consecutive Up/Down Strategy (TradingView built-in pattern)
// Counts consecutive UP or DOWN closes (close > close[1]), NOT green/red candles
var ups = 0.0
var dns = 0.0
ups := close > close[1] ? nz(ups[1]) + 1 : 0
dns := close < close[1] ? nz(dns[1]) + 1 : 0
entrySignal = {"ups >= 3" if is_long else "dns >= 3"}''',

        'big_candle': f'''// Big Candle Reversal Entry
atrValue = ta.atr(14)
candleRange = high - low
isGreen = close > open
isRed = close < open
bigCandle = candleRange > atrValue * 2
entrySignal = bigCandle and {"isRed" if is_long else "isGreen"}''',

        'doji_reversal': f'''// Doji Reversal Entry
body = math.abs(close - open)
candleRange = high - low
isDoji = body < candleRange * 0.1
wasRed = close[1] < open[1]
wasGreen = close[1] > open[1]
entrySignal = isDoji and {"wasRed" if is_long else "wasGreen"}''',

        'engulfing': f'''// Engulfing Pattern Entry
isGreen = close > open
isRed = close < open
bullishEngulfing = isGreen and isRed[1] and close > open[1] and open < close[1]
bearishEngulfing = isRed and isGreen[1] and close < open[1] and open > close[1]
entrySignal = {"bullishEngulfing" if is_long else "bearishEngulfing"}''',

        'inside_bar': f'''// InSide Bar Strategy (TradingView built-in pattern)
// if (high < high[1] and low > low[1]) - bar range inside previous bar
// if (close > open) -> long, if (close < open) -> short
insideBar = high < high[1] and low > low[1]
isGreen = close > open
isRed = close < open
entrySignal = insideBar and {"isGreen" if is_long else "isRed"}''',

        'outside_bar': f'''// OutSide Bar Strategy (TradingView built-in pattern)
// if (high > high[1] and low < low[1]) - bar range engulfs previous bar
// if (close > open) -> long, if (close < open) -> short
outsideBar = high > high[1] and low < low[1]
isGreen = close > open
isRed = close < open
entrySignal = outsideBar and {"isGreen" if is_long else "isRed"}''',

        'sma_cross': f'''// MovingAvg2Line Cross (TradingView built-in pattern)
// Fast SMA(9) crosses Slow SMA(18)
mafast = ta.sma(close, 9)
maslow = ta.sma(close, 18)
entrySignal = {"ta.crossover(mafast, maslow)" if is_long else "ta.crossunder(mafast, maslow)"}''',

        'atr_breakout': f'''// ATR Breakout Entry
atrValue = ta.atr(14)
priceMove = math.abs(close - close[1])
largeMove = priceMove > atrValue * 1.5
entrySignal = largeMove and {"close > close[1]" if is_long else "close < close[1]"}''',

        'low_volatility_breakout': f'''// Low Volatility Breakout Entry
atrValue = ta.atr(14)
avgAtr = ta.sma(atrValue, 20)
lowVol = atrValue < avgAtr * 0.7
entrySignal = lowVol[1] and {"close > high[1]" if is_long else "close < low[1]"}''',

        'higher_low': f'''// Higher Low/Lower High Entry
higherLow = low > low[1] and low[1] > low[2]
lowerHigh = high < high[1] and high[1] < high[2]
entrySignal = {"higherLow" if is_long else "lowerHigh"}''',

        'support_resistance': f'''// Support/Resistance Entry
recentLow = ta.lowest(low, 20)
recentHigh = ta.highest(high, 20)
entrySignal = {"close <= recentLow * 1.005" if is_long else "close >= recentHigh * 0.995"}''',

        # === NEW STRATEGIES ===
        'williams_r': f'''// Williams %R Extreme Entry
willrValue = ta.wpr(14)
entrySignal = {"willrValue < -80" if is_long else "willrValue > -20"}''',

        'cci_extreme': f'''// CCI Extreme Entry
cciValue = ta.cci(high, low, close, 20)
entrySignal = {"cciValue < -100" if is_long else "cciValue > 100"}''',

        'supertrend': f'''// Supertrend Strategy (TradingView built-in pattern)
// if ta.change(direction) < 0 -> long, if ta.change(direction) > 0 -> short
[supertrendValue, supertrendDir] = ta.supertrend(3, 10)
dirChange = ta.change(supertrendDir)
entrySignal = {"dirChange < 0" if is_long else "dirChange > 0"}''',

        'adx_strong_trend': f'''// ADX Strong Trend Entry
[diPlus, diMinus, adxValue] = ta.dmi(14, 14)
strongTrend = adxValue > 25
entrySignal = strongTrend and {"diPlus > diMinus" if is_long else "diMinus > diPlus"}''',

        'psar_reversal': f'''// Parabolic SAR Reversal Entry
psarValue = ta.sar(0.02, 0.02, 0.2)
entrySignal = {"close > psarValue and close[1] <= psarValue[1]" if is_long else "close < psarValue and close[1] >= psarValue[1]"}''',

        'vwap_bounce': f'''// VWAP Bounce Entry
vwapValue = ta.vwap(hlc3)
touchedBelow = low < vwapValue
touchedAbove = high > vwapValue
closedAbove = close > vwapValue
closedBelow = close < vwapValue
entrySignal = {"touchedBelow and closedAbove" if is_long else "touchedAbove and closedBelow"}''',

        'rsi_divergence': f'''// RSI Divergence Entry (simplified)
rsiValue = ta.rsi(close, 14)
lookback = 5
priceLowerLow = low < ta.lowest(low, lookback)[1]
rsiHigherLow = rsiValue > ta.lowest(rsiValue, lookback)[1]
priceHigherHigh = high > ta.highest(high, lookback)[1]
rsiLowerHigh = rsiValue < ta.highest(rsiValue, lookback)[1]
entrySignal = {"priceLowerLow and rsiHigherLow and rsiValue < 40" if is_long else "priceHigherHigh and rsiLowerHigh and rsiValue > 60"}''',

        # === ADDITIONAL TRADINGVIEW STRATEGIES ===
        'keltner_breakout': f'''// Keltner Channel Breakout Entry
[kcMid, kcUpper, kcLower] = ta.kc(close, 20, 2)
entrySignal = {"ta.crossover(close, kcUpper)" if is_long else "ta.crossunder(close, kcLower)"}''',

        'donchian_breakout': f'''// Donchian Channel Breakout Entry (Turtle Trading)
dcUpper = ta.highest(high, 20)
dcLower = ta.lowest(low, 20)
entrySignal = {"close > dcUpper[1]" if is_long else "close < dcLower[1]"}''',

        'ichimoku_cross': f'''// Ichimoku TK Cross Entry
donchian(len) => math.avg(ta.lowest(len), ta.highest(len))
tenkan = donchian(9)
kijun = donchian(26)
entrySignal = {"ta.crossover(tenkan, kijun)" if is_long else "ta.crossunder(tenkan, kijun)"}''',

        'ichimoku_cloud': f'''// Ichimoku Cloud Breakout Entry
donchian(len) => math.avg(ta.lowest(len), ta.highest(len))
tenkan = donchian(9)
kijun = donchian(26)
senkou_a = math.avg(tenkan, kijun)
senkou_b = donchian(52)
cloudTop = math.max(senkou_a, senkou_b)
cloudBottom = math.min(senkou_a, senkou_b)
entrySignal = {"ta.crossover(close, cloudTop)" if is_long else "ta.crossunder(close, cloudBottom)"}''',

        'aroon_cross': f'''// Aroon Cross Entry
[aroonUp, aroonDown] = ta.aroon(14)
entrySignal = {"ta.crossover(aroonUp, aroonDown)" if is_long else "ta.crossover(aroonDown, aroonUp)"}''',

        'momentum_zero': f'''// Momentum Zero Cross Entry
momValue = ta.mom(close, 10)
entrySignal = {"ta.crossover(momValue, 0)" if is_long else "ta.crossunder(momValue, 0)"}''',

        'roc_extreme': f'''// Rate of Change Extreme Entry
rocValue = ta.roc(close, 9)
entrySignal = {"rocValue < -5" if is_long else "rocValue > 5"}''',

        'uo_extreme': f'''// Ultimate Oscillator Extreme Entry
uoValue = ta.uo(7, 14, 28)
entrySignal = {"uoValue < 30" if is_long else "uoValue > 70"}''',

        'chop_trend': f'''// Choppiness Trend Entry
chopValue = ta.chop(14)
sma20 = ta.sma(close, 20)
isTrending = chopValue < 38.2
entrySignal = isTrending and {"close > sma20" if is_long else "close < sma20"}''',

        'double_ema_cross': f'''// Double EMA Cross Entry
ema12 = ta.ema(close, 12)
ema26 = ta.ema(close, 26)
entrySignal = {"ta.crossover(ema12, ema26)" if is_long else "ta.crossunder(ema12, ema26)"}''',

        'triple_ema': f'''// Triple EMA Alignment Entry
ema9 = ta.ema(close, 9)
ema21 = ta.ema(close, 21)
ema50 = ta.ema(close, 50)
{"aligned = ema9 > ema21 and ema21 > ema50" if is_long else "aligned = ema9 < ema21 and ema21 < ema50"}
{"wasNotAligned = not (ema9[1] > ema21[1] and ema21[1] > ema50[1])" if is_long else "wasNotAligned = not (ema9[1] < ema21[1] and ema21[1] < ema50[1])"}
entrySignal = aligned and wasNotAligned''',
    }

    # Get the entry condition code for this strategy
    entry_code = entry_conditions.get(entry_rule, 'entrySignal = true  // Unknown strategy')

    return f'''// =============================================================================
// {result.strategy_name.upper()}
// =============================================================================
// Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
//
// BACKTEST RESULTS (Python):
//   Trades: {result.total_trades} (Wins: {result.wins}, Losses: {result.losses})
//   Win Rate: {result.win_rate:.1f}%
//   Total Return: {result.total_pnl_percent:.1f}%
//   Profit Factor: {result.profit_factor}
//   Composite Score: {result.composite_score:.0f}
//
// EXACT-MATCH SETTINGS:
//   Direction: {result.direction.upper()}
//   Take Profit: {result.tp_percent}%
//   Stop Loss: {result.sl_percent}%
//   Entry Rule: {result.entry_rule}
//   Position Size: 75% of equity (compounding)
//   Commission: 0.1%
// =============================================================================

//@version=6
strategy("{result.strategy_name}",
         overlay=true,
         process_orders_on_close=true,
         default_qty_type=strategy.percent_of_equity,
         default_qty_value=75,
         initial_capital=1000,
         commission_type=strategy.commission.percent,
         commission_value=0.1)

// ============ RISK MANAGEMENT ============
tpPercent = input.float({result.tp_percent}, "Take Profit %", minval=0.1, step=0.1, group="Risk Management")
slPercent = input.float({result.sl_percent}, "Stop Loss %", minval=0.1, step=0.1, group="Risk Management")

// ============ DIRECTION ============
enableLongs = input.bool({str(is_long).lower()}, "Enable Long Trades", group="Direction")
enableShorts = input.bool({str(not is_long).lower()}, "Enable Short Trades", group="Direction")

// ============ VISUALS ============
showLabels = input.bool(true, "Show Labels", group="Visuals")
showTPSL = input.bool(true, "Show TP/SL Levels", group="Visuals")

// ============ ENTRY LOGIC ============
{entry_code}

// ============ POSITION MANAGEMENT ============
// Long Entry
if enableLongs and entrySignal and strategy.position_size == 0
    strategy.entry("Long", strategy.long)

// Short Entry
if enableShorts and entrySignal and strategy.position_size == 0
    strategy.entry("Short", strategy.short)

// Long Exit
if strategy.position_size > 0
    tpPrice = strategy.position_avg_price * (1 + tpPercent/100)
    slPrice = strategy.position_avg_price * (1 - slPercent/100)
    strategy.exit("Long Exit", "Long", limit=tpPrice, stop=slPrice)

    // Visualize TP/SL levels
    if showTPSL
        line.new(bar_index[1], tpPrice, bar_index, tpPrice, color=color.green, style=line.style_dotted)
        line.new(bar_index[1], slPrice, bar_index, slPrice, color=color.red, style=line.style_dotted)

// Short Exit
if strategy.position_size < 0
    tpPrice = strategy.position_avg_price * (1 - tpPercent/100)
    slPrice = strategy.position_avg_price * (1 + slPercent/100)
    strategy.exit("Short Exit", "Short", limit=tpPrice, stop=slPrice)

    // Visualize TP/SL levels
    if showTPSL
        line.new(bar_index[1], tpPrice, bar_index, tpPrice, color=color.green, style=line.style_dotted)
        line.new(bar_index[1], slPrice, bar_index, slPrice, color=color.red, style=line.style_dotted)

// ============ ENTRY LABELS ============
if showLabels and entrySignal and strategy.position_size == 0
    if enableLongs
        label.new(bar_index, low, "▲", style=label.style_label_up, color=color.green, textcolor=color.white)
    if enableShorts
        label.new(bar_index, high, "▼", style=label.style_label_down, color=color.red, textcolor=color.white)

// ============ INFO TABLE ============
var table infoTable = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 80), border_width=1)
if barstate.islast
    table.cell(infoTable, 0, 0, "Strategy", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 0, "{result.strategy_name}", text_color=color.white, text_size=size.small)
    table.cell(infoTable, 0, 1, "Entry Rule", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 1, "{result.entry_rule}", text_color=color.yellow, text_size=size.small)
    table.cell(infoTable, 0, 2, "Direction", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 2, "{result.direction.upper()}", text_color={"color.green" if is_long else "color.red"}, text_size=size.small)
    table.cell(infoTable, 0, 3, "TP / SL", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 3, str.tostring(tpPercent) + "% / " + str.tostring(slPercent) + "%", text_color=color.white, text_size=size.small)
    table.cell(infoTable, 0, 4, "Python WR", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 4, "{result.win_rate:.1f}%", text_color=color.lime, text_size=size.small)
    table.cell(infoTable, 0, 5, "Python Score", text_color=color.gray, text_size=size.small)
    table.cell(infoTable, 1, 5, "{result.composite_score:.0f}", text_color=color.lime, text_size=size.small)
'''


# Main entry point
def run_strategy_finder(df: pd.DataFrame,
                        status: Dict = None,
                        streaming_callback: Callable = None,
                        symbol: str = None,
                        timeframe: str = None,
                        exchange: str = None,
                        capital: float = 1000.0,
                        position_size_pct: float = 75.0,
                        engine: str = "tradingview",
                        n_trials: int = 300,
                        progress_min: int = 0,
                        progress_max: int = 100) -> Dict:
    """Main entry point for the strategy engine.

    Args:
        symbol: Trading symbol (e.g., 'BTCGBP')
        timeframe: Timeframe (e.g., '15m')
        exchange: Exchange name (e.g., 'KRAKEN', 'BINANCE')
        capital: Starting capital (from UI)
        position_size_pct: Position size as % of equity (from UI "Position Size %")
        engine: Calculation engine - "tradingview" or "native"
        n_trials: Controls TP/SL granularity (100=1%, 300=0.33%, 500=0.2% increments)
    """

    strategy_engine = StrategyEngine(df, status, streaming_callback,
                                     capital=capital, position_size_pct=position_size_pct,
                                     calc_engine=engine,
                                     progress_min=progress_min, progress_max=progress_max)

    # First, check for previous winners
    winners = strategy_engine.get_saved_winners(symbol=symbol, limit=10)
    if winners:
        print(f"Found {len(winners)} previous winning strategies to build on")

    # Find new strategies
    results = strategy_engine.find_strategies(
        min_trades=1,
        save_to_db=True,
        symbol=symbol,
        timeframe=timeframe,
        n_trials=n_trials
    )

    # Format report
    profitable = [r for r in results if r.total_pnl > 0]
    beats_bh = [r for r in profitable if r.beats_buy_hold]

    report = {
        'generated_at': datetime.now().isoformat(),
        'data_rows': len(df),
        'exchange': exchange,
        'symbol': symbol,
        'timeframe': timeframe,
        'engine': engine,  # Store calculation engine used for consistency with Pine Script
        'total_tested': len(results),
        'profitable_found': len(profitable),
        'beats_buy_hold_count': len(beats_bh),
        'previous_winners_used': len(winners),
        # Buy & Hold benchmark
        'buy_hold_return': strategy_engine.buy_hold_return,
        'workers_used': OPTIMAL_WORKERS,
        # Trading parameters (for Pine Script generation)
        'capital': capital,
        'position_size_pct': position_size_pct,
        'top_10': []
    }

    for i, r in enumerate(profitable[:10], 1):
        report['top_10'].append({
            'rank': i,
            'strategy_name': r.strategy_name,
            'strategy_category': r.strategy_category,
            'entry_rule': r.entry_rule,
            'direction': r.direction,
            'params': r.params,
            'metrics': {
                'total_trades': r.total_trades,
                'wins': r.wins,
                'losses': r.losses,
                'win_rate': round(r.win_rate, 1),
                'profit_factor': r.profit_factor,
                'total_pnl': round(r.total_pnl, 2),
                'total_pnl_percent': round(r.total_pnl_percent, 2),
                'max_drawdown': round(r.max_drawdown, 2),
                'max_drawdown_percent': round(r.max_drawdown_percent, 2),
                'avg_trade': round(r.avg_trade, 2),
                'composite_score': round(r.composite_score, 1),
                # Buy & Hold comparison
                'buy_hold_return': r.buy_hold_return,
                'vs_buy_hold': r.vs_buy_hold,
                'beats_buy_hold': r.beats_buy_hold,
            },
            'equity_curve': r.equity_curve if r.equity_curve else [],
            'trades_list': r.trades_list if r.trades_list else [],  # All trades for CSV export
            # Open position warning (shows if backtest ended with unclosed trade)
            'has_open_position': r.has_open_position,
            'open_position': r.open_position
        })

    return report
