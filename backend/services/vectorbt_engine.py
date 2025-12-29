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
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# Suppress VectorBT warnings during import
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import vectorbt as vbt
    from numba import njit
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    njit = lambda f: f  # No-op decorator if numba not available

from logging_config import log
from config import DEFAULT_TRADING_COSTS


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

        # Cache for signals
        self._signal_cache: Dict[str, pd.DataFrame] = {}

        log(f"[VectorBT] ✅ Engine initialized: {len(df)} bars, capital=${initial_capital:,.0f}, position_size={position_size_pct}%")
        log(f"[VectorBT] Trading costs: commission={self.commission_pct}%, spread={self.spread_pct}%, slippage={self.slippage_pct}%")

    def _prepare_dataframe(self):
        """Ensure DataFrame has required columns and format."""
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure time index for VectorBT
        if 'time' in self.df.columns:
            self.df.set_index('time', inplace=True)

        # Add helper columns
        self.df['range'] = self.df['high'] - self.df['low']
        self.df['body'] = abs(self.df['close'] - self.df['open'])
        self.df['green'] = self.df['close'] > self.df['open']
        self.df['red'] = self.df['close'] < self.df['open']

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

        log(f"[VectorBT] Calculated {len([c for c in df.columns if c not in ['open','high','low','close','volume']])} indicators")

    def _get_signals(self, strategy: str, direction: str) -> pd.Series:
        """
        Generate entry signals for a strategy.
        Mirrors StrategyEngine._get_signals() for compatibility.
        """
        df = self.df

        def safe_bool(series):
            return series.fillna(False).astype(bool)

        if strategy == 'always':
            return pd.Series(True, index=df.index)

        elif strategy == 'rsi_extreme':
            if direction == 'long':
                return safe_bool((df['rsi'] > 30) & (df['rsi'].shift(1) <= 30))
            else:
                return safe_bool((df['rsi'] < 70) & (df['rsi'].shift(1) >= 70))

        elif strategy == 'rsi_cross_50':
            if direction == 'long':
                return safe_bool((df['rsi'] > 50) & (df['rsi'].shift(1) <= 50))
            else:
                return safe_bool((df['rsi'] < 50) & (df['rsi'].shift(1) >= 50))

        elif strategy == 'stoch_extreme':
            if direction == 'long':
                k_cross = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
                return safe_bool(k_cross & (df['stoch_k'] < 20))
            else:
                k_cross = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
                return safe_bool(k_cross & (df['stoch_k'] > 80))

        elif strategy == 'bb_touch':
            if direction == 'long':
                return safe_bool((df['close'] > df['bb_lower']) & (df['close'].shift(1) <= df['bb_lower'].shift(1)))
            else:
                return safe_bool((df['close'] < df['bb_upper']) & (df['close'].shift(1) >= df['bb_upper'].shift(1)))

        elif strategy == 'bb_squeeze_breakout':
            squeeze = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8
            expanding = df['bb_width'] > df['bb_width'].shift(1)
            if direction == 'long':
                return safe_bool(squeeze.shift(1) & expanding & (df['close'] > df['bb_mid']))
            else:
                return safe_bool(squeeze.shift(1) & expanding & (df['close'] < df['bb_mid']))

        elif strategy == 'ema_cross':
            if direction == 'long':
                return safe_bool((df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1)))
            else:
                return safe_bool((df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1)))

        elif strategy == 'sma_cross':
            sma_fast = df['close'].rolling(9).mean()
            sma_slow = df['close'].rolling(18).mean()
            if direction == 'long':
                return safe_bool((sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1)))
            else:
                return safe_bool((sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1)))

        elif strategy == 'macd_cross':
            histogram = df['macd'] - df['macd_signal']
            if direction == 'long':
                return safe_bool((histogram > 0) & (histogram.shift(1) <= 0))
            else:
                return safe_bool((histogram < 0) & (histogram.shift(1) >= 0))

        elif strategy == 'supertrend':
            if direction == 'long':
                return safe_bool((df['supertrend_dir'] == 1) & (df['supertrend_dir'].shift(1) == -1))
            else:
                return safe_bool((df['supertrend_dir'] == -1) & (df['supertrend_dir'].shift(1) == 1))

        elif strategy == 'consecutive_candles':
            up_close = df['close'] > df['close'].shift(1)
            down_close = df['close'] < df['close'].shift(1)
            ups = up_close.astype(int).groupby((~up_close).cumsum()).cumsum()
            dns = down_close.astype(int).groupby((~down_close).cumsum()).cumsum()
            if direction == 'long':
                return safe_bool(ups >= 3)
            else:
                return safe_bool(dns >= 3)

        elif strategy == 'engulfing':
            if direction == 'long':
                return safe_bool(df['green'] & df['red'].shift(1) &
                               (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)))
            else:
                return safe_bool(df['red'] & df['green'].shift(1) &
                               (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1)))

        elif strategy == 'inside_bar':
            inside = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
            if direction == 'long':
                return safe_bool(inside & (df['close'] > df['open']))
            else:
                return safe_bool(inside & (df['close'] < df['open']))

        elif strategy == 'outside_bar':
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

        # Default: no signals
        return pd.Series(False, index=df.index)

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
                freq='1D',  # Will be adjusted based on data
            )

            # Extract metrics
            return self._extract_metrics(pf, strategy, direction, tp_percent, sl_percent)

        except Exception as e:
            log(f"[VectorBT] Backtest error for {strategy}/{direction}: {e}", level='WARNING')
            return self._empty_result(strategy, direction, tp_percent, sl_percent)

    def run_optimization(
        self,
        strategies: List[str] = None,
        directions: List[str] = None,
        tp_range: np.ndarray = None,
        sl_range: np.ndarray = None,
        mode: str = 'all',
        progress_callback: Callable = None,
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

        Returns:
            List of VectorBTResult sorted by composite score
        """
        if strategies is None:
            strategies = list(self.ENTRY_STRATEGIES.keys())

        if directions is None:
            directions = ['long', 'short']

        if tp_range is None:
            tp_range = np.arange(0.5, 5.1, 0.5)

        if sl_range is None:
            sl_range = np.arange(0.5, 5.1, 0.5)

        results = []
        total_combos = len(strategies) * len(directions) * len(tp_range) * len(sl_range)
        completed = 0

        import time
        start_time = time.time()

        log(f"[VectorBT] 🚀 Starting VECTORIZED optimization")
        log(f"[VectorBT] Parameters: {len(strategies)} strategies × {len(directions)} directions × {len(tp_range)} TPs × {len(sl_range)} SLs")
        log(f"[VectorBT] Total combinations: {total_combos:,} (using NumPy broadcasting for speed)")

        # Use VectorBT broadcasting for massive speedup
        for strategy in strategies:
            for direction in directions:
                # Get signals once per strategy/direction
                entries = self._get_signals(strategy, direction)

                if not entries.any():
                    completed += len(tp_range) * len(sl_range)
                    continue

                short = direction == 'short'

                # Broadcast across TP/SL combinations
                try:
                    pf = vbt.Portfolio.from_signals(
                        close=self.df['close'],
                        entries=entries if not short else pd.Series(False, index=self.df.index),
                        short_entries=entries if short else pd.Series(False, index=self.df.index),
                        sl_stop=sl_range / 100,  # Broadcasting!
                        tp_stop=tp_range / 100,  # Broadcasting!
                        size=self.position_size_pct / 100,
                        size_type='percent',
                        fees=self.total_fees,
                        init_cash=self.initial_capital,
                        freq='1D',
                    )

                    # Extract metrics for each combination
                    for i, tp in enumerate(tp_range):
                        for j, sl in enumerate(sl_range):
                            try:
                                # Get sub-portfolio for this TP/SL combination
                                sub_pf = pf[(i, j)] if hasattr(pf, '__getitem__') else pf
                                result = self._extract_metrics(sub_pf, strategy, direction, tp, sl)
                                if result.total_trades > 0:
                                    results.append(result)
                            except Exception as e:
                                log(f"[VectorBT] Error extracting metrics [{i},{j}]: {e}", level='DEBUG')

                            completed += 1

                            if progress_callback and completed % 100 == 0:
                                progress_callback(completed, total_combos)

                except Exception as e:
                    log(f"[VectorBT] Broadcast error for {strategy}/{direction}: {e}", level='WARNING')
                    completed += len(tp_range) * len(sl_range)

        # Sort by composite score
        results.sort(key=lambda r: r.composite_score, reverse=True)

        elapsed = time.time() - start_time
        combos_per_sec = total_combos / elapsed if elapsed > 0 else 0

        log(f"[VectorBT] ✅ Optimization COMPLETE")
        log(f"[VectorBT] Results: {len(results)} valid strategies from {total_combos:,} combinations")
        log(f"[VectorBT] Time: {elapsed:.1f}s ({combos_per_sec:,.0f} combinations/sec)")

        # Compare to estimated iterative time
        estimated_iterative = total_combos * 0.1  # ~100ms per combo in iterative mode
        speedup = estimated_iterative / elapsed if elapsed > 0 else 0
        if speedup > 1:
            log(f"[VectorBT] ⚡ Speedup: ~{speedup:.0f}x faster than iterative engine (est. {estimated_iterative/60:.1f} min iterative)")

        return results

    def _extract_metrics(
        self,
        pf,
        strategy: str,
        direction: str,
        tp_percent: float,
        sl_percent: float,
    ) -> VectorBTResult:
        """Extract metrics from VectorBT portfolio to VectorBTResult."""
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

            # Equity curve
            equity = pf.value().values.tolist()

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
                entry_rule=self.ENTRY_STRATEGIES.get(strategy, {}).get('description', ''),
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
                params={
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'direction': direction,
                    'entry_rule': self.ENTRY_STRATEGIES.get(strategy, {}).get('description', ''),
                }
            )

        except Exception as e:
            log(f"[VectorBT] Metric extraction error: {e}", level='WARNING')
            return self._empty_result(strategy, direction, tp_percent, sl_percent)

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
            entry_rule=self.ENTRY_STRATEGIES.get(strategy, {}).get('description', ''),
        )


def is_vectorbt_available() -> bool:
    """Check if VectorBT is available."""
    return VECTORBT_AVAILABLE
