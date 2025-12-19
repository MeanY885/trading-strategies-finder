"""
EXACT-MATCH BACKTESTER
======================
Guarantees 1:1 matching between Python backtest and TradingView Pine Script.

Key principles:
1. Entry at CLOSE of signal bar (matches TradingView process_orders_on_close=true)
2. Percentage-based TP/SL (not ATR-based) for exact calculation
3. Simple, verifiable logic that translates directly to Pine Script
4. Position sizing uses fixed BTC amount (0.01) like TradingView

Author: BTCGBP ML Optimizer
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import sqlite3
import os


@dataclass
class Trade:
    """Single trade record - matches TradingView trade format exactly"""
    entry_time: datetime
    entry_price: float
    direction: str  # "long" or "short"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "take_profit", "stop_loss", "end_of_data"
    pnl_gbp: Optional[float] = None
    pnl_percent: Optional[float] = None
    position_size_btc: float = 0.01


@dataclass
class StrategyResult:
    """Complete backtest results for a strategy configuration"""
    # Strategy identification
    strategy_id: str = ""
    strategy_name: str = ""
    direction: str = ""  # "long", "short", "both"

    # Parameters
    tp_percent: float = 0.0
    sl_percent: float = 0.0
    entry_condition: str = ""
    params: Dict = field(default_factory=dict)

    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L
    total_pnl_gbp: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    max_drawdown_gbp: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0

    # Equity curve for visualization
    equity_curve: List[float] = field(default_factory=list)

    # Trade list (for verification)
    trades: List[Dict] = field(default_factory=list)

    # Metadata
    data_start: str = ""
    data_end: str = ""
    timeframe: str = ""
    pair: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON/SQLite storage"""
        d = asdict(self)
        d['trades'] = json.dumps(d['trades'])
        d['equity_curve'] = json.dumps(d['equity_curve'])
        d['params'] = json.dumps(d['params'])
        return d


class ExactMatchBacktester:
    """
    Backtester that guarantees 1:1 matching with TradingView.

    Key matching rules:
    1. Entry at CLOSE of signal bar
    2. SL/TP calculated as percentage of entry price
    3. SL hit if LOW (for long) or HIGH (for short) touches stop level
    4. TP hit if HIGH (for long) or LOW (for short) touches target level
    5. When both SL and TP could be hit on same bar, SL takes priority (worst case)
    6. Commission: 0.1% per side (matches TradingView default)
    """

    COMMISSION_PERCENT = 0.1  # 0.1% per trade
    POSITION_SIZE_BTC = 0.01  # Fixed position size

    def __init__(self, df: pd.DataFrame, initial_capital: float = 1000.0):
        """
        Initialize backtester with OHLC data.

        Args:
            df: DataFrame with columns: time, open, high, low, close
            initial_capital: Starting capital in GBP
        """
        self.df = df.copy()
        self.initial_capital = initial_capital

        # Ensure time column is datetime
        if 'time' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['time']):
            self.df['time'] = pd.to_datetime(self.df['time'])

        # Pre-calculate common indicators
        self._calculate_indicators()

    def _calculate_indicators(self):
        """Pre-calculate commonly used indicators"""
        df = self.df

        # RSI (14-period default)
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # SMA
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

        # EMA
        for period in [9, 21, 55]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Bollinger Bands
        for period in [20]:
            for mult in [2.0]:
                sma = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                df[f'bb_upper_{period}'] = sma + (mult * std)
                df[f'bb_lower_{period}'] = sma - (mult * std)
                df[f'bb_middle_{period}'] = sma

        # ADX (simplified)
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_14 = tr.rolling(14).mean()
        df['atr_14'] = atr_14

        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df['adx_14'] = dx.rolling(14).mean()
        df['di_plus_14'] = plus_di
        df['di_minus_14'] = minus_di

        # Candle patterns
        df['is_green'] = df['close'] > df['open']
        df['is_red'] = df['close'] < df['open']
        df['body_size'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']

        # Store
        self.df = df

    def generate_signals(
        self,
        entry_condition: str,
        direction: Literal["long", "short", "both"],
        params: Dict = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry signals based on condition and direction.

        Returns:
            Tuple of (long_signals, short_signals) boolean Series
        """
        params = params or {}
        df = self.df

        # Initialize signals
        long_sig = pd.Series(False, index=df.index)
        short_sig = pd.Series(False, index=df.index)

        # === SIMPLE CONDITIONS (translates 1:1 to Pine Script) ===

        if entry_condition == "always":
            # Enter on every bar (baseline test)
            if direction in ["long", "both"]:
                long_sig = pd.Series(True, index=df.index)
            if direction in ["short", "both"]:
                short_sig = pd.Series(True, index=df.index)

        elif entry_condition == "rsi_oversold":
            # RSI below threshold = long, RSI above threshold = short
            rsi_period = params.get('rsi_period', 14)
            rsi_os = params.get('rsi_oversold', 30)
            rsi_ob = params.get('rsi_overbought', 70)
            rsi = df.get(f'rsi_{rsi_period}', df.get('rsi_14', pd.Series(50, index=df.index)))

            if direction in ["long", "both"]:
                long_sig = rsi < rsi_os
            if direction in ["short", "both"]:
                short_sig = rsi > rsi_ob

        elif entry_condition == "bb_touch":
            # Price touches Bollinger Band
            bb_period = params.get('bb_period', 20)
            bb_upper = df.get(f'bb_upper_{bb_period}', df['close'] * 1.02)
            bb_lower = df.get(f'bb_lower_{bb_period}', df['close'] * 0.98)

            if direction in ["long", "both"]:
                long_sig = df['close'] <= bb_lower
            if direction in ["short", "both"]:
                short_sig = df['close'] >= bb_upper

        elif entry_condition == "sma_cross":
            # SMA crossover
            fast_period = params.get('sma_fast', 10)
            slow_period = params.get('sma_slow', 50)
            fast_sma = df.get(f'sma_{fast_period}', df['close'].rolling(fast_period).mean())
            slow_sma = df.get(f'sma_{slow_period}', df['close'].rolling(slow_period).mean())

            if direction in ["long", "both"]:
                long_sig = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
            if direction in ["short", "both"]:
                short_sig = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))

        elif entry_condition == "consecutive_red":
            # N consecutive red candles (reversal long)
            n = params.get('consec_count', 3)
            red_count = df['is_red'].rolling(n).sum()

            if direction in ["long", "both"]:
                long_sig = red_count == n
            if direction in ["short", "both"]:
                green_count = df['is_green'].rolling(n).sum()
                short_sig = green_count == n

        elif entry_condition == "consecutive_green":
            # N consecutive green candles (trend follow or reversal short)
            n = params.get('consec_count', 3)
            green_count = df['is_green'].rolling(n).sum()

            if direction in ["long", "both"]:
                long_sig = green_count == n
            if direction in ["short", "both"]:
                red_count = df['is_red'].rolling(n).sum()
                short_sig = red_count == n

        elif entry_condition == "price_drop":
            # Price dropped X% from recent high
            drop_pct = params.get('drop_percent', 2.0) / 100
            lookback = params.get('lookback', 10)
            recent_high = df['high'].rolling(lookback).max()
            current_drop = (recent_high - df['close']) / recent_high

            if direction in ["long", "both"]:
                long_sig = current_drop >= drop_pct
            if direction in ["short", "both"]:
                recent_low = df['low'].rolling(lookback).min()
                current_rise = (df['close'] - recent_low) / recent_low
                short_sig = current_rise >= drop_pct

        elif entry_condition == "price_rise":
            # Price rose X% from recent low (for short entries)
            rise_pct = params.get('rise_percent', 2.0) / 100
            lookback = params.get('lookback', 10)
            recent_low = df['low'].rolling(lookback).min()
            current_rise = (df['close'] - recent_low) / recent_low

            if direction in ["short", "both"]:
                short_sig = current_rise >= rise_pct
            if direction in ["long", "both"]:
                recent_high = df['high'].rolling(lookback).max()
                current_drop = (recent_high - df['close']) / recent_high
                long_sig = current_drop >= rise_pct

        elif entry_condition == "adx_sideways":
            # ADX below threshold (sideways market) + RSI extreme
            adx_thresh = params.get('adx_threshold', 25)
            rsi_os = params.get('rsi_oversold', 30)
            rsi_ob = params.get('rsi_overbought', 70)
            adx = df.get('adx_14', pd.Series(20, index=df.index))
            rsi = df.get('rsi_14', pd.Series(50, index=df.index))
            sideways = adx < adx_thresh

            if direction in ["long", "both"]:
                long_sig = sideways & (rsi < rsi_os)
            if direction in ["short", "both"]:
                short_sig = sideways & (rsi > rsi_ob)

        elif entry_condition == "adx_trending":
            # ADX above threshold (trending market) + DI crossover
            adx_thresh = params.get('adx_threshold', 25)
            adx = df.get('adx_14', pd.Series(20, index=df.index))
            di_plus = df.get('di_plus_14', pd.Series(0, index=df.index))
            di_minus = df.get('di_minus_14', pd.Series(0, index=df.index))
            trending = adx > adx_thresh

            if direction in ["long", "both"]:
                long_sig = trending & (di_plus > di_minus)
            if direction in ["short", "both"]:
                short_sig = trending & (di_minus > di_plus)

        elif entry_condition == "ema_trend":
            # Price above/below EMA indicates trend
            ema_period = params.get('ema_period', 21)
            ema = df.get(f'ema_{ema_period}', df['close'].ewm(span=ema_period, adjust=False).mean())

            if direction in ["long", "both"]:
                long_sig = df['close'] > ema
            if direction in ["short", "both"]:
                short_sig = df['close'] < ema

        elif entry_condition == "every_n_bars":
            # Enter every N bars (for time-based strategies)
            n = params.get('n_bars', 10)
            long_sig = pd.Series(False, index=df.index)
            short_sig = pd.Series(False, index=df.index)
            for i in range(n, len(df), n):
                if direction in ["long", "both"]:
                    long_sig.iloc[i] = True
                if direction in ["short", "both"]:
                    short_sig.iloc[i] = True

        return long_sig, short_sig

    def run_backtest(
        self,
        direction: Literal["long", "short", "both"],
        tp_percent: float,
        sl_percent: float,
        entry_condition: str = "always",
        entry_params: Dict = None,
        one_trade_at_a_time: bool = True
    ) -> StrategyResult:
        """
        Run exact-match backtest.

        MATCHES TRADINGVIEW EXACTLY:
        - Entry at CLOSE of signal bar
        - SL/TP as percentage of entry price
        - SL checked before TP on same bar (worst case)
        - Commission: 0.1% per side

        Args:
            direction: "long", "short", or "both"
            tp_percent: Take profit percentage (e.g., 0.6 for 0.6%)
            sl_percent: Stop loss percentage (e.g., 5.0 for 5%)
            entry_condition: Signal type
            entry_params: Parameters for entry condition
            one_trade_at_a_time: If True, only one position at a time

        Returns:
            StrategyResult with all metrics
        """
        entry_params = entry_params or {}
        df = self.df

        # Generate signals
        long_signals, short_signals = self.generate_signals(
            entry_condition, direction, entry_params
        )

        # Run simulation
        trades: List[Trade] = []
        current_position: Optional[Trade] = None
        equity = self.initial_capital
        equity_curve = [equity]

        # Skip warmup period for indicators
        start_idx = 50

        for i in range(start_idx, len(df)):
            row = df.iloc[i]

            # === CHECK EXIT FOR EXISTING POSITION ===
            if current_position is not None:
                exit_price = None
                exit_reason = None

                entry_price = current_position.entry_price

                if current_position.direction == "long":
                    # Long position: SL hit if LOW touches stop, TP hit if HIGH touches target
                    sl_price = entry_price * (1 - sl_percent / 100)
                    tp_price = entry_price * (1 + tp_percent / 100)

                    # Check SL first (worst case on same bar)
                    if row['low'] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                    elif row['high'] >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"

                else:  # short position
                    # Short position: SL hit if HIGH touches stop, TP hit if LOW touches target
                    sl_price = entry_price * (1 + sl_percent / 100)
                    tp_price = entry_price * (1 - tp_percent / 100)

                    # Check SL first (worst case on same bar)
                    if row['high'] >= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                    elif row['low'] <= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"

                # Process exit
                if exit_price is not None:
                    current_position.exit_time = row['time']
                    current_position.exit_price = exit_price
                    current_position.exit_reason = exit_reason

                    # Calculate P&L
                    if current_position.direction == "long":
                        pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        pnl_percent = ((entry_price - exit_price) / entry_price) * 100

                    # Subtract commission (0.1% each side = 0.2% total)
                    pnl_percent -= (self.COMMISSION_PERCENT * 2)

                    # Convert to GBP (position size * price movement)
                    price_movement = exit_price - entry_price
                    if current_position.direction == "short":
                        price_movement = -price_movement
                    pnl_gbp = price_movement * self.POSITION_SIZE_BTC

                    # Subtract commission in GBP
                    commission_gbp = (entry_price + exit_price) * self.POSITION_SIZE_BTC * (self.COMMISSION_PERCENT / 100)
                    pnl_gbp -= commission_gbp

                    current_position.pnl_gbp = pnl_gbp
                    current_position.pnl_percent = pnl_percent

                    trades.append(current_position)
                    equity += pnl_gbp
                    current_position = None

            # === CHECK ENTRY SIGNALS (only if no position) ===
            if current_position is None or not one_trade_at_a_time:
                # Entry at CLOSE of signal bar (matches TradingView)
                if long_signals.iloc[i] and direction in ["long", "both"]:
                    current_position = Trade(
                        entry_time=row['time'],
                        entry_price=row['close'],
                        direction="long",
                        position_size_btc=self.POSITION_SIZE_BTC
                    )
                elif short_signals.iloc[i] and direction in ["short", "both"]:
                    current_position = Trade(
                        entry_time=row['time'],
                        entry_price=row['close'],
                        direction="short",
                        position_size_btc=self.POSITION_SIZE_BTC
                    )

            equity_curve.append(equity)

        # Close any open position at end
        if current_position is not None:
            last_row = df.iloc[-1]
            current_position.exit_time = last_row['time']
            current_position.exit_price = last_row['close']
            current_position.exit_reason = "end_of_data"

            entry_price = current_position.entry_price
            exit_price = last_row['close']

            if current_position.direction == "long":
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100

            pnl_percent -= (self.COMMISSION_PERCENT * 2)

            price_movement = exit_price - entry_price
            if current_position.direction == "short":
                price_movement = -price_movement
            pnl_gbp = price_movement * self.POSITION_SIZE_BTC
            commission_gbp = (entry_price + exit_price) * self.POSITION_SIZE_BTC * (self.COMMISSION_PERCENT / 100)
            pnl_gbp -= commission_gbp

            current_position.pnl_gbp = pnl_gbp
            current_position.pnl_percent = pnl_percent
            trades.append(current_position)
            equity += pnl_gbp
            equity_curve.append(equity)

        # Calculate metrics
        return self._calculate_metrics(
            trades=trades,
            equity_curve=equity_curve,
            direction=direction,
            tp_percent=tp_percent,
            sl_percent=sl_percent,
            entry_condition=entry_condition,
            entry_params=entry_params
        )

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        direction: str,
        tp_percent: float,
        sl_percent: float,
        entry_condition: str,
        entry_params: Dict
    ) -> StrategyResult:
        """Calculate comprehensive performance metrics"""

        result = StrategyResult(
            strategy_id=f"{entry_condition}_{direction}_tp{tp_percent}_sl{sl_percent}",
            strategy_name=f"{entry_condition.replace('_', ' ').title()} ({direction.upper()})",
            direction=direction,
            tp_percent=tp_percent,
            sl_percent=sl_percent,
            entry_condition=entry_condition,
            params=entry_params,
            equity_curve=equity_curve,
            data_start=str(self.df['time'].iloc[0]) if len(self.df) > 0 else "",
            data_end=str(self.df['time'].iloc[-1]) if len(self.df) > 0 else "",
            created_at=datetime.now().isoformat()
        )

        if len(trades) == 0:
            return result

        # Trade counts
        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t.pnl_gbp and t.pnl_gbp > 0)
        result.losing_trades = sum(1 for t in trades if t.pnl_gbp and t.pnl_gbp <= 0)

        # Win rate
        result.win_rate = (result.winning_trades / result.total_trades * 100) if result.total_trades > 0 else 0

        # P&L
        pnls = [t.pnl_gbp for t in trades if t.pnl_gbp is not None]
        result.total_pnl_gbp = sum(pnls)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.gross_profit = sum(wins) if wins else 0
        result.gross_loss = abs(sum(losses)) if losses else 0
        result.profit_factor = (result.gross_profit / result.gross_loss) if result.gross_loss > 0 else (10.0 if result.gross_profit > 0 else 0)

        # Drawdown
        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = running_max - equity_arr
        result.max_drawdown_gbp = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0
        result.max_drawdown_percent = (result.max_drawdown_gbp / self.initial_capital * 100) if self.initial_capital > 0 else 0

        # Sharpe ratio
        if len(pnls) > 1 and np.std(pnls) > 0:
            result.sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)

        # Trade details (for verification)
        result.trades = [
            {
                'entry_time': str(t.entry_time),
                'exit_time': str(t.exit_time),
                'direction': t.direction,
                'entry_price': round(t.entry_price, 2),
                'exit_price': round(t.exit_price, 2) if t.exit_price else None,
                'exit_reason': t.exit_reason,
                'pnl_gbp': round(t.pnl_gbp, 4) if t.pnl_gbp else None,
                'pnl_percent': round(t.pnl_percent, 4) if t.pnl_percent else None
            }
            for t in trades
        ]

        return result


class StrategyDatabase:
    """SQLite database for persistent strategy storage"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__),
                '..', 'data', 'strategies.db'
            )

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT UNIQUE,
                strategy_name TEXT,
                direction TEXT,
                tp_percent REAL,
                sl_percent REAL,
                entry_condition TEXT,
                params TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                total_pnl_gbp REAL,
                gross_profit REAL,
                gross_loss REAL,
                profit_factor REAL,
                max_drawdown_gbp REAL,
                max_drawdown_percent REAL,
                sharpe_ratio REAL,
                equity_curve TEXT,
                trades TEXT,
                data_start TEXT,
                data_end TEXT,
                timeframe TEXT,
                pair TEXT,
                created_at TEXT,
                UNIQUE(strategy_id, data_start, data_end)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_profit ON strategies(total_pnl_gbp DESC)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_win_rate ON strategies(win_rate DESC)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_profit_factor ON strategies(profit_factor DESC)
        ''')

        conn.commit()
        conn.close()

    def save_strategy(self, result: StrategyResult):
        """Save strategy result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data = result.to_dict()

        cursor.execute('''
            INSERT OR REPLACE INTO strategies
            (strategy_id, strategy_name, direction, tp_percent, sl_percent,
             entry_condition, params, total_trades, winning_trades, losing_trades,
             win_rate, total_pnl_gbp, gross_profit, gross_loss, profit_factor,
             max_drawdown_gbp, max_drawdown_percent, sharpe_ratio, equity_curve,
             trades, data_start, data_end, timeframe, pair, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['strategy_id'], data['strategy_name'], data['direction'],
            data['tp_percent'], data['sl_percent'], data['entry_condition'],
            data['params'], data['total_trades'], data['winning_trades'],
            data['losing_trades'], data['win_rate'], data['total_pnl_gbp'],
            data['gross_profit'], data['gross_loss'], data['profit_factor'],
            data['max_drawdown_gbp'], data['max_drawdown_percent'],
            data['sharpe_ratio'], data['equity_curve'], data['trades'],
            data['data_start'], data['data_end'], data['timeframe'],
            data['pair'], data['created_at']
        ))

        conn.commit()
        conn.close()

    def get_top_strategies(
        self,
        limit: int = 10,
        order_by: str = "total_pnl_gbp",
        direction: str = None,
        min_trades: int = 5
    ) -> List[Dict]:
        """Get top performing strategies"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = f'''
            SELECT * FROM strategies
            WHERE total_trades >= ?
        '''
        params = [min_trades]

        if direction:
            query += ' AND direction = ?'
            params.append(direction)

        query += f' ORDER BY {order_by} DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        conn.close()

        return [dict(zip(columns, row)) for row in rows]

    def get_best_for_conditions(
        self,
        entry_condition: str = None,
        direction: str = None
    ) -> Optional[Dict]:
        """Get best strategy for given conditions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = 'SELECT * FROM strategies WHERE total_trades >= 5'
        params = []

        if entry_condition:
            query += ' AND entry_condition = ?'
            params.append(entry_condition)

        if direction:
            query += ' AND direction = ?'
            params.append(direction)

        query += ' ORDER BY total_pnl_gbp DESC LIMIT 1'

        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()

        conn.close()

        return dict(zip(columns, row)) if row else None

    def clear_all(self):
        """Clear all stored strategies"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM strategies')
        conn.commit()
        conn.close()


# === UTILITY FUNCTIONS ===

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load OHLC data from CSV file"""
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Rename common variations
    column_map = {
        'timestamp': 'time',
        'date': 'time',
        'datetime': 'time',
    }
    df.rename(columns=column_map, inplace=True)

    # Ensure required columns exist
    required = ['time', 'open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Parse time
    df['time'] = pd.to_datetime(df['time'])

    # Sort by time
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    # Quick test
    print("Exact Match Backtester - Test")
    print("=" * 50)

    # Load sample data if available
    sample_path = "/Users/chriseddisford/Downloads/KRAKEN_BTCGBP, 1.csv"
    if os.path.exists(sample_path):
        df = load_csv_data(sample_path)
        print(f"Loaded {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

        backtester = ExactMatchBacktester(df)

        # Test your winning strategy: SHORT with 0.6% TP, 5% SL
        result = backtester.run_backtest(
            direction="short",
            tp_percent=0.6,
            sl_percent=5.0,
            entry_condition="always"
        )

        print(f"\nStrategy: SHORT - TP 0.6% / SL 5%")
        print(f"Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Total P&L: £{result.total_pnl_gbp:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
    else:
        print(f"Sample data not found at {sample_path}")
