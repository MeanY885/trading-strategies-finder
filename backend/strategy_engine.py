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
from dataclasses import dataclass
from datetime import datetime
import os
import psutil

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
class ExitConfig:
    """
    Configuration for exit strategy - supports both TP/SL and indicator-based exits.

    Exit Types:
    - 'fixed_tp_sl': Traditional fixed TP/SL percentages (current behavior)
    - 'trailing_stop': ATR-based trailing stop that ratchets with price
    - 'indicator_exit': Exit on indicator signal (trend reversal)
    """
    exit_type: str = 'fixed_tp_sl'  # 'fixed_tp_sl', 'trailing_stop', 'indicator_exit'

    # For fixed_tp_sl
    tp_percent: float = 2.0
    sl_percent: float = 1.0

    # For trailing_stop
    trailing_atr_mult: float = 2.0      # ATR multiplier for trailing distance
    use_protection_sl: bool = True      # Wide protection SL even with trailing
    protection_sl_atr_mult: float = 4.0 # Protection SL distance (wider than trailing)

    # For indicator_exit
    exit_indicator: str = None          # 'supertrend', 'ema_cross', 'psar', 'mcginley'
    exit_indicator_params: Dict = None  # Indicator-specific parameters

    # Pool classification
    pool: str = 'tp_sl'                 # 'tp_sl' or 'indicator_exit'


@dataclass
class TradingCosts:
    """
    Comprehensive trading costs configuration for realistic backtesting.

    These costs filter out marginal strategies that only work with zero costs
    and provide more realistic P&L expectations.
    """
    # Commission costs (per trade, applied to each entry and exit)
    commission_pct: float = 0.1         # 0.1% per trade (Binance spot default)

    # Spread costs (bid-ask spread, applied to entries/exits)
    spread_pct: float = 0.05            # 0.05% spread assumption

    # Slippage (market impact, worse fills than expected)
    slippage_pct: float = 0.03          # 0.03% slippage estimate

    # Enable/disable cost components
    apply_commission: bool = True
    apply_spread: bool = True
    apply_slippage: bool = True

    @property
    def total_entry_cost_pct(self) -> float:
        """Total cost applied at entry (worse entry price)"""
        cost = 0.0
        if self.apply_commission:
            cost += self.commission_pct
        if self.apply_spread:
            cost += self.spread_pct / 2  # Half spread at entry
        if self.apply_slippage:
            cost += self.slippage_pct
        return cost

    @property
    def total_exit_cost_pct(self) -> float:
        """Total cost applied at exit (worse exit price)"""
        cost = 0.0
        if self.apply_commission:
            cost += self.commission_pct
        if self.apply_spread:
            cost += self.spread_pct / 2  # Half spread at exit
        if self.apply_slippage:
            cost += self.slippage_pct
        return cost

    @property
    def total_round_trip_pct(self) -> float:
        """Total costs for a complete round-trip trade"""
        return self.total_entry_cost_pct + self.total_exit_cost_pct


# Default trading costs (realistic Binance spot trading)
DEFAULT_TRADING_COSTS = TradingCosts(
    commission_pct=0.1,
    spread_pct=0.05,
    slippage_pct=0.03
)

# Zero costs (for raw signal testing without friction)
ZERO_TRADING_COSTS = TradingCosts(
    commission_pct=0.0,
    spread_pct=0.0,
    slippage_pct=0.0,
    apply_commission=False,
    apply_spread=False,
    apply_slippage=False
)


@dataclass
class PositionSizing:
    """
    Position sizing configuration for risk management.

    Sizing Methods:
    - 'fixed': Fixed amount per trade (e.g., £1000)
    - 'percent_equity': Percentage of equity per trade (e.g., 2%)
    - 'percent_risk': Risk percentage of equity per trade, sized by SL distance
    - 'kelly': Kelly criterion based on win rate and risk:reward ratio
    - 'volatility_adjusted': Scale position based on ATR (lower volatility = larger position)

    Compounding:
    - When enabled, profits are reinvested for exponential growth
    - This is how strategies achieve 1000%+ returns
    """
    sizing_method: str = 'fixed'          # 'fixed', 'percent_equity', 'percent_risk', 'kelly', 'volatility_adjusted'

    # Initial capital
    initial_capital: float = 10000.0      # Starting equity

    # For 'fixed' method
    fixed_amount: float = 1000.0          # Fixed £ per trade

    # For 'percent_equity' method
    equity_percent: float = 10.0          # 10% of equity per trade

    # For 'percent_risk' method
    risk_percent: float = 1.0             # Risk 1% of equity per trade

    # For 'kelly' method
    kelly_fraction: float = 0.25          # Use 25% of Kelly (quarter Kelly for safety)

    # For 'volatility_adjusted' method
    target_risk_atr: float = 1.5          # Target risk in ATR units
    base_position_size: float = 1000.0    # Base position size to scale from

    # Compounding
    compound_profits: bool = True         # Reinvest profits (exponential growth)

    # Position limits
    max_position_pct: float = 50.0        # Max 50% of equity in single trade
    min_position_size: float = 10.0       # Minimum trade size

    def calculate_position_size(self,
                                 current_equity: float,
                                 sl_distance_pct: float = 1.0,
                                 win_rate: float = 0.5,
                                 avg_win_loss_ratio: float = 1.5,
                                 current_atr_pct: float = 2.0) -> float:
        """
        Calculate position size based on sizing method.

        Args:
            current_equity: Current account equity
            sl_distance_pct: Stop loss distance as percentage (for percent_risk)
            win_rate: Historical win rate (for Kelly)
            avg_win_loss_ratio: Average winner / average loser (for Kelly)
            current_atr_pct: Current ATR as percentage of price (for volatility_adjusted)

        Returns:
            Position size in currency units
        """
        if self.sizing_method == 'fixed':
            position = self.fixed_amount

        elif self.sizing_method == 'percent_equity':
            position = current_equity * (self.equity_percent / 100.0)

        elif self.sizing_method == 'percent_risk':
            # Size so that SL hit = risk_percent loss
            if sl_distance_pct <= 0:
                sl_distance_pct = 1.0  # Default 1% if not specified
            risk_amount = current_equity * (self.risk_percent / 100.0)
            position = risk_amount / (sl_distance_pct / 100.0)

        elif self.sizing_method == 'kelly':
            # Kelly Criterion: f* = (p * b - q) / b
            # Where p = win rate, q = 1 - p, b = avg win/loss ratio
            p = max(0.01, min(0.99, win_rate))  # Clamp to valid range
            q = 1 - p
            b = max(0.1, avg_win_loss_ratio)

            kelly_full = (p * b - q) / b if b > 0 else 0
            kelly_full = max(0, kelly_full)  # Can't be negative

            # Apply fractional Kelly for safety
            kelly_adjusted = kelly_full * self.kelly_fraction
            position = current_equity * kelly_adjusted

        elif self.sizing_method == 'volatility_adjusted':
            # Scale inversely with volatility
            # Lower ATR = larger position, higher ATR = smaller position
            if current_atr_pct <= 0:
                current_atr_pct = 2.0  # Default
            volatility_factor = self.target_risk_atr / current_atr_pct
            position = self.base_position_size * volatility_factor

        else:
            position = self.fixed_amount  # Fallback

        # Apply limits
        max_allowed = current_equity * (self.max_position_pct / 100.0)
        position = min(position, max_allowed)
        position = max(position, self.min_position_size)

        return position


@dataclass
class PortfolioRiskLimits:
    """
    Portfolio-level risk management limits.
    Prevents over-exposure and excessive drawdowns.
    """
    # Concurrent position limits
    max_concurrent_positions: int = 5          # Max open trades at once
    max_positions_per_pair: int = 1            # Max trades per trading pair

    # Drawdown limits
    max_daily_drawdown_pct: float = 5.0        # Pause trading if daily DD exceeds
    max_total_drawdown_pct: float = 20.0       # Stop trading if total DD exceeds

    # Correlation limits
    max_correlation: float = 0.7               # Don't run highly correlated strategies together

    # Exposure limits
    max_total_exposure_pct: float = 100.0      # Max total capital at risk
    max_single_pair_exposure_pct: float = 30.0 # Max exposure to single pair

    # Recovery rules
    reduce_size_after_losses: int = 3          # Reduce size after N consecutive losses
    size_reduction_factor: float = 0.5         # Cut size by 50% when reducing

    def check_daily_drawdown(self, daily_pnl_pct: float) -> bool:
        """Check if daily drawdown limit exceeded. Returns True if OK to trade."""
        return daily_pnl_pct > -self.max_daily_drawdown_pct

    def check_total_drawdown(self, total_drawdown_pct: float) -> bool:
        """Check if total drawdown limit exceeded. Returns True if OK to trade."""
        return total_drawdown_pct < self.max_total_drawdown_pct

    def get_size_multiplier(self, consecutive_losses: int) -> float:
        """Get position size multiplier based on consecutive losses."""
        if consecutive_losses >= self.reduce_size_after_losses:
            return self.size_reduction_factor
        return 1.0


# Default position sizing (fixed amount, no compounding)
DEFAULT_POSITION_SIZING = PositionSizing(
    sizing_method='fixed',
    fixed_amount=1000.0,
    compound_profits=False
)

# Compounding position sizing (for maximizing returns)
COMPOUNDING_POSITION_SIZING = PositionSizing(
    sizing_method='percent_equity',
    equity_percent=10.0,
    compound_profits=True,
    initial_capital=10000.0
)

# Risk-based position sizing (professional approach)
RISK_BASED_POSITION_SIZING = PositionSizing(
    sizing_method='percent_risk',
    risk_percent=1.0,
    compound_profits=True,
    initial_capital=10000.0
)

# Default portfolio risk limits
DEFAULT_PORTFOLIO_LIMITS = PortfolioRiskLimits()


@dataclass
class TradeResult:
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float  # Percentage gain/loss
    exit_reason: str    # 'tp', 'sl', 'trailing_stop', 'indicator_exit', 'flip', 'protection_sl'
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
    # GBP conversion fields (for USD source data)
    pnl_gbp: float = 0.0              # P&L converted to GBP
    position_size_gbp: float = 0.0    # Position size in GBP
    cumulative_pnl_gbp: float = 0.0   # Running total P&L in GBP
    usd_gbp_rate: float = 1.0         # Exchange rate used for this trade
    # NEW: Trend-following metrics
    trade_duration_bars: int = 0       # How many bars the trade lasted
    trade_duration_hours: float = 0.0  # Duration in hours (if time data available)
    mfe_capture_ratio: float = 0.0     # Actual P&L / Max Favorable Excursion (how much trend captured)
    exit_type: str = 'fixed_tp_sl'     # Exit strategy used: 'fixed_tp_sl', 'trailing_stop', 'indicator_exit'


def _generate_period_buckets(earliest: datetime, latest: datetime) -> List[Tuple[str, datetime, datetime]]:
    """
    Generate non-overlapping time buckets that auto-scale based on data span.
    Target: 6-12 periods max to fit UI without overlap.

    Returns: [(label, start_dt, end_dt), ...]

    Labels include date ranges for better clarity:
    - Weekly: "W1 (24-30 Nov)"
    - Bi-weekly: "W1-2 (24 Nov-7 Dec)"
    """
    from datetime import timedelta

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def format_date_range(start: datetime, end: datetime, compact: bool = True) -> str:
        """Format a date range like '24-30 Nov' or '24 Nov-7 Dec'"""
        # End date is exclusive, so subtract 1 day for display
        display_end = end - timedelta(days=1)
        if display_end < start:
            display_end = start

        if start.month == display_end.month:
            # Same month: "24-30 Nov"
            return f"{start.day}-{display_end.day} {month_names[start.month-1]}"
        else:
            # Different months: "24 Nov-7 Dec"
            return f"{start.day} {month_names[start.month-1]}-{display_end.day} {month_names[display_end.month-1]}"

    # Calculate span in days (inclusive)
    span_days = (latest - earliest).days

    # === SCALING RULES (based on actual day difference) ===
    # label_mode: 'index' = simple index, 'week' = W1 with dates, 'month' = month names, etc.
    if span_days <= 7:
        # Up to 1 week: Daily (D1, D2, D3...)
        chunk_days = 1
        label_mode = 'daily'
    elif span_days <= 14:
        # 8-14 days (~2 weeks): 2-day chunks
        chunk_days = 2
        label_mode = 'daily_range'
    elif span_days <= 21:
        # 15-21 days (~3 weeks): 3-day chunks
        chunk_days = 3
        label_mode = 'daily_range'
    elif span_days <= 45:
        # 22-45 days (1-1.5 months): Weekly with date ranges (W1, W2, W3...)
        chunk_days = 7
        label_mode = 'weekly'
    elif span_days <= 90:
        # 46-90 days (1.5-3 months): Bi-weekly with date ranges
        chunk_days = 14
        label_mode = 'biweekly'
    elif span_days <= 180:
        # 91-180 days (3-6 months): Monthly with actual month names
        chunk_days = 30
        label_mode = 'monthly'
    elif span_days <= 365:
        # 181-365 days (6-12 months): Monthly with actual month names
        chunk_days = 30
        label_mode = 'monthly'
    elif span_days <= 730:
        # 1-2 years: Bi-monthly with month names
        chunk_days = 60
        label_mode = 'bimonthly'
    elif span_days <= 1095:
        # 2-3 years: Quarterly (Q1, Q2...)
        chunk_days = 91
        label_mode = 'quarterly'
    elif span_days <= 1825:
        # 3-5 years: Semi-annual (H1, H2...)
        chunk_days = 182
        label_mode = 'semiannual'
    else:
        # 5+ years: Yearly (Y1, Y2...)
        chunk_days = 365
        label_mode = 'yearly'

    # Generate buckets - only create buckets that overlap with data range
    buckets = []
    current = earliest.replace(hour=0, minute=0, second=0, microsecond=0)  # Start of day
    latest_end = latest.replace(hour=23, minute=59, second=59, microsecond=999999)  # End of last day
    i = 0

    while current <= latest_end:
        bucket_end = current + timedelta(days=chunk_days)

        # Generate label based on mode (all include date ranges for clarity)
        date_range = format_date_range(current, bucket_end)

        if label_mode == 'daily':
            label = f'D{i+1}'
        elif label_mode == 'daily_range':
            label = f'D{i*chunk_days+1}-{i*chunk_days+chunk_days}'
        elif label_mode == 'weekly':
            label = f'W{i+1} ({date_range})'
        elif label_mode == 'biweekly':
            label = f'W{i*2+1}-{i*2+2} ({date_range})'
        elif label_mode == 'monthly':
            label = f'{month_names[current.month - 1]} ({date_range})'
        elif label_mode == 'bimonthly':
            label = f'{month_names[current.month-1]}-{month_names[(current.month % 12)][:3]} ({date_range})'
        elif label_mode == 'quarterly':
            label = f'Q{i+1} ({date_range})'
        elif label_mode == 'semiannual':
            label = f'H{i+1} ({date_range})'
        elif label_mode == 'yearly':
            label = f'Y{i+1} ({date_range})'
        else:
            label = f'P{i+1}'

        buckets.append((label, current, bucket_end))
        current = bucket_end
        i += 1

        # Safety cap: max 12 buckets
        if i >= 12:
            break

    return buckets


@dataclass
class PeriodMetrics:
    """Performance metrics for a specific time period."""
    period_name: str        # e.g., "1d", "3d", "1w", etc.
    period_days: int        # Days covered by this period
    has_data: bool          # Whether the data spans this period
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = None
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    profit_factor: float = None
    max_drawdown: float = 0.0      # Max drawdown £ in period
    max_drawdown_pct: float = 0.0  # Max drawdown % in period


# Auto-scaling configuration - USE ALL AVAILABLE RESOURCES
def get_system_resources() -> Dict:
    """
    Detect system resources (CPU, memory) using psutil for cross-platform support.
    Works on Linux, macOS, Windows, and inside Docker containers.
    """
    resources = {
        'cpu_cores': psutil.cpu_count(logical=True) or os.cpu_count() or 4,
        'cpu_cores_physical': psutil.cpu_count(logical=False) or 4,
        'memory_gb': 4.0,
        'memory_available_gb': 4.0,
        'container_memory_limit_gb': None,
        'is_container': False,
        'cpu_percent': 0.0
    }

    try:
        # Get memory info using psutil (cross-platform)
        mem = psutil.virtual_memory()
        resources['memory_gb'] = mem.total / (1024**3)
        resources['memory_available_gb'] = mem.available / (1024**3)
        resources['memory_percent_used'] = mem.percent

        # Get current CPU usage
        resources['cpu_percent'] = psutil.cpu_percent(interval=0.1)

        # Check for Docker/container memory limits (Linux cgroups)
        cgroup_paths = [
            '/sys/fs/cgroup/memory/memory.limit_in_bytes',  # cgroup v1
            '/sys/fs/cgroup/memory.max',  # cgroup v2
        ]
        for cgroup_path in cgroup_paths:
            if os.path.exists(cgroup_path):
                try:
                    with open(cgroup_path, 'r') as f:
                        content = f.read().strip()
                        if content != 'max':
                            limit_bytes = int(content)
                            # If limit is less than host memory, we're in a container
                            if limit_bytes < mem.total:
                                resources['container_memory_limit_gb'] = limit_bytes / (1024**3)
                                resources['is_container'] = True
                                break
                except (ValueError, IOError):
                    pass

    except Exception as e:
        print(f"Warning: Could not detect system resources: {e}")

    return resources


def get_optimal_workers() -> Tuple[int, Dict]:
    """
    Determine optimal number of workers - USE ALL AVAILABLE RESOURCES.

    Aggressive scaling for maximum performance:
    - Use 100% of logical CPU cores (hyperthreading helps for I/O-bound work)
    - Each worker needs ~300MB RAM (backtesting is mostly CPU-bound)
    - No artificial caps - use what's available
    - Minimum 2 workers for parallelism
    """
    resources = get_system_resources()

    # Use ALL logical CPU cores (including hyperthreading)
    cpu_workers = resources['cpu_cores']

    # Memory-based limit (~300MB per worker - backtesting is CPU-bound, not memory-heavy)
    mem_per_worker_gb = 0.3

    # Use container limit if available, otherwise system available memory
    if resources['container_memory_limit_gb']:
        available_mem = resources['container_memory_limit_gb']
    else:
        available_mem = resources['memory_available_gb']

    # Reserve 2GB for OS/system, use the rest
    usable_mem = max(1.0, available_mem - 2.0)
    mem_workers = max(2, int(usable_mem / mem_per_worker_gb))

    # Take the minimum of CPU and memory limits
    optimal = min(cpu_workers, mem_workers)

    # Minimum 2 workers, NO MAXIMUM CAP - use all available resources
    optimal = max(2, optimal)

    resources['cpu_based_workers'] = cpu_workers
    resources['memory_based_workers'] = mem_workers
    resources['optimal_workers'] = optimal
    resources['mem_per_worker_gb'] = mem_per_worker_gb
    resources['usable_memory_gb'] = usable_mem

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

    # Period-based performance metrics
    period_metrics: Dict = None      # Dict[str, PeriodMetrics]
    consistency_score: float = 0.0   # Performance consistency across time periods

    # GBP conversion fields (for USD source data)
    total_pnl_gbp: float = 0.0           # Total P&L in GBP
    max_drawdown_gbp: float = 0.0        # Max drawdown in GBP
    avg_trade_gbp: float = 0.0           # Average trade P&L in GBP
    equity_curve_gbp: List[float] = None # Equity curve in GBP
    source_currency: str = "USD"         # Currency of source data
    display_currencies: List[str] = None # Currencies to display (e.g., ["USD", "GBP"])

    # Bidirectional trading fields (direction can be 'long', 'short', or 'both')
    long_trades: int = 0                 # Number of long trades (for bidirectional)
    long_wins: int = 0                   # Long trade wins
    long_pnl: float = 0.0                # Long trades P&L
    short_trades: int = 0                # Number of short trades (for bidirectional)
    short_wins: int = 0                  # Short trade wins
    short_pnl: float = 0.0               # Short trades P&L
    flip_count: int = 0                  # Number of position flips (for bidirectional)

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
        if self.equity_curve_gbp is None:
            self.equity_curve_gbp = []
        if self.display_currencies is None:
            self.display_currencies = ["USD", "GBP"] if self.source_currency == "USD" else [self.source_currency]

        # Calculate period metrics first (needed for consistency score)
        self.period_metrics = self._calculate_period_metrics()

        # Calculate consistency score
        self.consistency_score = self._calculate_consistency_score()

        # Calculate composite score (includes consistency and max drawdown)
        self._calculate_composite_score()

    def _calculate_period_metrics(self) -> Dict:
        """
        Calculate NON-OVERLAPPING period metrics that auto-scale based on data span.

        Key changes from previous implementation:
        1. Includes ALL trades (even those with bar_ timestamps)
        2. Uses non-overlapping time buckets (trades sum to total)
        3. Auto-scales periods based on data span (daily for weeks, weekly for months, etc.)
        """
        from datetime import timedelta

        if not self.trades_list:
            return {}

        # === Get actual data date range from DataFrame ===
        data_start = None
        data_end = None
        if hasattr(self, 'df') and self.df is not None and len(self.df) > 0:
            if 'time' in self.df.columns:
                try:
                    data_start = pd.to_datetime(self.df['time'].iloc[0])
                    data_end = pd.to_datetime(self.df['time'].iloc[-1])
                except Exception:
                    pass
            elif self.df.index.name == 'time' or isinstance(self.df.index, pd.DatetimeIndex):
                try:
                    data_start = pd.to_datetime(self.df.index[0])
                    data_end = pd.to_datetime(self.df.index[-1])
                except Exception:
                    pass

        # === STEP 1: Parse ALL trade exit times (including bar_ timestamps) ===
        trades_with_times = []
        trades_with_bar_times = []  # Trades with bar_ timestamps (need estimation)

        for trade in self.trades_list:
            exit_time_str = trade.get('exit_time', '')

            if not exit_time_str:
                # No exit time at all - use trade_num for ordering
                trades_with_bar_times.append({**trade, '_bar_idx': trade.get('trade_num', 0)})
                continue

            exit_str = str(exit_time_str)

            if exit_str.startswith('bar_'):
                # Extract bar index for later estimation
                try:
                    bar_idx = int(exit_str.split('_')[1])
                    trades_with_bar_times.append({**trade, '_bar_idx': bar_idx})
                except (ValueError, IndexError):
                    trades_with_bar_times.append({**trade, '_bar_idx': trade.get('trade_num', 0)})
            else:
                # Parse datetime string
                try:
                    exit_str = exit_str.replace(' ', 'T')
                    if 'T' in exit_str:
                        exit_dt = datetime.fromisoformat(exit_str.split('+')[0].split('Z')[0])
                    else:
                        exit_dt = datetime.strptime(exit_str, '%Y-%m-%d %H:%M:%S')
                    trades_with_times.append({**trade, '_exit_dt': exit_dt})
                except Exception:
                    # Failed to parse - treat as bar_ trade
                    trades_with_bar_times.append({**trade, '_bar_idx': trade.get('trade_num', 0)})

        # === STEP 2: Estimate times for bar_ trades ===
        if trades_with_bar_times:
            if trades_with_times:
                # We have some valid timestamps - estimate bar duration
                sorted_valid = sorted(trades_with_times, key=lambda x: x['_exit_dt'])
                earliest_valid = sorted_valid[0]['_exit_dt']
                latest_valid = sorted_valid[-1]['_exit_dt']

                # Get bar indices from valid trades to estimate duration
                valid_with_idx = [t for t in trades_with_times if 'trade_num' in t]
                if len(valid_with_idx) >= 2:
                    # Estimate time per bar based on known data
                    time_span = (latest_valid - earliest_valid).total_seconds()
                    idx_span = max(t.get('trade_num', 0) for t in valid_with_idx) - min(t.get('trade_num', 0) for t in valid_with_idx)
                    if idx_span > 0:
                        seconds_per_trade = time_span / idx_span
                    else:
                        seconds_per_trade = 3600  # Default 1 hour
                else:
                    seconds_per_trade = 3600  # Default 1 hour

                # Estimate times for bar_ trades
                min_idx = min(t['_bar_idx'] for t in trades_with_bar_times)
                for trade in trades_with_bar_times:
                    offset_idx = trade['_bar_idx'] - min_idx
                    estimated_dt = earliest_valid + timedelta(seconds=offset_idx * seconds_per_trade)
                    trade['_exit_dt'] = estimated_dt
                    trades_with_times.append(trade)
            else:
                # No valid timestamps at all - use actual data date range from DataFrame
                # This ensures period labels match the selected backtest period
                num_trades = len(trades_with_bar_times)

                # Use actual data range if available, otherwise fallback to 7 days
                if data_start is not None and data_end is not None:
                    base_dt = data_start
                    span_seconds = (data_end - data_start).total_seconds()
                    # Distribute trades evenly across the actual data span
                    seconds_per_trade = span_seconds / max(num_trades - 1, 1) if num_trades > 1 else 0
                    for i, trade in enumerate(sorted(trades_with_bar_times, key=lambda x: x['_bar_idx'])):
                        trade['_exit_dt'] = base_dt + timedelta(seconds=i * seconds_per_trade)
                        trades_with_times.append(trade)
                else:
                    # Fallback: no date info available
                    base_dt = datetime.now() - timedelta(days=7)
                    for i, trade in enumerate(sorted(trades_with_bar_times, key=lambda x: x['_bar_idx'])):
                        trade['_exit_dt'] = base_dt + timedelta(hours=i * 12)
                        trades_with_times.append(trade)

        if not trades_with_times:
            return {}

        # === STEP 3: Generate dynamic non-overlapping buckets ===
        # Use FULL DATA DATE RANGE (not trade dates) for proper period scaling
        # This ensures a 1-month dataset shows W1-W4, not D1-D4 just because trades clustered
        if data_start and data_end:
            earliest = data_start
            latest = data_end
        else:
            # Fallback to trade dates if no data range available
            earliest = min(t['_exit_dt'] for t in trades_with_times)
            latest = max(t['_exit_dt'] for t in trades_with_times)

        buckets = _generate_period_buckets(earliest, latest)

        # === STEP 4: Assign each trade to exactly ONE bucket ===
        bucket_trades = {label: [] for label, _, _ in buckets}

        for trade in trades_with_times:
            exit_dt = trade['_exit_dt']
            for label, bucket_start, bucket_end in buckets:
                if bucket_start <= exit_dt < bucket_end:
                    bucket_trades[label].append(trade)
                    break
            else:
                # Trade falls outside all buckets - assign to last bucket
                if buckets:
                    bucket_trades[buckets[-1][0]].append(trade)

        # === STEP 5: Calculate metrics for each bucket ===
        period_metrics = {}
        data_span_days = (latest - earliest).days + 1

        for label, bucket_start, bucket_end in buckets:
            period_trades = bucket_trades[label]
            period_days = (bucket_end - bucket_start).days

            if not period_trades:
                continue  # Skip empty periods

            wins = [t for t in period_trades if t.get('result') == 'WIN']
            losses = [t for t in period_trades if t.get('result') == 'LOSS']
            total = len(period_trades)

            total_pnl = sum(t.get('pnl', 0) for t in period_trades)
            total_pnl_pct = sum(t.get('pnl_pct', 0) for t in period_trades)

            gross_profit = sum(t.get('pnl', 0) for t in wins) if wins else 0
            gross_loss = abs(sum(t.get('pnl', 0) for t in losses)) if losses else 0.001
            pf = gross_profit / gross_loss if gross_loss > 0.001 else (10 if gross_profit > 0 else 0)

            # Calculate max drawdown for this period
            # DD% is calculated relative to average position size (not peak PnL)
            sorted_trades = sorted(period_trades, key=lambda x: x['_exit_dt'])
            running_pnl = 0
            peak_pnl = 0
            max_dd = 0
            max_dd_pct = 0

            # Get average position size for this period (for meaningful DD%)
            position_sizes = [t.get('position_size', 0) for t in period_trades if t.get('position_size', 0) > 0]
            avg_position_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0

            for t in sorted_trades:
                running_pnl += t.get('pnl', 0)
                if running_pnl > peak_pnl:
                    peak_pnl = running_pnl
                drawdown = peak_pnl - running_pnl
                if drawdown > max_dd:
                    max_dd = drawdown
                    # Calculate DD% relative to position size (more meaningful)
                    if avg_position_size > 0:
                        max_dd_pct = (drawdown / avg_position_size) * 100
                    elif peak_pnl > 0:
                        max_dd_pct = (drawdown / peak_pnl) * 100

            period_metrics[label] = PeriodMetrics(
                period_name=label,
                period_days=period_days,
                has_data=True,
                total_trades=total,
                wins=len(wins),
                losses=len(losses),
                win_rate=round(len(wins) / total * 100, 1) if total > 0 else None,
                total_pnl=round(total_pnl, 2),
                total_pnl_pct=round(total_pnl_pct, 2),
                profit_factor=round(pf, 2) if total > 0 else None,
                max_drawdown=round(max_dd, 2),
                max_drawdown_pct=round(max_dd_pct, 2)
            )

        return period_metrics

    def _calculate_consistency_score(self) -> float:
        """
        Calculate a score for performance consistency across time periods.

        UPDATED: Made less punitive to avoid over-penalizing normal trading variability.

        Factors:
        - Win Rate Consistency (30%): Low standard deviation of win rates
        - Profitability Consistency (30%): % of periods that are profitable
        - Max Drawdown Score (25%): Lower max drawdowns = better
        - Trade Distribution (15%): Trades spread evenly vs clustered
        """
        if not self.period_metrics:
            return 50.0  # Neutral score if no period data

        valid_periods = [pm for pm in self.period_metrics.values()
                        if pm.has_data and pm.total_trades >= 3]

        if len(valid_periods) < 2:
            return 50.0  # Need at least 2 periods to measure consistency

        # 1. Win Rate Consistency (low standard deviation = consistent)
        # UPDATED: Reduced penalty multiplier from 4 to 2 for less punitive scoring
        win_rates = [pm.win_rate for pm in valid_periods if pm.win_rate is not None]
        if len(win_rates) >= 2:
            import statistics
            wr_mean = statistics.mean(win_rates)
            wr_stdev = statistics.stdev(win_rates)
            # Lower stdev = higher score (reduced penalty: 0-50 stdev maps to 100-0)
            wr_consistency = max(0, 100 - (wr_stdev * 2))
        else:
            wr_consistency = 50

        # 2. Profitability Consistency (how many periods are profitable)
        # UPDATED: 60%+ profitable periods is good (was requiring near 100%)
        profitable_periods = sum(1 for pm in valid_periods if pm.total_pnl is not None and pm.total_pnl > 0)
        profit_ratio = profitable_periods / len(valid_periods)
        # Scale so 60%+ profitable = 100 points, 40% = 66 points, 20% = 33 points
        if profit_ratio >= 0.6:
            pnl_consistency = 100
        else:
            pnl_consistency = (profit_ratio / 0.6) * 100

        # 3. Max Drawdown Score (lower is better)
        # UPDATED: More lenient thresholds
        max_dds = [pm.max_drawdown_pct for pm in valid_periods if pm.max_drawdown_pct is not None]
        if max_dds:
            avg_dd = sum(max_dds) / len(max_dds)
            # DD < 10%: 100, DD 10-20%: 80, DD 20-30%: 60, DD 30-40%: 40, DD > 40%: 20
            if avg_dd < 10:
                dd_score = 100
            elif avg_dd < 20:
                dd_score = 80
            elif avg_dd < 30:
                dd_score = 60
            elif avg_dd < 40:
                dd_score = 40
            else:
                dd_score = 20
        else:
            dd_score = 50

        # 4. Trade Distribution (trades per period - lower variance = more consistent)
        # UPDATED: Reduced penalty multiplier from 50 to 30 for less punitive scoring
        trade_counts = [pm.total_trades for pm in valid_periods]
        if len(trade_counts) >= 2:
            import statistics
            tc_mean = statistics.mean(trade_counts)
            tc_stdev = statistics.stdev(trade_counts)
            tc_cv = tc_stdev / tc_mean if tc_mean > 0 else 1
            # Lower CV = higher score (reduced penalty)
            distribution_score = max(0, 100 - (tc_cv * 30))
        else:
            distribution_score = 50

        # Combined consistency score
        return round(
            wr_consistency * 0.30 +
            pnl_consistency * 0.30 +
            dd_score * 0.25 +
            distribution_score * 0.15,
            1
        )

    def _calculate_composite_score(self):
        """
        Calculate a balanced composite score.

        Weights:
        - Win Rate: 25% (high win rate = reliable)
        - Profit Factor: 15% (risk-adjusted returns)
        - Total Return %: 15% (actual profitability)
        - Trade Count: 10% (statistical significance)
        - Max Drawdown: 15% (lower drawdown = better)
        - Consistency: 20% (stable performance across periods)
        """
        # Win Rate Score (0-100): Target 60%+, penalize below 40%
        win_rate = self.win_rate if self.win_rate is not None else 0
        if win_rate >= 60:
            wr_score = 100
        elif win_rate >= 50:
            wr_score = 60 + (win_rate - 50) * 4
        elif win_rate >= 40:
            wr_score = 40 + (win_rate - 40) * 2
        else:
            wr_score = max(0, win_rate)

        # Profit Factor Score (0-100): Target PF > 1.5
        profit_factor = self.profit_factor if self.profit_factor is not None else 0
        if profit_factor >= 2.0:
            pf_score = 100
        elif profit_factor >= 1.5:
            pf_score = 70 + (profit_factor - 1.5) * 60
        elif profit_factor >= 1.0:
            pf_score = 30 + (profit_factor - 1.0) * 80
        else:
            pf_score = max(0, profit_factor * 30)

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

        # Max Drawdown Score (0-100): Lower is better
        dd_pct = self.max_drawdown_percent if self.max_drawdown_percent else 0
        if dd_pct < 5:
            dd_score = 100
        elif dd_pct < 10:
            dd_score = 80
        elif dd_pct < 20:
            dd_score = 60
        elif dd_pct < 30:
            dd_score = 40
        else:
            dd_score = 20

        # Consistency Score (already calculated)
        consistency = self.consistency_score

        # Weighted composite - prioritizes capital preservation and edge detection
        # Win Rate (15%): Reduced - vanity metric, can profit at 30% WR with good R:R
        # Profit Factor (20%): Increased - better indicator of edge than win rate
        # Return % (10%): Reduced - returns meaningless without risk context
        # Trade Count (10%): Statistical significance
        # Max Drawdown (25%): Increased - capital preservation is #1 job
        # Consistency (20%): Filter out one-hit wonders
        base_score = (
            wr_score * 0.15 +
            pf_score * 0.20 +
            pnl_score * 0.10 +
            trades_score * 0.10 +
            dd_score * 0.25 +
            consistency * 0.20
        )

        # NOTE: Bidirectional bonus removed - was artificially inflating scores
        # All strategies should be evaluated on the same criteria without bonuses

        self.composite_score = round(base_score, 1)

    def calculate_trend_following_score(self, avg_winner_pct: float = 0, avg_loser_pct: float = 0,
                                        mfe_capture_ratio: float = 0) -> float:
        """
        Calculate score optimized for trend-following strategies.

        Key differences from standard scoring:
        - Lower win rate is acceptable (40% is fine if R:R is good)
        - Rewards large winners (MFE capture)
        - Penalizes cutting winners short
        - Values risk:reward ratio over win rate

        Weights:
        - Risk:Reward ratio: 30% (target 3:1 or better)
        - MFE Capture: 20% (how much of the trend was captured)
        - Profit Factor: 20% (overall profitability)
        - Win Rate: 10% (lower weight - trend strategies often have 40-50% WR)
        - Max Drawdown: 20% (capital preservation)

        Args:
            avg_winner_pct: Average winning trade percentage
            avg_loser_pct: Average losing trade percentage (positive number)
            mfe_capture_ratio: Ratio of actual profit to max favorable excursion

        Returns:
            Trend-following composite score (0-100)
        """
        # Risk:Reward Score (0-100): Target 3:1 or better
        if avg_loser_pct > 0:
            risk_reward = avg_winner_pct / avg_loser_pct
        else:
            risk_reward = avg_winner_pct if avg_winner_pct > 0 else 0

        if risk_reward >= 3.0:
            rr_score = 100
        elif risk_reward >= 2.0:
            rr_score = 70 + (risk_reward - 2.0) * 30
        elif risk_reward >= 1.5:
            rr_score = 40 + (risk_reward - 1.5) * 60
        elif risk_reward >= 1.0:
            rr_score = 20 + (risk_reward - 1.0) * 40
        else:
            rr_score = max(0, risk_reward * 20)

        # MFE Capture Score (0-100): How much of the move did we capture?
        mfe_score = min(100, mfe_capture_ratio * 100) if mfe_capture_ratio > 0 else 50

        # Profit Factor Score (same as standard)
        profit_factor = self.profit_factor if self.profit_factor is not None else 0
        if profit_factor >= 2.0:
            pf_score = 100
        elif profit_factor >= 1.5:
            pf_score = 70 + (profit_factor - 1.5) * 60
        elif profit_factor >= 1.0:
            pf_score = 30 + (profit_factor - 1.0) * 80
        else:
            pf_score = max(0, profit_factor * 30)

        # Win Rate Score (adjusted for trend strategies - 40% is acceptable)
        win_rate = self.win_rate if self.win_rate is not None else 0
        if win_rate >= 50:
            wr_score = 100
        elif win_rate >= 40:
            wr_score = 60 + (win_rate - 40) * 4
        elif win_rate >= 30:
            wr_score = 30 + (win_rate - 30) * 3
        else:
            wr_score = max(0, win_rate)

        # Max Drawdown Score (same as standard)
        dd_pct = self.max_drawdown_percent if self.max_drawdown_percent else 0
        if dd_pct < 10:
            dd_score = 100
        elif dd_pct < 20:
            dd_score = 70
        elif dd_pct < 30:
            dd_score = 40
        else:
            dd_score = max(0, 20 - (dd_pct - 30))

        # Trend-following composite weights
        trend_score = (
            rr_score * 0.30 +       # Risk:Reward is king for trend strategies
            mfe_score * 0.20 +      # Reward capturing the trend
            pf_score * 0.20 +       # Overall profitability
            wr_score * 0.10 +       # Win rate (lower weight)
            dd_score * 0.20         # Capital preservation
        )

        return round(trend_score, 1)


# =============================================================================
# PHASE 2: INDICATOR PARAMETER TUNING
# =============================================================================

# Default indicator parameters (used in Phase 1)
DEFAULT_INDICATOR_PARAMS = {
    'rsi_length': 14,
    'stoch_k': 14,
    'stoch_d': 3,
    'stoch_smooth': 3,
    'bb_length': 20,
    'bb_mult': 2.0,
    'atr_length': 14,
    'sma_fast': 9,
    'sma_slow': 18,
    'sma_20': 20,
    'sma_50': 50,
    'ema_fast': 9,
    'ema_slow': 21,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'willr_length': 14,
    'cci_length': 20,
    'adx_length': 14,
    'supertrend_factor': 3.0,
    'supertrend_atr': 10,
    'aroon_length': 14,
    'mom_length': 10,
    'roc_length': 9,
    'keltner_length': 20,
    'keltner_mult': 2.0,
    'keltner_atr': 10,
    'donchian_length': 20,
    'ichimoku_tenkan': 9,
    'ichimoku_kijun': 26,
    'ichimoku_senkou': 52,
    'uo_fast': 7,
    'uo_mid': 14,
    'uo_slow': 28,
    'chop_length': 14,
    'consecutive_bars': 3,
    # New indicator params for added strategies
    'mcginley_length': 14,
    'mcginley_k': 0.6,
    'hull_length': 20,
    'zlema_length': 20,
    'tsi_long': 25,
    'tsi_short': 13,
    'tsi_signal': 7,
    'fisher_length': 10,
    'ao_fast': 5,
    'ao_slow': 34,
    'chandelier_length': 22,
    'chandelier_mult': 3.0,
    'linreg_length': 50,
    'mfi_length': 14,
    'cmf_length': 20,
}

# Strategy to tunable parameters mapping
# Each strategy maps to the indicator params it uses + their test ranges
STRATEGY_PARAM_MAP = {
    # === MOMENTUM ===
    'rsi_extreme': {
        'params': ['rsi_length'],
        'ranges': {'rsi_length': [7, 10, 12, 14, 16, 20, 25]},
    },
    'rsi_cross_50': {
        'params': ['rsi_length'],
        'ranges': {'rsi_length': [7, 10, 12, 14, 16, 20, 25]},
    },
    'stoch_extreme': {
        'params': ['stoch_k', 'stoch_d', 'stoch_smooth'],
        'ranges': {
            'stoch_k': [10, 14, 20],
            'stoch_d': [3, 5, 7],
            'stoch_smooth': [3, 5],
        },
    },
    'williams_r': {
        'params': ['willr_length'],
        'ranges': {'willr_length': [10, 14, 20, 25]},
    },
    'cci_extreme': {
        'params': ['cci_length'],
        'ranges': {'cci_length': [14, 20, 25, 30]},
    },
    'momentum_zero': {
        'params': ['mom_length'],
        'ranges': {'mom_length': [5, 10, 14, 20]},
    },
    'roc_extreme': {
        'params': ['roc_length'],
        'ranges': {'roc_length': [5, 9, 12, 14]},
    },
    'uo_extreme': {
        'params': ['uo_fast', 'uo_mid', 'uo_slow'],
        'ranges': {
            'uo_fast': [5, 7, 9],
            'uo_mid': [10, 14, 18],
            'uo_slow': [21, 28, 35],
        },
    },

    # === MEAN REVERSION ===
    'bb_touch': {
        'params': ['bb_length', 'bb_mult'],
        'ranges': {
            'bb_length': [14, 18, 20, 25, 30],
            'bb_mult': [1.5, 2.0, 2.5, 3.0],
        },
    },
    'bb_squeeze_breakout': {
        'params': ['bb_length', 'bb_mult'],
        'ranges': {
            'bb_length': [14, 18, 20, 25],
            'bb_mult': [1.5, 2.0, 2.5],
        },
    },
    'price_vs_sma': {
        'params': ['sma_20'],
        'ranges': {'sma_20': [10, 15, 20, 25, 30]},
    },
    'vwap_bounce': {
        'params': [],  # VWAP has no length parameter
        'ranges': {},
    },

    # === TREND ===
    'ema_cross': {
        'params': ['ema_fast', 'ema_slow'],
        'ranges': {
            'ema_fast': [5, 7, 9, 12],
            'ema_slow': [15, 18, 21, 26, 30],
        },
    },
    'sma_cross': {
        'params': ['sma_fast', 'sma_slow'],
        'ranges': {
            'sma_fast': [5, 7, 9, 12],
            'sma_slow': [14, 18, 21, 26],
        },
    },
    'double_ema_cross': {
        'params': ['ema_fast', 'ema_slow'],
        'ranges': {
            'ema_fast': [8, 10, 12, 14],
            'ema_slow': [20, 24, 26, 30],
        },
    },
    'triple_ema': {
        'params': ['ema_fast', 'ema_slow', 'sma_50'],
        'ranges': {
            'ema_fast': [7, 9, 12],
            'ema_slow': [18, 21, 26],
            'sma_50': [40, 50, 60],
        },
    },
    'macd_cross': {
        'params': ['macd_fast', 'macd_slow', 'macd_signal'],
        'ranges': {
            'macd_fast': [8, 10, 12],
            'macd_slow': [20, 24, 26, 30],
            'macd_signal': [7, 9, 11],
        },
    },
    'price_above_sma': {
        'params': ['sma_20'],
        'ranges': {'sma_20': [10, 15, 20, 25, 30]},
    },
    'supertrend': {
        'params': ['supertrend_factor', 'supertrend_atr'],
        'ranges': {
            'supertrend_factor': [2.0, 2.5, 3.0, 3.5],
            'supertrend_atr': [7, 10, 14, 20],
        },
    },
    'adx_strong_trend': {
        'params': ['adx_length'],
        'ranges': {'adx_length': [10, 14, 20, 25]},
    },
    'psar_reversal': {
        'params': [],  # PSAR uses fixed params
        'ranges': {},
    },
    'aroon_cross': {
        'params': ['aroon_length'],
        'ranges': {'aroon_length': [10, 14, 20, 25]},
    },
    'donchian_breakout': {
        'params': ['donchian_length'],
        'ranges': {'donchian_length': [10, 15, 20, 25, 30]},
    },
    'ichimoku_cross': {
        'params': ['ichimoku_tenkan', 'ichimoku_kijun'],
        'ranges': {
            'ichimoku_tenkan': [7, 9, 12],
            'ichimoku_kijun': [20, 26, 30],
        },
    },
    'ichimoku_cloud': {
        'params': ['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou'],
        'ranges': {
            'ichimoku_tenkan': [7, 9, 12],
            'ichimoku_kijun': [20, 26, 30],
            'ichimoku_senkou': [44, 52, 60],
        },
    },

    # === PATTERN ===
    'consecutive_candles': {
        'params': ['consecutive_bars'],
        'ranges': {'consecutive_bars': [2, 3, 4, 5]},
    },
    'big_candle': {
        'params': ['atr_length'],
        'ranges': {'atr_length': [10, 14, 20]},
    },
    'doji_reversal': {
        'params': [],  # Pattern-based, no length
        'ranges': {},
    },
    'engulfing': {
        'params': [],  # Pattern-based, no length
        'ranges': {},
    },
    'inside_bar': {
        'params': [],  # Pattern-based, no length
        'ranges': {},
    },
    'outside_bar': {
        'params': [],  # Pattern-based, no length
        'ranges': {},
    },

    # === VOLATILITY ===
    'atr_breakout': {
        'params': ['atr_length'],
        'ranges': {'atr_length': [10, 14, 20, 25]},
    },
    'low_volatility_breakout': {
        'params': ['atr_length'],
        'ranges': {'atr_length': [10, 14, 20]},
    },
    'keltner_breakout': {
        'params': ['keltner_length', 'keltner_mult', 'keltner_atr'],
        'ranges': {
            'keltner_length': [15, 20, 25],
            'keltner_mult': [1.5, 2.0, 2.5],
            'keltner_atr': [7, 10, 14],
        },
    },
    'chop_trend': {
        'params': ['chop_length'],
        'ranges': {'chop_length': [10, 14, 20]},
    },

    # === PRICE ACTION ===
    'higher_low': {
        'params': [],  # Pure price action
        'ranges': {},
    },
    'support_resistance': {
        'params': ['sma_20'],  # Uses 20-bar lookback
        'ranges': {'sma_20': [10, 15, 20, 25, 30]},
    },

    # === BASELINE ===
    'always': {
        'params': [],  # No indicators
        'ranges': {},
    },

    # === DIVERGENCE ===
    'rsi_divergence': {
        'params': ['rsi_length'],
        'ranges': {'rsi_length': [10, 14, 20, 25]},
    },

    # === McGINLEY DYNAMIC STRATEGIES ===
    'mcginley_cross': {
        'params': ['mcginley_length', 'mcginley_k'],
        'ranges': {
            'mcginley_length': [10, 14, 20, 25],
            'mcginley_k': [0.4, 0.5, 0.6, 0.8],
        },
    },
    'mcginley_trend': {
        'params': ['mcginley_length', 'mcginley_k'],
        'ranges': {
            'mcginley_length': [10, 14, 20, 25],
            'mcginley_k': [0.4, 0.5, 0.6, 0.8],
        },
    },

    # === HULL MOVING AVERAGE ===
    'hull_ma_cross': {
        'params': ['hull_length'],
        'ranges': {'hull_length': [14, 20, 25, 30]},
    },
    'hull_ma_turn': {
        'params': ['hull_length'],
        'ranges': {'hull_length': [14, 20, 25, 30]},
    },

    # === ZLEMA ===
    'zlema_cross': {
        'params': ['zlema_length'],
        'ranges': {'zlema_length': [14, 20, 25, 30]},
    },

    # === CHANDELIER ===
    'chandelier_entry': {
        'params': ['chandelier_length', 'chandelier_mult'],
        'ranges': {
            'chandelier_length': [14, 20, 22, 25],
            'chandelier_mult': [2.0, 2.5, 3.0, 3.5],
        },
    },

    # === TSI ===
    'tsi_cross': {
        'params': ['tsi_long', 'tsi_short', 'tsi_signal'],
        'ranges': {
            'tsi_long': [20, 25, 30],
            'tsi_short': [10, 13, 15],
            'tsi_signal': [5, 7, 9],
        },
    },
    'tsi_zero': {
        'params': ['tsi_long', 'tsi_short'],
        'ranges': {
            'tsi_long': [20, 25, 30],
            'tsi_short': [10, 13, 15],
        },
    },

    # === CMF ===
    'cmf_cross': {
        'params': ['cmf_length'],
        'ranges': {'cmf_length': [14, 20, 25]},
    },

    # === MFI ===
    'mfi_extreme': {
        'params': ['mfi_length'],
        'ranges': {'mfi_length': [10, 14, 20]},
    },

    # === PPO ===
    'ppo_cross': {
        'params': [],  # Uses MACD periods (12, 26, 9)
        'ranges': {},
    },

    # === FISHER ===
    'fisher_cross': {
        'params': ['fisher_length'],
        'ranges': {'fisher_length': [8, 10, 12, 14]},
    },

    # === AWESOME OSCILLATOR ===
    'ao_zero_cross': {
        'params': ['ao_fast', 'ao_slow'],
        'ranges': {
            'ao_fast': [3, 5, 7],
            'ao_slow': [25, 34, 40],
        },
    },
    'ao_twin_peaks': {
        'params': ['ao_fast', 'ao_slow'],
        'ranges': {
            'ao_fast': [3, 5, 7],
            'ao_slow': [25, 34, 40],
        },
    },

    # === SQUEEZE MOMENTUM ===
    'squeeze_momentum': {
        'params': ['bb_length', 'bb_mult'],
        'ranges': {
            'bb_length': [15, 20, 25],
            'bb_mult': [1.5, 2.0, 2.5],
        },
    },

    # === LINEAR REGRESSION ===
    'linreg_channel': {
        'params': ['linreg_length'],
        'ranges': {'linreg_length': [30, 50, 75, 100]},
    },

    # === COMBO STRATEGIES ===
    'rsi_macd_combo': {
        'params': ['rsi_length'],
        'ranges': {'rsi_length': [10, 14, 20]},
    },
    'bb_rsi_combo': {
        'params': ['bb_length', 'rsi_length'],
        'ranges': {
            'bb_length': [15, 20, 25],
            'rsi_length': [10, 14, 20],
        },
    },
    'supertrend_adx_combo': {
        'params': ['supertrend_factor', 'adx_length'],
        'ranges': {
            'supertrend_factor': [2.0, 2.5, 3.0],
            'adx_length': [10, 14, 20],
        },
    },
    'ema_rsi_combo': {
        'params': ['ema_fast', 'ema_slow', 'rsi_length'],
        'ranges': {
            'ema_fast': [7, 9, 12],
            'ema_slow': [18, 21, 26],
            'rsi_length': [10, 14, 20],
        },
    },
    'macd_stoch_combo': {
        'params': ['macd_fast', 'macd_slow'],
        'ranges': {
            'macd_fast': [8, 12, 15],
            'macd_slow': [21, 26, 30],
        },
    },

    # === OTHER NEW STRATEGIES ===
    'vwap_cross': {
        'params': [],  # VWAP has no params
        'ranges': {},
    },
    'pivot_bounce': {
        'params': [],  # Pivot has no params
        'ranges': {},
    },
    'obv_trend': {
        'params': [],  # OBV has no params
        'ranges': {},
    },
    'elder_ray': {
        'params': [],  # Uses EMA 13 (fixed)
        'ranges': {},
    },
}


@dataclass
class TunedResult:
    """Result of Phase 2 indicator tuning for a strategy."""
    # Original Phase 1 result
    original_result: StrategyResult

    # Tuned parameters
    tuned_params: Dict  # {param_name: tuned_value}
    default_params: Dict  # {param_name: default_value}

    # Before/after metrics
    before_score: float
    after_score: float
    before_win_rate: float
    after_win_rate: float
    before_profit_factor: float
    after_profit_factor: float
    before_pnl_percent: float
    after_pnl_percent: float

    # Improvement metrics (calculated in __post_init__)
    score_improvement: float = 0  # Percentage improvement in composite score
    win_rate_improvement: float = 0
    profit_factor_improvement: float = 0
    pnl_improvement: float = 0

    # The tuned backtest result
    tuned_result: StrategyResult = None

    # Was tuning beneficial?
    is_improved: bool = False

    # Did the parameters actually change from defaults?
    params_changed: bool = False

    def __post_init__(self):
        # Calculate improvements
        if self.before_score > 0:
            self.score_improvement = ((self.after_score - self.before_score) / self.before_score) * 100
        else:
            self.score_improvement = 0

        self.win_rate_improvement = self.after_win_rate - self.before_win_rate

        if self.before_profit_factor > 0:
            self.profit_factor_improvement = ((self.after_profit_factor - self.before_profit_factor) / self.before_profit_factor) * 100
        else:
            self.profit_factor_improvement = 0

        self.pnl_improvement = self.after_pnl_percent - self.before_pnl_percent

        # Tuning is considered beneficial if score improved
        self.is_improved = self.after_score > self.before_score

        # Check if parameters actually changed from defaults
        self.params_changed = self.tuned_params != self.default_params

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy_name': self.original_result.strategy_name,
            'entry_rule': self.original_result.entry_rule,
            'direction': self.original_result.direction,
            'tp_percent': self.original_result.tp_percent,
            'sl_percent': self.original_result.sl_percent,
            'tuned_params': self.tuned_params,
            'default_params': self.default_params,
            'before': {
                'score': round(self.before_score, 2),
                'win_rate': round(self.before_win_rate, 2),
                'profit_factor': round(self.before_profit_factor, 2),
                'pnl_percent': round(self.before_pnl_percent, 2),
            },
            'after': {
                'score': round(self.after_score, 2),
                'win_rate': round(self.after_win_rate, 2),
                'profit_factor': round(self.after_profit_factor, 2),
                'pnl_percent': round(self.after_pnl_percent, 2),
            },
            'improvements': {
                'score': round(self.score_improvement, 2),
                'win_rate': round(self.win_rate_improvement, 2),
                'profit_factor': round(self.profit_factor_improvement, 2),
                'pnl': round(self.pnl_improvement, 2),
            },
            'is_improved': self.is_improved,
            'params_changed': self.params_changed,
        }


# =============================================================================
# EXIT STRATEGY HELPERS
# =============================================================================

def calculate_trailing_stop(current_stop: float, high: float, low: float,
                           atr: float, atr_mult: float, direction: str) -> float:
    """
    Calculate trailing stop that ratchets with price movement.
    Only moves in favorable direction - never backwards.

    Args:
        current_stop: Current trailing stop price (or None for initial)
        high: Current bar's high price
        low: Current bar's low price
        atr: Current ATR value
        atr_mult: ATR multiplier for stop distance
        direction: 'long' or 'short'

    Returns:
        New trailing stop price
    """
    if direction == 'long':
        # For longs: stop trails below price, ratchets UP
        new_stop = high - (atr * atr_mult)
        if current_stop is None or new_stop > current_stop:
            return new_stop
        return current_stop
    else:  # short
        # For shorts: stop trails above price, ratchets DOWN
        new_stop = low + (atr * atr_mult)
        if current_stop is None or new_stop < current_stop:
            return new_stop
        return current_stop


def check_indicator_exit(df: pd.DataFrame, bar_idx: int, indicator: str,
                         direction: str, params: dict = None) -> bool:
    """
    Check if an indicator-based exit signal has triggered.

    Args:
        df: DataFrame with indicator columns
        bar_idx: Current bar index
        indicator: Exit indicator type ('supertrend', 'ema_cross', 'psar', 'mcginley')
        direction: Current position direction ('long' or 'short')
        params: Optional indicator parameters

    Returns:
        True if exit signal triggered, False otherwise
    """
    params = params or {}
    row = df.iloc[bar_idx]
    prev_row = df.iloc[bar_idx - 1] if bar_idx > 0 else row

    if indicator == 'supertrend':
        # Exit when Supertrend direction flips
        # Supertrend: -1 = bearish (price below), 1 = bullish (price above)
        if 'supertrend_direction' in df.columns:
            current_dir = row.get('supertrend_direction', 0)
            if direction == 'long' and current_dir == -1:
                return True  # Supertrend turned bearish - exit long
            elif direction == 'short' and current_dir == 1:
                return True  # Supertrend turned bullish - exit short

    elif indicator == 'ema_cross':
        # Exit when fast EMA crosses slow EMA against position
        fast_col = f"ema_{params.get('fast', 9)}"
        slow_col = f"ema_{params.get('slow', 21)}"
        if fast_col in df.columns and slow_col in df.columns:
            fast_curr = row[fast_col]
            slow_curr = row[slow_col]
            fast_prev = prev_row[fast_col]
            slow_prev = prev_row[slow_col]
            if direction == 'long':
                # Exit long when fast crosses below slow
                if fast_prev >= slow_prev and fast_curr < slow_curr:
                    return True
            else:  # short
                # Exit short when fast crosses above slow
                if fast_prev <= slow_prev and fast_curr > slow_curr:
                    return True

    elif indicator == 'psar':
        # Exit when Parabolic SAR flips
        if 'psar' in df.columns:
            psar = row['psar']
            close = row['close']
            if direction == 'long' and psar > close:
                return True  # SAR above price = bearish
            elif direction == 'short' and psar < close:
                return True  # SAR below price = bullish

    elif indicator == 'mcginley':
        # Exit when McGinley Dynamic changes direction
        if 'mcginley' in df.columns:
            mcg_curr = row['mcginley']
            mcg_prev = prev_row['mcginley']
            if direction == 'long':
                # Exit long when McGinley turns from rising to falling
                if mcg_prev > df.iloc[bar_idx - 2]['mcginley'] if bar_idx > 1 else False:
                    if mcg_curr < mcg_prev:
                        return True
            else:  # short
                # Exit short when McGinley turns from falling to rising
                if mcg_prev < df.iloc[bar_idx - 2]['mcginley'] if bar_idx > 1 else False:
                    if mcg_curr > mcg_prev:
                        return True

    elif indicator == 'adx_di':
        # Exit when ADX weakens or DI cross reverses
        if 'adx' in df.columns and 'di_plus' in df.columns and 'di_minus' in df.columns:
            adx = row['adx']
            di_plus = row['di_plus']
            di_minus = row['di_minus']
            threshold = params.get('adx_threshold', 20)
            if adx < threshold:
                return True  # Trend weakening
            if direction == 'long' and di_minus > di_plus:
                return True  # DI flipped bearish
            elif direction == 'short' and di_plus > di_minus:
                return True  # DI flipped bullish

    return False


def calculate_mfe_capture_ratio(pnl_pct: float, run_up_pct: float) -> float:
    """
    Calculate how much of the maximum favorable excursion was captured.

    MFE Capture Ratio = Actual P&L / Max Favorable Excursion

    A ratio of 1.0 means we captured all of the best possible exit.
    A ratio of 0.5 means we only captured half of the potential.

    Returns 0.0 if run_up_pct is 0 or negative.
    """
    if run_up_pct <= 0:
        return 0.0
    # Can be > 1.0 if we happened to exit at the absolute peak
    return min(pnl_pct / run_up_pct, 1.0) if run_up_pct > 0 else 0.0


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

        # === McGINLEY DYNAMIC STRATEGIES ===
        'mcginley_cross': {
            'name': 'McGinley Cross',
            'category': 'Trend',
            'description': 'Price crosses McGinley Dynamic indicator',
            'pool': 'indicator_exit'  # Trend following with indicator exit
        },
        'mcginley_trend': {
            'name': 'McGinley Trend Direction',
            'category': 'Trend',
            'description': 'McGinley Dynamic changes direction (slope)',
            'pool': 'indicator_exit'
        },

        # === HULL MOVING AVERAGE ===
        'hull_ma_cross': {
            'name': 'Hull MA Cross',
            'category': 'Trend',
            'description': 'Price crosses Hull Moving Average',
            'pool': 'indicator_exit'
        },
        'hull_ma_turn': {
            'name': 'Hull MA Direction',
            'category': 'Trend',
            'description': 'Hull MA changes direction',
            'pool': 'indicator_exit'
        },

        # === ZLEMA (Zero-Lag EMA) ===
        'zlema_cross': {
            'name': 'ZLEMA Cross',
            'category': 'Trend',
            'description': 'Price crosses Zero-Lag EMA',
            'pool': 'indicator_exit'
        },

        # === CHANDELIER EXIT ===
        'chandelier_entry': {
            'name': 'Chandelier Entry',
            'category': 'Volatility',
            'description': 'Enter on Chandelier Exit signal',
            'pool': 'indicator_exit'
        },

        # === TSI (True Strength Index) ===
        'tsi_cross': {
            'name': 'TSI Cross',
            'category': 'Momentum',
            'description': 'TSI crosses signal line'
        },
        'tsi_zero': {
            'name': 'TSI Zero Cross',
            'category': 'Momentum',
            'description': 'TSI crosses zero line'
        },

        # === CMF (Chaikin Money Flow) ===
        'cmf_cross': {
            'name': 'CMF Zero Cross',
            'category': 'Momentum',
            'description': 'Chaikin Money Flow crosses zero'
        },

        # === OBV (On Balance Volume) ===
        'obv_trend': {
            'name': 'OBV Trend',
            'category': 'Momentum',
            'description': 'OBV makes new high/low with price'
        },

        # === MFI (Money Flow Index) ===
        'mfi_extreme': {
            'name': 'MFI Extreme',
            'category': 'Momentum',
            'description': 'MFI < 20 (long) or > 80 (short)'
        },

        # === PPO (Percentage Price Oscillator) ===
        'ppo_cross': {
            'name': 'PPO Signal Cross',
            'category': 'Momentum',
            'description': 'PPO crosses signal line'
        },

        # === FISHER TRANSFORM ===
        'fisher_cross': {
            'name': 'Fisher Transform Cross',
            'category': 'Momentum',
            'description': 'Fisher crosses signal line'
        },

        # === SQUEEZE MOMENTUM ===
        'squeeze_momentum': {
            'name': 'Squeeze Momentum',
            'category': 'Volatility',
            'description': 'BB inside Keltner + momentum direction',
            'pool': 'indicator_exit'
        },

        # === VWAP STRATEGIES ===
        'vwap_cross': {
            'name': 'VWAP Cross',
            'category': 'Mean Reversion',
            'description': 'Price crosses VWAP'
        },

        # === PIVOT POINTS ===
        'pivot_bounce': {
            'name': 'Pivot Point Bounce',
            'category': 'Price Action',
            'description': 'Price bounces off pivot point levels'
        },

        # === LINEAR REGRESSION ===
        'linreg_channel': {
            'name': 'Linear Regression Channel',
            'category': 'Trend',
            'description': 'Price touches/breaks linear regression channel',
            'pool': 'indicator_exit'
        },

        # === AWESOME OSCILLATOR ===
        'ao_zero_cross': {
            'name': 'AO Zero Cross',
            'category': 'Momentum',
            'description': 'Awesome Oscillator crosses zero'
        },
        'ao_twin_peaks': {
            'name': 'AO Twin Peaks',
            'category': 'Momentum',
            'description': 'Awesome Oscillator twin peaks pattern'
        },

        # === ELDER RAY ===
        'elder_ray': {
            'name': 'Elder Ray',
            'category': 'Momentum',
            'description': 'Bull/Bear power with EMA trend filter'
        },

        # === COMBO STRATEGIES ===
        'rsi_macd_combo': {
            'name': 'RSI + MACD Combo',
            'category': 'Momentum',
            'description': 'RSI extreme + MACD confirmation'
        },
        'bb_rsi_combo': {
            'name': 'BB + RSI Combo',
            'category': 'Mean Reversion',
            'description': 'Bollinger Band touch + RSI extreme'
        },
        'supertrend_adx_combo': {
            'name': 'Supertrend + ADX Combo',
            'category': 'Trend',
            'description': 'Supertrend signal + ADX > 25 filter',
            'pool': 'indicator_exit'
        },
        'ema_rsi_combo': {
            'name': 'EMA Cross + RSI Combo',
            'category': 'Trend',
            'description': 'EMA cross + RSI confirmation'
        },
        'macd_stoch_combo': {
            'name': 'MACD + Stochastic Combo',
            'category': 'Momentum',
            'description': 'MACD cross + Stochastic confirmation'
        },
    }

    def __init__(self, df: pd.DataFrame, status_callback: Dict = None,
                 streaming_callback: Callable = None,
                 capital: float = 1000.0,
                 position_size_pct: float = 100.0,
                 calc_engine: str = "tradingview",
                 progress_min: int = 0,
                 progress_max: int = 100,
                 source_currency: str = "USD",
                 fx_fetcher=None):
        self.df = df.copy()
        self.status = status_callback or {}
        self.streaming_callback = streaming_callback
        self.db = get_strategy_db() if HAS_DB else None

        # Store trading parameters from UI
        self.capital = capital
        self.position_size_pct = position_size_pct

        # Store calculation engine for indicator calculations
        self.calc_engine = calc_engine

        # Currency conversion parameters
        self.source_currency = source_currency
        self.fx_fetcher = fx_fetcher

        # Progress range for multi-engine mode (e.g., 0-33, 33-66, 66-100)
        self.progress_min = progress_min
        self.progress_max = progress_max

        # Calculate Buy & Hold benchmark
        self.buy_hold_return = self._calculate_buy_hold()

        # Extract data date range for database storage
        self.data_start = None
        self.data_end = None
        if len(df) > 0:
            try:
                # Check for 'time' column first (most common)
                if 'time' in df.columns:
                    self.data_start = str(pd.to_datetime(df['time'].iloc[0]))
                    self.data_end = str(pd.to_datetime(df['time'].iloc[-1]))
                # Check if index is DatetimeIndex
                elif isinstance(df.index, pd.DatetimeIndex):
                    self.data_start = str(df.index[0])
                    self.data_end = str(df.index[-1])
                # Check for 'datetime' or 'date' column
                elif 'datetime' in df.columns:
                    self.data_start = str(pd.to_datetime(df['datetime'].iloc[0]))
                    self.data_end = str(pd.to_datetime(df['datetime'].iloc[-1]))
                elif 'date' in df.columns:
                    self.data_start = str(pd.to_datetime(df['date'].iloc[0]))
                    self.data_end = str(pd.to_datetime(df['date'].iloc[-1]))
            except Exception as e:
                print(f"Warning: Could not extract date range: {e}")
                pass

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

    # =========================================================================
    # POOL CLASSIFICATION AND EXIT CONFIG
    # =========================================================================

    # Categories that naturally fit the TP/SL pool (mean reversion, fixed targets)
    TP_SL_CATEGORIES = {'Mean Reversion', 'Momentum', 'Pattern', 'Baseline'}

    # Categories that naturally fit the indicator exit pool (trend following)
    INDICATOR_EXIT_CATEGORIES = {'Trend', 'Volatility', 'Price Action'}

    @classmethod
    def get_strategy_pool(cls, strategy_name: str) -> str:
        """
        Determine which pool a strategy belongs to based on its category.

        Returns:
            'tp_sl' - Strategy best suited for fixed TP/SL exits
            'indicator_exit' - Strategy best suited for indicator-based exits
        """
        strategy_info = cls.ENTRY_STRATEGIES.get(strategy_name, {})
        category = strategy_info.get('category', 'Unknown')

        # Check explicit pool override first
        if 'pool' in strategy_info:
            return strategy_info['pool']

        # Otherwise classify by category
        if category in cls.TP_SL_CATEGORIES:
            return 'tp_sl'
        elif category in cls.INDICATOR_EXIT_CATEGORIES:
            return 'indicator_exit'
        else:
            # Default to TP/SL for unknown categories (safer default)
            return 'tp_sl'

    @classmethod
    def get_default_exit_config(cls, strategy_name: str, tp_percent: float = 2.0,
                                sl_percent: float = 1.0) -> 'ExitConfig':
        """
        Get the default exit configuration for a strategy based on its pool.

        Args:
            strategy_name: Name of the entry strategy
            tp_percent: Take profit percentage (for TP/SL pool)
            sl_percent: Stop loss percentage (for TP/SL pool)

        Returns:
            ExitConfig with appropriate defaults
        """
        pool = cls.get_strategy_pool(strategy_name)
        strategy_info = cls.ENTRY_STRATEGIES.get(strategy_name, {})

        if pool == 'tp_sl':
            return ExitConfig(
                exit_type='fixed_tp_sl',
                tp_percent=tp_percent,
                sl_percent=sl_percent,
                pool='tp_sl'
            )
        else:  # indicator_exit pool
            # Determine exit indicator based on strategy type
            category = strategy_info.get('category', 'Trend')

            if strategy_name in ['supertrend', 'supertrend_rider']:
                exit_indicator = 'supertrend'
            elif strategy_name in ['psar_reversal', 'parabolic_sar_trend']:
                exit_indicator = 'psar'
            elif strategy_name in ['ema_cross', 'double_ema_cross', 'triple_ema']:
                exit_indicator = 'ema_cross'
            elif strategy_name in ['adx_strong_trend', 'adx_di_trend']:
                exit_indicator = 'adx_di'
            elif strategy_name.startswith('mcginley'):
                exit_indicator = 'mcginley'
            else:
                # Default to trailing stop for trend strategies without specific indicator
                return ExitConfig(
                    exit_type='trailing_stop',
                    trailing_atr_mult=2.0,
                    use_protection_sl=True,
                    protection_sl_atr_mult=4.0,
                    pool='indicator_exit'
                )

            return ExitConfig(
                exit_type='indicator_exit',
                exit_indicator=exit_indicator,
                use_protection_sl=True,
                protection_sl_atr_mult=4.0,
                pool='indicator_exit'
            )

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
                # Serialize period metrics for JSON
                period_metrics_dict = {}
                if result.period_metrics:
                    for period_name, pm in result.period_metrics.items():
                        if pm.has_data and pm.total_trades > 0:
                            period_metrics_dict[period_name] = {
                                'trades': pm.total_trades,
                                'wins': pm.wins,
                                'losses': pm.losses,
                                'win_rate': pm.win_rate,
                                'pnl': pm.total_pnl,
                                'pnl_pct': pm.total_pnl_pct,
                                'pf': pm.profit_factor,
                                'max_dd': pm.max_drawdown,
                                'max_dd_pct': pm.max_drawdown_pct
                            }

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
                    # Period-based metrics
                    'period_metrics': period_metrics_dict,
                    'consistency_score': result.consistency_score,
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

        # === McGINLEY DYNAMIC ===
        # Pre-calculate for mcginley_cross, mcginley_trend strategies and indicator exits
        # Formula: MD = MD_prev + (Close - MD_prev) / (k * n * (Close/MD_prev)^4)
        try:
            length = 14
            k = 0.6
            close = df['close'].values
            n = len(close)
            md = np.full(n, np.nan)
            md[0] = close[0]
            for i in range(1, n):
                if md[i-1] == 0 or np.isnan(md[i-1]):
                    md[i] = close[i]
                else:
                    ratio = close[i] / md[i-1]
                    ratio = max(0.5, min(ratio, 2.0))  # Clamp ratio for stability
                    divisor = k * length * (ratio ** 4)
                    divisor = max(divisor, 0.001)  # Prevent division by zero
                    md[i] = md[i-1] + (close[i] - md[i-1]) / divisor
            df['mcginley'] = md
        except Exception:
            df['mcginley'] = df['ema_21']  # Fallback to EMA21

        # === CANDLE PROPERTIES ===
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['green'] = df['close'] > df['open']
        df['red'] = df['close'] < df['open']
        df['doji'] = df['body'] < df['range'] * 0.1

        # Price changes
        df['pct_change'] = df['close'].pct_change() * 100

        # Sanitize all indicator columns to prevent None comparison errors
        self._sanitize_indicators()

        print(f"Indicators ready (pandas-ta). {len(df)} bars, RSI: {df['rsi'].min():.1f}-{df['rsi'].max():.1f}")

    def _sanitize_indicators(self):
        """
        Sanitize all DataFrame columns to prevent 'None < float' comparison errors.
        Converts all None values to NaN and ensures numeric columns are float type.
        """
        self._sanitize_df(self.df)

    def _safe_col(self, col_name: str, default_value=np.nan) -> pd.Series:
        """
        Safely get a DataFrame column, converting None to NaN.
        Returns a series filled with default_value if column doesn't exist.
        """
        if col_name not in self.df.columns:
            return pd.Series(default_value, index=self.df.index)
        series = self.df[col_name]
        # Convert None to NaN and ensure numeric
        return pd.to_numeric(series, errors='coerce')

    def _get_signals(self, strategy: str, direction: str) -> pd.Series:
        """Get entry signals for a strategy. Simple and direct."""
        df = self.df

        # Helper to ensure boolean series with no NaN values
        def safe_bool(series):
            return series.fillna(False).astype(bool)

        # Helper to safely get a column with NaN instead of None
        def safe_col(col_name):
            if col_name not in df.columns:
                return pd.Series(np.nan, index=df.index)
            return pd.to_numeric(df[col_name], errors='coerce')

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
            # TradingView MovingAvg2Line Cross: SMA(fast) crosses SMA(slow)
            # Uses sma_fast and sma_slow columns if available (for tuning), else calculate
            if 'sma_fast' in df.columns and 'sma_slow' in df.columns:
                sma_fast = safe_col('sma_fast')
                sma_slow = safe_col('sma_slow')
            else:
                # Default to 9/18 for initial calculation
                sma_fast = df['close'].rolling(9).mean()
                sma_slow = df['close'].rolling(18).mean()
            if direction == 'long':
                return safe_bool((sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1)))
            else:
                return safe_bool((sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1)))

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
            # if (ups >= consecutiveBarsUp) - long entry after N UP closes
            # if (dns >= consecutiveBarsDown) - short entry after N DOWN closes
            # Note: This is consecutive CLOSES moving up/down, NOT green/red candles

            # Use tuned consecutive_bars from DataFrame column if available, else default to 3
            consecutive_bars = 3  # default
            if 'consecutive_bars' in df.columns:
                # Get the first non-null value (it's the same for all rows)
                consecutive_bars = int(df['consecutive_bars'].dropna().iloc[0]) if not df['consecutive_bars'].dropna().empty else 3

            up_close = df['close'] > df['close'].shift(1)
            down_close = df['close'] < df['close'].shift(1)

            # Count consecutive occurrences
            ups = up_close.astype(int).groupby((~up_close).cumsum()).cumsum()
            dns = down_close.astype(int).groupby((~down_close).cumsum()).cumsum()

            if direction == 'long':
                # Long after N+ consecutive up closes
                return safe_bool(ups >= consecutive_bars)
            else:
                # Short after N+ consecutive down closes
                return safe_bool(dns >= consecutive_bars)

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
            # ADAPTIVE: Uses 25th percentile of ATR to identify low volatility periods
            # This works across any pair/timeframe by adapting to the data's volatility profile
            atr_threshold = df['atr'].quantile(0.25)  # Bottom 25% = low volatility
            low_vol = df['atr'] < atr_threshold
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
            # Handle case where vwap might have None/NaN values
            if 'vwap' not in df.columns or df['vwap'].isna().all():
                return pd.Series(False, index=df.index)
            vwap = df['vwap'].ffill().fillna(df['close'])
            if direction == 'long':
                # Price touched below VWAP and bounced back above
                touched_below = df['low'] < vwap
                closed_above = df['close'] > vwap
                return safe_bool(touched_below & closed_above)
            else:
                # Price touched above VWAP and rejected
                touched_above = df['high'] > vwap
                closed_below = df['close'] < vwap
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
            # Uses CROSSOVER/CROSSUNDER logic to match TradingView Pine Script exactly:
            # Pine: breakoutUp = close > dcUpper and close[1] <= dcUpper[1]
            # Pine: breakoutDn = close < dcLower and close[1] >= dcLower[1]
            if direction == 'long':
                # Long: price CROSSES above upper Donchian channel
                # Current close > previous dc_upper AND previous close <= 2-bars-ago dc_upper
                return safe_bool(
                    (df['close'] > df['dc_upper'].shift(1)) &
                    (df['close'].shift(1) <= df['dc_upper'].shift(2))
                )
            else:
                # Short: price CROSSES below lower Donchian channel
                # Current close < previous dc_lower AND previous close >= 2-bars-ago dc_lower
                return safe_bool(
                    (df['close'] < df['dc_lower'].shift(1)) &
                    (df['close'].shift(1) >= df['dc_lower'].shift(2))
                )

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
            # Rate of Change extreme values - ADAPTIVE to any pair/timeframe
            # Uses 5th/95th percentile to identify extremes relative to the data
            roc_lower = df['roc'].quantile(0.05)  # Bottom 5% = oversold
            roc_upper = df['roc'].quantile(0.95)  # Top 5% = overbought
            if direction == 'long':
                # Long: ROC in bottom 5th percentile (oversold)
                return safe_bool(df['roc'] < roc_lower)
            else:
                # Short: ROC in top 95th percentile (overbought)
                return safe_bool(df['roc'] > roc_upper)

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

        # === McGINLEY DYNAMIC STRATEGIES ===
        elif strategy == 'mcginley_cross':
            # Price crosses McGinley Dynamic
            if 'mcginley' not in df.columns:
                # Calculate McGinley Dynamic inline
                # Formula: MD = MD_prev + (Close - MD_prev) / (k * n * (Close/MD_prev)^4)
                length = 14
                k = 0.6
                close = df['close'].values
                n = len(close)
                md = np.full(n, np.nan)
                md[0] = close[0]
                for i in range(1, n):
                    if md[i-1] == 0 or np.isnan(md[i-1]):
                        md[i] = close[i]
                    else:
                        ratio = close[i] / md[i-1]
                        ratio = max(0.5, min(ratio, 2.0))  # Clamp ratio for stability
                        divisor = k * length * (ratio ** 4)
                        divisor = max(divisor, 0.001)  # Prevent division by zero
                        md[i] = md[i-1] + (close[i] - md[i-1]) / divisor
                df['mcginley'] = md
            if direction == 'long':
                return safe_bool((df['close'] > df['mcginley']) & (df['close'].shift(1) <= df['mcginley'].shift(1)))
            else:
                return safe_bool((df['close'] < df['mcginley']) & (df['close'].shift(1) >= df['mcginley'].shift(1)))

        elif strategy == 'mcginley_trend':
            # McGinley changes direction (slope)
            if 'mcginley' not in df.columns:
                # Calculate McGinley Dynamic inline
                length = 14
                k = 0.6
                close = df['close'].values
                n = len(close)
                md = np.full(n, np.nan)
                md[0] = close[0]
                for i in range(1, n):
                    if md[i-1] == 0 or np.isnan(md[i-1]):
                        md[i] = close[i]
                    else:
                        ratio = close[i] / md[i-1]
                        ratio = max(0.5, min(ratio, 2.0))
                        divisor = k * length * (ratio ** 4)
                        divisor = max(divisor, 0.001)
                        md[i] = md[i-1] + (close[i] - md[i-1]) / divisor
                df['mcginley'] = md
            mcg_slope = df['mcginley'] - df['mcginley'].shift(1)
            mcg_slope_prev = df['mcginley'].shift(1) - df['mcginley'].shift(2)
            if direction == 'long':
                # Slope turns positive
                return safe_bool((mcg_slope > 0) & (mcg_slope_prev <= 0))
            else:
                # Slope turns negative
                return safe_bool((mcg_slope < 0) & (mcg_slope_prev >= 0))

        # === HULL MOVING AVERAGE ===
        elif strategy == 'hull_ma_cross':
            # Price crosses Hull MA
            if 'hull_ma' not in df.columns:
                # Calculate Hull MA if not present: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
                period = 20
                half_period = period // 2
                sqrt_period = int(np.sqrt(period))
                wma1 = df['close'].rolling(half_period).apply(lambda x: np.sum(x * np.arange(1, half_period+1)) / np.sum(np.arange(1, half_period+1)), raw=True)
                wma2 = df['close'].rolling(period).apply(lambda x: np.sum(x * np.arange(1, period+1)) / np.sum(np.arange(1, period+1)), raw=True)
                raw_hma = 2 * wma1 - wma2
                df['hull_ma'] = raw_hma.rolling(sqrt_period).apply(lambda x: np.sum(x * np.arange(1, sqrt_period+1)) / np.sum(np.arange(1, sqrt_period+1)), raw=True)
            if direction == 'long':
                return safe_bool((df['close'] > df['hull_ma']) & (df['close'].shift(1) <= df['hull_ma'].shift(1)))
            else:
                return safe_bool((df['close'] < df['hull_ma']) & (df['close'].shift(1) >= df['hull_ma'].shift(1)))

        elif strategy == 'hull_ma_turn':
            # Hull MA changes direction
            if 'hull_ma' not in df.columns:
                period = 20
                half_period = period // 2
                sqrt_period = int(np.sqrt(period))
                wma1 = df['close'].rolling(half_period).apply(lambda x: np.sum(x * np.arange(1, half_period+1)) / np.sum(np.arange(1, half_period+1)), raw=True)
                wma2 = df['close'].rolling(period).apply(lambda x: np.sum(x * np.arange(1, period+1)) / np.sum(np.arange(1, period+1)), raw=True)
                raw_hma = 2 * wma1 - wma2
                df['hull_ma'] = raw_hma.rolling(sqrt_period).apply(lambda x: np.sum(x * np.arange(1, sqrt_period+1)) / np.sum(np.arange(1, sqrt_period+1)), raw=True)
            hull_slope = df['hull_ma'] - df['hull_ma'].shift(1)
            hull_slope_prev = df['hull_ma'].shift(1) - df['hull_ma'].shift(2)
            if direction == 'long':
                return safe_bool((hull_slope > 0) & (hull_slope_prev <= 0))
            else:
                return safe_bool((hull_slope < 0) & (hull_slope_prev >= 0))

        # === ZLEMA (Zero-Lag EMA) ===
        elif strategy == 'zlema_cross':
            # Price crosses Zero-Lag EMA
            period = 20
            lag = (period - 1) // 2
            ema_data = df['close'] + (df['close'] - df['close'].shift(lag))
            zlema = ema_data.ewm(span=period, adjust=False).mean()
            if direction == 'long':
                return safe_bool((df['close'] > zlema) & (df['close'].shift(1) <= zlema.shift(1)))
            else:
                return safe_bool((df['close'] < zlema) & (df['close'].shift(1) >= zlema.shift(1)))

        # === CHANDELIER EXIT ===
        elif strategy == 'chandelier_entry':
            # Chandelier Exit signal
            period = 22
            mult = 3.0
            highest_high = df['high'].rolling(period).max()
            lowest_low = df['low'].rolling(period).min()
            chandelier_long = highest_high - df['atr'] * mult
            chandelier_short = lowest_low + df['atr'] * mult
            if direction == 'long':
                # Price crosses above chandelier long stop
                return safe_bool((df['close'] > chandelier_long) & (df['close'].shift(1) <= chandelier_long.shift(1)))
            else:
                # Price crosses below chandelier short stop
                return safe_bool((df['close'] < chandelier_short) & (df['close'].shift(1) >= chandelier_short.shift(1)))

        # === TSI (True Strength Index) ===
        elif strategy == 'tsi_cross':
            # TSI crosses signal line
            if 'tsi' not in df.columns:
                # Calculate TSI
                close_diff = df['close'].diff()
                double_smooth_pc = close_diff.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
                double_smooth_apc = close_diff.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
                df['tsi'] = 100 * (double_smooth_pc / double_smooth_apc.replace(0, np.nan))
                df['tsi_signal'] = df['tsi'].ewm(span=7, adjust=False).mean()
            if direction == 'long':
                return safe_bool((df['tsi'] > df['tsi_signal']) & (df['tsi'].shift(1) <= df['tsi_signal'].shift(1)))
            else:
                return safe_bool((df['tsi'] < df['tsi_signal']) & (df['tsi'].shift(1) >= df['tsi_signal'].shift(1)))

        elif strategy == 'tsi_zero':
            # TSI crosses zero
            if 'tsi' not in df.columns:
                close_diff = df['close'].diff()
                double_smooth_pc = close_diff.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
                double_smooth_apc = close_diff.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
                df['tsi'] = 100 * (double_smooth_pc / double_smooth_apc.replace(0, np.nan))
            if direction == 'long':
                return safe_bool((df['tsi'] > 0) & (df['tsi'].shift(1) <= 0))
            else:
                return safe_bool((df['tsi'] < 0) & (df['tsi'].shift(1) >= 0))

        # === CMF (Chaikin Money Flow) ===
        elif strategy == 'cmf_cross':
            # CMF crosses zero
            if 'cmf' not in df.columns:
                mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
                mfv = mfm * df['volume']
                df['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
            if direction == 'long':
                return safe_bool((df['cmf'] > 0) & (df['cmf'].shift(1) <= 0))
            else:
                return safe_bool((df['cmf'] < 0) & (df['cmf'].shift(1) >= 0))

        # === OBV (On Balance Volume) ===
        elif strategy == 'obv_trend':
            # OBV makes new high/low with price
            if 'obv' not in df.columns:
                obv_change = np.where(df['close'] > df['close'].shift(1), df['volume'],
                             np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
                df['obv'] = pd.Series(obv_change, index=df.index).cumsum()
            lookback = 20
            obv_high = df['obv'].rolling(lookback).max()
            obv_low = df['obv'].rolling(lookback).min()
            price_high = df['close'].rolling(lookback).max()
            price_low = df['close'].rolling(lookback).min()
            if direction == 'long':
                # OBV new high with price near highs
                return safe_bool((df['obv'] == obv_high) & (df['close'] >= price_high * 0.98))
            else:
                # OBV new low with price near lows
                return safe_bool((df['obv'] == obv_low) & (df['close'] <= price_low * 1.02))

        # === MFI (Money Flow Index) ===
        elif strategy == 'mfi_extreme':
            if 'mfi' not in df.columns:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                raw_mf = typical_price * df['volume']
                positive_mf = np.where(typical_price > typical_price.shift(1), raw_mf, 0)
                negative_mf = np.where(typical_price < typical_price.shift(1), raw_mf, 0)
                positive_mf_sum = pd.Series(positive_mf, index=df.index).rolling(14).sum()
                negative_mf_sum = pd.Series(negative_mf, index=df.index).rolling(14).sum()
                mfi_ratio = positive_mf_sum / negative_mf_sum.replace(0, np.nan)
                df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            if direction == 'long':
                return safe_bool((df['mfi'] > 20) & (df['mfi'].shift(1) <= 20))
            else:
                return safe_bool((df['mfi'] < 80) & (df['mfi'].shift(1) >= 80))

        # === PPO (Percentage Price Oscillator) ===
        elif strategy == 'ppo_cross':
            if 'ppo' not in df.columns:
                ema_fast = df['close'].ewm(span=12, adjust=False).mean()
                ema_slow = df['close'].ewm(span=26, adjust=False).mean()
                df['ppo'] = ((ema_fast - ema_slow) / ema_slow) * 100
                df['ppo_signal'] = df['ppo'].ewm(span=9, adjust=False).mean()
            if direction == 'long':
                return safe_bool((df['ppo'] > df['ppo_signal']) & (df['ppo'].shift(1) <= df['ppo_signal'].shift(1)))
            else:
                return safe_bool((df['ppo'] < df['ppo_signal']) & (df['ppo'].shift(1) >= df['ppo_signal'].shift(1)))

        # === FISHER TRANSFORM ===
        elif strategy == 'fisher_cross':
            if 'fisher' not in df.columns:
                period = 10
                hl2 = (df['high'] + df['low']) / 2
                max_high = hl2.rolling(period).max()
                min_low = hl2.rolling(period).min()
                value = 2 * ((hl2 - min_low) / (max_high - min_low).replace(0, np.nan)) - 1
                value = value.clip(-0.999, 0.999)
                df['fisher'] = (np.log((1 + value) / (1 - value)) / 2).ewm(span=1, adjust=False).mean()
                df['fisher_signal'] = df['fisher'].shift(1)
            if direction == 'long':
                return safe_bool((df['fisher'] > df['fisher_signal']) & (df['fisher'].shift(1) <= df['fisher_signal'].shift(1)))
            else:
                return safe_bool((df['fisher'] < df['fisher_signal']) & (df['fisher'].shift(1) >= df['fisher_signal'].shift(1)))

        # === SQUEEZE MOMENTUM ===
        elif strategy == 'squeeze_momentum':
            # BB inside Keltner + momentum direction
            bb_len, bb_mult = 20, 2.0
            kc_len, kc_mult = 20, 1.5
            sma = df['close'].rolling(bb_len).mean()
            std = df['close'].rolling(bb_len).std()
            bb_upper = sma + bb_mult * std
            bb_lower = sma - bb_mult * std
            kc_upper = sma + kc_mult * df['atr']
            kc_lower = sma - kc_mult * df['atr']
            squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
            squeeze_off = ~squeeze_on
            squeeze_fired = squeeze_off & squeeze_on.shift(1)
            mom = df['close'] - df['close'].rolling(20).mean()
            if direction == 'long':
                return safe_bool(squeeze_fired & (mom > 0))
            else:
                return safe_bool(squeeze_fired & (mom < 0))

        # === VWAP CROSS ===
        elif strategy == 'vwap_cross':
            if 'vwap' not in df.columns:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            if direction == 'long':
                return safe_bool((df['close'] > df['vwap']) & (df['close'].shift(1) <= df['vwap'].shift(1)))
            else:
                return safe_bool((df['close'] < df['vwap']) & (df['close'].shift(1) >= df['vwap'].shift(1)))

        # === PIVOT BOUNCE ===
        elif strategy == 'pivot_bounce':
            # Simple pivot calculation
            pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
            r1 = 2 * pivot - df['low'].shift(1)
            s1 = 2 * pivot - df['high'].shift(1)
            if direction == 'long':
                # Price bounces off S1
                near_s1 = (df['low'] <= s1 * 1.005) & (df['low'] >= s1 * 0.995)
                return safe_bool(near_s1 & (df['close'] > df['open']))
            else:
                # Price bounces off R1
                near_r1 = (df['high'] >= r1 * 0.995) & (df['high'] <= r1 * 1.005)
                return safe_bool(near_r1 & (df['close'] < df['open']))

        # === LINEAR REGRESSION CHANNEL ===
        elif strategy == 'linreg_channel':
            period = 50
            x = np.arange(period)
            def linreg_series(data):
                if len(data) < period:
                    return np.nan
                slope, intercept = np.polyfit(x, data[-period:], 1)
                return intercept + slope * (period - 1)
            linreg = df['close'].rolling(period).apply(linreg_series, raw=True)
            std = df['close'].rolling(period).std()
            upper = linreg + 2 * std
            lower = linreg - 2 * std
            if direction == 'long':
                return safe_bool((df['close'] > lower) & (df['close'].shift(1) <= lower.shift(1)))
            else:
                return safe_bool((df['close'] < upper) & (df['close'].shift(1) >= upper.shift(1)))

        # === AWESOME OSCILLATOR ===
        elif strategy == 'ao_zero_cross':
            if 'ao' not in df.columns:
                hl2 = (df['high'] + df['low']) / 2
                df['ao'] = hl2.rolling(5).mean() - hl2.rolling(34).mean()
            if direction == 'long':
                return safe_bool((df['ao'] > 0) & (df['ao'].shift(1) <= 0))
            else:
                return safe_bool((df['ao'] < 0) & (df['ao'].shift(1) >= 0))

        elif strategy == 'ao_twin_peaks':
            # AO twin peaks pattern (simplified)
            if 'ao' not in df.columns:
                hl2 = (df['high'] + df['low']) / 2
                df['ao'] = hl2.rolling(5).mean() - hl2.rolling(34).mean()
            lookback = 20
            if direction == 'long':
                # Two lows below zero, second higher than first, AO turning up
                ao_low = df['ao'].rolling(lookback).min()
                ao_rising = df['ao'] > df['ao'].shift(1)
                return safe_bool((df['ao'] < 0) & (df['ao'] > ao_low) & ao_rising)
            else:
                # Two highs above zero, second lower than first, AO turning down
                ao_high = df['ao'].rolling(lookback).max()
                ao_falling = df['ao'] < df['ao'].shift(1)
                return safe_bool((df['ao'] > 0) & (df['ao'] < ao_high) & ao_falling)

        # === ELDER RAY ===
        elif strategy == 'elder_ray':
            ema_13 = df['close'].ewm(span=13, adjust=False).mean()
            bull_power = df['high'] - ema_13
            bear_power = df['low'] - ema_13
            if direction == 'long':
                # EMA rising, bear power negative but rising
                ema_rising = ema_13 > ema_13.shift(1)
                bear_rising = bear_power > bear_power.shift(1)
                return safe_bool(ema_rising & (bear_power < 0) & bear_rising)
            else:
                # EMA falling, bull power positive but falling
                ema_falling = ema_13 < ema_13.shift(1)
                bull_falling = bull_power < bull_power.shift(1)
                return safe_bool(ema_falling & (bull_power > 0) & bull_falling)

        # === COMBO STRATEGIES ===
        elif strategy == 'rsi_macd_combo':
            # RSI extreme + MACD confirmation
            histogram = df['macd'] - df['macd_signal']
            if direction == 'long':
                rsi_oversold = df['rsi'] < 30
                macd_bullish = histogram > histogram.shift(1)
                return safe_bool(rsi_oversold & macd_bullish)
            else:
                rsi_overbought = df['rsi'] > 70
                macd_bearish = histogram < histogram.shift(1)
                return safe_bool(rsi_overbought & macd_bearish)

        elif strategy == 'bb_rsi_combo':
            # BB touch + RSI extreme
            if direction == 'long':
                bb_touch = df['close'] <= df['bb_lower']
                rsi_oversold = df['rsi'] < 35
                return safe_bool(bb_touch & rsi_oversold)
            else:
                bb_touch = df['close'] >= df['bb_upper']
                rsi_overbought = df['rsi'] > 65
                return safe_bool(bb_touch & rsi_overbought)

        elif strategy == 'supertrend_adx_combo':
            # Supertrend signal + ADX > 25 filter
            direction_change = df['supertrend_dir'] - df['supertrend_dir'].shift(1)
            adx_strong = df['adx'] > 25
            if direction == 'long':
                return safe_bool((direction_change > 0) & adx_strong)
            else:
                return safe_bool((direction_change < 0) & adx_strong)

        elif strategy == 'ema_rsi_combo':
            # EMA cross + RSI confirmation
            ema_cross_up = (df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))
            ema_cross_down = (df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1))
            if direction == 'long':
                return safe_bool(ema_cross_up & (df['rsi'] > 50))
            else:
                return safe_bool(ema_cross_down & (df['rsi'] < 50))

        elif strategy == 'macd_stoch_combo':
            # MACD cross + Stochastic confirmation
            macd_cross_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            if direction == 'long':
                return safe_bool(macd_cross_up & (df['stoch_k'] < 50))
            else:
                return safe_bool(macd_cross_down & (df['stoch_k'] > 50))

        return pd.Series(False, index=df.index)

    def backtest(self, strategy: str, direction: str,
                 tp_percent: float, sl_percent: float,
                 initial_capital: float = 1000.0,
                 position_size_pct: float = 100.0,
                 commission_pct: float = 0.1,
                 source_currency: str = "USD",
                 fx_fetcher=None) -> StrategyResult:
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
            source_currency: Currency of source data ("USD" for BTCUSDT, "GBP" for BTCGBP)
            fx_fetcher: Exchange rate fetcher instance (optional, for USD->GBP conversion)
        """
        df = self.df
        signals = self._get_signals(strategy, direction)

        trades = []
        position = None
        equity = initial_capital
        equity_curve = [initial_capital]
        cumulative_pnl = 0.0
        trade_num = 0

        # GBP tracking (for USD source data conversion)
        needs_conversion = source_currency == "USD" and fx_fetcher is not None
        equity_gbp = initial_capital  # Start same, will diverge with exchange rate
        equity_curve_gbp = [initial_capital]
        cumulative_pnl_gbp = 0.0

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

                        # GBP conversion
                        exit_dt = pd.to_datetime(row['time']) if 'time' in row else None
                        usd_gbp_rate = fx_fetcher.get_rate_for_date(exit_dt) if needs_conversion else 1.0
                        pnl_gbp = pnl * usd_gbp_rate if needs_conversion else pnl
                        pos_size_gbp = pos_size * usd_gbp_rate if needs_conversion else pos_size
                        equity_gbp += pnl_gbp
                        cumulative_pnl_gbp += pnl_gbp

                        trades.append(TradeResult(
                            'long', entry, sl_price, pnl, loss_pct, 'sl',
                            trade_num=trade_num, entry_time=entry_time, exit_time=exit_time,
                            position_size=pos_size, position_qty=pos_qty,
                            run_up=run_up, run_up_pct=run_up_pct,
                            drawdown=dd, drawdown_pct=dd_pct,
                            cumulative_pnl=cumulative_pnl,
                            pnl_gbp=pnl_gbp, position_size_gbp=pos_size_gbp,
                            cumulative_pnl_gbp=cumulative_pnl_gbp, usd_gbp_rate=usd_gbp_rate
                        ))
                        equity_curve.append(equity)
                        equity_curve_gbp.append(equity_gbp)
                        position = None
                    elif row['high'] >= tp_price:
                        gain_pct = tp_percent
                        pnl = pos_size * (gain_pct / 100)
                        pnl -= pos_size * (commission_pct / 100) * 2  # Entry + Exit commission (TradingView applies per side)
                        equity += pnl
                        cumulative_pnl += pnl
                        trade_num += 1
                        exit_time = str(row['time']) if 'time' in row else f"bar_{i}"

                        # GBP conversion
                        exit_dt = pd.to_datetime(row['time']) if 'time' in row else None
                        usd_gbp_rate = fx_fetcher.get_rate_for_date(exit_dt) if needs_conversion else 1.0
                        pnl_gbp = pnl * usd_gbp_rate if needs_conversion else pnl
                        pos_size_gbp = pos_size * usd_gbp_rate if needs_conversion else pos_size
                        equity_gbp += pnl_gbp
                        cumulative_pnl_gbp += pnl_gbp

                        trades.append(TradeResult(
                            'long', entry, tp_price, pnl, gain_pct, 'tp',
                            trade_num=trade_num, entry_time=entry_time, exit_time=exit_time,
                            position_size=pos_size, position_qty=pos_qty,
                            run_up=run_up, run_up_pct=run_up_pct,
                            drawdown=dd, drawdown_pct=dd_pct,
                            cumulative_pnl=cumulative_pnl,
                            pnl_gbp=pnl_gbp, position_size_gbp=pos_size_gbp,
                            cumulative_pnl_gbp=cumulative_pnl_gbp, usd_gbp_rate=usd_gbp_rate
                        ))
                        equity_curve.append(equity)
                        equity_curve_gbp.append(equity_gbp)
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

                        # GBP conversion
                        exit_dt = pd.to_datetime(row['time']) if 'time' in row else None
                        usd_gbp_rate = fx_fetcher.get_rate_for_date(exit_dt) if needs_conversion else 1.0
                        pnl_gbp = pnl * usd_gbp_rate if needs_conversion else pnl
                        pos_size_gbp = pos_size * usd_gbp_rate if needs_conversion else pos_size
                        equity_gbp += pnl_gbp
                        cumulative_pnl_gbp += pnl_gbp

                        trades.append(TradeResult(
                            'short', entry, sl_price, pnl, loss_pct, 'sl',
                            trade_num=trade_num, entry_time=entry_time, exit_time=exit_time,
                            position_size=pos_size, position_qty=pos_qty,
                            run_up=run_up, run_up_pct=run_up_pct,
                            drawdown=dd, drawdown_pct=dd_pct,
                            cumulative_pnl=cumulative_pnl,
                            pnl_gbp=pnl_gbp, position_size_gbp=pos_size_gbp,
                            cumulative_pnl_gbp=cumulative_pnl_gbp, usd_gbp_rate=usd_gbp_rate
                        ))
                        equity_curve.append(equity)
                        equity_curve_gbp.append(equity_gbp)
                        position = None
                    elif row['low'] <= tp_price:
                        gain_pct = tp_percent
                        pnl = pos_size * (gain_pct / 100)
                        pnl -= pos_size * (commission_pct / 100) * 2  # Entry + Exit commission (TradingView applies per side)
                        equity += pnl
                        cumulative_pnl += pnl
                        trade_num += 1
                        exit_time = str(row['time']) if 'time' in row else f"bar_{i}"

                        # GBP conversion
                        exit_dt = pd.to_datetime(row['time']) if 'time' in row else None
                        usd_gbp_rate = fx_fetcher.get_rate_for_date(exit_dt) if needs_conversion else 1.0
                        pnl_gbp = pnl * usd_gbp_rate if needs_conversion else pnl
                        pos_size_gbp = pos_size * usd_gbp_rate if needs_conversion else pos_size
                        equity_gbp += pnl_gbp
                        cumulative_pnl_gbp += pnl_gbp

                        trades.append(TradeResult(
                            'short', entry, tp_price, pnl, gain_pct, 'tp',
                            trade_num=trade_num, entry_time=entry_time, exit_time=exit_time,
                            position_size=pos_size, position_qty=pos_qty,
                            run_up=run_up, run_up_pct=run_up_pct,
                            drawdown=dd, drawdown_pct=dd_pct,
                            cumulative_pnl=cumulative_pnl,
                            pnl_gbp=pnl_gbp, position_size_gbp=pos_size_gbp,
                            cumulative_pnl_gbp=cumulative_pnl_gbp, usd_gbp_rate=usd_gbp_rate
                        ))
                        equity_curve.append(equity)
                        equity_curve_gbp.append(equity_gbp)
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
                beats_buy_hold=False,
                # GBP fields (zeros for empty trades)
                total_pnl_gbp=0, max_drawdown_gbp=0, avg_trade_gbp=0,
                equity_curve_gbp=equity_curve_gbp,
                source_currency=source_currency,
                display_currencies=["USD", "GBP"] if needs_conversion else [source_currency]
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
            'position_size_gbp': round(t.position_size_gbp, 2),
            'position_qty': round(t.position_qty, 5),
            'pnl': round(t.pnl, 2),
            'pnl_gbp': round(t.pnl_gbp, 2),
            'pnl_pct': round(t.pnl_percent, 2),
            'run_up': round(t.run_up, 2),
            'run_up_pct': round(t.run_up_pct, 2),
            'drawdown': round(t.drawdown, 2),
            'drawdown_pct': round(t.drawdown_pct, 2),
            'cumulative_pnl': round(t.cumulative_pnl, 2),
            'cumulative_pnl_gbp': round(t.cumulative_pnl_gbp, 2),
            'usd_gbp_rate': round(t.usd_gbp_rate, 4),
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

        # Calculate GBP-converted summary metrics
        total_pnl_gbp = sum(t.pnl_gbp for t in trades) if trades else 0.0
        avg_trade_gbp = total_pnl_gbp / len(trades) if trades else 0.0

        # Max drawdown in GBP
        if equity_curve_gbp and len(equity_curve_gbp) > 1:
            equity_arr_gbp = np.array(equity_curve_gbp)
            peak_gbp = np.maximum.accumulate(equity_arr_gbp)
            drawdown_gbp = peak_gbp - equity_arr_gbp
            max_dd_gbp = drawdown_gbp.max()
        else:
            max_dd_gbp = 0.0

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
            open_position=open_position_data,
            # GBP conversion fields
            total_pnl_gbp=round(total_pnl_gbp, 2),
            max_drawdown_gbp=round(max_dd_gbp, 2),
            avg_trade_gbp=round(avg_trade_gbp, 2),
            equity_curve_gbp=equity_curve_gbp,
            source_currency=source_currency,
            display_currencies=["USD", "GBP"] if needs_conversion else [source_currency]
        )

    def backtest_bidirectional(self, strategy: str,
                               tp_percent: float, sl_percent: float,
                               initial_capital: float = 1000.0,
                               position_size_pct: float = 100.0,
                               commission_pct: float = 0.1,
                               source_currency: str = "USD",
                               fx_fetcher=None) -> StrategyResult:
        """
        Run bidirectional backtest - strategy can take both long AND short trades.

        Position handling: Flip-style (one position at a time).
        - An opposite signal closes current position and opens new one.
        - Conflicting signals (both fire) are skipped.

        Args:
            strategy: Entry strategy name
            tp_percent: Take profit percentage (same for both directions)
            sl_percent: Stop loss percentage (same for both directions)
            initial_capital: Starting capital (default £1000)
            position_size_pct: Position size as % of equity (default 75%)
            commission_pct: Commission per trade (default 0.1%)
            source_currency: Currency of source data ("USD" for BTCUSDT, "GBP" for BTCGBP)
            fx_fetcher: Exchange rate fetcher instance (optional, for USD->GBP conversion)
        """
        df = self.df

        # Get BOTH long and short signals upfront
        long_signals = self._get_signals(strategy, 'long')
        short_signals = self._get_signals(strategy, 'short')

        trades = []
        position = None  # None, or dict with 'direction' key
        equity = initial_capital
        equity_curve = [initial_capital]
        cumulative_pnl = 0.0
        trade_num = 0
        flip_count = 0

        # Per-direction tracking
        long_trade_count = 0
        long_win_count = 0
        long_pnl_total = 0.0
        short_trade_count = 0
        short_win_count = 0
        short_pnl_total = 0.0

        # GBP tracking (for USD source data conversion)
        needs_conversion = source_currency == "USD" and fx_fetcher is not None
        equity_gbp = initial_capital
        equity_curve_gbp = [initial_capital]
        cumulative_pnl_gbp = 0.0

        def close_position(row, exit_reason, exit_price_override=None):
            """Helper to close current position and record trade."""
            nonlocal position, equity, cumulative_pnl, trade_num
            nonlocal equity_gbp, cumulative_pnl_gbp
            nonlocal long_trade_count, long_win_count, long_pnl_total
            nonlocal short_trade_count, short_win_count, short_pnl_total

            if position is None:
                return None

            entry = position['entry_price']
            pos_size = position['position_size']
            entry_time = position['entry_time']
            pos_qty = position['position_qty']
            direction = position['direction']

            # Calculate exit price and P&L based on exit reason
            if exit_reason == 'tp':
                if direction == 'long':
                    exit_price = entry * (1 + tp_percent / 100)
                    pnl_pct = tp_percent
                else:  # short
                    exit_price = entry * (1 - tp_percent / 100)
                    pnl_pct = tp_percent
            elif exit_reason == 'sl':
                if direction == 'long':
                    exit_price = entry * (1 - sl_percent / 100)
                    pnl_pct = -sl_percent
                else:  # short
                    exit_price = entry * (1 + sl_percent / 100)
                    pnl_pct = -sl_percent
            else:  # flip - close at current price
                exit_price = exit_price_override or row['close']
                if direction == 'long':
                    pnl_pct = ((exit_price - entry) / entry) * 100
                else:  # short
                    pnl_pct = ((entry - exit_price) / entry) * 100

            pnl = pos_size * (pnl_pct / 100)
            pnl -= pos_size * (commission_pct / 100) * 2  # Entry + Exit commission
            equity += pnl
            cumulative_pnl += pnl
            trade_num += 1
            exit_time = str(row['time']) if 'time' in row else f"bar_{position['entry_bar']}"

            # Per-direction tracking
            if direction == 'long':
                long_trade_count += 1
                long_pnl_total += pnl
                if pnl > 0:
                    long_win_count += 1
            else:
                short_trade_count += 1
                short_pnl_total += pnl
                if pnl > 0:
                    short_win_count += 1

            # Run-up and drawdown
            if direction == 'long':
                run_up_pct = ((position['best_price'] - entry) / entry) * 100
                dd_pct = ((entry - position['worst_price']) / entry) * 100
            else:
                run_up_pct = ((entry - position['best_price']) / entry) * 100
                dd_pct = ((position['worst_price'] - entry) / entry) * 100
            run_up = pos_size * (run_up_pct / 100)
            dd = pos_size * (dd_pct / 100)

            # GBP conversion
            exit_dt = pd.to_datetime(row['time']) if 'time' in row else None
            usd_gbp_rate = fx_fetcher.get_rate_for_date(exit_dt) if needs_conversion else 1.0
            pnl_gbp = pnl * usd_gbp_rate if needs_conversion else pnl
            pos_size_gbp = pos_size * usd_gbp_rate if needs_conversion else pos_size
            equity_gbp += pnl_gbp
            cumulative_pnl_gbp += pnl_gbp

            trade = TradeResult(
                direction, entry, exit_price, pnl, pnl_pct, exit_reason,
                trade_num=trade_num, entry_time=entry_time, exit_time=exit_time,
                position_size=pos_size, position_qty=pos_qty,
                run_up=run_up, run_up_pct=run_up_pct,
                drawdown=dd, drawdown_pct=dd_pct,
                cumulative_pnl=cumulative_pnl,
                pnl_gbp=pnl_gbp, position_size_gbp=pos_size_gbp,
                cumulative_pnl_gbp=cumulative_pnl_gbp, usd_gbp_rate=usd_gbp_rate
            )
            trades.append(trade)
            equity_curve.append(equity)
            equity_curve_gbp.append(equity_gbp)
            position = None
            return trade

        def open_position(row, direction, bar_idx):
            """Helper to open a new position."""
            nonlocal position
            pos_size = equity * (position_size_pct / 100)
            entry_price = row['close']
            pos_qty = pos_size / entry_price
            entry_time = str(row['time']) if 'time' in row else f"bar_{bar_idx}"
            position = {
                'direction': direction,
                'entry_price': entry_price,
                'position_size': pos_size,
                'position_qty': pos_qty,
                'entry_bar': bar_idx,
                'entry_time': entry_time,
                'best_price': entry_price,
                'worst_price': entry_price
            }

        # Main loop
        for i in range(50, len(df)):
            row = df.iloc[i]

            # Track run-up/drawdown while in position
            if position is not None:
                entry = position['entry_price']
                if position['direction'] == 'long':
                    best_price = max(position.get('best_price', entry), row['high'])
                    worst_price = min(position.get('worst_price', entry), row['low'])
                else:  # short
                    best_price = min(position.get('best_price', entry), row['low'])
                    worst_price = max(position.get('worst_price', entry), row['high'])
                position['best_price'] = best_price
                position['worst_price'] = worst_price

            # Check TP/SL exits
            if position is not None:
                entry = position['entry_price']
                direction = position['direction']

                if direction == 'long':
                    tp_price = entry * (1 + tp_percent / 100)
                    sl_price = entry * (1 - sl_percent / 100)
                    # SL first (conservative)
                    if row['low'] <= sl_price:
                        close_position(row, 'sl')
                    elif row['high'] >= tp_price:
                        close_position(row, 'tp')
                else:  # short
                    tp_price = entry * (1 - tp_percent / 100)
                    sl_price = entry * (1 + sl_percent / 100)
                    if row['high'] >= sl_price:
                        close_position(row, 'sl')
                    elif row['low'] <= tp_price:
                        close_position(row, 'tp')

            # Get current signals
            long_sig = long_signals.iloc[i]
            short_sig = short_signals.iloc[i]

            # Skip conflicting signals
            if long_sig and short_sig:
                continue

            # Entry/Flip logic
            if equity > 0:
                if position is None:
                    # No position - check for new entry
                    if long_sig:
                        open_position(row, 'long', i)
                    elif short_sig:
                        open_position(row, 'short', i)
                else:
                    # In position - check for flip signal
                    current_dir = position['direction']
                    if current_dir == 'long' and short_sig:
                        # Flip from long to short
                        close_position(row, 'flip', row['close'])
                        flip_count += 1
                        if equity > 0:
                            open_position(row, 'short', i)
                    elif current_dir == 'short' and long_sig:
                        # Flip from short to long
                        close_position(row, 'flip', row['close'])
                        flip_count += 1
                        if equity > 0:
                            open_position(row, 'long', i)

        # Calculate metrics
        if not trades:
            return StrategyResult(
                strategy_name=f"{strategy}_both",
                strategy_category=self.ENTRY_STRATEGIES.get(strategy, {}).get('category', 'Unknown'),
                direction='both',
                tp_percent=tp_percent,
                sl_percent=sl_percent,
                entry_rule=strategy,
                total_trades=0, wins=0, losses=0,
                win_rate=0, total_pnl=0, total_pnl_percent=0,
                profit_factor=0, max_drawdown=0, max_drawdown_percent=0,
                avg_trade=0, avg_trade_percent=0,
                buy_hold_return=self.buy_hold_return,
                vs_buy_hold=-self.buy_hold_return,
                beats_buy_hold=False,
                total_pnl_gbp=0, max_drawdown_gbp=0, avg_trade_gbp=0,
                equity_curve_gbp=equity_curve_gbp,
                source_currency=source_currency,
                display_currencies=["USD", "GBP"] if needs_conversion else [source_currency],
                # Bidirectional fields
                long_trades=0, long_wins=0, long_pnl=0,
                short_trades=0, short_wins=0, short_pnl=0,
                flip_count=0
            )

        # Use exit_reason for consistent profit factor calculation (matches regular backtest)
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

        # Build trades list
        trades_list = [{
            'trade_num': t.trade_num,
            'direction': t.direction,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry': round(t.entry_price, 2),
            'exit': round(t.exit_price, 2),
            'position_size': round(t.position_size, 2),
            'position_size_gbp': round(t.position_size_gbp, 2),
            'position_qty': round(t.position_qty, 5),
            'pnl': round(t.pnl, 2),
            'pnl_gbp': round(t.pnl_gbp, 2),
            'pnl_pct': round(t.pnl_percent, 2),
            'run_up': round(t.run_up, 2),
            'run_up_pct': round(t.run_up_pct, 2),
            'drawdown': round(t.drawdown, 2),
            'drawdown_pct': round(t.drawdown_pct, 2),
            'cumulative_pnl': round(t.cumulative_pnl, 2),
            'cumulative_pnl_gbp': round(t.cumulative_pnl_gbp, 2),
            'usd_gbp_rate': round(t.usd_gbp_rate, 4),
            'result': 'WIN' if t.pnl > 0 else 'LOSS',
            'exit_reason': t.exit_reason
        } for t in trades]

        # Open position tracking
        open_position_data = None
        has_open = False
        if position is not None:
            has_open = True
            last_row = self.df.iloc[-1]
            current_price = last_row['close']
            entry_price = position['entry_price']
            pos_size = position['position_size']
            if position['direction'] == 'long':
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
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

        vs_bh = round(total_pnl_percent - self.buy_hold_return, 2)

        # GBP metrics
        total_pnl_gbp = sum(t.pnl_gbp for t in trades) if trades else 0.0
        avg_trade_gbp = total_pnl_gbp / len(trades) if trades else 0.0
        if equity_curve_gbp and len(equity_curve_gbp) > 1:
            equity_arr_gbp = np.array(equity_curve_gbp)
            peak_gbp = np.maximum.accumulate(equity_arr_gbp)
            drawdown_gbp = peak_gbp - equity_arr_gbp
            max_dd_gbp = drawdown_gbp.max()
        else:
            max_dd_gbp = 0.0

        return StrategyResult(
            strategy_name=f"{strategy}_both",
            strategy_category=self.ENTRY_STRATEGIES.get(strategy, {}).get('category', 'Unknown'),
            direction='both',
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
            beats_buy_hold=bool(total_pnl_percent > self.buy_hold_return),
            equity_curve=equity_curve,
            trades_list=trades_list,
            has_open_position=has_open,
            open_position=open_position_data,
            total_pnl_gbp=round(total_pnl_gbp, 2),
            max_drawdown_gbp=round(max_dd_gbp, 2),
            avg_trade_gbp=round(avg_trade_gbp, 2),
            equity_curve_gbp=equity_curve_gbp,
            source_currency=source_currency,
            display_currencies=["USD", "GBP"] if needs_conversion else [source_currency],
            # Bidirectional fields
            long_trades=long_trade_count,
            long_wins=long_win_count,
            long_pnl=round(long_pnl_total, 2),
            short_trades=short_trade_count,
            short_wins=short_win_count,
            short_pnl=round(short_pnl_total, 2),
            flip_count=flip_count
        )

    def find_strategies(self, min_trades: int = 3,
                        min_win_rate: float = 0,
                        save_to_db: bool = True,
                        symbol: str = None,
                        timeframe: str = None,
                        n_trials: int = 300,
                        mode: str = "all") -> List[StrategyResult]:
        """
        Find all profitable strategies.
        Saves winners to database for future reference.

        n_trials controls granularity of TP/SL testing:
        - 100: 1.0% increments (fast)
        - 225: 0.67% increments
        - 400: 0.5% increments (thorough)
        - 625: 0.4% increments
        - 10000: 0.1% increments (exhaustive)

        mode controls direction testing:
        - "separate": Test long and short independently (default, current behavior)
        - "bidirectional": Test combined long+short strategies only
        - "all": Run both separate and bidirectional modes
        """
        strategies = list(self.ENTRY_STRATEGIES.keys())
        directions = ['long', 'short'] if mode in ['separate', 'all'] else []

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

        # Calculate total combinations based on mode
        separate_total = num_strategies * num_directions * num_tp * num_sl if directions else 0
        bidirectional_total = num_strategies * num_tp * num_sl if mode in ['bidirectional', 'all'] else 0
        total = separate_total + bidirectional_total

        tested = 0
        profitable_count = 0

        # Progress phases:
        # 0-2%: Initialization (already done)
        # 2-90%: Testing combinations (main work)
        # 90-95%: Sorting/filtering
        # 95-100%: Saving to DB

        mode_desc = "bidirectional" if mode == "bidirectional" else ("separate + bidirectional" if mode == "all" else "separate (long/short)")
        self._update_status(f"Testing {total:,} combinations ({num_strategies} strategies, {mode_desc}, {num_tp}×{num_sl} TP/SL @ {increment:.2f}% steps)...", 2)

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

        # === SEPARATE DIRECTION TESTING (long/short independently) ===
        if directions:  # Only run if mode is "separate" or "all"
            for strat_idx, strategy in enumerate(strategies):
                # Check for abort signal
                if self.status and self.status.get("abort"):
                    self._update_status("Optimization aborted by user", 95)
                    break

                for dir_idx, direction in enumerate(directions):
                    # Check for abort signal
                    if self.status and self.status.get("abort"):
                        break

                    # Update at start of each strategy/direction combination
                    combo_num = strat_idx * num_directions + dir_idx + 1
                    combo_total = num_strategies * num_directions

                    for tp in tp_range:
                        # Check for abort signal
                        if self.status and self.status.get("abort"):
                            break

                        for sl in sl_range:
                            # Check for abort signal
                            if self.status and self.status.get("abort"):
                                break

                            result = self.backtest(strategy, direction, tp, sl,
                                                   initial_capital=self.capital,
                                                   position_size_pct=self.position_size_pct,
                                                   source_currency=self.source_currency,
                                                   fx_fetcher=self.fx_fetcher)
                            tested += 1

                            # Update progress more frequently
                            if tested % update_interval == 0 or tested == total:
                                # Progress from 2% to 90% during testing phase
                                progress = int(2 + (tested / total) * 88)
                                # Calculate actual overall progress percentage for display
                                overall_pct = self.progress_min + ((progress / 100) * (self.progress_max - self.progress_min))
                                self._update_status(
                                    f"[{combo_num}/{combo_total}] {strategy} {direction.upper()} | {tested:,}/{total:,} ({overall_pct:.1f}%) | Found: {profitable_count}",
                                    progress
                                )

                            if result.total_trades >= min_trades and (result.win_rate or 0) >= min_win_rate:
                                results.append(result)

                                # Stream profitable ones
                                if result.total_pnl is not None and result.total_pnl > 0:
                                    profitable_count += 1
                                    self._publish_result(result)

        # === BIDIRECTIONAL TESTING (combined long+short) ===
        if mode in ['bidirectional', 'all']:
            for strat_idx, strategy in enumerate(strategies):
                # Check for abort signal
                if self.status and self.status.get("abort"):
                    self._update_status("Optimization aborted by user", 95)
                    break

                combo_num = strat_idx + 1

                for tp in tp_range:
                    # Check for abort signal
                    if self.status and self.status.get("abort"):
                        break

                    for sl in sl_range:
                        # Check for abort signal
                        if self.status and self.status.get("abort"):
                            break

                        result = self.backtest_bidirectional(strategy, tp, sl,
                                                             initial_capital=self.capital,
                                                             position_size_pct=self.position_size_pct,
                                                             source_currency=self.source_currency,
                                                             fx_fetcher=self.fx_fetcher)
                        tested += 1

                        # Update progress more frequently
                        if tested % update_interval == 0 or tested == total:
                            progress = int(2 + (tested / total) * 88)
                            overall_pct = self.progress_min + ((progress / 100) * (self.progress_max - self.progress_min))
                            self._update_status(
                                f"[{combo_num}/{num_strategies}] {strategy} BIDIRECTIONAL | {tested:,}/{total:,} ({overall_pct:.1f}%) | Found: {profitable_count}",
                                progress
                            )

                        if result.total_trades >= min_trades and (result.win_rate or 0) >= min_win_rate:
                            results.append(result)

                            # Stream profitable ones
                            if result.total_pnl is not None and result.total_pnl > 0:
                                profitable_count += 1
                                self._publish_result(result)

        # Phase: Sorting results (90-95%)
        self._update_status(f"Sorting {len(results):,} results by composite score...", 90)

        # Sort by COMPOSITE SCORE (not just PnL)
        # This ensures high win rate + good PF strategies rank higher
        results.sort(key=lambda x: x.composite_score if x.composite_score is not None else 0, reverse=True)

        self._update_status(f"Filtering profitable strategies...", 92)

        # Save profitable strategies to database
        profitable = [r for r in results if r.total_pnl is not None and r.total_pnl > 0]

        # Phase: Saving to DB (95-100%)
        if self.db and save_to_db and profitable:
            self._update_status(f"Saving top {min(50, len(profitable))} strategies to database...", 95)

            for i, result in enumerate(profitable[:50]):  # Save top 50
                self.db.save_strategy(
                    result,
                    run_id=db_run_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    data_start=self.data_start,
                    data_end=self.data_end
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

                    result = self.backtest(entry_rule, direction, tp, sl,
                                           source_currency=self.source_currency,
                                           fx_fetcher=self.fx_fetcher)
                    if result.total_trades >= 3:
                        results.append(result)

        results.sort(key=lambda x: x.total_pnl if x.total_pnl is not None else 0, reverse=True)
        return results

    # =========================================================================
    # PHASE 2: INDICATOR PARAMETER TUNING
    # =========================================================================

    def _recalculate_indicators_for_tuning(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Recalculate specific indicators with custom parameters.
        Only recalculates what's needed based on params provided.
        """
        engine = self.calc_engine

        # Use MultiEngineCalculator if available
        if HAS_MULTI_ENGINE and engine in ['tradingview', 'native']:
            calc = MultiEngineCalculator(df)

            # RSI
            if 'rsi_length' in params:
                length = params['rsi_length']
                if engine == 'tradingview':
                    df['rsi'] = calc.rsi_tradingview(length)
                else:
                    df['rsi'] = calc.rsi_native(length)

            # Stochastic
            if any(k in params for k in ['stoch_k', 'stoch_d', 'stoch_smooth']):
                k = params.get('stoch_k', DEFAULT_INDICATOR_PARAMS['stoch_k'])
                d = params.get('stoch_d', DEFAULT_INDICATOR_PARAMS['stoch_d'])
                smooth = params.get('stoch_smooth', DEFAULT_INDICATOR_PARAMS['stoch_smooth'])
                if engine == 'tradingview':
                    stoch_k, stoch_d = calc.stoch_tradingview(k, d, smooth)
                else:
                    stoch_k, stoch_d = calc.stoch_native(k, d, smooth)
                df['stoch_k'] = stoch_k
                df['stoch_d'] = stoch_d

            # Bollinger Bands
            if any(k in params for k in ['bb_length', 'bb_mult']):
                length = params.get('bb_length', DEFAULT_INDICATOR_PARAMS['bb_length'])
                mult = params.get('bb_mult', DEFAULT_INDICATOR_PARAMS['bb_mult'])
                if engine == 'tradingview':
                    bb_mid, bb_upper, bb_lower = calc.bbands_tradingview(length, mult)
                else:
                    bb_mid, bb_upper, bb_lower = calc.bbands_native(length, mult)
                df['bb_upper'] = bb_upper
                df['bb_lower'] = bb_lower
                df['bb_mid'] = bb_mid
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

            # ATR
            if 'atr_length' in params:
                length = params['atr_length']
                if engine == 'tradingview':
                    df['atr'] = calc.atr_tradingview(length)
                else:
                    df['atr'] = calc.atr_native(length)

            # EMA
            if 'ema_fast' in params or 'ema_slow' in params:
                fast = params.get('ema_fast', DEFAULT_INDICATOR_PARAMS['ema_fast'])
                slow = params.get('ema_slow', DEFAULT_INDICATOR_PARAMS['ema_slow'])
                if engine == 'tradingview':
                    df['ema_9'] = calc.ema_tradingview(fast)
                    df['ema_21'] = calc.ema_tradingview(slow)
                else:
                    df['ema_9'] = calc.ema_native(fast)
                    df['ema_21'] = calc.ema_native(slow)

            # SMA
            if 'sma_20' in params:
                length = params['sma_20']
                if engine == 'tradingview':
                    df['sma_20'] = calc.sma_tradingview(length)
                else:
                    df['sma_20'] = calc.sma_native(length)

            if 'sma_50' in params:
                length = params['sma_50']
                if engine == 'tradingview':
                    df['sma_50'] = calc.sma_tradingview(length)
                else:
                    df['sma_50'] = calc.sma_native(length)

            # SMA fast/slow (for sma_cross strategy)
            if 'sma_fast' in params or 'sma_slow' in params:
                fast = params.get('sma_fast', DEFAULT_INDICATOR_PARAMS['sma_fast'])
                slow = params.get('sma_slow', DEFAULT_INDICATOR_PARAMS['sma_slow'])
                if engine == 'tradingview':
                    df['sma_fast'] = calc.sma_tradingview(fast)
                    df['sma_slow'] = calc.sma_tradingview(slow)
                else:
                    df['sma_fast'] = calc.sma_native(fast)
                    df['sma_slow'] = calc.sma_native(slow)

            # MACD
            if any(k in params for k in ['macd_fast', 'macd_slow', 'macd_signal']):
                fast = params.get('macd_fast', DEFAULT_INDICATOR_PARAMS['macd_fast'])
                slow = params.get('macd_slow', DEFAULT_INDICATOR_PARAMS['macd_slow'])
                signal = params.get('macd_signal', DEFAULT_INDICATOR_PARAMS['macd_signal'])
                if engine == 'tradingview':
                    macd_line, signal_line, histogram = calc.macd_tradingview(fast, slow, signal)
                else:
                    macd_line, signal_line, histogram = calc.macd_native(fast, slow, signal)
                df['macd'] = macd_line
                df['macd_signal'] = signal_line
                df['macd_hist'] = histogram

            # Williams %R
            if 'willr_length' in params:
                length = params['willr_length']
                if engine == 'tradingview':
                    df['willr'] = calc.willr_tradingview(length)
                else:
                    df['willr'] = calc.willr_native(length)

            # CCI
            if 'cci_length' in params:
                length = params['cci_length']
                if engine == 'tradingview':
                    df['cci'] = calc.cci_tradingview(length)
                else:
                    df['cci'] = calc.cci_native(length)

            # ADX
            if 'adx_length' in params:
                length = params['adx_length']
                if engine == 'tradingview':
                    adx, di_plus, di_minus = calc.adx_tradingview(length)
                else:
                    adx, di_plus, di_minus = calc.adx_native(length)
                df['adx'] = adx
                df['di_plus'] = di_plus
                df['di_minus'] = di_minus

            # Supertrend
            if any(k in params for k in ['supertrend_factor', 'supertrend_atr']):
                factor = params.get('supertrend_factor', DEFAULT_INDICATOR_PARAMS['supertrend_factor'])
                atr_len = params.get('supertrend_atr', DEFAULT_INDICATOR_PARAMS['supertrend_atr'])
                if engine == 'tradingview':
                    supertrend, supertrend_dir = calc.supertrend_tradingview(factor, atr_len)
                else:
                    supertrend, supertrend_dir = calc.supertrend_native(factor, atr_len)
                df['supertrend'] = supertrend
                df['supertrend_dir'] = supertrend_dir

            # Aroon
            if 'aroon_length' in params:
                length = params['aroon_length']
                if engine == 'tradingview':
                    aroon_up, aroon_down, aroon_osc = calc.aroon_tradingview(length)
                    df['aroon_up'] = aroon_up
                    df['aroon_down'] = aroon_down
                    df['aroon_osc'] = aroon_osc
                else:
                    # Fall back to pandas_ta for native
                    aroon = ta.aroon(df['high'], df['low'], length=length)
                    aroon_up_col = [c for c in aroon.columns if 'AROONU' in c][0]
                    aroon_down_col = [c for c in aroon.columns if 'AROOND' in c][0]
                    aroon_osc_col = [c for c in aroon.columns if 'AROONOSC' in c][0]
                    df['aroon_up'] = aroon[aroon_up_col]
                    df['aroon_down'] = aroon[aroon_down_col]
                    df['aroon_osc'] = aroon[aroon_osc_col]

            # Momentum
            if 'mom_length' in params:
                length = params['mom_length']
                if engine == 'tradingview':
                    df['mom'] = calc.mom_tradingview(length)
                else:
                    df['mom'] = calc.mom_native(length)

            # ROC
            if 'roc_length' in params:
                length = params['roc_length']
                if engine == 'tradingview':
                    df['roc'] = calc.roc_tradingview(length)
                else:
                    df['roc'] = calc.roc_native(length)

            # Keltner Channels
            if any(k in params for k in ['keltner_length', 'keltner_mult', 'keltner_atr']):
                length = params.get('keltner_length', DEFAULT_INDICATOR_PARAMS['keltner_length'])
                mult = params.get('keltner_mult', DEFAULT_INDICATOR_PARAMS['keltner_mult'])
                atr_len = params.get('keltner_atr', DEFAULT_INDICATOR_PARAMS['keltner_atr'])
                if engine == 'tradingview':
                    kc_mid, kc_upper, kc_lower = calc.keltner_tradingview(length, mult, atr_len)
                    df['kc_mid'] = kc_mid
                    df['kc_upper'] = kc_upper
                    df['kc_lower'] = kc_lower

            # Donchian Channels
            if 'donchian_length' in params:
                length = params['donchian_length']
                if engine == 'tradingview':
                    dc_mid, dc_upper, dc_lower = calc.donchian_tradingview(length)
                    df['dc_mid'] = dc_mid
                    df['dc_upper'] = dc_upper
                    df['dc_lower'] = dc_lower

            # Ichimoku
            if any(k in params for k in ['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou']):
                tenkan = params.get('ichimoku_tenkan', DEFAULT_INDICATOR_PARAMS['ichimoku_tenkan'])
                kijun = params.get('ichimoku_kijun', DEFAULT_INDICATOR_PARAMS['ichimoku_kijun'])
                senkou = params.get('ichimoku_senkou', DEFAULT_INDICATOR_PARAMS['ichimoku_senkou'])
                if engine == 'tradingview':
                    ichimoku = calc.ichimoku_tradingview(tenkan, kijun, senkou)
                    df['tenkan'] = ichimoku['tenkan']
                    df['kijun'] = ichimoku['kijun']
                    df['senkou_a'] = ichimoku['senkou_a']
                    df['senkou_b'] = ichimoku['senkou_b']

            # Ultimate Oscillator
            if any(k in params for k in ['uo_fast', 'uo_mid', 'uo_slow']):
                fast = params.get('uo_fast', DEFAULT_INDICATOR_PARAMS['uo_fast'])
                mid = params.get('uo_mid', DEFAULT_INDICATOR_PARAMS['uo_mid'])
                slow = params.get('uo_slow', DEFAULT_INDICATOR_PARAMS['uo_slow'])
                if engine == 'tradingview':
                    df['uo'] = calc.uo_tradingview(fast, mid, slow)

            # Choppiness
            if 'chop_length' in params:
                length = params['chop_length']
                if engine == 'tradingview':
                    df['chop'] = calc.chop_tradingview(length)

        else:
            # Fallback to pandas_ta
            if 'rsi_length' in params:
                df['rsi'] = ta.rsi(df['close'], length=params['rsi_length'])

            if any(k in params for k in ['stoch_k', 'stoch_d', 'stoch_smooth']):
                k = params.get('stoch_k', DEFAULT_INDICATOR_PARAMS['stoch_k'])
                d = params.get('stoch_d', DEFAULT_INDICATOR_PARAMS['stoch_d'])
                smooth = params.get('stoch_smooth', DEFAULT_INDICATOR_PARAMS['stoch_smooth'])
                stoch = ta.stoch(df['high'], df['low'], df['close'], k=k, d=d, smooth_k=smooth)
                stoch_k_col = [c for c in stoch.columns if c.startswith('STOCHk_')][0]
                stoch_d_col = [c for c in stoch.columns if c.startswith('STOCHd_')][0]
                df['stoch_k'] = stoch[stoch_k_col]
                df['stoch_d'] = stoch[stoch_d_col]

            if any(k in params for k in ['bb_length', 'bb_mult']):
                length = params.get('bb_length', DEFAULT_INDICATOR_PARAMS['bb_length'])
                mult = params.get('bb_mult', DEFAULT_INDICATOR_PARAMS['bb_mult'])
                bb = ta.bbands(df['close'], length=length, std=mult)
                bb_upper_col = [c for c in bb.columns if c.startswith('BBU_')][0]
                bb_lower_col = [c for c in bb.columns if c.startswith('BBL_')][0]
                bb_mid_col = [c for c in bb.columns if c.startswith('BBM_')][0]
                df['bb_upper'] = bb[bb_upper_col]
                df['bb_lower'] = bb[bb_lower_col]
                df['bb_mid'] = bb[bb_mid_col]
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

            if 'atr_length' in params:
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=params['atr_length'])

            if 'ema_fast' in params or 'ema_slow' in params:
                fast = params.get('ema_fast', DEFAULT_INDICATOR_PARAMS['ema_fast'])
                slow = params.get('ema_slow', DEFAULT_INDICATOR_PARAMS['ema_slow'])
                df['ema_9'] = ta.ema(df['close'], length=fast)
                df['ema_21'] = ta.ema(df['close'], length=slow)

            if 'sma_20' in params:
                df['sma_20'] = ta.sma(df['close'], length=params['sma_20'])

            # SMA fast/slow (for sma_cross strategy)
            if 'sma_fast' in params or 'sma_slow' in params:
                fast = params.get('sma_fast', DEFAULT_INDICATOR_PARAMS['sma_fast'])
                slow = params.get('sma_slow', DEFAULT_INDICATOR_PARAMS['sma_slow'])
                df['sma_fast'] = ta.sma(df['close'], length=fast)
                df['sma_slow'] = ta.sma(df['close'], length=slow)

            if any(k in params for k in ['macd_fast', 'macd_slow', 'macd_signal']):
                fast = params.get('macd_fast', DEFAULT_INDICATOR_PARAMS['macd_fast'])
                slow = params.get('macd_slow', DEFAULT_INDICATOR_PARAMS['macd_slow'])
                signal = params.get('macd_signal', DEFAULT_INDICATOR_PARAMS['macd_signal'])
                macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
                macd_col = [c for c in macd.columns if c.startswith('MACD_') and not c.startswith('MACDs') and not c.startswith('MACDh')][0]
                macd_signal_col = [c for c in macd.columns if c.startswith('MACDs_')][0]
                macd_hist_col = [c for c in macd.columns if c.startswith('MACDh_')][0]
                df['macd'] = macd[macd_col]
                df['macd_signal'] = macd[macd_signal_col]
                df['macd_hist'] = macd[macd_hist_col]

        # Store non-indicator parameters as constant columns for signal functions
        if 'consecutive_bars' in params:
            df['consecutive_bars'] = params['consecutive_bars']

        # Sanitize all indicator columns to prevent None comparison errors
        self._sanitize_df(df)

        return df

    def _sanitize_df(self, df: pd.DataFrame):
        """
        Sanitize a DataFrame's indicator columns to prevent 'None < float' comparison errors.
        Converts all None values to NaN and ensures numeric columns are float type.
        """
        # List of all indicator columns that should be numeric
        numeric_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'stoch_k', 'stoch_d', 'atr',
            'bb_upper', 'bb_lower', 'bb_mid', 'bb_width',
            'sma_20', 'sma_50', 'sma_fast', 'sma_slow', 'ema_9', 'ema_21',
            'macd', 'macd_signal', 'macd_hist',
            'willr', 'cci', 'mom', 'roc',
            'adx', 'di_plus', 'di_minus',
            'aroon_up', 'aroon_down', 'aroon_osc',
            'supertrend', 'supertrend_dir',
            'psar', 'vwap',
            'kc_mid', 'kc_upper', 'kc_lower',
            'dc_mid', 'dc_upper', 'dc_lower',
            'tenkan', 'kijun', 'senkou_a', 'senkou_b',
            'uo', 'chop',
            'body', 'range', 'pct_change',
            'ht_trendmode', 'ht_dcperiod',
            'consecutive_bars'
        ]

        for col in numeric_columns:
            if col in df.columns:
                # Convert None to NaN and ensure float type
                df[col] = pd.to_numeric(df[col], errors='coerce')

    def _backtest_with_custom_df(self, df: pd.DataFrame, strategy: str, direction: str,
                                  tp_percent: float, sl_percent: float,
                                  initial_capital: float = 1000.0,
                                  position_size_pct: float = 100.0,
                                  commission_pct: float = 0.1) -> StrategyResult:
        """
        Run backtest using a custom dataframe (for tuning).
        This is a copy of backtest() that uses the provided df instead of self.df.
        """
        # Temporarily swap the dataframe
        original_df = self.df
        self.df = df

        # Run backtest
        result = self.backtest(strategy, direction, tp_percent, sl_percent,
                               initial_capital, position_size_pct, commission_pct,
                               source_currency=self.source_currency,
                               fx_fetcher=self.fx_fetcher)

        # Restore original
        self.df = original_df

        return result

    def tune_single_strategy(self, strategy_result: StrategyResult,
                             streaming_callback: Callable = None) -> TunedResult:
        """
        Phase 2: Tune indicator parameters for a single winning strategy.

        Takes a Phase 1 winner and optimizes its indicator lengths.
        Returns a TunedResult with before/after comparison.
        """
        from itertools import product

        entry_rule = strategy_result.entry_rule
        direction = strategy_result.direction
        tp = strategy_result.tp_percent
        sl = strategy_result.sl_percent

        # Get parameter config for this strategy
        param_config = STRATEGY_PARAM_MAP.get(entry_rule, {'params': [], 'ranges': {}})
        param_names = param_config['params']
        param_ranges = param_config['ranges']

        # If no tunable parameters, return with no change
        if not param_names:
            default_params = {}
            # Handle None values by defaulting to 0
            score = strategy_result.composite_score if strategy_result.composite_score is not None else 0
            win_rate = strategy_result.win_rate if strategy_result.win_rate is not None else 0
            pf = strategy_result.profit_factor if strategy_result.profit_factor is not None else 0
            pnl_pct = strategy_result.total_pnl_percent if strategy_result.total_pnl_percent is not None else 0
            return TunedResult(
                original_result=strategy_result,
                tuned_params=default_params,
                default_params=default_params,
                before_score=score,
                after_score=score,
                before_win_rate=win_rate,
                after_win_rate=win_rate,
                before_profit_factor=pf,
                after_profit_factor=pf,
                before_pnl_percent=pnl_pct,
                after_pnl_percent=pnl_pct,
                tuned_result=strategy_result,
            )

        # Get default params for this strategy
        default_params = {p: DEFAULT_INDICATOR_PARAMS[p] for p in param_names if p in DEFAULT_INDICATOR_PARAMS}

        # Baseline metrics (handle None by defaulting to 0)
        baseline_score = strategy_result.composite_score if strategy_result.composite_score is not None else 0
        best_score = baseline_score
        best_params = default_params.copy()
        best_result = strategy_result

        # Generate all parameter combinations
        param_values = [param_ranges[p] for p in param_names]
        combinations = list(product(*param_values))

        # Test each combination
        tested_count = 0
        improved_combos = []

        print(f"  [Tuning] {entry_rule}_{direction}: Testing {len(combinations)} combinations for params: {param_names}")
        print(f"  [Tuning] Default params: {default_params}, Baseline score: {baseline_score:.1f}")

        import gc

        for combo in combinations:
            param_dict = dict(zip(param_names, combo))

            # Skip the default combination (already tested in Phase 1)
            if param_dict == default_params:
                continue

            tested_count += 1

            # Create a copy of the dataframe and recalculate indicators
            df_copy = self.df.copy()
            df_copy = self._recalculate_indicators_for_tuning(df_copy, param_dict)

            # Run backtest with new indicators
            result = self._backtest_with_custom_df(
                df_copy, entry_rule, direction, tp, sl,
                initial_capital=self.capital,
                position_size_pct=self.position_size_pct
            )

            # Check if this is better (handle None by defaulting to 0)
            result_score = result.composite_score if result.composite_score is not None else 0
            result_pnl = result.total_pnl_percent if result.total_pnl_percent is not None else 0
            result_trades = result.total_trades if result.total_trades is not None else 0

            # Log each combination result
            if result_score != baseline_score or result_trades > 0:
                print(f"    {param_dict} -> Score: {result_score:.1f}, PnL: {result_pnl:.1f}%, Trades: {result_trades}")

            if result_score > best_score:
                improved_combos.append((param_dict.copy(), result_score))
                best_score = result_score
                best_params = param_dict.copy()
                best_result = result
                print(f"    *** NEW BEST: {param_dict} -> Score: {result_score:.1f} (was {baseline_score:.1f})")

            # Clean up to prevent memory buildup
            del df_copy
            if tested_count % 10 == 0:
                gc.collect()

        print(f"  [Tuning] Tested {tested_count} combos, {len(improved_combos)} improved. Best score: {best_score:.1f} (baseline: {baseline_score:.1f})")

        # Create TunedResult (handle None values by defaulting to 0)
        before_wr = strategy_result.win_rate if strategy_result.win_rate is not None else 0
        after_wr = best_result.win_rate if best_result.win_rate is not None else 0
        before_pf = strategy_result.profit_factor if strategy_result.profit_factor is not None else 0
        after_pf = best_result.profit_factor if best_result.profit_factor is not None else 0
        before_pnl = strategy_result.total_pnl_percent if strategy_result.total_pnl_percent is not None else 0
        after_pnl = best_result.total_pnl_percent if best_result.total_pnl_percent is not None else 0

        tuned_result = TunedResult(
            original_result=strategy_result,
            tuned_params=best_params,
            default_params=default_params,
            before_score=baseline_score,
            after_score=best_score,
            before_win_rate=before_wr,
            after_win_rate=after_wr,
            before_profit_factor=before_pf,
            after_profit_factor=after_pf,
            before_pnl_percent=before_pnl,
            after_pnl_percent=after_pnl,
            tuned_result=best_result,
        )

        return tuned_result

    def tune_top_strategies(self, phase1_results: List[StrategyResult],
                           top_n: int = 20,
                           streaming_callback: Callable = None) -> List[TunedResult]:
        """
        Phase 2: Tune indicator parameters for top N winning strategies.

        Args:
            phase1_results: List of StrategyResult from Phase 1
            top_n: Number of top strategies to tune (default 20)
            streaming_callback: Optional callback for progress updates

        Returns:
            List of TunedResult with before/after comparisons
        """
        # Filter to profitable strategies and take top N by composite score
        profitable = [r for r in phase1_results if r.total_pnl is not None and r.total_pnl > 0]
        profitable.sort(key=lambda x: x.composite_score if x.composite_score is not None else 0, reverse=True)
        top_strategies = profitable[:top_n]

        tuned_results = []
        total = len(top_strategies)

        for i, strategy_result in enumerate(top_strategies):
            # Get tuning parameters info for this strategy
            entry_rule = strategy_result.entry_rule
            param_config = STRATEGY_PARAM_MAP.get(entry_rule, {'params': [], 'ranges': {}})
            param_names = param_config['params']
            param_ranges = param_config['ranges']

            # Calculate number of combinations
            from itertools import product
            if param_names:
                param_values = [param_ranges.get(p, []) for p in param_names]
                num_combinations = len(list(product(*param_values)))
            else:
                num_combinations = 0

            # Update progress with detailed tuning info
            if streaming_callback:
                progress_pct = int((i / total) * 100)
                streaming_callback({
                    'type': 'tuning_progress',
                    'current': i + 1,
                    'total': total,
                    'strategy': strategy_result.strategy_name,
                    'entry_rule': strategy_result.entry_rule,
                    'direction': strategy_result.direction,
                    'progress': progress_pct,
                    # New detailed tuning info
                    'params_being_tuned': param_names,
                    'param_ranges': {p: list(param_ranges.get(p, [])) for p in param_names},
                    'num_combinations': num_combinations,
                    'original_score': strategy_result.composite_score,
                    'original_pnl': strategy_result.total_pnl_percent,
                })

            # Tune this strategy
            tuned = self.tune_single_strategy(strategy_result)
            tuned_results.append(tuned)

            # Stream the result
            if streaming_callback:
                streaming_callback({
                    'type': 'tuning_result',
                    'rank': i + 1,
                    'tuning': tuned.to_dict(),
                })

        # Sort by tuned score
        tuned_results.sort(key=lambda x: x.after_score if x.after_score is not None else 0, reverse=True)

        return tuned_results


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

        'low_volatility_breakout': f'''// Low Volatility Breakout Entry (ADAPTIVE)
atrValue = ta.atr(14)
atrThreshold = ta.percentile_linear_interpolation(atrValue, 100, 25)  // Bottom 25%
lowVol = atrValue < atrThreshold
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

        'roc_extreme': f'''// Rate of Change Extreme Entry (ADAPTIVE)
rocValue = ta.roc(close, 9)
rocLower = ta.percentile_linear_interpolation(rocValue, 100, 5)  // Bottom 5%
rocUpper = ta.percentile_linear_interpolation(rocValue, 100, 95) // Top 5%
entrySignal = {"rocValue < rocLower" if is_long else "rocValue > rocUpper"}''',

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
                        position_size_pct: float = 100.0,
                        engine: str = "tradingview",
                        n_trials: int = 300,
                        progress_min: int = 0,
                        progress_max: int = 100,
                        source_currency: str = "USD",
                        fx_fetcher=None) -> Dict:
    """Main entry point for the strategy engine.

    Args:
        symbol: Trading symbol (e.g., 'BTCGBP')
        timeframe: Timeframe (e.g., '15m')
        exchange: Exchange name (e.g., 'KRAKEN', 'BINANCE')
        capital: Starting capital (from UI)
        position_size_pct: Position size as % of equity (from UI "Position Size %")
        engine: Calculation engine - "tradingview" or "native"
        n_trials: Controls TP/SL granularity (100=1%, 300=0.33%, 500=0.2% increments)
        source_currency: Currency of source data ("USD" for BTCUSDT, "GBP" for BTCGBP)
        fx_fetcher: Exchange rate fetcher instance for USD->GBP conversion
    """

    strategy_engine = StrategyEngine(df, status, streaming_callback,
                                     capital=capital, position_size_pct=position_size_pct,
                                     calc_engine=engine,
                                     progress_min=progress_min, progress_max=progress_max,
                                     source_currency=source_currency, fx_fetcher=fx_fetcher)

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
    profitable = [r for r in results if r.total_pnl is not None and r.total_pnl > 0]
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
        # Currency conversion info
        'source_currency': source_currency,
        'display_currencies': ["USD", "GBP"] if source_currency == "USD" and fx_fetcher else [source_currency],
        'currency_conversion_enabled': source_currency == "USD" and fx_fetcher is not None,
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
                'win_rate': round(r.win_rate, 1) if r.win_rate is not None else 0,
                'profit_factor': r.profit_factor if r.profit_factor is not None else 0,
                'total_pnl': round(r.total_pnl, 2) if r.total_pnl is not None else 0,
                'total_pnl_percent': round(r.total_pnl_percent, 2) if r.total_pnl_percent is not None else 0,
                'max_drawdown': round(r.max_drawdown, 2) if r.max_drawdown is not None else 0,
                'max_drawdown_percent': round(r.max_drawdown_percent, 2) if r.max_drawdown_percent is not None else 0,
                'avg_trade': round(r.avg_trade, 2) if r.avg_trade is not None else 0,
                'composite_score': round(r.composite_score, 1) if r.composite_score is not None else 0,
                # Buy & Hold comparison
                'buy_hold_return': r.buy_hold_return,
                'vs_buy_hold': r.vs_buy_hold,
                'beats_buy_hold': r.beats_buy_hold,
                # GBP conversion metrics
                'total_pnl_gbp': round(r.total_pnl_gbp, 2) if r.total_pnl_gbp is not None else 0,
                'max_drawdown_gbp': round(r.max_drawdown_gbp, 2) if r.max_drawdown_gbp is not None else 0,
                'avg_trade_gbp': round(r.avg_trade_gbp, 2) if r.avg_trade_gbp is not None else 0,
                'source_currency': r.source_currency,
                'display_currencies': r.display_currencies or [source_currency],
            },
            'equity_curve': r.equity_curve if r.equity_curve else [],
            'equity_curve_gbp': r.equity_curve_gbp if r.equity_curve_gbp else [],
            'trades_list': r.trades_list if r.trades_list else [],  # All trades for CSV export
            # Open position warning (shows if backtest ended with unclosed trade)
            'has_open_position': r.has_open_position,
            'open_position': r.open_position
        })

    # Include all profitable results for Phase 2 tuning
    # (StrategyResult objects - not serialized, for internal use)
    report['all_results'] = profitable

    return report
