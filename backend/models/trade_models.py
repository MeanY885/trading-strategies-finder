"""
TRADE MODELS
=============
Data classes for trading configuration and results.
Extracted from strategy_engine.py for better modularity.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional


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
    - 'volatility_adjusted': Scale position based on ATR
    """
    sizing_method: str = 'fixed'
    initial_capital: float = 10000.0
    fixed_amount: float = 1000.0
    equity_percent: float = 10.0
    risk_percent: float = 1.0
    kelly_fraction: float = 0.25
    target_risk_atr: float = 1.5
    base_position_size: float = 1000.0
    compound_profits: bool = True
    max_position_pct: float = 50.0
    min_position_size: float = 10.0

    def calculate_position_size(self,
                                 current_equity: float,
                                 sl_distance_pct: float = 1.0,
                                 win_rate: float = 0.5,
                                 avg_win_loss_ratio: float = 1.5,
                                 current_atr_pct: float = 2.0) -> float:
        """Calculate position size based on sizing method."""
        if self.sizing_method == 'fixed':
            position = self.fixed_amount
        elif self.sizing_method == 'percent_equity':
            position = current_equity * (self.equity_percent / 100.0)
        elif self.sizing_method == 'percent_risk':
            if sl_distance_pct <= 0:
                sl_distance_pct = 1.0
            risk_amount = current_equity * (self.risk_percent / 100.0)
            position = risk_amount / (sl_distance_pct / 100.0)
        elif self.sizing_method == 'kelly':
            p = max(0.01, min(0.99, win_rate))
            q = 1 - p
            b = max(0.1, avg_win_loss_ratio)
            kelly_full = (p * b - q) / b if b > 0 else 0
            kelly_full = max(0, kelly_full)
            kelly_adjusted = kelly_full * self.kelly_fraction
            position = current_equity * kelly_adjusted
        elif self.sizing_method == 'volatility_adjusted':
            if current_atr_pct <= 0:
                current_atr_pct = 2.0
            volatility_factor = self.target_risk_atr / current_atr_pct
            position = self.base_position_size * volatility_factor
        else:
            position = self.fixed_amount

        # Apply limits
        max_allowed = current_equity * (self.max_position_pct / 100.0)
        position = min(position, max_allowed)
        position = max(position, self.min_position_size)
        return position


@dataclass
class PortfolioRiskLimits:
    """Portfolio-level risk management limits."""
    max_concurrent_positions: int = 5
    max_positions_per_pair: int = 1
    max_daily_drawdown_pct: float = 5.0
    max_total_drawdown_pct: float = 20.0
    max_correlation: float = 0.7
    max_total_exposure_pct: float = 100.0
    max_single_pair_exposure_pct: float = 30.0
    reduce_size_after_losses: int = 3
    size_reduction_factor: float = 0.5

    def check_daily_drawdown(self, daily_pnl_pct: float) -> bool:
        """Check if daily drawdown limit exceeded. Returns True if OK."""
        return daily_pnl_pct > -self.max_daily_drawdown_pct

    def check_total_drawdown(self, total_drawdown_pct: float) -> bool:
        """Check if total drawdown limit exceeded. Returns True if OK."""
        return total_drawdown_pct < self.max_total_drawdown_pct

    def get_size_multiplier(self, consecutive_losses: int) -> float:
        """Get position size multiplier based on consecutive losses."""
        if consecutive_losses >= self.reduce_size_after_losses:
            return self.size_reduction_factor
        return 1.0


@dataclass
class TradeResult:
    """Result of a single trade execution."""
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    exit_reason: str  # 'tp', 'sl', 'trailing_stop', 'indicator_exit', 'flip', 'protection_sl'
    trade_num: int = 0
    entry_time: str = None
    exit_time: str = None
    position_size: float = 0.0
    position_qty: float = 0.0
    run_up: float = 0.0
    run_up_pct: float = 0.0
    drawdown: float = 0.0
    drawdown_pct: float = 0.0
    cumulative_pnl: float = 0.0
    pnl_gbp: float = 0.0
    position_size_gbp: float = 0.0
    cumulative_pnl_gbp: float = 0.0
    usd_gbp_rate: float = 1.0
    trade_duration_bars: int = 0
    trade_duration_hours: float = 0.0
    mfe_capture_ratio: float = 0.0
    exit_type: str = 'fixed_tp_sl'


@dataclass
class PeriodMetrics:
    """Performance metrics for a specific time period."""
    period_name: str
    period_days: int
    has_data: bool
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = None
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    profit_factor: float = None
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0


# Default configurations
DEFAULT_POSITION_SIZING = PositionSizing(
    sizing_method='fixed',
    fixed_amount=1000.0,
    compound_profits=False
)

COMPOUNDING_POSITION_SIZING = PositionSizing(
    sizing_method='percent_equity',
    equity_percent=10.0,
    compound_profits=True,
    initial_capital=10000.0
)

RISK_BASED_POSITION_SIZING = PositionSizing(
    sizing_method='percent_risk',
    risk_percent=1.0,
    compound_profits=True,
    initial_capital=10000.0
)

DEFAULT_PORTFOLIO_LIMITS = PortfolioRiskLimits()
