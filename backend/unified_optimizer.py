"""
UNIFIED STRATEGY OPTIMIZER
===========================
The ONE system that tests EVERYTHING:
- 75+ strategy types from all sources
- 3 ML optimization methods (Random, Bayesian TPE, CMA-ES)
- Walk-forward validation
- Consensus approach (strategies multiple methods agree on)
- Full parameter optimization with wide ranges

This replaces: optimizer.py, ensemble_optimizer.py, indicator_research.py
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os

# Import strategy database for persistence
try:
    from strategy_database import get_strategy_db, StrategyDatabase
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False
    print("Warning: Strategy database not available")

# Import psutil for auto-detecting optimal workers
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_optimal_workers() -> int:
    """Auto-detect optimal number of parallel workers based on CPU and memory.

    Returns:
        Optimal number of workers (between 2 and 12)
    """
    cpus = os.cpu_count() or 4

    if HAS_PSUTIL:
        # Get available memory in GB
        available_gb = psutil.virtual_memory().available / (1024**3)
        # Each worker uses ~500MB
        memory_limited = int(available_gb / 0.5)
        # Use 75% of CPUs, cap by memory, max 12
        optimal = min(int(cpus * 0.75), memory_limited, 12)
    else:
        # Fallback: use 75% of CPUs, max 8
        optimal = min(int(cpus * 0.75), 8)

    return max(2, optimal)

from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import ADXIndicator, MACD, CCIIndicator, SMAIndicator, EMAIndicator

# Import ALL indicator modules
from daviddtech_indicators import (
    calculate_daviddtech_indicators, jma, mcginley_dynamic, stiffness,
    tdfi, volatility_quality, trendilo, range_filter, flat_market_detector,
    t3, zlema, hma, supertrend, alma, lwpi, ema, sma
)
from advanced_indicators import calculate_all_advanced_indicators
from ml_indicators import calculate_ml_indicators

# Import Backtrader engine for industry-standard backtesting
try:
    from backtrader_engine import BacktraderEngine, BacktestResult as BTBacktestResult
    HAS_BACKTRADER = True
except ImportError:
    HAS_BACKTRADER = False
    print("Warning: Backtrader engine not available")

# Import ML models for automatic training
try:
    from ml_models.gradient_boosting import GradientBoostingPredictor
    HAS_ML_MODELS = True
except ImportError:
    HAS_ML_MODELS = False
    print("Warning: ML models not available (missing ml_models package)")

# Global storage for trained ML models (shared with get_signals)
trained_ml_models: Dict[str, Any] = {}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TradeRecord:
    """Single trade for backtest"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str


@dataclass
class StrategyResult:
    """Complete results for a strategy + parameter combination"""
    strategy_name: str
    strategy_category: str
    params: Dict[str, Any]

    # Core metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0

    # Advanced metrics
    equity_r_squared: float = 0.0  # Smoothness of equity curve
    recovery_factor: float = 0.0   # PnL / Max Drawdown
    sharpe_ratio: float = 0.0

    # Composite score (what we optimize for)
    composite_score: float = 0.0

    # Score breakdown for UI display (Balanced Triangle: 33% each)
    # Shows individual component scores and penalties applied
    score_breakdown: Dict[str, Any] = field(default_factory=dict)

    # Which optimization methods found this strategy
    found_by: List[str] = field(default_factory=list)

    # Validation results (out-of-sample)
    val_pnl: float = 0.0
    val_profit_factor: float = 0.0
    val_win_rate: float = 0.0

    # Trade details
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


# =============================================================================
# MASTER STRATEGY REGISTRY - ALL 75+ STRATEGIES
# =============================================================================

class StrategyRegistry:
    """
    Central registry of ALL strategy types we can test.
    Each strategy is a function that takes indicators and returns (long_signal, short_signal)
    """
    
    # Strategy categories for organization
    CATEGORIES = {
        "mean_reversion": "Mean Reversion",
        "trend_following": "Trend Following",
        "momentum": "Momentum",
        "breakout": "Breakout",
        "oscillator": "Oscillator",
        "daviddtech": "DaviddTech Pro",
        "confluence": "Multi-Indicator Confluence",
        "volatility": "Volatility-Based",
        "volume": "Volume-Based",
        "ml_based": "ML-Based Indicators",
        "simple": "Simple (Beginner-Friendly)"
    }
    
    @staticmethod
    def get_all_strategies() -> List[Dict]:
        """Return list of all strategy definitions"""
        strategies = []
        
        # ===== MEAN REVERSION STRATEGIES =====
        strategies.extend([
            {
                "name": "bb_rsi_classic",
                "display_name": "BB(20,2) + RSI(14) Classic",
                "category": "mean_reversion",
                "params": ["bb_length", "bb_mult", "rsi_length", "rsi_oversold", "rsi_overbought", "adx_threshold"],
            },
            {
                "name": "bb_rsi_tight",
                "display_name": "BB + RSI Tight (Conservative)",
                "category": "mean_reversion",
                "params": ["bb_length", "bb_mult", "rsi_length", "rsi_oversold", "rsi_overbought", "adx_threshold"],
            },
            {
                "name": "bb_stoch",
                "display_name": "BB + Stochastic",
                "category": "mean_reversion",
                "params": ["bb_length", "bb_mult", "stoch_k", "stoch_d", "stoch_oversold", "stoch_overbought", "adx_threshold"],
            },
            {
                "name": "keltner_rsi",
                "display_name": "Keltner Channel + RSI",
                "category": "mean_reversion",
                "params": ["kc_length", "kc_mult", "rsi_length", "rsi_oversold", "rsi_overbought", "adx_threshold"],
            },
            {
                "name": "donchian_fade",
                "display_name": "Donchian Channel Fade",
                "category": "mean_reversion",
                "params": ["dc_length", "rsi_length", "adx_threshold"],
            },
            {
                "name": "z_score_reversion",
                "display_name": "Z-Score Mean Reversion",
                "category": "mean_reversion",
                "params": ["z_period", "z_threshold", "adx_threshold"],
            },
            {
                "name": "bb_percent_b_extreme",
                "display_name": "Bollinger %B Extreme",
                "category": "mean_reversion",
                "params": ["bb_length", "bb_mult", "adx_threshold"],
            },
            {
                "name": "vwap_mean_reversion",
                "display_name": "VWAP Distance Reversion",
                "category": "mean_reversion",
                "params": ["vwap_threshold", "rsi_length", "adx_threshold"],
            },
        ])
        
        # ===== OSCILLATOR STRATEGIES =====
        strategies.extend([
            {
                "name": "stoch_extreme",
                "display_name": "Stochastic Extremes",
                "category": "oscillator",
                "params": ["stoch_k", "stoch_d", "stoch_oversold", "stoch_overbought", "adx_threshold"],
            },
            {
                "name": "rsi_extreme",
                "display_name": "RSI Extremes",
                "category": "oscillator",
                "params": ["rsi_length", "rsi_oversold", "rsi_overbought", "adx_threshold"],
            },
            {
                "name": "williams_r",
                "display_name": "Williams %R",
                "category": "oscillator",
                "params": ["willr_length", "willr_oversold", "willr_overbought", "adx_threshold"],
            },
            {
                "name": "cci_extreme",
                "display_name": "CCI Extremes",
                "category": "oscillator",
                "params": ["cci_length", "cci_threshold", "adx_threshold"],
            },
            {
                "name": "stoch_rsi",
                "display_name": "Stochastic RSI",
                "category": "oscillator",
                "params": ["stoch_rsi_length", "stoch_oversold", "stoch_overbought", "adx_threshold"],
            },
            {
                "name": "fisher_transform",
                "display_name": "Fisher Transform Reversal",
                "category": "oscillator",
                "params": ["fisher_length", "fisher_threshold", "adx_threshold"],
            },
            {
                "name": "awesome_oscillator",
                "display_name": "Awesome Oscillator Zero Cross",
                "category": "oscillator",
                "params": ["adx_threshold"],
            },
        ])
        
        # ===== TREND FOLLOWING STRATEGIES =====
        strategies.extend([
            {
                "name": "supertrend_follow",
                "display_name": "Supertrend Follow",
                "category": "trend_following",
                "params": ["st_length", "st_mult", "rsi_length"],
            },
            {
                "name": "macd_trend",
                "display_name": "MACD Trend",
                "category": "trend_following",
                "params": ["macd_fast", "macd_slow", "macd_signal"],
            },
            {
                "name": "adx_trend",
                "display_name": "ADX + DI Trend",
                "category": "trend_following",
                "params": ["adx_length", "adx_threshold", "rsi_length"],
            },
            {
                "name": "ema_crossover",
                "display_name": "EMA Crossover",
                "category": "trend_following",
                "params": ["ema_fast", "ema_slow", "adx_threshold"],
            },
            {
                "name": "hma_trend",
                "display_name": "Hull MA Trend",
                "category": "trend_following",
                "params": ["hma_length", "rsi_length", "adx_threshold"],
            },
            {
                "name": "zlema_momentum",
                "display_name": "ZLEMA Momentum",
                "category": "trend_following",
                "params": ["zlema_fast", "zlema_slow", "rsi_length"],
            },
            {
                "name": "multi_ma_confluence",
                "display_name": "Multi-MA Confluence (EMA9/21/55)",
                "category": "trend_following",
                "params": ["adx_threshold"],
            },
        ])
        
        # ===== BREAKOUT STRATEGIES =====
        strategies.extend([
            {
                "name": "bb_squeeze_breakout",
                "display_name": "BB Squeeze Breakout",
                "category": "breakout",
                "params": ["bb_length", "bb_mult", "squeeze_threshold"],
            },
            {
                "name": "donchian_breakout",
                "display_name": "Donchian Channel Breakout",
                "category": "breakout",
                "params": ["dc_length", "adx_threshold"],
            },
            {
                "name": "volatility_breakout",
                "display_name": "ATR Volatility Breakout",
                "category": "breakout",
                "params": ["atr_length", "atr_mult", "rsi_length"],
            },
            {
                "name": "trendilo_breakout",
                "display_name": "Trendilo Band Breakout",
                "category": "breakout",
                "params": ["trendilo_length"],
            },
        ])
        
        # ===== DAVIDDTECH PRO STRATEGIES =====
        strategies.extend([
            {
                "name": "stiff_surge_v1",
                "display_name": "Stiff Surge v1 (JMA+Stiff+TDFI)",
                "category": "daviddtech",
                "params": ["jma_length", "jma_phase", "stiff_length", "stiff_ma", "tdfi_length"],
            },
            {
                "name": "stiff_surge_v2",
                "display_name": "Stiff Surge v2 (JMA+VQ)",
                "category": "daviddtech",
                "params": ["jma_length", "jma_phase", "vq_length"],
            },
            {
                "name": "mcginley_trend",
                "display_name": "McGinley Trend Followers",
                "category": "daviddtech",
                "params": ["mcginley_length", "mcginley_k", "lwpi_length"],
            },
            {
                "name": "trendhoo_v1",
                "display_name": "Trendhoo (Trendilo+HMA+T3)",
                "category": "daviddtech",
                "params": ["trendilo_length", "hma_length", "t3_length"],
            },
            {
                "name": "trendhoo_v2",
                "display_name": "Trendhoo v2 (ALMA Optimized)",
                "category": "daviddtech",
                "params": ["alma_length", "alma_offset", "hma_length"],
            },
            {
                "name": "macd_liquidity_spectrum",
                "display_name": "MACD Liquidity Spectrum",
                "category": "daviddtech",
                "params": ["macd_fast", "macd_slow", "macd_signal", "rf_period", "rf_mult"],
            },
            {
                "name": "precision_trend_mastery",
                "display_name": "Precision Trend Mastery",
                "category": "daviddtech",
                "params": ["adx_length", "adx_threshold", "rf_period", "rf_mult"],
            },
            {
                "name": "t3_nexus_stiff",
                "display_name": "T3 Nexus + Stiffness",
                "category": "daviddtech",
                "params": ["t3_fast", "t3_slow", "stiff_length", "stiff_ma"],
            },
            {
                "name": "range_filter_adx",
                "display_name": "Range Filter + ADX",
                "category": "daviddtech",
                "params": ["rf_period", "rf_mult", "adx_threshold"],
            },
            {
                "name": "jma_volatility_quality",
                "display_name": "JMA + Volatility Quality",
                "category": "daviddtech",
                "params": ["jma_length", "jma_phase", "vq_length", "vq_threshold"],
            },
            {
                "name": "hma_stiffness",
                "display_name": "HMA + Stiffness",
                "category": "daviddtech",
                "params": ["hma_length", "stiff_length", "stiff_ma", "stiff_threshold"],
            },
            {
                "name": "lwpi_trend",
                "display_name": "LWPI Trend",
                "category": "daviddtech",
                "params": ["lwpi_length", "mcginley_length"],
            },
            {
                "name": "supertrend_confluence",
                "display_name": "Supertrend Multi-Confluence",
                "category": "daviddtech",
                "params": ["st_length", "st_mult", "rsi_length"],
            },
            {
                "name": "flat_market_mean_reversion",
                "display_name": "Flat Market Mean Reversion",
                "category": "daviddtech",
                "params": ["flat_threshold", "bb_length", "bb_mult", "rsi_length"],
            },
            {
                "name": "adaptive_momentum",
                "display_name": "Adaptive Multi-Indicator Score",
                "category": "daviddtech",
                "params": ["score_threshold", "rsi_length"],
            },
        ])
        
        # ===== CONFLUENCE STRATEGIES (Multiple Indicators) =====
        strategies.extend([
            {
                "name": "triple_confirm_bb_rsi_stoch",
                "display_name": "Triple: BB + RSI + Stochastic",
                "category": "confluence",
                "params": ["bb_length", "bb_mult", "rsi_length", "rsi_oversold", "rsi_overbought", 
                          "stoch_k", "stoch_oversold", "stoch_overbought", "adx_threshold"],
            },
            {
                "name": "quad_oscillator",
                "display_name": "Quad: RSI + StochRSI + CCI + Williams",
                "category": "confluence",
                "params": ["rsi_length", "rsi_oversold", "rsi_overbought", "cci_threshold", 
                          "willr_oversold", "willr_overbought", "adx_threshold"],
            },
            {
                "name": "volume_price_combo",
                "display_name": "Volume + Price Combo (CMF + Force + BB)",
                "category": "confluence",
                "params": ["cmf_length", "bb_length", "rsi_length", "adx_threshold"],
            },
            {
                "name": "trend_alignment",
                "display_name": "Trend Alignment (Supertrend + HMA + ZLEMA)",
                "category": "confluence",
                "params": ["st_length", "st_mult", "hma_length", "zlema_length"],
            },
            {
                "name": "mean_reversion_suite",
                "display_name": "Mean Reversion Suite (Z-Score + BB%B + VWAP)",
                "category": "confluence",
                "params": ["z_period", "bb_length", "rsi_length", "adx_threshold"],
            },
            {
                "name": "squeeze_momentum",
                "display_name": "Squeeze + Momentum (BB + AO + RSI)",
                "category": "confluence",
                "params": ["bb_length", "rsi_length"],
            },
            {
                "name": "fisher_stoch_keltner",
                "display_name": "Fisher + Stochastic + Keltner",
                "category": "confluence",
                "params": ["fisher_length", "stoch_k", "stoch_oversold", "stoch_overbought", 
                          "kc_length", "kc_mult", "adx_threshold"],
            },
            {
                "name": "hurst_regime_multi",
                "display_name": "Hurst Regime + Multi-Indicator",
                "category": "confluence",
                "params": ["hurst_period", "rsi_length", "rsi_oversold", "rsi_overbought", "bb_length"],
            },
            {
                "name": "atr_spike_extreme",
                "display_name": "ATR Spike + Extreme Readings",
                "category": "confluence",
                "params": ["atr_length", "atr_spike_mult", "rsi_length", "rsi_oversold", "rsi_overbought", "adx_threshold"],
            },
            {
                "name": "ultimate_confluence",
                "display_name": "Ultimate 6-Indicator Confluence",
                "category": "confluence",
                "params": ["bb_length", "bb_mult", "rsi_length", "rsi_oversold", "rsi_overbought",
                          "stoch_k", "stoch_oversold", "stoch_overbought", "cci_threshold", "adx_threshold"],
            },
        ])
        
        # ===== MACD-BASED STRATEGIES =====
        strategies.extend([
            {
                "name": "macd_bb",
                "display_name": "MACD + Bollinger Bands",
                "category": "momentum",
                "params": ["macd_fast", "macd_slow", "macd_signal", "bb_length", "bb_mult"],
            },
            {
                "name": "macd_histogram_divergence",
                "display_name": "MACD Histogram Momentum",
                "category": "momentum",
                "params": ["macd_fast", "macd_slow", "macd_signal", "rsi_length"],
            },
        ])
        
        # ===== VOLUME-BASED STRATEGIES =====
        strategies.extend([
            {
                "name": "volume_confirmed_trend",
                "display_name": "Volume Confirmed Trend",
                "category": "volume",
                "params": ["st_length", "st_mult", "rsi_length", "vol_length"],
            },
            {
                "name": "cmf_pressure",
                "display_name": "CMF Accumulation/Distribution",
                "category": "volume",
                "params": ["cmf_length", "cmf_threshold", "rsi_length", "adx_threshold"],
            },
            {
                "name": "force_index_trend",
                "display_name": "Force Index Trend",
                "category": "volume",
                "params": ["force_length", "rsi_length"],
            },
        ])
        
        # ===== ML-BASED STRATEGIES =====
        strategies.extend([
            {
                "name": "nadaraya_watson_reversion",
                "display_name": "Nadaraya-Watson Mean Reversion",
                "category": "ml_based",
                "params": ["nw_bandwidth", "nw_mult", "rsi_length", "adx_threshold"],
            },
            {
                "name": "divergence_3wave",
                "display_name": "3-Wave Divergence (Vdubus)",
                "category": "ml_based",
                "params": ["div_lookback", "rsi_length", "adx_threshold"],
            },
            {
                "name": "cvd_divergence",
                "display_name": "CVD-Price Divergence",
                "category": "ml_based",
                "params": ["rsi_length", "adx_threshold"],
            },
            {
                "name": "order_block_bounce",
                "display_name": "Order Block Bounce (SMC)",
                "category": "ml_based",
                "params": ["ob_lookback", "ob_min_move", "rsi_length", "adx_threshold"],
            },
            {
                "name": "fvg_fill",
                "display_name": "Fair Value Gap Fill (SMC)",
                "category": "ml_based",
                "params": ["rsi_length", "adx_threshold"],
            },
            {
                "name": "squeeze_momentum_breakout",
                "display_name": "Squeeze Momentum Breakout",
                "category": "ml_based",
                "params": ["squeeze_kc_mult", "rsi_length"],
            },
            {
                "name": "connors_rsi_extreme",
                "display_name": "Connors RSI Extreme",
                "category": "ml_based",
                "params": ["crsi_rsi_len", "crsi_streak_len", "crsi_roc_len", "adx_threshold"],
            },
            {
                "name": "smc_divergence_confluence",
                "display_name": "SMC + Divergence Confluence",
                "category": "ml_based",
                "params": ["div_lookback", "rsi_length", "adx_threshold"],
            },
            {
                "name": "kernel_trend_follow",
                "display_name": "Kernel Regression Trend",
                "category": "ml_based",
                "params": ["nw_bandwidth", "rsi_length"],
            },
            {
                "name": "ml_feature_ensemble",
                "display_name": "ML Feature Ensemble",
                "category": "ml_based",
                "params": ["rsi_length", "adx_threshold"],
            },
        ])

        # ===== SIMPLE STRATEGIES (Beginner-Friendly, Proven Patterns) =====
        # These are basic strategies that often outperform complex ones
        strategies.extend([
            # --- Percentage Drop/Rise Strategies ---
            {
                "name": "pct_drop_buy",
                "display_name": "Percentage Drop Buy (1-5%)",
                "category": "simple",
                "complexity": "simple",
                "params": ["drop_percent", "rise_percent", "pct_lookback"],
                "description": "Buy after X% drop from recent high, sell after Y% rise from entry"
            },
            {
                "name": "pct_drop_consecutive",
                "display_name": "Consecutive Red Candles + Drop",
                "category": "simple",
                "complexity": "simple",
                "params": ["red_candle_count", "min_drop_percent", "profit_target_percent"],
                "description": "Buy after N consecutive red candles with X% total drop"
            },

            # --- Simple Moving Average Strategies ---
            {
                "name": "simple_sma_cross",
                "display_name": "Simple SMA Cross (Fast/Slow)",
                "category": "simple",
                "complexity": "simple",
                "params": ["sma_fast", "sma_slow"],
                "description": "Classic fast/slow SMA crossover"
            },
            {
                "name": "price_vs_sma",
                "display_name": "Price vs SMA",
                "category": "simple",
                "complexity": "simple",
                "params": ["single_sma_period", "confirm_candles"],
                "description": "Buy when price crosses above SMA, sell when crosses below"
            },
            {
                "name": "triple_sma_align",
                "display_name": "Triple SMA Alignment",
                "category": "simple",
                "complexity": "simple",
                "params": ["triple_fast", "triple_medium", "triple_slow"],
                "description": "All three SMAs aligned (fast > medium > slow for long)"
            },

            # --- Price Action Strategies ---
            {
                "name": "consecutive_candles_reversal",
                "display_name": "Consecutive Candle Reversal",
                "category": "simple",
                "complexity": "simple",
                "params": ["consec_count", "reversal_size_mult"],
                "description": "N red candles followed by bullish reversal candle"
            },
            {
                "name": "inside_bar_breakout",
                "display_name": "Inside Bar Breakout",
                "category": "simple",
                "complexity": "simple",
                "params": ["min_inside_bars"],
                "description": "Trade breakout from inside bar consolidation"
            },
            {
                "name": "range_breakout_simple",
                "display_name": "Simple Range Breakout",
                "category": "simple",
                "complexity": "simple",
                "params": ["range_lookback", "breakout_buffer_pct"],
                "description": "Break above recent high / below recent low"
            },
            {
                "name": "support_resistance_simple",
                "display_name": "Simple S/R Bounce",
                "category": "simple",
                "complexity": "simple",
                "params": ["sr_lookback", "sr_tolerance_pct"],
                "description": "Buy at support level, sell at resistance"
            },

            # --- Simple Momentum Strategies ---
            {
                "name": "simple_rsi_extreme",
                "display_name": "Simple RSI Extremes",
                "category": "simple",
                "complexity": "simple",
                "params": ["simple_rsi_period", "simple_oversold", "simple_overbought"],
                "description": "Buy when RSI oversold, sell when overbought"
            },
            {
                "name": "candle_ratio_momentum",
                "display_name": "Green/Red Candle Ratio",
                "category": "simple",
                "complexity": "simple",
                "params": ["ratio_lookback", "ratio_threshold"],
                "description": "Buy when green candle ratio exceeds threshold after selling pressure"
            },
            {
                "name": "engulfing_pattern",
                "display_name": "Engulfing Candle Pattern",
                "category": "simple",
                "complexity": "simple",
                "params": ["engulf_min_size_mult"],
                "description": "Trade bullish/bearish engulfing patterns"
            },
            {
                "name": "doji_reversal",
                "display_name": "Doji Reversal Pattern",
                "category": "simple",
                "complexity": "simple",
                "params": ["doji_body_pct", "doji_confirm_candles"],
                "description": "Trade reversal after doji at extremes"
            },
        ])

        # ===== ML-BASED STRATEGIES (Deep Learning Models) =====
        # These strategies use pre-trained ML models for signal generation
        # Models must be trained via /api/ml-train before use
        strategies.extend([
            {
                "name": "ml_xgboost",
                "display_name": "XGBoost Classifier",
                "category": "ml_based",
                "complexity": "ml",
                "params": [],  # ML models have no optimizable params - they're pre-trained
                "description": "Gradient boosting classifier predicting price direction"
            },
            {
                "name": "ml_lightgbm",
                "display_name": "LightGBM Classifier",
                "category": "ml_based",
                "complexity": "ml",
                "params": [],
                "description": "Microsoft's gradient boosting for fast classification"
            },
            {
                "name": "ml_catboost",
                "display_name": "CatBoost Classifier",
                "category": "ml_based",
                "complexity": "ml",
                "params": [],
                "description": "Yandex's gradient boosting with categorical feature handling"
            },
            {
                "name": "ml_lstm",
                "display_name": "LSTM Neural Network",
                "category": "ml_based",
                "complexity": "ml",
                "params": [],
                "description": "Recurrent neural network for sequence prediction"
            },
            {
                "name": "ml_gru",
                "display_name": "GRU Neural Network",
                "category": "ml_based",
                "complexity": "ml",
                "params": [],
                "description": "Gated recurrent unit - lighter alternative to LSTM"
            },
            {
                "name": "ml_transformer",
                "display_name": "Transformer Time-Series",
                "category": "ml_based",
                "complexity": "ml",
                "params": [],
                "description": "Attention-based model for time-series prediction"
            },
            {
                "name": "ml_rl_ppo",
                "display_name": "RL Agent (PPO)",
                "category": "ml_based",
                "complexity": "ml",
                "params": [],
                "description": "Reinforcement learning agent using PPO algorithm"
            },
            {
                "name": "ml_rl_dqn",
                "display_name": "RL Agent (DQN)",
                "category": "ml_based",
                "complexity": "ml",
                "params": [],
                "description": "Reinforcement learning agent using DQN algorithm"
            },
            # === ML ENSEMBLE STRATEGIES ===
            {
                "name": "ml_ensemble_majority",
                "display_name": "ML Ensemble (Majority Vote)",
                "category": "ml_based",
                "complexity": "ml",
                "params": [],
                "description": "Trade when 2+ of 3 ML models agree on direction"
            },
            {
                "name": "ml_ensemble_unanimous",
                "display_name": "ML Ensemble (Unanimous)",
                "category": "ml_based",
                "complexity": "ml",
                "params": [],
                "description": "Trade only when ALL 3 ML models agree (highest confidence)"
            },
            {
                "name": "ml_xgboost_rsi_confirmed",
                "display_name": "XGBoost + RSI Confirmation",
                "category": "ml_based",
                "complexity": "ml",
                "params": ["rsi_length", "rsi_oversold", "rsi_overbought"],
                "description": "XGBoost signal confirmed by RSI extreme"
            },
            {
                "name": "ml_lightgbm_bb_confirmed",
                "display_name": "LightGBM + BB Confirmation",
                "category": "ml_based",
                "complexity": "ml",
                "params": ["bb_length", "bb_mult"],
                "description": "LightGBM signal confirmed by Bollinger Band touch"
            },
            {
                "name": "ml_ensemble_adx_filter",
                "display_name": "ML Ensemble + ADX Filter",
                "category": "ml_based",
                "complexity": "ml",
                "params": ["adx_threshold"],
                "description": "ML majority vote filtered by ADX trend strength"
            },
        ])

        return strategies


# =============================================================================
# PARAMETER RANGES - Wide ranges for exhaustive testing
# =============================================================================

PARAM_RANGES = {
    # Bollinger Bands
    "bb_length": (5, 50),
    "bb_mult": (0.5, 4.0),
    
    # RSI
    "rsi_length": (3, 30),
    "rsi_oversold": (5, 45),
    "rsi_overbought": (55, 95),
    
    # Stochastic
    "stoch_k": (3, 21),
    "stoch_d": (2, 9),
    "stoch_oversold": (5, 35),
    "stoch_overbought": (65, 95),
    "stoch_rsi_length": (7, 21),
    
    # ADX
    "adx_length": (5, 30),
    "adx_threshold": (10, 50),
    
    # CCI
    "cci_length": (5, 30),
    "cci_threshold": (50, 200),
    
    # Williams %R
    "willr_length": (5, 30),
    "willr_oversold": (-95, -70),
    "willr_overbought": (-30, -5),
    
    # Keltner
    "kc_length": (5, 30),
    "kc_mult": (0.5, 3.0),
    
    # Donchian
    "dc_length": (10, 50),
    
    # MACD
    "macd_fast": (5, 20),
    "macd_slow": (15, 40),
    "macd_signal": (3, 15),
    
    # Moving Averages
    "ema_fast": (5, 20),
    "ema_slow": (20, 100),
    "hma_length": (10, 100),
    "zlema_fast": (5, 20),
    "zlema_slow": (20, 100),
    
    # Supertrend
    "st_length": (5, 20),
    "st_mult": (1.0, 5.0),
    
    # DaviddTech
    "jma_length": (4, 50),
    "jma_phase": (0, 100),
    "mcginley_length": (10, 200),
    "mcginley_k": (0.4, 0.8),
    "stiff_length": (20, 100),
    "stiff_ma": (30, 150),
    "stiff_threshold": (40, 80),
    "tdfi_length": (5, 30),
    "vq_length": (10, 50),
    "vq_threshold": (0.3, 1.0),
    "trendilo_length": (30, 80),
    "t3_fast": (3, 10),
    "t3_slow": (15, 40),
    "t3_length": (3, 20),
    "rf_period": (50, 200),
    "rf_mult": (1.5, 5.0),
    "lwpi_length": (10, 150),
    "alma_length": (5, 20),
    "alma_offset": (0.3, 1.0),
    "flat_threshold": (15, 40),
    
    # Advanced
    "fisher_length": (5, 20),
    "fisher_threshold": (1.0, 2.5),
    "z_period": (10, 50),
    "z_threshold": (1.5, 3.0),
    "vwap_threshold": (1.0, 3.0),
    "squeeze_threshold": (0.5, 0.9),
    "atr_length": (10, 20),
    "atr_mult": (1.5, 4.0),
    "atr_spike_mult": (1.5, 3.0),
    "hurst_period": (50, 150),
    "cmf_length": (10, 30),
    "cmf_threshold": (0.05, 0.2),
    "force_length": (10, 20),
    "vol_length": (20, 80),
    "score_threshold": (3, 5),
    
    # Risk Management - PERCENTAGE-BASED (matches TradingView exactly)
    "tp_percent": (0.3, 5.0),   # Take Profit as % of entry price
    "sl_percent": (1.0, 10.0),  # Stop Loss as % of entry price

    # Legacy ATR-based (kept for backwards compatibility)
    "sl_atr_mult": (0.5, 5.0),
    "tp_ratio": (0.5, 5.0),
    
    # ML-Based Indicators
    "nw_bandwidth": (5, 20),
    "nw_mult": (1.5, 4.0),
    "div_lookback": (20, 100),
    "squeeze_kc_mult": (1.0, 2.5),
    "crsi_rsi_len": (2, 5),
    "crsi_streak_len": (2, 5),
    "crsi_roc_len": (50, 150),
    "ob_lookback": (5, 20),
    "ob_min_move": (1.0, 3.0),

    # === SIMPLE STRATEGY PARAMETERS ===
    # Percentage-based strategies
    "drop_percent": (0.5, 5.0),           # Buy after this % drop
    "rise_percent": (0.5, 5.0),           # Sell after this % rise
    "pct_lookback": (5, 30),              # Lookback for recent high
    "red_candle_count": (2, 6),           # Consecutive red candles
    "min_drop_percent": (0.5, 4.0),       # Min drop during red candles
    "profit_target_percent": (0.5, 3.0),  # Profit target after entry

    # Simple MA strategies
    "sma_fast": (5, 15),                  # Fast SMA period
    "sma_slow": (15, 50),                 # Slow SMA period
    "single_sma_period": (20, 100),       # Single SMA for price crossover
    "confirm_candles": (1, 3),            # Candles to confirm crossover
    "triple_fast": (5, 12),               # Triple SMA fast
    "triple_medium": (12, 25),            # Triple SMA medium
    "triple_slow": (25, 55),              # Triple SMA slow

    # Price action strategies
    "consec_count": (2, 5),               # Consecutive candles for pattern
    "reversal_size_mult": (0.5, 2.0),     # Reversal candle size multiplier
    "min_inside_bars": (1, 3),            # Min inside bars before breakout
    "range_lookback": (10, 30),           # Lookback for range high/low
    "breakout_buffer_pct": (0.1, 1.0),    # Buffer % for breakout confirmation
    "sr_lookback": (20, 100),             # S/R detection lookback
    "sr_tolerance_pct": (0.2, 1.0),       # Tolerance for S/R level bounce

    # Simple momentum strategies
    "simple_rsi_period": (7, 21),         # RSI period for simple strategy
    "simple_oversold": (20, 35),          # Simple oversold threshold
    "simple_overbought": (65, 80),        # Simple overbought threshold
    "ratio_lookback": (5, 20),            # Lookback for candle ratio
    "ratio_threshold": (0.6, 0.85),       # Green candle ratio threshold
    "engulf_min_size_mult": (1.2, 2.5),   # Min size for engulfing candle
    "doji_body_pct": (0.05, 0.15),        # Doji body % of range
    "doji_confirm_candles": (1, 3),       # Candles to confirm doji reversal
}


# =============================================================================
# UNIFIED BACKTESTER
# =============================================================================

class UnifiedBacktester:
    """
    Unified backtester that can test ANY strategy with ANY parameters.
    Calculates comprehensive metrics including equity curve smoothness.
    """
    
    def __init__(self, df: pd.DataFrame, capital: float = 1000.0, risk_percent: float = 2.0):
        self.df = df.copy()
        self.capital = capital
        self.risk_percent = risk_percent
        
        # Pre-calculate ATR
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        self.df['atr'] = atr.average_true_range()
        
        # Pre-calculate ALL indicators once (expensive but done once)
        self._calculate_all_indicators()
    
    def _calculate_all_indicators(self):
        """Calculate ALL indicators upfront for efficiency"""
        df = self.df
        
        # === Standard TA Indicators ===
        
        # RSI variants
        for length in [6, 7, 14, 21]:
            df[f'rsi_{length}'] = RSIIndicator(df['close'], window=length).rsi()
        
        # Stochastic
        for k in [9, 14]:
            for d in [3, 5]:
                stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=k, smooth_window=d)
                df[f'stoch_k_{k}_{d}'] = stoch.stoch()
                df[f'stoch_d_{k}_{d}'] = stoch.stoch_signal()
        
        # ADX
        for length in [10, 14, 20]:
            adx = ADXIndicator(df['high'], df['low'], df['close'], window=length)
            df[f'adx_{length}'] = adx.adx()
            df[f'di_plus_{length}'] = adx.adx_pos()
            df[f'di_minus_{length}'] = adx.adx_neg()
        
        # Bollinger Bands
        for length in [14, 20, 25]:
            for mult in [1.5, 2.0, 2.5]:
                bb = BollingerBands(df['close'], window=length, window_dev=mult)
                df[f'bb_upper_{length}_{mult}'] = bb.bollinger_hband()
                df[f'bb_lower_{length}_{mult}'] = bb.bollinger_lband()
                df[f'bb_mid_{length}_{mult}'] = bb.bollinger_mavg()
                df[f'bb_width_{length}_{mult}'] = bb.bollinger_wband()
        
        # Keltner Channel
        for length in [14, 20]:
            kc = KeltnerChannel(df['high'], df['low'], df['close'], window=length, window_atr=length)
            df[f'kc_upper_{length}'] = kc.keltner_channel_hband()
            df[f'kc_lower_{length}'] = kc.keltner_channel_lband()
        
        # CCI
        for length in [14, 20]:
            df[f'cci_{length}'] = CCIIndicator(df['high'], df['low'], df['close'], window=length).cci()
        
        # Williams %R
        for length in [10, 14, 21]:
            df[f'willr_{length}'] = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=length).williams_r()
        
        # MACD
        for fast, slow, signal in [(12, 26, 9), (8, 21, 5)]:
            macd = MACD(df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
            df[f'macd_{fast}_{slow}_{signal}'] = macd.macd()
            df[f'macd_signal_{fast}_{slow}_{signal}'] = macd.macd_signal()
            df[f'macd_hist_{fast}_{slow}_{signal}'] = macd.macd_diff()
        
        # Moving Averages
        for length in [9, 21, 55, 100, 200]:
            df[f'ema_{length}'] = EMAIndicator(df['close'], window=length).ema_indicator()
            df[f'sma_{length}'] = SMAIndicator(df['close'], window=length).sma_indicator()
        
        # Donchian
        for length in [10, 20, 50]:
            dc = DonchianChannel(df['high'], df['low'], df['close'], window=length)
            df[f'dc_upper_{length}'] = dc.donchian_channel_hband()
            df[f'dc_lower_{length}'] = dc.donchian_channel_lband()
        
        # === DaviddTech Indicators ===
        try:
            self.df = calculate_daviddtech_indicators(self.df)
        except Exception as e:
            print(f"Warning: DaviddTech indicators failed: {e}")
        
        # === Advanced ML Indicators ===
        try:
            self.df = calculate_all_advanced_indicators(self.df)
        except Exception as e:
            print(f"Warning: Advanced indicators failed: {e}")
        
        # === ML-Based Indicators (Nadaraya-Watson, CVD, SMC, etc.) ===
        try:
            self.df = calculate_ml_indicators(self.df)
        except Exception as e:
            print(f"Warning: ML indicators failed: {e}")
        
        # Store reference
        df = self.df

    def get_signals(self, strategy_name: str, params: Dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate long/short signals for any strategy.
        Returns: (long_signal, short_signal, filter_signal)
        """
        df = self.df
        
        # Helper to get indicator with fallback
        def get_ind(name, default=None):
            if name in df.columns:
                return df[name]
            return pd.Series(default if default is not None else 0, index=df.index)
        
        # Get common params with defaults
        adx_thresh = params.get('adx_threshold', 25)
        rsi_os = params.get('rsi_oversold', 30)
        rsi_ob = params.get('rsi_overbought', 70)
        
        # Default sideways filter
        adx = get_ind('adx_14', 20)
        sideways = adx < adx_thresh
        
        # === MEAN REVERSION ===
        if strategy_name == "bb_rsi_classic":
            bb_len = params.get('bb_length', 20)
            bb_mult = params.get('bb_mult', 2.0)
            rsi_len = params.get('rsi_length', 14)
            bb_upper = get_ind(f'bb_upper_{bb_len}_{bb_mult}', df['close'] * 1.02)
            bb_lower = get_ind(f'bb_lower_{bb_len}_{bb_mult}', df['close'] * 0.98)
            rsi = get_ind(f'rsi_{rsi_len}', 50)
            long_sig = (df['close'] <= bb_lower) & (rsi < rsi_os)
            short_sig = (df['close'] >= bb_upper) & (rsi > rsi_ob)
            
        elif strategy_name == "bb_rsi_tight":
            bb_len = params.get('bb_length', 20)
            bb_mult = params.get('bb_mult', 2.0)
            bb_upper = get_ind(f'bb_upper_{bb_len}_{bb_mult}', df['close'] * 1.02)
            bb_lower = get_ind(f'bb_lower_{bb_len}_{bb_mult}', df['close'] * 0.98)
            rsi = get_ind('rsi_14', 50)
            long_sig = (df['close'] <= bb_lower) & (rsi < 25)
            short_sig = (df['close'] >= bb_upper) & (rsi > 75)
            sideways = adx < 20  # Tighter ADX filter
            
        elif strategy_name == "bb_stoch":
            bb_len = params.get('bb_length', 20)
            bb_mult = params.get('bb_mult', 2.0)
            stoch_k = params.get('stoch_k', 14)
            stoch_os = params.get('stoch_oversold', 20)
            stoch_ob = params.get('stoch_overbought', 80)
            bb_upper = get_ind(f'bb_upper_{bb_len}_{bb_mult}', df['close'] * 1.02)
            bb_lower = get_ind(f'bb_lower_{bb_len}_{bb_mult}', df['close'] * 0.98)
            stoch = get_ind(f'stoch_k_{stoch_k}_3', 50)
            long_sig = (df['close'] <= bb_lower) & (stoch < stoch_os)
            short_sig = (df['close'] >= bb_upper) & (stoch > stoch_ob)
            
        elif strategy_name == "keltner_rsi":
            kc_len = params.get('kc_length', 20)
            rsi_len = params.get('rsi_length', 14)
            kc_upper = get_ind(f'kc_upper_{kc_len}', df['close'] * 1.02)
            kc_lower = get_ind(f'kc_lower_{kc_len}', df['close'] * 0.98)
            rsi = get_ind(f'rsi_{rsi_len}', 50)
            long_sig = (df['close'] <= kc_lower) & (rsi < rsi_os)
            short_sig = (df['close'] >= kc_upper) & (rsi > rsi_ob)
            
        elif strategy_name == "z_score_reversion":
            z_thresh = params.get('z_threshold', 2.0)
            z_score = get_ind('z_score', 0)
            long_sig = z_score < -z_thresh
            short_sig = z_score > z_thresh
            
        elif strategy_name == "bb_percent_b_extreme":
            bb_pct = get_ind('bb_percent_b', 0.5)
            long_sig = bb_pct < 0
            short_sig = bb_pct > 1
            
        elif strategy_name == "vwap_mean_reversion":
            vwap_dist = get_ind('vwap_distance', 0)
            vwap_thresh = params.get('vwap_threshold', 2.0)
            long_sig = vwap_dist < -vwap_thresh
            short_sig = vwap_dist > vwap_thresh
            
        elif strategy_name == "donchian_fade":
            dc_len = params.get('dc_length', 20)
            dc_upper = get_ind(f'dc_upper_{dc_len}', df['close'] * 1.02)
            dc_lower = get_ind(f'dc_lower_{dc_len}', df['close'] * 0.98)
            long_sig = df['close'] <= dc_lower
            short_sig = df['close'] >= dc_upper
            
        # === OSCILLATOR ===
        elif strategy_name == "stoch_extreme":
            stoch_k = params.get('stoch_k', 14)
            stoch_os = params.get('stoch_oversold', 20)
            stoch_ob = params.get('stoch_overbought', 80)
            stoch_k_val = get_ind(f'stoch_k_{stoch_k}_3', 50)
            stoch_d_val = get_ind(f'stoch_d_{stoch_k}_3', 50)
            long_sig = (stoch_k_val < stoch_os) & (stoch_d_val < stoch_os)
            short_sig = (stoch_k_val > stoch_ob) & (stoch_d_val > stoch_ob)
            
        elif strategy_name == "rsi_extreme":
            rsi_len = params.get('rsi_length', 14)
            rsi = get_ind(f'rsi_{rsi_len}', 50)
            long_sig = rsi < rsi_os
            short_sig = rsi > rsi_ob
            
        elif strategy_name == "williams_r":
            willr_len = params.get('willr_length', 14)
            willr_os = params.get('willr_oversold', -80)
            willr_ob = params.get('willr_overbought', -20)
            willr = get_ind(f'willr_{willr_len}', -50)
            long_sig = willr < willr_os
            short_sig = willr > willr_ob
            
        elif strategy_name == "cci_extreme":
            cci_len = params.get('cci_length', 20)
            cci_thresh = params.get('cci_threshold', 100)
            cci = get_ind(f'cci_{cci_len}', 0)
            long_sig = cci < -cci_thresh
            short_sig = cci > cci_thresh
            
        elif strategy_name == "stoch_rsi":
            stoch_rsi = get_ind('stoch_rsi_k', 50)
            stoch_os = params.get('stoch_oversold', 20)
            stoch_ob = params.get('stoch_overbought', 80)
            long_sig = stoch_rsi < stoch_os
            short_sig = stoch_rsi > stoch_ob
            
        elif strategy_name == "fisher_transform":
            fisher = get_ind('fisher', 0)
            fisher_thresh = params.get('fisher_threshold', 1.5)
            long_sig = (fisher < -fisher_thresh) & (fisher > fisher.shift(1))
            short_sig = (fisher > fisher_thresh) & (fisher < fisher.shift(1))
            
        elif strategy_name == "awesome_oscillator":
            ao = get_ind('ao', 0)
            ao_cross = get_ind('ao_cross', 0)
            long_sig = (ao_cross == 1) & (ao_cross.shift(1) == -1)
            short_sig = (ao_cross == -1) & (ao_cross.shift(1) == 1)
            
        # === TREND FOLLOWING ===
        elif strategy_name == "supertrend_follow":
            st_dir = get_ind('supertrend_dir', 1)
            rsi = get_ind('rsi_14', 50)
            long_sig = (st_dir > 0) & (rsi > 50)
            short_sig = (st_dir < 0) & (rsi < 50)
            sideways = pd.Series(True, index=df.index)  # No sideways filter for trend following
            
        elif strategy_name == "macd_trend":
            macd = get_ind('macd_12_26_9', 0)
            macd_sig = get_ind('macd_signal_12_26_9', 0)
            macd_hist = get_ind('macd_hist_12_26_9', 0)
            long_sig = (macd > macd_sig) & (macd_hist > 0)
            short_sig = (macd < macd_sig) & (macd_hist < 0)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "adx_trend":
            adx_len = params.get('adx_length', 14)
            adx_val = get_ind(f'adx_{adx_len}', 20)
            di_plus = get_ind(f'di_plus_{adx_len}', 0)
            di_minus = get_ind(f'di_minus_{adx_len}', 0)
            rsi = get_ind('rsi_14', 50)
            long_sig = (adx_val > adx_thresh) & (di_plus > di_minus) & (rsi > 50)
            short_sig = (adx_val > adx_thresh) & (di_minus > di_plus) & (rsi < 50)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "ema_crossover":
            ema_fast = params.get('ema_fast', 9)
            ema_slow = params.get('ema_slow', 21)
            ema_f = get_ind(f'ema_{ema_fast}', df['close'])
            ema_s = get_ind(f'ema_{ema_slow}', df['close'])
            long_sig = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
            short_sig = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))
            
        elif strategy_name == "hma_trend":
            hma_val = get_ind('hma', df['close'])
            hma_slope = get_ind('hma_slope', 0)
            rsi = get_ind('rsi_14', 50)
            long_sig = (hma_slope > 0) & (df['close'] > hma_val) & (rsi > 50)
            short_sig = (hma_slope < 0) & (df['close'] < hma_val) & (rsi < 50)
            
        elif strategy_name == "zlema_momentum":
            zlema_cross = get_ind('zlema_cross', 0)
            rsi = get_ind('rsi_14', 50)
            long_sig = (zlema_cross > 0) & (rsi > 50)
            short_sig = (zlema_cross < 0) & (rsi < 50)
            
        elif strategy_name == "multi_ma_confluence":
            ema9 = get_ind('ema_9', df['close'])
            ema21 = get_ind('ema_21', df['close'])
            ema55 = get_ind('ema_55', df['close'])
            sma200 = get_ind('sma_200', df['close'])
            long_sig = (ema9 > ema21) & (ema21 > ema55) & (df['close'] > sma200)
            short_sig = (ema9 < ema21) & (ema21 < ema55) & (df['close'] < sma200)
            
        # === BREAKOUT ===
        elif strategy_name == "bb_squeeze_breakout":
            bb_width = get_ind('bb_width_20_2.0', 0.05)
            bb_mid = get_ind('bb_mid_20_2.0', df['close'])
            avg_width = bb_width.rolling(50).mean()
            squeeze_thresh = params.get('squeeze_threshold', 0.75)
            in_squeeze = bb_width < avg_width * squeeze_thresh
            squeeze_release = in_squeeze.shift(1) & ~in_squeeze
            long_sig = squeeze_release & (df['close'] > bb_mid)
            short_sig = squeeze_release & (df['close'] < bb_mid)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "donchian_breakout":
            dc_len = params.get('dc_length', 20)
            dc_upper = get_ind(f'dc_upper_{dc_len}', df['close'] * 1.02)
            dc_lower = get_ind(f'dc_lower_{dc_len}', df['close'] * 0.98)
            long_sig = (df['close'] > dc_upper) & (df['close'].shift(1) <= dc_upper.shift(1))
            short_sig = (df['close'] < dc_lower) & (df['close'].shift(1) >= dc_lower.shift(1))
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "trendilo_breakout":
            trendilo_val = get_ind('trendilo_52', df['close'])
            upper = get_ind('trendilo_upper_52', df['close'] * 1.02)
            lower = get_ind('trendilo_lower_52', df['close'] * 0.98)
            long_sig = (df['close'] > upper) & (df['close'].shift(1) <= upper.shift(1))
            short_sig = (df['close'] < lower) & (df['close'].shift(1) >= lower.shift(1))
            sideways = pd.Series(True, index=df.index)
            
        # === DAVIDDTECH ===
        elif strategy_name == "stiff_surge_v1":
            jma_val = get_ind('jma_43_84', df['close'])
            stiff = get_ind('stiff_smooth_60_100', 50)
            tdfi_val = get_ind('tdfi_15', 0)
            is_flat = get_ind('is_flat_15', False)
            long_sig = (df['close'] > jma_val) & (stiff > 50) & (tdfi_val > 0.05)
            short_sig = (df['close'] < jma_val) & (stiff < 50) & (tdfi_val < -0.05)
            sideways = ~is_flat
            
        elif strategy_name == "stiff_surge_v2":
            jma_val = get_ind('jma_7_50', df['close'])
            vq = get_ind('vq_14', 0)
            long_sig = (jma_val > jma_val.shift(1)) & (vq > 0) & (df['close'] > jma_val)
            short_sig = (jma_val < jma_val.shift(1)) & (vq < 0) & (df['close'] < jma_val)
            
        elif strategy_name == "mcginley_trend":
            md = get_ind('mcginley_130_0.6', df['close'])
            lwpi_val = get_ind('lwpi_130', -50)
            long_sig = (df['close'] > md) & (md > md.shift(1)) & (lwpi_val > lwpi_val.shift(1))
            short_sig = (df['close'] < md) & (md < md.shift(1)) & (lwpi_val < lwpi_val.shift(1))
            
        elif strategy_name == "trendhoo_v1":
            trendilo_val = get_ind('trendilo_52', df['close'])
            hma_val = get_ind('hma_65', df['close'])
            t3_val = get_ind('t3_5', df['close'])
            stoch = get_ind('stoch_k_14_3', 50)
            long_sig = (df['close'] > trendilo_val) & (df['close'] > hma_val) & (df['close'] > t3_val) & (stoch < 40)
            short_sig = (df['close'] < trendilo_val) & (df['close'] < hma_val) & (df['close'] < t3_val) & (stoch > 60)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "trendhoo_v2":
            alma_val = get_ind('alma_11_0.85', df['close'])
            hma_val = get_ind('hma_65', df['close'])
            stoch = get_ind('stoch_k_14_3', 50)
            long_sig = (df['close'] > alma_val) & (alma_val > alma_val.shift(1)) & (df['close'] > hma_val) & (stoch < 70)
            short_sig = (df['close'] < alma_val) & (alma_val < alma_val.shift(1)) & (df['close'] < hma_val) & (stoch > 30)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "macd_liquidity_spectrum":
            macd = get_ind('macd_12_26_9', 0)
            macd_sig = get_ind('macd_signal_12_26_9', 0)
            macd_hist = get_ind('macd_hist_12_26_9', 0)
            rf_dir = get_ind('rf_dir_100_3.0', 0)
            macd_cross_up = (macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1))
            macd_cross_dn = (macd < macd_sig) & (macd.shift(1) >= macd_sig.shift(1))
            long_sig = (macd_cross_up | ((macd > 0) & (macd_hist > 0))) & (rf_dir > 0)
            short_sig = (macd_cross_dn | ((macd < 0) & (macd_hist < 0))) & (rf_dir < 0)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "precision_trend_mastery":
            adx_val = get_ind('adx_14', 20)
            di_plus = get_ind('di_plus_14', 0)
            di_minus = get_ind('di_minus_14', 0)
            rf_dir = get_ind('rf_dir_164_4.5', get_ind('rf_dir_100_3.0', 0))
            is_flat = get_ind('is_flat_40', False)
            long_sig = (adx_val > 34) & (di_plus > di_minus) & (rf_dir > 0)
            short_sig = (adx_val > 34) & (di_minus > di_plus) & (rf_dir < 0)
            sideways = ~is_flat
            
        elif strategy_name == "t3_nexus_stiff":
            t3_fast = get_ind('t3_5', df['close'])
            t3_slow = get_ind('t3_20', df['close'])
            stiff = get_ind('stiff_smooth_60_100', 50)
            t3_bullish = t3_fast > t3_slow
            t3_cross_up = t3_bullish & ~t3_bullish.shift(1).fillna(False)
            t3_cross_dn = ~t3_bullish & t3_bullish.shift(1).fillna(True)
            long_sig = (t3_cross_up | (t3_bullish & (df['close'] > t3_fast))) & (stiff > 60)
            short_sig = (t3_cross_dn | (~t3_bullish & (df['close'] < t3_fast))) & (stiff < 40)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "range_filter_adx":
            rf_dir = get_ind('rf_dir_100_3.0', 0)
            adx_val = get_ind('adx_14', 20)
            rf_change_up = (rf_dir > 0) & (rf_dir.shift(1) <= 0)
            rf_change_dn = (rf_dir < 0) & (rf_dir.shift(1) >= 0)
            long_sig = (rf_change_up | (rf_dir > 0)) & (adx_val > adx_thresh)
            short_sig = (rf_change_dn | (rf_dir < 0)) & (adx_val > adx_thresh)
            is_flat = get_ind('is_flat_15', False)
            sideways = ~is_flat
            
        elif strategy_name == "jma_volatility_quality":
            jma_val = get_ind('jma_14_50', df['close'])
            vq = get_ind('vq_29', 0)
            vq_thresh = params.get('vq_threshold', 0.5)
            long_sig = (df['close'] > jma_val) & (vq > vq_thresh)
            short_sig = (df['close'] < jma_val) & (vq < -vq_thresh)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "hma_stiffness":
            hma_val = get_ind('hma_100', df['close'])
            stiff = get_ind('stiff_smooth_39_50', get_ind('stiff_smooth_60_100', 50))
            stiff_thresh = params.get('stiff_threshold', 70)
            long_sig = (df['close'] > hma_val) & (hma_val > hma_val.shift(1)) & (stiff > stiff_thresh)
            short_sig = (df['close'] < hma_val) & (hma_val < hma_val.shift(1)) & (stiff < 100 - stiff_thresh)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "lwpi_trend":
            lwpi_val = get_ind('lwpi_13', -50)
            md = get_ind('mcginley_14_0.6', df['close'])
            lwpi_oversold = lwpi_val < -80
            lwpi_overbought = lwpi_val > -20
            lwpi_rising = lwpi_val > lwpi_val.shift(1)
            lwpi_falling = lwpi_val < lwpi_val.shift(1)
            long_sig = lwpi_oversold.shift(1) & lwpi_rising & (df['close'] > md)
            short_sig = lwpi_overbought.shift(1) & lwpi_falling & (df['close'] < md)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "supertrend_confluence":
            st_dir = get_ind('st_dir_10_3.0', 1)
            rsi = get_ind('rsi_14', 50)
            st_flip_up = (st_dir > 0) & (st_dir.shift(1) <= 0)
            st_flip_dn = (st_dir < 0) & (st_dir.shift(1) >= 0)
            long_sig = (st_flip_up | (st_dir > 0)) & (rsi > 50)
            short_sig = (st_flip_dn | (st_dir < 0)) & (rsi < 50)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "flat_market_mean_reversion":
            is_flat = get_ind('is_flat_30', False)
            bb_upper = get_ind('bb_upper_20_2.0', df['close'] * 1.02)
            bb_lower = get_ind('bb_lower_20_2.0', df['close'] * 0.98)
            rsi = get_ind('rsi_14', 50)
            long_sig = is_flat & (df['close'] <= bb_lower) & (rsi < 35)
            short_sig = is_flat & (df['close'] >= bb_upper) & (rsi > 65)
            sideways = is_flat  # Only trade in flat markets
            
        elif strategy_name == "adaptive_momentum":
            # Score multiple indicators
            score = pd.Series(0.0, index=df.index)
            rsi = get_ind('rsi_14', 50)
            macd_hist = get_ind('macd_hist_12_26_9', 0)
            di_plus = get_ind('di_plus_14', 0)
            di_minus = get_ind('di_minus_14', 0)
            rf_dir = get_ind('rf_dir_100_3.0', 0)
            st_dir = get_ind('st_dir_10_3.0', 0)
            
            score = score + np.where(rsi > 50, 1, -1)
            score = score + np.where(macd_hist > 0, 1, -1)
            score = score + np.where(di_plus > di_minus, 1, -1)
            score = score + rf_dir
            score = score + st_dir
            
            score_thresh = params.get('score_threshold', 4)
            long_sig = score >= score_thresh
            short_sig = score <= -score_thresh
            sideways = pd.Series(True, index=df.index)
            
        # === CONFLUENCE ===
        elif strategy_name == "triple_confirm_bb_rsi_stoch":
            bb_upper = get_ind('bb_upper_20_2.0', df['close'] * 1.02)
            bb_lower = get_ind('bb_lower_20_2.0', df['close'] * 0.98)
            rsi = get_ind('rsi_14', 50)
            stoch = get_ind('stoch_k_14_3', 50)
            stoch_os = params.get('stoch_oversold', 25)
            stoch_ob = params.get('stoch_overbought', 75)
            long_sig = (df['close'] <= bb_lower) & (rsi < rsi_os) & (stoch < stoch_os)
            short_sig = (df['close'] >= bb_upper) & (rsi > rsi_ob) & (stoch > stoch_ob)
            
        elif strategy_name == "quad_oscillator":
            rsi = get_ind('rsi_14', 50)
            stoch_rsi_zone = get_ind('stoch_rsi_zone', 0)
            cci = get_ind('cci_20', 0)
            willr = get_ind('willr_14', -50)
            cci_thresh = params.get('cci_threshold', 100)
            willr_os = params.get('willr_oversold', -80)
            willr_ob = params.get('willr_overbought', -20)
            long_sig = (rsi < rsi_os) & (stoch_rsi_zone == -1) & (cci < -cci_thresh) & (willr < willr_os)
            short_sig = (rsi > rsi_ob) & (stoch_rsi_zone == 1) & (cci > cci_thresh) & (willr > willr_ob)
            
        elif strategy_name == "volume_price_combo":
            cmf = get_ind('cmf', 0)
            force = get_ind('force_index', 0)
            bb_pct = get_ind('bb_percent_b', 0.5)
            rsi = get_ind('rsi_14', 50)
            long_sig = (cmf > 0) & (force > 0) & (bb_pct < 0.2) & (rsi < 40)
            short_sig = (cmf < 0) & (force < 0) & (bb_pct > 0.8) & (rsi > 60)
            
        elif strategy_name == "trend_alignment":
            st_dir = get_ind('supertrend_dir', 1)
            hma_slope = get_ind('hma_slope', 0)
            zlema_cross = get_ind('zlema_cross', 0)
            hma_val = get_ind('hma', df['close'])
            long_sig = (st_dir == 1) & (hma_slope > 0) & (zlema_cross > 0) & (df['close'] < hma_val)
            short_sig = (st_dir == -1) & (hma_slope < 0) & (zlema_cross < 0) & (df['close'] > hma_val)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "mean_reversion_suite":
            z_score = get_ind('z_score', 0)
            bb_pct = get_ind('bb_percent_b', 0.5)
            vwap_dist = get_ind('vwap_distance', 0)
            rsi = get_ind('rsi_14', 50)
            long_sig = (z_score < -1.5) & (bb_pct < 0.1) & (vwap_dist < -1) & (rsi < 35)
            short_sig = (z_score > 1.5) & (bb_pct > 0.9) & (vwap_dist > 1) & (rsi > 65)
            
        elif strategy_name == "squeeze_momentum":
            bb_bw = get_ind('bb_bandwidth', 0.5)
            ao_cross = get_ind('ao_cross', 0)
            rsi = get_ind('rsi_14', 50)
            bb_mid = get_ind('bb_mid_20_2.0', df['close'])
            long_sig = (bb_bw < 0.3) & (ao_cross == 1) & (rsi > 50) & (df['close'] > bb_mid)
            short_sig = (bb_bw < 0.3) & (ao_cross == -1) & (rsi < 50) & (df['close'] < bb_mid)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "fisher_stoch_keltner":
            fisher = get_ind('fisher', 0)
            stoch = get_ind('stoch_k_14_3', 50)
            kc_upper = get_ind('kc_upper_20', df['close'] * 1.02)
            kc_lower = get_ind('kc_lower_20', df['close'] * 0.98)
            stoch_os = params.get('stoch_oversold', 25)
            stoch_ob = params.get('stoch_overbought', 75)
            long_sig = (fisher < -1) & (stoch < stoch_os) & (df['close'] <= kc_lower)
            short_sig = (fisher > 1) & (stoch > stoch_ob) & (df['close'] >= kc_upper)
            
        elif strategy_name == "hurst_regime_multi":
            hurst = get_ind('hurst', 0.5)
            rsi = get_ind('rsi_14', 50)
            bb_pct = get_ind('bb_percent_b', 0.5)
            z_score = get_ind('z_score', 0)
            # Mean reverting regime (H < 0.45)
            mr_regime = hurst < 0.45
            long_sig = mr_regime & (rsi < rsi_os) & (bb_pct < 0.15) & (z_score < -1.5)
            short_sig = mr_regime & (rsi > rsi_ob) & (bb_pct > 0.85) & (z_score > 1.5)
            sideways = hurst < 0.5
            
        elif strategy_name == "atr_spike_extreme":
            atr_ratio = get_ind('atr_ratio', 1)
            atr_spike = params.get('atr_spike_mult', 1.5)
            rsi = get_ind('rsi_14', 50)
            stoch_rsi_zone = get_ind('stoch_rsi_zone', 0)
            donchian_prox = get_ind('donchian_prox', 0.5)
            long_sig = (atr_ratio > atr_spike) & (rsi < 25) & (stoch_rsi_zone == -1) & (donchian_prox < 0.15)
            short_sig = (atr_ratio > atr_spike) & (rsi > 75) & (stoch_rsi_zone == 1) & (donchian_prox > 0.85)
            
        elif strategy_name == "ultimate_confluence":
            bb_upper = get_ind('bb_upper_20_2.0', df['close'] * 1.02)
            bb_lower = get_ind('bb_lower_20_2.0', df['close'] * 0.98)
            rsi = get_ind('rsi_14', 50)
            stoch = get_ind('stoch_k_14_3', 50)
            cci = get_ind('cci_20', 0)
            z_score = get_ind('z_score', 0)
            bb_pct = get_ind('bb_percent_b', 0.5)
            cci_thresh = params.get('cci_threshold', 100)
            stoch_os = params.get('stoch_oversold', 20)
            stoch_ob = params.get('stoch_overbought', 80)
            long_sig = ((df['close'] <= bb_lower) & (rsi < rsi_os) & (stoch < stoch_os) & 
                       (cci < -cci_thresh) & (z_score < -1.5) & (bb_pct < 0.1))
            short_sig = ((df['close'] >= bb_upper) & (rsi > rsi_ob) & (stoch > stoch_ob) & 
                        (cci > cci_thresh) & (z_score > 1.5) & (bb_pct > 0.9))
            
        # === MACD-BASED ===
        elif strategy_name == "macd_bb":
            bb_upper = get_ind('bb_upper_20_2.0', df['close'] * 1.02)
            bb_lower = get_ind('bb_lower_20_2.0', df['close'] * 0.98)
            macd_hist = get_ind('macd_hist_12_26_9', 0)
            long_sig = (df['close'] <= bb_lower) & (macd_hist > macd_hist.shift(1))
            short_sig = (df['close'] >= bb_upper) & (macd_hist < macd_hist.shift(1))
            
        elif strategy_name == "macd_histogram_divergence":
            macd_hist = get_ind('macd_hist_12_26_9', 0)
            rsi = get_ind('rsi_14', 50)
            long_sig = (macd_hist > 0) & (macd_hist > macd_hist.shift(1)) & (rsi > 50)
            short_sig = (macd_hist < 0) & (macd_hist < macd_hist.shift(1)) & (rsi < 50)
            sideways = pd.Series(True, index=df.index)
            
        # === VOLUME-BASED ===
        elif strategy_name == "volume_confirmed_trend":
            high_vol = get_ind('high_vol', False)
            st_dir = get_ind('st_dir_14_2.5', get_ind('st_dir_10_3.0', 1))
            rsi = get_ind('rsi_14', 50)
            long_sig = (st_dir > 0) & high_vol & (rsi > 50) & (rsi < 70)
            short_sig = (st_dir < 0) & high_vol & (rsi < 50) & (rsi > 30)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "cmf_pressure":
            cmf = get_ind('cmf', 0)
            cmf_thresh = params.get('cmf_threshold', 0.1)
            rsi = get_ind('rsi_14', 50)
            long_sig = (cmf > cmf_thresh) & (rsi < 40)
            short_sig = (cmf < -cmf_thresh) & (rsi > 60)
            
        elif strategy_name == "force_index_trend":
            force = get_ind('force_index', 0)
            rsi = get_ind('rsi_14', 50)
            long_sig = (force > 1) & (rsi < 50)
            short_sig = (force < -1) & (rsi > 50)
            sideways = pd.Series(True, index=df.index)
        
        # === ML-BASED STRATEGIES ===
        elif strategy_name == "nadaraya_watson_reversion":
            # Mean reversion using Nadaraya-Watson kernel envelope
            nw_upper = get_ind('nw_upper', df['close'] * 1.02)
            nw_lower = get_ind('nw_lower', df['close'] * 0.98)
            nw_center = get_ind('nw_center', df['close'])
            rsi = get_ind(f'rsi_{params.get("rsi_length", 14)}', 50)
            long_sig = (df['close'] <= nw_lower) & (rsi < rsi_os)
            short_sig = (df['close'] >= nw_upper) & (rsi > rsi_ob)
            
        elif strategy_name == "divergence_3wave":
            # 3-Wave divergence detection (Vdubus style)
            bull_div = get_ind('bullish_div_3wave', False)
            bear_div = get_ind('bearish_div_3wave', False)
            rsi = get_ind('rsi_14', 50)
            long_sig = bull_div & (rsi < 40)
            short_sig = bear_div & (rsi > 60)
            
        elif strategy_name == "cvd_divergence":
            # CVD-Price divergence
            bull_cvd_div = get_ind('bullish_cvd_div', False)
            bear_cvd_div = get_ind('bearish_cvd_div', False)
            rsi = get_ind('rsi_14', 50)
            long_sig = bull_cvd_div & (rsi < 40)
            short_sig = bear_cvd_div & (rsi > 60)
            
        elif strategy_name == "order_block_bounce":
            # Order Block (SMC) bounce entries
            bull_ob = get_ind('bullish_ob', False)
            bear_ob = get_ind('bearish_ob', False)
            ob_top = get_ind('ob_top', df['close'] * 1.02)
            ob_bottom = get_ind('ob_bottom', df['close'] * 0.98)
            rsi = get_ind('rsi_14', 50)
            # Enter long when price touches bullish OB zone
            long_sig = bull_ob & (df['close'] <= ob_top) & (rsi < 50)
            short_sig = bear_ob & (rsi > 50)
            
        elif strategy_name == "fvg_fill":
            # Fair Value Gap fill strategy
            bull_fvg = get_ind('bullish_fvg', False)
            bear_fvg = get_ind('bearish_fvg', False)
            fvg_top = get_ind('fvg_top', df['close'])
            fvg_bottom = get_ind('fvg_bottom', df['close'])
            rsi = get_ind('rsi_14', 50)
            # Enter on FVG detection with RSI confirmation
            long_sig = bull_fvg.shift(1).fillna(False) & (df['close'] < fvg_top.shift(1)) & (rsi < 50)
            short_sig = bear_fvg.shift(1).fillna(False) & (df['close'] > fvg_bottom.shift(1)) & (rsi > 50)
            
        elif strategy_name == "squeeze_momentum_breakout":
            # Squeeze release with momentum confirmation
            squeeze_on = get_ind('squeeze_on', False)
            squeeze_mom = get_ind('squeeze_mom', 0)
            squeeze_dir = get_ind('squeeze_dir', 0)
            rsi = get_ind('rsi_14', 50)
            # Enter when squeeze releases with momentum
            squeeze_release = squeeze_on.shift(1).fillna(False) & ~squeeze_on
            long_sig = squeeze_release & (squeeze_mom > 0) & (squeeze_dir > 0) & (rsi > 50)
            short_sig = squeeze_release & (squeeze_mom < 0) & (squeeze_dir < 0) & (rsi < 50)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "connors_rsi_extreme":
            # Connors RSI extreme readings
            crsi = get_ind('connors_rsi', 50)
            crsi_os = 15  # Oversold threshold
            crsi_ob = 85  # Overbought threshold
            long_sig = crsi < crsi_os
            short_sig = crsi > crsi_ob
            
        elif strategy_name == "smc_divergence_confluence":
            # SMC zones + divergence confluence
            bull_ob = get_ind('bullish_ob', False)
            bear_ob = get_ind('bearish_ob', False)
            bull_div = get_ind('bullish_div_3wave', False) | get_ind('bullish_div_rsi', False)
            bear_div = get_ind('bearish_div_3wave', False) | get_ind('bearish_div_rsi', False)
            rsi = get_ind('rsi_14', 50)
            # High confluence: SMC zone + divergence
            long_sig = (bull_ob | bull_div) & (rsi < 40)
            short_sig = (bear_ob | bear_div) & (rsi > 60)
            
        elif strategy_name == "kernel_trend_follow":
            # Kernel regression trend following
            nw_center = get_ind('nw_center', df['close'])
            kernel_ma = get_ind('kernel_ma_20', df['close'])
            rsi = get_ind('rsi_14', 50)
            # Price above kernel center and rising
            trend_up = (nw_center > nw_center.shift(1)) & (df['close'] > nw_center)
            trend_dn = (nw_center < nw_center.shift(1)) & (df['close'] < nw_center)
            long_sig = trend_up & (rsi > 50) & (rsi < 70)
            short_sig = trend_dn & (rsi < 50) & (rsi > 30)
            sideways = pd.Series(True, index=df.index)
            
        elif strategy_name == "ml_feature_ensemble":
            # Ensemble of multiple ML signals
            score = pd.Series(0.0, index=df.index)
            
            # Nadaraya-Watson: price at extreme
            nw_upper = get_ind('nw_upper', df['close'] * 1.02)
            nw_lower = get_ind('nw_lower', df['close'] * 0.98)
            score = score + np.where(df['close'] <= nw_lower, 1, np.where(df['close'] >= nw_upper, -1, 0))
            
            # Divergence signals
            bull_div = get_ind('bullish_div_3wave', False) | get_ind('bullish_div_rsi', False)
            bear_div = get_ind('bearish_div_3wave', False) | get_ind('bearish_div_rsi', False)
            score = score + np.where(bull_div, 1, np.where(bear_div, -1, 0))
            
            # Squeeze momentum
            squeeze_mom = get_ind('squeeze_mom', 0)
            score = score + np.where(squeeze_mom > 0, 0.5, np.where(squeeze_mom < 0, -0.5, 0))
            
            # Connors RSI
            crsi = get_ind('connors_rsi', 50)
            score = score + np.where(crsi < 20, 1, np.where(crsi > 80, -1, 0))
            
            # SMC Order Blocks
            bull_ob = get_ind('bullish_ob', False)
            bear_ob = get_ind('bearish_ob', False)
            score = score + np.where(bull_ob, 0.5, np.where(bear_ob, -0.5, 0))
            
            # Liquidity zones
            liq_below = get_ind('liquidity_below', 0)
            liq_above = get_ind('liquidity_above', 0)
            score = score + np.where(liq_below > 0.5, 0.5, 0) - np.where(liq_above > 0.5, 0.5, 0)
            
            # Need strong consensus (3+ signals)
            long_sig = score >= 2.5
            short_sig = score <= -2.5

        # =================================================================
        # === SIMPLE STRATEGIES (Beginner-Friendly, Often Outperform) ===
        # =================================================================

        # --- Percentage Drop/Rise Strategies ---
        elif strategy_name == "pct_drop_buy":
            # Buy after X% drop from recent high, sell after Y% rise from entry
            drop_pct = params.get('drop_percent', 2.0) / 100
            lookback = int(params.get('pct_lookback', 10))
            rolling_high = df['high'].rolling(lookback).max()
            current_drop = (rolling_high - df['close']) / rolling_high
            long_sig = current_drop >= drop_pct
            short_sig = pd.Series(False, index=df.index)  # Long-only strategy
            sideways = pd.Series(True, index=df.index)

        elif strategy_name == "pct_drop_consecutive":
            # Buy after N consecutive red candles with X% total drop
            red_count = int(params.get('red_candle_count', 3))
            min_drop = params.get('min_drop_percent', 2.0) / 100
            is_red = df['close'] < df['open']
            red_streak = is_red.rolling(red_count).sum() == red_count
            price_change = (df['close'] - df['close'].shift(red_count)) / df['close'].shift(red_count)
            long_sig = red_streak & (price_change <= -min_drop)
            short_sig = pd.Series(False, index=df.index)
            sideways = pd.Series(True, index=df.index)

        # --- Simple Moving Average Strategies ---
        elif strategy_name == "simple_sma_cross":
            # Classic fast/slow SMA crossover
            fast_len = int(params.get('sma_fast', 9))
            slow_len = int(params.get('sma_slow', 21))
            sma_fast = df['close'].rolling(fast_len).mean()
            sma_slow = df['close'].rolling(slow_len).mean()
            long_sig = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
            short_sig = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))
            sideways = pd.Series(True, index=df.index)

        elif strategy_name == "price_vs_sma":
            # Price crosses above/below single SMA
            sma_len = int(params.get('single_sma_period', 50))
            confirm = int(params.get('confirm_candles', 2))
            sma_val = df['close'].rolling(sma_len).mean()
            above_sma = df['close'] > sma_val
            confirmed_above = above_sma.rolling(confirm).sum() == confirm
            confirmed_below = (~above_sma).rolling(confirm).sum() == confirm
            long_sig = confirmed_above & ~confirmed_above.shift(1).fillna(False)
            short_sig = confirmed_below & ~confirmed_below.shift(1).fillna(False)
            sideways = pd.Series(True, index=df.index)

        elif strategy_name == "triple_sma_align":
            # Triple SMA alignment (fast > medium > slow for long)
            fast = int(params.get('triple_fast', 9))
            medium = int(params.get('triple_medium', 21))
            slow = int(params.get('triple_slow', 50))
            sma_f = df['close'].rolling(fast).mean()
            sma_m = df['close'].rolling(medium).mean()
            sma_s = df['close'].rolling(slow).mean()
            aligned_bull = (sma_f > sma_m) & (sma_m > sma_s)
            aligned_bear = (sma_f < sma_m) & (sma_m < sma_s)
            # Signal on new alignment
            long_sig = aligned_bull & ~aligned_bull.shift(1).fillna(False)
            short_sig = aligned_bear & ~aligned_bear.shift(1).fillna(False)
            sideways = pd.Series(True, index=df.index)

        # --- Price Action Strategies ---
        elif strategy_name == "consecutive_candles_reversal":
            # N red candles followed by bullish reversal
            n_candles = int(params.get('consec_count', 3))
            is_red = df['close'] < df['open']
            is_green = df['close'] > df['open']
            red_streak = is_red.rolling(n_candles).sum() == n_candles
            long_sig = red_streak.shift(1).fillna(False) & is_green
            # Opposite for shorts
            green_streak = is_green.rolling(n_candles).sum() == n_candles
            short_sig = green_streak.shift(1).fillna(False) & is_red
            sideways = pd.Series(True, index=df.index)

        elif strategy_name == "inside_bar_breakout":
            # Inside bar: current bar's high/low within previous bar's range
            min_bars = int(params.get('min_inside_bars', 1))
            inside = (df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))
            inside_count = inside.rolling(min_bars).sum()
            had_inside = inside_count >= min_bars
            # Breakout after inside bar(s)
            breakout_up = had_inside.shift(1).fillna(False) & (df['close'] > df['high'].shift(1))
            breakout_dn = had_inside.shift(1).fillna(False) & (df['close'] < df['low'].shift(1))
            long_sig = breakout_up
            short_sig = breakout_dn
            sideways = pd.Series(True, index=df.index)

        elif strategy_name == "range_breakout_simple":
            # Break above/below recent high/low
            lookback = int(params.get('range_lookback', 20))
            buffer_pct = params.get('breakout_buffer_pct', 0.5) / 100
            recent_high = df['high'].rolling(lookback).max()
            recent_low = df['low'].rolling(lookback).min()
            # Add buffer for breakout confirmation
            breakout_level_up = recent_high.shift(1) * (1 + buffer_pct)
            breakout_level_dn = recent_low.shift(1) * (1 - buffer_pct)
            long_sig = df['close'] > breakout_level_up
            short_sig = df['close'] < breakout_level_dn
            sideways = pd.Series(True, index=df.index)

        elif strategy_name == "support_resistance_simple":
            # Simple S/R using pivot points
            lookback = int(params.get('sr_lookback', 50))
            tolerance = params.get('sr_tolerance_pct', 0.5) / 100
            # Simple: rolling min = support, rolling max = resistance
            support = df['low'].rolling(lookback).min()
            resistance = df['high'].rolling(lookback).max()
            near_support = (df['close'] - support) / support <= tolerance
            near_resistance = (resistance - df['close']) / resistance <= tolerance
            # Buy near support with bounce confirmation
            bouncing_up = df['close'] > df['open']
            bouncing_dn = df['close'] < df['open']
            long_sig = near_support & bouncing_up
            short_sig = near_resistance & bouncing_dn
            sideways = pd.Series(True, index=df.index)

        # --- Simple Momentum Strategies ---
        elif strategy_name == "simple_rsi_extreme":
            # Pure RSI: buy oversold, sell overbought
            rsi_len = int(params.get('simple_rsi_period', 14))
            os_level = params.get('simple_oversold', 30)
            ob_level = params.get('simple_overbought', 70)
            rsi = get_ind(f'rsi_{rsi_len}', 50)
            long_sig = rsi < os_level
            short_sig = rsi > ob_level
            sideways = pd.Series(True, index=df.index)

        elif strategy_name == "candle_ratio_momentum":
            # Buy when green candle ratio exceeds threshold after selling pressure
            lookback = int(params.get('ratio_lookback', 10))
            threshold = params.get('ratio_threshold', 0.7)
            is_green = (df['close'] > df['open']).astype(float)
            green_ratio = is_green.rolling(lookback).mean()
            # Reversal signal: was mostly red, now turning green
            was_mostly_red = green_ratio.shift(1) < (1 - threshold)
            now_bullish = green_ratio > threshold
            long_sig = was_mostly_red & now_bullish
            # Opposite for shorts
            was_mostly_green = green_ratio.shift(1) > threshold
            now_bearish = green_ratio < (1 - threshold)
            short_sig = was_mostly_green & now_bearish
            sideways = pd.Series(True, index=df.index)

        elif strategy_name == "engulfing_pattern":
            # Bullish/Bearish engulfing candle pattern
            min_size = params.get('engulf_min_size_mult', 1.5)
            body_prev = abs(df['close'].shift(1) - df['open'].shift(1))
            body_curr = abs(df['close'] - df['open'])
            # Bullish engulfing: prev red, current green, current body >= prev body * mult
            prev_red = df['close'].shift(1) < df['open'].shift(1)
            curr_green = df['close'] > df['open']
            engulf_bull = prev_red & curr_green & (body_curr >= body_prev * min_size)
            engulf_bull = engulf_bull & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
            # Bearish engulfing
            prev_green = df['close'].shift(1) > df['open'].shift(1)
            curr_red = df['close'] < df['open']
            engulf_bear = prev_green & curr_red & (body_curr >= body_prev * min_size)
            engulf_bear = engulf_bear & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
            long_sig = engulf_bull
            short_sig = engulf_bear
            sideways = pd.Series(True, index=df.index)

        elif strategy_name == "doji_reversal":
            # Doji at extremes signals reversal
            body_pct = params.get('doji_body_pct', 0.1)
            confirm = int(params.get('doji_confirm_candles', 2))
            candle_range = df['high'] - df['low']
            body_size = abs(df['close'] - df['open'])
            is_doji = (body_size / candle_range.replace(0, np.nan)) < body_pct
            is_doji = is_doji.fillna(False)
            # Doji at low (bullish) - price was falling
            was_falling = df['close'].shift(1) < df['close'].shift(confirm + 1)
            doji_at_low = is_doji.shift(1).fillna(False) & was_falling
            # Confirm with green candle after
            long_sig = doji_at_low & (df['close'] > df['open'])
            # Doji at high (bearish)
            was_rising = df['close'].shift(1) > df['close'].shift(confirm + 1)
            doji_at_high = is_doji.shift(1).fillna(False) & was_rising
            short_sig = doji_at_high & (df['close'] < df['open'])
            sideways = pd.Series(True, index=df.index)

        # === ML-BASED STRATEGIES (XGBoost, LightGBM, CatBoost) ===
        elif strategy_name.startswith("ml_"):
            # Use pre-trained ML models from global trained_ml_models
            global trained_ml_models

            # Helper to get ML predictions
            def get_ml_signals(model_name):
                if model_name in trained_ml_models:
                    try:
                        result = trained_ml_models[model_name].predict(df)
                        return result.signals
                    except:
                        pass
                return pd.Series(0, index=df.index)

            # ML Ensemble Strategies
            if strategy_name == "ml_ensemble_majority":
                # Majority vote: 2+ of 3 models agree
                xgb_sig = get_ml_signals('ml_xgboost')
                lgb_sig = get_ml_signals('ml_lightgbm')
                cat_sig = get_ml_signals('ml_catboost')
                combined = xgb_sig + lgb_sig + cat_sig
                long_sig = combined >= 2  # At least 2 say long
                short_sig = combined <= -2  # At least 2 say short
                sideways = pd.Series(True, index=df.index)

            elif strategy_name == "ml_ensemble_unanimous":
                # All 3 models must agree
                xgb_sig = get_ml_signals('ml_xgboost')
                lgb_sig = get_ml_signals('ml_lightgbm')
                cat_sig = get_ml_signals('ml_catboost')
                long_sig = (xgb_sig == 1) & (lgb_sig == 1) & (cat_sig == 1)
                short_sig = (xgb_sig == -1) & (lgb_sig == -1) & (cat_sig == -1)
                sideways = pd.Series(True, index=df.index)

            elif strategy_name == "ml_xgboost_rsi_confirmed":
                # XGBoost + RSI confirmation
                xgb_sig = get_ml_signals('ml_xgboost')
                rsi_len = params.get('rsi_length', 14)
                rsi_os = params.get('rsi_oversold', 30)
                rsi_ob = params.get('rsi_overbought', 70)
                rsi = get_ind(f'rsi_{rsi_len}', 50)
                long_sig = (xgb_sig == 1) & (rsi < rsi_os)
                short_sig = (xgb_sig == -1) & (rsi > rsi_ob)
                sideways = pd.Series(True, index=df.index)

            elif strategy_name == "ml_lightgbm_bb_confirmed":
                # LightGBM + BB confirmation
                lgb_sig = get_ml_signals('ml_lightgbm')
                bb_len = params.get('bb_length', 20)
                bb_mult = params.get('bb_mult', 2.0)
                bb_upper = get_ind(f'bb_upper_{bb_len}_{bb_mult}', df['close'] * 1.02)
                bb_lower = get_ind(f'bb_lower_{bb_len}_{bb_mult}', df['close'] * 0.98)
                long_sig = (lgb_sig == 1) & (df['close'] <= bb_lower)
                short_sig = (lgb_sig == -1) & (df['close'] >= bb_upper)
                sideways = pd.Series(True, index=df.index)

            elif strategy_name == "ml_ensemble_adx_filter":
                # ML majority + ADX trend filter
                xgb_sig = get_ml_signals('ml_xgboost')
                lgb_sig = get_ml_signals('ml_lightgbm')
                cat_sig = get_ml_signals('ml_catboost')
                combined = xgb_sig + lgb_sig + cat_sig
                adx_thresh = params.get('adx_threshold', 25)
                adx = get_ind('adx_14', 20)
                # Only trade when ADX shows trend
                trend_present = adx > adx_thresh
                long_sig = (combined >= 2) & trend_present
                short_sig = (combined <= -2) & trend_present
                sideways = pd.Series(True, index=df.index)

            elif strategy_name in trained_ml_models:
                # Single ML model (XGBoost, LightGBM, CatBoost)
                try:
                    model = trained_ml_models[strategy_name]
                    result = model.predict(df)
                    signals = result.signals

                    # Convert signals to long/short
                    long_sig = signals == 1
                    short_sig = signals == -1
                    sideways = pd.Series(True, index=df.index)

                except Exception as e:
                    print(f"ML prediction error for {strategy_name}: {e}")
                    long_sig = pd.Series(False, index=df.index)
                    short_sig = pd.Series(False, index=df.index)
                    sideways = pd.Series(True, index=df.index)
            else:
                # Model not trained - no signals
                long_sig = pd.Series(False, index=df.index)
                short_sig = pd.Series(False, index=df.index)
                sideways = pd.Series(True, index=df.index)

        # === DEFAULT FALLBACK ===
        else:
            long_sig = pd.Series(False, index=df.index)
            short_sig = pd.Series(False, index=df.index)

        return long_sig, short_sig, sideways

    def run_backtest(self, strategy_name: str, category: str, params: Dict) -> StrategyResult:
        """
        Run full backtest for a strategy with given parameters.

        EXACT-MATCH MODE (matches TradingView exactly):
        - Entry at CLOSE of signal bar (process_orders_on_close=true)
        - Percentage-based TP/SL (not ATR-based)
        - Fixed position size: 0.01 BTC
        - Commission: 0.1% per side
        """
        df = self.df

        # Get signals
        long_sig, short_sig, sideways = self.get_signals(strategy_name, params)

        # Risk parameters - PERCENTAGE-BASED for exact TradingView match
        tp_percent = params.get('tp_percent', 1.0)
        sl_percent = params.get('sl_percent', 3.0)

        # Fixed position size (matches TradingView strategy settings)
        POSITION_SIZE_BTC = 0.01
        COMMISSION_PERCENT = 0.1  # 0.1% per side

        trades = []
        position = None

        for i in range(50, len(df)):
            row = df.iloc[i]

            # Manage existing position FIRST (check exits)
            if position is not None:
                exit_price = None
                exit_reason = None
                entry_price = position['entry_price']

                # PERCENTAGE-BASED TP/SL (matches TradingView exactly)
                if position['direction'] == 'long':
                    sl_price = entry_price * (1 - sl_percent / 100)
                    tp_price = entry_price * (1 + tp_percent / 100)

                    # Check SL first (worst case on same bar)
                    if row['low'] <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'stop_loss'
                    elif row['high'] >= tp_price:
                        exit_price = tp_price
                        exit_reason = 'take_profit'
                else:  # short
                    sl_price = entry_price * (1 + sl_percent / 100)
                    tp_price = entry_price * (1 - tp_percent / 100)

                    # Check SL first (worst case on same bar)
                    if row['high'] >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'stop_loss'
                    elif row['low'] <= tp_price:
                        exit_price = tp_price
                        exit_reason = 'take_profit'

                if exit_price:
                    # Calculate P&L with fixed position size
                    if position['direction'] == 'long':
                        price_movement = exit_price - entry_price
                    else:
                        price_movement = entry_price - exit_price

                    pnl = price_movement * POSITION_SIZE_BTC

                    # Subtract commission (0.1% each side = entry + exit)
                    commission = (entry_price + exit_price) * POSITION_SIZE_BTC * (COMMISSION_PERCENT / 100)
                    pnl -= commission

                    trades.append(TradeRecord(
                        entry_time=position['entry_time'],
                        exit_time=row['time'],
                        direction=position['direction'],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        exit_reason=exit_reason
                    ))
                    position = None

            # Check for new entry signal (ENTRY AT CLOSE - matches TradingView)
            if position is None:
                # Apply sideways filter if applicable
                if sideways is not None and not sideways.iloc[i]:
                    continue

                # ENTRY AT CLOSE of signal bar (matches process_orders_on_close=true)
                if long_sig.iloc[i]:
                    position = {
                        'direction': 'long',
                        'entry_time': row['time'],
                        'entry_price': row['close']  # CLOSE, not next bar's open
                    }
                elif short_sig.iloc[i]:
                    position = {
                        'direction': 'short',
                        'entry_time': row['time'],
                        'entry_price': row['close']  # CLOSE, not next bar's open
                    }

        # Close any open position at end of data
        if position is not None:
            last_row = df.iloc[-1]
            entry_price = position['entry_price']
            exit_price = last_row['close']

            if position['direction'] == 'long':
                price_movement = exit_price - entry_price
            else:
                price_movement = entry_price - exit_price

            pnl = price_movement * POSITION_SIZE_BTC
            commission = (entry_price + exit_price) * POSITION_SIZE_BTC * (COMMISSION_PERCENT / 100)
            pnl -= commission

            trades.append(TradeRecord(
                entry_time=position['entry_time'],
                exit_time=last_row['time'],
                direction=position['direction'],
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                exit_reason='end_of_data'
            ))

        # Calculate metrics
        return self._calculate_metrics(strategy_name, category, params, trades)
    
    def _calculate_metrics(self, strategy_name: str, category: str, 
                          params: Dict, trades: List[TradeRecord]) -> StrategyResult:
        """Calculate comprehensive metrics including equity smoothness."""
        
        result = StrategyResult(
            strategy_name=strategy_name,
            strategy_category=category,
            params=params.copy(),
            trades=trades
        )
        
        if len(trades) < 5:
            result.composite_score = -100
            return result
        
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        result.total_trades = len(trades)
        result.total_pnl = sum(pnls)
        result.win_rate = len(wins) / len(trades) * 100 if trades else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10 if gross_profit > 0 else 0)
        
        # Equity curve
        equity = np.cumsum(pnls)
        result.equity_curve = equity.tolist()
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = running_max - equity
        result.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Recovery factor
        result.recovery_factor = result.total_pnl / (result.max_drawdown + 1)
        
        # Sharpe ratio
        if len(pnls) > 1 and np.std(pnls) > 0:
            result.sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        
        # Equity curve smoothness (R² of linear fit)
        if len(equity) > 1:
            x = np.arange(len(equity))
            slope, intercept = np.polyfit(x, equity, 1)
            predicted = slope * x + intercept
            ss_res = np.sum((equity - predicted) ** 2)
            ss_tot = np.sum((equity - np.mean(equity)) ** 2)
            result.equity_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            if slope < 0:
                result.equity_r_squared *= -1
        
        # =====================================================
        # COMPOSITE SCORE - BALANCED TRIANGLE APPROACH
        # Equal weight to: Win Rate, Equity Smoothness, Total Profit
        # User requested: 33% each for these three key metrics
        # =====================================================

        # ===== WIN RATE SCORE (33.33% weight) =====
        # Normalize to 0-1 scale with bonus for >60% win rate
        wr_score = result.win_rate / 100
        if result.win_rate >= 60:
            wr_score = min(1.0, wr_score * 1.2)  # 20% bonus for >60%

        # ===== EQUITY SMOOTHNESS SCORE (33.33% weight) =====
        # R² of equity curve: higher = steadier growth
        eq_score = max(0, result.equity_r_squared)
        # Bonus for very smooth curves (R² > 0.7)
        if result.equity_r_squared > 0.7:
            eq_score = min(1.0, eq_score * 1.15)

        # ===== TOTAL PROFIT SCORE (33.33% weight) =====
        # Use sigmoid scaling for profit to handle varying magnitudes
        # Scale: £500 profit = ~0.5 score, £2000+ profit = ~1.0 score
        pnl_normalized = result.total_pnl / self.capital  # Relative to starting capital
        pnl_score = 2 / (1 + np.exp(-pnl_normalized)) - 1  # Sigmoid: -1 to 1
        pnl_score = max(0, (pnl_score + 1) / 2)  # Shift to 0-1 scale
        # Bonus for very profitable strategies (>100% return)
        if result.total_pnl > self.capital:
            pnl_score = min(1.0, pnl_score * 1.1)

        # ===== MINIMUM REQUIREMENTS =====
        # Reject strategies that don't meet basic criteria
        data_months = len(self.df) / (4 * 24 * 30)  # Assuming 15min candles
        trades_per_month = result.total_trades / max(data_months, 1)

        min_trades_ok = trades_per_month >= 3  # At least 3 trades/month
        profitable_ok = result.total_pnl > 0  # Must be profitable
        profit_factor_ok = result.profit_factor >= 1.0  # Must have PF >= 1

        # ===== CALCULATE BALANCED COMPOSITE SCORE =====
        result.composite_score = (
            0.3333 * wr_score +     # Win Rate (33.33%)
            0.3333 * eq_score +     # Equity Smoothness (33.33%)
            0.3334 * pnl_score      # Total Profit (33.34%)
        )

        # Apply penalties for not meeting minimum requirements
        if result.win_rate < 50:
            result.composite_score *= 0.6  # 40% penalty for <50% win rate
        if not profitable_ok:
            result.composite_score *= 0.3  # 70% penalty for losing money
        if not profit_factor_ok:
            result.composite_score *= 0.5  # 50% penalty for PF < 1
        if not min_trades_ok:
            result.composite_score *= 0.7  # 30% penalty for too few trades

        # Store score breakdown for UI display
        result.score_breakdown = {
            'win_rate_score': round(wr_score * 100, 1),
            'equity_smoothness_score': round(eq_score * 100, 1),
            'total_profit_score': round(pnl_score * 100, 1),
            'win_rate_weight': 33.33,
            'equity_smoothness_weight': 33.33,
            'total_profit_weight': 33.34,
            'penalties_applied': {
                'low_win_rate': result.win_rate < 50,
                'not_profitable': not profitable_ok,
                'low_profit_factor': not profit_factor_ok,
                'too_few_trades': not min_trades_ok
            }
        }
        
        return result

    def run_backtest_backtrader(self, strategy_name: str, category: str, params: Dict) -> StrategyResult:
        """
        Run backtest using Backtrader engine (industry-standard backtesting).

        This provides more accurate backtesting with:
        - Proper order execution and slippage handling
        - TA-Lib indicators that match TradingView
        - Correct position sizing
        - Better handling of edge cases

        Args:
            strategy_name: Name of the strategy
            category: Strategy category
            params: Strategy parameters

        Returns:
            StrategyResult with backtest metrics
        """
        if not HAS_BACKTRADER:
            # Fall back to custom backtester if Backtrader not available
            return self.run_backtest(strategy_name, category, params)

        try:
            # Create Backtrader engine
            bt_engine = BacktraderEngine(self.df, capital=self.capital, commission=0.001)

            # Run backtest
            bt_result = bt_engine.run_backtest(strategy_name, category, params)

            # Convert BacktraderEngine result to StrategyResult
            result = StrategyResult(
                strategy_name=strategy_name,
                strategy_category=category,
                params=params.copy(),
                trades=[]  # Backtrader doesn't return same format
            )

            # Copy metrics from Backtrader result
            result.total_trades = bt_result.total_trades
            result.total_pnl = bt_result.total_pnl
            result.win_rate = bt_result.win_rate
            result.profit_factor = bt_result.profit_factor
            result.max_drawdown = bt_result.max_drawdown
            result.sharpe_ratio = bt_result.sharpe_ratio
            result.equity_r_squared = bt_result.equity_r_squared
            result.recovery_factor = bt_result.recovery_factor
            result.equity_curve = bt_result.equity_curve
            result.composite_score = bt_result.composite_score
            result.score_breakdown = bt_result.score_breakdown

            # Ensure minimum trades requirement
            if result.total_trades < 5:
                result.composite_score = -100

            return result

        except Exception as e:
            print(f"Backtrader error for {strategy_name}: {e}")
            # Fall back to custom backtester on error
            return self.run_backtest(strategy_name, category, params)


# =============================================================================
# UNIFIED OPTIMIZER - Combines ALL methods
# =============================================================================

class UnifiedOptimizer:
    """
    THE unified optimizer that tests EVERYTHING:
    - All 75+ strategy types
    - 3 optimization methods (Random, Bayesian TPE, CMA-ES)
    - Consensus approach
    - Walk-forward validation
    """
    
    def __init__(self, df: pd.DataFrame, capital: float = 1000.0,
                 risk_percent: float = 2.0, status_callback: Optional[Dict] = None,
                 streaming_callback: Optional[Callable] = None,
                 use_backtrader: bool = True):
        self.df = df
        self.capital = capital
        self.risk_percent = risk_percent
        self.status = status_callback or {}
        self.streaming_callback = streaming_callback  # For SSE real-time streaming
        self.use_backtrader = use_backtrader and HAS_BACKTRADER  # Use Backtrader if available

        if self.use_backtrader:
            print("Using Backtrader engine (industry-standard backtesting)")
        else:
            print("Using custom backtester")

        # Split data: 70% train, 30% validation
        split_idx = int(len(df) * 0.7)
        self.train_df = df.iloc[:split_idx].copy()
        self.val_df = df.iloc[split_idx:].copy()

        print(f"Training data: {len(self.train_df)} candles")
        print(f"Validation data: {len(self.val_df)} candles")

        # Train ML models FIRST (before creating backtesters)
        self._train_ml_models()

        # Create backtesters
        self._update_status("Calculating indicators (this may take a minute)...", 2)
        self.train_backtester = UnifiedBacktester(self.train_df, capital, risk_percent)
        self.val_backtester = UnifiedBacktester(self.val_df, capital, risk_percent)

        # Get all strategies
        self.strategies = StrategyRegistry.get_all_strategies()
        print(f"Loaded {len(self.strategies)} strategy types")
        
        # Results storage
        self.random_results: List[StrategyResult] = []
        self.bayesian_results: List[StrategyResult] = []
        self.cmaes_results: List[StrategyResult] = []
        self.consensus_results: List[StrategyResult] = []

        # Database persistence
        self.db: Optional[StrategyDatabase] = None
        self.db_run_id: Optional[int] = None
        self._data_metadata = {
            'symbol': None,
            'timeframe': None,
            'data_source': None,
            'data_start': None,
            'data_end': None,
        }
        if HAS_DATABASE:
            try:
                self.db = get_strategy_db()
                print(f"Strategy database connected: {self.db.db_path}")
            except Exception as e:
                print(f"Warning: Could not connect to strategy database: {e}")

    def set_data_metadata(self, symbol: str = None, timeframe: str = None,
                          data_source: str = None, data_start: str = None,
                          data_end: str = None):
        """Set metadata about the data being analyzed (for database storage)."""
        if symbol:
            self._data_metadata['symbol'] = symbol
        if timeframe:
            self._data_metadata['timeframe'] = timeframe
        if data_source:
            self._data_metadata['data_source'] = data_source
        if data_start:
            self._data_metadata['data_start'] = data_start
        if data_end:
            self._data_metadata['data_end'] = data_end

    def _update_status(self, message: str, progress: int):
        if self.status:
            self.status['message'] = message
            self.status['progress'] = progress

    def _publish_result(self, result: 'StrategyResult', method: str):
        """Publish a strategy result to SSE stream for real-time updates."""
        print(f"[OPTIMIZER] _publish_result called: {result.strategy_name} score={result.composite_score:.4f} callback={'SET' if self.streaming_callback else 'NONE'}")
        if self.streaming_callback:
            try:
                self.streaming_callback({
                    'type': 'strategy_result',
                    'strategy_name': result.strategy_name,
                    'strategy_category': result.strategy_category,
                    'method': method,
                    'composite_score': round(result.composite_score, 4),
                    'win_rate': round(result.win_rate, 2),
                    'profit_factor': round(result.profit_factor, 2),
                    'total_pnl': round(result.total_pnl, 2),
                    'total_trades': result.total_trades,
                    'params': {k: round(v, 2) if isinstance(v, float) else v
                              for k, v in result.params.items()},
                })
                print(f"[OPTIMIZER] Callback called successfully")
            except Exception as e:
                print(f"Streaming error: {e}")
        else:
            print(f"[OPTIMIZER] No streaming callback set!")

    def _train_ml_models(self):
        """
        Train ML models (XGBoost, LightGBM, CatBoost) automatically.
        These will be used by ML-based strategies.
        """
        global trained_ml_models

        if not HAS_ML_MODELS:
            print("ML models not available - skipping ML training")
            return

        self._update_status("Training ML models (XGBoost, LightGBM, CatBoost)...", 1)
        print("\n" + "="*60)
        print("TRAINING ML MODELS")
        print("="*60)

        models_to_train = [
            ('ml_xgboost', 'xgboost', 'XGBoost'),
            ('ml_lightgbm', 'lightgbm', 'LightGBM'),
            ('ml_catboost', 'catboost', 'CatBoost'),
        ]

        for model_id, model_type, display_name in models_to_train:
            try:
                self._update_status(f"Training {display_name}...", 1)
                print(f"\nTraining {display_name}...")

                model = GradientBoostingPredictor(
                    model_type=model_type,
                    task='classification'
                )

                # Train on the training data
                result = model.train(
                    self.train_df,
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.1,
                    target_horizon=1,
                    target_threshold=0.001
                )

                if result.success:
                    trained_ml_models[model_id] = model
                    print(f"  ✓ {display_name} trained - Val Accuracy: {result.val_accuracy:.2%}")
                else:
                    print(f"  ✗ {display_name} training failed: {result.message}")

            except Exception as e:
                print(f"  ✗ {display_name} error: {str(e)}")

        print(f"\nML Models trained: {len(trained_ml_models)}")
        print("="*60 + "\n")

    def _create_trial_params(self, trial: optuna.Trial, strategy: Dict) -> Dict:
        """Create parameters from Optuna trial based on strategy requirements."""
        params = {}
        
        # Get required params for this strategy
        required_params = strategy.get('params', [])
        
        for param_name in required_params:
            if param_name in PARAM_RANGES:
                min_val, max_val = PARAM_RANGES[param_name]
                
                # Determine type based on values
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    step = 0.1 if max_val - min_val > 1 else 0.05
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val, step=step)
        
        # Risk management params - PERCENTAGE-BASED (matches TradingView exactly)
        params['tp_percent'] = trial.suggest_float('tp_percent', 0.3, 5.0, step=0.1)
        params['sl_percent'] = trial.suggest_float('sl_percent', 1.0, 10.0, step=0.5)

        return params
    
    def _run_optimization_method(self, method: str, sampler, n_trials: int,
                                 strategy_subset: List[Dict]) -> List[StrategyResult]:
        """Run one optimization method across a subset of strategies.

        Args:
            method: Name of optimization method (for logging)
            sampler: Optuna sampler instance
            n_trials: Number of trials PER STRATEGY (not total)
            strategy_subset: List of strategies to optimize
        """
        results = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Choose backtesting method based on configuration
        backtest_fn = (self.train_backtester.run_backtest_backtrader
                      if self.use_backtrader
                      else self.train_backtester.run_backtest)

        # FIXED: Use full n_trials per strategy (not divided)
        # Each strategy now gets the full trial allocation for thorough optimization
        trials_per_strategy = n_trials

        for strategy in strategy_subset:
            strategy_name = strategy['name']
            category = StrategyRegistry.CATEGORIES.get(strategy['category'], strategy['category'])

            try:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=sampler
                )

                def objective(trial):
                    params = self._create_trial_params(trial, strategy)
                    result = backtest_fn(strategy_name, category, params)
                    return result.composite_score

                study.optimize(objective, n_trials=trials_per_strategy, show_progress_bar=False)

                # Get best result - stream ALL results, even negative scores
                if study.best_trial:
                    best_params = dict(study.best_params)
                    result = backtest_fn(strategy_name, category, best_params)
                    result.found_by = [method]
                    results.append(result)

                    # Stream result immediately for real-time feedback
                    self._publish_result(result, method)

            except Exception as e:
                print(f"Error optimizing {strategy_name} with {method}: {e}")
                continue

        return results

    def _optimize_single_strategy(self, args: Tuple) -> Optional[StrategyResult]:
        """
        Optimize a single strategy - used for parallel processing.

        Args:
            args: Tuple of (strategy, method_name, sampler_class, n_trials, seed)

        Returns:
            Best StrategyResult or None if optimization failed
        """
        strategy, method_name, sampler_class, n_trials, seed = args
        strategy_name = strategy['name']
        category = StrategyRegistry.CATEGORIES.get(strategy['category'], strategy['category'])

        # Choose backtesting method based on configuration
        backtest_fn = (self.train_backtester.run_backtest_backtrader
                      if self.use_backtrader
                      else self.train_backtester.run_backtest)

        try:
            # Create sampler with seed
            if sampler_class == RandomSampler:
                sampler = RandomSampler(seed=seed)
            elif sampler_class == TPESampler:
                sampler = TPESampler(seed=seed)
            else:  # CmaEsSampler
                sampler = CmaEsSampler(seed=seed)

            study = optuna.create_study(
                direction="maximize",
                sampler=sampler
            )

            def objective(trial):
                params = self._create_trial_params(trial, strategy)
                result = backtest_fn(strategy_name, category, params)
                return result.composite_score

            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            # Return best result regardless of score (was filtering out negative scores)
            if study.best_trial:
                best_params = dict(study.best_params)
                result = backtest_fn(strategy_name, category, best_params)
                result.found_by = [method_name]
                return result

        except Exception as e:
            print(f"Error optimizing {strategy_name} with {method_name}: {e}")

        return None

    def _run_parallel_optimization(self, method_name: str, sampler_class,
                                    n_trials: int, base_progress: int,
                                    progress_range: int) -> List[StrategyResult]:
        """
        Run optimization in parallel using ThreadPoolExecutor.

        Args:
            method_name: Name of the method (for logging)
            sampler_class: Class of the sampler to use
            n_trials: Trials per strategy
            base_progress: Starting progress percentage
            progress_range: Range of progress to cover

        Returns:
            List of StrategyResults
        """
        results = []
        total_strategies = len(self.strategies)

        # Auto-detect optimal workers based on CPU and memory
        num_workers = get_optimal_workers()

        # Prepare arguments for parallel execution
        args_list = [
            (strategy, method_name, sampler_class, n_trials, 42 + i)
            for i, strategy in enumerate(self.strategies)
        ]

        completed = 0

        # Use ThreadPoolExecutor for parallel strategy optimization
        # (ProcessPoolExecutor has pickle issues with complex objects)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_strategy = {
                executor.submit(self._optimize_single_strategy, args): args[0]
                for args in args_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                completed += 1

                progress = base_progress + int(progress_range * completed / total_strategies)
                self._update_status(
                    f"{method_name}: {strategy['display_name']} ({completed}/{total_strategies}) - {num_workers} parallel workers",
                    progress
                )

                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        # Stream result immediately
                        self._publish_result(result, method_name)
                except Exception as e:
                    print(f"Parallel task error for {strategy['name']}: {e}")

        return results

    def run_random_search(self, n_trials: int = 500, parallel: bool = True) -> List[StrategyResult]:
        """Phase 1a: Random search - unbiased exploration.

        IMPORTANT: n_trials is now trials PER STRATEGY, not total.
        This ensures thorough exploration of each strategy's parameter space.

        Args:
            n_trials: Number of trials per strategy
            parallel: Whether to use parallel processing
        """
        self._update_status("Random Search: Testing all strategies...", 5)

        if parallel:
            results = self._run_parallel_optimization(
                "random", RandomSampler, n_trials,
                base_progress=5, progress_range=20
            )
        else:
            results = []
            total_strategies = len(self.strategies)
            trials_per = n_trials

            for i, strategy in enumerate(self.strategies):
                progress = 5 + int(20 * i / total_strategies)
                self._update_status(f"Random: {strategy['display_name']} ({i+1}/{total_strategies}) - {trials_per} trials", progress)

                batch = self._run_optimization_method(
                    "random",
                    RandomSampler(seed=42 + i),
                    trials_per,
                    [strategy]
                )
                results.extend(batch)

        results.sort(key=lambda x: x.composite_score, reverse=True)
        self.random_results = results[:100]
        return self.random_results
    
    def run_bayesian_search(self, n_trials: int = 500, parallel: bool = True) -> List[StrategyResult]:
        """Phase 1b: Bayesian TPE - smart directed search.

        IMPORTANT: n_trials is now trials PER STRATEGY, not total.
        TPE sampler learns from previous trials, so more trials = better optimization.

        Args:
            n_trials: Number of trials per strategy
            parallel: Whether to use parallel processing
        """
        self._update_status("Bayesian TPE: Smart optimization...", 30)

        if parallel:
            results = self._run_parallel_optimization(
                "bayesian", TPESampler, n_trials,
                base_progress=30, progress_range=25
            )
        else:
            results = []
            total_strategies = len(self.strategies)
            trials_per = n_trials

            for i, strategy in enumerate(self.strategies):
                progress = 30 + int(25 * i / total_strategies)
                self._update_status(f"Bayesian: {strategy['display_name']} ({i+1}/{total_strategies}) - {trials_per} trials", progress)

                batch = self._run_optimization_method(
                    "bayesian",
                    TPESampler(seed=42 + i),
                    trials_per,
                    [strategy]
                )
                results.extend(batch)

        results.sort(key=lambda x: x.composite_score, reverse=True)
        self.bayesian_results = results[:100]
        return self.bayesian_results

    def run_cmaes_search(self, n_trials: int = 300, parallel: bool = True) -> List[StrategyResult]:
        """Phase 1c: CMA-ES - best for continuous parameters.

        IMPORTANT: n_trials is now trials PER STRATEGY, not total.
        CMA-ES is an evolution strategy that benefits from more generations.

        Args:
            n_trials: Number of trials per strategy
            parallel: Whether to use parallel processing
        """
        self._update_status("CMA-ES: Evolution strategy...", 55)

        if parallel:
            results = self._run_parallel_optimization(
                "cmaes", CmaEsSampler, n_trials,
                base_progress=55, progress_range=20
            )
        else:
            results = []
            total_strategies = len(self.strategies)
            trials_per = n_trials

            for i, strategy in enumerate(self.strategies):
                progress = 55 + int(20 * i / total_strategies)
                self._update_status(f"CMA-ES: {strategy['display_name']} ({i+1}/{total_strategies}) - {trials_per} trials", progress)

                try:
                    batch = self._run_optimization_method(
                        "cmaes",
                        CmaEsSampler(seed=42 + i),
                        trials_per,
                        [strategy]
                    )
                    results.extend(batch)
                except Exception as e:
                    print(f"CMA-ES failed for {strategy['name']}: {e}")
                    continue

        results.sort(key=lambda x: x.composite_score, reverse=True)
        self.cmaes_results = results[:100]
        return self.cmaes_results
    
    def find_consensus(self, min_methods: int = 1) -> List[StrategyResult]:
        """
        Phase 2: Find best strategies from all methods.

        FIXED: Now groups by STRATEGY NAME ONLY (not parameters).
        This shows more results because:
        - Different methods find different optimal parameters
        - If multiple methods find the same strategy profitable, that's valuable
        - User sees ALL promising strategies, sorted by score

        Args:
            min_methods: Minimum number of methods that found this strategy (default=1 to show all)
        """
        self._update_status("Collecting all strategies...", 80)

        # Group results by STRATEGY NAME ONLY (not parameters)
        # This is the key fix - we don't require exact parameter matches
        strategy_groups = defaultdict(list)

        all_results = self.random_results + self.bayesian_results + self.cmaes_results

        for result in all_results:
            # Group by strategy name only - ignore parameters
            sig = result.strategy_name
            strategy_groups[sig].append(result)

        # Collect best result for each strategy
        all_strategies = []
        for strategy_name, candidates in strategy_groups.items():
            # Find which methods discovered this strategy
            methods = set()
            for c in candidates:
                methods.update(c.found_by)

            # Get best result for this strategy
            best = max(candidates, key=lambda x: x.composite_score)
            best.found_by = list(methods)

            # Add if meets minimum methods requirement
            if len(methods) >= min_methods:
                all_strategies.append(best)

        # Sort by composite score (highest first)
        all_strategies.sort(key=lambda x: x.composite_score, reverse=True)

        # Log how many strategies from each method count
        multi_method = sum(1 for s in all_strategies if len(s.found_by) >= 2)
        three_method = sum(1 for s in all_strategies if len(s.found_by) >= 3)

        self._update_status(
            f"Found {len(all_strategies)} strategies ({multi_method} by 2+ methods, {three_method} by all 3)",
            85
        )

        return all_strategies
    
    def validate_on_holdout(self, candidates: List[StrategyResult]) -> List[StrategyResult]:
        """
        Phase 3: Validate ALL candidates on held-out data.

        FIXED: Now includes ALL strategies (no filtering).
        User requested: "No minimum profit - show all strategies sorted by score"
        Validation results are recorded but don't filter out strategies.
        """
        self._update_status("Validating on held-out data...", 88)

        # Choose backtesting method based on configuration
        val_backtest_fn = (self.val_backtester.run_backtest_backtrader
                         if self.use_backtrader
                         else self.val_backtester.run_backtest)

        validated = []
        profitable_count = 0

        for i, candidate in enumerate(candidates[:75]):  # Validate top 75 strategies
            if i % 10 == 0:
                self._update_status(f"Validating strategy {i+1}/{min(75, len(candidates))}...", 88 + int(i/75*10))

            try:
                val_result = val_backtest_fn(
                    candidate.strategy_name,
                    candidate.strategy_category,
                    candidate.params
                )

                # Record validation results (but don't filter)
                candidate.val_pnl = val_result.total_pnl
                candidate.val_profit_factor = val_result.profit_factor
                candidate.val_win_rate = val_result.win_rate

                # Track if profitable on validation
                is_profitable = val_result.total_pnl > 0 and val_result.profit_factor >= 1
                if is_profitable:
                    profitable_count += 1
                    # Give bonus to strategies that validate well
                    candidate.composite_score = (
                        0.6 * candidate.composite_score +
                        0.4 * val_result.composite_score
                    )
                else:
                    # Slight penalty for strategies that don't validate (but still include them)
                    candidate.composite_score = (
                        0.8 * candidate.composite_score +
                        0.2 * max(0, val_result.composite_score)
                    )

                validated.append(candidate)

            except Exception as e:
                # Still include strategy even if validation fails
                candidate.val_pnl = 0
                candidate.val_profit_factor = 0
                candidate.val_win_rate = 0
                validated.append(candidate)

        validated.sort(key=lambda x: x.composite_score, reverse=True)

        self._update_status(
            f"Validated {len(validated)} strategies ({profitable_count} profitable on holdout)",
            98
        )

        return validated
    
    def optimize(self, n_trials_per_method: int = 500) -> List[StrategyResult]:
        """
        Run the full unified optimization pipeline.

        UPDATED: Now returns ALL strategies (not just top 10), sorted by score.
        User requested: "No minimum profit - show all strategies sorted by score"
        """
        self._update_status("Starting Unified Optimization...", 0)
        print(f"\n{'='*60}")
        print("UNIFIED OPTIMIZER - Testing ALL {0} strategies".format(len(self.strategies)))
        print(f"{'='*60}\n")

        # Phase 1: Run all optimization methods
        self._update_status("Phase 1: Running 3 optimization methods...", 2)

        self.run_random_search(n_trials_per_method)
        print(f"Random Search: Found {len(self.random_results)} candidates")

        self.run_bayesian_search(n_trials_per_method)
        print(f"Bayesian TPE: Found {len(self.bayesian_results)} candidates")

        self.run_cmaes_search(int(n_trials_per_method * 0.6))
        print(f"CMA-ES: Found {len(self.cmaes_results)} candidates")

        # Phase 2: Collect ALL strategies (min_methods=1 shows everything)
        self._update_status("Phase 2: Collecting all strategies...", 78)
        all_strategies = self.find_consensus(min_methods=1)  # Show ALL strategies
        print(f"Total strategies found: {len(all_strategies)}")

        # Phase 3: Validate all strategies
        self._update_status("Phase 3: Validation on held-out data...", 85)
        validated = self.validate_on_holdout(all_strategies)
        print(f"Validated: {len(validated)} total strategies")

        # Keep ALL validated strategies (not just top 10)
        # UI will paginate or limit display as needed
        self.consensus_results = validated

        # Count multi-method strategies for summary
        multi_method = sum(1 for s in validated if len(s.found_by) >= 2)
        all_three = sum(1 for s in validated if len(s.found_by) >= 3)

        self._update_status(
            f"Complete! {len(validated)} strategies ({multi_method} by 2+ methods, {all_three} by all 3)",
            100
        )

        # Save results to database
        self._save_results_to_database()

        return self.consensus_results

    def _save_results_to_database(self):
        """Save optimization results to SQLite database for persistence."""
        if not self.db:
            print("Database not available - results not persisted")
            return

        try:
            # Start optimization run
            self.db_run_id = self.db.start_optimization_run(
                symbol=self._data_metadata.get('symbol'),
                timeframe=self._data_metadata.get('timeframe'),
                data_source=self._data_metadata.get('data_source'),
                data_rows=len(self.df),
                capital=self.capital,
                risk_percent=self.risk_percent
            )

            # Save all consensus results (profitable strategies)
            profitable = [r for r in self.consensus_results if r.total_pnl > 0]
            saved = self.db.save_strategies_batch(
                profitable,
                run_id=self.db_run_id,
                symbol=self._data_metadata.get('symbol'),
                timeframe=self._data_metadata.get('timeframe'),
                data_source=self._data_metadata.get('data_source'),
                data_start=self._data_metadata.get('data_start'),
                data_end=self._data_metadata.get('data_end')
            )

            # Complete the run
            total_tested = len(self.random_results) + len(self.bayesian_results) + len(self.cmaes_results)
            self.db.complete_optimization_run(
                self.db_run_id,
                strategies_tested=total_tested,
                profitable_found=len(profitable)
            )

            print(f"Saved {saved} profitable strategies to database (run #{self.db_run_id})")

        except Exception as e:
            print(f"Error saving to database: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "capital": self.capital,
                "risk_percent": self.risk_percent,
                "train_candles": len(self.train_df),
                "validation_candles": len(self.val_df),
                "strategies_tested": len(self.strategies),
            },
            "methods_used": ["Random Search", "Bayesian TPE", "CMA-ES"],
            "optimization_stats": {
                "random_candidates": len(self.random_results),
                "bayesian_candidates": len(self.bayesian_results),
                "cmaes_candidates": len(self.cmaes_results),
                "consensus_found": len(self.consensus_results),
            },
            "top_10": []
        }
        
        for i, result in enumerate(self.consensus_results[:10], 1):
            strat_report = {
                "rank": i,
                "strategy_name": result.strategy_name,
                "strategy_category": result.strategy_category,
                "found_by_methods": result.found_by,
                "params": {k: round(v, 2) if isinstance(v, float) else v 
                         for k, v in result.params.items()},
                "metrics": {
                    "total_trades": result.total_trades,
                    "win_rate": round(result.win_rate, 1),
                    "profit_factor": round(result.profit_factor, 2),
                    "total_pnl": round(result.total_pnl, 2),
                    "max_drawdown": round(result.max_drawdown, 2),
                    "equity_smoothness": round(result.equity_r_squared, 3),
                    "recovery_factor": round(result.recovery_factor, 2),
                    "sharpe_ratio": round(result.sharpe_ratio, 2),
                    "composite_score": round(result.composite_score, 3),
                    # Score breakdown for UI display (Balanced Triangle: 33% each)
                    "score_breakdown": result.score_breakdown if result.score_breakdown else None,
                },
                "validation": {
                    "val_pnl": round(result.val_pnl, 2),
                    "val_profit_factor": round(result.val_profit_factor, 2),
                    "val_win_rate": round(result.val_win_rate, 1),
                },
                "equity_curve": result.equity_curve[-100:] if result.equity_curve else []
            }
            report["top_10"].append(strat_report)
        
        # Category breakdown
        category_counts = defaultdict(int)
        for result in self.consensus_results:
            category_counts[result.strategy_category] += 1
        report["category_breakdown"] = dict(category_counts)
        
        return report


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_unified_optimization(df: pd.DataFrame, capital: float = 1000.0,
                             risk_percent: float = 2.0, n_trials: int = 500,
                             status: Optional[Dict] = None,
                             streaming_callback: Optional[Callable] = None,
                             use_backtrader: bool = True,
                             symbol: str = None, timeframe: str = None,
                             data_source: str = None) -> Dict[str, Any]:
    """
    Main entry point for unified optimization.

    Args:
        df: OHLCV DataFrame with 'time', 'open', 'high', 'low', 'close' columns
        capital: Starting capital
        risk_percent: Risk per trade as percentage
        n_trials: Trials per optimization method (more = slower but better)
        status: Optional dict for progress updates
        streaming_callback: Optional callback for real-time SSE streaming
        use_backtrader: Use Backtrader engine for industry-standard backtesting
        symbol: Trading pair (e.g., BTCGBP) for database storage
        timeframe: Candle timeframe (e.g., 15m) for database storage
        data_source: Data source (e.g., Kraken) for database storage

    Returns:
        Report with all strategies
    """
    optimizer = UnifiedOptimizer(df, capital, risk_percent, status, streaming_callback, use_backtrader)

    # Set metadata for database persistence
    if symbol or timeframe or data_source:
        # Try to extract date range from dataframe
        data_start = None
        data_end = None
        if 'time' in df.columns:
            data_start = str(df['time'].min())[:10]
            data_end = str(df['time'].max())[:10]

        optimizer.set_data_metadata(
            symbol=symbol,
            timeframe=timeframe,
            data_source=data_source,
            data_start=data_start,
            data_end=data_end
        )

    optimizer.optimize(n_trials_per_method=n_trials)
    return optimizer.generate_report()


if __name__ == "__main__":
    print("Unified Strategy Optimizer")
    print("="*50)
    print("This module combines ALL strategies and optimization methods.")
    print("Use via the web interface or import run_unified_optimization().")

