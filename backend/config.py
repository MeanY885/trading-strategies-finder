"""
CONFIGURATION MODULE
====================
All configuration constants for the BTCGBP ML Optimizer.
Extracted from main.py to reduce file size and improve maintainability.
"""
import os
from pathlib import Path
import psutil

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent

# Check if running in Docker (look for /app directory)
if Path("/app").exists():
    DATA_DIR = Path("/app/data")
    OUTPUT_DIR = Path("/app/output")
else:
    DATA_DIR = PROJECT_DIR / "data"
    OUTPUT_DIR = PROJECT_DIR / "output"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# SYSTEM RESOURCES
# =============================================================================

CPU_CORES = os.cpu_count() or 4
MEMORY_TOTAL_GB = psutil.virtual_memory().total / (1024**3)
MEMORY_AVAILABLE_GB = psutil.virtual_memory().available / (1024**3)

# =============================================================================
# HISTORY AND LIMITS
# =============================================================================

MAX_HISTORY_SIZE = 500  # Keep last 500 runs in memory
MAX_QUEUE_COMPLETED = 20  # Keep last 20 completed in queue display
MAX_SKIPPED_VALIDATIONS = 100  # Keep last 100 skipped validations

# =============================================================================
# VALIDATION PERIODS (Used by both Elite validation and Autonomous optimizer)
# =============================================================================

VALIDATION_PERIODS = [
    {"period": "1 week", "months": 0.25, "days": 7},
    {"period": "2 weeks", "months": 0.5, "days": 14},
    {"period": "1 month", "months": 1.0, "days": 30},
    {"period": "3 months", "months": 3.0, "days": 90},
    {"period": "6 months", "months": 6.0, "days": 180},
    {"period": "9 months", "months": 9.0, "days": 270},
    {"period": "1 year", "months": 12.0, "days": 365},
    {"period": "2 years", "months": 24.0, "days": 730},
    {"period": "3 years", "months": 36.0, "days": 1095},
    {"period": "5 years", "months": 60.0, "days": 1825},
]

# =============================================================================
# DATA SOURCE LIMITS (days of data available per timeframe)
# =============================================================================

# Yahoo Finance limits
YAHOO_DATA_LIMITS = {1: 7, 5: 60, 15: 60, 30: 60, 60: 730, 1440: 9999}

# Binance limits: 1m=1yr (slow fetch), 5m+=5-7yrs (fast pagination)
BINANCE_DATA_LIMITS = {
    1: 365,      # 1m: 1 year (525k candles, slow)
    5: 1825,     # 5m: 5 years (525k candles)
    15: 2555,    # 15m: 7 years (245k candles)
    30: 2555,    # 30m: 7 years (122k candles)
    60: 2555,    # 1h: 7 years (61k candles)
    240: 2555,   # 4h: 7 years (15k candles)
    1440: 3650,  # 1d: 10 years (3.6k candles)
}

# Supported quote currencies for Elite validation
SUPPORTED_QUOTE_CURRENCIES = ['USDT', 'USDC', 'BUSD']

# =============================================================================
# AUTONOMOUS OPTIMIZER CONFIGURATION
# =============================================================================

AUTONOMOUS_CONFIG = {
    # Capital settings
    "capital": 1000.0,
    "position_size_pct": 100.0,

    # Single source: Binance (via CCXT)
    # All data is USDT pairs - use BINANCE:SYMBOL on TradingView
    "sources": ["binance"],

    # USDT pairs only - excellent historical depth
    "pairs": {
        "binance": [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
            "SOLUSDT", "ADAUSDT", "DOGEUSDT", "DOTUSDT",
            "MATICUSDT", "LTCUSDT", "AVAXUSDT", "LINKUSDT"
        ],
    },

    # Historical periods - Binance has years of data
    "periods": [
        {"label": "1 week", "months": 0.25},
        {"label": "2 weeks", "months": 0.5},
        {"label": "1 month", "months": 1.0},
        {"label": "3 months", "months": 3.0},
        {"label": "6 months", "months": 6.0},
        {"label": "9 months", "months": 9.0},
        {"label": "12 months", "months": 12.0},
    ],

    # Timeframes - all Binance supported
    "timeframes": [
        {"label": "15m", "minutes": 15},
        {"label": "5m", "minutes": 5},
        {"label": "30m", "minutes": 30},
        {"label": "1h", "minutes": 60},
        {"label": "4h", "minutes": 240},
    ],

    # Granularity options - 0.5% first for quick coverage, then finer
    "granularities": [
        {"label": "0.5%", "n_trials": 400},    # Start here - good balance
        {"label": "0.2%", "n_trials": 2500},   # Finer detail
        {"label": "0.1%", "n_trials": 10000},  # Exhaustive
        {"label": "0.7%", "n_trials": 200},    # Coarse
        {"label": "1.0%", "n_trials": 100},    # Coarsest
    ],
}

# =============================================================================
# RESOURCE MONITOR THRESHOLDS
# =============================================================================

# These can be overridden via environment variables for different deployments
RESOURCE_THRESHOLDS = {
    "cpu_target_usage": int(os.getenv("CPU_TARGET_USAGE", "70")),      # Target CPU usage percentage
    "cpu_max_usage": int(os.getenv("CPU_MAX_USAGE", "85")),            # Max before scaling down
    "mem_min_available_gb": float(os.getenv("MEM_MIN_AVAILABLE_GB", "2")),   # Minimum free memory to maintain
    "mem_per_worker_gb": float(os.getenv("MEM_PER_WORKER_GB", "0.3")),    # Estimated memory per optimization
    "sample_window": 10,         # Number of CPU samples to average
    "adjustment_cooldown": 15,   # Seconds between scaling adjustments (faster response)
}

# =============================================================================
# CONCURRENCY CONFIGURATION
# =============================================================================

# Max concurrent optimizations - 0 means auto-detect based on CPU cores
# Can be overridden via environment variable for powerful machines
MAX_CONCURRENT_OPTIMIZATIONS = int(os.getenv("MAX_CONCURRENT", "0"))

# Max concurrent data fetches - Binance rate limit is ~10/sec, so 5 is safe
MAX_CONCURRENT_FETCHES = int(os.getenv("MAX_FETCH_CONCURRENT", "5"))

# =============================================================================
# WEBSOCKET CONFIGURATION
# =============================================================================

WEBSOCKET_CONFIG = {
    "keepalive_timeout": 30.0,   # Seconds for keepalive timeout
    "broadcast_throttle": 1,     # Minimum seconds between broadcasts (1 second)
}

# =============================================================================
# DATA FETCH CONFIGURATION
# =============================================================================

DATA_FETCH_CONFIG = {
    "timeout_seconds": 300,      # 5 minutes max for data fetch
    "min_candles": 100,          # Minimum candles required
    "coverage_threshold": 90,    # Minimum coverage percentage
}

# =============================================================================
# TRADING COSTS (realistic Binance spot trading)
# =============================================================================

DEFAULT_TRADING_COSTS = {
    "commission_pct": 0.1,       # 0.1% per trade
    "spread_pct": 0.05,          # 0.05% spread assumption
    "slippage_pct": 0.03,        # 0.03% slippage estimate
}

# =============================================================================
# WATCHDOG CONFIGURATION
# =============================================================================
# Settings for task monitoring, timeouts, and orphan cleanup

WATCHDOG_CONFIG = {
    # Base timeouts in seconds (scaled by TimeoutCalculator)
    "base_optimization_timeout": 300,    # 5 minutes base for optimization tasks
    "base_data_fetch_timeout": 120,      # 2 minutes base for data fetching
    "base_backtest_timeout": 60,         # 1 minute base for single backtest

    # Absolute timeout limits
    "min_timeout": 60,                   # 1 minute minimum timeout
    "max_timeout": 3600,                 # 1 hour maximum timeout

    # Progress monitoring thresholds
    "no_progress_warning_seconds": 300,  # 5 minutes without progress = warning
    "no_progress_abort_seconds": 600,    # 10 minutes without progress = abort

    # Orphan cleanup settings
    "orphan_cleanup_interval": 60,       # Check for orphans every minute
    "orphan_threshold_seconds": 1800,    # 30 minutes without update = orphan

    # Resource wait timeouts
    "manual_optimizer_wait_timeout": 600,  # 10 minutes max wait for manual optimizer
    "resource_wait_timeout": 300,          # 5 minutes max wait for resources

    # Strategy engine stall detection (see also strategy_engine.py)
    "stall_check_interval": 30,          # Check every 30 seconds
    "stall_timeout": 300,                # 5 minutes before declaring stall
}
