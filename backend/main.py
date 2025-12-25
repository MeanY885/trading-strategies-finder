"""
BTCGBP ML Optimizer - Main FastAPI Application
"""
import os
import json
import asyncio
import concurrent.futures
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import io
import queue
import threading

# Thread pool for running blocking optimization
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

from data_fetcher import BinanceDataFetcher, YFinanceDataFetcher, KrakenDataFetcher
from pinescript_generator import PineScriptGenerator
from strategy_engine import (
    run_strategy_finder, generate_pinescript, StrategyEngine,
    TunedResult, STRATEGY_PARAM_MAP, DEFAULT_INDICATOR_PARAMS
)
from strategy_database import get_strategy_db
from exchange_rate_fetcher import preload_exchange_rates, get_exchange_fetcher, reset_exchange_fetcher
HAS_DATABASE = True

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    print("[Startup] Application starting...")

    # Define delayed startup to avoid blocking during app initialization
    async def delayed_startup():
        # Wait for the app to be fully ready
        await asyncio.sleep(5)
        print("[Startup] Starting background tasks...")
        asyncio.create_task(start_auto_elite_validation())
        print("[Startup] Elite auto-validation loop started")
        asyncio.create_task(start_autonomous_optimizer())
        print("[Startup] Autonomous optimizer auto-started")

    # Schedule delayed startup without blocking
    asyncio.create_task(delayed_startup())

    yield
    # Shutdown
    print("[Shutdown] Application shutting down...")

# Initialize FastAPI with lifespan
app = FastAPI(title="BTCGBP ML Optimizer", version="1.0.0", lifespan=lifespan)

# Global state
unified_status = {
    "running": False,
    "progress": 0,
    "message": "Ready",
    "report": None
}

# SSE streaming queue for real-time results
streaming_results_queue = queue.Queue()
streaming_lock = threading.Lock()
streaming_clients = []  # Track connected SSE clients

data_status = {
    "loaded": False,
    "rows": 0,
    "start_date": None,
    "end_date": None,
    "message": "No data loaded",
    "stats": None,
    "progress": 0,
    "fetching": False
}

# Elite validation status - runs AUTOMATICALLY in background when optimizer is idle
elite_validation_status = {
    "running": False,
    "paused": False,
    "current_strategy_id": None,
    "processed": 0,
    "total": 0,
    "message": "Idle",
    "auto_running": False  # Track if auto-validation loop is active
}

# Autonomous optimizer status - cycles through all combinations automatically
autonomous_optimizer_status = {
    "auto_running": False,      # Is the auto-loop active?
    "running": False,           # Is an optimization currently in progress?
    "paused": False,            # Paused waiting for manual optimizer?
    "enabled": True,            # Auto-enabled on startup
    "message": "Idle",
    "progress": 0,

    # Current optimization parameters
    "current_source": None,      # "binance" or "kraken"
    "current_pair": None,        # e.g., "BTCUSDT"
    "current_period": None,      # e.g., "1 month"
    "current_timeframe": None,   # e.g., "15m"
    "current_granularity": None, # e.g., "0.5%"

    # Trial progress (for real-time feedback)
    "trial_current": 0,          # Current trial number
    "trial_total": 0,            # Total trials for this run

    # Data validation
    "data_validation": None,     # Dict with date range validation info

    # Cycling state
    "cycle_index": 0,            # Current position in the full cycle
    "total_combinations": 0,     # Total combinations to process
    "completed_count": 0,        # Successful optimizations
    "error_count": 0,            # Failed optimizations
    "skipped_count": 0,          # Skipped due to data validation failure

    # Results summary
    "last_result": None,         # Last optimization result summary
    "last_completed_at": None,   # Timestamp of last completion
    "best_strategy_found": None, # Best strategy from current session

    # Skipped validations log (for UI investigation)
    "skipped_validations": [],   # List of skipped combos with reasons
}

# History of autonomous optimizer runs (kept in memory, most recent first)
autonomous_runs_history = []
MAX_HISTORY_SIZE = 500  # Keep last 500 runs in memory

# Autonomous optimizer configuration
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
        {"label": "1 month", "months": 1.0},
        {"label": "1 week", "months": 0.25},
        {"label": "3 months", "months": 3.0},
        {"label": "6 months", "months": 6.0},
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

def calculate_data_stats(df: pd.DataFrame) -> dict:
    """Calculate statistics about the loaded data"""
    close = df['close']
    high = df['high']
    low = df['low']

    # Basic price stats
    min_price = float(low.min())
    max_price = float(high.max())
    avg_price = float(close.mean())

    # Calculate swings (percentage moves)
    returns = close.pct_change() * 100
    max_up_swing = float(returns.max())
    max_down_swing = float(returns.min())

    # Calculate ATR-based volatility percentage
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    volatility_pct = float((atr / close * 100).mean())

    # Calculate daily bull/bear trend
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['time']).dt.date
    daily = df_copy.groupby('date').agg({
        'open': 'first',
        'close': 'last'
    })
    daily['trend'] = (daily['close'] > daily['open']).map({True: 'bull', False: 'bear'})
    # Mark flat days (< 0.1% change)
    daily['pct_change'] = ((daily['close'] - daily['open']) / daily['open'] * 100).abs()
    daily.loc[daily['pct_change'] < 0.1, 'trend'] = 'flat'

    daily_trends = [
        {"date": str(date), "trend": row['trend']}
        for date, row in daily.iterrows()
    ]

    return {
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "avg_price": round(avg_price, 2),
        "max_up_swing": round(max_up_swing, 2),
        "max_down_swing": round(max_down_swing, 2),
        "volatility_pct": round(volatility_pct, 2),
        "daily_trends": daily_trends
    }

# Paths - Use relative paths for local dev, absolute for Docker
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent

# Check if running in Docker (look for /app directory)
if Path("/app").exists():
    DATA_DIR = Path("/app/data")
    OUTPUT_DIR = Path("/app/output")
    FRONTEND_DIR = Path("/app/frontend")
else:
    DATA_DIR = PROJECT_DIR / "data"
    OUTPUT_DIR = PROJECT_DIR / "output"
    FRONTEND_DIR = PROJECT_DIR / "frontend"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Models
class DataFetchRequest(BaseModel):
    source: str = "yfinance"  # "binance" or "yfinance"
    pair: str = "BTC-GBP"     # e.g., BTCUSDT (Binance) or BTC-GBP (yfinance)
    interval: int = 15        # Candle interval in minutes
    months: float = 3         # Historical period (supports decimals: 0.03=1day, 0.25=1week)

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main UI"""
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return {
        "optimization": unified_status,
        "data": data_status
    }


@app.get("/api/exchange-rate")
async def get_exchange_rate():
    """Get current USD/GBP exchange rate from Frankfurter API"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.frankfurter.dev/v1/latest",
                params={"base": "USD", "symbols": "GBP"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    rate = data.get("rates", {}).get("GBP", 0.79)
                    return {"usd_to_gbp": rate, "source": "frankfurter"}
                else:
                    return {"usd_to_gbp": 0.79, "source": "fallback"}
    except Exception as e:
        print(f"Exchange rate fetch error: {e}")
        return {"usd_to_gbp": 0.79, "source": "fallback"}


@app.post("/api/fetch-data")
async def fetch_data(request: DataFetchRequest, background_tasks: BackgroundTasks):
    """Fetch historical data from Binance, Kraken, or Yahoo Finance"""
    global data_status

    source_names = {"binance": "Binance", "kraken": "Kraken", "yfinance": "Yahoo Finance"}
    source_name = source_names.get(request.source, request.source)
    data_status["message"] = f"Fetching {request.interval}m {request.pair} from {source_name}..."
    data_status["source"] = request.source
    data_status["pair"] = request.pair
    data_status["interval"] = request.interval
    data_status["progress"] = 0
    data_status["fetching"] = True
    data_status["loaded"] = False

    background_tasks.add_task(
        fetch_data_task,
        request.source,
        request.pair,
        request.interval,
        request.months
    )

    return {
        "status": "started",
        "message": f"Fetching {request.months} months of {request.interval}m {request.pair} from {source_name}"
    }


async def fetch_data_task(source: str, pair: str, interval: int, months: float):
    """Background task to fetch data"""
    global data_status

    source_names = {"binance": "Binance", "kraken": "Kraken", "yfinance": "Yahoo Finance"}
    source_name = source_names.get(source, source)

    def status_callback(message: str, progress: int = None):
        """Update status for frontend polling"""
        # Clean up source prefixes from messages
        clean_msg = message
        for prefix in ["[YFinance] ", "[Binance] ", "[Kraken] "]:
            clean_msg = clean_msg.replace(prefix, "")
        data_status["message"] = clean_msg
        if progress is not None:
            data_status["progress"] = progress

    try:
        # Create fetcher with status callback (all sources support it)
        if source == "binance":
            fetcher = BinanceDataFetcher(status_callback=status_callback)
        elif source == "kraken":
            fetcher = KrakenDataFetcher(status_callback=status_callback)
        else:
            fetcher = YFinanceDataFetcher(status_callback=status_callback)

        df = await fetcher.fetch_ohlcv(pair=pair, interval=interval, months=months)

        if len(df) == 0:
            # Only set generic error if fetcher didn't already set a specific error
            if not data_status["message"].startswith("Error:"):
                data_status["message"] = f"Error: No data returned for {pair}"
            data_status["loaded"] = False
            data_status["fetching"] = False
            data_status["progress"] = 0
            return

        # Save to CSV
        pair_clean = pair.lower().replace("/", "").replace("-", "")
        csv_path = DATA_DIR / f"{pair_clean}_{interval}m.csv"
        df.to_csv(csv_path, index=False)

        days = (df['time'].max() - df['time'].min()).days

        data_status["loaded"] = True
        data_status["fetching"] = False
        data_status["progress"] = 100
        data_status["rows"] = len(df)
        data_status["interval"] = interval
        data_status["pair"] = pair
        data_status["source"] = source
        data_status["start_date"] = df['time'].min().isoformat()
        data_status["end_date"] = df['time'].max().isoformat()
        data_status["stats"] = calculate_data_stats(df)
        data_status["message"] = f"✓ {len(df)} candles ({days} days) from {source_name}"

    except Exception as e:
        data_status["message"] = f"Error: {str(e)}"
        data_status["loaded"] = False
        data_status["fetching"] = False
        data_status["progress"] = 0
        import traceback
        traceback.print_exc()

@app.post("/api/load-existing-data")
async def load_existing_data():
    """Load existing CSV data - finds most recent file"""
    global data_status
    
    import re
    
    # Find any CSV data file
    data_files = list(DATA_DIR.glob("*_*m.csv"))
    
    if not data_files:
        raise HTTPException(status_code=404, detail="No data file found. Fetch data or upload CSV first.")
    
    # Use most recently modified
    csv_path = max(data_files, key=lambda p: p.stat().st_mtime)
    
    # Extract pair and interval from filename (e.g., btcusdt_15m.csv)
    match = re.search(r'([a-z]+)_(\d+)m\.csv', csv_path.name.lower())
    pair = match.group(1).upper() if match else "UNKNOWN"
    interval = int(match.group(2)) if match else 15
    
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    days = (df['time'].max() - df['time'].min()).days
    
    data_status["loaded"] = True
    data_status["rows"] = len(df)
    data_status["interval"] = interval
    data_status["pair"] = pair
    data_status["start_date"] = df['time'].min().isoformat()
    data_status["end_date"] = df['time'].max().isoformat()
    data_status["stats"] = calculate_data_stats(df)
    data_status["message"] = f"✓ Loaded {len(df)} {pair} candles ({days} days)"

    return data_status


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload TradingView exported CSV file

    Handles TradingView premium export format:
    - Filename: EXCHANGE_PAIR, TIMEFRAME.csv (e.g., KRAKEN_BTCGBP, 15.csv)
    - Columns: time,open,high,low,close,Volume
    - Time format: ISO 8601 with Z suffix (2025-11-19T02:30:00Z)
    """
    global data_status

    import re

    try:
        # Parse TradingView filename to extract metadata
        # Pattern: EXCHANGE_PAIR, TIMEFRAME.csv or EXCHANGE_PAIR, TIMEFRAME (N).csv
        filename = file.filename or ""
        exchange = None
        pair = None
        interval_from_filename = None

        # Try TradingView format: KRAKEN_BTCGBP, 15.csv or KRAKEN_BTCGBP, 15 (1).csv
        tv_match = re.match(r'([A-Z]+)_([A-Z]+),\s*(\d+)(?:\s*\(\d+\))?\.csv', filename, re.IGNORECASE)
        if tv_match:
            exchange = tv_match.group(1).upper()
            pair = tv_match.group(2).upper()
            interval_from_filename = int(tv_match.group(3))

        # Read the uploaded file
        contents = await file.read()

        # Parse CSV - TradingView exports with specific format
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

        # TradingView exports columns like: time, open, high, low, close, Volume
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Rename 'volume' if it exists with different casing
        if 'volume' not in df.columns:
            for col in df.columns:
                if 'vol' in col.lower():
                    df = df.rename(columns={col: 'volume'})
                    break

        # Handle TradingView time format (ISO 8601 with Z suffix)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True)
        elif 'date' in df.columns:
            df['time'] = pd.to_datetime(df['date'], utc=True)
            df = df.drop('date', axis=1)
        else:
            raise HTTPException(status_code=400, detail="CSV must have 'time' or 'date' column")

        # Convert to timezone-naive for consistency with rest of system
        if df['time'].dt.tz is not None:
            df['time'] = df['time'].dt.tz_localize(None)

        # Validate required columns
        required = ['time', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=['time'], keep='first')

        # Determine interval: prefer filename, fallback to auto-detection
        if interval_from_filename:
            interval = interval_from_filename
        elif len(df) >= 2:
            time_diff = (df['time'].iloc[1] - df['time'].iloc[0]).total_seconds() / 60
            interval = int(round(time_diff))
        else:
            interval = 15

        # Save to CSV with extracted pair name (or default)
        pair_clean = (pair or 'btcgbp').lower()
        csv_path = DATA_DIR / f"{pair_clean}_{interval}m.csv"
        df.to_csv(csv_path, index=False)

        # Update status with extracted metadata
        data_status["loaded"] = True
        data_status["rows"] = len(df)
        data_status["interval"] = interval
        data_status["pair"] = pair.upper() if pair else "UNKNOWN"
        data_status["source"] = exchange or "TradingView"
        data_status["start_date"] = df['time'].min().isoformat()
        data_status["end_date"] = df['time'].max().isoformat()
        data_status["stats"] = calculate_data_stats(df)

        days_of_data = (df['time'].max() - df['time'].min()).days
        pair_display = f"{pair}/" if pair else ""
        source_display = f" from {exchange}" if exchange else ""
        data_status["message"] = f"✓ Uploaded {len(df)} {pair_display}{interval}min candles ({days_of_data} days){source_display}"

        return {
            "status": "success",
            "message": data_status["message"],
            "rows": len(df),
            "interval": interval,
            "pair": data_status["pair"],
            "source": data_status["source"],
            "start_date": data_status["start_date"],
            "end_date": data_status["end_date"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


# ============ STRATEGY FINDER ============

class DateRange(BaseModel):
    enabled: bool = True
    startDate: str = None
    startTime: str = "00:00"
    endDate: str = None
    endTime: str = "23:59"

class UnifiedRequest(BaseModel):
    capital: float = 1000.0
    risk_percent: float = 2.0
    n_trials: int = 300  # Trials per optimization method
    engine: str = "all"  # "all" to compare all engines, or specific: tradingview, native
    date_range: DateRange = None  # Optional date range for Pine Script generation


class ValidateStrategyRequest(BaseModel):
    strategy_id: int
    capital: Optional[float] = None  # Override starting capital (default: use original)
    position_size_pct: Optional[float] = None  # Override position size % (default: use original)
    # No period selection - tests ALL periods automatically: 1w, 1m, 3m, 6m, 1yr, 2yr


@app.post("/api/run-unified")
async def start_unified_optimization(request: UnifiedRequest, background_tasks: BackgroundTasks):
    """
    Run strategy finder - tests 18 entry strategies with optimized TP/SL combinations.

    Tests combinations of:
    - Entry strategies: RSI, BB, MACD, EMA cross, engulfing patterns, etc.
    - Directions: Long and Short
    - TP/SL: Small percentages (0.3% - 5%) that actually get hit

    Args:
        capital: Starting capital
        risk_percent: Position size as % of equity
        n_trials: Number of optimization trials
        engine: Calculation engine - "tradingview" or "native"

    Returns profitable strategies ranked by P&L.
    """
    global unified_status

    # Validate engine parameter
    valid_engines = ["tradingview", "native", "all"]
    if request.engine not in valid_engines:
        raise HTTPException(status_code=400, detail=f"Invalid engine. Must be one of: {valid_engines}")

    if unified_status["running"]:
        raise HTTPException(status_code=400, detail="Optimization already running")

    if not data_status["loaded"]:
        raise HTTPException(status_code=400, detail="No data loaded. Please load data first.")

    unified_status["running"] = True
    unified_status["progress"] = 0
    unified_status["message"] = f"Starting unified optimization [{request.engine.upper()} engine]..."
    unified_status["report"] = None

    # Convert date_range to dict for storage
    date_range_dict = None
    if request.date_range:
        date_range_dict = {
            "enabled": request.date_range.enabled,
            "startDate": request.date_range.startDate,
            "startTime": request.date_range.startTime,
            "endDate": request.date_range.endDate,
            "endTime": request.date_range.endTime
        }

    background_tasks.add_task(
        run_unified_task,
        request.capital,
        request.risk_percent,
        request.n_trials,
        request.engine,
        date_range_dict
    )
    
    return {
        "status": "started",
        "message": f"Finding profitable strategies - testing 18 entry rules x 2 directions x TP/SL combinations"
    }


def run_unified_sync(capital: float, risk_percent: float, n_trials: int, engine: str = "all", date_range: dict = None):
    """
    Synchronous unified optimization that runs in thread pool.

    When engine="all", runs all three engines for comparison.
    Otherwise runs just the specified engine.

    Args:
        date_range: Optional date range dict for Pine Script generation
    """
    global unified_status

    try:
        # Load data - find most recent data file
        data_files = list(DATA_DIR.glob("*_*m.csv"))
        if not data_files:
            unified_status["message"] = "Error: No data file found"
            unified_status["running"] = False
            return

        csv_path = max(data_files, key=lambda p: p.stat().st_mtime)
        unified_status["message"] = f"Loading data from {csv_path.name}..."

        # Extract metadata from filename
        filename = csv_path.stem
        parts = filename.split("_")

        if len(parts) >= 3:
            data_source = parts[0]
            symbol = parts[1].upper()
            timeframe = parts[2]
        elif len(parts) == 2:
            symbol = parts[0].upper()
            timeframe = parts[1]
            if "GBP" in symbol:
                data_source = "KRAKEN"
            elif "USDT" in symbol:
                data_source = "BINANCE"
            else:
                data_source = None
        else:
            data_source = None
            symbol = filename.upper()
            timeframe = "15m"

        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])

        # Apply date range filter if provided (MUST match TradingView date filter)
        if date_range and date_range.get('enabled'):
            start_date = date_range.get('startDate')
            end_date = date_range.get('endDate')
            start_time = date_range.get('startTime', '00:00')
            end_time = date_range.get('endTime', '23:59')

            if start_date:
                start_datetime = pd.to_datetime(f"{start_date} {start_time}")
                df = df[df['time'] >= start_datetime]
            if end_date:
                end_datetime = pd.to_datetime(f"{end_date} {end_time}")
                df = df[df['time'] <= end_datetime]

            unified_status["message"] = f"Filtered to {len(df)} candles ({start_date} to {end_date})"

        # Determine source currency from symbol (USD for USDT pairs, GBP for GBP pairs)
        source_currency = "USD" if "USD" in symbol.upper() or "USDT" in symbol.upper() else "GBP"
        fx_fetcher = None

        # Preload exchange rates if using USD data
        if source_currency == "USD":
            unified_status["message"] = "Loading historical USD/GBP exchange rates..."
            unified_status["progress"] = 5

            # Reset and preload exchange rates for the data period
            reset_exchange_fetcher()

            start_date_dt = df['time'].min()
            end_date_dt = df['time'].max()

            # Run async preload in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(preload_exchange_rates(start_date_dt, end_date_dt))
                fx_fetcher = get_exchange_fetcher()
                if fx_fetcher.is_loaded():
                    stats = fx_fetcher.get_cache_stats()
                    unified_status["message"] = f"Loaded {stats['count']} exchange rates ({stats['start_date']} to {stats['end_date']})"
                else:
                    unified_status["message"] = "Warning: Using default exchange rate (API unavailable)"
            except Exception as e:
                unified_status["message"] = f"Warning: Exchange rate loading failed - {e}"
            finally:
                loop.close()

        # Determine which engines to run
        if engine == "all":
            engines_to_run = ["tradingview", "native"]
        else:
            engines_to_run = [engine]

        all_reports = {}
        total_engines = len(engines_to_run)

        for idx, eng in enumerate(engines_to_run):
            engine_label = eng.upper()
            # Short tags for display: TV, NT
            engine_tag = {"tradingview": "TV", "native": "NT"}.get(eng, eng[:2].upper())

            # Calculate progress range for this engine (continuous 0-100 across all engines)
            progress_min = int((idx / total_engines) * 95)
            progress_max = int(((idx + 1) / total_engines) * 95)

            unified_status["message"] = f"[{engine_tag}] Finding strategies on {len(df)} candles..."
            unified_status["progress"] = progress_min

            # Create engine-aware streaming callback
            def make_callback(tag, engine_name):
                def callback(result):
                    result['engine'] = engine_name
                    result['engine_tag'] = tag
                    publish_strategy_result(result)
                return callback

            # Run strategy finder with this engine - stream ALL engines
            report = run_strategy_finder(
                df=df.copy(),  # Use copy to avoid any state issues
                status=unified_status,
                streaming_callback=make_callback(engine_tag, eng),
                symbol=symbol,
                timeframe=timeframe,
                exchange=data_source.upper() if data_source else None,
                capital=capital,
                position_size_pct=risk_percent,
                engine=eng,
                n_trials=n_trials,
                progress_min=progress_min,
                progress_max=progress_max,
                source_currency=source_currency,
                fx_fetcher=fx_fetcher
            )

            all_reports[eng] = report

        # Create combined report for comparison mode
        if engine == "all":
            # Find the best performing engine
            best_engine = None
            best_pnl = float('-inf')
            for eng, report in all_reports.items():
                top_10 = report.get("top_10", [])
                if top_10:
                    engine_best_pnl = top_10[0]["metrics"]["total_pnl"]
                    if engine_best_pnl is not None and engine_best_pnl > best_pnl:
                        best_pnl = engine_best_pnl
                        best_engine = eng

            # Create comparison report structure
            combined_report = {
                'generated_at': datetime.now().isoformat(),
                'data_rows': len(df),
                'exchange': data_source.upper() if data_source else None,
                'symbol': symbol,
                'timeframe': timeframe,
                'mode': 'comparison',
                'engines': engines_to_run,
                'best_engine': best_engine,
                'capital': capital,
                'position_size_pct': risk_percent,
                # Include all engine reports
                'engine_reports': all_reports,
                # Also include the best engine's results as the "main" report for backward compatibility
                'engine': best_engine or 'tradingview',
                'top_10': all_reports.get(best_engine or 'tradingview', {}).get('top_10', []),
                'total_tested': all_reports.get(best_engine or 'tradingview', {}).get('total_tested', 0),
                'profitable_found': all_reports.get(best_engine or 'tradingview', {}).get('profitable_found', 0),
                'beats_buy_hold_count': all_reports.get(best_engine or 'tradingview', {}).get('beats_buy_hold_count', 0),
                'buy_hold_return': all_reports.get(best_engine or 'tradingview', {}).get('buy_hold_return', 0),
            }
            unified_status["report"] = combined_report
        else:
            # Single engine mode - use original format
            unified_status["report"] = all_reports[engine]

        # Store date_range for Pine Script generation
        if date_range:
            unified_status["report"]["date_range"] = date_range

        # =====================================================================
        # PHASE 2: INDICATOR PARAMETER TUNING
        # =====================================================================
        unified_status["message"] = "Phase 2: Tuning indicator parameters for top 20 strategies..."
        unified_status["progress"] = 95

        # Collect all Phase 1 results for tuning
        all_phase1_results = []
        for eng, report in all_reports.items():
            if 'all_results' in report and report['all_results']:
                all_phase1_results.extend(report['all_results'])

        # CRITICAL: Deduplicate strategies before tuning
        # Keep only the best result for each unique (entry_rule, direction) combination
        # This prevents tuning the same strategy 20+ times with different TP/SL
        if all_phase1_results:
            seen_strategies = {}
            for result in all_phase1_results:
                key = (result.entry_rule, result.direction)
                score = result.composite_score if result.composite_score is not None else 0
                if key not in seen_strategies or score > (seen_strategies[key].composite_score or 0):
                    seen_strategies[key] = result
            all_phase1_results = list(seen_strategies.values())
            print(f"[Tuning] Deduplicated to {len(all_phase1_results)} unique strategies for tuning")

        # Only proceed with tuning if we have results
        tuning_results = []
        if all_phase1_results:
            # Create a StrategyEngine instance for tuning
            # Use the first available engine's data
            first_engine = engines_to_run[0]

            # Create tuning callback for SSE streaming
            def tuning_callback(data):
                data['type'] = data.get('type', 'tuning_update')
                publish_strategy_result(data)

            try:
                # Create engine for tuning (reuse df)
                tuning_engine = StrategyEngine(
                    df=df.copy(),
                    status_callback=unified_status,
                    streaming_callback=None,
                    capital=capital,
                    position_size_pct=risk_percent,
                    calc_engine=first_engine
                )

                # Run Phase 2 tuning
                tuning_results = tuning_engine.tune_top_strategies(
                    phase1_results=all_phase1_results,
                    top_n=20,
                    streaming_callback=tuning_callback
                )

                # Convert TunedResult objects to dicts for JSON serialization
                tuning_data = [tr.to_dict() for tr in tuning_results]

                # Add tuning results to report
                if unified_status["report"]:
                    unified_status["report"]["tuning_results"] = tuning_data
                    unified_status["report"]["tuning_complete"] = True

                    # Count improvements
                    improved_count = sum(1 for tr in tuning_results if tr.is_improved)
                    unified_status["report"]["tuning_improved_count"] = improved_count

                unified_status["message"] = f"Phase 2 complete: {improved_count}/{len(tuning_results)} strategies improved"

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Phase 2 tuning error: {e}")
                if unified_status["report"]:
                    unified_status["report"]["tuning_results"] = []
                    unified_status["report"]["tuning_complete"] = False
                    unified_status["report"]["tuning_error"] = str(e)
        else:
            if unified_status["report"]:
                unified_status["report"]["tuning_results"] = []
                unified_status["report"]["tuning_complete"] = False

        unified_status["progress"] = 100

        # Final message
        if engine == "all":
            summary_parts = []
            for eng in engines_to_run:
                top_10 = all_reports[eng].get("top_10", [])
                if top_10:
                    best = top_10[0]
                    summary_parts.append(f"{eng.upper()}: £{best['metrics']['total_pnl']:.0f}")
                else:
                    summary_parts.append(f"{eng.upper()}: No profitable")
            unified_status["message"] = f"Complete! {' | '.join(summary_parts)}"
        else:
            report = unified_status["report"]
            top_count = len(report.get("top_10", []))
            if top_count > 0:
                best = report["top_10"][0]
                unified_status["message"] = f"Complete! #{1} {best['strategy_name']} (PF: {best['metrics']['profit_factor']}, PnL: £{best['metrics']['total_pnl']:.2f})"
            else:
                unified_status["message"] = "Complete! No profitable strategies found."

    except Exception as e:
        import traceback
        traceback.print_exc()
        unified_status["message"] = f"Error: {str(e)}"
    finally:
        unified_status["running"] = False


async def run_unified_task(capital: float, risk_percent: float, n_trials: int, engine: str = "all", date_range: dict = None):
    """Background task for unified optimization - runs sync code in thread pool"""
    global unified_status

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        thread_pool,
        run_unified_sync,
        capital,
        risk_percent,
        n_trials,
        engine,
        date_range
    )


@app.get("/api/unified-status")
async def get_unified_status():
    """Get current unified optimization status"""
    return unified_status


# =============================================================================
# SSE STREAMING ENDPOINTS - Real-time results as they complete
# =============================================================================

def publish_strategy_result(result: dict):
    """Publish a strategy result to all connected SSE clients"""
    with streaming_lock:
        client_count = len(streaming_clients)
        if not streaming_clients:
            return  # No clients, skip silently
        # Iterate over a copy to avoid issues if list changes
        clients_copy = list(streaming_clients)

    # Log outside the lock to avoid holding it too long
    strategy_name = result.get('strategy_name', result.get('type', 'unknown'))
    print(f"[SSE] Publishing: {strategy_name} to {client_count} clients")

    for q in clients_copy:
        try:
            q.put_nowait(result)
        except queue.Full:
            print(f"[SSE] Queue full, skipping")


async def generate_sse_stream():
    """Generator function for SSE streaming"""
    client_queue = queue.Queue(maxsize=100)

    with streaming_lock:
        streaming_clients.append(client_queue)

    try:
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'message': 'SSE stream connected'})}\n\n"

        while True:
            try:
                # Non-blocking check for new results
                result = client_queue.get(timeout=1.0)
                yield f"data: {json.dumps(result)}\n\n"
            except queue.Empty:
                # Send heartbeat to keep connection alive
                yield f": heartbeat\n\n"

            # Check if optimization is complete
            if not unified_status["running"] and client_queue.empty():
                yield f"data: {json.dumps({'type': 'complete', 'message': 'Optimization complete'})}\n\n"
                break

            await asyncio.sleep(0.1)
    finally:
        with streaming_lock:
            try:
                streaming_clients.remove(client_queue)
            except ValueError:
                pass  # Already removed, ignore


@app.get("/api/unified-stream")
async def stream_unified_results():
    """
    SSE endpoint for streaming optimization results in real-time.

    Connect to this endpoint to receive results as each strategy completes.

    Event types:
    - connected: Initial connection confirmation
    - strategy_result: A strategy has completed optimization
    - progress: Progress update
    - complete: Optimization finished

    Example usage (JavaScript):
    ```javascript
    const eventSource = new EventSource('/api/unified-stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'strategy_result') {
            console.log('New strategy:', data.strategy_name, 'Score:', data.composite_score);
        }
    };
    ```
    """
    return StreamingResponse(
        generate_sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/unified-report")
async def get_unified_report():
    """Get full unified optimization report with Top 10 strategies"""
    if unified_status["report"] is None:
        raise HTTPException(status_code=404, detail="No report available. Run unified optimization first.")
    return unified_status["report"]


@app.get("/api/unified-pinescript/{rank}")
async def get_unified_pinescript(rank: int = 1, engine: str = "tradingview"):
    """
    Generate Pine Script for any of the Top 10 unified optimization results.

    Args:
        rank: Strategy rank (1-10)
        engine: Calculation engine - "tradingview" or "native"

    Returns:
        Pine Script code for the specified strategy
    """
    if unified_status["report"] is None:
        raise HTTPException(status_code=404, detail="No optimization results. Run unified optimization first.")

    report = unified_status["report"]

    # Validate engine parameter
    valid_engines = ["tradingview", "native"]
    if engine not in valid_engines:
        raise HTTPException(status_code=400, detail=f"Invalid engine. Must be one of: {valid_engines}")

    # In comparison mode, look up from the specific engine's top_10
    if report.get("mode") == "comparison" and report.get("engine_reports"):
        engine_report = report["engine_reports"].get(engine, {})
        top_10 = engine_report.get("top_10", [])
    else:
        top_10 = report.get("top_10", [])

    if rank < 1 or rank > len(top_10):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_10)}")

    strategy_data = top_10[rank - 1]
    strategy_name = strategy_data["strategy_name"]
    params = strategy_data["params"]
    metrics = strategy_data["metrics"]
    entry_rule = strategy_data.get("entry_rule")
    direction = strategy_data.get("direction")

    # Get trading parameters from report (passed from UI)
    position_size_pct = report.get("position_size_pct", 75.0)
    capital = report.get("capital", 1000.0)
    date_range = report.get("date_range")

    # Use EXACT-MATCH generator for guaranteed TradingView compatibility
    generator = PineScriptGenerator()
    pine_script = generator.generate_exact_match(strategy_name, params, metrics, entry_rule, direction,
                                                  position_size_pct=position_size_pct, capital=capital,
                                                  engine=engine, date_range=date_range)

    return {
        "rank": rank,
        "strategy_name": strategy_name,
        "strategy_category": strategy_data.get("strategy_category", "unknown"),
        "tp_percent": params.get("tp_percent", 1.0),
        "sl_percent": params.get("sl_percent", 3.0),
        "metrics": metrics,
        "engine": engine,
        "pinescript": pine_script
    }


@app.get("/api/unified-pinescript-all")
async def get_all_unified_pinescripts(engine: str = "tradingview"):
    """
    Generate Pine Scripts for ALL Top 10 strategies.

    Args:
        engine: Calculation engine - "tradingview" (default), "pandas_ta", or "mihakralj"

    Returns:
        List of Pine Script codes for all top strategies
    """
    if unified_status["report"] is None:
        raise HTTPException(status_code=404, detail="No optimization results. Run unified optimization first.")

    # Validate engine parameter
    valid_engines = ["tradingview", "pandas_ta", "mihakralj"]
    if engine not in valid_engines:
        raise HTTPException(status_code=400, detail=f"Invalid engine. Must be one of: {valid_engines}")

    top_10 = unified_status["report"].get("top_10", [])
    generator = PineScriptGenerator()

    # Get trading parameters from report (passed from UI)
    position_size_pct = unified_status["report"].get("position_size_pct", 75.0)
    capital = unified_status["report"].get("capital", 1000.0)
    date_range = unified_status["report"].get("date_range")

    results = []
    for i, strategy_data in enumerate(top_10, 1):
        strategy_name = strategy_data["strategy_name"]
        params = strategy_data["params"]
        metrics = strategy_data["metrics"]
        entry_rule = strategy_data.get("entry_rule")
        direction = strategy_data.get("direction")

        # Use EXACT-MATCH generator for guaranteed TradingView compatibility
        pine_script = generator.generate_exact_match(strategy_name, params, metrics, entry_rule, direction,
                                                      position_size_pct=position_size_pct, capital=capital,
                                                      engine=engine, date_range=date_range)

        results.append({
            "rank": i,
            "strategy_name": strategy_name,
            "strategy_category": strategy_data.get("strategy_category", "unknown"),
            "tp_percent": params.get("tp_percent", 1.0),
            "sl_percent": params.get("sl_percent", 3.0),
            "metrics": metrics,
            "engine": engine,
            "pinescript": pine_script
        })

    return {"strategies": results, "engine": engine}


@app.get("/api/download-unified-pinescript/{rank}")
async def download_unified_pinescript(rank: int = 1, engine: str = "tradingview"):
    """Download Pine Script file for a specific ranked strategy"""
    result = await get_unified_pinescript(rank, engine=engine)

    # Create output file
    filename = f"{result['strategy_name']}_rank{rank}.pine"
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w") as f:
        f.write(result["pinescript"])

    return FileResponse(
        output_path,
        media_type="text/plain",
        filename=filename
    )


@app.get("/api/unified-trades-csv/{rank}")
async def download_trades_csv(rank: int = 1):
    """
    Download trades list as CSV for a specific ranked strategy.
    Matches TradingView's trade list export format.
    """
    if unified_status["report"] is None:
        raise HTTPException(status_code=404, detail="No optimization results. Run unified optimization first.")

    top_10 = unified_status["report"].get("top_10", [])

    if rank < 1 or rank > len(top_10):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_10)}")

    strategy_data = top_10[rank - 1]
    strategy_name = strategy_data["strategy_name"]
    trades_list = strategy_data.get("trades_list", [])

    if not trades_list:
        raise HTTPException(status_code=404, detail="No trades found for this strategy")

    # Get currency info
    source_currency = unified_status["report"].get("source_currency", "USD")
    has_conversion = unified_status["report"].get("currency_conversion_enabled", False)

    # Build CSV content (TradingView-style format with dual currency support)
    if has_conversion:
        csv_lines = [
            "Trade #,Type,Entry Time,Exit Time,Entry Price,Exit Price,"
            "Position Size (USD),Position Size (GBP),Position Size (qty),"
            "Net P&L USD,Net P&L GBP,Net P&L %,"
            "Run-up USD,Run-up %,Drawdown USD,Drawdown %,"
            "Cumulative P&L USD,Cumulative P&L GBP,USD/GBP Rate,Result"
        ]
    else:
        csv_lines = [
            f"Trade #,Type,Entry Time,Exit Time,Entry Price,Exit Price,"
            f"Position Size ({source_currency}),Position Size (qty),"
            f"Net P&L {source_currency},Net P&L %,"
            f"Run-up {source_currency},Run-up %,Drawdown {source_currency},Drawdown %,"
            f"Cumulative P&L {source_currency},Result"
        ]

    for trade in trades_list:
        if has_conversion:
            csv_lines.append(
                f"{trade['trade_num']},"
                f"{trade['direction'].upper()},"
                f"{trade['entry_time']},"
                f"{trade['exit_time']},"
                f"{trade['entry']},"
                f"{trade['exit']},"
                f"{trade['position_size']},"
                f"{trade.get('position_size_gbp', trade['position_size'])},"
                f"{trade['position_qty']},"
                f"{trade['pnl']},"
                f"{trade.get('pnl_gbp', trade['pnl'])},"
                f"{trade['pnl_pct']},"
                f"{trade['run_up']},"
                f"{trade['run_up_pct']},"
                f"{trade['drawdown']},"
                f"{trade['drawdown_pct']},"
                f"{trade['cumulative_pnl']},"
                f"{trade.get('cumulative_pnl_gbp', trade['cumulative_pnl'])},"
                f"{trade.get('usd_gbp_rate', 1.0)},"
                f"{trade['result']}"
            )
        else:
            csv_lines.append(
                f"{trade['trade_num']},"
                f"{trade['direction'].upper()},"
                f"{trade['entry_time']},"
                f"{trade['exit_time']},"
                f"{trade['entry']},"
                f"{trade['exit']},"
                f"{trade['position_size']},"
                f"{trade['position_qty']},"
                f"{trade['pnl']},"
                f"{trade['pnl_pct']},"
                f"{trade['run_up']},"
                f"{trade['run_up_pct']},"
                f"{trade['drawdown']},"
                f"{trade['drawdown_pct']},"
                f"{trade['cumulative_pnl']},"
                f"{trade['result']}"
            )

    csv_content = "\n".join(csv_lines)

    # Create output file
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{strategy_name}_{date_str}.csv"
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w") as f:
        f.write(csv_content)

    return FileResponse(
        output_path,
        media_type="text/csv",
        filename=filename
    )


@app.get("/api/tradingview-link/{rank}")
async def get_tradingview_link(rank: int = 1, engine: str = "tradingview"):
    """
    Generate TradingView deep link for a strategy.
    Opens TradingView with the correct exchange:symbol and timeframe.
    Also returns the Pine Script for clipboard copy.

    Args:
        rank: Strategy rank (1-10)
        engine: Calculation engine - "tradingview" or "native"
    """
    if unified_status["report"] is None:
        raise HTTPException(status_code=404, detail="No optimization results. Run unified optimization first.")

    report = unified_status["report"]

    # Validate engine parameter
    valid_engines = ["tradingview", "native"]
    if engine not in valid_engines:
        raise HTTPException(status_code=400, detail=f"Invalid engine. Must be one of: {valid_engines}")

    # In comparison mode, look up from the specific engine's top_10
    if report.get("mode") == "comparison" and report.get("engine_reports"):
        engine_report = report["engine_reports"].get(engine, {})
        top_10 = engine_report.get("top_10", [])
    else:
        top_10 = report.get("top_10", [])

    if rank < 1 or rank > len(top_10):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_10)}")

    # Get chart settings from report
    raw_exchange = report.get("exchange", "")
    symbol = report.get("symbol", "BTCUSDT")
    timeframe = report.get("timeframe", "15m")

    # Map data source to TradingView-supported exchange
    # TradingView supports: BINANCE, KRAKEN, COINBASE, BITSTAMP, etc.
    # Yahoo Finance and other unsupported sources need mapping
    tv_exchange_map = {
        "BINANCE": "BINANCE",
        "KRAKEN": "KRAKEN",
        "COINBASE": "COINBASE",
        "BITSTAMP": "BITSTAMP",
        "BLOFIN": "BLOFIN",
    }

    # Check if exchange is supported, otherwise infer from symbol
    exchange = tv_exchange_map.get(raw_exchange.upper() if raw_exchange else "", None)
    exchange_warning = None

    if not exchange:
        # Infer exchange from symbol
        if symbol and "GBP" in symbol.upper():
            exchange = "KRAKEN"  # KRAKEN has GBP pairs
        elif symbol and "USDT" in symbol.upper():
            exchange = "BINANCE"  # BINANCE has most USDT pairs
        elif symbol and "USD" in symbol.upper():
            exchange = "COINBASE"  # COINBASE has USD pairs
        else:
            exchange = "BINANCE"  # Default fallback

        if raw_exchange and raw_exchange.upper() not in tv_exchange_map:
            exchange_warning = f"Data source '{raw_exchange}' not available on TradingView. Using {exchange} instead - slight price differences may occur."

    # Convert timeframe to TradingView format: "15m" → "15", "1h" → "60"
    tf_map = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "4h": "240", "1d": "D", "1w": "W"
    }
    tv_interval = tf_map.get(timeframe, "15")

    # Build TradingView URL
    tv_url = f"https://www.tradingview.com/chart/?symbol={exchange}:{symbol}&interval={tv_interval}"

    # Get strategy data and generate Pine Script
    strategy_data = top_10[rank - 1]
    strategy_name = strategy_data["strategy_name"]
    params = strategy_data["params"]
    metrics = strategy_data["metrics"]
    entry_rule = strategy_data.get("entry_rule")
    direction = strategy_data.get("direction")

    # Get trading parameters from report
    position_size_pct = report.get("position_size_pct", 75.0)
    capital = report.get("capital", 1000.0)
    date_range = report.get("date_range")

    # Validate engine parameter
    valid_engines = ["tradingview", "pandas_ta", "mihakralj"]
    if engine not in valid_engines:
        raise HTTPException(status_code=400, detail=f"Invalid engine. Must be one of: {valid_engines}")

    # Generate Pine Script
    generator = PineScriptGenerator()
    pine_script = generator.generate_exact_match(
        strategy_name, params, metrics, entry_rule, direction,
        position_size_pct=position_size_pct, capital=capital,
        engine=engine, date_range=date_range
    )

    return {
        "tradingview_url": tv_url,
        "pinescript": pine_script,
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_name": strategy_name,
        "rank": rank,
        "engine": engine,
        "exchange_warning": exchange_warning  # None if no warning
    }


# =============================================================================
# DATABASE API ENDPOINTS - Strategy Persistence
# =============================================================================

@app.get("/api/db/stats")
async def get_database_stats():
    """Get overall database statistics."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/db/filter-options")
async def get_filter_options():
    """Get distinct symbols, timeframes, and date range for filter dropdowns."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()

        # Get distinct symbols
        cursor.execute("SELECT DISTINCT symbol FROM strategies WHERE symbol IS NOT NULL ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]

        # Get distinct timeframes
        cursor.execute("SELECT DISTINCT timeframe FROM strategies WHERE timeframe IS NOT NULL ORDER BY timeframe")
        timeframes = [row[0] for row in cursor.fetchall()]

        # Get date range
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM strategies")
        date_row = cursor.fetchone()
        date_range = {
            "min": date_row[0] if date_row[0] else None,
            "max": date_row[1] if date_row[1] else None
        }

        conn.close()
        return {
            "symbols": symbols,
            "timeframes": timeframes,
            "date_range": date_range
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/db/strategies")
async def get_saved_strategies(
    limit: int = 20,
    symbol: str = None,
    timeframe: str = None,
    min_win_rate: float = 0.0
):
    """Get top strategies from database by composite score."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_top_strategies(
            limit=limit,
            symbol=symbol,
            timeframe=timeframe,
            min_trades=0,
            min_win_rate=min_win_rate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/db/strategies/best-win-rate")
async def get_best_by_win_rate(limit: int = 10):
    """Get strategies with highest win rate."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_best_by_win_rate(limit=limit, min_trades=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/db/strategies/best-profit-factor")
async def get_best_by_profit_factor(limit: int = 10):
    """Get strategies with highest profit factor."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_best_by_profit_factor(limit=limit, min_trades=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/db/strategies/search")
async def search_strategies(
    strategy_name: str = None,
    category: str = None,
    min_win_rate: float = None,
    min_pnl: float = None,
    symbol: str = None,
    timeframe: str = None
):
    """Search strategies with various filters."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.search_strategies(
            strategy_name=strategy_name,
            category=category,
            min_win_rate=min_win_rate,
            min_pnl=min_pnl,
            symbol=symbol,
            timeframe=timeframe
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/db/strategies/{strategy_id}")
async def get_strategy_by_id(strategy_id: int):
    """Get a single strategy by its ID."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        strategy = db.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/db/strategies/{strategy_id}/pinescript")
async def get_strategy_pinescript_from_db(strategy_id: int):
    """Generate Pine Script for a saved strategy."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        strategy = db.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Extract entry_rule and direction from params
        params = strategy['params']
        entry_rule = params.get('entry_rule')
        direction = params.get('direction') or strategy.get('trade_mode', 'long')

        # Get position_size_pct from optimization run (stored as risk_percent)
        position_size_pct = 75.0  # default
        capital = 1000.0  # default
        run_id = strategy.get('optimization_run_id')
        if run_id:
            run = db.get_optimization_run_by_id(run_id)
            if run:
                position_size_pct = run.get('risk_percent', 75.0)
                capital = run.get('capital', 1000.0)

        # Also check if stored in params (override if present)
        position_size_pct = params.get('position_size_pct', position_size_pct)
        capital = params.get('capital', capital)

        # Build date range from strategy's data_start/data_end
        date_range = None
        data_start = strategy.get('data_start')
        data_end = strategy.get('data_end')
        if data_start and data_end:
            # Parse ISO format dates and extract date/time components
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(data_start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(data_end.replace('Z', '+00:00'))
                date_range = {
                    'enabled': True,
                    'startDate': start_dt.strftime('%Y-%m-%d'),
                    'startTime': start_dt.strftime('%H:%M'),
                    'endDate': end_dt.strftime('%Y-%m-%d'),
                    'endTime': end_dt.strftime('%H:%M'),
                }
            except Exception as e:
                print(f"Date range parsing error: {e}")

        generator = PineScriptGenerator()
        pinescript = generator.generate_exact_match(
            strategy['strategy_name'],
            params,
            {
                'total_trades': strategy['total_trades'],
                'win_rate': strategy['win_rate'],
                'total_pnl': strategy['total_pnl'],
                'profit_factor': strategy['profit_factor'],
                'max_drawdown': strategy['max_drawdown'],
            },
            entry_rule=entry_rule,
            direction=direction,
            position_size_pct=position_size_pct,
            capital=capital,
            date_range=date_range
        )

        return {
            'strategy_id': strategy_id,
            'strategy_name': strategy['strategy_name'],
            'tp_percent': strategy['tp_percent'],
            'sl_percent': strategy['sl_percent'],
            'pinescript': pinescript
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/validate-strategy")
async def validate_strategy(request: ValidateStrategyRequest):
    """
    Validate a strategy by testing it against different time periods.
    Uses EXACT SAME configuration from original:
    - Same data source (Binance/Yahoo)
    - Same symbol & timeframe
    - Same TP% and SL%
    - Same entry_rule and direction
    - Same indicator parameters
    """
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        strategy = db.get_strategy_by_id(request.strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Extract ALL parameters from the strategy
        params = strategy.get('params', {})
        symbol = strategy.get('symbol', 'BTCUSDT')  # Default to USDT

        # === CHECK FOR SUPPORTED PAIRS (Binance USDT only) ===
        supported_quotes = ['USDT', 'USDC', 'BUSD']
        symbol_supported = any(symbol.endswith(q) for q in supported_quotes)
        if not symbol_supported:
            raise HTTPException(
                status_code=400,
                detail=f"Symbol {symbol} not supported. Only Binance USDT pairs are available."
            )

        # Detect data source from symbol format if not stored
        # Binance symbols: BTCUSDT, ETHUSDT (no hyphen, ends in USDT/BUSD)
        # Yahoo symbols: BTC-GBP, BTC-USD (has hyphen)
        data_source = strategy.get('data_source')
        if not data_source:
            if '-' in symbol:
                data_source = 'yahoo'
            elif symbol.endswith('USDT') or symbol.endswith('BUSD') or symbol.endswith('BTC'):
                data_source = 'binance'
            else:
                data_source = 'binance'  # Default to Binance for most crypto pairs
        timeframe = strategy.get('timeframe', '15m')
        tp_percent = strategy.get('tp_percent', 2.0)
        sl_percent = strategy.get('sl_percent', 5.0)
        entry_rule = params.get('entry_rule', 'rsi_oversold')
        direction = params.get('direction', strategy.get('trade_mode', 'long'))

        # Get position size and original period from optimization run
        original_position_size_pct = 75.0
        original_capital = 1000.0
        original_months = 1.0  # Default to 1 month if not found
        run_id = strategy.get('optimization_run_id')
        if run_id:
            run = db.get_optimization_run_by_id(run_id)
            if run:
                original_position_size_pct = run.get('risk_percent', 75.0)
                original_capital = run.get('capital', 1000.0)
                original_months = run.get('months', 1.0)
        original_position_size_pct = params.get('position_size_pct', original_position_size_pct)
        original_capital = params.get('capital', original_capital)

        # Use request overrides if provided, otherwise use original values
        capital = request.capital if request.capital is not None else original_capital
        position_size_pct = request.position_size_pct if request.position_size_pct is not None else original_position_size_pct

        # Calculate original period display text
        original_days = int(original_months * 30)
        if original_days < 7:
            original_period_text = f"{original_days} days"
        elif original_days < 30:
            original_period_text = f"{original_days // 7} week{'s' if original_days >= 14 else ''}"
        elif original_days < 60:
            original_period_text = f"{original_days} days"
        else:
            original_period_text = f"{original_months:.1f} months".replace('.0 ', ' ')

        # Convert timeframe to minutes for data limit check
        tf_minutes = int(timeframe.replace('m', '').replace('h', '')) if 'm' in timeframe else int(timeframe.replace('h', '')) * 60

        # Data source limits (days)
        if data_source and 'yahoo' in data_source.lower():
            data_limits = {1: 7, 5: 60, 15: 60, 30: 60, 60: 730, 1440: 9999}
        else:
            # Binance typically has more history
            data_limits = {1: 365, 5: 365, 15: 365, 30: 365, 60: 730, 1440: 9999}
        max_days = data_limits.get(tf_minutes, 60)

        # Validation periods: name, months, days
        validation_periods = [
            {"period": "1 week", "months": 0.25, "days": 7},
            {"period": "1 month", "months": 1.0, "days": 30},
            {"period": "3 months", "months": 3.0, "days": 90},
            {"period": "6 months", "months": 6.0, "days": 180},
            {"period": "1 year", "months": 12.0, "days": 365},
            {"period": "2 years", "months": 24.0, "days": 730},
        ]

        # Original metrics (baseline)
        original_metrics = {
            "total_trades": strategy.get('total_trades', 0),
            "win_rate": strategy.get('win_rate', 0),
            "total_pnl": strategy.get('total_pnl', 0),
            "profit_factor": strategy.get('profit_factor', 0),
            "max_drawdown": strategy.get('max_drawdown', 0),
        }

        # Run validation for each period
        validations = []
        for vp in validation_periods:
            if vp["days"] > max_days:
                # Period exceeds data source limit
                validations.append({
                    "period": vp["period"],
                    "months": vp["months"],
                    "metrics": None,
                    "status": "limit_exceeded",
                    "message": f"Exceeds {data_source} {tf_minutes}m limit of {max_days} days"
                })
                continue

            try:
                # Fetch fresh data for this period
                if data_source and 'binance' in data_source.lower():
                    fetcher = BinanceDataFetcher()
                else:
                    fetcher = YFinanceDataFetcher()

                df = await fetcher.fetch_ohlcv(pair=symbol, interval=tf_minutes, months=vp["months"])

                if len(df) < 50:
                    validations.append({
                        "period": vp["period"],
                        "months": vp["months"],
                        "metrics": None,
                        "status": "insufficient_data",
                        "message": f"Only {len(df)} candles available"
                    })
                    continue

                # Create engine and run backtest
                engine = StrategyEngine(df)
                result = engine.backtest(
                    strategy=entry_rule,
                    direction=direction,
                    tp_percent=tp_percent,
                    sl_percent=sl_percent,
                    initial_capital=capital,
                    position_size_pct=position_size_pct,
                    commission_pct=0.1
                )

                # Determine status based on comparison with original
                status = "consistent"
                if result.total_trades == 0:
                    status = "no_trades"
                elif result.win_rate < original_metrics["win_rate"] * 0.8:
                    status = "degraded"
                elif result.profit_factor < original_metrics["profit_factor"] * 0.8:
                    status = "degraded"
                elif result.win_rate < original_metrics["win_rate"] * 0.95:
                    status = "minor_drop"

                validations.append({
                    "period": vp["period"],
                    "months": vp["months"],
                    "metrics": {
                        "total_trades": result.total_trades,
                        "win_rate": round(result.win_rate, 2),
                        "total_pnl": round(result.total_pnl, 2),
                        "profit_factor": round(result.profit_factor, 2),
                        "max_drawdown": round(result.max_drawdown, 2),
                    },
                    "status": status,
                    "message": None
                })

            except Exception as e:
                validations.append({
                    "period": vp["period"],
                    "months": vp["months"],
                    "metrics": None,
                    "status": "error",
                    "message": str(e)
                })

        return {
            "strategy": {
                "id": strategy.get('id'),
                "name": strategy.get('strategy_name'),
                "symbol": symbol,
                "timeframe": timeframe,
                "data_source": data_source,
                "tp_percent": tp_percent,
                "sl_percent": sl_percent,
                "entry_rule": entry_rule,
                "direction": direction,
                "capital": capital,
                "position_size_pct": position_size_pct,
            },
            "original": {
                **original_metrics,
                "period": original_period_text,
            },
            "validations": validations
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/db/runs")
async def get_optimization_runs(limit: int = 20):
    """Get recent optimization runs."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_optimization_runs(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/db/strategies/{strategy_id}")
async def delete_strategy(strategy_id: int):
    """Delete a strategy from the database."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        if db.delete_strategy(strategy_id):
            return {"message": f"Strategy {strategy_id} deleted"}
        raise HTTPException(status_code=404, detail="Strategy not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/db/clear")
async def clear_database():
    """Clear all strategies from the database."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        strategies = db.get_all_strategies()
        deleted = 0
        for s in strategies:
            if db.delete_strategy(s['id']):
                deleted += 1
        return {"message": f"Deleted {deleted} strategies", "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/db/remove-duplicates")
async def remove_duplicate_strategies():
    """Remove duplicate strategies from the database, keeping only the most recent."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        removed = db.remove_duplicates()
        return {"message": f"Removed {removed} duplicate strategies", "removed_count": removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TRADINGVIEW COMPARISON FEATURE - Compare our results vs TradingView
# =============================================================================

# Store comparison results
comparison_data = {
    "tv_trades": None,
    "our_trades": None,
    "comparison": None,
    "strategy_rank": None
}


@app.post("/api/upload-tv-comparison/{rank}")
async def upload_tv_comparison(rank: int, file: UploadFile = File(...)):
    """
    Upload TradingView trade list export CSV for comparison.

    TradingView format (Entry long + Exit long rows per trade):
    Trade #,Type,Date/Time,Signal,Price GBP,Position size (qty),...

    Args:
        rank: Strategy rank to compare against (1-10)
        file: TradingView exported trade list CSV
    """
    global comparison_data

    if unified_status["report"] is None:
        raise HTTPException(status_code=400, detail="Run optimization first before comparing")

    top_10 = unified_status["report"].get("top_10", [])
    if rank < 1 or rank > len(top_10):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_10)}")

    try:
        contents = await file.read()

        # Parse TradingView CSV
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df.columns = df.columns.str.strip()

        # TradingView exports Entry + Exit rows per trade
        # Group by Trade # to combine entry/exit into single trade records
        tv_trades = []

        for trade_num in df['Trade #'].unique():
            trade_rows = df[df['Trade #'] == trade_num]

            entry_row = trade_rows[trade_rows['Type'].str.contains('Entry', case=False, na=False)]
            exit_row = trade_rows[trade_rows['Type'].str.contains('Exit', case=False, na=False)]

            if len(entry_row) == 0 or len(exit_row) == 0:
                continue

            entry_row = entry_row.iloc[0]
            exit_row = exit_row.iloc[0]

            # Extract direction from Signal or Type
            direction = 'long' if 'long' in str(entry_row.get('Signal', '')).lower() or 'long' in str(entry_row.get('Type', '')).lower() else 'short'

            tv_trades.append({
                'trade_num': int(trade_num),
                'direction': direction.upper(),
                'entry_time': str(entry_row.get('Date/Time', '')),
                'exit_time': str(exit_row.get('Date/Time', '')),
                'entry_price': float(entry_row.get('Price GBP', 0)),
                'exit_price': float(exit_row.get('Price GBP', 0)),
                'pnl': float(exit_row.get('Net P&L GBP', 0)),
                'pnl_pct': float(str(exit_row.get('Net P&L %', '0')).replace('%', '')),
                'cumulative_pnl': float(exit_row.get('Cumulative P&L GBP', 0))
            })

        # Get our trades for this strategy
        our_strategy = top_10[rank - 1]
        our_trades_list = our_strategy.get("trades_list", [])

        # Perform comparison
        comparison = compare_trades(tv_trades, our_trades_list)

        # Store for UI access
        comparison_data["tv_trades"] = tv_trades
        comparison_data["our_trades"] = our_trades_list
        comparison_data["comparison"] = comparison
        comparison_data["strategy_rank"] = rank
        comparison_data["strategy_name"] = our_strategy["strategy_name"]

        return {
            "status": "success",
            "tv_trade_count": len(tv_trades),
            "our_trade_count": len(our_trades_list),
            "comparison": comparison
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")


def compare_trades(tv_trades: list, our_trades: list) -> dict:
    """
    Compare TradingView trades vs our backtester trades.

    Returns detailed comparison with:
    - Summary stats
    - Trade-by-trade matching
    - Discrepancies highlighted
    """
    # Summary comparison
    tv_total_pnl = sum(t['pnl'] for t in tv_trades) if tv_trades else 0
    our_total_pnl = sum(t['pnl'] for t in our_trades) if our_trades else 0

    tv_wins = len([t for t in tv_trades if t['pnl'] > 0])
    our_wins = len([t for t in our_trades if t.get('pnl', 0) > 0])

    tv_win_rate = (tv_wins / len(tv_trades) * 100) if tv_trades else 0
    our_win_rate = (our_wins / len(our_trades) * 100) if our_trades else 0

    # Try to match trades by time (within 1 hour tolerance)
    matched_trades = []
    unmatched_tv = []
    unmatched_ours = []

    our_trades_copy = list(our_trades)

    for tv_trade in tv_trades:
        tv_entry = pd.to_datetime(tv_trade['entry_time'])
        matched = False

        for i, our_trade in enumerate(our_trades_copy):
            our_entry = pd.to_datetime(our_trade['entry_time'])
            time_diff = abs((tv_entry - our_entry).total_seconds() / 3600)  # hours

            if time_diff <= 1:  # Within 1 hour
                pnl_diff = tv_trade['pnl'] - our_trade['pnl']
                matched_trades.append({
                    'tv_trade': tv_trade,
                    'our_trade': our_trade,
                    'time_diff_hours': round(time_diff, 2),
                    'pnl_diff': round(pnl_diff, 2),
                    'pnl_diff_pct': round((pnl_diff / abs(our_trade['pnl']) * 100) if our_trade['pnl'] != 0 else 0, 1),
                    'match_quality': 'exact' if time_diff < 0.1 else 'close'
                })
                our_trades_copy.pop(i)
                matched = True
                break

        if not matched:
            unmatched_tv.append(tv_trade)

    unmatched_ours = our_trades_copy

    return {
        "summary": {
            "tradingview": {
                "trade_count": len(tv_trades),
                "total_pnl": round(tv_total_pnl, 2),
                "win_rate": round(tv_win_rate, 1),
                "wins": tv_wins,
                "losses": len(tv_trades) - tv_wins
            },
            "our_system": {
                "trade_count": len(our_trades),
                "total_pnl": round(our_total_pnl, 2),
                "win_rate": round(our_win_rate, 1),
                "wins": our_wins,
                "losses": len(our_trades) - our_wins
            },
            "difference": {
                "trade_count_diff": len(our_trades) - len(tv_trades),
                "pnl_diff": round(our_total_pnl - tv_total_pnl, 2),
                "win_rate_diff": round(our_win_rate - tv_win_rate, 1)
            }
        },
        "matched_trades": matched_trades,
        "unmatched_tv": unmatched_tv,
        "unmatched_ours": unmatched_ours,
        "match_rate": round(len(matched_trades) / len(tv_trades) * 100, 1) if tv_trades else 0
    }


@app.get("/api/comparison")
async def get_comparison():
    """Get the current comparison data between TradingView and our system"""
    if comparison_data["comparison"] is None:
        raise HTTPException(status_code=404, detail="No comparison data. Upload TradingView trade export first.")

    return {
        "strategy_rank": comparison_data["strategy_rank"],
        "strategy_name": comparison_data["strategy_name"],
        "comparison": comparison_data["comparison"],
        "tv_trades": comparison_data["tv_trades"],
        "our_trades": comparison_data["our_trades"]
    }


@app.delete("/api/comparison")
async def clear_comparison():
    """Clear comparison data"""
    global comparison_data
    comparison_data = {
        "tv_trades": None,
        "our_trades": None,
        "comparison": None,
        "strategy_rank": None
    }
    return {"status": "cleared"}


# =============================================================================
# INDICATOR ENGINE COMPARISON - Compare TV Default vs pandas_ta vs mihakralj
# =============================================================================

from indicator_engines import MultiEngineCalculator, IndicatorEngine

@app.get("/api/indicator-comparison")
async def get_indicator_comparison():
    """
    Compare indicator calculations across all three engines:
    - TradingView Default (matches ta.* functions)
    - pandas_ta (current library)
    - mihakralj (mathematically rigorous)

    Returns comparison for last 100 bars of loaded data.
    """
    global current_df

    if current_df is None or len(current_df) == 0:
        raise HTTPException(status_code=400, detail="No data loaded. Fetch data first.")

    try:
        # Use last 100 bars for comparison
        df_sample = current_df.tail(100).copy()
        calc = MultiEngineCalculator(df_sample)

        # Get comparison summary
        summary = calc.get_comparison_summary()

        # Get detailed comparison for each indicator (last 20 values)
        detailed = {}

        # RSI comparison
        rsi_df = calc.compare_indicator('rsi', length=14).tail(20)
        detailed['rsi'] = {
            'values': {
                'tradingview': rsi_df['tradingview'].round(4).tolist(),
                'pandas_ta': rsi_df['pandas_ta'].round(4).tolist(),
                'mihakralj': rsi_df['mihakralj'].round(4).tolist(),
            },
            'timestamps': [str(t) for t in rsi_df.index.tolist()],
        }

        # EMA comparison
        ema_df = calc.compare_indicator('ema', length=20).tail(20)
        detailed['ema'] = {
            'values': {
                'tradingview': ema_df['tradingview'].round(2).tolist(),
                'pandas_ta': ema_df['pandas_ta'].round(2).tolist(),
                'mihakralj': ema_df['mihakralj'].round(2).tolist(),
            },
            'timestamps': [str(t) for t in ema_df.index.tolist()],
        }

        # SMA comparison
        sma_df = calc.compare_indicator('sma', length=20).tail(20)
        detailed['sma'] = {
            'values': {
                'tradingview': sma_df['tradingview'].round(2).tolist(),
                'pandas_ta': sma_df['pandas_ta'].round(2).tolist(),
                'mihakralj': sma_df['mihakralj'].round(2).tolist(),
            },
            'timestamps': [str(t) for t in sma_df.index.tolist()],
        }

        # ATR comparison
        atr_df = calc.compare_indicator('atr', length=14).tail(20)
        detailed['atr'] = {
            'values': {
                'tradingview': atr_df['tradingview'].round(2).tolist(),
                'pandas_ta': atr_df['pandas_ta'].round(2).tolist(),
                'mihakralj': atr_df['mihakralj'].round(2).tolist(),
            },
            'timestamps': [str(t) for t in atr_df.index.tolist()],
        }

        # MACD comparison (special handling for tuple return)
        macd_tv = calc.macd_tradingview()
        macd_pta = calc.macd_pandas_ta()
        macd_mih = calc.macd_mihakralj()

        detailed['macd'] = {
            'values': {
                'tradingview': macd_tv[0].tail(20).round(2).tolist(),
                'pandas_ta': macd_pta[0].tail(20).round(2).tolist(),
                'mihakralj': macd_mih[0].tail(20).round(2).tolist(),
            },
            'timestamps': [str(t) for t in macd_tv[0].tail(20).index.tolist()],
        }

        # Stochastic comparison
        stoch_tv = calc.stoch_tradingview()
        stoch_pta = calc.stoch_pandas_ta()
        stoch_mih = calc.stoch_mihakralj()

        detailed['stochastic'] = {
            'values': {
                'tradingview': stoch_tv[0].tail(20).round(2).tolist(),
                'pandas_ta': stoch_pta[0].tail(20).round(2).tolist(),
                'mihakralj': stoch_mih[0].tail(20).round(2).tolist(),
            },
            'timestamps': [str(t) for t in stoch_tv[0].tail(20).index.tolist()],
        }

        # Bollinger Bands comparison
        bb_tv = calc.bbands_tradingview()
        bb_pta = calc.bbands_pandas_ta()
        bb_mih = calc.bbands_mihakralj()

        detailed['bollinger_upper'] = {
            'values': {
                'tradingview': bb_tv[1].tail(20).round(2).tolist(),
                'pandas_ta': bb_pta[1].tail(20).round(2).tolist(),
                'mihakralj': bb_mih[1].tail(20).round(2).tolist(),
            },
            'timestamps': [str(t) for t in bb_tv[1].tail(20).index.tolist()],
        }

        return {
            "summary": summary,
            "detailed": detailed,
            "sample_size": len(df_sample),
            "engines": ["tradingview", "pandas_ta", "mihakralj"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error comparing indicators: {str(e)}")


@app.get("/api/indicator-engines")
async def list_indicator_engines():
    """List available indicator calculation engines"""
    return {
        "engines": [
            {
                "id": "tradingview",
                "name": "TradingView Default",
                "description": "Matches TradingView's built-in ta.* functions exactly"
            },
            {
                "id": "pandas_ta",
                "name": "pandas_ta",
                "description": "pandas_ta library implementation (current default)"
            },
            {
                "id": "mihakralj",
                "name": "mihakralj/QuanTAlib",
                "description": "Mathematically rigorous implementations with proper warmup"
            }
        ],
        "current": "pandas_ta",
        "recommendation": "mihakralj - best for matching Pine Script output"
    }


# =============================================================================
# ELITE STRATEGIES - Automated Multi-Period Validation
# =============================================================================

@app.get("/api/elite/status")
async def get_elite_status():
    """
    Get elite validation status and counts.
    Returns validation progress and strategy counts by elite status.
    """
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        strategies = db.get_all_strategies()

        return {
            "total": len(strategies),
            "elite": sum(1 for s in strategies if s.get('elite_status') == 'elite'),
            "partial": sum(1 for s in strategies if s.get('elite_status') == 'partial'),
            "failed": sum(1 for s in strategies if s.get('elite_status') == 'failed'),
            "pending": sum(1 for s in strategies if s.get('elite_status') in [None, 'pending']),
            # Validation progress
            "validation_running": elite_validation_status["running"],
            "validation_paused": elite_validation_status["paused"],
            "validation_processed": elite_validation_status["processed"],
            "validation_total": elite_validation_status["total"],
            "validation_message": elite_validation_status["message"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/elite/strategies")
async def get_elite_strategies():
    """Return all validated strategies, sorted by elite_score descending"""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        strategies = db.get_all_strategies()
        # Filter validated (not pending) and sort by score descending
        validated = [s for s in strategies if s.get('elite_status') not in [None, 'pending']]
        validated.sort(key=lambda x: x.get('elite_score', 0), reverse=True)
        return validated
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/elite/run-validation")
async def run_elite_validation(background_tasks: BackgroundTasks):
    """
    Manually trigger validation (auto-validation runs continuously anyway).
    """
    global elite_validation_status

    if unified_status["running"]:
        return {"status": "blocked", "message": "Optimizer is running. Validation will auto-resume when idle."}

    if elite_validation_status["running"]:
        return {"status": "already_running", "message": "Elite validation already in progress"}

    # Force a validation cycle now
    background_tasks.add_task(validate_all_strategies_for_elite)
    return {"status": "started", "message": "Validation started"}


# =============================================================================
# AUTONOMOUS OPTIMIZER ENDPOINTS
# =============================================================================

@app.get("/api/autonomous/status")
async def get_autonomous_status():
    """Get current autonomous optimizer status"""
    return autonomous_optimizer_status


@app.post("/api/autonomous/toggle")
async def toggle_autonomous_optimizer():
    """Toggle the autonomous optimizer on/off"""
    global autonomous_optimizer_status

    if autonomous_optimizer_status["enabled"]:
        # Disable
        autonomous_optimizer_status["enabled"] = False
        autonomous_optimizer_status["auto_running"] = False
        autonomous_optimizer_status["message"] = "Disabled by user"
        return {"status": "disabled", "message": "Autonomous optimizer disabled"}
    else:
        # Enable
        autonomous_optimizer_status["enabled"] = True
        autonomous_optimizer_status["message"] = "Starting..."

        # Start the loop if not already running
        if not autonomous_optimizer_status["auto_running"]:
            asyncio.create_task(start_autonomous_optimizer())

        return {"status": "enabled", "message": "Autonomous optimizer enabled"}


@app.post("/api/autonomous/start")
async def start_autonomous():
    """Explicitly start autonomous optimizer"""
    global autonomous_optimizer_status

    autonomous_optimizer_status["enabled"] = True

    if not autonomous_optimizer_status["auto_running"]:
        asyncio.create_task(start_autonomous_optimizer())
        return {"status": "started", "message": "Autonomous optimizer started"}
    else:
        return {"status": "already_running", "message": "Already running"}


@app.post("/api/autonomous/stop")
async def stop_autonomous():
    """Stop autonomous optimizer"""
    global autonomous_optimizer_status

    autonomous_optimizer_status["enabled"] = False
    autonomous_optimizer_status["auto_running"] = False
    autonomous_optimizer_status["message"] = "Stopped by user"

    return {"status": "stopped", "message": "Autonomous optimizer stopped"}


@app.post("/api/autonomous/reset-cycle")
async def reset_autonomous_cycle():
    """Reset autonomous optimizer cycle to beginning"""
    global autonomous_optimizer_status

    autonomous_optimizer_status["cycle_index"] = 0
    autonomous_optimizer_status["completed_count"] = 0
    autonomous_optimizer_status["skipped_count"] = 0
    autonomous_optimizer_status["error_count"] = 0
    autonomous_optimizer_status["skipped_validations"] = []
    autonomous_optimizer_status["message"] = "Cycle reset - starting from beginning"

    return {"status": "reset", "message": "Cycle reset to beginning"}


@app.get("/api/autonomous/results")
async def get_autonomous_results():
    """Get summary of autonomous optimization results"""
    return {
        "completed_count": autonomous_optimizer_status["completed_count"],
        "error_count": autonomous_optimizer_status["error_count"],
        "total_combinations": autonomous_optimizer_status["total_combinations"],
        "cycle_index": autonomous_optimizer_status["cycle_index"],
        "best_strategy": autonomous_optimizer_status["best_strategy_found"],
        "last_result": autonomous_optimizer_status["last_result"],
        "last_completed_at": autonomous_optimizer_status["last_completed_at"],
    }


@app.get("/api/autonomous/history")
async def get_autonomous_history(limit: int = 50):
    """Get history of autonomous optimization runs"""
    return {
        "history": autonomous_runs_history[:limit],
        "total_runs": len(autonomous_runs_history),
    }


# =============================================================================
# AUTO-VALIDATION LOOP - Runs continuously in background
# =============================================================================

async def start_auto_elite_validation():
    """
    Continuous background loop that automatically validates strategies.
    Runs when optimizer is idle, pauses when optimizer is active.
    Checks for new pending strategies every 60 seconds.
    """
    global elite_validation_status

    if elite_validation_status["auto_running"]:
        return  # Already running

    elite_validation_status["auto_running"] = True
    print("[Elite Auto-Validation] Starting continuous background validation...")

    while elite_validation_status["auto_running"]:
        try:
            # Wait for optimizer to be idle
            while unified_status["running"]:
                elite_validation_status["message"] = "Waiting for optimizer to finish..."
                await asyncio.sleep(5)

            # Check if there are pending strategies
            db = get_strategy_db()
            strategies = db.get_all_strategies()
            pending = [s for s in strategies if s.get('elite_status') in [None, 'pending']]

            if pending:
                elite_validation_status["message"] = f"Found {len(pending)} pending strategies to validate"
                await validate_all_strategies_for_elite()
            else:
                # No pending - re-validate oldest strategy to keep data fresh
                validated = [s for s in strategies if s.get('elite_validated_at')]
                if validated:
                    # Sort by validated_at ascending (oldest first)
                    validated.sort(key=lambda x: x.get('elite_validated_at', ''))
                    oldest = validated[0]

                    # Reset to pending and re-validate
                    db.update_elite_status(
                        strategy_id=oldest['id'],
                        elite_status='pending',
                        periods_passed=0,
                        periods_total=0,
                        validation_data=None,
                        elite_score=0
                    )
                    elite_validation_status["message"] = f"Re-validating: {oldest['strategy_name']} (oldest)"
                    await validate_all_strategies_for_elite()
                else:
                    elite_validation_status["message"] = "No strategies to validate"
                    await asyncio.sleep(60)

        except Exception as e:
            print(f"[Elite Auto-Validation] Error: {e}")
            elite_validation_status["message"] = f"Error: {str(e)}"
            await asyncio.sleep(30)  # Wait before retrying

    print("[Elite Auto-Validation] Stopped")


async def validate_all_strategies_for_elite():
    """
    Background task: Validate ALL pending strategies across multiple time periods.
    Pauses when optimizer is running, resumes when idle.
    """
    global elite_validation_status

    validation_periods = [
        {"period": "1 week", "months": 0.25, "days": 7},
        {"period": "1 month", "months": 1.0, "days": 30},
        {"period": "3 months", "months": 3.0, "days": 90},
        {"period": "6 months", "months": 6.0, "days": 180},
        {"period": "1 year", "months": 12.0, "days": 365},
        {"period": "2 years", "months": 24.0, "days": 730},
    ]

    try:
        db = get_strategy_db()
        strategies = db.get_all_strategies()
        pending = [s for s in strategies if s.get('elite_status') in [None, 'pending']]

        if not pending:
            elite_validation_status["message"] = "No pending strategies to validate"
            return

        elite_validation_status["running"] = True
        elite_validation_status["total"] = len(pending)
        elite_validation_status["processed"] = 0
        elite_validation_status["message"] = f"Validating {len(pending)} strategies..."

        for strategy in pending:
            # === PRIORITY CHECK: Pause if optimizer starts ===
            while unified_status["running"]:
                elite_validation_status["paused"] = True
                elite_validation_status["message"] = "Paused (optimizer running)"
                await asyncio.sleep(2)  # Check every 2 seconds
            elite_validation_status["paused"] = False

            strategy_id = strategy.get('id')
            strategy_name = strategy.get('strategy_name', 'Unknown')
            elite_validation_status["current_strategy_id"] = strategy_id
            elite_validation_status["message"] = f"Validating: {strategy_name}"

            # Extract strategy parameters
            params = strategy.get('params', {})
            symbol = strategy.get('symbol', 'BTCGBP')
            data_source = strategy.get('data_source')

            # === SKIP NON-USDT PAIRS (Binance only supports USDT now) ===
            # Old strategies with GBP pairs won't work with Binance-only data fetcher
            supported_quotes = ['USDT', 'USDC', 'BUSD']
            symbol_supported = any(symbol.endswith(q) for q in supported_quotes)
            if not symbol_supported:
                print(f"[Elite Validation] Skipping {strategy_name} - {symbol} not supported (USDT pairs only)")
                elite_validation_status["processed"] += 1
                # Mark as "validated" to prevent infinite retry loops
                try:
                    db.update_elite_status(
                        strategy_id=strategy_id,
                        elite_status="skipped",
                        periods_passed=0,
                        periods_total=0,
                        validation_data=json.dumps({
                            "status": "skipped",
                            "reason": f"Symbol {symbol} not supported - Binance USDT pairs only"
                        }),
                        elite_score=0
                    )
                except Exception as e:
                    print(f"[Elite Validation] Error updating skipped strategy: {e}")
                continue

            # Detect data source from symbol format if not stored
            if not data_source:
                if '-' in symbol:
                    data_source = 'yahoo'
                elif symbol.endswith('USDT') or symbol.endswith('BUSD') or symbol.endswith('BTC'):
                    data_source = 'binance'
                else:
                    data_source = 'binance'

            timeframe = strategy.get('timeframe', '15m')
            tp_percent = strategy.get('tp_percent', 2.0)
            sl_percent = strategy.get('sl_percent', 5.0)
            entry_rule = params.get('entry_rule', 'rsi_oversold')
            direction = params.get('direction', strategy.get('trade_mode', 'long'))

            # Convert timeframe to minutes
            tf_minutes = int(timeframe.replace('m', '').replace('h', '')) if 'm' in timeframe else int(timeframe.replace('h', '')) * 60

            # Data source limits (days)
            if data_source and 'yahoo' in data_source.lower():
                data_limits = {1: 7, 5: 60, 15: 60, 30: 60, 60: 730, 1440: 9999}
            else:
                data_limits = {1: 365, 5: 365, 15: 365, 30: 365, 60: 730, 1440: 9999}
            max_days = data_limits.get(tf_minutes, 60)

            # Original metrics for comparison
            original_metrics = {
                "win_rate": strategy.get('win_rate', 0),
                "profit_factor": strategy.get('profit_factor', 0),
            }

            passed = 0
            total_testable = 0
            results = []

            for vp in validation_periods:
                # Check optimizer again before each period
                if unified_status["running"]:
                    break

                if vp["days"] > max_days:
                    # Skip - exceeds data limit
                    results.append({
                        "period": vp["period"],
                        "status": "limit_exceeded"
                    })
                    continue

                try:
                    # Fetch fresh data
                    if data_source and 'binance' in data_source.lower():
                        fetcher = BinanceDataFetcher()
                    else:
                        fetcher = YFinanceDataFetcher()

                    df = await fetcher.fetch_ohlcv(pair=symbol, interval=tf_minutes, months=vp["months"])

                    if len(df) < 50:
                        results.append({
                            "period": vp["period"],
                            "status": "insufficient_data"
                        })
                        continue

                    # Create engine and run backtest
                    engine = StrategyEngine(df)
                    result = engine.backtest(
                        strategy=entry_rule,
                        direction=direction,
                        tp_percent=tp_percent,
                        sl_percent=sl_percent,
                        initial_capital=1000.0,
                        position_size_pct=75.0,
                        commission_pct=0.1
                    )

                    # Determine status
                    status = "consistent"
                    if result.total_trades == 0:
                        status = "no_trades"
                    elif result.win_rate < original_metrics["win_rate"] * 0.8:
                        status = "degraded"
                    elif result.profit_factor < original_metrics["profit_factor"] * 0.8:
                        status = "degraded"
                    elif result.win_rate < original_metrics["win_rate"] * 0.95:
                        status = "minor_drop"

                    total_testable += 1
                    if status in ['consistent', 'minor_drop']:
                        passed += 1

                    results.append({
                        "period": vp["period"],
                        "status": status,
                        "trades": result.total_trades,
                        "win_rate": round(result.win_rate, 2),
                        "pnl": round(result.total_pnl, 2),
                    })

                except Exception as e:
                    results.append({
                        "period": vp["period"],
                        "status": "error",
                        "message": str(e)
                    })

            # Calculate elite score:
            # - Consistency points: 1 point per successful period (max 6)
            # - Profit bonus: total positive P&L / 100 (to scale £ to points)
            consistency_points = passed  # Number of periods passed (0-6)

            # Sum up P&L from all successful periods
            total_pnl = 0
            for result in results:
                if result.get('status') in ['consistent', 'minor_drop']:
                    pnl = result.get('pnl', 0)
                    if pnl > 0:
                        total_pnl += pnl

            # Profit bonus: £100 = 1 point
            profit_bonus = total_pnl / 100

            # Final score = consistency + profit bonus
            elite_score = consistency_points + profit_bonus

            # Determine elite status
            if total_testable == 0:
                elite_status = 'pending'
            elif passed == total_testable:
                elite_status = 'elite'
            elif passed >= total_testable * 0.7:
                elite_status = 'partial'
            else:
                elite_status = 'failed'

            # Update database with score
            db.update_elite_status(
                strategy_id=strategy_id,
                elite_status=elite_status,
                periods_passed=passed,
                periods_total=total_testable,
                validation_data=json.dumps(results),
                elite_score=elite_score
            )

            elite_validation_status["processed"] += 1

        elite_validation_status["message"] = f"Complete! Validated {elite_validation_status['processed']} strategies"

    except Exception as e:
        import traceback
        traceback.print_exc()
        elite_validation_status["message"] = f"Error: {str(e)}"

    finally:
        elite_validation_status["running"] = False
        elite_validation_status["paused"] = False
        elite_validation_status["current_strategy_id"] = None


# =============================================================================
# AUTONOMOUS OPTIMIZER - Cycles through all combinations automatically
# =============================================================================

def build_optimization_combinations():
    """
    Build priority-ordered list of all optimization combinations.

    Priority order: Granularity -> Timeframe -> Period -> Pairs

    This means:
    1. All pairs at 15m/0.5% (1 month period)
    2. All pairs at 5m/0.5%
    3. All pairs at 30m/0.5%
    4. ... continue through all timeframes at 0.5%
    5. Then all timeframes at 0.2%
    6. Then all timeframes at 0.1% (exhaustive)
    """
    config = AUTONOMOUS_CONFIG
    combinations = []

    for granularity in config["granularities"]:
        for timeframe in config["timeframes"]:
            for period in config["periods"]:
                for source in config["sources"]:
                    pairs = config["pairs"].get(source, [])
                    for pair in pairs:
                        combinations.append({
                            "source": source,
                            "pair": pair,
                            "period": period,
                            "timeframe": timeframe,
                            "granularity": granularity,
                        })

    return combinations


def validate_data_range(df: pd.DataFrame, period: dict, timeframe: dict) -> dict:
    """
    Validate that the fetched data covers the expected date range.

    Args:
        df: DataFrame with 'time' column
        period: Period config dict with 'label' and 'months' keys
        timeframe: Timeframe config dict with 'label' and 'minutes' keys

    Returns:
        dict with validation results:
        - valid: bool - True if data matches expected range
        - expected_days: float - Expected days of data
        - actual_days: float - Actual days of data
        - coverage_pct: float - Percentage of expected data received
        - start_date: str - Actual data start date
        - end_date: str - Actual data end date
        - message: str - Human readable message
    """
    try:
        if df is None or len(df) == 0:
            return {
                "valid": False,
                "expected_days": 0,
                "actual_days": 0,
                "coverage_pct": 0,
                "start_date": None,
                "end_date": None,
                "candles": 0,
                "message": "No data available"
            }

        # Calculate expected days from period
        expected_days = period["months"] * 30  # Approximate

        # Get actual date range from data
        if 'time' in df.columns:
            start_time = pd.to_datetime(df['time'].min())
            end_time = pd.to_datetime(df['time'].max())
        else:
            # Try index if time column not present
            start_time = pd.to_datetime(df.index.min())
            end_time = pd.to_datetime(df.index.max())

        actual_days = (end_time - start_time).total_seconds() / 86400  # Days
        coverage_pct = (actual_days / expected_days) * 100 if expected_days > 0 else 0

        # Calculate expected candles
        candles_per_day = 1440 / timeframe["minutes"]  # Minutes per day / timeframe
        expected_candles = int(expected_days * candles_per_day)
        actual_candles = len(df)

        # Validation: Allow 10% tolerance
        is_valid = coverage_pct >= 90

        # Build message
        if is_valid:
            message = f"✓ {actual_days:.1f} days ({actual_candles:,} candles)"
        else:
            message = f"⚠ Expected ~{expected_days:.0f} days, got {actual_days:.1f} days ({coverage_pct:.0f}% coverage)"

        return {
            "valid": is_valid,
            "expected_days": round(expected_days, 1),
            "actual_days": round(actual_days, 1),
            "coverage_pct": round(coverage_pct, 1),
            "expected_candles": expected_candles,
            "actual_candles": actual_candles,
            "start_date": start_time.strftime("%Y-%m-%d %H:%M"),
            "end_date": end_time.strftime("%Y-%m-%d %H:%M"),
            "message": message
        }
    except Exception as e:
        return {
            "valid": False,
            "expected_days": 0,
            "actual_days": 0,
            "coverage_pct": 0,
            "start_date": None,
            "end_date": None,
            "candles": len(df) if df is not None else 0,
            "message": f"Validation error: {str(e)}"
        }


def run_autonomous_optimization_sync(df, combo, status):
    """
    Synchronous optimization that runs in thread pool.
    Reuses existing run_strategy_finder logic.
    """
    try:
        config = AUTONOMOUS_CONFIG

        # All pairs are USDT - source currency is always USD
        source_currency = "USD"

        # Run strategy finder
        report = run_strategy_finder(
            df=df,
            status=status,
            streaming_callback=None,  # No streaming for autonomous
            symbol=combo["pair"],
            timeframe=combo["timeframe"]["label"],
            exchange="BINANCE",  # Always Binance - match on TradingView
            capital=config["capital"],
            position_size_pct=config["position_size_pct"],
            engine="tradingview",  # Use TV engine for consistency
            n_trials=combo["granularity"]["n_trials"],
            progress_min=30,
            progress_max=95,
            source_currency=source_currency,
            fx_fetcher=None  # Not needed for USD
        )

        status["report"] = report

    except Exception as e:
        import traceback
        traceback.print_exc()
        status["message"] = f"Error: {str(e)}"


async def run_autonomous_optimization(combo: dict) -> str:
    """
    Run a single optimization for the given combination.
    Fetches data, runs optimizer, stores results.

    Returns: "completed", "skipped", or "error"
    """
    global autonomous_optimizer_status

    source = combo["source"]
    pair = combo["pair"]
    period = combo["period"]
    timeframe = combo["timeframe"]
    granularity = combo["granularity"]

    # Step 1: Fetch data from Binance via CCXT
    # Single source: Binance USDT pairs - use BINANCE:SYMBOL on TradingView
    from data_fetcher import BinanceDataFetcher

    autonomous_optimizer_status["message"] = f"Fetching {pair} from Binance..."
    autonomous_optimizer_status["progress"] = 5
    autonomous_optimizer_status["trial_current"] = 0
    autonomous_optimizer_status["trial_total"] = granularity["n_trials"]

    fetcher = BinanceDataFetcher()

    try:
        df = await fetcher.fetch_ohlcv(
            pair=pair,
            interval=timeframe["minutes"],
            months=period["months"]
        )
    except Exception as e:
        autonomous_optimizer_status["message"] = f"Data fetch error: {str(e)}"
        autonomous_optimizer_status["data_validation"] = None
        return "error"

    if df is None or len(df) < 100:
        autonomous_optimizer_status["message"] = f"Insufficient data for {pair} ({len(df) if df is not None else 0} candles)"
        autonomous_optimizer_status["data_validation"] = None
        return "error"

    # Step 1.5: Validate data date range
    autonomous_optimizer_status["progress"] = 10
    data_validation = validate_data_range(df, period, timeframe)
    autonomous_optimizer_status["data_validation"] = data_validation

    if not data_validation["valid"]:
        # SKIP this optimization - data is insufficient
        skip_reason = data_validation['message']
        skip_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "pair": pair,
            "period": period["label"],
            "timeframe": timeframe["label"],
            "granularity": granularity["label"],
            "reason": skip_reason,
            "expected_days": data_validation.get("expected_days", 0),
            "actual_days": data_validation.get("actual_days", 0),
            "coverage_pct": data_validation.get("coverage_pct", 0),
        }

        # Add to skipped log (keep last 100)
        autonomous_optimizer_status["skipped_validations"].insert(0, skip_entry)
        if len(autonomous_optimizer_status["skipped_validations"]) > 100:
            autonomous_optimizer_status["skipped_validations"] = autonomous_optimizer_status["skipped_validations"][:100]

        autonomous_optimizer_status["skipped_count"] += 1
        autonomous_optimizer_status["message"] = f"⚠ SKIPPED {pair} {period['label']} - {skip_reason}"

        print(f"[Autonomous] SKIPPED {pair} {period['label']} {timeframe['label']}: {skip_reason}")
        return "skipped"  # Skip to next combination

    # Step 2: Run optimization
    n_trials = granularity["n_trials"]
    autonomous_optimizer_status["message"] = f"Optimizing {pair} {timeframe['label']} - Trial 0/{n_trials}..."
    autonomous_optimizer_status["progress"] = 15

    # Create a temporary status dict for the optimizer
    temp_status = {
        "running": True,
        "progress": 0,
        "message": "",
        "report": None
    }

    # Background task to update progress from temp_status
    async def update_progress():
        while temp_status["running"]:
            # Map temp_status progress (0-100) to our range (15-95)
            inner_progress = temp_status.get("progress", 0)
            mapped_progress = 15 + int(inner_progress * 0.8)  # 15 to 95
            autonomous_optimizer_status["progress"] = mapped_progress

            # Extract progress info from message if available
            # Format: "[1/5] RSI_14 LONG | 50,000/400,000 (45.2%) | Found: 3"
            msg = temp_status.get("message", "")
            import re
            # Match pattern like "50/400" or "50,000/400,000" (with optional commas)
            match = re.search(r'\|\s*([\d,]+)\s*/\s*([\d,]+)\s*\(', msg)
            if match:
                current_trial = int(match.group(1).replace(',', ''))
                total_trials = int(match.group(2).replace(',', ''))
                autonomous_optimizer_status["trial_current"] = current_trial
                autonomous_optimizer_status["trial_total"] = total_trials
                autonomous_optimizer_status["message"] = f"Optimizing {pair} {timeframe['label']} - {current_trial:,}/{total_trials:,}..."
            elif msg:
                autonomous_optimizer_status["message"] = f"Optimizing {pair} {timeframe['label']} ({granularity['label']})..."

            await asyncio.sleep(0.5)  # Update every 500ms

    # Run optimization and progress updater concurrently
    loop = asyncio.get_event_loop()
    progress_task = asyncio.create_task(update_progress())

    try:
        await loop.run_in_executor(
            thread_pool,
            run_autonomous_optimization_sync,
            df,
            combo,
            temp_status
        )
    finally:
        temp_status["running"] = False
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    # Step 3: Process results
    if temp_status.get("report"):
        report = temp_status["report"]
        top_strategies = report.get("top_10", [])

        if top_strategies:
            best = top_strategies[0]
            autonomous_optimizer_status["last_result"] = {
                "pair": pair,
                "timeframe": timeframe["label"],
                "period": period["label"],
                "granularity": granularity["label"],
                "strategy": best.get("strategy_name", "Unknown"),
                "pnl": best.get("metrics", {}).get("total_pnl", 0),
                "win_rate": best.get("metrics", {}).get("win_rate", 0),
            }

            # Update best strategy if this is better
            current_best = autonomous_optimizer_status["best_strategy_found"]
            best_pnl = best.get("metrics", {}).get("total_pnl", 0)
            if current_best is None or best_pnl > current_best.get("pnl", 0):
                autonomous_optimizer_status["best_strategy_found"] = autonomous_optimizer_status["last_result"].copy()

    autonomous_optimizer_status["progress"] = 100
    autonomous_optimizer_status["message"] = f"Completed {pair} {timeframe['label']}"

    # Record to history
    global autonomous_runs_history
    strategies_found = len(temp_status.get("report", {}).get("top_10", [])) if temp_status.get("report") else 0
    best_pnl = temp_status.get("report", {}).get("top_10", [{}])[0].get("metrics", {}).get("total_pnl", 0) if strategies_found > 0 else 0

    history_entry = {
        "completed_at": datetime.now().isoformat(),
        "source": source,
        "pair": pair,
        "period": period["label"],
        "timeframe": timeframe["label"],
        "granularity": granularity["label"],
        "strategies_found": strategies_found,
        "best_pnl": best_pnl,
        "status": "success" if strategies_found > 0 else "no_results"
    }
    autonomous_runs_history.insert(0, history_entry)  # Most recent first

    # Trim history if too large
    if len(autonomous_runs_history) > MAX_HISTORY_SIZE:
        autonomous_runs_history = autonomous_runs_history[:MAX_HISTORY_SIZE]

    return "completed"


async def wait_for_elite_validation():
    """
    Wait for Elite validation to process strategies from the last optimization.
    This ensures newly generated strategies get validated before we start
    the next optimization run.

    Strategy: Validate a BATCH of strategies (up to 10) from the most recent
    optimization before continuing. This prevents the queue from growing forever.
    """
    global elite_validation_status, autonomous_optimizer_status

    db = get_strategy_db()

    # Get pending strategies
    strategies = db.get_all_strategies()
    pending = [s for s in strategies if s.get('elite_status') in [None, 'pending']]

    if not pending:
        print("[Autonomous Optimizer] No pending strategies, continuing...")
        await asyncio.sleep(2)
        return

    initial_pending = len(pending)

    # Validate a batch of strategies (up to 10) before continuing
    # This ensures Elite validation makes progress between optimizations
    batch_size = min(10, initial_pending)
    target_remaining = initial_pending - batch_size

    autonomous_optimizer_status["message"] = f"Elite validation: {initial_pending} pending, validating {batch_size}..."
    print(f"[Autonomous Optimizer] Waiting for Elite to validate {batch_size} strategies ({initial_pending} pending)...")

    # Wait for Elite validation to process the batch (with timeout)
    max_wait = 600  # 10 minute max wait for batch
    check_interval = 3
    elapsed = 0

    while elapsed < max_wait:
        # Let Elite validation run
        await asyncio.sleep(check_interval)
        elapsed += check_interval

        # Update status message
        if elite_validation_status["running"]:
            autonomous_optimizer_status["message"] = f"Elite: {elite_validation_status['message']}"

        # Check progress
        strategies = db.get_all_strategies()
        current_pending = len([s for s in strategies if s.get('elite_status') in [None, 'pending']])
        validated_count = initial_pending - current_pending

        if validated_count >= batch_size:
            print(f"[Autonomous Optimizer] Elite validated {validated_count} strategies, continuing...")
            break

        if current_pending == 0:
            print(f"[Autonomous Optimizer] All strategies validated!")
            break

        # Progress update every 30 seconds
        if elapsed % 30 == 0:
            print(f"[Autonomous Optimizer] Elite progress: {validated_count}/{batch_size} validated, {current_pending} pending")

    if elapsed >= max_wait:
        print(f"[Autonomous Optimizer] Timeout after {max_wait}s, continuing anyway...")

    autonomous_optimizer_status["message"] = "Resuming optimization..."
    await asyncio.sleep(2)


async def start_autonomous_optimizer():
    """
    Continuous background loop that automatically optimizes strategies
    across all source/pair/period/timeframe/granularity combinations.

    Runs when manual optimizer is idle, pauses when optimizer is active.
    Uses priority ordering: starts with 1 month, 15min, 0.5% granularity.
    """
    global autonomous_optimizer_status, unified_status

    if autonomous_optimizer_status["auto_running"]:
        return  # Already running

    autonomous_optimizer_status["auto_running"] = True
    print("[Autonomous Optimizer] Starting continuous background optimization...")

    # Build the full combination list with priority ordering
    combinations = build_optimization_combinations()
    autonomous_optimizer_status["total_combinations"] = len(combinations)
    autonomous_optimizer_status["cycle_index"] = 0

    print(f"[Autonomous Optimizer] Built {len(combinations)} combinations to process")

    while autonomous_optimizer_status["auto_running"] and autonomous_optimizer_status["enabled"]:
        try:
            # Wait for manual optimizer to be idle
            while unified_status["running"]:
                autonomous_optimizer_status["paused"] = True
                autonomous_optimizer_status["message"] = "Paused - waiting for manual optimizer..."
                await asyncio.sleep(5)

            autonomous_optimizer_status["paused"] = False

            # Get current combination
            if autonomous_optimizer_status["cycle_index"] >= len(combinations):
                # Completed full cycle, restart
                autonomous_optimizer_status["cycle_index"] = 0
                autonomous_optimizer_status["message"] = "Completed full cycle, restarting..."
                print("[Autonomous Optimizer] Completed full cycle, restarting from beginning")
                await asyncio.sleep(60)  # Brief pause before restarting
                continue

            combo = combinations[autonomous_optimizer_status["cycle_index"]]

            # Update status
            autonomous_optimizer_status["current_source"] = combo["source"]
            autonomous_optimizer_status["current_pair"] = combo["pair"]
            autonomous_optimizer_status["current_period"] = combo["period"]["label"]
            autonomous_optimizer_status["current_timeframe"] = combo["timeframe"]["label"]
            autonomous_optimizer_status["current_granularity"] = combo["granularity"]["label"]
            autonomous_optimizer_status["running"] = True

            # Set unified_status to signal we're running (for Elite validation interleaving)
            unified_status["running"] = True
            autonomous_optimizer_status["message"] = f"Optimizing {combo['pair']} {combo['timeframe']['label']} ({combo['period']['label']})..."

            print(f"[Autonomous Optimizer] {autonomous_optimizer_status['cycle_index']+1}/{len(combinations)}: "
                  f"{combo['pair']} {combo['timeframe']['label']} {combo['period']['label']} {combo['granularity']['label']}")

            # Run the optimization
            try:
                result = await run_autonomous_optimization(combo)
                if result == "completed":
                    autonomous_optimizer_status["completed_count"] += 1
                    autonomous_optimizer_status["last_completed_at"] = datetime.now().isoformat()
                    print(f"[Autonomous Optimizer] Completed {combo['pair']} - waiting for Elite validation...")
                elif result == "skipped":
                    # Already logged and counted in run_autonomous_optimization
                    print(f"[Autonomous Optimizer] Skipped {combo['pair']} - moving to next combination...")
                elif result == "error":
                    autonomous_optimizer_status["error_count"] += 1
            except Exception as e:
                print(f"[Autonomous Optimizer] Error: {e}")
                autonomous_optimizer_status["error_count"] += 1
                autonomous_optimizer_status["message"] = f"Error: {str(e)}"

            # Move to next combination
            autonomous_optimizer_status["cycle_index"] += 1
            autonomous_optimizer_status["running"] = False
            unified_status["running"] = False  # Allow Elite validation to run

            # Wait for Elite validation to process any pending strategies
            if result == "completed":
                await wait_for_elite_validation()
            else:
                # For skipped/error, just a brief pause
                await asyncio.sleep(2)

        except Exception as e:
            print(f"[Autonomous Optimizer] Loop error: {e}")
            autonomous_optimizer_status["message"] = f"Error: {str(e)}"
            unified_status["running"] = False
            await asyncio.sleep(30)  # Wait before retrying

    autonomous_optimizer_status["auto_running"] = False
    autonomous_optimizer_status["running"] = False
    unified_status["running"] = False
    print("[Autonomous Optimizer] Stopped")


# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

