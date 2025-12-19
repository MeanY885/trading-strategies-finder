"""
BTCGBP ML Optimizer - Main FastAPI Application
"""
import os
import json
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path

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

from data_fetcher import BinanceDataFetcher, YFinanceDataFetcher
from strategy import SidewaysScalperStrategy
from optimizer import StrategyOptimizer
from pinescript_generator import PineScriptGenerator
# Unified optimizer that tests ALL 75+ strategies with 3 ML methods
from unified_optimizer import run_unified_optimization
# Strategy database for persistence
try:
    from strategy_database import get_strategy_db
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False
    print("Warning: Strategy database not available")

# Initialize FastAPI
app = FastAPI(title="BTCGBP ML Optimizer", version="1.0.0")

# Global state
optimization_status = {
    "running": False,
    "progress": 0,
    "current_trial": 0,
    "total_trials": 0,
    "best_params": None,
    "best_score": None,
    "message": "Ready",
    "results": []
}

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
    "message": "No data loaded"
}

# Paths
DATA_DIR = Path("/app/data")
OUTPUT_DIR = Path("/app/output")
FRONTEND_DIR = Path("/app/frontend")

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Models
class DataFetchRequest(BaseModel):
    source: str = "yfinance"  # "binance" or "yfinance"
    pair: str = "BTC-GBP"     # e.g., BTCUSDT (Binance) or BTC-GBP (yfinance)
    interval: int = 15        # Candle interval in minutes
    months: float = 3         # Historical period (supports decimals: 0.03=1day, 0.25=1week)

class OptimizationRequest(BaseModel):
    n_trials: int = 100
    timeout: int = 300  # seconds

class ParameterRanges(BaseModel):
    adx_threshold_min: int = 15
    adx_threshold_max: int = 35
    bb_length_min: int = 10
    bb_length_max: int = 30
    bb_mult_min: float = 1.5
    bb_mult_max: float = 3.0
    rsi_oversold_min: int = 20
    rsi_oversold_max: int = 40
    rsi_overbought_min: int = 60
    rsi_overbought_max: int = 80
    sl_fixed_min: float = 50
    sl_fixed_max: float = 300
    tp_ratio_min: float = 1.0
    tp_ratio_max: float = 3.0

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main UI"""
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return {
        "optimization": optimization_status,
        "data": data_status
    }

@app.post("/api/fetch-data")
async def fetch_data(request: DataFetchRequest, background_tasks: BackgroundTasks):
    """Fetch historical data from Binance or Yahoo Finance"""
    global data_status
    
    source_name = "Binance" if request.source == "binance" else "Yahoo Finance"
    data_status["message"] = f"Fetching {request.interval}m {request.pair} from {source_name}..."
    data_status["source"] = request.source
    data_status["pair"] = request.pair
    data_status["interval"] = request.interval
    
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
    
    source_name = "Binance" if source == "binance" else "Yahoo Finance"
    
    def status_callback(message: str, progress: int = None):
        """Update status for frontend polling"""
        data_status["message"] = message.replace("[YFinance] ", "").replace("[Binance] ", "")
        if progress is not None:
            data_status["progress"] = progress
    
    try:
        # Create fetcher with status callback
        if source == "binance":
            fetcher = BinanceDataFetcher()
        else:
            fetcher = YFinanceDataFetcher(status_callback=status_callback)
        
        df = await fetcher.fetch_ohlcv(pair=pair, interval=interval, months=months)
        
        if len(df) == 0:
            # Only set generic error if fetcher didn't already set a specific error
            if not data_status["message"].startswith("Error:"):
                data_status["message"] = f"Error: No data returned for {pair}"
            data_status["loaded"] = False
            return
        
        # Save to CSV
        pair_clean = pair.lower().replace("/", "").replace("-", "")
        csv_path = DATA_DIR / f"{pair_clean}_{interval}m.csv"
        df.to_csv(csv_path, index=False)
        
        days = (df['time'].max() - df['time'].min()).days
        
        data_status["loaded"] = True
        data_status["rows"] = len(df)
        data_status["interval"] = interval
        data_status["pair"] = pair
        data_status["source"] = source
        data_status["start_date"] = df['time'].min().isoformat()
        data_status["end_date"] = df['time'].max().isoformat()
        data_status["message"] = f"✓ {len(df)} candles ({days} days) from {source_name}"
        
    except Exception as e:
        data_status["message"] = f"Error: {str(e)}"
        data_status["loaded"] = False
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


@app.post("/api/optimize")
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start ML optimization"""
    global optimization_status
    
    if optimization_status["running"]:
        raise HTTPException(status_code=400, detail="Optimization already running")
    
    if not data_status["loaded"]:
        raise HTTPException(status_code=400, detail="No data loaded. Please fetch data first.")
    
    optimization_status["running"] = True
    optimization_status["progress"] = 0
    optimization_status["total_trials"] = request.n_trials
    optimization_status["message"] = "Starting optimization..."
    
    background_tasks.add_task(run_optimization_task, request.n_trials, request.timeout)
    
    return {"status": "started", "message": f"Running {request.n_trials} optimization trials"}

def run_optimization_sync(n_trials: int, timeout: int):
    """Synchronous optimization task that runs in thread pool"""
    global optimization_status
    
    try:
        import pandas as pd
        
        # Load data - find most recent data file
        data_files = list(DATA_DIR.glob("*_*m.csv"))
        if not data_files:
            optimization_status["message"] = "Error: No data file found"
            optimization_status["running"] = False
            return
        
        csv_path = max(data_files, key=lambda p: p.stat().st_mtime)
        optimization_status["message"] = f"Loading data from {csv_path.name}..."
        
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        
        optimization_status["message"] = f"Creating optimizer with {len(df)} candles..."
        
        # Create optimizer
        optimizer = StrategyOptimizer(df)
        
        optimization_status["message"] = f"Starting {n_trials} optimization trials..."
        
        # Progress callback - updates global state from worker thread
        def progress_callback(trial_num, total, params, score):
            optimization_status["current_trial"] = trial_num
            optimization_status["progress"] = int((trial_num / total) * 100)
            optimization_status["message"] = f"Trial {trial_num}/{total} - Score: {score:.4f}"
            
            if optimization_status["best_score"] is None or score > optimization_status["best_score"]:
                optimization_status["best_score"] = score
                optimization_status["best_params"] = dict(params) if params else None
        
        # Run optimization
        best_params, best_score, all_results = optimizer.optimize(
            n_trials=n_trials,
            timeout=timeout,
            callback=progress_callback
        )
        
        optimization_status["best_params"] = best_params
        optimization_status["best_score"] = best_score
        optimization_status["results"] = all_results[-20:]  # Keep last 20 results
        optimization_status["message"] = f"Optimization complete! Best score: {best_score:.4f}"
        
        # Generate Pine Script
        generator = PineScriptGenerator()
        pine_script = generator.generate(best_params)
        
        # Save to file
        output_path = OUTPUT_DIR / "optimized_strategy.pine"
        with open(output_path, "w") as f:
            f.write(pine_script)
        
        optimization_status["message"] = f"Complete! Best profit factor: {best_score:.4f}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        optimization_status["message"] = f"Error: {str(e)}"
    finally:
        optimization_status["running"] = False


async def run_optimization_task(n_trials: int, timeout: int):
    """Background task for optimization - runs sync code in thread pool"""
    global optimization_status
    
    # Run the blocking optimization in a thread pool so it doesn't block the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(thread_pool, run_optimization_sync, n_trials, timeout)

@app.get("/api/optimization-results")
async def get_optimization_results():
    """Get detailed optimization results"""
    return {
        "status": optimization_status,
        "best_params": optimization_status.get("best_params"),
        "best_score": optimization_status.get("best_score")
    }


# ============ UNIFIED OPTIMIZATION (Tests ALL 75+ Strategies) ============

class UnifiedRequest(BaseModel):
    capital: float = 1000.0
    risk_percent: float = 2.0
    n_trials: int = 300  # Trials per optimization method


@app.post("/api/run-unified")
async def start_unified_optimization(request: UnifiedRequest, background_tasks: BackgroundTasks):
    """
    Run UNIFIED optimization - tests ALL 75+ strategies with ALL methods.
    
    This is THE optimization endpoint that:
    1. Tests 75+ strategy types (Mean Reversion, Trend, Momentum, DaviddTech, etc.)
    2. Uses 3 ML methods (Random Search, Bayesian TPE, CMA-ES)
    3. Finds consensus strategies that multiple methods agree on
    4. Validates on held-out data
    5. Returns Top 10 most robust strategies
    """
    global unified_status
    
    if unified_status["running"]:
        raise HTTPException(status_code=400, detail="Optimization already running")
    
    if not data_status["loaded"]:
        raise HTTPException(status_code=400, detail="No data loaded. Please load data first.")
    
    unified_status["running"] = True
    unified_status["progress"] = 0
    unified_status["message"] = "Starting unified optimization (testing ALL strategies)..."
    unified_status["report"] = None
    
    background_tasks.add_task(
        run_unified_task, 
        request.capital, 
        request.risk_percent, 
        request.n_trials
    )
    
    return {
        "status": "started", 
        "message": f"Running UNIFIED optimization: £{request.capital} capital, {request.risk_percent}% risk, {request.n_trials} trials/method, 75+ strategies"
    }


def run_unified_sync(capital: float, risk_percent: float, n_trials: int):
    """Synchronous unified optimization that runs in thread pool"""
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

        # Extract metadata from filename (e.g., "kraken_btcgbp_15m.csv")
        filename = csv_path.stem  # e.g., "kraken_btcgbp_15m"
        parts = filename.split("_")
        data_source = parts[0] if len(parts) > 0 else None
        symbol = parts[1].upper() if len(parts) > 1 else None
        timeframe = parts[2] if len(parts) > 2 else None

        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])

        unified_status["message"] = f"Starting UNIFIED optimization on {len(df)} candles..."
        unified_status["progress"] = 2

        # Run unified optimization with streaming callback for real-time results
        report = run_unified_optimization(
            df=df,
            capital=capital,
            risk_percent=risk_percent,
            n_trials=n_trials,
            status=unified_status,
            streaming_callback=publish_strategy_result,
            symbol=symbol,
            timeframe=timeframe,
            data_source=data_source
        )
        
        unified_status["report"] = report
        unified_status["progress"] = 100
        
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


async def run_unified_task(capital: float, risk_percent: float, n_trials: int):
    """Background task for unified optimization - runs sync code in thread pool"""
    global unified_status
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        thread_pool, 
        run_unified_sync, 
        capital, 
        risk_percent, 
        n_trials
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
    print(f"[SSE] Publishing result: {result.get('strategy_name', 'unknown')} score={result.get('composite_score', 0):.4f} to {len(streaming_clients)} clients")
    with streaming_lock:
        if not streaming_clients:
            print("[SSE] WARNING: No SSE clients connected!")
        for q in streaming_clients:
            try:
                q.put_nowait(result)
                print(f"[SSE] Result queued successfully")
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
            streaming_clients.remove(client_queue)


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
async def get_unified_pinescript(rank: int = 1):
    """
    Generate Pine Script for any of the Top 10 unified optimization results.

    Args:
        rank: Strategy rank (1-10)

    Returns:
        Pine Script code for the specified strategy
    """
    if unified_status["report"] is None:
        raise HTTPException(status_code=404, detail="No optimization results. Run unified optimization first.")

    top_10 = unified_status["report"].get("top_10", [])

    if rank < 1 or rank > len(top_10):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_10)}")

    strategy_data = top_10[rank - 1]
    strategy_name = strategy_data["strategy_name"]
    params = strategy_data["params"]
    metrics = strategy_data["metrics"]

    # Use EXACT-MATCH generator for guaranteed TradingView compatibility
    generator = PineScriptGenerator()
    pine_script = generator.generate_exact_match(strategy_name, params, metrics)

    return {
        "rank": rank,
        "strategy_name": strategy_name,
        "strategy_category": strategy_data.get("strategy_category", "unknown"),
        "tp_percent": params.get("tp_percent", 1.0),
        "sl_percent": params.get("sl_percent", 3.0),
        "metrics": metrics,
        "pinescript": pine_script
    }


@app.get("/api/unified-pinescript-all")
async def get_all_unified_pinescripts():
    """
    Generate Pine Scripts for ALL Top 10 strategies.

    Returns:
        List of Pine Script codes for all top strategies
    """
    if unified_status["report"] is None:
        raise HTTPException(status_code=404, detail="No optimization results. Run unified optimization first.")

    top_10 = unified_status["report"].get("top_10", [])
    generator = PineScriptGenerator()

    results = []
    for i, strategy_data in enumerate(top_10, 1):
        strategy_name = strategy_data["strategy_name"]
        params = strategy_data["params"]
        metrics = strategy_data["metrics"]

        # Use EXACT-MATCH generator for guaranteed TradingView compatibility
        pine_script = generator.generate_exact_match(strategy_name, params, metrics)

        results.append({
            "rank": i,
            "strategy_name": strategy_name,
            "strategy_category": strategy_data.get("strategy_category", "unknown"),
            "tp_percent": params.get("tp_percent", 1.0),
            "sl_percent": params.get("sl_percent", 3.0),
            "metrics": metrics,
            "pinescript": pine_script
        })

    return {"strategies": results}


@app.get("/api/download-unified-pinescript/{rank}")
async def download_unified_pinescript(rank: int = 1):
    """Download Pine Script file for a specific ranked strategy"""
    result = await get_unified_pinescript(rank)

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


# =============================================================================
# ML MODEL TRAINING ENDPOINTS
# =============================================================================

# ML Training Status
ml_training_status = {
    "running": False,
    "progress": 0,
    "message": "Ready",
    "current_model": None,
    "models_trained": [],
    "models_failed": [],
    "results": {}
}


class MLTrainingRequest(BaseModel):
    """Request model for ML training"""
    models: List[str] = ["ml_xgboost", "ml_lightgbm", "ml_catboost"]  # Models to train
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    target_horizon: int = 1
    target_threshold: float = 0.001


@app.post("/api/ml-train")
async def start_ml_training(request: MLTrainingRequest, background_tasks: BackgroundTasks):
    """Start ML model training"""
    global ml_training_status

    if not data_status["loaded"]:
        raise HTTPException(status_code=400, detail="No data loaded. Load data first.")

    if ml_training_status["running"]:
        raise HTTPException(status_code=400, detail="ML training already running")

    ml_training_status["running"] = True
    ml_training_status["progress"] = 0
    ml_training_status["message"] = "Starting ML training..."
    ml_training_status["models_trained"] = []
    ml_training_status["models_failed"] = []
    ml_training_status["results"] = {}

    # Run training in background
    background_tasks.add_task(
        run_ml_training_task,
        request.models,
        request.n_estimators,
        request.max_depth,
        request.learning_rate,
        request.target_horizon,
        request.target_threshold
    )

    return {"status": "started", "message": f"Training {len(request.models)} ML models..."}


async def run_ml_training_task(models: List[str], n_estimators: int, max_depth: int,
                                learning_rate: float, target_horizon: int, target_threshold: float):
    """Background task to train ML models"""
    global ml_training_status

    try:
        # Load data
        data_files = list(DATA_DIR.glob("*_*m.csv"))
        if not data_files:
            ml_training_status["running"] = False
            ml_training_status["message"] = "Error: No data file found"
            return

        csv_path = max(data_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])

        # Import and run ML optimizer
        from ml_optimizer import MLOptimizer

        def status_callback(status):
            ml_training_status.update(status)

        optimizer = MLOptimizer(df)
        results = optimizer.train_models(
            models,
            status_callback=status_callback,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            target_horizon=target_horizon,
            target_threshold=target_threshold
        )

        ml_training_status["results"] = results

    except Exception as e:
        import traceback
        traceback.print_exc()
        ml_training_status["message"] = f"Error: {str(e)}"

    finally:
        ml_training_status["running"] = False
        ml_training_status["progress"] = 100


@app.get("/api/ml-status")
async def get_ml_status():
    """Get current ML training status"""
    return ml_training_status


@app.get("/api/ml-models")
async def get_available_ml_models():
    """Get list of available ML models"""
    return {
        "available_models": {
            "ml_xgboost": {"name": "XGBoost Classifier", "installed": True},
            "ml_lightgbm": {"name": "LightGBM Classifier", "installed": True},
            "ml_catboost": {"name": "CatBoost Classifier", "installed": True},
            "ml_lstm": {"name": "LSTM Neural Network", "installed": False, "requires": "torch"},
            "ml_gru": {"name": "GRU Neural Network", "installed": False, "requires": "torch"},
            "ml_transformer": {"name": "Transformer Time-Series", "installed": False, "requires": "torch"},
            "ml_rl_ppo": {"name": "RL Agent (PPO)", "installed": False, "requires": "stable-baselines3"},
            "ml_rl_dqn": {"name": "RL Agent (DQN)", "installed": False, "requires": "stable-baselines3"},
        },
        "trained_models": ml_training_status.get("models_trained", [])
    }


# =============================================================================
# END ML MODEL TRAINING ENDPOINTS
# =============================================================================


@app.get("/api/backtest")
async def run_backtest(
    adx_threshold: int = 25,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    rsi_oversold: int = 35,
    rsi_overbought: int = 65,
    sl_fixed: float = 100,
    tp_ratio: float = 1.5
):
    """Run a single backtest with specified parameters"""
    if not data_status["loaded"]:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    import pandas as pd
    
    # Find most recent data file
    data_files = list(DATA_DIR.glob("btcgbp_*m.csv"))
    if not data_files:
        raise HTTPException(status_code=400, detail="No data file found")
    
    csv_path = max(data_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    strategy = SidewaysScalperStrategy(
        adx_threshold=adx_threshold,
        adx_length=14,
        bb_length=bb_length,
        bb_mult=bb_mult,
        rsi_length=14,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        sl_fixed=sl_fixed,
        tp_ratio=tp_ratio
    )
    
    results = strategy.backtest(df)
    
    return results

@app.get("/api/pinescript")
async def get_pinescript():
    """Get the generated Pine Script"""
    output_path = OUTPUT_DIR / "optimized_strategy.pine"
    
    if not output_path.exists():
        if optimization_status.get("best_params"):
            # Generate on the fly
            generator = PineScriptGenerator()
            return {"pinescript": generator.generate(optimization_status["best_params"])}
        raise HTTPException(status_code=404, detail="No optimized strategy found. Run optimization first.")
    
    with open(output_path, "r") as f:
        content = f.read()
    
    return {"pinescript": content}

@app.get("/api/download-pinescript")
async def download_pinescript():
    """Download the Pine Script file"""
    output_path = OUTPUT_DIR / "optimized_strategy.pine"
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="No optimized strategy found")
    
    return FileResponse(
        output_path,
        media_type="text/plain",
        filename="btcgbp_optimized_scalper.pine"
    )

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


@app.get("/api/db/strategies")
async def get_saved_strategies(
    limit: int = 20,
    symbol: str = None,
    timeframe: str = None,
    min_trades: int = 3,
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
            min_trades=min_trades,
            min_win_rate=min_win_rate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/db/strategies/best-win-rate")
async def get_best_by_win_rate(limit: int = 10, min_trades: int = 5):
    """Get strategies with highest win rate."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_best_by_win_rate(limit=limit, min_trades=min_trades)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/db/strategies/best-profit-factor")
async def get_best_by_profit_factor(limit: int = 10, min_trades: int = 5):
    """Get strategies with highest profit factor."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_best_by_profit_factor(limit=limit, min_trades=min_trades)
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

        generator = PineScriptGenerator()
        pinescript = generator.generate_exact_match(
            strategy['strategy_name'],
            strategy['params'],
            {
                'total_trades': strategy['total_trades'],
                'win_rate': strategy['win_rate'],
                'total_pnl': strategy['total_pnl'],
                'profit_factor': strategy['profit_factor'],
                'max_drawdown': strategy['max_drawdown'],
            }
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


# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

