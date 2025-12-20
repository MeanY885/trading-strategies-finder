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
from pinescript_generator import PineScriptGenerator
from strategy_engine import run_strategy_finder, generate_pinescript, StrategyEngine
from strategy_database import get_strategy_db
HAS_DATABASE = True

# Initialize FastAPI
app = FastAPI(title="BTCGBP ML Optimizer", version="1.0.0")

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
    "stats": None
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
        data_status["stats"] = calculate_data_stats(df)
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
    engine: str = "all"  # "all" to compare all engines, or specific: tradingview, pandas_ta, mihakralj
    date_range: DateRange = None  # Optional date range for Pine Script generation


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
        engine: Calculation engine - "tradingview", "pandas_ta", or "mihakralj"

    Returns profitable strategies ranked by P&L.
    """
    global unified_status

    # Validate engine parameter
    valid_engines = ["tradingview", "pandas_ta", "mihakralj", "all"]
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

        # Determine which engines to run
        if engine == "all":
            engines_to_run = ["tradingview", "pandas_ta", "mihakralj"]
        else:
            engines_to_run = [engine]

        all_reports = {}
        total_engines = len(engines_to_run)

        for idx, eng in enumerate(engines_to_run):
            engine_label = eng.upper()
            # Short tags for display: TV, PT, MH
            engine_tag = {"tradingview": "TV", "pandas_ta": "PT", "mihakralj": "MH"}.get(eng, eng[:2].upper())

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
                progress_min=progress_min,
                progress_max=progress_max
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
                    if engine_best_pnl > best_pnl:
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
async def get_unified_pinescript(rank: int = 1, engine: str = "tradingview"):
    """
    Generate Pine Script for any of the Top 10 unified optimization results.

    Args:
        rank: Strategy rank (1-10)
        engine: Calculation engine - "tradingview" (default), "pandas_ta", or "mihakralj"

    Returns:
        Pine Script code for the specified strategy
    """
    if unified_status["report"] is None:
        raise HTTPException(status_code=404, detail="No optimization results. Run unified optimization first.")

    top_10 = unified_status["report"].get("top_10", [])

    if rank < 1 or rank > len(top_10):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_10)}")

    # Validate engine parameter
    valid_engines = ["tradingview", "pandas_ta", "mihakralj"]
    if engine not in valid_engines:
        raise HTTPException(status_code=400, detail=f"Invalid engine. Must be one of: {valid_engines}")

    strategy_data = top_10[rank - 1]
    strategy_name = strategy_data["strategy_name"]
    params = strategy_data["params"]
    metrics = strategy_data["metrics"]
    entry_rule = strategy_data.get("entry_rule")
    direction = strategy_data.get("direction")

    # Get trading parameters from report (passed from UI)
    position_size_pct = unified_status["report"].get("position_size_pct", 75.0)
    capital = unified_status["report"].get("capital", 1000.0)
    date_range = unified_status["report"].get("date_range")

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

    # Build CSV content (TradingView-style format)
    csv_lines = [
        "Trade #,Type,Entry Time,Exit Time,Entry Price,Exit Price,Position Size (GBP),Position Size (qty),Net P&L GBP,Net P&L %,Run-up GBP,Run-up %,Drawdown GBP,Drawdown %,Cumulative P&L GBP,Result"
    ]

    for trade in trades_list:
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
        engine: Calculation engine - "tradingview" (default), "pandas_ta", or "mihakralj"
    """
    if unified_status["report"] is None:
        raise HTTPException(status_code=404, detail="No optimization results. Run unified optimization first.")

    report = unified_status["report"]
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


# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

