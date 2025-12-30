"""
DATA ROUTES
===========
API endpoints for data fetching and management.
Extracted from main.py for better modularity.
"""
import io
import re
import aiohttp
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd

from config import DATA_DIR
from logging_config import log
from state import app_state
from services.websocket_manager import broadcast_data_status

router = APIRouter(prefix="/api", tags=["data"])


# =============================================================================
# REQUEST MODELS
# =============================================================================

class DataFetchRequest(BaseModel):
    source: str = "binance"  # "binance" or "yfinance"
    pair: str = "BTCUSDT"    # e.g., BTCUSDT (Binance) or BTC-GBP (yfinance)
    interval: int = 15       # Candle interval in minutes
    months: float = 3        # Historical period


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_data_stats(df: pd.DataFrame) -> dict:
    """Calculate statistics about the loaded data."""
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


def update_data_status_from_df(df: pd.DataFrame, message: str = "Data loaded") -> None:
    """Update global data status from a dataframe."""
    stats = calculate_data_stats(df)

    app_state.update_data_status(
        loaded=True,
        rows=len(df),
        start_date=str(df['time'].min()),
        end_date=str(df['time'].max()),
        message=message,
        stats=stats,
        progress=100,
        fetching=False
    )
    app_state.set_dataframe(df)
    broadcast_data_status(app_state.get_data_status())


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/data-status")
async def get_data_status():
    """Get current data loading status."""
    return app_state.get_data_status()


@router.post("/fetch-data")
async def fetch_data(request: DataFetchRequest):
    """
    Fetch historical OHLCV data from exchange.

    Supports:
    - Binance (via CCXT) - USDT pairs
    - Yahoo Finance - Traditional pairs like BTC-GBP
    """
    from data_fetcher import BinanceDataFetcher, YFinanceDataFetcher

    app_state.update_data_status(
        fetching=True,
        progress=0,
        message=f"Fetching {request.pair} from {request.source}..."
    )
    broadcast_data_status(app_state.get_data_status())

    try:
        # Select fetcher based on source
        if request.source.lower() == "binance":
            fetcher = BinanceDataFetcher()
        else:
            fetcher = YFinanceDataFetcher()

        # Fetch data
        app_state.update_data_status(progress=20, message="Connecting to exchange...")
        broadcast_data_status(app_state.get_data_status())

        df = await fetcher.fetch_ohlcv(
            pair=request.pair,
            interval=request.interval,
            months=request.months
        )

        if df is None or len(df) < 10:
            app_state.update_data_status(
                fetching=False,
                progress=0,
                message=f"Failed to fetch data for {request.pair}"
            )
            broadcast_data_status(app_state.get_data_status())
            raise HTTPException(status_code=400, detail="Insufficient data returned")

        # Update status
        update_data_status_from_df(df, f"Loaded {len(df)} candles from {request.source}")

        return {
            "success": True,
            "rows": len(df),
            "start_date": str(df['time'].min()),
            "end_date": str(df['time'].max()),
            "message": f"Loaded {len(df)} {request.interval}m candles"
        }

    except Exception as e:
        app_state.update_data_status(
            fetching=False,
            progress=0,
            message=f"Error: {str(e)}"
        )
        broadcast_data_status(app_state.get_data_status())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with OHLCV data.

    Expected columns: time, open, high, low, close, volume
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Read the file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validate columns
        required_cols = ['time', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Try common alternative column names
            col_mapping = {
                'timestamp': 'time',
                'date': 'time',
                'datetime': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            for old_col, new_col in col_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})

            # Check again
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required columns: {missing_cols}"
                )

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'volume' not in df.columns:
            df['volume'] = 0

        # Parse time column
        df['time'] = pd.to_datetime(df['time'])

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Drop rows with NaN in price columns
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data after cleaning")

        # Save to data directory
        output_path = DATA_DIR / file.filename
        df.to_csv(output_path, index=False)

        # Update status
        update_data_status_from_df(df, f"Uploaded {file.filename}")

        return {
            "success": True,
            "rows": len(df),
            "start_date": str(df['time'].min()),
            "end_date": str(df['time'].max()),
            "saved_to": str(output_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-data")
async def get_available_data():
    """List available data files in the data directory."""
    try:
        files = []
        for path in DATA_DIR.glob("*.csv"):
            try:
                # Quick read to get row count and date range
                df = pd.read_csv(path, nrows=1)
                df_full = pd.read_csv(path)
                files.append({
                    "filename": path.name,
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "rows": len(df_full),
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                })
            except Exception:
                files.append({
                    "filename": path.name,
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "rows": None,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                })

        return {"files": sorted(files, key=lambda x: x['modified'], reverse=True)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-file/{filename}")
async def load_data_file(filename: str):
    """Load a specific data file from the data directory."""
    import os

    # Sanitize filename - remove any path components to prevent directory traversal
    safe_filename = os.path.basename(filename)
    file_path = DATA_DIR / safe_filename

    # Verify the resolved path is still within DATA_DIR
    if not file_path.resolve().is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    if not safe_filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        df = pd.read_csv(file_path)

        # Ensure time column is datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

        # Update status
        update_data_status_from_df(df, f"Loaded {filename}")

        return {
            "success": True,
            "rows": len(df),
            "start_date": str(df['time'].min()),
            "end_date": str(df['time'].max())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-existing-data")
def load_existing_data():
    """Load existing CSV data - finds most recent file."""
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

    # Calculate stats
    stats = calculate_data_stats(df)

    # Update state
    app_state.update_data_status(
        loaded=True,
        rows=len(df),
        start_date=df['time'].min().isoformat(),
        end_date=df['time'].max().isoformat(),
        message=f"Loaded {len(df)} {pair} candles ({days} days)",
        stats=stats,
        progress=100,
        fetching=False
    )
    app_state.set_dataframe(df)
    broadcast_data_status(app_state.get_data_status())

    return app_state.get_data_status()


@router.get("/exchange-rate")
async def get_exchange_rate():
    """Get current USD/GBP exchange rate from Frankfurter API."""
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
        log(f"[Data] Exchange rate fetch error: {e}", level='ERROR')
        return {"usd_to_gbp": 0.79, "source": "fallback", "error": str(e)}
