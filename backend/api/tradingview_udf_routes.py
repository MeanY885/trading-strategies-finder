"""
TRADINGVIEW UDF (Universal Data Feed) ROUTES
============================================
Backend endpoints for TradingView Charting Library integration.
Provides OHLCV data and strategy markers from Binance via CCXT.

UDF Protocol Reference:
https://www.tradingview.com/charting-library-docs/latest/connecting_data/UDF

Endpoints:
- /api/udf/config - Library configuration
- /api/udf/time - Server time
- /api/udf/symbols - Symbol info
- /api/udf/search - Symbol search
- /api/udf/history - OHLCV data
- /api/udf/marks - Trade markers
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from logging_config import log
from state import app_state

router = APIRouter(prefix="/api/udf", tags=["tradingview-udf"])


# =============================================================================
# SUPPORTED SYMBOLS (from BinanceDataFetcher.CORE_PAIRS)
# =============================================================================

SUPPORTED_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'LTCUSDT', 'ATOMUSDT', 'UNIUSDT', 'NEARUSDT',
    'APTUSDT', 'ARBUSDT', 'OPUSDT', 'INJUSDT', 'SUIUSDT',
]

# Resolution mapping (TradingView resolution -> minutes)
RESOLUTION_MAP = {
    "1": 1, "3": 3, "5": 5, "15": 15, "30": 30,
    "60": 60, "120": 120, "240": 240, "360": 360,
    "480": 480, "720": 720, "D": 1440, "1D": 1440,
    "W": 10080, "1W": 10080,
}

SUPPORTED_RESOLUTIONS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W"]


# =============================================================================
# CONFIGURATION ENDPOINT
# =============================================================================

@router.get("/config")
async def get_config():
    """
    TradingView UDF configuration endpoint.

    Returns library capabilities and supported features.
    """
    return {
        "supports_search": True,
        "supports_group_request": False,
        "supports_marks": True,
        "supports_timescale_marks": True,
        "supports_time": True,
        "exchanges": [
            {"value": "BINANCE", "name": "Binance", "desc": "Binance Exchange"}
        ],
        "symbols_types": [
            {"name": "crypto", "value": "crypto"}
        ],
        "supported_resolutions": SUPPORTED_RESOLUTIONS,
        "currency_codes": ["USD", "USDT", "GBP"],
    }


# =============================================================================
# SERVER TIME ENDPOINT
# =============================================================================

@router.get("/time")
async def get_time():
    """
    Server time endpoint.

    Returns current Unix timestamp in seconds.
    """
    return int(time.time())


# =============================================================================
# SYMBOL INFO ENDPOINT
# =============================================================================

@router.get("/symbols")
async def get_symbol_info(symbol: str):
    """
    Symbol information endpoint.

    Returns detailed information about a trading symbol.

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
    """
    # Normalize symbol
    symbol = symbol.upper().replace("/", "").replace("-", "")

    # Determine price scale based on typical price ranges
    if symbol.startswith("BTC"):
        pricescale = 100  # $0.01 precision for BTC
    elif symbol.startswith("ETH"):
        pricescale = 100  # $0.01 precision for ETH
    elif symbol in ["DOGEUSDT", "XRPUSDT", "ADAUSDT", "MATICUSDT"]:
        pricescale = 100000  # $0.00001 precision for low-price coins
    else:
        pricescale = 1000  # $0.001 precision for mid-range coins

    return {
        "name": symbol,
        "ticker": symbol,
        "description": f"{symbol} on Binance",
        "type": "crypto",
        "session": "24x7",
        "exchange": "BINANCE",
        "listed_exchange": "BINANCE",
        "timezone": "Etc/UTC",
        "minmov": 1,
        "pricescale": pricescale,
        "has_intraday": True,
        "has_daily": True,
        "has_weekly_and_monthly": True,
        "supported_resolutions": SUPPORTED_RESOLUTIONS,
        "volume_precision": 8,
        "data_status": "streaming",
        "currency_code": "USDT",
    }


# =============================================================================
# SYMBOL SEARCH ENDPOINT
# =============================================================================

@router.get("/search")
async def search_symbols(
    query: str = "",
    type: str = "",
    exchange: str = "",
    limit: int = 30
):
    """
    Symbol search endpoint.

    Returns matching symbols based on search query.

    Args:
        query: Search string
        type: Symbol type filter (optional)
        exchange: Exchange filter (optional)
        limit: Maximum results to return
    """
    query = query.upper()

    # Filter symbols matching the query
    matches = [s for s in SUPPORTED_SYMBOLS if query in s][:limit]

    return [
        {
            "symbol": s,
            "full_name": f"BINANCE:{s}",
            "description": f"{s} on Binance",
            "exchange": "BINANCE",
            "type": "crypto",
        }
        for s in matches
    ]


# =============================================================================
# HISTORY ENDPOINT (OHLCV DATA)
# =============================================================================

@router.get("/history")
async def get_history(
    symbol: str,
    resolution: str,
    from_ts: int = Query(..., alias="from"),
    to_ts: int = Query(..., alias="to"),
    countback: Optional[int] = None
):
    """
    OHLCV history endpoint - fetches from Binance via CCXT.

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        resolution: Candle resolution (1, 5, 15, 30, 60, 240, D, W)
        from_ts: Start Unix timestamp (seconds)
        to_ts: End Unix timestamp (seconds)
        countback: Optional number of bars to return

    Returns:
        UDF-format OHLCV data with status
    """
    from data_fetcher import BinanceDataFetcher

    # Normalize symbol
    symbol = symbol.upper().replace("/", "").replace("-", "")

    # Convert resolution to minutes
    timeframe_minutes = RESOLUTION_MAP.get(resolution, 60)

    log(f"[UDF] History request: {symbol} {resolution} from={from_ts} to={to_ts}")

    try:
        # Calculate months needed from timestamp range
        from_dt = datetime.fromtimestamp(from_ts, tz=timezone.utc)
        to_dt = datetime.fromtimestamp(to_ts, tz=timezone.utc)
        days_needed = (to_dt - from_dt).days + 1
        months_needed = max(0.5, days_needed / 30)  # Minimum 0.5 months

        # Cap at reasonable maximum
        months_needed = min(months_needed, 24)  # Max 2 years

        log(f"[UDF] Fetching {months_needed:.1f} months of {symbol} {timeframe_minutes}m data")

        # Create fetcher and get data
        fetcher = BinanceDataFetcher()
        df = await fetcher.fetch_ohlcv(
            pair=symbol,
            interval=timeframe_minutes,
            months=months_needed
        )

        if df is None or len(df) == 0:
            log(f"[UDF] No data returned for {symbol}")
            return {"s": "no_data"}

        # Convert time to Unix timestamps (seconds)
        df['timestamp'] = df['time'].astype('int64') // 10**9

        # Filter by requested time range
        df = df[(df['timestamp'] >= from_ts) & (df['timestamp'] <= to_ts)]

        if len(df) == 0:
            log(f"[UDF] No data in requested range for {symbol}")
            return {"s": "no_data"}

        # Apply countback limit if specified
        if countback is not None and countback > 0:
            df = df.tail(countback)

        log(f"[UDF] Returning {len(df)} candles for {symbol}")

        return {
            "s": "ok",
            "t": df['timestamp'].tolist(),
            "o": df['open'].tolist(),
            "h": df['high'].tolist(),
            "l": df['low'].tolist(),
            "c": df['close'].tolist(),
            "v": df['volume'].tolist(),
        }

    except Exception as e:
        log(f"[UDF] History error for {symbol}: {e}", level='ERROR')
        return {"s": "error", "errmsg": str(e)}


# =============================================================================
# MARKS ENDPOINT (TRADE MARKERS)
# =============================================================================

@router.get("/marks")
async def get_marks(
    symbol: str,
    resolution: str,
    from_ts: int = Query(..., alias="from"),
    to_ts: int = Query(..., alias="to")
):
    """
    Trade markers for strategy visualization.

    Returns entry/exit markers from current strategy results.

    Args:
        symbol: Trading pair symbol
        resolution: Candle resolution
        from_ts: Start Unix timestamp (seconds)
        to_ts: End Unix timestamp (seconds)

    Returns:
        List of mark objects for TradingView
    """
    # Get current strategy results from app state
    unified = app_state.get_unified_status()

    if not unified.get("report"):
        return []

    marks = []
    top_10 = unified["report"].get("top_10", [])

    # Process first strategy's trades (most relevant for visualization)
    if top_10:
        strategy = top_10[0]
        trades_list = strategy.get("trades_list", [])

        for trade in trades_list:
            try:
                # Get entry time
                entry_time = trade.get("entry_time")
                if entry_time:
                    if isinstance(entry_time, str):
                        entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                        entry_ts = int(entry_dt.timestamp())
                    elif isinstance(entry_time, datetime):
                        entry_ts = int(entry_time.timestamp())
                    else:
                        entry_ts = int(entry_time)

                    # Check if within requested range
                    if from_ts <= entry_ts <= to_ts:
                        direction = trade.get("direction", "LONG")
                        trade_num = trade.get("trade_num", 0)

                        # Entry mark
                        marks.append({
                            "id": f"entry_{trade_num}",
                            "time": entry_ts,
                            "color": {"background": "#22c55e" if direction == "LONG" else "#ef4444"},
                            "text": f"{direction} Entry",
                            "label": "E",
                            "labelFontColor": "#ffffff",
                            "minSize": 14,
                        })

                # Get exit time
                exit_time = trade.get("exit_time")
                if exit_time:
                    if isinstance(exit_time, str):
                        exit_dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                        exit_ts = int(exit_dt.timestamp())
                    elif isinstance(exit_time, datetime):
                        exit_ts = int(exit_time.timestamp())
                    else:
                        exit_ts = int(exit_time)

                    # Check if within requested range
                    if from_ts <= exit_ts <= to_ts:
                        pnl = trade.get("pnl", 0)
                        trade_num = trade.get("trade_num", 0)
                        is_win = pnl > 0

                        # Exit mark
                        marks.append({
                            "id": f"exit_{trade_num}",
                            "time": exit_ts,
                            "color": {"background": "#22c55e" if is_win else "#ef4444"},
                            "text": f"{'Win' if is_win else 'Loss'} ${abs(pnl):.2f}",
                            "label": "X",
                            "labelFontColor": "#ffffff",
                            "minSize": 14,
                        })

            except Exception as e:
                log(f"[UDF] Error processing trade mark: {e}", level='WARNING')
                continue

    log(f"[UDF] Returning {len(marks)} marks for {symbol}")
    return marks


# =============================================================================
# TIMESCALE MARKS ENDPOINT
# =============================================================================

@router.get("/timescale_marks")
async def get_timescale_marks(
    symbol: str,
    resolution: str,
    from_ts: int = Query(..., alias="from"),
    to_ts: int = Query(..., alias="to")
):
    """
    Timescale marks for important events (shown on time axis).

    Currently returns empty list - can be extended for significant events.
    """
    return []
