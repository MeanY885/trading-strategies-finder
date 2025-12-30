"""
OPTIMIZATION ROUTES
==================
API endpoints for manual optimization runs.
"""
import asyncio
import concurrent.futures
import io
import csv
from typing import Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict

from config import OUTPUT_DIR
from logging_config import log
from state import app_state, concurrency_config
from services.websocket_manager import broadcast_optimization_status, broadcast_strategy_result
from utils.converters import dict_to_strategy_result

router = APIRouter(prefix="/api", tags=["optimization"])

# Thread pool for optimization
max_workers = concurrency_config.get("cpu_cores", 4)
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)


# =============================================================================
# REQUEST MODELS
# =============================================================================

class DateRangeConfig(BaseModel):
    """Date range configuration for backtesting."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    enabled: bool = False
    start_date: Optional[str] = Field(default=None, alias="startDate")
    start_time: Optional[str] = Field(default=None, alias="startTime")
    end_date: Optional[str] = Field(default=None, alias="endDate")
    end_time: Optional[str] = Field(default=None, alias="endTime")
    start_timestamp: Optional[str] = Field(default=None, alias="startTimestamp")
    end_timestamp: Optional[str] = Field(default=None, alias="endTimestamp")


class UnifiedOptimizationRequest(BaseModel):
    """
    Request model for optimization runs.

    Accepts both snake_case and camelCase parameter names for frontend compatibility.
    All parameters have sensible defaults and validation.

    Parameters:
        symbol: Trading pair symbol (e.g., "BTCGBP", "BTCUSDT")
        timeframe: Candlestick timeframe (e.g., "15m", "1h", "4h")
        exchange: Exchange name (e.g., "BINANCE")
        capital: Starting capital for backtesting
        position_size_pct: Position size as percentage of capital (alias: positionSizePct)
        engine: Backtesting engine - 'tradingview', 'native', or 'all'
        n_trials: Number of optimization trials to run (alias: nTrials)
        source_currency: Base currency for display (e.g., "GBP", "USD")
        use_vectorbt: Enable VectorBT for faster backtesting (alias: useVectorbt)
        risk_percent: Risk percentage per trade (alias: riskPercent)
        date_range: Date range configuration for backtesting (alias: dateRange)
    """
    model_config = ConfigDict(
        populate_by_name=True,  # Accept both snake_case and camelCase
        extra="ignore",  # Ignore unknown fields gracefully
    )

    # Core parameters
    symbol: str = Field(default="BTCGBP", description="Trading pair symbol")
    timeframe: str = Field(default="15m", description="Candlestick timeframe")
    exchange: str = Field(default="BINANCE", description="Exchange name")
    capital: float = Field(default=1000.0, ge=0, description="Starting capital for backtesting")

    # Position sizing
    position_size_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        alias="positionSizePct",
        description="Position size as percentage of capital"
    )

    # Engine configuration
    engine: str = Field(
        default="tradingview",
        description="Backtesting engine: 'tradingview', 'native', or 'all'"
    )
    n_trials: int = Field(
        default=400,
        ge=1,
        le=10000,
        alias="nTrials",
        description="Number of optimization trials"
    )

    # Currency settings
    source_currency: str = Field(
        default="GBP",
        alias="sourceCurrency",
        description="Base currency for display"
    )

    # Performance options
    use_vectorbt: bool = Field(
        default=False,
        alias="useVectorbt",
        description="Enable VectorBT for faster backtesting"
    )

    # Risk settings
    risk_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        alias="riskPercent",
        description="Risk percentage per trade (overrides position_size_pct if set)"
    )

    # Date range filtering
    date_range: Optional[Any] = Field(
        default=None,
        alias="dateRange",
        description="Date range configuration for backtesting"
    )

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        """Validate engine is one of the allowed values."""
        valid_engines = {"tradingview", "native", "all"}
        if v.lower() not in valid_engines:
            raise ValueError(
                f"Invalid engine '{v}'. Must be one of: {', '.join(valid_engines)}"
            )
        return v.lower()

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe format."""
        valid_timeframes = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}
        if v.lower() not in valid_timeframes:
            # Allow it anyway but log a warning
            import logging
            logging.warning(f"Unusual timeframe '{v}' - valid options: {', '.join(valid_timeframes)}")
        return v

    @field_validator("date_range", mode="before")
    @classmethod
    def normalize_date_range(cls, v: Any) -> Optional[dict]:
        """Normalize date range to consistent format."""
        if v is None:
            return None
        if isinstance(v, dict):
            # Normalize camelCase keys to snake_case for internal use
            normalized = {
                "enabled": v.get("enabled", False),
                "startDate": v.get("startDate") or v.get("start_date"),
                "startTime": v.get("startTime") or v.get("start_time") or "00:00",
                "endDate": v.get("endDate") or v.get("end_date"),
                "endTime": v.get("endTime") or v.get("end_time") or "23:59",
                "startTimestamp": v.get("startTimestamp") or v.get("start_timestamp"),
                "endTimestamp": v.get("endTimestamp") or v.get("end_timestamp"),
            }
            return normalized
        return v


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_optimization_sync(request: UnifiedOptimizationRequest, status: dict, streaming_callback=None):
    """
    Synchronous optimization wrapper that runs in thread pool.
    """
    from strategy_engine import run_strategy_finder
    import pandas as pd

    try:
        df = app_state.get_dataframe()
        if df is None or len(df) == 0:
            status["message"] = "No data loaded"
            status["running"] = False
            return

        # Apply date range filter if provided
        if request.date_range and request.date_range.get("enabled"):
            try:
                start_str = request.date_range.get("startTimestamp") or request.date_range.get("startDate")
                end_str = request.date_range.get("endTimestamp") or request.date_range.get("endDate")

                if start_str and end_str:
                    start_dt = pd.to_datetime(start_str)
                    end_dt = pd.to_datetime(end_str)

                    # Filter dataframe by date range
                    if 'datetime' in df.columns:
                        df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()
                    elif df.index.name == 'datetime' or isinstance(df.index, pd.DatetimeIndex):
                        df = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()

                    if len(df) == 0:
                        status["message"] = "No data in selected date range"
                        status["running"] = False
                        return
            except Exception as e:
                # Log but continue with full dataframe if date filtering fails
                log(f"[Data] Date range filtering failed, using full dataset: {e}", level='WARNING')

        # If risk_percent is provided, use it as position_size_pct
        position_size = request.risk_percent if request.risk_percent is not None else request.position_size_pct

        report = run_strategy_finder(
            df=df,
            status=status,
            streaming_callback=streaming_callback,
            symbol=request.symbol,
            timeframe=request.timeframe,
            exchange=request.exchange,
            capital=request.capital,
            position_size_pct=position_size,
            engine=request.engine,
            n_trials=request.n_trials,
            source_currency=request.source_currency,
            use_vectorbt=request.use_vectorbt,
        )

        status["report"] = report
        status["running"] = False
        status["progress"] = 100
        status["message"] = "Complete"

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        log(f"[Data] Optimization run failed: {e}\n{error_traceback}", level='ERROR')
        status["running"] = False
        status["message"] = f"Error: {str(e)}"


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/unified-status")
async def get_unified_status():
    """Get current optimization status."""
    return app_state.get_unified_status()


@router.post("/run-unified")
async def run_unified_optimization(request: UnifiedOptimizationRequest, background_tasks: BackgroundTasks):
    """
    Start a manual optimization run.

    This runs the strategy finder with the specified parameters.
    Progress updates are broadcast via WebSocket.
    """
    # Check if already running
    if app_state.is_optimization_running():
        raise HTTPException(status_code=400, detail="Optimization already running")

    # Check if data is loaded
    df = app_state.get_dataframe()
    if df is None or len(df) == 0:
        raise HTTPException(status_code=400, detail="No data loaded. Please load data first.")

    # Initialize status
    app_state.update_unified_status(
        running=True,
        progress=0,
        message="Starting optimization...",
        report=None
    )
    broadcast_optimization_status(app_state.get_unified_status())

    # Create a status dict that will be updated by the thread
    status = {
        "running": True,
        "progress": 0,
        "message": "Starting...",
        "report": None
    }
    app_state.set_current_optimization(status)

    def streaming_callback(result):
        """Callback to broadcast strategy results in real-time."""
        if result.get("type") == "strategy_result":
            broadcast_strategy_result(result)

        # Update global status
        app_state.update_unified_status(
            progress=status.get("progress", 0),
            message=status.get("message", "")
        )
        broadcast_optimization_status(app_state.get_unified_status())

    async def run_in_background():
        """Async wrapper to run optimization and update state."""
        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(
                thread_pool,
                run_optimization_sync,
                request,
                status,
                streaming_callback
            )
        finally:
            # Update global state from thread status
            app_state.update_unified_status(
                running=False,
                progress=100,
                message=status.get("message", "Complete"),
                report=status.get("report")
            )
            app_state.set_current_optimization(None)
            broadcast_optimization_status(app_state.get_unified_status())

    # Start background task
    background_tasks.add_task(run_in_background)

    return {
        "success": True,
        "message": "Optimization started",
        "n_trials": request.n_trials
    }


@router.post("/stop-unified")
async def stop_unified_optimization():
    """Stop the current optimization run."""
    if not app_state.is_optimization_running():
        return {"success": False, "message": "No optimization running"}

    if app_state.signal_abort():
        app_state.update_unified_status(message="Stopping...")
        broadcast_optimization_status(app_state.get_unified_status())
        return {"success": True, "message": "Stop signal sent"}

    return {"success": False, "message": "Could not signal abort"}


@router.get("/unified-report")
async def get_unified_report():
    """Get the last optimization report."""
    status = app_state.get_unified_status()
    if status.get("report"):
        return status["report"]
    return {"error": "No report available"}


@router.get("/unified-pinescript/{rank}")
async def get_unified_pinescript(rank: int = 1):
    """
    Get Pine Script for a specific ranked strategy from the last run.

    Args:
        rank: Strategy rank (1 = best, 2 = second best, etc.)
    """
    from strategy_engine import generate_pinescript

    status = app_state.get_unified_status()
    report = status.get("report")

    if not report:
        raise HTTPException(status_code=404, detail="No optimization report available")

    top_strategies = report.get("top_10", [])
    if not top_strategies:
        raise HTTPException(status_code=404, detail="No strategies in report")

    if rank < 1 or rank > len(top_strategies):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rank. Must be 1-{len(top_strategies)}"
        )

    strategy = top_strategies[rank - 1]

    try:
        # Convert dict to StrategyResult if needed
        result = dict_to_strategy_result(strategy) if isinstance(strategy, dict) else strategy

        pinescript = generate_pinescript(result)

        return {
            "rank": rank,
            "strategy_name": result.strategy_name,
            "pinescript": pinescript
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/unified-pinescript-all")
async def get_all_pinescripts():
    """Get Pine Scripts for all top strategies from the last run."""
    from strategy_engine import generate_pinescript

    status = app_state.get_unified_status()
    report = status.get("report")

    if not report:
        raise HTTPException(status_code=404, detail="No optimization report available")

    top_strategies = report.get("top_10", [])
    if not top_strategies:
        raise HTTPException(status_code=404, detail="No strategies in report")

    results = []
    for i, strategy in enumerate(top_strategies):
        try:
            result = dict_to_strategy_result(strategy) if isinstance(strategy, dict) else strategy

            pinescript = generate_pinescript(result)
            results.append({
                "rank": i + 1,
                "strategy_name": result.strategy_name,
                "pinescript": pinescript
            })
        except Exception as e:
            results.append({
                "rank": i + 1,
                "strategy_name": strategy.get("strategy_name", "Unknown") if isinstance(strategy, dict) else "Unknown",
                "error": str(e)
            })

    return {"strategies": results}


@router.get("/download-unified-pinescript/{rank}")
async def download_pinescript(rank: int = 1):
    """Download Pine Script as a file for a specific ranked strategy."""
    from strategy_engine import generate_pinescript

    status = app_state.get_unified_status()
    report = status.get("report")

    if not report:
        raise HTTPException(status_code=404, detail="No optimization report available")

    top_strategies = report.get("top_10", [])
    if rank < 1 or rank > len(top_strategies):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_strategies)}")

    strategy = top_strategies[rank - 1]

    try:
        result = dict_to_strategy_result(strategy) if isinstance(strategy, dict) else strategy

        pinescript = generate_pinescript(result)
        filename = f"strategy_{rank}_{result.strategy_name.replace(' ', '_')}.pine"

        return PlainTextResponse(
            content=pinescript,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/unified-trades-csv/{rank}")
async def get_trades_csv(rank: int = 1):
    """Download trades as CSV for a specific ranked strategy."""
    status = app_state.get_unified_status()
    report = status.get("report")

    if not report:
        raise HTTPException(status_code=404, detail="No optimization report available")

    top_strategies = report.get("top_10", [])
    if rank < 1 or rank > len(top_strategies):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_strategies)}")

    strategy = top_strategies[rank - 1]
    trades = strategy.get("trades_list", [])

    if not trades:
        raise HTTPException(status_code=404, detail="No trades available for this strategy")

    # Create CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "trade_num", "direction", "entry_time", "entry_price",
        "exit_time", "exit_price", "pnl", "pnl_pct", "cumulative_pnl"
    ])
    writer.writeheader()

    for i, trade in enumerate(trades):
        writer.writerow({
            "trade_num": i + 1,
            "direction": trade.get("direction", "LONG"),
            "entry_time": trade.get("entry_time", ""),
            "entry_price": trade.get("entry_price", 0),
            "exit_time": trade.get("exit_time", ""),
            "exit_price": trade.get("exit_price", 0),
            "pnl": round(trade.get("pnl", 0), 2),
            "pnl_pct": round(trade.get("pnl_pct", 0), 2),
            "cumulative_pnl": round(trade.get("cumulative_pnl", 0), 2)
        })

    strategy_name = strategy.get("strategy_name", "strategy").replace(" ", "_")
    filename = f"trades_{rank}_{strategy_name}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/export-trades/{strategy_id}")
async def export_trades_from_db(strategy_id: int):
    """Export trades for a strategy from the database."""
    from strategy_database import get_strategy_db

    try:
        db = get_strategy_db()
        strategy = db.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        trades = strategy.get("trades_list", []) or []

        if not trades:
            raise HTTPException(status_code=404, detail="No trades available for this strategy")

        # Create CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "trade_num", "direction", "entry_time", "entry_price",
            "exit_time", "exit_price", "pnl", "pnl_pct"
        ])
        writer.writeheader()

        for i, trade in enumerate(trades):
            writer.writerow({
                "trade_num": i + 1,
                "direction": trade.get("direction", "LONG"),
                "entry_time": trade.get("entry_time", ""),
                "entry_price": trade.get("entry_price", 0),
                "exit_time": trade.get("exit_time", ""),
                "exit_price": trade.get("exit_price", 0),
                "pnl": round(trade.get("pnl", 0), 2),
                "pnl_pct": round(trade.get("pnl_pct", 0), 2)
            })

        strategy_name = strategy.get("strategy_name", "strategy").replace(" ", "_")
        filename = f"trades_{strategy_id}_{strategy_name}.csv"

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tradingview-link/{rank}")
async def get_tradingview_link(rank: int = 1):
    """Generate a TradingView chart link for a strategy."""
    status = app_state.get_unified_status()
    report = status.get("report")

    if not report:
        raise HTTPException(status_code=404, detail="No optimization report available")

    top_strategies = report.get("top_10", [])
    if rank < 1 or rank > len(top_strategies):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_strategies)}")

    strategy = top_strategies[rank - 1]

    # Get symbol and exchange from report metadata
    symbol = report.get("symbol", "BTCUSDT")
    exchange = report.get("exchange", "BINANCE")
    timeframe = report.get("timeframe", "15")

    # Construct TradingView URL
    tv_symbol = f"{exchange}:{symbol}"
    tv_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}&interval={timeframe}"

    return {
        "rank": rank,
        "strategy_name": strategy.get("strategy_name", "Unknown"),
        "symbol": tv_symbol,
        "timeframe": timeframe,
        "url": tv_url
    }
