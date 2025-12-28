"""
OPTIMIZATION ROUTES
==================
API endpoints for manual optimization runs.
"""
import asyncio
import concurrent.futures
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import io
import csv

from config import OUTPUT_DIR
from state import app_state, concurrency_config
from services.websocket_manager import broadcast_optimization_status, broadcast_strategy_result

router = APIRouter(prefix="/api", tags=["optimization"])

# Thread pool for optimization
max_workers = concurrency_config.get("cpu_cores", 4)
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)


# =============================================================================
# REQUEST MODELS
# =============================================================================

class UnifiedOptimizationRequest(BaseModel):
    symbol: str = "BTCGBP"
    timeframe: str = "15m"
    exchange: str = "BINANCE"
    capital: float = 1000.0
    position_size_pct: float = 100.0
    engine: str = "tradingview"  # 'tradingview' or 'native'
    n_trials: int = 400  # Number of optimization trials
    source_currency: str = "GBP"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_optimization_sync(request: UnifiedOptimizationRequest, status: dict, streaming_callback=None):
    """
    Synchronous optimization wrapper that runs in thread pool.
    """
    from strategy_engine import run_strategy_finder

    try:
        df = app_state.get_dataframe()
        if df is None or len(df) == 0:
            status["message"] = "No data loaded"
            status["running"] = False
            return

        report = run_strategy_finder(
            df=df,
            status=status,
            streaming_callback=streaming_callback,
            symbol=request.symbol,
            timeframe=request.timeframe,
            exchange=request.exchange,
            capital=request.capital,
            position_size_pct=request.position_size_pct,
            engine=request.engine,
            n_trials=request.n_trials,
            source_currency=request.source_currency,
        )

        status["report"] = report
        status["running"] = False
        status["progress"] = 100
        status["message"] = "Complete"

    except Exception as e:
        import traceback
        traceback.print_exc()
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
        loop = asyncio.get_event_loop()

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
        # Generate Pine Script
        from strategy_engine import StrategyResult

        # Convert dict to StrategyResult if needed
        if isinstance(strategy, dict):
            result = StrategyResult(
                strategy_name=strategy.get("strategy_name", "Unknown"),
                direction=strategy.get("direction", "long"),
                entry_rule=strategy.get("entry_rule", ""),
                tp_percent=strategy.get("tp_percent", 2.0),
                sl_percent=strategy.get("sl_percent", 1.0),
                total_trades=strategy.get("metrics", {}).get("total_trades", 0),
                wins=strategy.get("metrics", {}).get("wins", 0),
                losses=strategy.get("metrics", {}).get("losses", 0),
                win_rate=strategy.get("metrics", {}).get("win_rate", 0),
                total_pnl=strategy.get("metrics", {}).get("total_pnl", 0),
                total_pnl_percent=strategy.get("metrics", {}).get("total_pnl_percent", 0),
                profit_factor=strategy.get("metrics", {}).get("profit_factor", 0),
                max_drawdown=strategy.get("metrics", {}).get("max_drawdown", 0),
                max_drawdown_pct=strategy.get("metrics", {}).get("max_drawdown_pct", 0),
                trades=[],
                composite_score=strategy.get("metrics", {}).get("composite_score", 0),
            )
        else:
            result = strategy

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
    from strategy_engine import generate_pinescript, StrategyResult

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
            if isinstance(strategy, dict):
                result = StrategyResult(
                    strategy_name=strategy.get("strategy_name", "Unknown"),
                    direction=strategy.get("direction", "long"),
                    entry_rule=strategy.get("entry_rule", ""),
                    tp_percent=strategy.get("tp_percent", 2.0),
                    sl_percent=strategy.get("sl_percent", 1.0),
                    total_trades=strategy.get("metrics", {}).get("total_trades", 0),
                    wins=strategy.get("metrics", {}).get("wins", 0),
                    losses=strategy.get("metrics", {}).get("losses", 0),
                    win_rate=strategy.get("metrics", {}).get("win_rate", 0),
                    total_pnl=strategy.get("metrics", {}).get("total_pnl", 0),
                    total_pnl_percent=strategy.get("metrics", {}).get("total_pnl_percent", 0),
                    profit_factor=strategy.get("metrics", {}).get("profit_factor", 0),
                    max_drawdown=strategy.get("metrics", {}).get("max_drawdown", 0),
                    max_drawdown_pct=strategy.get("metrics", {}).get("max_drawdown_pct", 0),
                    trades=[],
                    composite_score=strategy.get("metrics", {}).get("composite_score", 0),
                )
            else:
                result = strategy

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
    from strategy_engine import generate_pinescript, StrategyResult

    status = app_state.get_unified_status()
    report = status.get("report")

    if not report:
        raise HTTPException(status_code=404, detail="No optimization report available")

    top_strategies = report.get("top_10", [])
    if rank < 1 or rank > len(top_strategies):
        raise HTTPException(status_code=400, detail=f"Invalid rank. Must be 1-{len(top_strategies)}")

    strategy = top_strategies[rank - 1]

    try:
        if isinstance(strategy, dict):
            result = StrategyResult(
                strategy_name=strategy.get("strategy_name", "Unknown"),
                direction=strategy.get("direction", "long"),
                entry_rule=strategy.get("entry_rule", ""),
                tp_percent=strategy.get("tp_percent", 2.0),
                sl_percent=strategy.get("sl_percent", 1.0),
                total_trades=strategy.get("metrics", {}).get("total_trades", 0),
                wins=strategy.get("metrics", {}).get("wins", 0),
                losses=strategy.get("metrics", {}).get("losses", 0),
                win_rate=strategy.get("metrics", {}).get("win_rate", 0),
                total_pnl=strategy.get("metrics", {}).get("total_pnl", 0),
                total_pnl_percent=strategy.get("metrics", {}).get("total_pnl_percent", 0),
                profit_factor=strategy.get("metrics", {}).get("profit_factor", 0),
                max_drawdown=strategy.get("metrics", {}).get("max_drawdown", 0),
                max_drawdown_pct=strategy.get("metrics", {}).get("max_drawdown_pct", 0),
                trades=[],
                composite_score=strategy.get("metrics", {}).get("composite_score", 0),
            )
        else:
            result = strategy

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
