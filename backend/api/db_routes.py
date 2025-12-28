"""
DATABASE ROUTES
===============
API endpoints for strategy database operations.
Extracted from main.py for better modularity.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from strategy_database import get_strategy_db
from pinescript_generator import PineScriptGenerator

router = APIRouter(prefix="/api/db", tags=["database"])

# Check database availability
try:
    get_strategy_db()
    HAS_DATABASE = True
except Exception:
    HAS_DATABASE = False


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ValidateStrategyRequest(BaseModel):
    strategy_id: int
    capital: Optional[float] = None
    position_size_pct: Optional[float] = None


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/stats")
def get_database_stats():
    """Get overall database statistics."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/filter-options")
def get_filter_options():
    """Get distinct symbols, timeframes, and date range for filter dropdowns."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_filter_options()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
def get_saved_strategies(
    symbol: str = None,
    timeframe: str = None,
    min_win_rate: float = 0.0,
    limit: int = 500
):
    """Get top 10 strategies per pair/timeframe from database."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()

        # Use optimized SQL query with window functions
        # Gets top 10 per (symbol, timeframe) at the database level
        return db.get_top_strategies_per_market(
            top_n=10,
            symbol=symbol,
            timeframe=timeframe,
            min_win_rate=min_win_rate,
            total_limit=limit
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/best-win-rate")
def get_best_by_win_rate(limit: int = 10):
    """Get strategies with highest win rate."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_best_by_win_rate(limit=limit, min_trades=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/best-profit-factor")
def get_best_by_profit_factor(limit: int = 10):
    """Get strategies with highest profit factor."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_best_by_profit_factor(limit=limit, min_trades=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/search")
def search_strategies(
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


@router.get("/strategies/{strategy_id}")
def get_strategy_by_id(strategy_id: int):
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


@router.get("/strategies/{strategy_id}/pinescript")
def get_strategy_pinescript_from_db(strategy_id: int):
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
        position_size_pct = 100.0  # default
        capital = 1000.0  # default
        run_id = strategy.get('optimization_run_id')
        if run_id:
            run = db.get_optimization_run_by_id(run_id)
            if run:
                position_size_pct = run.get('risk_percent', 100.0)
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


@router.delete("/strategies/{strategy_id}")
def delete_strategy(strategy_id: int):
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


@router.get("/runs")
def get_optimization_runs(limit: int = 20):
    """Get recent optimization runs."""
    if not HAS_DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db = get_strategy_db()
        return db.get_optimization_runs(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
