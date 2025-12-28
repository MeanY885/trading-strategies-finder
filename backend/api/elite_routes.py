"""
ELITE VALIDATION ROUTES
======================
API endpoints for Elite strategy validation.
"""
import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException

from state import app_state

router = APIRouter(prefix="/api/elite", tags=["elite"])


@router.get("/status")
async def get_elite_status():
    """Get current Elite validation status."""
    return app_state.get_elite_status()


@router.get("/strategies")
async def get_elite_strategies(status_filter: Optional[str] = None, limit: int = 50):
    """
    Get Elite validated strategies.

    Args:
        status_filter: Filter by status ('elite', 'partial', 'failed', 'pending')
        limit: Maximum number of strategies to return
    """
    try:
        from strategy_database import get_strategy_db
        db = get_strategy_db()

        # Use optimized SQL query with WHERE clause instead of loading all
        strategies = db.get_elite_strategies_filtered(
            status_filter=status_filter,
            limit=limit
        )

        return {"strategies": strategies}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard")
async def get_elite_leaderboard(limit: int = 20):
    """Get top Elite strategies sorted by score."""
    try:
        from strategy_database import get_strategy_db
        db = get_strategy_db()

        # Use optimized SQL query instead of loading all strategies
        return db.get_elite_leaderboard(limit=limit)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/{strategy_id}")
async def trigger_validation(strategy_id: int):
    """
    Manually trigger validation for a specific strategy.
    Sets the strategy to 'pending' status so it will be picked up by the background validator.
    """
    try:
        from strategy_database import get_strategy_db
        db = get_strategy_db()

        # Set strategy to pending
        db.update_elite_status(
            strategy_id=strategy_id,
            elite_status='pending',
            periods_passed=0,
            periods_total=0,
            validation_data=None,
            elite_score=0
        )

        return {
            "success": True,
            "message": f"Strategy {strategy_id} queued for validation"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset/{strategy_id}")
async def reset_validation(strategy_id: int):
    """Reset Elite validation for a specific strategy."""
    try:
        from strategy_database import get_strategy_db
        db = get_strategy_db()

        db.update_elite_status(
            strategy_id=strategy_id,
            elite_status=None,
            periods_passed=0,
            periods_total=0,
            validation_data=None,
            elite_score=0
        )

        return {
            "success": True,
            "message": f"Strategy {strategy_id} validation reset"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_elite_stats():
    """Get Elite validation statistics."""
    try:
        from strategy_database import get_strategy_db
        db = get_strategy_db()

        # Use optimized SQL aggregation instead of loading all strategies
        stats = db.get_elite_stats_optimized()
        stats["validation_running"] = app_state.is_elite_running()

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_elite_validation():
    """Start the Elite validation background service."""
    if app_state.is_elite_auto_running():
        return {"status": "already_running", "message": "Elite validation already running"}

    from services.elite_validator import start_auto_elite_validation
    asyncio.create_task(start_auto_elite_validation())

    return {"status": "started", "message": "Elite validation started"}


@router.post("/stop")
async def stop_elite_validation():
    """Stop the Elite validation background service."""
    from services.elite_validator import stop_elite_validation
    await stop_elite_validation()

    return {"status": "stopped", "message": "Elite validation stopped"}


@router.post("/validate-all")
async def trigger_validate_all():
    """Trigger immediate validation of all pending strategies."""
    from services.elite_validator import validate_all_strategies
    asyncio.create_task(validate_all_strategies())

    return {"status": "started", "message": "Validation of all pending strategies started"}
