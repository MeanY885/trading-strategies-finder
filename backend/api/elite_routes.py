"""
ELITE VALIDATION ROUTES
======================
API endpoints for Elite strategy validation.
Uses AsyncDatabase for non-blocking database operations.
"""
import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException

from state import app_state
from async_database import AsyncDatabase

router = APIRouter(prefix="/api/elite", tags=["elite"])


@router.get("/status")
async def get_elite_status():
    """Get current Elite validation status."""
    return app_state.get_elite_status()


@router.get("/strategies")
async def get_elite_strategies(status_filter: Optional[str] = None, limit: int = 50):
    """
    Get Elite validated strategies (non-blocking).

    Args:
        status_filter: Filter by status ('validated', 'pending', 'untestable', 'skipped')
        limit: Maximum number of strategies to return
    """
    try:
        strategies = await AsyncDatabase.get_elite_strategies_filtered(
            status_filter=status_filter,
            limit=limit
        )
        return {"strategies": strategies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard")
async def get_elite_leaderboard(limit: int = 20):
    """Get top Elite strategies sorted by score (non-blocking)."""
    try:
        # Use existing optimized async method
        strategies = await AsyncDatabase.get_elite_strategies_optimized(top_n_per_market=limit)
        return {"strategies": strategies[:limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/{strategy_id}")
async def trigger_validation(strategy_id: int):
    """
    Manually trigger validation for a specific strategy (non-blocking).
    Sets the strategy to 'pending' status so it will be picked up by the background validator.
    """
    try:
        await AsyncDatabase.update_elite_status(
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
    """Reset Elite validation for a specific strategy (non-blocking)."""
    try:
        await AsyncDatabase.update_elite_status(
            strategy_id=strategy_id,
            elite_status='pending',  # Set to pending for re-validation
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
    """Get Elite validation statistics (non-blocking)."""
    try:
        counts = await AsyncDatabase.get_elite_counts()
        stats = {
            "total": sum(counts.values()),
            "validated_count": counts.get('validated', 0),
            "pending_count": counts.get('pending', 0),
            "untestable_count": counts.get('untestable', 0),
            "skipped_count": counts.get('skipped', 0),
            "validation_running": app_state.is_elite_running()
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/toggle")
async def toggle_elite_validation():
    """Toggle the Elite validation service on/off."""
    from services.websocket_manager import broadcast_elite_status
    from logging_config import log

    status = app_state.get_elite_status()
    currently_running = status.get("auto_running", False)

    if currently_running:
        # Stop the service
        from services.elite_validator import stop_elite_validation
        await stop_elite_validation()
        log("[Elite Validation] Stopped via toggle")
        return {"status": "disabled", "message": "Elite validation stopped"}
    else:
        # Start the service
        app_state.update_elite_status(message="Starting...")
        broadcast_elite_status(app_state.get_elite_status())

        from services.elite_validator import start_auto_elite_validation
        asyncio.create_task(start_auto_elite_validation())
        log("[Elite Validation] Started via toggle")
        return {"status": "enabled", "message": "Elite validation started"}


@router.post("/start")
async def start_elite_validation():
    """Start the Elite validation background service."""
    from services.websocket_manager import broadcast_elite_status
    from logging_config import log

    if app_state.is_elite_auto_running():
        return {"status": "already_running", "message": "Elite validation already running"}

    app_state.update_elite_status(message="Starting...")
    broadcast_elite_status(app_state.get_elite_status())

    from services.elite_validator import start_auto_elite_validation
    asyncio.create_task(start_auto_elite_validation())
    log("[Elite Validation] Started via /start endpoint")

    return {"status": "started", "message": "Elite validation started"}


@router.post("/stop")
async def stop_elite_validation():
    """Stop the Elite validation background service."""
    from services.elite_validator import stop_elite_validation
    from logging_config import log

    await stop_elite_validation()
    log("[Elite Validation] Stopped via /stop endpoint")

    return {"status": "stopped", "message": "Elite validation stopped"}


@router.post("/pause")
async def pause_elite_validation():
    """Toggle pause state - pause if running, resume if paused."""
    from services.elite_validator import pause_elite_validation, resume_elite_validation
    from services.websocket_manager import broadcast_elite_status
    from logging_config import log

    status = app_state.get_elite_status()
    currently_paused = status.get("paused", False)
    is_running = status.get("auto_running", False)

    if not is_running:
        return {"status": "not_running", "message": "Elite validation is not running"}

    if currently_paused:
        # Resume
        resume_elite_validation()
        log("[Elite Validation] Resumed via /pause toggle")
        broadcast_elite_status(app_state.get_elite_status())
        return {"status": "resumed", "message": "Elite validation resumed"}
    else:
        # Pause
        pause_elite_validation()
        log("[Elite Validation] Paused via /pause toggle")
        broadcast_elite_status(app_state.get_elite_status())
        return {"status": "paused", "message": "Elite validation paused"}


@router.post("/validate-all")
async def trigger_validate_all():
    """Trigger immediate validation of all pending strategies."""
    from services.elite_validator import validate_all_strategies
    asyncio.create_task(validate_all_strategies())

    return {"status": "started", "message": "Validation of all pending strategies started"}


@router.post("/reset-all")
async def reset_all_elite_validation():
    """
    Reset all elite validation data (non-blocking).
    Sets all strategies to 'pending' status so they will be re-validated.
    """
    try:
        total_count = await AsyncDatabase.reset_all_elite_validation()

        # Invalidate cache
        from services.cache import invalidate_counts_cache
        invalidate_counts_cache()

        return {
            "success": True,
            "reset_count": total_count,
            "message": f"Reset {total_count} strategies to pending validation"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
