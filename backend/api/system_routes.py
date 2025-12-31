"""
SYSTEM ROUTES
=============
API endpoints for system status and configuration.
"""
from fastapi import APIRouter

from state import app_state, concurrency_config
from services.resource_monitor import resource_monitor

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/status")
async def get_status():
    """Get current system status (all components)."""
    # Get autonomous status but exclude the massive combinations_list
    autonomous = app_state.get_autonomous_status()
    # Remove large arrays to keep response lean
    autonomous.pop("combinations_list", None)
    autonomous.pop("queue_completed", None)

    return {
        "optimization": app_state.get_unified_status(),
        "data": app_state.get_data_status(),
        "autonomous": autonomous,
        "elite": app_state.get_elite_status(),
    }


@router.get("/system")
async def get_system_info():
    """Get system resource information."""
    running_count = app_state.get_running_count()
    return resource_monitor.get_status(running_count)


@router.get("/concurrency")
async def get_concurrency_config():
    """Get current concurrency configuration."""
    return concurrency_config.get_all()


@router.post("/concurrency")
async def update_concurrency_config(
    max_concurrent: int = None,
    parallel_enabled: bool = None,
    elite_parallel: bool = None,
    adaptive_scaling: bool = None
):
    """Update concurrency configuration."""
    if max_concurrent is not None:
        concurrency_config["max_concurrent"] = max(1, min(max_concurrent, resource_monitor.cpu_cores))

    if parallel_enabled is not None:
        concurrency_config["parallel_enabled"] = parallel_enabled

    if elite_parallel is not None:
        concurrency_config["elite_parallel"] = elite_parallel

    if adaptive_scaling is not None:
        concurrency_config["adaptive_scaling"] = adaptive_scaling

    return concurrency_config.get_all()


@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


@router.post("/db/clear")
async def clear_database():
    """
    Complete database NUKE - clears everything and resets to fresh defaults.

    This will:
    - Delete all strategies
    - Delete all optimization history
    - Delete all completed optimization tracking
    - Reset priority pairs to defaults (from config)
    - Reset priority periods to defaults
    - Reset priority timeframes to defaults
    - Reset priority granularities to defaults
    """
    from fastapi import HTTPException
    try:
        from strategy_database import get_strategy_db
        from config import AUTONOMOUS_CONFIG

        db = get_strategy_db()

        # NUKE everything
        deleted = db.clear_all()

        # Reset priority settings to fresh defaults from config
        config = AUTONOMOUS_CONFIG
        db.reset_priority_pairs(config["pairs"].get("binance", []))
        db.reset_priority_periods(config["periods"])
        db.reset_priority_timeframes(config["timeframes"])
        db.reset_priority_granularities(config["granularities"])

        return {
            "success": True,
            "message": f"Database nuked: {deleted} strategies deleted, priority settings reset to defaults",
            "deleted": deleted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/db/remove-duplicates")
async def remove_duplicate_strategies():
    """Remove duplicate strategies from the database, keeping only the most recent."""
    from fastapi import HTTPException
    try:
        from strategy_database import get_strategy_db
        db = get_strategy_db()
        removed = db.remove_duplicates()
        return {"success": True, "message": f"Removed {removed} duplicate strategies", "removed_count": removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vectorbt-status")
async def get_vectorbt_status():
    """Check if VectorBT is available for high-performance backtesting."""
    from services.vectorbt_engine import is_vectorbt_available
    available = is_vectorbt_available()
    return {
        "available": available,
        "speedup": "100x" if available else None,
        "message": "VectorBT enabled for 100x faster backtesting" if available else "VectorBT not installed - using standard engine"
    }
