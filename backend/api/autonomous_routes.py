"""
AUTONOMOUS OPTIMIZER ROUTES
===========================
API endpoints for the autonomous optimizer that cycles through combinations.
"""
import asyncio
from fastapi import APIRouter, HTTPException

from config import AUTONOMOUS_CONFIG
from state import app_state

router = APIRouter(prefix="/api/autonomous", tags=["autonomous"])

# Reference to thread pool (set by main.py during startup)
_thread_pool = None

def set_thread_pool(pool):
    """Set the thread pool reference for use by start endpoint."""
    global _thread_pool
    _thread_pool = pool


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/status")
async def get_autonomous_status():
    """Get current autonomous optimizer status."""
    return app_state.get_autonomous_status()


@router.post("/toggle")
async def toggle_autonomous():
    """Toggle the autonomous optimizer on/off."""
    from services.websocket_manager import broadcast_autonomous_status
    from logging_config import log

    status = app_state.get_autonomous_status()
    currently_running = status.get("auto_running", False) or status.get("enabled", False)

    if currently_running:
        # Disable - use the proper stop function
        from services.autonomous_optimizer import stop_autonomous_optimizer
        await stop_autonomous_optimizer()

        # Also clear any running optimization tracking
        app_state.clear_running_optimizations()

        # Clear unified status so Elite validation can run immediately
        app_state.update_unified_status(running=False)

        log("[Autonomous Optimizer] Stopped via toggle")
        return {"status": "disabled", "message": "Autonomous optimizer stopped"}
    else:
        # Enable
        app_state.update_autonomous_status(
            enabled=True,
            message="Starting..."
        )
        broadcast_autonomous_status(app_state.get_autonomous_status())

        # Start the loop if not already running
        if not app_state.is_autonomous_running():
            from services.autonomous_optimizer import start_autonomous_optimizer
            if _thread_pool is None:
                raise HTTPException(status_code=500, detail="Thread pool not initialized")

            async def start_with_error_handling():
                try:
                    log("[Autonomous Optimizer] Task starting...")
                    await start_autonomous_optimizer(_thread_pool)
                except Exception as e:
                    import traceback
                    log(f"[Autonomous Optimizer] Task CRASHED: {e}", level='ERROR')
                    traceback.print_exc()
                    # Reset state on crash
                    app_state.update_autonomous_status(
                        auto_running=False,
                        running=False,
                        enabled=False,
                        message=f"Error: {str(e)}"
                    )
                    broadcast_autonomous_status(app_state.get_autonomous_status())

            asyncio.create_task(start_with_error_handling())
            log("[Autonomous Optimizer] Task created via toggle")

        return {"status": "enabled", "message": "Autonomous optimizer enabled"}


@router.get("/config")
async def get_autonomous_config():
    """Get autonomous optimizer configuration."""
    return AUTONOMOUS_CONFIG


@router.get("/history")
async def get_autonomous_history(limit: int = 50):
    """Get history of autonomous optimization runs."""
    history = app_state.get_history()
    return {"history": history[:limit]}


@router.get("/queue")
async def get_queue_status():
    """Get queue state for UI task list display - supports parallel processing."""
    from state import concurrency_config

    status = app_state.get_autonomous_status()
    cycle_index = status.get("cycle_index", 0)
    combinations = status.get("combinations_list", [])
    total = len(combinations)

    # Get currently running items (parallel processing)
    parallel_running = status.get("parallel_running", [])

    # For backwards compatibility, also check queue_current
    current = status.get("queue_current")

    # Get pending items (next 5 after current index)
    pending_start = cycle_index
    pending_items = []
    running_indices = {r.get("index") for r in parallel_running}
    # Also exclude pairs that are currently running to avoid showing same pair twice
    running_pairs = {r.get("pair") for r in parallel_running}

    for i in range(pending_start, min(pending_start + 10, total)):
        if i < total and i not in running_indices:
            combo = combinations[i]
            pair = combo.get("pair", "")
            # Skip if this pair is already running (even with different settings)
            if pair in running_pairs:
                continue
            pending_items.append({
                "index": i,
                "pair": pair,
                "period": combo.get("period", ""),
                "timeframe": combo.get("timeframe", ""),
                "granularity": combo.get("granularity", ""),
                "status": "pending"
            })
            if len(pending_items) >= 5:
                break

    return {
        "completed": status.get("queue_completed", []),
        "current": current,  # Legacy single item
        "running": parallel_running,  # Multiple parallel items
        "parallel_count": len(parallel_running),
        "max_parallel": concurrency_config.get("max_concurrent", 4),
        "pending": pending_items,
        "pending_remaining": max(0, total - cycle_index - len(pending_items)),
        "total": total,
        "cycle_index": cycle_index,
        # Current progress info (aggregated from parallel tasks)
        "trial_current": status.get("trial_current", 0),
        "trial_total": status.get("trial_total", 0),
        "current_strategy": status.get("current_strategy"),
        "message": status.get("message", ""),
    }


@router.get("/skipped")
async def get_skipped_validations(limit: int = 50):
    """Get list of skipped validations with reasons."""
    status = app_state.get_autonomous_status()
    skipped = status.get("skipped_validations", [])
    return {"skipped": skipped[:limit]}


@router.post("/reset-counters")
async def reset_counters():
    """Reset completed/error/skipped counters."""
    app_state.update_autonomous_status(
        completed_count=0,
        error_count=0,
        skipped_count=0,
        skipped_validations=[],
        queue_completed=[]
    )

    from services.websocket_manager import broadcast_autonomous_status
    broadcast_autonomous_status(app_state.get_autonomous_status())

    return {"success": True, "message": "Counters reset"}


@router.post("/enable")
async def enable_autonomous():
    """Enable autonomous optimizer."""
    app_state.update_autonomous_status(enabled=True)

    from services.websocket_manager import broadcast_autonomous_status
    broadcast_autonomous_status(app_state.get_autonomous_status())

    return {"status": "enabled", "message": "Autonomous optimizer enabled"}


@router.post("/start")
async def start_autonomous():
    """Explicitly start autonomous optimizer."""
    from services.websocket_manager import broadcast_autonomous_status
    from logging_config import log

    app_state.update_autonomous_status(enabled=True, message="Starting...")
    broadcast_autonomous_status(app_state.get_autonomous_status())

    if not app_state.is_autonomous_running():
        from services.autonomous_optimizer import start_autonomous_optimizer
        if _thread_pool is None:
            raise HTTPException(status_code=500, detail="Thread pool not initialized")

        async def start_with_error_handling():
            try:
                log("[Autonomous Optimizer] Task starting via /start...")
                await start_autonomous_optimizer(_thread_pool)
            except Exception as e:
                import traceback
                log(f"[Autonomous Optimizer] Task CRASHED: {e}", level='ERROR')
                traceback.print_exc()
                app_state.update_autonomous_status(
                    auto_running=False,
                    running=False,
                    enabled=False,
                    message=f"Error: {str(e)}"
                )
                broadcast_autonomous_status(app_state.get_autonomous_status())

        asyncio.create_task(start_with_error_handling())
        log("[Autonomous Optimizer] Task created via /start endpoint")
        return {"status": "started", "message": "Autonomous optimizer started"}
    else:
        return {"status": "already_running", "message": "Already running"}


@router.post("/stop")
async def stop_autonomous():
    """Stop autonomous optimizer."""
    from services.autonomous_optimizer import stop_autonomous_optimizer
    await stop_autonomous_optimizer()

    return {"status": "stopped", "message": "Autonomous optimizer stopped"}


@router.post("/reset-cycle")
async def reset_autonomous_cycle():
    """Reset autonomous optimizer cycle to beginning."""
    app_state.update_autonomous_status(
        cycle_index=0,
        completed_count=0,
        skipped_count=0,
        error_count=0,
        skipped_validations=[],
        message="Cycle reset - starting from beginning"
    )

    from services.websocket_manager import broadcast_autonomous_status
    broadcast_autonomous_status(app_state.get_autonomous_status())

    return {"status": "reset", "message": "Cycle reset to beginning"}


@router.get("/results")
async def get_autonomous_results():
    """Get summary of autonomous optimization results."""
    status = app_state.get_autonomous_status()
    return {
        "completed_count": status.get("completed_count", 0),
        "error_count": status.get("error_count", 0),
        "total_combinations": status.get("total_combinations", 0),
        "cycle_index": status.get("cycle_index", 0),
        "best_strategy": status.get("best_strategy_found"),
        "last_result": status.get("last_result"),
        "last_completed_at": status.get("last_completed_at"),
    }
