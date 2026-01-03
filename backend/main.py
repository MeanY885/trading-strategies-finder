"""
BTCGBP ML Optimizer - Main FastAPI Application (Refactored)
===========================================================
Clean, modular entry point for the application.

All business logic has been extracted to separate modules:
- config.py: Configuration constants
- state.py: Thread-safe state management
- api/: API route handlers
- services/: Business logic services
- engine/: Strategy engine components
- models/: Data classes

This file is now only responsible for:
1. Initializing the FastAPI app
2. Registering routes
3. Setting up WebSocket endpoint
4. Managing application lifecycle
"""
import os
import json
import asyncio
import concurrent.futures
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Configuration and state
from config import DATA_DIR, OUTPUT_DIR
from state import app_state, concurrency_config
from logging_config import log, UVICORN_LOG_CONFIG
from strategy_database import get_strategy_db
from async_database import AsyncDatabase

# Services
from services.websocket_manager import ws_manager, broadcast_full_state, serialize_for_json
from services.resource_monitor import resource_monitor
from services.cache import (
    strategies_cache, counts_cache, stats_cache, priority_cache,
    CacheKeys, invalidate_strategy_caches
)

# API routes
from api import register_routes

# Initialize logging
log("[Startup] Trading Optimizer v2.0.0 (Refactored)")
log(f"[Startup] CPU cores: {resource_monitor.cpu_cores}")
log(f"[Startup] Memory: {resource_monitor.mem_total_gb:.1f} GB")
log(f"[Startup] Max workers: {resource_monitor.current_max}")

# Thread pool for blocking operations
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=resource_monitor.current_max)

# Update concurrency config with detected values
concurrency_config.update(
    max_concurrent=resource_monitor.current_max,
    auto_detected=resource_monitor.current_max,
    cpu_cores=resource_monitor.cpu_cores,
    memory_total_gb=round(resource_monitor.mem_total_gb, 1),
    memory_available_gb=round(resource_monitor.get_current_resources()["memory_available_gb"], 1),
)


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def warm_up_caches():
    """Pre-populate caches on startup for instant first requests."""
    log("[Startup] Warming up caches...")
    try:
        db = get_strategy_db()

        # Warm up elite counts cache
        counts = db.get_elite_counts()
        counts_cache.set(CacheKeys.ELITE_COUNTS, counts, ttl=60)
        log(f"[Cache] Elite counts cached: {counts}")

        # Warm up DB stats cache
        db_stats = db.get_db_stats_optimized()
        stats_cache.set(CacheKeys.DB_STATS, db_stats, ttl=60)
        log(f"[Cache] DB stats cached: {db_stats['total_strategies']} strategies")

        # Warm up elite strategies cache
        elite_strategies = db.get_elite_strategies_optimized(top_n_per_market=10)
        strategies_cache.set(CacheKeys.ELITE_STRATEGIES, elite_strategies, ttl=300)
        log(f"[Cache] Elite strategies cached: {len(elite_strategies)} strategies")

        # Warm up priority lists cache (auto-populate if empty)
        priority_data = db.get_all_priority_lists()
        if not priority_data.get('populated'):
            from config import AUTONOMOUS_CONFIG
            config = AUTONOMOUS_CONFIG
            db.reset_priority_pairs(config["pairs"].get("binance", []))
            db.reset_priority_periods(config["periods"])
            db.reset_priority_timeframes(config["timeframes"])
            db.reset_priority_granularities(config["granularities"])
            priority_data = db.get_all_priority_lists()
            log(f"[Cache] Priority lists auto-populated with defaults")
        # Convert enabled to bool for JSON serialization
        for p in priority_data.get('pairs', []):
            p['enabled'] = bool(p.get('enabled', 1))
        for p in priority_data.get('periods', []):
            p['enabled'] = bool(p.get('enabled', 1))
        for t in priority_data.get('timeframes', []):
            t['enabled'] = bool(t.get('enabled', 1))
        for g in priority_data.get('granularities', []):
            g['enabled'] = bool(g.get('enabled', 1))
        priority_cache.set(CacheKeys.PRIORITY_LISTS, priority_data, ttl=300)
        log(f"[Cache] Priority lists cached")

        log("[Startup] Cache warm-up complete!")
    except Exception as e:
        log(f"[Startup] Cache warm-up failed (non-critical): {e}", level='WARNING')


async def start_background_services():
    """Start background services after app initialization."""
    await asyncio.sleep(5)  # Wait for app to be fully ready

    log("[Startup] Starting background services...")

    try:
        # Initialize autonomous optimizer async primitives
        from services.autonomous_optimizer import init_async_primitives
        init_async_primitives()

        # Note: Autonomous optimizer is NOT started automatically
        # User must explicitly enable it via the UI toggle
        log("[Startup] Autonomous optimizer ready (requires manual start)")

        # Note: Elite validation is NOT started automatically
        # User must explicitly enable it via the UI toggle
        log("[Startup] Elite validation ready (requires manual start)")

    except Exception as e:
        log(f"[Startup] Error starting background services: {e}", level='WARNING')
        import traceback
        traceback.print_exc()


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

def run_auto_tune_if_needed():
    """Run auto-tune benchmark if no cached config exists."""
    from pathlib import Path
    import os

    auto_tuned_path = Path(__file__).parent / ".auto_tuned"
    force_tune = os.getenv("FORCE_AUTOTUNE", "0") == "1"
    has_env_override = os.getenv("CORES_PER_TASK") is not None

    if has_env_override:
        log("[Auto-Tune] CORES_PER_TASK set via environment, skipping benchmark")
        return

    if auto_tuned_path.exists() and not force_tune:
        log("[Auto-Tune] Using cached settings from previous run")
        return

    log("[Auto-Tune] Running startup benchmark to detect optimal settings...")
    try:
        from tools.auto_tune import main as auto_tune_main
        auto_tune_main()
    except Exception as e:
        log(f"[Auto-Tune] Benchmark failed (using defaults): {e}", level='WARNING')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    log("[Startup] Application starting...")

    # Run auto-tune benchmark if needed (first boot or FORCE_AUTOTUNE=1)
    run_auto_tune_if_needed()

    # Reload config after potential auto-tune
    # (reimport to get updated values)
    import importlib
    import config as cfg_module
    importlib.reload(cfg_module)

    from config import (
        CORES_PER_TASK, MEMORY_PER_TASK_GB, RESERVED_CORES,
        MAX_CONCURRENT_CALCULATED, CPU_CORES, MEMORY_AVAILABLE_GB
    )
    log("[Startup] " + "=" * 50)
    log("[Startup] OPTIMIZER CONFIGURATION")
    log("[Startup] " + "=" * 50)
    log(f"[Startup]   System: {CPU_CORES} cores, {MEMORY_AVAILABLE_GB:.1f}GB available")
    log(f"[Startup]   CORES_PER_TASK: {CORES_PER_TASK}")
    log(f"[Startup]   MAX_CONCURRENT: {MAX_CONCURRENT_CALCULATED}")
    log(f"[Startup]   Total threads: {MAX_CONCURRENT_CALCULATED * CORES_PER_TASK}")
    log("[Startup] " + "=" * 50)

    # Initialize async database pool FIRST (critical for non-blocking operations)
    try:
        await AsyncDatabase.init_pool()
        log("[Startup] Async database pool initialized")
    except Exception as e:
        log(f"[Startup] Async database pool failed: {e}", level='ERROR')

    # Register main event loop for cross-thread WebSocket broadcasts
    from services.websocket_manager import ws_manager
    ws_manager.set_main_loop(asyncio.get_running_loop())

    # Warm up caches immediately for fast first requests
    await warm_up_caches()

    # Schedule delayed startup for background services
    asyncio.create_task(start_background_services())

    yield

    # Shutdown
    log("[Shutdown] Application shutting down...")
    await AsyncDatabase.close_pool()
    log("[Shutdown] Async database pool closed")
    thread_pool.shutdown(wait=False)


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Trading Optimizer",
    version="2.0.0",
    description="ML-powered cryptocurrency trading strategy optimizer",
    lifespan=lifespan
)

# Add CORS middleware
# Note: For production, restrict allow_origins to specific domains instead of "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routes
register_routes(app)

# Set thread pool reference for autonomous routes
from api.autonomous_routes import set_thread_pool
set_thread_pool(thread_pool)


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """
    WebSocket endpoint for real-time status updates.

    Message types:
    - full_state: Complete state (sent on connect)
    - data_status: Data loading updates
    - optimization_status: Manual optimization updates
    - autonomous_status: Autonomous optimizer updates
    - elite_status: Elite validation updates
    """
    await ws_manager.connect(websocket)

    # Send initial full state
    try:
        full_state = app_state.get_full_state()
        # Serialize to handle datetime objects
        await websocket.send_json(serialize_for_json({"type": "full_state", **full_state}))
    except Exception as e:
        log(f"[WebSocket] Error sending initial state: {e}", level='WARNING')

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                # Handle ping/pong for keepalive
                if message == "ping":
                    await websocket.send_text("pong")
                    continue

                # Handle JSON requests
                try:
                    request = json.loads(message)
                    msg_type = request.get("type")
                    request_id = request.get("id")
                    loop = asyncio.get_event_loop()

                    if msg_type == "get_state":
                        full_state = app_state.get_full_state()
                        await websocket.send_json(serialize_for_json({"type": "full_state", **full_state}))

                    elif msg_type == "get_strategies":
                        # Get strategy history - cached with async DB
                        cached = strategies_cache.get(CacheKeys.STRATEGIES_ALL)
                        if cached is not None:
                            strategies = cached
                        else:
                            result = await AsyncDatabase.get_strategies_paginated(limit=1000, offset=0)
                            strategies = result['strategies']
                            strategies_cache.set(CacheKeys.STRATEGIES_ALL, strategies, ttl=300)

                        await websocket.send_json(serialize_for_json({
                            "type": "strategies_data",
                            "id": request_id,
                            "data": strategies
                        }))

                    elif msg_type == "get_elite":
                        # Get elite strategies and status - cached with async DB
                        cached_counts = counts_cache.get(CacheKeys.ELITE_COUNTS)
                        cached_strategies = strategies_cache.get(CacheKeys.ELITE_STRATEGIES)

                        # Use async SQL aggregation for counts
                        if cached_counts is None:
                            cached_counts = await AsyncDatabase.get_elite_counts()
                            counts_cache.set(CacheKeys.ELITE_COUNTS, cached_counts, ttl=60)

                        # Get elite strategies (already ranked by score)
                        if cached_strategies is None:
                            cached_strategies = await AsyncDatabase.get_elite_strategies_optimized(top_n_per_market=10)
                            strategies_cache.set(CacheKeys.ELITE_STRATEGIES, cached_strategies, ttl=300)

                        elite_data = {
                            "status": {
                                # New simplified status model
                                "validated": cached_counts.get('validated', 0),
                                "pending": cached_counts.get('pending', 0),
                                "untestable": cached_counts.get('untestable', 0),
                                "skipped": cached_counts.get('skipped', 0),
                                **app_state.get_elite_status()
                            },
                            "strategies": cached_strategies
                        }

                        await websocket.send_json(serialize_for_json({
                            "type": "elite_data",
                            "id": request_id,
                            **elite_data
                        }))

                    elif msg_type == "get_priority":
                        # Get priority lists - cached with async DB
                        from config import AUTONOMOUS_CONFIG

                        cached = priority_cache.get(CacheKeys.PRIORITY_LISTS)
                        if cached is not None:
                            priority_data = cached
                        else:
                            data = await AsyncDatabase.get_all_priority_lists()

                            # Auto-populate if empty (rare, use sync fallback)
                            if not data.get('populated'):
                                db = get_strategy_db()
                                config = AUTONOMOUS_CONFIG
                                db.reset_priority_pairs(config["pairs"].get("binance", []))
                                db.reset_priority_periods(config["periods"])
                                db.reset_priority_timeframes(config["timeframes"])
                                db.reset_priority_granularities(config["granularities"])
                                data = await AsyncDatabase.get_all_priority_lists()

                            # Convert enabled to bool for JSON serialization
                            for p in data.get('pairs', []):
                                p['enabled'] = bool(p.get('enabled', 1))
                            for p in data.get('periods', []):
                                p['enabled'] = bool(p.get('enabled', 1))
                            for t in data.get('timeframes', []):
                                t['enabled'] = bool(t.get('enabled', 1))
                            for g in data.get('granularities', []):
                                g['enabled'] = bool(g.get('enabled', 1))

                            priority_cache.set(CacheKeys.PRIORITY_LISTS, data, ttl=300)
                            priority_data = data

                        await websocket.send_json(serialize_for_json({
                            "type": "priority_data",
                            "id": request_id,
                            "data": priority_data
                        }))

                    elif msg_type == "get_db_stats":
                        # Get database stats for Tools page - cached with async DB
                        cached = stats_cache.get(CacheKeys.DB_STATS)
                        if cached is not None:
                            db_stats = cached
                        else:
                            db_stats = await AsyncDatabase.get_db_stats_optimized()
                            stats_cache.set(CacheKeys.DB_STATS, db_stats, ttl=60)

                        await websocket.send_json(serialize_for_json({
                            "type": "db_stats_data",
                            "id": request_id,
                            "data": db_stats
                        }))

                    elif msg_type == "get_queue":
                        status = app_state.get_autonomous_status()
                        combinations = status.get("combinations_list", [])
                        cycle_index = status.get("cycle_index", 0)
                        total = len(combinations)

                        # Get pending items (next 5 after current index)
                        pending_items = []
                        running_indices = {r.get("index") for r in status.get("parallel_running", [])}
                        for i in range(cycle_index, min(cycle_index + 10, total)):
                            if i not in running_indices:
                                combo = combinations[i] if i < len(combinations) else None
                                if combo:
                                    pending_items.append({
                                        "index": i,
                                        "pair": combo.get("pair", ""),
                                        "period": combo.get("period", ""),
                                        "timeframe": combo.get("timeframe", ""),
                                        "granularity": combo.get("granularity", ""),
                                        "status": "pending"
                                    })
                                if len(pending_items) >= 5:
                                    break

                        parallel_running = status.get("parallel_running", [])
                        await websocket.send_json(serialize_for_json({
                            "type": "queue_data",
                            "id": request.get("id"),
                            "data": {
                                "completed": status.get("queue_completed", [])[-10:],
                                "running": parallel_running,
                                "pending": pending_items,
                                "total": total,
                                "cycle_index": cycle_index,
                                "pending_remaining": max(0, total - cycle_index - len(pending_items)),
                                "parallel_count": len(parallel_running),
                                "max_parallel": status.get("max_parallel", 4),
                            }
                        }))

                except json.JSONDecodeError:
                    pass  # Ignore non-JSON messages
                except Exception as e:
                    # Send error response instead of crashing the connection
                    log(f"[WebSocket] Request handler error: {e}", level='WARNING')
                    import traceback
                    traceback.print_exc()
                    try:
                        await websocket.send_json(serialize_for_json({
                            "type": "error",
                            "id": locals().get('request_id'),
                            "error": str(e)
                        }))
                    except Exception:
                        pass  # Connection may already be broken

            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_text("ping")
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log(f"[WebSocket] Error: {e}", level='WARNING')
    finally:
        await ws_manager.disconnect(websocket)


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.get("/api/ping")
async def ping():
    """Simple ping endpoint for health checks."""
    return {"pong": True, "timestamp": datetime.now().isoformat()}


# =============================================================================
# WATCHDOG STATUS ENDPOINTS
# =============================================================================

@app.get("/api/watchdog/status")
async def get_watchdog_status():
    """Get current watchdog and task monitoring status."""
    from services.autonomous_optimizer import (
        running_optimizations,
        orphan_cleaner_instance
    )
    from services.elite_validator import running_validations
    from config import WATCHDOG_CONFIG

    return {
        "watchdog": {
            "orphan_cleaner_running": orphan_cleaner_instance is not None,
            "orphans_cleaned": orphan_cleaner_instance.total_cleaned if orphan_cleaner_instance else 0,
        },
        "running_optimizations": len(running_optimizations),
        "running_validations": len(running_validations),
        "config": {
            "orphan_threshold_seconds": WATCHDOG_CONFIG["orphan_threshold_seconds"],
            "no_progress_abort_seconds": WATCHDOG_CONFIG["no_progress_abort_seconds"],
            "stall_timeout": WATCHDOG_CONFIG["stall_timeout"],
        }
    }


@app.post("/api/watchdog/cleanup")
async def trigger_watchdog_cleanup():
    """Manually trigger orphan cleanup for running tasks."""
    from services.autonomous_optimizer import (
        running_optimizations,
        running_optimizations_async_lock,
        orphan_cleaner_instance
    )
    from services.elite_validator import (
        running_validations,
        running_validations_async_lock
    )

    results = {
        "optimizations_cleaned": 0,
        "validations_cleaned": 0,
    }

    # Clean orphaned optimizations
    if orphan_cleaner_instance:
        results["optimizations_cleaned"] = await orphan_cleaner_instance.cleanup_orphans()
    else:
        # Manual cleanup if orphan cleaner not running
        if running_optimizations_async_lock:
            async with running_optimizations_async_lock:
                count = len(running_optimizations)
                running_optimizations.clear()
                results["optimizations_cleaned"] = count

    # Clean orphaned validations
    if running_validations_async_lock:
        async with running_validations_async_lock:
            count = len(running_validations)
            running_validations.clear()
            results["validations_cleaned"] = count

    log(f"[Watchdog] Manual cleanup: {results}")
    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_config=UVICORN_LOG_CONFIG
    )
