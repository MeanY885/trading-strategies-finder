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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configuration and state
from config import FRONTEND_DIR, DATA_DIR, OUTPUT_DIR
from state import app_state, concurrency_config
from logging_config import log, UVICORN_LOG_CONFIG
from strategy_database import get_strategy_db

# Services
from services.websocket_manager import ws_manager, broadcast_full_state
from services.resource_monitor import resource_monitor
from services.cache import (
    strategies_cache, counts_cache, stats_cache, priority_cache,
    CacheKeys, invalidate_strategy_caches
)

# API routes
from api import register_routes

# Initialize logging
log("[Startup] BTCGBP ML Optimizer v2.0.0 (Refactored)")
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

        # Warm up priority lists cache
        priority_data = db.get_all_priority_lists()
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

        # Start Elite validation in background
        from services.elite_validator import start_auto_elite_validation
        asyncio.create_task(start_auto_elite_validation())
        log("[Startup] Elite validation service started")

        # Start Autonomous optimizer in background
        from services.autonomous_optimizer import start_autonomous_optimizer
        asyncio.create_task(start_autonomous_optimizer(thread_pool))
        log("[Startup] Autonomous optimizer service started")

    except Exception as e:
        log(f"[Startup] Error starting background services: {e}", level='WARNING')
        import traceback
        traceback.print_exc()


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    log("[Startup] Application starting...")

    # Warm up caches immediately for fast first requests
    await warm_up_caches()

    # Schedule delayed startup for background services
    asyncio.create_task(start_background_services())

    yield

    # Shutdown
    log("[Shutdown] Application shutting down...")
    thread_pool.shutdown(wait=False)


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="BTCGBP ML Optimizer",
    version="2.0.0",
    description="ML-powered cryptocurrency trading strategy optimizer",
    lifespan=lifespan
)

# Register all API routes
register_routes(app)

# Set thread pool reference for autonomous routes
from api.autonomous_routes import set_thread_pool
set_thread_pool(thread_pool)


# =============================================================================
# FRONTEND ROUTES
# =============================================================================

@app.get("/")
async def serve_frontend():
    """Serve the main UI."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


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
        await websocket.send_json({"type": "full_state", **full_state})
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
                        await websocket.send_json({"type": "full_state", **full_state})

                    elif msg_type == "get_strategies":
                        # Get strategy history - cached with pagination
                        def fetch_strategies():
                            cached = strategies_cache.get(CacheKeys.STRATEGIES_ALL)
                            if cached is not None:
                                return cached
                            db = get_strategy_db()
                            # Use paginated query - much faster than get_all_strategies()
                            result = db.get_strategies_paginated(limit=1000, offset=0)
                            strategies_cache.set(CacheKeys.STRATEGIES_ALL, result['strategies'], ttl=300)
                            return result['strategies']

                        strategies = await loop.run_in_executor(thread_pool, fetch_strategies)
                        await websocket.send_json({
                            "type": "strategies_data",
                            "id": request_id,
                            "data": strategies
                        })

                    elif msg_type == "get_elite":
                        # Get elite strategies and status - cached and optimized
                        def fetch_elite_data():
                            # Check cache first
                            cached_counts = counts_cache.get(CacheKeys.ELITE_COUNTS)
                            cached_strategies = strategies_cache.get(CacheKeys.ELITE_STRATEGIES)

                            db = get_strategy_db()

                            # Use SQL aggregation for counts (much faster)
                            if cached_counts is None:
                                cached_counts = db.get_elite_counts()
                                counts_cache.set(CacheKeys.ELITE_COUNTS, cached_counts, ttl=60)

                            # Get elite strategies (already ranked by score)
                            if cached_strategies is None:
                                cached_strategies = db.get_elite_strategies_optimized(top_n_per_market=10)
                                strategies_cache.set(CacheKeys.ELITE_STRATEGIES, cached_strategies, ttl=300)

                            return {
                                "status": {
                                    "elite_count": cached_counts.get('elite', 0),
                                    "partial_count": cached_counts.get('partial', 0),
                                    "failed_count": cached_counts.get('failed', 0),
                                    "pending_count": cached_counts.get('pending', 0),
                                    **app_state.get_elite_status()
                                },
                                "strategies": cached_strategies
                            }

                        elite_data = await loop.run_in_executor(thread_pool, fetch_elite_data)
                        await websocket.send_json({
                            "type": "elite_data",
                            "id": request_id,
                            **elite_data
                        })

                    elif msg_type == "get_priority":
                        # Get priority lists - cached and properly formatted
                        from config import AUTONOMOUS_CONFIG

                        def fetch_priority_data():
                            cached = priority_cache.get(CacheKeys.PRIORITY_LISTS)
                            if cached is not None:
                                return cached

                            db = get_strategy_db()
                            data = db.get_all_priority_lists()

                            # Auto-populate if empty
                            if not data.get('populated'):
                                config = AUTONOMOUS_CONFIG
                                db.reset_priority_pairs(config["pairs"].get("binance", []))
                                db.reset_priority_periods(config["periods"])
                                db.reset_priority_timeframes(config["timeframes"])
                                db.reset_priority_granularities(config["granularities"])
                                data = db.get_all_priority_lists()

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
                            return data

                        priority_data = await loop.run_in_executor(thread_pool, fetch_priority_data)
                        await websocket.send_json({
                            "type": "priority_data",
                            "id": request_id,
                            "data": priority_data
                        })

                    elif msg_type == "get_db_stats":
                        # Get database stats for Tools page - cached and optimized
                        def fetch_db_stats():
                            cached = stats_cache.get(CacheKeys.DB_STATS)
                            if cached is not None:
                                return cached
                            db = get_strategy_db()
                            # Use SQL aggregation - much faster than loading all strategies
                            data = db.get_db_stats_optimized()
                            stats_cache.set(CacheKeys.DB_STATS, data, ttl=60)
                            return data

                        db_stats = await loop.run_in_executor(thread_pool, fetch_db_stats)
                        await websocket.send_json({
                            "type": "db_stats_data",
                            "id": request_id,
                            "data": db_stats
                        })

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
                        await websocket.send_json({
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
                        })

                except json.JSONDecodeError:
                    pass  # Ignore non-JSON messages

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
