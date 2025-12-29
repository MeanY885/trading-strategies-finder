"""
ELITE VALIDATOR SERVICE
=======================
Background service that validates strategies across multiple time periods
to determine their consistency and reliability.

Features:
- Multi-period validation (1 week to 5 years)
- Period boundary detection (only re-validates when calendar boundaries crossed)
- Parallel processing with configurable concurrency
- Elite scoring based on consistency and profitability
"""
import asyncio
import json
import threading
import psutil
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from config import VALIDATION_PERIODS
from state import app_state, concurrency_config
from logging_config import log
from services.autonomous_optimizer import has_period_boundary_crossed, get_stale_periods
from services.cache import invalidate_counts_cache

# =============================================================================
# PARALLEL VALIDATION SUPPORT (Resource-Aware)
# =============================================================================

# Track running validations for queue display
running_validations: Dict[int, dict] = {}
running_validations_lock = threading.Lock()
running_validations_async_lock: Optional[asyncio.Lock] = None

# Pending strategies list for queue display
pending_strategies_list: List[dict] = []

# =============================================================================
# RESOURCE THRESHOLDS FOR ELITE VALIDATION
# =============================================================================
# These thresholds ensure we reserve resources for critical shared services:
#
# RESERVED RESOURCES (always kept free):
#   - PostgreSQL Database: ~300-500MB RAM, variable CPU for queries
#   - Nginx Frontend: ~50MB RAM, minimal CPU
#   - System overhead: ~500MB RAM
#   - WebSocket connections: ~100MB RAM
#   Total reserved: ~1.0-1.5GB minimum
#
# The thresholds below ensure these services always have resources available.
# =============================================================================

ELITE_CPU_THRESHOLD = 70      # Only spawn if CPU < 70% (reserve 30% for DB queries, system)
ELITE_MEM_THRESHOLD = 3.0     # Only spawn if > 3GB available (reserves ~1.5GB for DB/Frontend/System)
ELITE_MEM_PER_VALIDATION = 0.5  # Each validation task uses ~500MB
ELITE_BASE_CONCURRENT = 2     # Base concurrent validations
ELITE_MAX_CONCURRENT = 4      # Maximum concurrent validations

# Reserved memory breakdown (in GB) - used for calculations
RESERVED_FOR_DATABASE = 0.5   # PostgreSQL shared buffers, work mem, connections
RESERVED_FOR_FRONTEND = 0.1   # Nginx + static file serving
RESERVED_FOR_WEBSOCKET = 0.2  # WebSocket connections and message queues
RESERVED_FOR_SYSTEM = 0.5     # OS, Docker overhead, buffers
TOTAL_RESERVED_MEMORY = RESERVED_FOR_DATABASE + RESERVED_FOR_FRONTEND + RESERVED_FOR_WEBSOCKET + RESERVED_FOR_SYSTEM


def get_max_concurrent_validations() -> int:
    """
    Determine max concurrent validations based on system resources
    AND awareness of other running services (Autonomous Optimizer, etc).

    Memory calculation:
        available_for_validation = mem_available - TOTAL_RESERVED_MEMORY
        memory_based_max = available_for_validation / ELITE_MEM_PER_VALIDATION

    This ensures Database, Frontend, WebSocket, and System always have resources.
    """
    from services.resource_monitor import resource_monitor

    # Check config override first
    config_max = concurrency_config.get("elite_max_concurrent", 0)
    if config_max > 0:
        return config_max

    # Get current resource state
    resources = resource_monitor.get_current_resources()
    cpu_percent = resources["cpu_percent"]
    mem_available_gb = resources["memory_available_gb"]
    cpu_cores = resources["cpu_cores"]

    # Check if Autonomous Optimizer is running
    autonomous_running = app_state.is_autonomous_running()
    autonomous_parallel = 0
    if autonomous_running:
        auto_status = app_state.get_autonomous_status()
        autonomous_parallel = auto_status.get("parallel_count", 0) or len(auto_status.get("parallel_running", []))

    # Check if manual optimizer is running
    optimizer_running = app_state.is_running()

    # =================================================================
    # MEMORY-BASED CALCULATION (respects reserved resources)
    # =================================================================
    # Calculate memory available for validation after reserving for shared services
    available_for_validation = mem_available_gb - TOTAL_RESERVED_MEMORY

    # If Autonomous Optimizer is running, reserve additional memory for it
    if autonomous_running and autonomous_parallel > 0:
        # Assume each autonomous task uses ~0.8GB
        autonomous_mem_usage = autonomous_parallel * 0.8
        available_for_validation -= autonomous_mem_usage

    # Calculate max validations based on available memory
    if available_for_validation > ELITE_MEM_PER_VALIDATION:
        memory_based_max = int(available_for_validation / ELITE_MEM_PER_VALIDATION)
    else:
        memory_based_max = 1  # Always allow at least 1 if any memory available

    # =================================================================
    # CPU-BASED CALCULATION
    # =================================================================
    if cpu_cores >= 16:
        cpu_based_max = 4
    elif cpu_cores >= 8:
        cpu_based_max = 3
    elif cpu_cores >= 4:
        cpu_based_max = 2
    else:
        cpu_based_max = 1

    # Reduce CPU-based max for other running services
    if autonomous_running:
        # Reserve CPU for autonomous optimizer tasks + 1 for overhead
        reduction = min(autonomous_parallel + 1, cpu_based_max - 1)
        cpu_based_max = max(1, cpu_based_max - reduction)
        log(f"[Elite Validation] Autonomous running ({autonomous_parallel} parallel), CPU max reduced to {cpu_based_max}")

    if optimizer_running:
        # Manual optimizer running - reduce further
        cpu_based_max = max(1, cpu_based_max - 1)
        log(f"[Elite Validation] Manual optimizer running, CPU max reduced to {cpu_based_max}")

    # Further reduce if CPU is already high
    if cpu_percent > ELITE_CPU_THRESHOLD:
        cpu_based_max = max(1, cpu_based_max - 1)

    # =================================================================
    # FINAL CALCULATION: Use minimum of memory and CPU limits
    # =================================================================
    final_max = min(memory_based_max, cpu_based_max, ELITE_MAX_CONCURRENT)
    final_max = max(1, final_max)  # Always allow at least 1

    log(f"[Elite Validation] Max concurrent: {final_max} (mem_based={memory_based_max}, cpu_based={cpu_based_max}, available_mem={available_for_validation:.1f}GB)")

    return final_max


def can_spawn_validation() -> tuple[bool, str]:
    """
    Check if we can spawn a new validation task based on current resources.
    Returns (can_spawn, reason).

    Memory check ensures:
        available_mem > TOTAL_RESERVED_MEMORY + ELITE_MEM_PER_VALIDATION
    This guarantees Database, Frontend, WebSocket, and System have their resources.
    """
    from services.resource_monitor import resource_monitor

    resources = resource_monitor.get_current_resources()
    cpu_percent = resources["cpu_percent"]
    mem_available_gb = resources["memory_available_gb"]

    # Check CPU - reserve headroom for DB queries and system
    if cpu_percent > ELITE_CPU_THRESHOLD:
        return False, f"CPU too high ({cpu_percent:.0f}% > {ELITE_CPU_THRESHOLD}%)"

    # =================================================================
    # MEMORY CHECK: Ensure reserved resources are protected
    # =================================================================
    # Calculate minimum memory needed = reserved + 1 validation task
    min_memory_needed = TOTAL_RESERVED_MEMORY + ELITE_MEM_PER_VALIDATION

    if mem_available_gb < min_memory_needed:
        return False, f"Memory too low ({mem_available_gb:.1f}GB < {min_memory_needed:.1f}GB needed)"

    # Additional check: ensure we have enough for current running + 1 more
    current_running = len(running_validations)
    memory_for_all_tasks = TOTAL_RESERVED_MEMORY + ((current_running + 1) * ELITE_MEM_PER_VALIDATION)

    if mem_available_gb < memory_for_all_tasks:
        return False, f"Insufficient memory for additional task ({mem_available_gb:.1f}GB < {memory_for_all_tasks:.1f}GB)"

    # Check current running count vs max
    max_concurrent = get_max_concurrent_validations()

    if current_running >= max_concurrent:
        return False, f"Max concurrent reached ({current_running}/{max_concurrent})"

    return True, "OK"


def init_elite_async_primitives():
    """Initialize async primitives for parallel validation."""
    global running_validations_async_lock

    running_validations_async_lock = asyncio.Lock()

    max_concurrent = get_max_concurrent_validations()
    log(f"[Elite Validation] Initialized: max_concurrent={max_concurrent} (resource-aware)")


# =============================================================================
# VALIDATION STATUS
# =============================================================================

def get_elite_status() -> dict:
    """Get current elite validation status."""
    return app_state.get_elite_status()


# =============================================================================
# ELITE VALIDATION
# =============================================================================

async def validate_strategy(
    strategy: dict,
    validation_periods: List[dict]
) -> dict:
    """
    Validate a single strategy across multiple time periods.

    Returns:
        dict with validation results including elite_status, score, periods passed
    """
    from data_fetcher import BinanceDataFetcher, YFinanceDataFetcher
    from strategy_engine import StrategyEngine

    strategy_id = strategy.get('id')
    strategy_name = strategy.get('strategy_name', 'Unknown')

    # Extract parameters
    params = strategy.get('params', {})
    symbol = strategy.get('symbol', 'BTCGBP')
    data_source = strategy.get('data_source')
    timeframe = strategy.get('timeframe', '15m')
    tp_percent = strategy.get('tp_percent', 2.0)
    sl_percent = strategy.get('sl_percent', 5.0)
    entry_rule = params.get('entry_rule', 'rsi_oversold')
    direction = params.get('direction', strategy.get('trade_mode', 'long'))

    # Check for unsupported pairs
    supported_quotes = ['USDT', 'USDC', 'BUSD']
    if not any(symbol.endswith(q) for q in supported_quotes):
        return {
            "elite_status": "skipped",
            "periods_passed": 0,
            "periods_total": 0,
            "elite_score": 0,
            "validation_data": json.dumps({
                "status": "skipped",
                "reason": f"Symbol {symbol} not supported - Binance USDT pairs only"
            })
        }

    # Detect data source
    if not data_source:
        if '-' in symbol:
            data_source = 'yahoo'
        elif symbol.endswith('USDT') or symbol.endswith('BUSD'):
            data_source = 'binance'
        else:
            data_source = 'binance'

    # Convert timeframe to minutes
    if 'h' in timeframe:
        tf_minutes = int(timeframe.replace('h', '')) * 60
    else:
        tf_minutes = int(timeframe.replace('m', ''))

    # Data source limits
    if data_source and 'yahoo' in data_source.lower():
        data_limits = {1: 7, 5: 60, 15: 60, 30: 60, 60: 730, 1440: 9999}
    else:
        data_limits = {
            1: 365, 5: 1825, 15: 2555, 30: 2555,
            60: 2555, 240: 2555, 1440: 3650,
        }
    max_days = data_limits.get(tf_minutes, 1825)

    # Original metrics
    original_metrics = {
        "win_rate": strategy.get('win_rate', 0),
        "profit_factor": strategy.get('profit_factor', 0),
    }

    # Load existing validation data
    existing_validation_data = strategy.get('elite_validation_data')
    existing_results = []
    if existing_validation_data:
        try:
            existing_results = json.loads(existing_validation_data) if isinstance(existing_validation_data, str) else existing_validation_data
        except:
            existing_results = []

    # Determine stale periods
    stale_periods = get_stale_periods(existing_results, validation_periods)
    stale_period_names = [p["period"] for p in stale_periods]

    if not stale_periods and existing_results:
        log(f"[Elite Validation] Strategy #{strategy_id}: All periods fresh, skipping")
        return None  # No update needed

    passed = 0
    total_testable = 0
    results = []

    # Process each period
    for vp in validation_periods:
        # Keep fresh results
        if vp["period"] not in stale_period_names and existing_results:
            for er in existing_results:
                if er.get("period") == vp["period"]:
                    results.append(er)
                    if er.get("status") not in ["limit_exceeded", "insufficient_data", "error"]:
                        total_testable += 1
                        if er.get("status") in ['consistent', 'minor_drop']:
                            passed += 1
                    break
            continue

        app_state.update_elite_status(
            message=f"Validating: {strategy_name} ({vp['period']})"
        )

        if vp["days"] > max_days:
            results.append({
                "period": vp["period"],
                "status": "limit_exceeded",
                "validated_at": datetime.now().isoformat()
            })
            continue

        try:
            # Fetch data
            if data_source and 'binance' in data_source.lower():
                fetcher = BinanceDataFetcher()
            else:
                fetcher = YFinanceDataFetcher()

            df = await fetcher.fetch_ohlcv(pair=symbol, interval=tf_minutes, months=vp["months"])

            if len(df) < 50:
                results.append({
                    "period": vp["period"],
                    "status": "insufficient_data",
                    "validated_at": datetime.now().isoformat()
                })
                continue

            # Run backtest
            engine = StrategyEngine(df)
            result = engine.backtest(
                strategy=entry_rule,
                direction=direction,
                tp_percent=tp_percent,
                sl_percent=sl_percent,
                initial_capital=1000.0,
                position_size_pct=100.0,
                commission_pct=0.1
            )

            # Determine status
            status = "consistent"
            if result.total_trades == 0:
                status = "no_trades"
            elif result.win_rate < original_metrics["win_rate"] * 0.8:
                status = "degraded"
            elif result.profit_factor < original_metrics["profit_factor"] * 0.8:
                status = "degraded"
            elif result.win_rate < original_metrics["win_rate"] * 0.95:
                status = "minor_drop"

            total_testable += 1
            if status in ['consistent', 'minor_drop']:
                passed += 1

            return_pct = round((result.total_pnl / 1000.0) * 100, 1)

            results.append({
                "period": vp["period"],
                "status": status,
                "trades": result.total_trades,
                "win_rate": round(result.win_rate, 2),
                "pnl": round(result.total_pnl, 2),
                "return_pct": return_pct,
                "validated_at": datetime.now().isoformat()
            })

        except Exception as e:
            results.append({
                "period": vp["period"],
                "status": "error",
                "message": str(e),
                "validated_at": datetime.now().isoformat()
            })

    # Calculate elite score
    consistency_points = passed
    total_pnl = sum(
        r.get('pnl', 0) for r in results
        if r.get('status') in ['consistent', 'minor_drop'] and r.get('pnl', 0) > 0
    )
    profit_bonus = total_pnl / 100
    elite_score = consistency_points + profit_bonus

    # Determine final status
    if total_testable == 0:
        elite_status = 'untestable'
    elif passed == total_testable:
        elite_status = 'elite'
    elif passed >= total_testable * 0.7:
        elite_status = 'partial'
    else:
        elite_status = 'failed'

    return {
        "elite_status": elite_status,
        "periods_passed": passed,
        "periods_total": total_testable,
        "elite_score": elite_score,
        "validation_data": json.dumps(results)
    }


async def validate_single_strategy_worker(strategy: dict, db, processed_count: list, task_slot: asyncio.Semaphore):
    """
    Worker function to validate a single strategy with resource-aware control.
    Uses a dynamic semaphore that respects other running services.
    """
    global running_validations, pending_strategies_list
    from services.websocket_manager import broadcast_elite_status

    strategy_id = strategy.get('id')
    strategy_name = strategy.get('strategy_name', 'Unknown')
    symbol = strategy.get('symbol', '')
    timeframe = strategy.get('timeframe', '')

    # Wait for a slot - semaphore is dynamically sized
    async with task_slot:
        # Additional resource check before starting
        can_spawn, reason = can_spawn_validation()
        while not can_spawn:
            log(f"[Elite Validation] Waiting for resources: {reason}")
            await asyncio.sleep(5)
            can_spawn, reason = can_spawn_validation()

        # Add to running validations
        async with running_validations_async_lock:
            running_validations[strategy_id] = {
                "id": strategy_id,
                "name": strategy_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "validating",
                "progress": 0
            }
            # Remove from pending list if present
            pending_strategies_list[:] = [s for s in pending_strategies_list if s.get('id') != strategy_id]

            # Update status with queue info
            _update_queue_status()

        log(f"[Elite Validation] Validating: {strategy_name} (#{strategy_id})")

        try:
            result = await validate_strategy(strategy, VALIDATION_PERIODS)

            if result:
                db.update_elite_status(
                    strategy_id=strategy_id,
                    elite_status=result["elite_status"],
                    periods_passed=result["periods_passed"],
                    periods_total=result["periods_total"],
                    validation_data=result["validation_data"],
                    elite_score=result["elite_score"]
                )
                invalidate_counts_cache()
                log(f"[Elite Validation] Result: {result['elite_status'].upper()} - {result['periods_passed']}/{result['periods_total']} periods")

        except Exception as e:
            log(f"[Elite Validation] Error validating {strategy_name}: {e}", level='ERROR')

        finally:
            # Remove from running validations
            async with running_validations_async_lock:
                if strategy_id in running_validations:
                    del running_validations[strategy_id]

            # Update processed count
            processed_count[0] += 1
            app_state.update_elite_status(
                processed=processed_count[0]
            )
            _update_queue_status()
            broadcast_elite_status(app_state.get_elite_status())


def _update_queue_status():
    """Update elite status with current queue information and resource state."""
    global running_validations, pending_strategies_list
    from services.resource_monitor import resource_monitor

    running_list = list(running_validations.values())
    pending_preview = pending_strategies_list[:5]  # Next 5 pending

    # Get current resource state
    resources = resource_monitor.get_current_resources()
    mem_available = resources["memory_available_gb"]

    # Calculate available memory for validation (after reserved)
    mem_for_validation = max(0, mem_available - TOTAL_RESERVED_MEMORY)

    app_state.update_elite_status(
        running_validations=running_list,
        pending_queue=pending_preview,
        parallel_count=len(running_list),
        max_parallel=get_max_concurrent_validations(),
        cpu_percent=resources["cpu_percent"],
        memory_available_gb=mem_available,
        memory_for_validation_gb=round(mem_for_validation, 1),
        reserved_memory_gb=TOTAL_RESERVED_MEMORY
    )


async def validate_all_strategies():
    """
    Background task: Validate ALL pending strategies with resource-aware parallel processing.
    Dynamically adjusts concurrency based on CPU, memory, and other running services.
    """
    global pending_strategies_list, running_validations
    from strategy_database import get_strategy_db
    from services.resource_monitor import resource_monitor

    # Initialize async primitives if needed
    if running_validations_async_lock is None:
        init_elite_async_primitives()

    try:
        db = get_strategy_db()

        # Use optimized query instead of loading all strategies
        pending = db.get_strategies_pending_validation(limit=1000)

        if not pending:
            app_state.update_elite_status(message="No pending strategies to validate")
            return

        # Sort by priority if available
        try:
            priority_list = db.get_priority_list()
            if priority_list:
                priority_lookup = {}
                for item in priority_list:
                    if item['enabled']:
                        key = (item['pair'], item['timeframe_label'])
                        if key not in priority_lookup:
                            priority_lookup[key] = item['position']

                def get_priority(s):
                    key = (s.get('symbol', ''), s.get('timeframe', ''))
                    return priority_lookup.get(key, 999999)

                pending.sort(key=get_priority)
        except:
            pass

        # Store pending list for queue display
        pending_strategies_list = [{
            "id": s.get('id'),
            "name": s.get('strategy_name', 'Unknown'),
            "symbol": s.get('symbol', ''),
            "timeframe": s.get('timeframe', ''),
            "status": "pending"
        } for s in pending]

        # Get resource-aware max concurrent (considers Autonomous Optimizer, etc.)
        max_concurrent = get_max_concurrent_validations()
        resources = resource_monitor.get_current_resources()

        app_state.update_elite_status(
            running=True,
            total=len(pending),
            processed=0,
            max_parallel=max_concurrent,
            cpu_percent=resources["cpu_percent"],
            memory_available_gb=resources["memory_available_gb"],
            message=f"Validating {len(pending)} strategies ({max_concurrent} parallel, resource-aware)..."
        )

        from services.websocket_manager import broadcast_elite_status
        _update_queue_status()
        broadcast_elite_status(app_state.get_elite_status())

        # Use shared counter for processed count
        processed_count = [0]

        # Create dynamic semaphore - will be checked against current resources
        task_slot = asyncio.Semaphore(max_concurrent)

        # Create tasks for all strategies (pass semaphore for resource control)
        tasks = [
            validate_single_strategy_worker(strategy, db, processed_count, task_slot)
            for strategy in pending
        ]

        # Run all tasks (semaphore controls concurrency)
        await asyncio.gather(*tasks, return_exceptions=True)

        app_state.update_elite_status(
            message=f"Complete! Validated {processed_count[0]} strategies"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        app_state.update_elite_status(message=f"Error: {str(e)}")

    finally:
        # Clear queue tracking
        running_validations.clear()
        pending_strategies_list.clear()

        app_state.update_elite_status(
            running=False,
            paused=False,
            current_strategy_id=None,
            running_validations=[],
            pending_queue=[],
            parallel_count=0
        )
        from services.websocket_manager import broadcast_elite_status
        broadcast_elite_status(app_state.get_elite_status())


async def start_auto_elite_validation():
    """
    Continuous background loop that automatically validates strategies.

    PARALLEL MODE: Runs alongside the optimizer using shared resources.
    Checks for new pending strategies every 60 seconds.
    """
    if app_state.is_elite_auto_running():
        return

    app_state.update_elite_status(auto_running=True)
    log("[Elite Validation] Starting continuous background validation...")

    while app_state.is_elite_auto_running():
        try:
            # Check parallel mode
            if concurrency_config.get("elite_parallel", True):
                app_state.update_elite_status(paused=False)
            else:
                while app_state.is_unified_running():
                    app_state.update_elite_status(
                        message="Waiting for optimizer to finish...",
                        paused=True
                    )
                    await asyncio.sleep(5)
                app_state.update_elite_status(paused=False)

            log("[Elite Validation] Checking for pending strategies...")

            from strategy_database import get_strategy_db
            db = get_strategy_db()

            # Use optimized query to get pending count first
            pending_count = db.get_pending_validation_count()

            if pending_count > 0:
                log(f"[Elite Validation] Found {pending_count} pending strategies")
                app_state.update_elite_status(
                    message=f"Found {pending_count} pending strategies to validate"
                )
                await validate_all_strategies()
                await asyncio.sleep(5)
            else:
                # Check for stale periods - use optimized query for validated strategies only
                validated = db.get_elite_strategies_filtered(status_filter=None, limit=500)
                validated = [s for s in validated if s.get('elite_validated_at')]
                strategies_with_stale = []

                for s in validated:
                    existing_data = s.get('elite_validation_data')
                    existing_results = []
                    if existing_data:
                        try:
                            existing_results = json.loads(existing_data) if isinstance(existing_data, str) else existing_data
                        except:
                            existing_results = []

                    stale = get_stale_periods(existing_results, VALIDATION_PERIODS)
                    if stale:
                        strategies_with_stale.append({
                            'strategy': s,
                            'stale_count': len(stale)
                        })

                if strategies_with_stale:
                    strategies_with_stale.sort(key=lambda x: -x['stale_count'])
                    top = strategies_with_stale[0]
                    strategy = top['strategy']

                    log(f"[Elite Validation] Re-validating {strategy['strategy_name']}: {top['stale_count']} stale periods")

                    db.update_elite_status(
                        strategy_id=strategy['id'],
                        elite_status='pending',
                        periods_passed=strategy.get('elite_periods_passed', 0),
                        periods_total=strategy.get('elite_periods_total', 0),
                        validation_data=strategy.get('elite_validation_data'),
                        elite_score=strategy.get('elite_score', 0)
                    )
                    # Invalidate cache after elite status update
                    invalidate_counts_cache()

                    await validate_all_strategies()
                    await asyncio.sleep(10)
                else:
                    app_state.update_elite_status(
                        message="All periods fresh, waiting..."
                    )
                    log(f"[Elite Validation] All {len(validated)} strategies fresh, checking in 1 hour")
                    await asyncio.sleep(3600)

        except Exception as e:
            log(f"[Elite Validation] Error: {e}", level='ERROR')
            app_state.update_elite_status(message=f"Error: {str(e)}")
            await asyncio.sleep(30)

    log("[Elite Validation] Stopped")


async def stop_elite_validation():
    """Stop the elite validation service."""
    app_state.update_elite_status(
        auto_running=False,
        running=False,
        paused=False,
        message="Stopped"
    )

    from services.websocket_manager import broadcast_elite_status
    broadcast_elite_status(app_state.get_elite_status())

    log("[Elite Validation] Stop signal sent")
