"""
ELITE VALIDATOR SERVICE
=======================
Background service that validates strategies across multiple time periods
to determine their consistency and reliability.

Features:
- Multi-period validation (1 week to 2 years)
- Period boundary detection (only re-validates when calendar boundaries crossed)
- Parallel processing with configurable concurrency
- Elite scoring based on consistency and profitability
"""
import asyncio
import json
import time
import psutil
from datetime import datetime
from typing import Dict, List, Optional

from config import VALIDATION_PERIODS
from state import app_state, concurrency_config
from logging_config import log
from services.autonomous_optimizer import has_period_boundary_crossed, get_stale_periods
from services.task_watchdog import TimeoutCalculator
from services.progress_watchdog import ProgressBasedWatchdog, notify_task_completed
from services.cache import invalidate_counts_cache
from async_database import AsyncDatabase
from services.vectorbt_engine import VectorBTEngine, is_vectorbt_available

# =============================================================================
# PARALLEL VALIDATION SUPPORT (Resource-Aware)
# =============================================================================

# Track running validations for queue display
running_validations: Dict[int, dict] = {}
# Note: threading.Lock removed - using async lock exclusively for async context
running_validations_async_lock: Optional[asyncio.Lock] = None

# Pending strategies list for queue display
pending_strategies_list: List[dict] = []

# Resource cache to avoid blocking psutil calls
_cached_resources = {"data": None, "timestamp": 0}
RESOURCE_CACHE_TTL = 2  # 2 second cache

# Timeout constants for validation
RESOURCE_WAIT_TIMEOUT = 300  # 5 minutes max wait for resources

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

ELITE_CPU_THRESHOLD = 75      # Only spawn if CPU < 75% (reserve 25% for DB queries, system)
# Note: Memory threshold now calculated dynamically using TOTAL_RESERVED_MEMORY + ELITE_MEM_PER_VALIDATION
ELITE_MEM_PER_VALIDATION = 0.5  # Each validation task uses ~500MB
ELITE_BASE_CONCURRENT = 2     # Base concurrent validations
ELITE_MAX_CONCURRENT = 8      # Maximum concurrent validations (increased from 4 for high-spec systems)

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
    optimizer_running = app_state.is_optimization_running()

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
    # CPU-BASED CALCULATION (scaled for high-spec systems)
    # =================================================================
    if cpu_cores >= 32:
        cpu_based_max = 8
    elif cpu_cores >= 16:
        cpu_based_max = 6
    elif cpu_cores >= 8:
        cpu_based_max = 4
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
    validation_periods: List[dict],
    progress_callback=None
) -> dict:
    """
    Validate a single strategy across multiple time periods.

    Args:
        strategy: Strategy dict with id, name, params, etc.
        validation_periods: List of periods to validate
        progress_callback: Optional async callback(period_index, period_name, total_periods)

    Returns:
        dict with validation results including elite_status, score, periods passed
    """
    from data_fetcher import BinanceDataFetcher
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

    # Always use Binance as data source (only USDT/USDC/BUSD pairs are supported)
    data_source = 'binance'

    # Convert timeframe to minutes
    if 'h' in timeframe:
        tf_minutes = int(timeframe.replace('h', '')) * 60
    else:
        tf_minutes = int(timeframe.replace('m', ''))

    # Binance data limits by timeframe (in days)
    data_limits = {
        1: 365, 5: 1825, 15: 2555, 30: 2555,
        60: 2555, 240: 2555, 1440: 3650,
    }
    max_days = data_limits.get(tf_minutes, 1825)

    # Note: We no longer compare to original metrics
    # New system: simply check if each period is profitable (PnL > 0)

    # Load existing validation data
    existing_validation_data = strategy.get('elite_validation_data')
    existing_results = []
    if existing_validation_data:
        try:
            existing_results = json.loads(existing_validation_data) if isinstance(existing_validation_data, str) else existing_validation_data
        except json.JSONDecodeError:
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

    # =================================================================
    # PROGRESS-BASED WATCHDOG SETUP (like Auto Strat)
    # NO time-based timeouts - uses measurement-count stall detection
    # =================================================================
    total_periods = len(validation_periods)
    watchdog_status = {
        "progress": 0,
        "running": True,
        "abort": False,
        "current_period": "",
        "periods_completed": 0,
        "total_periods": total_periods
    }

    # Use extended mode for longer periods (6m+, 1yr, 2yr)
    # These have more data to process and need higher thresholds
    has_long_periods = any(vp.get("days", 0) >= 180 for vp in validation_periods)

    # Create progress-based watchdog for this validation
    watchdog_task_id = f"elite_{strategy_id}_{symbol}"
    watchdog = ProgressBasedWatchdog(
        task_id=watchdog_task_id,
        status_dict=watchdog_status,
        total_combinations=total_periods,  # Each period is a "combination"
        progress_key="progress",
        abort_key="abort",
        running_key="running",
        extended_mode=has_long_periods  # Use extended thresholds for long periods
    )

    # Start watchdog in background
    watchdog_task = asyncio.create_task(watchdog.start())

    try:
        # Process each period
        for period_idx, vp in enumerate(validation_periods):
            # Check if watchdog triggered abort
            if watchdog_status.get("abort"):
                log(f"[Elite Validation] Watchdog aborted validation for {strategy_name}", level='WARNING')
                break

            # Update progress for watchdog (0-100 scale)
            # Each period has 3 phases: start (0%), data fetch (33%), backtest (66%), complete (100%)
            # So period progress = (period_idx + phase_fraction) / total_periods * 100
            phase_fraction = 0.0  # Start of period
            progress_pct = ((period_idx + phase_fraction) / total_periods) * 100
            watchdog_status["progress"] = progress_pct
            watchdog_status["current_period"] = vp["period"]
            watchdog_status["current_phase"] = "starting"
            watchdog_status["periods_completed"] = period_idx

            # Report progress via callback
            if progress_callback:
                try:
                    await progress_callback(period_idx, vp["period"], total_periods)
                except Exception:
                    pass

            # Keep fresh results
            if vp["period"] not in stale_period_names and existing_results:
                for er in existing_results:
                    if er.get("period") == vp["period"]:
                        results.append(er)
                        if er.get("status") not in ["limit_exceeded", "insufficient_data", "error", "no_trades"]:
                            total_testable += 1
                            if er.get("status") == 'profitable':
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
                # Update progress: data fetch phase (33% of period)
                phase_fraction = 0.1
                watchdog_status["progress"] = ((period_idx + phase_fraction) / total_periods) * 100
                watchdog_status["current_phase"] = "fetching_data"

                # Fetch data - NO timeout wrapper (progress-based watchdog handles stalls)
                fetcher = BinanceDataFetcher()

                # NO asyncio.wait_for() - progress-based watchdog monitors stalls
                df = await fetcher.fetch_ohlcv(pair=symbol, interval=tf_minutes, months=vp["months"])

                # Update progress: data fetched (40% of period)
                phase_fraction = 0.4
                watchdog_status["progress"] = ((period_idx + phase_fraction) / total_periods) * 100
                watchdog_status["current_phase"] = "data_ready"
                watchdog_status["candles_fetched"] = len(df)

                if len(df) < 50:
                    results.append({
                        "period": vp["period"],
                        "status": "insufficient_data",
                        "validated_at": datetime.now().isoformat()
                    })
                    continue

                # Check abort again after data fetch
                if watchdog_status.get("abort"):
                    log(f"[Elite Validation] Watchdog aborted after data fetch for {strategy_name}", level='WARNING')
                    break

                # Update progress: backtest starting (50% of period)
                phase_fraction = 0.5
                watchdog_status["progress"] = ((period_idx + phase_fraction) / total_periods) * 100
                watchdog_status["current_phase"] = "backtesting"

                # Run backtest - NO timeout wrapper (progress-based watchdog handles stalls)
                def run_backtest():
                    # Try VectorBT first for 100x faster backtesting
                    if is_vectorbt_available():
                        try:
                            vbt_engine = VectorBTEngine(
                                df,
                                initial_capital=1000.0,
                                position_size_pct=100.0,
                                commission_pct=0.1
                            )
                            vbt_result = vbt_engine.run_single_backtest(
                                strategy=entry_rule,
                                direction=direction,
                                tp_percent=tp_percent,
                                sl_percent=sl_percent
                            )
                            return vbt_result
                        except Exception as e:
                            log(f"[Elite Validation] VectorBT failed, falling back to StrategyEngine: {e}", level='WARNING')

                    # Fallback to standard StrategyEngine
                    engine = StrategyEngine(df)
                    return engine.backtest(
                        strategy=entry_rule,
                        direction=direction,
                        tp_percent=tp_percent,
                        sl_percent=sl_percent,
                        initial_capital=1000.0,
                        position_size_pct=100.0,
                        commission_pct=0.1
                    )

                loop = asyncio.get_running_loop()
                # NO asyncio.wait_for() - progress-based watchdog monitors stalls
                result = await loop.run_in_executor(None, run_backtest)

                # Update progress: backtest complete (90% of period)
                phase_fraction = 0.9
                watchdog_status["progress"] = ((period_idx + phase_fraction) / total_periods) * 100
                watchdog_status["current_phase"] = "processing_results"

                # Simple profitability check - no comparison to original metrics
                # A period passes if it made any profit (PnL > 0)
                if result.total_trades == 0:
                    status = "no_trades"
                elif result.total_pnl > 0:
                    status = "profitable"
                else:
                    status = "unprofitable"

                # Only count periods with actual trades as testable
                if status != "no_trades":
                    total_testable += 1
                    if status == 'profitable':
                        passed += 1

                return_pct = round((result.total_pnl / 1000.0) * 100, 1)

                results.append({
                    "period": vp["period"],
                    "status": status,
                    "trades": result.total_trades,
                    "win_rate": round(result.win_rate, 2),
                    "pnl": round(result.total_pnl, 2),
                    "return_pct": return_pct,
                    "max_drawdown": round(result.max_drawdown, 2),
                    "validated_at": datetime.now().isoformat()
                })

                # MEMORY CLEANUP: Explicitly free DataFrame and result after each period
                # This prevents memory accumulation when validating 8 periods sequentially
                del df
                del result
                import gc
                gc.collect()

            except Exception as e:
                results.append({
                    "period": vp["period"],
                    "status": "error",
                    "message": str(e),
                    "validated_at": datetime.now().isoformat()
                })

        # Mark progress as complete
        watchdog_status["progress"] = 100
        watchdog_status["periods_completed"] = total_periods

    finally:
        # Stop watchdog and notify coordinator
        watchdog_status["running"] = False
        await watchdog.stop()
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass
        await notify_task_completed(watchdog_task_id)

    # =================================================================
    # NEW ELITE SCORING SYSTEM
    # =================================================================
    # 1. Profitability points: +1 for each profitable period
    # 2. Avg return bonus: average return % / 10
    # 3. Drawdown bonus: +3 (<5%), +2 (<10%), +1 (<15%)
    # =================================================================

    # 1. Profitability points (1 per profitable period)
    profitability_points = passed

    # 2. Average return bonus
    profitable_returns = [r.get('return_pct', 0) for r in results if r.get('status') == 'profitable']
    avg_return = sum(profitable_returns) / len(profitable_returns) if profitable_returns else 0
    profit_bonus = avg_return / 10

    # 3. Drawdown bonus (based on worst drawdown across all tested periods)
    all_drawdowns = [r.get('max_drawdown', 100) for r in results if r.get('status') in ['profitable', 'unprofitable']]
    worst_drawdown = max(all_drawdowns) if all_drawdowns else 100

    if worst_drawdown < 5:
        drawdown_bonus = 3
    elif worst_drawdown < 10:
        drawdown_bonus = 2
    elif worst_drawdown < 15:
        drawdown_bonus = 1
    else:
        drawdown_bonus = 0

    elite_score = profitability_points + profit_bonus + drawdown_bonus

    # Simplified status: just 'validated' or 'untestable'
    # The score does the ranking - no need for elite/partial/failed categories
    if total_testable == 0:
        elite_status = 'untestable'
    else:
        elite_status = 'validated'

    return {
        "elite_status": elite_status,
        "periods_passed": passed,
        "periods_total": total_testable,
        "elite_score": elite_score,
        "validation_data": json.dumps(results)
    }


async def validate_single_strategy_worker(strategy: dict, processed_count: list, task_slot: asyncio.Semaphore):
    """
    Worker function to validate a single strategy with resource-aware control.
    Uses a dynamic semaphore that respects other running services.

    FIX: Only add to running_validations AFTER acquiring semaphore slot,
    so the count reflects actually executing tasks, not waiting ones.
    """
    global running_validations, pending_strategies_list
    from services.websocket_manager import broadcast_elite_status

    strategy_id = strategy.get('id')
    strategy_name = strategy.get('strategy_name', 'Unknown')
    symbol = strategy.get('symbol', '')
    timeframe = strategy.get('timeframe', '')
    tp_percent = strategy.get('tp_percent', 0)
    sl_percent = strategy.get('sl_percent', 0)
    composite_score = strategy.get('composite_score', 0)
    rank = strategy.get('rank', 0)

    # Wait for a slot - semaphore is dynamically sized
    async with task_slot:
        # Track start time for ETA calculation (AFTER acquiring slot)
        validation_start_time = time.time()

        # Add to running validations AFTER acquiring slot (not before)
        # This fixes the race condition where waiting tasks inflated the count
        async with running_validations_async_lock:
            running_validations[strategy_id] = {
                "id": strategy_id,
                "name": strategy_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "tp_sl": f"{tp_percent:.1f}/{sl_percent:.1f}",
                "score": round(composite_score, 1),
                "rank": rank,
                "status": "validating",
                "progress": 0,
                "start_time": validation_start_time,
                "estimated_remaining": None
            }
            # Remove from pending list if present
            pending_strategies_list[:] = [s for s in pending_strategies_list if s.get('id') != strategy_id]

            # Update status with queue info
            _update_queue_status()

        log(f"[Elite Validation] Validating: {strategy_name} (#{strategy_id})")

        # Progress callback to update running_validations with ETA
        async def on_period_progress(period_idx, period_name, total_periods):
            progress_pct = int((period_idx / total_periods) * 100) if total_periods > 0 else 0

            # Calculate ETA based on elapsed time and periods completed
            estimated_remaining = None
            if period_idx > 0:
                elapsed = time.time() - validation_start_time
                avg_per_period = elapsed / period_idx
                remaining_periods = total_periods - period_idx
                estimated_remaining = int(avg_per_period * remaining_periods)

            async with running_validations_async_lock:
                if strategy_id in running_validations:
                    running_validations[strategy_id]["progress"] = progress_pct
                    running_validations[strategy_id]["current_period"] = period_name
                    running_validations[strategy_id]["period_index"] = period_idx
                    running_validations[strategy_id]["total_periods"] = total_periods
                    running_validations[strategy_id]["estimated_remaining"] = estimated_remaining
            _update_queue_status()
            broadcast_elite_status(app_state.get_elite_status())

        try:
            result = await validate_strategy(strategy, VALIDATION_PERIODS, progress_callback=on_period_progress)

            if result:
                # Use async database to avoid blocking the event loop
                await AsyncDatabase.update_elite_status(
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
    global running_validations, pending_strategies_list, _cached_resources
    from services.resource_monitor import resource_monitor

    running_list = list(running_validations.values())
    pending_preview = pending_strategies_list[:5]  # Next 5 pending

    # Get current resource state with caching to avoid blocking psutil calls
    now = time.time()
    if now - _cached_resources["timestamp"] < RESOURCE_CACHE_TTL and _cached_resources["data"]:
        resources = _cached_resources["data"]
    else:
        resources = resource_monitor.get_current_resources()
        _cached_resources = {"data": resources, "timestamp": now}

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
    from services.resource_monitor import resource_monitor

    # Initialize async primitives if needed
    if running_validations_async_lock is None:
        init_elite_async_primitives()

    try:
        # Use async database to avoid blocking the event loop
        # Get top 3 strategies per (symbol, timeframe) combination
        pending = await AsyncDatabase.get_top_pending_per_market(top_n=3)

        if not pending:
            app_state.update_elite_status(message="No pending strategies to validate")
            return

        # Sort by priority if available
        try:
            priority_list = await AsyncDatabase.get_priority_list()
            if priority_list:
                priority_lookup = {}
                for item in priority_list:
                    if item.get('enabled'):
                        key = (item['pair'], item['timeframe_label'])
                        if key not in priority_lookup:
                            priority_lookup[key] = item['position']

                def get_priority(s):
                    key = (s.get('symbol', ''), s.get('timeframe', ''))
                    return priority_lookup.get(key, 999999)

                pending.sort(key=get_priority)
        except Exception as e:
            log(f"[Elite Validator] Error sorting pending strategies by priority: {e}", level='ERROR')

        # Store pending list for queue display (with richer info for better UI)
        pending_strategies_list = [{
            "id": s.get('id'),
            "name": s.get('strategy_name', 'Unknown'),
            "symbol": s.get('symbol', ''),
            "timeframe": s.get('timeframe', ''),
            "tp_sl": f"{s.get('tp_percent', 0):.1f}/{s.get('sl_percent', 0):.1f}",
            "score": round(s.get('composite_score', 0), 1),
            "rank": s.get('rank', 0),
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
            validate_single_strategy_worker(strategy, processed_count, task_slot)
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

            # Use async database to avoid blocking the event loop
            pending_count = await AsyncDatabase.get_pending_validation_count()

            if pending_count > 0:
                log(f"[Elite Validation] Found {pending_count} pending strategies")
                app_state.update_elite_status(
                    message=f"Found {pending_count} pending strategies to validate"
                )
                await validate_all_strategies()
                await asyncio.sleep(5)
            else:
                # Check for stale periods - use async query for validated strategies only
                validated = await AsyncDatabase.get_elite_strategies_filtered(status_filter=None, limit=500)
                validated = [s for s in validated if s.get('elite_validated_at')]
                strategies_with_stale = []

                for s in validated:
                    existing_data = s.get('elite_validation_data')
                    existing_results = []
                    if existing_data:
                        try:
                            existing_results = json.loads(existing_data) if isinstance(existing_data, str) else existing_data
                        except json.JSONDecodeError:
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

                    # Use async database to avoid blocking the event loop
                    await AsyncDatabase.update_elite_status(
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
