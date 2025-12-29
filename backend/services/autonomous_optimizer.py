"""
AUTONOMOUS OPTIMIZER SERVICE
=============================
Background service that cycles through all trading pair/timeframe/period combinations
and runs optimization for each.

Features:
- Parallel processing with configurable concurrency
- Adaptive scaling based on CPU/memory usage
- Smart resume from where it left off
- Period boundary detection for re-optimization
"""
import asyncio
import threading
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd

from config import AUTONOMOUS_CONFIG, MAX_HISTORY_SIZE
from state import app_state, concurrency_config
from logging_config import log


# =============================================================================
# MODULE STATE
# =============================================================================

# Semaphore for limiting parallel optimizations
optimization_semaphore: Optional[asyncio.Semaphore] = None

# Semaphore for data fetching (allow multiple concurrent fetches within rate limits)
# Binance allows ~10 req/sec, so 3 concurrent is safe
data_fetch_semaphore: Optional[asyncio.Semaphore] = None

# Running optimizations tracking (for parallel mode)
running_optimizations: Dict[str, dict] = {}
running_optimizations_lock = threading.Lock()  # For sync contexts
running_optimizations_async_lock: Optional[asyncio.Lock] = None  # For async contexts

# History of completed optimizations
autonomous_runs_history: List[dict] = []

# Reference to current optimization status (for abort signaling)
current_optimization_status: Optional[dict] = None


def init_async_primitives():
    """Initialize asyncio primitives. Must be called from async context."""
    global optimization_semaphore, data_fetch_semaphore, running_optimizations_async_lock

    from config import MAX_CONCURRENT_OPTIMIZATIONS, MAX_CONCURRENT_FETCHES
    import psutil

    # Initialize async lock for running_optimizations in async contexts
    running_optimizations_async_lock = asyncio.Lock()

    # Auto-detect if not specified, or use configured value
    if MAX_CONCURRENT_OPTIMIZATIONS > 0:
        max_concurrent = MAX_CONCURRENT_OPTIMIZATIONS
    else:
        # Auto-detect based on CPU cores
        cpu_count = psutil.cpu_count(logical=True) or 4
        if cpu_count >= 32:
            max_concurrent = cpu_count - 4
        elif cpu_count >= 16:
            max_concurrent = cpu_count - 2
        elif cpu_count >= 8:
            max_concurrent = cpu_count - 1
        else:
            max_concurrent = max(2, cpu_count)

    # Override from state config if set
    state_max = concurrency_config.get("max_concurrent", 0)
    if state_max > 0:
        max_concurrent = state_max

    optimization_semaphore = asyncio.Semaphore(max_concurrent)
    data_fetch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    mem = psutil.virtual_memory()
    log(f"[Autonomous Optimizer] Initialized: max_concurrent={max_concurrent}, fetch_concurrent={MAX_CONCURRENT_FETCHES}")
    log(f"[Autonomous Optimizer] System: {psutil.cpu_count()} cores, {mem.total / (1024**3):.1f} GB RAM ({mem.available / (1024**3):.1f} GB available)")


# =============================================================================
# PERIOD BOUNDARY DETECTION
# =============================================================================

def has_period_boundary_crossed(period: str, last_validated_at: str) -> bool:
    """
    Check if a period boundary has been crossed since last validation.

    Logic:
    - 1 week: New week started (crossed a Monday)
    - 2 weeks: 2 weeks have passed since last validation
    - 1 month: New month started
    - 3 months: New quarter started (Jan/Apr/Jul/Oct)
    - 6 months: Crossed Jan 1 or Jul 1
    - 9 months: 9 months have passed
    - 1 year: New year started (crossed Jan 1)
    - 2 years: 2 Jan 1sts have passed
    """
    if not last_validated_at:
        return True  # Never validated

    try:
        # Parse the validation timestamp
        if 'T' in last_validated_at:
            validated = datetime.fromisoformat(last_validated_at.replace('Z', '+00:00'))
            if validated.tzinfo:
                validated = validated.replace(tzinfo=None)
        else:
            validated = datetime.strptime(last_validated_at, '%Y-%m-%d %H:%M:%S')

        now = datetime.now()

        if period == "1 week":
            days_since_monday = now.weekday()
            this_monday = now - timedelta(days=days_since_monday)
            this_monday = this_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            return validated < this_monday

        elif period == "2 weeks":
            return (now - validated).days >= 14

        elif period == "1 month":
            return (now.year, now.month) != (validated.year, validated.month)

        elif period == "3 months":
            def get_quarter(dt):
                return (dt.year, (dt.month - 1) // 3)
            return get_quarter(now) != get_quarter(validated)

        elif period == "6 months":
            def get_half(dt):
                return (dt.year, 0 if dt.month < 7 else 1)
            return get_half(now) != get_half(validated)

        elif period == "9 months":
            months_diff = (now.year - validated.year) * 12 + (now.month - validated.month)
            return months_diff >= 9

        elif period in ["1 year", "2 years", "3 years", "5 years"]:
            return now.year != validated.year

        else:
            return True  # Unknown period, always refresh

    except Exception as e:
        log(f"[Period Boundary] Error parsing date: {e}", level='WARNING')
        return True


def get_stale_periods(validation_data: dict, validation_periods: list) -> list:
    """Return list of periods that need refreshing."""
    if not validation_data:
        return validation_periods

    stale = []
    for vp in validation_periods:
        period_name = vp["period"]

        period_result = None
        if isinstance(validation_data, list):
            for result in validation_data:
                if result.get("period") == period_name:
                    period_result = result
                    break

        if not period_result:
            stale.append(vp)
            continue

        validated_at = period_result.get("validated_at")
        if has_period_boundary_crossed(period_name, validated_at):
            stale.append(vp)

    return stale


# =============================================================================
# COMBINATION BUILDING
# =============================================================================

def build_optimization_combinations() -> List[dict]:
    """
    Build priority-ordered list of all optimization combinations.

    Priority order:
    1. NEW 4-list system (from database priority tables)
    2. Legacy priority queue
    3. Fallback: Hardcoded order (Granularity -> Timeframe -> Period -> Pair)
    """
    config = AUTONOMOUS_CONFIG

    try:
        from strategy_database import get_strategy_db
        db = get_strategy_db()

        # Try NEW 4-list priority system first
        if db.has_priority_lists_populated():
            pairs = db.get_enabled_priority_pairs()
            periods = db.get_enabled_priority_periods()
            timeframes = db.get_enabled_priority_timeframes()
            granularities = db.get_enabled_priority_granularities()

            if not granularities:
                granularities = [{"label": g["label"], "n_trials": g["n_trials"]}
                                 for g in config["granularities"]]

            if pairs and periods and timeframes and granularities:
                combinations = []
                for gran in granularities:
                    for tf in timeframes:
                        for period in periods:
                            for pair in pairs:
                                combinations.append({
                                    "source": "binance",
                                    "pair": pair["value"],
                                    "period": {"label": period["label"], "months": period["months"]},
                                    "timeframe": {"label": tf["label"], "minutes": tf["minutes"]},
                                    "granularity": {"label": gran["label"], "n_trials": gran["n_trials"]}
                                })

                log(f"[Autonomous Optimizer] Built {len(combinations)} combinations from priority system")
                return combinations

        # Try legacy single priority list
        priority_combos = db.get_enabled_priority_combinations()
        if priority_combos:
            combinations = [{
                "source": combo["source"],
                "pair": combo["pair"],
                "period": combo["period"],
                "timeframe": combo["timeframe"],
                "granularity": combo["granularity"],
            } for combo in priority_combos]

            log(f"[Autonomous Optimizer] Using {len(combinations)} priority items from legacy database")
            return combinations

    except Exception as e:
        log(f"[Autonomous Optimizer] Error loading priority list: {e}, using defaults", level='WARNING')

    # Fallback: Hardcoded order
    combinations = []
    for granularity in config["granularities"]:
        for timeframe in config["timeframes"]:
            for period in config["periods"]:
                for source in config["sources"]:
                    pairs = config["pairs"].get(source, [])
                    for pair in pairs:
                        combinations.append({
                            "source": source,
                            "pair": pair,
                            "period": period,
                            "timeframe": timeframe,
                            "granularity": granularity,
                        })

    log(f"[Autonomous Optimizer] Using default order: {len(combinations)} combinations")
    return combinations


def find_resume_index(combinations: list, db) -> int:
    """Find the index to resume from by checking completed optimizations."""
    completed = db.get_completed_optimizations(with_timestamps=True)
    log(f"[Resume] Found {len(completed)} completed optimization records in database")

    skipped_fresh = 0
    for i, combo in enumerate(combinations):
        key = (
            combo['pair'],
            combo['period']['label'],
            combo['timeframe']['label'],
            combo['granularity']['label']
        )

        if key not in completed:
            return i

        completed_at = completed[key]
        period_label = combo['period']['label']

        if has_period_boundary_crossed(period_label, completed_at):
            log(f"[Resume] {combo['pair']} {period_label} needs re-optimization (period boundary crossed)")
            return i

        skipped_fresh += 1

    log(f"[Resume] All {skipped_fresh} combinations are still fresh (within period boundaries)")
    return 0


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data_range(df: pd.DataFrame, period: dict, timeframe: dict) -> dict:
    """Validate that fetched data covers expected date range."""
    try:
        if df is None or len(df) == 0:
            return {
                "valid": False,
                "expected_days": 0,
                "actual_days": 0,
                "coverage_pct": 0,
                "message": "No data available"
            }

        expected_days = period["months"] * 30

        if 'time' in df.columns:
            start_time = pd.to_datetime(df['time'].min())
            end_time = pd.to_datetime(df['time'].max())
        else:
            start_time = pd.to_datetime(df.index.min())
            end_time = pd.to_datetime(df.index.max())

        actual_days = (end_time - start_time).total_seconds() / 86400
        coverage_pct = (actual_days / expected_days) * 100 if expected_days > 0 else 0

        candles_per_day = 1440 / timeframe["minutes"]
        expected_candles = int(expected_days * candles_per_day)
        actual_candles = len(df)

        is_valid = coverage_pct >= 90

        if is_valid:
            message = f"OK: {actual_days:.1f} days ({actual_candles:,} candles)"
        else:
            message = f"Expected ~{expected_days:.0f} days, got {actual_days:.1f} days ({coverage_pct:.0f}%)"

        return {
            "valid": is_valid,
            "expected_days": round(expected_days, 1),
            "actual_days": round(actual_days, 1),
            "coverage_pct": round(coverage_pct, 1),
            "expected_candles": expected_candles,
            "actual_candles": actual_candles,
            "start_date": start_time.strftime("%Y-%m-%d %H:%M"),
            "end_date": end_time.strftime("%Y-%m-%d %H:%M"),
            "message": message
        }
    except Exception as e:
        return {
            "valid": False,
            "message": f"Validation error: {str(e)}"
        }


# =============================================================================
# OPTIMIZATION RUNNER
# =============================================================================

def run_optimization_sync(
    df: pd.DataFrame,
    combo: dict,
    status: dict,
    run_strategy_finder_func: Callable,
    thread_pool
):
    """
    Synchronous optimization that runs in thread pool.
    """
    try:
        config = AUTONOMOUS_CONFIG

        from strategy_engine import run_strategy_finder as rsf
        report = rsf(
            df=df,
            status=status,
            streaming_callback=None,
            symbol=combo["pair"],
            timeframe=combo["timeframe"]["label"],
            exchange="BINANCE",
            capital=config["capital"],
            position_size_pct=config["position_size_pct"],
            engine="tradingview",
            n_trials=combo["granularity"]["n_trials"],
            progress_min=30,
            progress_max=95,
            source_currency="USD",
            fx_fetcher=None
        )
        status["report"] = report

    except Exception as e:
        import traceback
        traceback.print_exc()
        status["message"] = f"Error: {str(e)}"


async def run_single_optimization(
    combo: dict,
    combo_id: str,
    thread_pool
) -> str:
    """
    Run a single optimization for the given combination.
    Returns: "completed", "skipped", or "error"
    """
    global current_optimization_status, running_optimizations
    from services.websocket_manager import ws_manager, _get_queue_data_from_status

    source = combo["source"]
    pair = combo["pair"]
    period = combo["period"]
    timeframe = combo["timeframe"]
    granularity = combo["granularity"]

    status = app_state.get_autonomous_status()

    last_broadcast_time = [0]  # Use list to allow mutation in nested function

    async def update_parallel_status(message: str, progress: int = None):
        if combo_id and combo_id in running_optimizations:
            async with running_optimizations_async_lock:
                if combo_id in running_optimizations:
                    running_optimizations[combo_id]["message"] = message
                    if progress is not None:
                        running_optimizations[combo_id]["progress"] = progress
                    app_state.update_autonomous_status(
                        parallel_running=list(running_optimizations.values())
                    )

            # Broadcast to WebSocket (throttled to every 0.5s to avoid flooding)
            import time
            now = time.time()
            if now - last_broadcast_time[0] >= 0.5:
                last_broadcast_time[0] = now
                status = app_state.get_autonomous_status()
                await ws_manager.broadcast("autonomous_status", {
                    "autonomous": status,
                    "queue": _get_queue_data_from_status(status)
                })

    # Fetch data (with caching)
    from data_fetcher import BinanceDataFetcher
    from services.ohlcv_cache import ohlcv_cache

    await update_parallel_status(f"Loading {pair}...", 5)
    app_state.update_autonomous_status(
        message=f"Loading {pair}...",
        progress=5,
        trial_current=0,
        trial_total=granularity["n_trials"]
    )

    # Check cache first (no lock needed for reads)
    df = ohlcv_cache.get(pair, timeframe["minutes"], period["months"])

    if df is not None:
        log(f"[Autonomous Optimizer] Cache HIT: {pair} {timeframe['label']} {period['label']}")
        await update_parallel_status(f"Cached {pair}", 10)
    else:
        # Cache miss - fetch from Binance (with semaphore for rate limiting)
        await update_parallel_status(f"Fetching {pair}...", 5)
        async with data_fetch_semaphore:
            fetcher = BinanceDataFetcher()
            try:
                df = await asyncio.wait_for(
                    fetcher.fetch_ohlcv(
                        pair=pair,
                        interval=timeframe["minutes"],
                        months=period["months"]
                    ),
                    timeout=300
                )
            except asyncio.TimeoutError:
                log(f"[Autonomous Optimizer] Data fetch TIMEOUT for {pair}", level='ERROR')
                return "error"
            except Exception as e:
                log(f"[Autonomous Optimizer] Data fetch error: {e}", level='ERROR')
                return "error"

            if df is not None and len(df) >= 100:
                # Store in cache for future use
                ohlcv_cache.set(pair, timeframe["minutes"], period["months"], df)
                log(f"[Autonomous Optimizer] Cached: {pair} {timeframe['label']} {period['label']} ({len(df):,} rows)")

    if df is None or len(df) < 100:
        return "error"

    data_validation = validate_data_range(df, period, timeframe)

    app_state.update_autonomous_status(data_validation=data_validation)

    if not data_validation["valid"]:
        app_state.update_autonomous_status(
            skipped_count=status.get("skipped_count", 0) + 1,
            message=f"SKIPPED {pair} - {data_validation['message']}"
        )
        return "skipped"

    # Run optimization
    log(f"[Autonomous Optimizer] Starting optimization for {pair} ({granularity['n_trials']} trials)...")
    temp_status = {"running": True, "progress": 0, "message": "", "report": None, "abort": False}
    current_optimization_status = temp_status

    async def update_progress():
        import re

        while temp_status["running"]:
            if not app_state.is_autonomous_enabled():
                temp_status["abort"] = True

            inner_progress = temp_status.get("progress", 0)
            mapped_progress = 15 + int(inner_progress * 0.8)
            app_state.update_autonomous_status(progress=mapped_progress)

            # Parse progress from message - format: "[Parallel] 49,400/79,200 (42.0%) | Found: 150"
            msg = temp_status.get("message", "")
            current_trial = 0
            total_trials = 0

            # Match the actual format from strategy_engine.py
            match = re.search(r'\[Parallel\]\s*([\d,]+)\s*/\s*([\d,]+)', msg)
            if match:
                current_trial = int(match.group(1).replace(',', ''))
                total_trials = int(match.group(2).replace(',', ''))

            # Calculate progress percentage
            progress_pct = int(inner_progress) if inner_progress else 0
            if current_trial > 0 and total_trials > 0:
                progress_pct = int((current_trial / total_trials) * 100)

            # Build status message
            if current_trial > 0 and total_trials > 0:
                status_msg = f"{pair} - {current_trial:,}/{total_trials:,}"
            elif progress_pct > 0:
                status_msg = f"Optimizing {pair} ({progress_pct}%)"
            else:
                status_msg = f"Optimizing {pair}..."

            app_state.update_autonomous_status(
                trial_current=current_trial,
                trial_total=total_trials,
                message=status_msg
            )
            await update_parallel_status(status_msg, progress_pct)

            await asyncio.sleep(0.3)

    loop = asyncio.get_event_loop()
    progress_task = asyncio.create_task(update_progress())

    try:
        from strategy_engine import run_strategy_finder
        log(f"[Autonomous Optimizer] {pair} - Submitting to thread pool...")
        await loop.run_in_executor(
            thread_pool,
            run_optimization_sync,
            df, combo, temp_status, run_strategy_finder, thread_pool
        )
        log(f"[Autonomous Optimizer] {pair} - Thread pool execution completed")
    except Exception as e:
        log(f"[Autonomous Optimizer] {pair} - EXCEPTION: {e}", level='ERROR')
        import traceback
        traceback.print_exc()
    finally:
        temp_status["running"] = False
        current_optimization_status = None
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    if temp_status.get("abort"):
        log(f"[Autonomous Optimizer] {pair} - Aborted")
        return "aborted"

    # Process results
    if temp_status.get("report"):
        report = temp_status["report"]
        top_strategies = report.get("top_10", [])

        if top_strategies:
            best = top_strategies[0]
            app_state.update_autonomous_status(
                last_result={
                    "pair": pair,
                    "timeframe": timeframe["label"],
                    "period": period["label"],
                    "granularity": granularity["label"],
                    "strategy": best.get("strategy_name", "Unknown"),
                    "pnl": best.get("metrics", {}).get("total_pnl", 0),
                    "win_rate": best.get("metrics", {}).get("win_rate", 0),
                }
            )

    # Record to history
    global autonomous_runs_history
    strategies_found = len(temp_status.get("report", {}).get("top_10", [])) if temp_status.get("report") else 0

    history_entry = {
        "completed_at": datetime.now().isoformat(),
        "source": source,
        "pair": pair,
        "period": period["label"],
        "timeframe": timeframe["label"],
        "granularity": granularity["label"],
        "strategies_found": strategies_found,
        "status": "success" if strategies_found > 0 else "no_results"
    }
    autonomous_runs_history.insert(0, history_entry)
    if len(autonomous_runs_history) > MAX_HISTORY_SIZE:
        autonomous_runs_history = autonomous_runs_history[:MAX_HISTORY_SIZE]

    log(f"[Autonomous Optimizer] Completed {pair} - {strategies_found} strategies found")
    return "completed"


async def process_single_combination(
    combo: dict,
    combo_index: int,
    combinations: list,
    thread_pool
):
    """Process a single optimization with semaphore control."""
    global running_optimizations

    combo_id = f"auto_{combo_index}_{combo['pair']}_{combo['timeframe']['label']}"

    async with optimization_semaphore:
        if not app_state.is_autonomous_enabled():
            return "aborted"

        # Wait for manual optimizer if running
        while app_state.is_unified_running():
            if not app_state.is_autonomous_enabled():
                return "aborted"
            await asyncio.sleep(1)

        # Register this optimization
        combo_status = {
            "id": combo_id,
            "index": combo_index,
            "pair": combo["pair"],
            "period": combo["period"]["label"],
            "timeframe": combo["timeframe"]["label"],
            "granularity": combo["granularity"]["label"],
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "progress": 0,
            "message": f"Starting {combo['pair']}...",
        }

        async with running_optimizations_async_lock:
            running_optimizations[combo_id] = combo_status
            app_state.update_autonomous_status(
                parallel_running=list(running_optimizations.values()),
                parallel_count=len(running_optimizations)
            )

        log(f"[Parallel Optimizer] Starting: {combo['pair']} {combo['timeframe']['label']}")

        try:
            result = await run_single_optimization(combo, combo_id, thread_pool)

            completed_item = {
                "index": combo_index,
                "pair": combo["pair"],
                "period": combo["period"]["label"],
                "timeframe": combo["timeframe"]["label"],
                "granularity": combo["granularity"]["label"],
                "completed_at": datetime.now().isoformat(),
                "status": result,
            }

            if result == "completed":
                status = app_state.get_autonomous_status()
                app_state.update_autonomous_status(
                    completed_count=status.get("completed_count", 0) + 1,
                    last_completed_at=datetime.now().isoformat()
                )

                # Record in database
                try:
                    from strategy_database import get_strategy_db
                    db = get_strategy_db()
                    db.record_completed_optimization(
                        pair=combo["pair"],
                        period_label=combo["period"]["label"],
                        timeframe_label=combo["timeframe"]["label"],
                        granularity_label=combo["granularity"]["label"],
                        strategies_found=0,
                        source=combo.get("source", "binance")
                    )
                except Exception as e:
                    log(f"[Parallel Optimizer] Error recording completion: {e}", level='WARNING')

            elif result == "error":
                status = app_state.get_autonomous_status()
                app_state.update_autonomous_status(
                    error_count=status.get("error_count", 0) + 1
                )

            # Add to completed queue
            status = app_state.get_autonomous_status()
            queue_completed = status.get("queue_completed", [])
            queue_completed.insert(0, completed_item)
            app_state.update_autonomous_status(
                queue_completed=queue_completed[:20]
            )

            return result

        except Exception as e:
            log(f"[Parallel Optimizer] Exception: {e}", level='ERROR')
            return "error"

        finally:
            async with running_optimizations_async_lock:
                if combo_id in running_optimizations:
                    del running_optimizations[combo_id]
                app_state.update_autonomous_status(
                    parallel_running=list(running_optimizations.values()),
                    parallel_count=len(running_optimizations)
                )


async def start_autonomous_optimizer(thread_pool):
    """
    Main autonomous optimizer loop.
    Runs multiple optimizations in parallel.
    """
    global running_optimizations

    log("[Parallel Optimizer] start_autonomous_optimizer() called")

    if app_state.is_autonomous_running():
        log("[Parallel Optimizer] Already running, exiting early")
        return

    log("[Parallel Optimizer] Initializing...")

    # Initialize async primitives if needed
    if optimization_semaphore is None:
        log("[Parallel Optimizer] Initializing async primitives...")
        init_async_primitives()

    log(f"[Parallel Optimizer] Setting auto_running=True, max_parallel={concurrency_config['max_concurrent']}")

    app_state.update_autonomous_status(
        auto_running=True,
        running=True,
        max_parallel=concurrency_config["max_concurrent"],
        message="Initializing optimizer..."
    )

    # Broadcast immediately so UI updates
    from services.websocket_manager import broadcast_autonomous_status
    broadcast_autonomous_status(app_state.get_autonomous_status())

    log("[Parallel Optimizer] Starting (waiting 3s for stability)...")

    await asyncio.sleep(3)

    # Build combinations in thread pool to avoid blocking event loop
    log("[Parallel Optimizer] Building combinations...")
    loop = asyncio.get_event_loop()
    combinations = await loop.run_in_executor(thread_pool, build_optimization_combinations)
    log(f"[Parallel Optimizer] Built {len(combinations)} combinations")

    app_state.update_autonomous_status(
        total_combinations=len(combinations),
        combinations_list=[{
            "pair": c["pair"],
            "period": c["period"]["label"],
            "timeframe": c["timeframe"]["label"],
            "granularity": c["granularity"]["label"],
        } for c in combinations],
        queue_completed=[],
        parallel_running=[],
        parallel_count=0,
        message=f"Ready - {len(combinations)} combinations to process"
    )
    broadcast_autonomous_status(app_state.get_autonomous_status())

    # Find resume point (in thread pool to avoid blocking)
    def find_resume():
        try:
            from strategy_database import get_strategy_db
            db = get_strategy_db()
            return find_resume_index(combinations, db)
        except:
            return 0

    start_index = await loop.run_in_executor(thread_pool, find_resume)

    app_state.update_autonomous_status(cycle_index=start_index)

    if start_index > 0:
        log(f"[Parallel Optimizer] Resuming from {start_index+1}/{len(combinations)}")

    active_tasks = set()
    current_index = start_index
    last_spawn_time = 0  # Allow immediate first spawn

    log(f"[Parallel Optimizer] Entering main loop. auto_running={app_state.is_autonomous_running()}, enabled={app_state.is_autonomous_enabled()}")

    while app_state.is_autonomous_running() and app_state.is_autonomous_enabled():
        try:
            log(f"[Parallel Optimizer] Loop iteration: {len(active_tasks)} active tasks, index={current_index}")

            if not app_state.is_autonomous_enabled():
                log("[Parallel Optimizer] Exiting: not enabled")
                break

            # Wait for manual optimizer
            if app_state.is_unified_running():
                app_state.update_autonomous_status(
                    paused=True,
                    message="Paused - waiting for manual optimizer..."
                )
                await asyncio.sleep(2)
                continue

            app_state.update_autonomous_status(paused=False)

            # Check cycle completion
            if current_index >= len(combinations):
                if active_tasks:
                    done, active_tasks = await asyncio.wait(active_tasks, timeout=1)
                    continue

                current_index = 0
                app_state.update_autonomous_status(
                    cycle_index=0,
                    message="Completed full cycle, restarting..."
                )
                log("[Parallel Optimizer] Completed full cycle")
                await asyncio.sleep(30)
                continue

            # Clean up completed tasks
            if active_tasks:
                done, active_tasks = await asyncio.wait(
                    active_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )

            # Dynamic resource-based spawning
            from services.resource_monitor import resource_monitor

            resources = resource_monitor.get_current_resources()
            cpu_percent = resources["cpu_percent"]
            mem_available_gb = resources["memory_available_gb"]

            # Resource thresholds for spawning
            CPU_SPAWN_THRESHOLD = 80  # Only spawn if CPU < 80%
            MEM_SPAWN_THRESHOLD = 4.0  # Only spawn if > 4GB available
            SPAWN_COOLDOWN = 30  # Seconds between spawns
            MAX_CONCURRENT = 1  # TEST MODE: Only 1 task at a time

            # Check if we can spawn based on resources
            can_spawn_cpu = cpu_percent < CPU_SPAWN_THRESHOLD
            can_spawn_mem = mem_available_gb > MEM_SPAWN_THRESHOLD
            can_spawn_slots = len(active_tasks) < MAX_CONCURRENT
            time_since_spawn = time.time() - last_spawn_time

            # Spawn if: resources available AND slots available AND cooldown elapsed AND work remaining
            if can_spawn_cpu and can_spawn_mem and can_spawn_slots and time_since_spawn >= SPAWN_COOLDOWN and current_index < len(combinations):
                combo = combinations[current_index]
                task = asyncio.create_task(
                    process_single_combination(combo, current_index, combinations, thread_pool)
                )
                active_tasks.add(task)
                current_index += 1
                last_spawn_time = time.time()
                app_state.update_autonomous_status(cycle_index=current_index)

                log(f"[Parallel Optimizer] Spawned task #{current_index} | CPU: {cpu_percent:.1f}% | Mem: {mem_available_gb:.1f}GB")

            # Update status with resource info
            status_msg = f"Running {len(active_tasks)} tasks | CPU: {cpu_percent:.0f}% | Mem: {mem_available_gb:.1f}GB"
            if not can_spawn_cpu:
                status_msg += " | CPU high, waiting..."
            elif not can_spawn_mem:
                status_msg += " | Memory low, waiting..."
            elif time_since_spawn < SPAWN_COOLDOWN and current_index < len(combinations):
                status_msg += f" | Next spawn in {SPAWN_COOLDOWN - int(time_since_spawn)}s"

            app_state.update_autonomous_status(
                running=len(active_tasks) > 0,
                max_parallel=len(active_tasks),  # Dynamic - no fixed max
                message=status_msg,
                cpu_percent=cpu_percent,
                memory_available_gb=mem_available_gb
            )

            log(f"[Parallel Optimizer] Broadcasting... active={len(active_tasks)}, index={current_index}")
            # Use async broadcast directly since we're in async context
            from services.websocket_manager import ws_manager, _get_queue_data_from_status
            status = app_state.get_autonomous_status()
            await ws_manager.broadcast("autonomous_status", {
                "autonomous": status,
                "queue": _get_queue_data_from_status(status)
            })

            log("[Parallel Optimizer] Sleeping 0.5s...")
            await asyncio.sleep(0.5)
            log("[Parallel Optimizer] Woke up, continuing loop")

        except Exception as e:
            import traceback
            log(f"[Parallel Optimizer] Loop error: {e}", level='ERROR')
            traceback.print_exc()
            await asyncio.sleep(5)

    log(f"[Parallel Optimizer] Exited loop! auto_running={app_state.is_autonomous_running()}, enabled={app_state.is_autonomous_enabled()}")

    # Cleanup
    if active_tasks:
        log(f"[Parallel Optimizer] Cancelling {len(active_tasks)} tasks...")
        for task in active_tasks:
            task.cancel()
        await asyncio.gather(*active_tasks, return_exceptions=True)

    with running_optimizations_lock:
        running_optimizations.clear()

    app_state.update_autonomous_status(
        auto_running=False,
        running=False,
        parallel_running=[],
        parallel_count=0,
        message="Stopped"
    )

    from services.websocket_manager import broadcast_autonomous_status
    broadcast_autonomous_status(app_state.get_autonomous_status())

    log("[Parallel Optimizer] Stopped")


async def stop_autonomous_optimizer():
    """Stop the autonomous optimizer."""
    global running_optimizations

    app_state.update_autonomous_status(
        enabled=False,
        auto_running=False,
        running=False,
        paused=False,
        message="Stopped by user"
    )

    # Signal current optimization to abort
    if current_optimization_status:
        current_optimization_status["abort"] = True

    # Signal all running optimizations to abort
    with running_optimizations_lock:
        for combo_id, status in running_optimizations.items():
            status["abort"] = True
        running_optimizations.clear()

    # Clear app state tracking
    app_state.clear_running_optimizations()
    app_state.update_autonomous_status(
        parallel_running=[],
        parallel_count=0
    )

    from services.websocket_manager import broadcast_autonomous_status
    broadcast_autonomous_status(app_state.get_autonomous_status())

    log("[Autonomous Optimizer] Stop signal sent")
