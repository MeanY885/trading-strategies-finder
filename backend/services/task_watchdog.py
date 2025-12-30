"""
Task Watchdog Module

Provides centralized timeout and monitoring infrastructure for long-running tasks
in the Autonomous Optimizer and Elite Strategies validation systems.

Classes:
    - TimeoutCalculator: Dynamic timeout calculation based on task parameters
    - InterruptibleSleep: Event-based sleep that can be woken early
    - TaskWatchdog: Monitor task progress and force-abort stalled tasks
    - OrphanCleaner: Background cleanup of stale/orphaned task entries
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


def log(message: str, level: str = 'INFO'):
    """Log with consistent formatting."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    if level == 'ERROR':
        logger.error(f"[{timestamp}] {message}")
    elif level == 'WARNING':
        logger.warning(f"[{timestamp}] {message}")
    elif level == 'DEBUG':
        logger.debug(f"[{timestamp}] {message}")
    else:
        logger.info(f"[{timestamp}] {message}")


class TimeoutCalculator:
    """
    Calculate dynamic timeouts based on task parameters.

    Timeouts scale with:
    - Timeframe: Lower timeframes = more data points = longer processing
    - Period: Longer periods = more historical data = longer processing
    - Granularity: More trials = longer optimization time
    """

    # Multipliers for different timeframes (minutes -> multiplier)
    # Lower timeframes have more data points, so higher multiplier
    TIMEFRAME_MULTIPLIERS = {
        1: 1.0,      # 1 minute - most data
        3: 0.9,
        5: 0.8,
        15: 0.6,
        30: 0.5,
        60: 0.4,     # 1 hour
        120: 0.35,   # 2 hours
        240: 0.3,    # 4 hours
        1440: 0.2,   # 1 day - least data
    }

    # Multipliers for different periods (months -> multiplier)
    PERIOD_MULTIPLIERS = {
        0.25: 0.5,   # 1 week
        0.5: 0.7,    # 2 weeks
        1: 1.0,      # 1 month
        2: 1.3,
        3: 1.5,      # 3 months
        6: 2.0,      # 6 months
        9: 2.5,      # 9 months
        12: 3.0,     # 1 year
        24: 4.0,     # 2 years
        36: 5.0,     # 3 years
    }

    # Multipliers for granularity (n_trials -> multiplier)
    GRANULARITY_MULTIPLIERS = {
        100: 0.5,
        200: 0.7,
        400: 1.0,
        1000: 1.5,
        2500: 2.5,
        5000: 3.5,
        10000: 5.0,
    }

    # Base timeouts in seconds
    # PHILOSOPHY: Progress-based detection is PRIMARY, absolute timeout is SAFETY NET
    # If task is making progress, let it run. Only abort if truly stuck or hits max.
    # VectorBT optimizations can take 40+ minutes for some combinations.
    BASE_OPTIMIZATION_TIMEOUT = 7200  # 2 hours base (generous, relies on progress detection)
    BASE_DATA_FETCH_TIMEOUT = 300     # 5 minutes (network can be slow)
    BASE_BACKTEST_TIMEOUT = 120       # 2 minutes

    # Absolute limits - these are SAFETY NETS, not expected durations
    MIN_TIMEOUT = 300     # 5 minute minimum
    MAX_TIMEOUT = 14400   # 4 hours maximum (safety net only)

    @classmethod
    def _interpolate_multiplier(cls, value: float, multipliers: Dict[float, float]) -> float:
        """Interpolate multiplier for values between defined points."""
        keys = sorted(multipliers.keys())

        # Below minimum
        if value <= keys[0]:
            return multipliers[keys[0]]

        # Above maximum
        if value >= keys[-1]:
            return multipliers[keys[-1]]

        # Find surrounding keys and interpolate
        for i in range(len(keys) - 1):
            if keys[i] <= value <= keys[i + 1]:
                lower_key, upper_key = keys[i], keys[i + 1]
                lower_mult, upper_mult = multipliers[lower_key], multipliers[upper_key]

                # Linear interpolation
                ratio = (value - lower_key) / (upper_key - lower_key)
                return lower_mult + ratio * (upper_mult - lower_mult)

        return 1.0  # Default fallback

    @classmethod
    def get_optimization_timeout(cls,
                                  timeframe_minutes: int,
                                  period_months: float,
                                  n_trials: int) -> int:
        """
        Calculate timeout for a full optimization task.

        Args:
            timeframe_minutes: Chart timeframe in minutes (1, 5, 15, 60, etc.)
            period_months: Data period in months (1, 3, 6, 12, etc.)
            n_trials: Number of optimization trials (100, 400, 2500, etc.)

        Returns:
            Timeout in seconds, clamped to MIN_TIMEOUT..MAX_TIMEOUT
        """
        tf_mult = cls._interpolate_multiplier(timeframe_minutes, cls.TIMEFRAME_MULTIPLIERS)
        period_mult = cls._interpolate_multiplier(period_months, cls.PERIOD_MULTIPLIERS)
        gran_mult = cls._interpolate_multiplier(n_trials, cls.GRANULARITY_MULTIPLIERS)

        timeout = cls.BASE_OPTIMIZATION_TIMEOUT * tf_mult * period_mult * gran_mult

        # Clamp to limits
        timeout = max(cls.MIN_TIMEOUT, min(cls.MAX_TIMEOUT, int(timeout)))

        log(f"[Watchdog] Optimization timeout: {timeout}s "
            f"(tf={timeframe_minutes}m×{tf_mult:.2f}, period={period_months}mo×{period_mult:.2f}, "
            f"trials={n_trials}×{gran_mult:.2f})", level='DEBUG')

        return timeout

    @classmethod
    def get_data_fetch_timeout(cls, period_months: float) -> int:
        """
        Calculate timeout for data fetching.

        Scales primarily with period (more months = more data to fetch).
        """
        period_mult = cls._interpolate_multiplier(period_months, cls.PERIOD_MULTIPLIERS)
        timeout = cls.BASE_DATA_FETCH_TIMEOUT * period_mult

        # Data fetch has its own limits (30s to 10 min)
        timeout = max(30, min(600, int(timeout)))

        return timeout

    @classmethod
    def get_backtest_timeout(cls, period_months: float) -> int:
        """
        Calculate timeout for a single backtest operation.

        Scales with period (more data points to process).
        """
        period_mult = cls._interpolate_multiplier(period_months, cls.PERIOD_MULTIPLIERS)
        timeout = cls.BASE_BACKTEST_TIMEOUT * period_mult

        # Backtest has its own limits (30s to 5 min)
        timeout = max(30, min(300, int(timeout)))

        return timeout


class InterruptibleSleep:
    """
    Event-based sleep that can be interrupted early.

    Replaces long asyncio.sleep() calls with interruptible waiting,
    allowing tasks to respond quickly to stop signals.
    """

    def __init__(self):
        self._stop_event: Optional[asyncio.Event] = None
        self._interrupted = False

    async def sleep(self,
                    seconds: float,
                    check_callback: Optional[Callable[[], bool]] = None,
                    check_interval: float = 1.0) -> bool:
        """
        Sleep for up to `seconds`, checking for interrupts.

        Args:
            seconds: Maximum time to sleep
            check_callback: Optional function returning True to continue, False to stop
            check_interval: How often to check for interrupts (default 1s)

        Returns:
            True if sleep completed normally, False if interrupted
        """
        if self._stop_event is None:
            self._stop_event = asyncio.Event()

        self._interrupted = False
        elapsed = 0.0

        while elapsed < seconds:
            remaining = min(check_interval, seconds - elapsed)

            try:
                # Wait for either the interval or the stop event
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=remaining
                )
                # Event was set - we're interrupted
                self._interrupted = True
                return False
            except asyncio.TimeoutError:
                # Normal timeout - continue sleeping
                pass

            elapsed += remaining

            # Check callback if provided
            if check_callback is not None and not check_callback():
                self._interrupted = True
                return False

        return True  # Sleep completed normally

    def interrupt(self):
        """Wake up the sleeper immediately."""
        self._interrupted = True
        if self._stop_event is not None:
            self._stop_event.set()

    def reset(self):
        """Reset for reuse."""
        self._interrupted = False
        if self._stop_event is not None:
            self._stop_event.clear()

    @property
    def was_interrupted(self) -> bool:
        """Check if the last sleep was interrupted."""
        return self._interrupted


class TaskWatchdog:
    """
    Monitor task progress and force-abort stalled tasks.

    Watches a status dictionary for progress updates. If no progress
    is detected for too long, sets the abort flag to stop the task.
    """

    # Default thresholds - THIS IS THE PRIMARY ABORT MECHANISM
    # Tasks are aborted when progress stalls, not based on total time
    DEFAULT_WARNING_SECONDS = 300    # 5 minutes without progress = warning
    DEFAULT_ABORT_SECONDS = 600      # 10 minutes without progress = abort (PRIMARY)
    CHECK_INTERVAL = 10              # Check every 10 seconds

    def __init__(self,
                 task_id: str,
                 status_dict: dict,
                 timeout_seconds: int,
                 progress_key: str = "progress",
                 abort_key: str = "abort",
                 running_key: str = "running",
                 no_progress_warning_seconds: int = None,
                 no_progress_abort_seconds: int = None):
        """
        Initialize watchdog for a task.

        Args:
            task_id: Identifier for logging
            status_dict: Dictionary to monitor (must have progress_key and abort_key)
            timeout_seconds: Absolute timeout for the task
            progress_key: Key in status_dict for progress value (0-100)
            abort_key: Key in status_dict to set True when aborting
            running_key: Key in status_dict indicating if task is running
            no_progress_warning_seconds: Seconds without progress before warning
            no_progress_abort_seconds: Seconds without progress before abort
        """
        self.task_id = task_id
        self.status_dict = status_dict
        self.timeout_seconds = timeout_seconds
        self.progress_key = progress_key
        self.abort_key = abort_key
        self.running_key = running_key

        self.no_progress_warning = no_progress_warning_seconds or self.DEFAULT_WARNING_SECONDS
        self.no_progress_abort = no_progress_abort_seconds or self.DEFAULT_ABORT_SECONDS

        self._running = False
        self._aborted = False
        self._abort_reason: Optional[str] = None
        self._start_time: Optional[float] = None
        self._last_progress: float = 0
        self._last_progress_time: float = 0
        self._warning_logged = False

    async def start(self):
        """Start watching the task. Runs until task completes or is aborted."""
        self._running = True
        self._aborted = False
        self._start_time = time.time()
        self._last_progress = self.status_dict.get(self.progress_key, 0)
        self._last_progress_time = self._start_time
        self._warning_logged = False

        log(f"[Watchdog] Started monitoring task {self.task_id} "
            f"(timeout={self.timeout_seconds}s, no_progress_abort={self.no_progress_abort}s)")

        while self._running:
            await asyncio.sleep(self.CHECK_INTERVAL)

            # Check if task is still supposed to be running
            if not self.status_dict.get(self.running_key, True):
                log(f"[Watchdog] Task {self.task_id} marked as not running, stopping watchdog")
                break

            now = time.time()
            elapsed = now - self._start_time
            current_progress = self.status_dict.get(self.progress_key, 0)

            # Check absolute timeout
            if elapsed > self.timeout_seconds:
                self._trigger_abort(f"Absolute timeout ({self.timeout_seconds}s) exceeded")
                break

            # Check progress
            if current_progress > self._last_progress:
                # Progress made - reset timer
                self._last_progress = current_progress
                self._last_progress_time = now
                self._warning_logged = False
            else:
                # No progress - check duration
                no_progress_duration = now - self._last_progress_time

                if no_progress_duration > self.no_progress_abort:
                    self._trigger_abort(
                        f"No progress for {int(no_progress_duration)}s "
                        f"(stuck at {current_progress:.1f}%)"
                    )
                    break
                elif no_progress_duration > self.no_progress_warning and not self._warning_logged:
                    log(f"[Watchdog] WARNING: Task {self.task_id} has no progress "
                        f"for {int(no_progress_duration)}s (at {current_progress:.1f}%)",
                        level='WARNING')
                    self._warning_logged = True

        self._running = False

    def _trigger_abort(self, reason: str):
        """Trigger abort on the monitored task."""
        self._aborted = True
        self._abort_reason = reason
        self.status_dict[self.abort_key] = True
        log(f"[Watchdog] ABORT triggered for {self.task_id}: {reason}", level='ERROR')

    async def stop(self):
        """Stop watching (task completed normally)."""
        self._running = False
        if self._start_time:
            duration = time.time() - self._start_time
            log(f"[Watchdog] Stopped monitoring {self.task_id} after {duration:.1f}s "
                f"(aborted={self._aborted})")

    def is_aborted(self) -> bool:
        """Check if watchdog triggered an abort."""
        return self._aborted

    @property
    def abort_reason(self) -> Optional[str]:
        """Get the reason for abort, if any."""
        return self._abort_reason


class OrphanCleaner:
    """
    Background task to clean orphaned/stale task entries.

    Tasks can leave orphaned entries in tracking dictionaries if they
    crash or are interrupted before cleanup. This cleaner periodically
    removes entries that haven't been updated for too long.
    """

    CLEANUP_INTERVAL = 60          # Check every minute
    ORPHAN_THRESHOLD = 1800        # 30 minutes without update = orphan

    def __init__(self,
                 running_dict: dict,
                 lock: asyncio.Lock,
                 started_at_key: str = "started_at",
                 last_update_key: str = "last_update",
                 on_cleanup_callback: Optional[Callable[[str, dict], None]] = None):
        """
        Initialize orphan cleaner.

        Args:
            running_dict: Dictionary of running tasks to monitor
            lock: Async lock protecting the dictionary
            started_at_key: Key in task dict for start timestamp
            last_update_key: Key in task dict for last update timestamp
            on_cleanup_callback: Called with (task_id, task_data) when orphan is cleaned
        """
        self.running_dict = running_dict
        self.lock = lock
        self.started_at_key = started_at_key
        self.last_update_key = last_update_key
        self.on_cleanup_callback = on_cleanup_callback

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._total_cleaned = 0

    async def start_cleanup_loop(self):
        """Start the background cleanup loop."""
        self._running = True
        log("[OrphanCleaner] Started background cleanup loop")

        while self._running:
            await asyncio.sleep(self.CLEANUP_INTERVAL)

            if not self._running:
                break

            try:
                cleaned = await self.cleanup_orphans()
                if cleaned > 0:
                    log(f"[OrphanCleaner] Cleaned {cleaned} orphaned task(s)")
            except Exception as e:
                log(f"[OrphanCleaner] Error during cleanup: {e}", level='ERROR')

    async def stop(self):
        """Stop the cleanup loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log(f"[OrphanCleaner] Stopped (total cleaned: {self._total_cleaned})")

    async def cleanup_orphans(self) -> int:
        """
        Clean orphaned entries from the running dictionary.

        Returns:
            Number of orphans removed
        """
        now = time.time()
        orphans = []

        async with self.lock:
            for task_id, task_data in list(self.running_dict.items()):
                # Get the most recent timestamp
                last_update = None

                if self.last_update_key in task_data:
                    last_update = task_data[self.last_update_key]
                elif self.started_at_key in task_data:
                    last_update = task_data[self.started_at_key]

                if last_update is None:
                    # No timestamp - consider it an orphan
                    orphans.append((task_id, task_data, "no timestamp"))
                    continue

                # Parse timestamp if it's a string
                if isinstance(last_update, str):
                    try:
                        dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        last_update = dt.timestamp()
                    except ValueError:
                        orphans.append((task_id, task_data, "invalid timestamp"))
                        continue

                # Check if orphaned
                age = now - last_update
                if age > self.ORPHAN_THRESHOLD:
                    orphans.append((task_id, task_data, f"stale for {int(age)}s"))

            # Remove orphans
            for task_id, task_data, reason in orphans:
                del self.running_dict[task_id]
                log(f"[OrphanCleaner] Removed orphan: {task_id} ({reason})", level='WARNING')

                if self.on_cleanup_callback:
                    try:
                        self.on_cleanup_callback(task_id, task_data)
                    except Exception as e:
                        log(f"[OrphanCleaner] Cleanup callback error: {e}", level='ERROR')

        self._total_cleaned += len(orphans)
        return len(orphans)

    @property
    def total_cleaned(self) -> int:
        """Total number of orphans cleaned since start."""
        return self._total_cleaned


# Global orphan cleaner instance (initialized by main.py)
orphan_cleaner: Optional[OrphanCleaner] = None


async def init_orphan_cleaner(running_dict: dict, lock: asyncio.Lock) -> OrphanCleaner:
    """
    Initialize and start the global orphan cleaner.

    Call this from main.py startup.
    """
    global orphan_cleaner

    orphan_cleaner = OrphanCleaner(running_dict, lock)

    # Start in background
    asyncio.create_task(orphan_cleaner.start_cleanup_loop())

    return orphan_cleaner


async def stop_orphan_cleaner():
    """Stop the global orphan cleaner."""
    global orphan_cleaner

    if orphan_cleaner:
        await orphan_cleaner.stop()
        orphan_cleaner = None
