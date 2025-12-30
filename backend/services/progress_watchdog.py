"""
PROGRESS-BASED WATCHDOG
========================
A watchdog system that NEVER uses absolute time limits.

This replaces time-based timeout detection with:
1. Progress velocity monitoring
2. Signal-count-based stall detection
3. Event-driven completion detection
4. Relative progress comparison

Designed for VectorBT optimizations with 52,800+ combinations
where progress updates can be sparse.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List, Any, Deque
from collections import deque
from enum import Enum
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


# =============================================================================
# PROGRESS VELOCITY TRACKER
# =============================================================================

@dataclass
class VelocityDataPoint:
    """Single velocity measurement."""
    progress: float
    velocity: float  # Progress units gained this measurement
    acceleration: float  # Change in velocity


class ProgressVelocityTracker:
    """
    Track progress velocity (progress per measurement, NOT per second).

    The key insight is that if we're gaining ANY progress between measurements,
    the task is healthy - regardless of wall clock time.
    """

    # How many measurements for velocity calculation
    # Increased for batch processing - need a wider window to detect trends
    WINDOW_SIZE = 100

    # Minimum velocity to consider "making progress"
    # This is progress-per-measurement, not per-second
    # Set very low because batch updates are sparse
    MIN_VELOCITY = 0.00001  # 0.00001% per measurement

    def __init__(self):
        self.history: Deque[VelocityDataPoint] = deque(maxlen=500)
        self._last_progress = 0.0
        self._measurement_count = 0

    def update(self, progress: float) -> VelocityDataPoint:
        """
        Record new progress measurement.

        Args:
            progress: Current progress (0-100 or raw combo count)

        Returns:
            VelocityDataPoint with current velocity metrics
        """
        self._measurement_count += 1

        # Velocity = how much progress since last measurement
        velocity = progress - self._last_progress
        self._last_progress = progress

        # Acceleration = change in velocity
        acceleration = 0.0
        if self.history:
            prev_velocity = self.history[-1].velocity
            acceleration = velocity - prev_velocity

        data_point = VelocityDataPoint(
            progress=progress,
            velocity=velocity,
            acceleration=acceleration
        )
        self.history.append(data_point)

        return data_point

    def get_rolling_velocity(self) -> float:
        """Get average velocity over recent window."""
        if len(self.history) < 2:
            return 0.0

        recent = list(self.history)[-self.WINDOW_SIZE:]
        return sum(dp.velocity for dp in recent) / len(recent)

    def is_making_progress(self) -> bool:
        """Check if task is making meaningful progress."""
        return self.get_rolling_velocity() > self.MIN_VELOCITY

    def get_trend(self) -> str:
        """
        Analyze velocity trend.

        Returns: "accelerating", "steady", "decelerating", "stalled", "collecting_data"
        """
        if len(self.history) < self.WINDOW_SIZE:
            return "collecting_data"

        recent = list(self.history)[-self.WINDOW_SIZE:]
        avg_acceleration = sum(dp.acceleration for dp in recent) / len(recent)
        rolling_velocity = self.get_rolling_velocity()

        if rolling_velocity <= self.MIN_VELOCITY:
            return "stalled"
        elif avg_acceleration > 0.0001:
            return "accelerating"
        elif avg_acceleration < -0.0001:
            return "decelerating"
        else:
            return "steady"

    @property
    def measurement_count(self) -> int:
        return self._measurement_count


# =============================================================================
# SIGNAL-COUNT STALL DETECTOR
# =============================================================================

class SignalCountStallDetector:
    """
    Detect stalls based on consecutive unchanged measurements.

    Instead of "no progress for X seconds", we use
    "no progress change for Y consecutive measurements".

    This is completely time-independent.
    """

    # Consecutive unchanged measurements for warning
    # VectorBT batch processing can take several minutes between updates
    # At CHECK_INTERVAL=0.5s, 1800 measurements = ~15 minutes
    UNCHANGED_WARNING = 1800

    # Consecutive unchanged measurements for abort
    # Extended for batch processing: 3600 measurements = ~30 minutes
    # This gives enough time for large optimizations with 52,800+ combinations
    UNCHANGED_ABORT = 3600

    def __init__(self):
        self._last_progress_value = 0.0
        self._consecutive_unchanged = 0
        self._total_measurements = 0

    def update(self, progress: float) -> Dict:
        """
        Update with new progress measurement.

        Returns:
            Dict with stall detection status
        """
        self._total_measurements += 1

        if progress == self._last_progress_value:
            self._consecutive_unchanged += 1
        else:
            self._consecutive_unchanged = 0
            self._last_progress_value = progress

        return {
            "consecutive_unchanged": self._consecutive_unchanged,
            "warning": self._consecutive_unchanged >= self.UNCHANGED_WARNING,
            "should_abort": self._consecutive_unchanged >= self.UNCHANGED_ABORT,
            "total_measurements": self._total_measurements
        }

    def reset(self):
        """Reset the detector."""
        self._consecutive_unchanged = 0

    @property
    def is_stalled(self) -> bool:
        return self._consecutive_unchanged >= self.UNCHANGED_WARNING


# =============================================================================
# COMPLETION EVENT DETECTOR
# =============================================================================

class CompletionType(Enum):
    """Types of completion signals."""
    PROGRESS_100 = "progress_100"
    RESULT_AVAILABLE = "result_available"
    RUNNING_FALSE = "running_false"
    ALL_COMBOS_DONE = "all_combos_done"


class CompletionEventDetector:
    """
    Detect task completion through multiple independent signals.

    Requires 2+ completion signals for confident detection.
    """

    # How many signals needed for confident completion
    SIGNALS_NEEDED = 2

    def __init__(self):
        self._signals_received: Dict[CompletionType, bool] = {
            t: False for t in CompletionType
        }

    def check_progress(self, progress: float) -> bool:
        """Check if progress indicates completion."""
        if progress >= 100:
            self._signals_received[CompletionType.PROGRESS_100] = True
            return True
        return False

    def check_result(self, result: Any) -> bool:
        """Check if result is available."""
        if result is not None:
            self._signals_received[CompletionType.RESULT_AVAILABLE] = True
            return True
        return False

    def check_running_flag(self, running: bool) -> bool:
        """Check if running flag is False."""
        if not running:
            self._signals_received[CompletionType.RUNNING_FALSE] = True
            return True
        return False

    def check_combo_count(self, completed: int, total: int) -> bool:
        """Check if all combinations are done."""
        if total > 0 and completed >= total:
            self._signals_received[CompletionType.ALL_COMBOS_DONE] = True
            return True
        return False

    @property
    def signal_count(self) -> int:
        return sum(1 for v in self._signals_received.values() if v)

    @property
    def is_completed(self) -> bool:
        return self.signal_count >= self.SIGNALS_NEEDED

    @property
    def signals_received(self) -> List[str]:
        return [t.value for t, v in self._signals_received.items() if v]


# =============================================================================
# INTEGRATED PROGRESS-BASED WATCHDOG
# =============================================================================

class ProgressBasedWatchdog:
    """
    Watchdog that monitors task health through progress signals only.

    NO ABSOLUTE TIME LIMITS.

    Abort conditions (ALL must be true):
    1. Velocity is zero (no progress per measurement)
    2. Signal-count detector shows sustained stall
    3. Multiple measurements have passed with no change

    The key difference from time-based watchdog:
    - Old: "No progress for 20 minutes" -> abort
    - New: "No progress for 50 consecutive measurements" -> abort

    If the task is making ANY progress, it runs forever.
    """

    # How often to check (just for CPU relief, not timing)
    CHECK_INTERVAL = 0.5

    def __init__(
        self,
        task_id: str,
        status_dict: dict,
        total_combinations: int = 0,
        progress_key: str = "progress",
        abort_key: str = "abort",
        running_key: str = "running",
        result_key: str = "report",
        combo_current_key: str = None,
        combo_total_key: str = None
    ):
        """
        Initialize progress-based watchdog.

        Args:
            task_id: Identifier for logging
            status_dict: Dictionary to monitor
            total_combinations: Total combos to process (for ETA)
            progress_key: Key for progress percentage
            abort_key: Key to set True when aborting
            running_key: Key indicating if task is running
            result_key: Key for result/report object
            combo_current_key: Optional key for current combo count
            combo_total_key: Optional key for total combo count
        """
        self.task_id = task_id
        self.status_dict = status_dict
        self.total_combinations = total_combinations
        self.progress_key = progress_key
        self.abort_key = abort_key
        self.running_key = running_key
        self.result_key = result_key
        self.combo_current_key = combo_current_key
        self.combo_total_key = combo_total_key

        # Detection systems
        self.velocity_tracker = ProgressVelocityTracker()
        self.stall_detector = SignalCountStallDetector()
        self.completion_detector = CompletionEventDetector()

        # State
        self._running = False
        self._aborted = False
        self._abort_reason: Optional[str] = None
        self._measurement_count = 0
        self._last_log_measurement = 0

    async def start(self):
        """Start watching the task (runs until completion or abort)."""
        self._running = True
        self._aborted = False

        log(f"[ProgressWatchdog] Started monitoring {self.task_id}")
        log(f"[ProgressWatchdog] Mode: PROGRESS-BASED (no absolute time limits)")
        if self.total_combinations > 0:
            log(f"[ProgressWatchdog] Total combinations: {self.total_combinations:,}")

        while self._running:
            # Check if task is supposed to be running
            if not self.status_dict.get(self.running_key, True):
                log(f"[ProgressWatchdog] Task {self.task_id} marked as not running")
                break

            # Get current state
            self._measurement_count += 1
            progress = self.status_dict.get(self.progress_key, 0)
            result = self.status_dict.get(self.result_key)
            running = self.status_dict.get(self.running_key, True)

            # Get combo counts if available
            combo_current = 0
            combo_total = 0
            if self.combo_current_key:
                combo_current = self.status_dict.get(self.combo_current_key, 0)
            if self.combo_total_key:
                combo_total = self.status_dict.get(self.combo_total_key, 0)

            # === Check completion ===
            self.completion_detector.check_progress(progress)
            self.completion_detector.check_result(result)
            self.completion_detector.check_running_flag(running)
            if combo_total > 0:
                self.completion_detector.check_combo_count(combo_current, combo_total)

            if self.completion_detector.is_completed:
                log(f"[ProgressWatchdog] Completion detected for {self.task_id}")
                log(f"[ProgressWatchdog] Signals: {self.completion_detector.signals_received}")
                break

            # === Update velocity tracker ===
            velocity_data = self.velocity_tracker.update(progress)
            velocity_trend = self.velocity_tracker.get_trend()
            making_progress = self.velocity_tracker.is_making_progress()

            # === Update stall detector ===
            stall_status = self.stall_detector.update(progress)

            # === Decision logic ===

            # If making progress, log occasionally but keep running
            if making_progress:
                # Log every 100 measurements
                if self._measurement_count - self._last_log_measurement >= 100:
                    self._last_log_measurement = self._measurement_count
                    rolling_velocity = self.velocity_tracker.get_rolling_velocity()
                    log(f"[ProgressWatchdog] {self.task_id}: {progress:.1f}% | "
                        f"Velocity: {velocity_trend} ({rolling_velocity:.4f}/meas) | "
                        f"Measurements: {self._measurement_count}")

            # Warning if stall detector triggered but velocity still shows progress
            elif stall_status["warning"] and not making_progress:
                log(f"[ProgressWatchdog] WARNING: {self.task_id} appears stalled", level='WARNING')
                log(f"[ProgressWatchdog]   Progress: {progress:.1f}%", level='WARNING')
                log(f"[ProgressWatchdog]   Unchanged for: {stall_status['consecutive_unchanged']} measurements", level='WARNING')
                log(f"[ProgressWatchdog]   Velocity trend: {velocity_trend}", level='WARNING')

            # Abort only if both detectors agree task is stuck
            if stall_status["should_abort"] and not making_progress:
                self._trigger_abort(
                    f"Task stalled at {progress:.1f}% for "
                    f"{stall_status['consecutive_unchanged']} consecutive measurements"
                )
                break

            # Event-driven wait (just for CPU, not timing)
            await asyncio.sleep(self.CHECK_INTERVAL)

        self._running = False

    def _trigger_abort(self, reason: str):
        """Trigger abort on the monitored task."""
        self._aborted = True
        self._abort_reason = reason
        self.status_dict[self.abort_key] = True
        log(f"[ProgressWatchdog] ABORT: {self.task_id} - {reason}", level='ERROR')

    async def stop(self):
        """Stop watching (task completed normally)."""
        self._running = False
        log(f"[ProgressWatchdog] Stopped monitoring {self.task_id} "
            f"(measurements: {self._measurement_count}, aborted: {self._aborted})")

    @property
    def is_aborted(self) -> bool:
        return self._aborted

    @property
    def abort_reason(self) -> Optional[str]:
        return self._abort_reason


# =============================================================================
# MULTI-TASK WATCHDOG COORDINATOR
# =============================================================================

class WatchdogCoordinator:
    """
    Coordinate multiple watchdogs for parallel optimization.

    Uses cross-task comparison: if other tasks are completing
    but one is stuck, that's a stronger stall signal.
    """

    def __init__(self):
        self.watchdogs: Dict[str, ProgressBasedWatchdog] = {}
        self._completed_count = 0
        self._lock = asyncio.Lock()

    def register(self, watchdog: ProgressBasedWatchdog):
        """Register a watchdog."""
        self.watchdogs[watchdog.task_id] = watchdog

    def unregister(self, task_id: str):
        """Unregister a watchdog (task completed)."""
        if task_id in self.watchdogs:
            del self.watchdogs[task_id]
            self._completed_count += 1

    async def heartbeat(self):
        """
        Called when any task completes - drives stall detection.

        This is the "tick" mechanism for signal-count-based detection.
        When one task completes, we check if others are making progress.
        """
        async with self._lock:
            for task_id, watchdog in self.watchdogs.items():
                # If another task completed but this one hasn't progressed,
                # that's a relative stall signal
                if watchdog.stall_detector.is_stalled:
                    log(f"[WatchdogCoordinator] Task {task_id} may be stuck "
                        f"(other tasks completing, this one stalled)", level='WARNING')

    def get_status(self) -> Dict:
        """Get status of all watched tasks."""
        return {
            "active_watchdogs": len(self.watchdogs),
            "completed_count": self._completed_count,
            "tasks": {
                tid: {
                    "measurements": wd.velocity_tracker.measurement_count,
                    "velocity_trend": wd.velocity_tracker.get_trend(),
                    "stall_status": wd.stall_detector._consecutive_unchanged
                }
                for tid, wd in self.watchdogs.items()
            }
        }


# Global coordinator instance
watchdog_coordinator = WatchdogCoordinator()


# =============================================================================
# FACTORY FUNCTION FOR EASY CREATION
# =============================================================================

def create_progress_watchdog(
    task_id: str,
    status_dict: dict,
    total_combinations: int = 0,
    **kwargs
) -> ProgressBasedWatchdog:
    """
    Create a progress-based watchdog and register it with the coordinator.

    Args:
        task_id: Unique identifier for the task
        status_dict: Status dictionary to monitor
        total_combinations: Total combinations to process
        **kwargs: Additional arguments for ProgressBasedWatchdog

    Returns:
        Configured ProgressBasedWatchdog instance
    """
    watchdog = ProgressBasedWatchdog(
        task_id=task_id,
        status_dict=status_dict,
        total_combinations=total_combinations,
        **kwargs
    )
    watchdog_coordinator.register(watchdog)
    return watchdog


async def notify_task_completed(task_id: str):
    """
    Notify coordinator that a task completed.

    Call this when any optimization task finishes to drive
    relative stall detection for remaining tasks.
    """
    watchdog_coordinator.unregister(task_id)
    await watchdog_coordinator.heartbeat()
