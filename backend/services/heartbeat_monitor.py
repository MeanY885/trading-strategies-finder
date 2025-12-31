"""
HEARTBEAT-BASED TASK MONITORING SYSTEM
======================================

Alternative to time-based timeouts that monitors task "liveness" through
explicit heartbeat signals rather than progress percentage updates.

Key Concepts:
-------------
1. HEARTBEAT vs PROGRESS:
   - Progress: "I'm 50% done" (tells you how far)
   - Heartbeat: "I'm alive and working" (tells you the task is responsive)

   Tasks can report heartbeats even when progress is hard to measure
   (e.g., VectorBT processes thousands of combinations between callbacks).

2. ADAPTIVE THRESHOLDS:
   - Learn from historical task completion times
   - Adjust warning/abort thresholds based on task characteristics
   - Handle variance in execution times gracefully

3. KUBERNETES-INSPIRED PROBES:
   - Liveness: Is the task still running? (heartbeat-based)
   - Readiness: Is the task making progress? (progress-based)
   - Startup: Allow longer initial delay for complex tasks

Design Philosophy:
------------------
Based on research from Celery, Kubernetes, and distributed systems:
- Celery workers send heartbeats every 60 seconds, offline after 120 seconds
- Kubernetes uses configurable probe intervals with failure thresholds
- Separate concerns: "alive" != "making progress" != "will finish soon"

Usage:
------
    from services.heartbeat_monitor import (
        HeartbeatMonitor,
        TaskHeartbeat,
        AdaptiveThresholds,
        HeartbeatWatchdog
    )

    # Create a heartbeat for a task
    heartbeat = TaskHeartbeat(task_id="opt_123")

    # In your task, periodically call:
    heartbeat.beat()  # or heartbeat.beat(message="Processing strategy 15/50")

    # Monitor with watchdog
    watchdog = HeartbeatWatchdog(heartbeat, on_stall=handle_stall)
    await watchdog.start()
"""

import asyncio
import time
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

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
# ENUMS AND TYPES
# =============================================================================

class TaskState(Enum):
    """Task lifecycle states (Kubernetes-inspired)."""
    PENDING = "pending"       # Task created but not started
    STARTING = "starting"     # Task initializing (startup probe phase)
    RUNNING = "running"       # Task actively processing
    STALLING = "stalling"     # No heartbeat received (warning state)
    STALLED = "stalled"       # Heartbeat timeout exceeded (abort state)
    COMPLETED = "completed"   # Task finished successfully
    FAILED = "failed"         # Task failed with error
    ABORTED = "aborted"       # Task was force-aborted


class ProbeType(Enum):
    """Types of health probes (Kubernetes terminology)."""
    LIVENESS = "liveness"     # Is the task alive? (heartbeat-based)
    READINESS = "readiness"   # Is the task making progress? (progress-based)
    STARTUP = "startup"       # Initial startup grace period


@dataclass
class ProbeResult:
    """Result of a health probe check."""
    probe_type: ProbeType
    success: bool
    message: str
    timestamp: float = field(default_factory=time.time)
    consecutive_failures: int = 0


# =============================================================================
# TASK HEARTBEAT
# =============================================================================

class TaskHeartbeat:
    """
    Heartbeat tracker for a single task.

    Tasks call beat() periodically to signal they're alive.
    This is independent of progress percentage - a task can be
    "alive" without having made measurable progress.

    Example:
        heartbeat = TaskHeartbeat("optimization_123")

        # In task loop:
        for batch in batches:
            process(batch)
            heartbeat.beat()  # Signal alive

        heartbeat.complete()
    """

    # Default intervals (in seconds)
    DEFAULT_BEAT_INTERVAL = 30      # Expected beat frequency
    DEFAULT_WARN_AFTER = 120        # Warning after 2 minutes without beat
    DEFAULT_ABORT_AFTER = 300       # Abort after 5 minutes without beat
    DEFAULT_STARTUP_GRACE = 60      # Extra grace period during startup

    def __init__(
        self,
        task_id: str,
        expected_interval: float = None,
        warn_after: float = None,
        abort_after: float = None,
        startup_grace: float = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize heartbeat tracker.

        Args:
            task_id: Unique identifier for the task
            expected_interval: How often beats are expected (seconds)
            warn_after: Seconds without beat before warning
            abort_after: Seconds without beat before abort recommendation
            startup_grace: Extra time allowed during startup
            metadata: Optional task metadata (pair, timeframe, etc.)
        """
        self.task_id = task_id
        self.expected_interval = expected_interval or self.DEFAULT_BEAT_INTERVAL
        self.warn_after = warn_after or self.DEFAULT_WARN_AFTER
        self.abort_after = abort_after or self.DEFAULT_ABORT_AFTER
        self.startup_grace = startup_grace or self.DEFAULT_STARTUP_GRACE
        self.metadata = metadata or {}

        # State
        self._lock = threading.RLock()
        self._state = TaskState.PENDING
        self._created_at = time.time()
        self._started_at: Optional[float] = None
        self._last_beat: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._beat_count = 0
        self._last_message: str = ""

        # Beat history for interval analysis
        self._beat_times: deque = deque(maxlen=100)

        # Progress tracking (optional - separate from heartbeat)
        self._progress: float = 0.0
        self._progress_updated_at: Optional[float] = None

        # Callbacks
        self._on_stall_callbacks: List[Callable] = []
        self._on_abort_callbacks: List[Callable] = []

    def start(self) -> None:
        """Mark task as started. Begins startup grace period."""
        with self._lock:
            self._state = TaskState.STARTING
            self._started_at = time.time()
            self._last_beat = time.time()  # Initial "beat"
            log(f"[Heartbeat] Task {self.task_id} started (startup grace: {self.startup_grace}s)")

    def beat(self, message: str = None, progress: float = None) -> None:
        """
        Record a heartbeat - task is alive.

        Args:
            message: Optional status message
            progress: Optional progress percentage (0-100)
        """
        with self._lock:
            now = time.time()

            # Transition from STARTING to RUNNING after first explicit beat
            if self._state == TaskState.STARTING:
                self._state = TaskState.RUNNING
            elif self._state == TaskState.STALLING:
                self._state = TaskState.RUNNING
                log(f"[Heartbeat] Task {self.task_id} recovered from stalling state")

            self._last_beat = now
            self._beat_count += 1
            self._beat_times.append(now)

            if message:
                self._last_message = message

            if progress is not None:
                self._progress = progress
                self._progress_updated_at = now

    def complete(self, success: bool = True) -> None:
        """Mark task as completed."""
        with self._lock:
            self._state = TaskState.COMPLETED if success else TaskState.FAILED
            self._completed_at = time.time()

            duration = self._completed_at - self._started_at if self._started_at else 0
            log(f"[Heartbeat] Task {self.task_id} completed "
                f"(duration: {duration:.1f}s, beats: {self._beat_count})")

    def abort(self, reason: str = "Heartbeat timeout") -> None:
        """Mark task as aborted."""
        with self._lock:
            self._state = TaskState.ABORTED
            self._completed_at = time.time()
            log(f"[Heartbeat] Task {self.task_id} aborted: {reason}", level='WARNING')

    def check_liveness(self) -> ProbeResult:
        """
        Check if task is alive (liveness probe).

        Returns:
            ProbeResult indicating if task is considered alive
        """
        with self._lock:
            if self._state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.ABORTED):
                return ProbeResult(
                    probe_type=ProbeType.LIVENESS,
                    success=False,
                    message=f"Task is {self._state.value}"
                )

            if self._last_beat is None:
                return ProbeResult(
                    probe_type=ProbeType.LIVENESS,
                    success=False,
                    message="No heartbeat received yet"
                )

            now = time.time()
            silence = now - self._last_beat

            # Apply startup grace period
            effective_abort = self.abort_after
            if self._state == TaskState.STARTING:
                effective_abort += self.startup_grace

            if silence > effective_abort:
                self._state = TaskState.STALLED
                return ProbeResult(
                    probe_type=ProbeType.LIVENESS,
                    success=False,
                    message=f"No heartbeat for {silence:.0f}s (threshold: {effective_abort}s)"
                )

            if silence > self.warn_after:
                self._state = TaskState.STALLING
                return ProbeResult(
                    probe_type=ProbeType.LIVENESS,
                    success=True,
                    message=f"Heartbeat delayed {silence:.0f}s (warning: {self.warn_after}s)"
                )

            return ProbeResult(
                probe_type=ProbeType.LIVENESS,
                success=True,
                message=f"Last beat {silence:.0f}s ago"
            )

    def check_readiness(self, progress_threshold: float = 0.01) -> ProbeResult:
        """
        Check if task is making progress (readiness probe).

        This is separate from liveness - a task can be alive but stuck.

        Args:
            progress_threshold: Minimum progress increase to consider "making progress"
        """
        with self._lock:
            if self._progress_updated_at is None:
                return ProbeResult(
                    probe_type=ProbeType.READINESS,
                    success=False,
                    message="No progress updates received"
                )

            now = time.time()
            silence = now - self._progress_updated_at

            if silence > self.warn_after:
                return ProbeResult(
                    probe_type=ProbeType.READINESS,
                    success=False,
                    message=f"No progress update for {silence:.0f}s (at {self._progress:.1f}%)"
                )

            return ProbeResult(
                probe_type=ProbeType.READINESS,
                success=True,
                message=f"Progress: {self._progress:.1f}%, last update {silence:.0f}s ago"
            )

    def get_average_beat_interval(self) -> Optional[float]:
        """Calculate average interval between beats."""
        with self._lock:
            if len(self._beat_times) < 2:
                return None

            intervals = []
            times = list(self._beat_times)
            for i in range(1, len(times)):
                intervals.append(times[i] - times[i-1])

            return statistics.mean(intervals) if intervals else None

    @property
    def state(self) -> TaskState:
        with self._lock:
            return self._state

    @property
    def time_since_last_beat(self) -> Optional[float]:
        with self._lock:
            if self._last_beat is None:
                return None
            return time.time() - self._last_beat

    @property
    def elapsed(self) -> Optional[float]:
        with self._lock:
            if self._started_at is None:
                return None
            end = self._completed_at or time.time()
            return end - self._started_at

    def to_dict(self) -> Dict:
        """Serialize heartbeat state for WebSocket/API."""
        with self._lock:
            return {
                "task_id": self.task_id,
                "state": self._state.value,
                "started_at": datetime.fromtimestamp(self._started_at).isoformat() if self._started_at else None,
                "last_beat": datetime.fromtimestamp(self._last_beat).isoformat() if self._last_beat else None,
                "time_since_beat": self.time_since_last_beat,
                "beat_count": self._beat_count,
                "progress": self._progress,
                "message": self._last_message,
                "elapsed": self.elapsed,
                "metadata": self.metadata,
            }


# =============================================================================
# ADAPTIVE THRESHOLDS
# =============================================================================

class AdaptiveThresholds:
    """
    Learn optimal timeout thresholds from historical task completion patterns.

    Based on research from:
    - Probability-Guaranteed Adaptive Timeout (PGAT) algorithms
    - Q-learning approaches for adaptive scheduling
    - Statistical analysis of completion time distributions

    Key insight: Different task types (timeframe, period, granularity) have
    different execution time distributions. We track these separately and
    adapt thresholds to minimize false positives (premature aborts) while
    catching truly stuck tasks.
    """

    # Number of historical samples to keep per task type
    HISTORY_SIZE = 50

    # Percentiles for threshold calculation
    WARN_PERCENTILE = 90    # Warn if longer than 90% of similar tasks
    ABORT_PERCENTILE = 99   # Abort if longer than 99% of similar tasks

    # Minimum thresholds (floor)
    MIN_WARN_SECONDS = 120
    MIN_ABORT_SECONDS = 300

    # Maximum thresholds (ceiling)
    MAX_WARN_SECONDS = 1800   # 30 minutes
    MAX_ABORT_SECONDS = 7200  # 2 hours

    def __init__(self):
        self._lock = threading.RLock()

        # Historical completion times keyed by task type signature
        # Signature format: "{timeframe}_{period}_{granularity}"
        self._history: Dict[str, deque] = {}

        # Cached thresholds (recomputed when history changes)
        self._thresholds: Dict[str, Tuple[float, float]] = {}  # (warn, abort)

        # Learning rate for exponential moving average (for online learning)
        self.ema_alpha = 0.1

        # Running statistics
        self._stats: Dict[str, Dict] = {}

    def _get_task_signature(self, metadata: Dict) -> str:
        """Generate a unique signature for task type grouping."""
        timeframe = metadata.get("timeframe", "unknown")
        period = metadata.get("period", "unknown")
        granularity = metadata.get("granularity", "unknown")

        # Normalize period to a canonical form
        period_str = str(period).replace(" ", "_").lower()

        return f"{timeframe}_{period_str}_{granularity}"

    def record_completion(self, metadata: Dict, duration: float) -> None:
        """
        Record a task completion for threshold learning.

        Args:
            metadata: Task metadata (timeframe, period, granularity)
            duration: How long the task took in seconds
        """
        signature = self._get_task_signature(metadata)

        with self._lock:
            if signature not in self._history:
                self._history[signature] = deque(maxlen=self.HISTORY_SIZE)
                self._stats[signature] = {
                    "count": 0,
                    "sum": 0,
                    "sum_sq": 0,
                    "min": float('inf'),
                    "max": 0,
                    "ema": None,
                }

            self._history[signature].append(duration)

            # Update running statistics
            stats = self._stats[signature]
            stats["count"] += 1
            stats["sum"] += duration
            stats["sum_sq"] += duration * duration
            stats["min"] = min(stats["min"], duration)
            stats["max"] = max(stats["max"], duration)

            # Update EMA
            if stats["ema"] is None:
                stats["ema"] = duration
            else:
                stats["ema"] = self.ema_alpha * duration + (1 - self.ema_alpha) * stats["ema"]

            # Invalidate cached threshold
            self._thresholds.pop(signature, None)

            log(f"[AdaptiveThresholds] Recorded {signature}: {duration:.1f}s "
                f"(samples: {len(self._history[signature])}, EMA: {stats['ema']:.1f}s)")

    def get_thresholds(self, metadata: Dict) -> Tuple[float, float]:
        """
        Get adaptive warn and abort thresholds for a task type.

        Args:
            metadata: Task metadata (timeframe, period, granularity)

        Returns:
            (warn_seconds, abort_seconds) tuple
        """
        signature = self._get_task_signature(metadata)

        with self._lock:
            # Return cached if available
            if signature in self._thresholds:
                return self._thresholds[signature]

            # Check if we have enough history
            if signature not in self._history or len(self._history[signature]) < 5:
                # Not enough data - use static defaults
                warn = TaskHeartbeat.DEFAULT_WARN_AFTER
                abort = TaskHeartbeat.DEFAULT_ABORT_AFTER

                log(f"[AdaptiveThresholds] No history for {signature}, using defaults "
                    f"(warn: {warn}s, abort: {abort}s)")

                return (warn, abort)

            # Calculate percentile-based thresholds
            durations = sorted(self._history[signature])

            # Calculate percentile indices
            warn_idx = int(len(durations) * self.WARN_PERCENTILE / 100)
            abort_idx = int(len(durations) * self.ABORT_PERCENTILE / 100)

            # Get base values from percentiles
            warn_base = durations[min(warn_idx, len(durations) - 1)]
            abort_base = durations[min(abort_idx, len(durations) - 1)]

            # Add buffer (1.5x for warn, 2x for abort) to account for variance
            warn = warn_base * 1.5
            abort = abort_base * 2.0

            # Clamp to min/max bounds
            warn = max(self.MIN_WARN_SECONDS, min(self.MAX_WARN_SECONDS, warn))
            abort = max(self.MIN_ABORT_SECONDS, min(self.MAX_ABORT_SECONDS, abort))

            # Ensure abort > warn
            if abort <= warn:
                abort = warn * 2

            # Cache and return
            self._thresholds[signature] = (warn, abort)

            log(f"[AdaptiveThresholds] Calculated for {signature}: "
                f"warn={warn:.0f}s (p{self.WARN_PERCENTILE}), "
                f"abort={abort:.0f}s (p{self.ABORT_PERCENTILE})")

            return (warn, abort)

    def get_estimated_duration(self, metadata: Dict) -> Optional[float]:
        """
        Get estimated duration for a task type based on historical EMA.

        Returns None if not enough history.
        """
        signature = self._get_task_signature(metadata)

        with self._lock:
            if signature in self._stats and self._stats[signature]["ema"]:
                return self._stats[signature]["ema"]
            return None

    def get_statistics(self, metadata: Dict = None) -> Dict:
        """Get statistics for all task types or a specific one."""
        with self._lock:
            if metadata:
                signature = self._get_task_signature(metadata)
                if signature in self._stats:
                    stats = self._stats[signature].copy()
                    stats["signature"] = signature
                    stats["samples"] = len(self._history.get(signature, []))
                    if stats["count"] > 0:
                        stats["mean"] = stats["sum"] / stats["count"]
                        variance = (stats["sum_sq"] / stats["count"]) - (stats["mean"] ** 2)
                        stats["std"] = variance ** 0.5 if variance > 0 else 0
                    return stats
                return {}

            # Return all statistics
            result = {}
            for sig, stats in self._stats.items():
                entry = stats.copy()
                entry["samples"] = len(self._history.get(sig, []))
                if entry["count"] > 0:
                    entry["mean"] = entry["sum"] / entry["count"]
                result[sig] = entry
            return result


# =============================================================================
# HEARTBEAT WATCHDOG
# =============================================================================

class HeartbeatWatchdog:
    """
    Async watchdog that monitors a TaskHeartbeat and takes action on stalls.

    Inspired by Kubernetes probe behavior:
    - periodSeconds: How often to check (default 10s)
    - failureThreshold: How many failures before action (default 3)
    - successThreshold: How many successes to recover (default 1)

    Usage:
        heartbeat = TaskHeartbeat("task_123")
        watchdog = HeartbeatWatchdog(
            heartbeat,
            on_warn=lambda hb: log("Warning: task stalling"),
            on_abort=lambda hb: abort_task(hb.task_id)
        )

        asyncio.create_task(watchdog.start())

        # ... task runs and calls heartbeat.beat() ...

        await watchdog.stop()
    """

    DEFAULT_PERIOD_SECONDS = 10
    DEFAULT_FAILURE_THRESHOLD = 3
    DEFAULT_SUCCESS_THRESHOLD = 1

    def __init__(
        self,
        heartbeat: TaskHeartbeat,
        period_seconds: float = None,
        failure_threshold: int = None,
        success_threshold: int = None,
        on_warn: Optional[Callable[[TaskHeartbeat], None]] = None,
        on_abort: Optional[Callable[[TaskHeartbeat], None]] = None,
        on_recover: Optional[Callable[[TaskHeartbeat], None]] = None,
        adaptive_thresholds: Optional[AdaptiveThresholds] = None,
    ):
        """
        Initialize watchdog.

        Args:
            heartbeat: TaskHeartbeat to monitor
            period_seconds: How often to check (default 10s)
            failure_threshold: Consecutive failures before action
            success_threshold: Consecutive successes to recover
            on_warn: Callback when task enters stalling state
            on_abort: Callback when task should be aborted
            on_recover: Callback when task recovers from stalling
            adaptive_thresholds: Optional AdaptiveThresholds for dynamic timeouts
        """
        self.heartbeat = heartbeat
        self.period_seconds = period_seconds or self.DEFAULT_PERIOD_SECONDS
        self.failure_threshold = failure_threshold or self.DEFAULT_FAILURE_THRESHOLD
        self.success_threshold = success_threshold or self.DEFAULT_SUCCESS_THRESHOLD

        self.on_warn = on_warn
        self.on_abort = on_abort
        self.on_recover = on_recover
        self.adaptive_thresholds = adaptive_thresholds

        # State
        self._running = False
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._warned = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start monitoring the heartbeat."""
        self._running = True
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._warned = False

        # Apply adaptive thresholds if available
        if self.adaptive_thresholds:
            warn, abort = self.adaptive_thresholds.get_thresholds(self.heartbeat.metadata)
            self.heartbeat.warn_after = warn
            self.heartbeat.abort_after = abort
            log(f"[HeartbeatWatchdog] Using adaptive thresholds for {self.heartbeat.task_id}: "
                f"warn={warn:.0f}s, abort={abort:.0f}s")

        log(f"[HeartbeatWatchdog] Started monitoring {self.heartbeat.task_id} "
            f"(period={self.period_seconds}s, failure_threshold={self.failure_threshold})")

        while self._running:
            await asyncio.sleep(self.period_seconds)

            if not self._running:
                break

            # Check if task is already done
            if self.heartbeat.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.ABORTED):
                log(f"[HeartbeatWatchdog] Task {self.heartbeat.task_id} finished, stopping")
                break

            # Perform liveness check
            result = self.heartbeat.check_liveness()

            if result.success:
                self._consecutive_failures = 0
                self._consecutive_successes += 1

                # Recovery handling
                if self._warned and self._consecutive_successes >= self.success_threshold:
                    self._warned = False
                    if self.on_recover:
                        try:
                            self.on_recover(self.heartbeat)
                        except Exception as e:
                            log(f"[HeartbeatWatchdog] Recovery callback error: {e}", level='ERROR')
            else:
                self._consecutive_successes = 0
                self._consecutive_failures += 1

                log(f"[HeartbeatWatchdog] Liveness check failed for {self.heartbeat.task_id}: "
                    f"{result.message} (failures: {self._consecutive_failures}/{self.failure_threshold})")

                # Warning handling
                if not self._warned and self.heartbeat.state == TaskState.STALLING:
                    self._warned = True
                    if self.on_warn:
                        try:
                            self.on_warn(self.heartbeat)
                        except Exception as e:
                            log(f"[HeartbeatWatchdog] Warn callback error: {e}", level='ERROR')

                # Abort handling
                if self._consecutive_failures >= self.failure_threshold:
                    if self.heartbeat.state == TaskState.STALLED:
                        log(f"[HeartbeatWatchdog] Aborting {self.heartbeat.task_id} after "
                            f"{self._consecutive_failures} consecutive failures", level='ERROR')

                        self.heartbeat.abort(f"Liveness probe failed {self._consecutive_failures} times")

                        if self.on_abort:
                            try:
                                self.on_abort(self.heartbeat)
                            except Exception as e:
                                log(f"[HeartbeatWatchdog] Abort callback error: {e}", level='ERROR')

                        break

        self._running = False

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

        # Record completion for adaptive thresholds if applicable
        if self.adaptive_thresholds and self.heartbeat.elapsed:
            if self.heartbeat.state == TaskState.COMPLETED:
                self.adaptive_thresholds.record_completion(
                    self.heartbeat.metadata,
                    self.heartbeat.elapsed
                )

        log(f"[HeartbeatWatchdog] Stopped monitoring {self.heartbeat.task_id}")

    @property
    def is_running(self) -> bool:
        return self._running


# =============================================================================
# HEARTBEAT REGISTRY
# =============================================================================

class HeartbeatRegistry:
    """
    Global registry for all active task heartbeats.

    Provides a central place to:
    - Track all running tasks
    - Query task states
    - Broadcast heartbeat updates
    - Manage watchdogs
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._heartbeats: Dict[str, TaskHeartbeat] = {}
        self._watchdogs: Dict[str, HeartbeatWatchdog] = {}
        self._adaptive_thresholds = AdaptiveThresholds()

    def register(
        self,
        task_id: str,
        metadata: Optional[Dict] = None,
        auto_watchdog: bool = True,
        on_warn: Optional[Callable] = None,
        on_abort: Optional[Callable] = None,
    ) -> TaskHeartbeat:
        """
        Register a new task and create its heartbeat.

        Args:
            task_id: Unique task identifier
            metadata: Task metadata for adaptive thresholds
            auto_watchdog: Whether to automatically start a watchdog
            on_warn: Callback for warning state
            on_abort: Callback for abort

        Returns:
            TaskHeartbeat instance to use in the task
        """
        metadata = metadata or {}

        # Get adaptive thresholds based on metadata
        warn, abort = self._adaptive_thresholds.get_thresholds(metadata)

        heartbeat = TaskHeartbeat(
            task_id=task_id,
            warn_after=warn,
            abort_after=abort,
            metadata=metadata
        )

        with self._lock:
            self._heartbeats[task_id] = heartbeat

        log(f"[HeartbeatRegistry] Registered task {task_id}")

        return heartbeat

    async def start_watchdog(
        self,
        task_id: str,
        on_warn: Optional[Callable] = None,
        on_abort: Optional[Callable] = None,
    ) -> Optional[HeartbeatWatchdog]:
        """Start a watchdog for a registered task."""
        heartbeat = self.get(task_id)
        if not heartbeat:
            log(f"[HeartbeatRegistry] Cannot start watchdog - task {task_id} not found",
                level='WARNING')
            return None

        watchdog = HeartbeatWatchdog(
            heartbeat=heartbeat,
            on_warn=on_warn,
            on_abort=on_abort,
            adaptive_thresholds=self._adaptive_thresholds,
        )

        with self._lock:
            self._watchdogs[task_id] = watchdog

        # Start in background
        asyncio.create_task(watchdog.start())

        return watchdog

    def get(self, task_id: str) -> Optional[TaskHeartbeat]:
        """Get a heartbeat by task ID."""
        with self._lock:
            return self._heartbeats.get(task_id)

    def unregister(self, task_id: str) -> None:
        """Remove a task from the registry."""
        with self._lock:
            # Stop watchdog if running
            if task_id in self._watchdogs:
                watchdog = self._watchdogs.pop(task_id)
                asyncio.create_task(watchdog.stop())

            # Record completion for adaptive learning
            if task_id in self._heartbeats:
                heartbeat = self._heartbeats[task_id]
                if heartbeat.elapsed and heartbeat.state == TaskState.COMPLETED:
                    self._adaptive_thresholds.record_completion(
                        heartbeat.metadata,
                        heartbeat.elapsed
                    )
                del self._heartbeats[task_id]

        log(f"[HeartbeatRegistry] Unregistered task {task_id}")

    def get_all_states(self) -> Dict[str, Dict]:
        """Get serialized state of all heartbeats."""
        with self._lock:
            return {
                task_id: hb.to_dict()
                for task_id, hb in self._heartbeats.items()
            }

    def get_stalling_tasks(self) -> List[TaskHeartbeat]:
        """Get list of tasks in STALLING or STALLED state."""
        with self._lock:
            return [
                hb for hb in self._heartbeats.values()
                if hb.state in (TaskState.STALLING, TaskState.STALLED)
            ]

    def get_statistics(self) -> Dict:
        """Get adaptive threshold statistics."""
        return self._adaptive_thresholds.get_statistics()


# =============================================================================
# GLOBAL REGISTRY INSTANCE
# =============================================================================

# Global heartbeat registry singleton
heartbeat_registry = HeartbeatRegistry()


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_heartbeat_for_optimization(
    combo_id: str,
    pair: str,
    timeframe: str,
    period: str,
    granularity: int,
) -> TaskHeartbeat:
    """
    Convenience function to create a heartbeat for an optimization task.

    Usage:
        heartbeat = create_heartbeat_for_optimization(
            combo_id="opt_123",
            pair="BTC/GBP",
            timeframe="1h",
            period="3 months",
            granularity=400
        )
        heartbeat.start()

        # In optimization loop:
        for batch in batches:
            process(batch)
            heartbeat.beat(f"Processed {batch_num}/{total}")

        heartbeat.complete()
    """
    metadata = {
        "pair": pair,
        "timeframe": timeframe,
        "period": period,
        "granularity": granularity,
    }

    return heartbeat_registry.register(
        task_id=combo_id,
        metadata=metadata
    )


class HeartbeatContext:
    """
    Context manager for heartbeat lifecycle.

    Usage:
        async with HeartbeatContext("task_123", metadata) as heartbeat:
            for batch in batches:
                process(batch)
                heartbeat.beat()
    """

    def __init__(
        self,
        task_id: str,
        metadata: Optional[Dict] = None,
        on_warn: Optional[Callable] = None,
        on_abort: Optional[Callable] = None,
    ):
        self.task_id = task_id
        self.metadata = metadata
        self.on_warn = on_warn
        self.on_abort = on_abort
        self.heartbeat: Optional[TaskHeartbeat] = None
        self.watchdog: Optional[HeartbeatWatchdog] = None

    async def __aenter__(self) -> TaskHeartbeat:
        self.heartbeat = heartbeat_registry.register(
            task_id=self.task_id,
            metadata=self.metadata
        )
        self.heartbeat.start()

        self.watchdog = await heartbeat_registry.start_watchdog(
            self.task_id,
            on_warn=self.on_warn,
            on_abort=self.on_abort
        )

        return self.heartbeat

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.heartbeat:
            if exc_type:
                self.heartbeat.abort(f"Exception: {exc_type.__name__}")
            else:
                self.heartbeat.complete()

        heartbeat_registry.unregister(self.task_id)
        return False  # Don't suppress exceptions
