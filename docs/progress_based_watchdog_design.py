"""
PROGRESS-BASED WATCHDOG SYSTEM DESIGN
======================================
A watchdog system that NEVER uses absolute time limits.

PROBLEM STATEMENT:
- VectorBT optimizations process 52,800+ combinations
- Progress updates can be sparse (VectorBT broadcasts every 10 combos)
- Time-based timeouts fail because:
  1. Legitimate tasks can take hours on slower hardware
  2. Different data sizes have vastly different processing times
  3. System load affects processing speed unpredictably

PHILOSOPHY:
- If progress is being made, let it run forever
- Only abort when task is TRULY stuck (no progress signals at all)
- Use progress velocity, not duration
- Event-driven completion detection

FOUR KEY APPROACHES:
1. Progress Signal Detection (primary)
2. Progress Velocity Monitoring
3. Stall Detection Without Time (using signal absence)
4. Event-Driven Completion Detection
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List, Any
from collections import deque
from enum import Enum
from datetime import datetime


# =============================================================================
# APPROACH 1: PROGRESS SIGNAL DETECTION
# =============================================================================

class ProgressSignalType(Enum):
    """Types of progress signals that indicate task is alive."""
    PERCENTAGE_UPDATE = "percentage"      # Progress % changed
    COMBINATION_COUNT = "combination"     # Combos processed increased
    RESULT_FOUND = "result"               # New strategy result discovered
    STAGE_CHANGE = "stage"                # Task entered new phase
    HEARTBEAT = "heartbeat"               # Generic "I'm alive" signal
    COMPLETION = "completion"             # Task finished (100% or result ready)


@dataclass
class ProgressSignal:
    """A single progress signal from a task."""
    signal_type: ProgressSignalType
    timestamp: float
    value: Any  # Progress %, combo count, etc.
    metadata: Dict = field(default_factory=dict)


class ProgressSignalDetector:
    """
    Detects task completion purely through progress signals.

    NEVER uses absolute time limits. Instead:
    - Monitors signal frequency
    - Detects completion states
    - Identifies stall patterns (consecutive identical signals)
    """

    # Consecutive identical signals before stall alert
    IDENTICAL_SIGNAL_THRESHOLD = 10

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.signals: deque = deque(maxlen=100)  # Recent signals
        self._completed = False
        self._completion_reason: Optional[str] = None
        self._stall_detected = False
        self._callbacks: List[Callable] = []

    def register_signal(self, signal: ProgressSignal) -> Dict:
        """
        Register a new progress signal.

        Returns:
            Dict with detection results:
            - completed: bool
            - stalled: bool
            - progress_active: bool
            - completion_reason: str or None
        """
        self.signals.append(signal)

        # Check for completion signals
        if self._check_completion(signal):
            return {
                "completed": True,
                "stalled": False,
                "progress_active": True,
                "completion_reason": self._completion_reason
            }

        # Check for stall pattern (many identical signals)
        stalled = self._check_stall_pattern()

        return {
            "completed": False,
            "stalled": stalled,
            "progress_active": len(self.signals) > 0,
            "completion_reason": None
        }

    def _check_completion(self, signal: ProgressSignal) -> bool:
        """Check if this signal indicates task completion."""

        # Direct completion signal
        if signal.signal_type == ProgressSignalType.COMPLETION:
            self._completed = True
            self._completion_reason = signal.metadata.get("reason", "Completion signal received")
            return True

        # Progress reached 100%
        if signal.signal_type == ProgressSignalType.PERCENTAGE_UPDATE:
            if signal.value >= 100:
                self._completed = True
                self._completion_reason = "Progress reached 100%"
                return True

        # All combinations processed
        if signal.signal_type == ProgressSignalType.COMBINATION_COUNT:
            total = signal.metadata.get("total", 0)
            if total > 0 and signal.value >= total:
                self._completed = True
                self._completion_reason = f"All {total} combinations processed"
                return True

        return False

    def _check_stall_pattern(self) -> bool:
        """
        Detect stall WITHOUT using time.

        A stall is detected when we receive many consecutive IDENTICAL signals.
        This indicates the task is emitting signals but not making progress.
        """
        if len(self.signals) < self.IDENTICAL_SIGNAL_THRESHOLD:
            return False

        # Get recent signals
        recent = list(self.signals)[-self.IDENTICAL_SIGNAL_THRESHOLD:]

        # Check if all recent signals are identical (same value)
        first_value = recent[0].value
        all_identical = all(s.value == first_value for s in recent)

        if all_identical:
            self._stall_detected = True
            return True

        self._stall_detected = False
        return False

    @property
    def is_completed(self) -> bool:
        return self._completed

    @property
    def is_stalled(self) -> bool:
        return self._stall_detected


# =============================================================================
# APPROACH 2: PROGRESS VELOCITY MONITORING
# =============================================================================

@dataclass
class VelocitySnapshot:
    """A snapshot of progress velocity at a point in time."""
    timestamp: float
    progress: float  # Current progress (0-100 or combo count)
    velocity: float  # Progress units per second
    acceleration: float  # Change in velocity


class ProgressVelocityMonitor:
    """
    Monitor progress VELOCITY instead of progress time.

    KEY INSIGHT: If velocity > 0, task is healthy regardless of duration.

    - Tracks progress delta over rolling windows
    - Detects velocity trends (accelerating, decelerating, stalled)
    - Uses velocity thresholds, not time thresholds
    """

    # Rolling window for velocity calculation
    VELOCITY_WINDOW_SIZE = 10

    # Minimum velocity to consider "making progress" (progress units per signal)
    # This is NOT time-based - it's progress-per-signal-based
    MIN_VELOCITY_PER_SIGNAL = 0.001  # 0.001% per signal minimum

    def __init__(self, task_id: str, total_units: int = 100):
        self.task_id = task_id
        self.total_units = total_units
        self.snapshots: deque = deque(maxlen=100)
        self._current_progress = 0
        self._signal_count = 0

    def update(self, progress: float) -> VelocitySnapshot:
        """
        Update with new progress value.

        Args:
            progress: Current progress (percentage or combo count)

        Returns:
            VelocitySnapshot with current velocity metrics
        """
        self._signal_count += 1
        progress_delta = progress - self._current_progress
        self._current_progress = progress

        # Calculate velocity as progress-per-signal (NOT per-time)
        # This removes time dependency entirely
        velocity = progress_delta  # Progress units gained this signal

        # Calculate acceleration from previous velocity
        acceleration = 0.0
        if self.snapshots:
            prev_velocity = self.snapshots[-1].velocity
            acceleration = velocity - prev_velocity

        snapshot = VelocitySnapshot(
            timestamp=time.time(),
            progress=progress,
            velocity=velocity,
            acceleration=acceleration
        )
        self.snapshots.append(snapshot)

        return snapshot

    def get_rolling_velocity(self) -> float:
        """
        Get average velocity over recent window.

        Returns progress units per signal (NOT per second).
        """
        if len(self.snapshots) < 2:
            return 0.0

        recent = list(self.snapshots)[-self.VELOCITY_WINDOW_SIZE:]
        total_velocity = sum(s.velocity for s in recent)
        return total_velocity / len(recent)

    def is_making_progress(self) -> bool:
        """
        Check if task is making meaningful progress.

        Uses velocity threshold, NOT time threshold.
        """
        rolling_velocity = self.get_rolling_velocity()
        return rolling_velocity > self.MIN_VELOCITY_PER_SIGNAL

    def get_velocity_trend(self) -> str:
        """
        Analyze velocity trend.

        Returns: "accelerating", "steady", "decelerating", "stalled"
        """
        if len(self.snapshots) < self.VELOCITY_WINDOW_SIZE:
            return "collecting_data"

        recent = list(self.snapshots)[-self.VELOCITY_WINDOW_SIZE:]
        avg_acceleration = sum(s.acceleration for s in recent) / len(recent)
        rolling_velocity = self.get_rolling_velocity()

        if rolling_velocity <= self.MIN_VELOCITY_PER_SIGNAL:
            return "stalled"
        elif avg_acceleration > 0.01:
            return "accelerating"
        elif avg_acceleration < -0.01:
            return "decelerating"
        else:
            return "steady"

    def estimate_remaining_signals(self) -> Optional[int]:
        """
        Estimate how many more signals until completion.

        NOT time-based - signal-based estimation.
        """
        remaining_progress = self.total_units - self._current_progress
        rolling_velocity = self.get_rolling_velocity()

        if rolling_velocity <= 0:
            return None  # Cannot estimate if stalled

        return int(remaining_progress / rolling_velocity)


# =============================================================================
# APPROACH 3: STALL DETECTION WITHOUT TIME
# =============================================================================

class SignalAbsenceDetector:
    """
    Detect stalls through signal ABSENCE, not time elapsed.

    KEY CONCEPT: Instead of "no progress for X seconds", use
    "no progress for Y consecutive check cycles".

    The "check cycles" are event-driven (triggered by something else
    making progress, or by external heartbeat checks).
    """

    # Number of check cycles with no signal before alert
    CYCLES_WITHOUT_SIGNAL_WARNING = 5
    CYCLES_WITHOUT_SIGNAL_ABORT = 10

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._last_signal_cycle = 0
        self._current_cycle = 0
        self._cycles_without_signal = 0
        self._signal_received_this_cycle = False

    def tick_cycle(self) -> Dict:
        """
        Called by external event (other task completing, heartbeat, etc.)

        This is NOT time-based - it's event-based.

        Returns:
            Dict with stall detection results
        """
        self._current_cycle += 1

        if not self._signal_received_this_cycle:
            self._cycles_without_signal += 1
        else:
            self._cycles_without_signal = 0
            self._last_signal_cycle = self._current_cycle

        # Reset for next cycle
        self._signal_received_this_cycle = False

        return {
            "cycles_without_signal": self._cycles_without_signal,
            "warning": self._cycles_without_signal >= self.CYCLES_WITHOUT_SIGNAL_WARNING,
            "should_abort": self._cycles_without_signal >= self.CYCLES_WITHOUT_SIGNAL_ABORT,
            "last_active_cycle": self._last_signal_cycle,
            "current_cycle": self._current_cycle
        }

    def signal_received(self):
        """Mark that a signal was received this cycle."""
        self._signal_received_this_cycle = True

    def get_status(self) -> Dict:
        """Get current stall detection status."""
        return {
            "healthy": self._cycles_without_signal < self.CYCLES_WITHOUT_SIGNAL_WARNING,
            "warning": self._cycles_without_signal >= self.CYCLES_WITHOUT_SIGNAL_WARNING,
            "critical": self._cycles_without_signal >= self.CYCLES_WITHOUT_SIGNAL_ABORT,
            "cycles_stalled": self._cycles_without_signal
        }


class RelativeStallDetector:
    """
    Detect stalls by comparing to OTHER tasks' progress.

    If other tasks are completing but this one is stuck at same progress,
    it's likely stalled - regardless of absolute time.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._progress_at_last_comparison = 0
        self._other_tasks_completed_since = 0
        self._comparisons_without_progress = 0

    def compare_progress(self, current_progress: float, other_tasks_completed: int) -> Dict:
        """
        Compare this task's progress relative to other tasks completing.

        Args:
            current_progress: This task's current progress
            other_tasks_completed: Count of other tasks that have completed

        Returns:
            Relative stall detection results
        """
        made_progress = current_progress > self._progress_at_last_comparison
        others_advanced = other_tasks_completed > self._other_tasks_completed_since

        if others_advanced:
            # Other tasks completed - use this as a "tick" for comparison
            if not made_progress:
                self._comparisons_without_progress += 1
            else:
                self._comparisons_without_progress = 0

            self._other_tasks_completed_since = other_tasks_completed

        self._progress_at_last_comparison = current_progress

        # Stall if we haven't progressed while 3+ other tasks completed
        is_relatively_stalled = self._comparisons_without_progress >= 3

        return {
            "relatively_stalled": is_relatively_stalled,
            "comparisons_without_progress": self._comparisons_without_progress,
            "other_tasks_completed": other_tasks_completed,
            "current_progress": current_progress
        }


# =============================================================================
# APPROACH 4: EVENT-DRIVEN COMPLETION DETECTION
# =============================================================================

class CompletionEventType(Enum):
    """Types of events that indicate task completion."""
    PROGRESS_100 = "progress_100"
    RESULT_READY = "result_ready"
    STATUS_FLAG = "status_flag"
    OUTPUT_DETECTED = "output_detected"
    COMBINATION_COUNT = "combo_complete"
    EXPLICIT_DONE = "explicit_done"


@dataclass
class CompletionEvent:
    """An event indicating task completion."""
    event_type: CompletionEventType
    timestamp: float
    details: Dict = field(default_factory=dict)


class EventDrivenCompletionDetector:
    """
    Detect task completion through events, not timeouts.

    Multiple completion signals that together confirm task is done:
    1. Progress reaching 100%
    2. Result object becoming available
    3. Status flag changing to "completed"
    4. Expected output file/data appearing
    5. Explicit "done" signal from task

    Requires N of M signals for confident completion detection.
    """

    # How many completion signals needed for confident detection
    SIGNALS_FOR_CONFIDENT_COMPLETION = 2

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.events: List[CompletionEvent] = []
        self._completion_signals: Dict[CompletionEventType, bool] = {
            t: False for t in CompletionEventType
        }

    def register_event(self, event: CompletionEvent) -> Dict:
        """
        Register a completion-related event.

        Returns:
            Dict with completion confidence assessment
        """
        self.events.append(event)
        self._completion_signals[event.event_type] = True

        # Count how many different completion signals we have
        signal_count = sum(1 for v in self._completion_signals.values() if v)

        return {
            "completed": signal_count >= self.SIGNALS_FOR_CONFIDENT_COMPLETION,
            "signal_count": signal_count,
            "signals_needed": self.SIGNALS_FOR_CONFIDENT_COMPLETION,
            "signals_received": [
                t.value for t, received in self._completion_signals.items() if received
            ],
            "confidence": signal_count / len(CompletionEventType)
        }

    def check_progress_completion(self, progress: float) -> Optional[CompletionEvent]:
        """Check if progress indicates completion."""
        if progress >= 100:
            return CompletionEvent(
                event_type=CompletionEventType.PROGRESS_100,
                timestamp=time.time(),
                details={"progress": progress}
            )
        return None

    def check_result_available(self, result: Any) -> Optional[CompletionEvent]:
        """Check if result object is available and valid."""
        if result is not None:
            return CompletionEvent(
                event_type=CompletionEventType.RESULT_READY,
                timestamp=time.time(),
                details={"result_type": type(result).__name__}
            )
        return None

    def check_status_flag(self, status: Dict) -> Optional[CompletionEvent]:
        """Check if status flags indicate completion."""
        if not status.get("running", True):
            return CompletionEvent(
                event_type=CompletionEventType.STATUS_FLAG,
                timestamp=time.time(),
                details={"running": False}
            )
        return None

    def check_combination_count(self, completed: int, total: int) -> Optional[CompletionEvent]:
        """Check if all combinations are processed."""
        if total > 0 and completed >= total:
            return CompletionEvent(
                event_type=CompletionEventType.COMBINATION_COUNT,
                timestamp=time.time(),
                details={"completed": completed, "total": total}
            )
        return None


# =============================================================================
# INTEGRATED PROGRESS-BASED WATCHDOG
# =============================================================================

class ProgressBasedWatchdog:
    """
    Integrated watchdog that combines all four approaches.

    NEVER uses absolute time limits.

    Decision flow:
    1. If completion detected (any approach) -> Task done
    2. If velocity positive -> Task healthy, let it continue
    3. If stall detected (multiple approaches agree) -> Investigate
    4. Only abort when confident task is truly stuck
    """

    def __init__(
        self,
        task_id: str,
        status_dict: dict,
        total_combinations: int,
        progress_key: str = "progress",
        abort_key: str = "abort",
        running_key: str = "running",
        result_key: str = "report"
    ):
        self.task_id = task_id
        self.status_dict = status_dict
        self.total_combinations = total_combinations
        self.progress_key = progress_key
        self.abort_key = abort_key
        self.running_key = running_key
        self.result_key = result_key

        # Initialize all detection systems
        self.signal_detector = ProgressSignalDetector(task_id)
        self.velocity_monitor = ProgressVelocityMonitor(task_id, total_combinations)
        self.absence_detector = SignalAbsenceDetector(task_id)
        self.completion_detector = EventDrivenCompletionDetector(task_id)

        self._running = False
        self._aborted = False
        self._abort_reason: Optional[str] = None
        self._last_progress = 0

    async def start(self):
        """Start watching the task."""
        self._running = True

        print(f"[ProgressWatchdog] Started for {self.task_id}")
        print(f"[ProgressWatchdog] Total combinations: {self.total_combinations:,}")
        print(f"[ProgressWatchdog] NO absolute time limits - progress-based only")

        while self._running:
            # Check if task should still be running
            if not self.status_dict.get(self.running_key, True):
                print(f"[ProgressWatchdog] Task {self.task_id} marked as not running")
                break

            # Get current state
            current_progress = self.status_dict.get(self.progress_key, 0)
            result = self.status_dict.get(self.result_key)

            # === Run all detection systems ===

            # 1. Check for completion events
            completion_events = [
                self.completion_detector.check_progress_completion(current_progress),
                self.completion_detector.check_result_available(result),
                self.completion_detector.check_status_flag(self.status_dict),
            ]

            for event in completion_events:
                if event:
                    result = self.completion_detector.register_event(event)
                    if result["completed"]:
                        print(f"[ProgressWatchdog] Completion detected: {result['signals_received']}")
                        self._running = False
                        return

            # 2. Update velocity monitor
            velocity_snapshot = self.velocity_monitor.update(current_progress)
            velocity_trend = self.velocity_monitor.get_velocity_trend()

            # 3. Register progress signal if progress changed
            if current_progress != self._last_progress:
                signal = ProgressSignal(
                    signal_type=ProgressSignalType.PERCENTAGE_UPDATE,
                    timestamp=time.time(),
                    value=current_progress
                )
                signal_result = self.signal_detector.register_signal(signal)
                self.absence_detector.signal_received()

                if signal_result["completed"]:
                    print(f"[ProgressWatchdog] Signal detector: completion detected")
                    self._running = False
                    return

                self._last_progress = current_progress

            # 4. Tick absence detector
            absence_result = self.absence_detector.tick_cycle()

            # === Decision logic ===

            # If making progress, continue regardless of anything else
            if self.velocity_monitor.is_making_progress():
                # Log periodically
                if velocity_snapshot.velocity > 0:
                    remaining = self.velocity_monitor.estimate_remaining_signals()
                    print(f"[ProgressWatchdog] Progress: {current_progress:.1f}% | "
                          f"Velocity: {velocity_trend} | "
                          f"Est. signals remaining: {remaining or 'N/A'}")

            # Only consider abort if multiple systems agree task is stuck
            stall_indicators = [
                self.signal_detector.is_stalled,
                absence_result["warning"],
                velocity_trend == "stalled"
            ]

            stall_count = sum(1 for s in stall_indicators if s)

            if stall_count >= 2:
                # Multiple systems detect stall - warning
                print(f"[ProgressWatchdog] WARNING: Stall detected by {stall_count} systems")
                print(f"[ProgressWatchdog]   Progress stuck at {current_progress:.1f}%")
                print(f"[ProgressWatchdog]   Cycles without progress: {absence_result['cycles_without_signal']}")

            # Only abort if ALL systems agree AND abort threshold reached
            if stall_count >= 3 and absence_result["should_abort"]:
                self._trigger_abort(
                    f"All detection systems agree: stalled at {current_progress:.1f}% "
                    f"for {absence_result['cycles_without_signal']} cycles"
                )
                break

            # Event-driven wait - NOT time-based
            # This wait is just for CPU relief, not timing
            await asyncio.sleep(0.5)

        self._running = False

    def _trigger_abort(self, reason: str):
        """Trigger abort on the monitored task."""
        self._aborted = True
        self._abort_reason = reason
        self.status_dict[self.abort_key] = True
        print(f"[ProgressWatchdog] ABORT: {self.task_id} - {reason}")

    async def stop(self):
        """Stop watching."""
        self._running = False

    @property
    def is_aborted(self) -> bool:
        return self._aborted

    @property
    def abort_reason(self) -> Optional[str]:
        return self._abort_reason


# =============================================================================
# HELPER: HEARTBEAT-BASED CYCLE DRIVER
# =============================================================================

class HeartbeatCycleDriver:
    """
    Drives check cycles through heartbeats, not time.

    External events (other tasks completing, user actions, etc.)
    trigger heartbeats which drive the cycle counter.
    """

    def __init__(self):
        self.detectors: List[SignalAbsenceDetector] = []
        self._cycle_count = 0

    def register_detector(self, detector: SignalAbsenceDetector):
        """Register a detector to receive heartbeats."""
        self.detectors.append(detector)

    def heartbeat(self, source: str = "unknown"):
        """
        Trigger a heartbeat (advances cycle for all detectors).

        Call this when:
        - Another task completes
        - User interacts with system
        - External event occurs
        """
        self._cycle_count += 1
        results = []

        for detector in self.detectors:
            result = detector.tick_cycle()
            result["task_id"] = detector.task_id
            results.append(result)

        return {
            "cycle": self._cycle_count,
            "source": source,
            "detector_results": results
        }


# =============================================================================
# USAGE EXAMPLE FOR VECTORBT OPTIMIZATION
# =============================================================================

async def example_vectorbt_watchdog():
    """
    Example showing how to use progress-based watchdog for VectorBT.

    VectorBT characteristics:
    - 52,800+ combinations (55 strategies x 2 directions x 10 TPs x 10 SLs x ~5 extras)
    - Progress updates every 10 combinations via callback
    - Large batches between updates (sparse signals)
    """

    # Simulated status dict (in real code, this comes from the optimizer)
    status = {
        "running": True,
        "progress": 0,
        "report": None,
        "abort": False
    }

    # Create watchdog with total combinations
    watchdog = ProgressBasedWatchdog(
        task_id="vectorbt_optimization_1",
        status_dict=status,
        total_combinations=52800,
        progress_key="progress",
        abort_key="abort",
        running_key="running",
        result_key="report"
    )

    # Start watchdog in background
    watchdog_task = asyncio.create_task(watchdog.start())

    # Simulate optimization progress
    # In real code, VectorBT would update status["progress"]
    for i in range(100):
        await asyncio.sleep(0.1)  # Simulate work
        status["progress"] = i + 1

        if i == 99:
            # Simulate completion
            status["running"] = False
            status["report"] = {"top_10": []}

    # Wait for watchdog to detect completion
    await watchdog_task

    print(f"Watchdog finished. Aborted: {watchdog.is_aborted}")


# =============================================================================
# KEY DIFFERENCES FROM TIME-BASED WATCHDOG
# =============================================================================

"""
TIME-BASED (current TaskWatchdog):
- Uses DEFAULT_WARNING_SECONDS = 600 (10 minutes)
- Uses DEFAULT_ABORT_SECONDS = 1200 (20 minutes)
- Aborts if no progress for N seconds
- PROBLEM: Legitimate tasks can have sparse updates

PROGRESS-BASED (new design):
- NO time thresholds
- Tracks progress VELOCITY (progress per signal, not per second)
- Uses CYCLES not SECONDS for stall detection
- Cycles are driven by EVENTS (other tasks completing), not time
- Only aborts when multiple independent systems agree task is stuck

BENEFITS:
1. Works regardless of hardware speed
2. Handles sparse VectorBT updates correctly
3. Self-adjusting based on actual system behavior
4. Never aborts a task that's making any progress at all
5. More robust stall detection through consensus
"""


if __name__ == "__main__":
    # Run example
    asyncio.run(example_vectorbt_watchdog())
