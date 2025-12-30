# Progress-Based Watchdog System Design

## Executive Summary

This document proposes a watchdog system that **NEVER uses absolute time limits** for task monitoring. Instead, it relies entirely on progress signals to determine task health and completion.

## Problem Statement

The current `TaskWatchdog` in `backend/services/task_watchdog.py` uses time-based thresholds:
- `DEFAULT_WARNING_SECONDS = 600` (10 minutes without progress = warning)
- `DEFAULT_ABORT_SECONDS = 1200` (20 minutes without progress = abort)

**Why this fails for VectorBT optimizations:**

1. **52,800+ combinations** need processing (55 strategies x 2 directions x 10 TPs x 10 SLs x ~5 extras)
2. **Sparse progress updates** - VectorBT only reports every 10 combinations
3. **Variable processing time** - depends on hardware, system load, data size
4. **Legitimate long runs** - can take 60+ minutes on slower systems

## Proposed Solution: Progress-Based Detection

### Core Philosophy

> If progress is being made, let it run forever.
> Only abort when multiple independent systems agree the task is truly stuck.

### Four Key Approaches

#### 1. Progress Velocity Monitoring

Track **progress per measurement**, not progress per second:

```python
class ProgressVelocityTracker:
    """Track velocity as progress-per-measurement, NOT per-second."""

    MIN_VELOCITY = 0.0001  # 0.0001% per measurement minimum

    def update(self, progress: float) -> VelocityDataPoint:
        velocity = progress - self._last_progress  # Progress gained
        self._last_progress = progress
        return VelocityDataPoint(progress, velocity, acceleration)

    def is_making_progress(self) -> bool:
        return self.get_rolling_velocity() > self.MIN_VELOCITY
```

**Why this works:** If *any* progress is made between measurements, the task is healthy - regardless of wall clock time.

#### 2. Signal-Count Stall Detection

Replace time-based thresholds with measurement-count thresholds:

```python
class SignalCountStallDetector:
    """Detect stalls by counting consecutive unchanged measurements."""

    UNCHANGED_WARNING = 20   # 20 consecutive unchanged = warning
    UNCHANGED_ABORT = 50     # 50 consecutive unchanged = abort

    def update(self, progress: float) -> Dict:
        if progress == self._last_progress_value:
            self._consecutive_unchanged += 1
        else:
            self._consecutive_unchanged = 0
```

**Why this works:** Counts measurements, not seconds. Works identically on fast and slow systems.

#### 3. Multi-Signal Completion Detection

Require multiple independent signals to confirm completion:

```python
class CompletionEventDetector:
    """Require 2+ signals for confident completion."""

    SIGNALS_NEEDED = 2

    signals = [
        PROGRESS_100,      # Progress reached 100%
        RESULT_AVAILABLE,  # Result object available
        RUNNING_FALSE,     # Running flag is False
        ALL_COMBOS_DONE    # All combinations processed
    ]
```

**Why this works:** Prevents false positives from single signal glitches.

#### 4. Cross-Task Comparison

Use other tasks completing as a "tick" mechanism:

```python
class WatchdogCoordinator:
    """If other tasks complete but one is stuck, that's suspicious."""

    async def heartbeat(self):
        # Called when any task completes
        for watchdog in self.watchdogs.values():
            if watchdog.stall_detector.is_stalled:
                log("This task may be stuck - others are completing")
```

**Why this works:** Relative comparison catches stalls that absolute thresholds miss.

## Implementation

### New File: `backend/services/progress_watchdog.py`

Contains:
- `ProgressVelocityTracker` - Velocity monitoring
- `SignalCountStallDetector` - Measurement-count stall detection
- `CompletionEventDetector` - Multi-signal completion
- `ProgressBasedWatchdog` - Integrated watchdog
- `WatchdogCoordinator` - Multi-task coordination

### Usage Example

```python
from services.progress_watchdog import ProgressBasedWatchdog

# Create status dict (same as current)
status = {
    "running": True,
    "progress": 0,
    "report": None,
    "abort": False
}

# Create progress-based watchdog (NO time limits)
watchdog = ProgressBasedWatchdog(
    task_id="vectorbt_opt_1",
    status_dict=status,
    total_combinations=52800,
    progress_key="progress",
    abort_key="abort"
)

# Start monitoring
watchdog_task = asyncio.create_task(watchdog.start())

# ... run optimization ...

# Watchdog will:
# - Detect completion when progress reaches 100% AND result is available
# - Only abort if 50+ consecutive measurements show zero progress
# - NEVER abort a task that's making any progress
```

## Comparison: Old vs New

| Aspect | Time-Based (Old) | Progress-Based (New) |
|--------|------------------|---------------------|
| Stall detection | 20 minutes no progress | 50 consecutive unchanged measurements |
| Abort trigger | Time elapsed | Measurement count |
| Slow hardware | May false-abort | Works correctly |
| Fast hardware | Works | Works |
| Sparse updates | May false-abort | Handles correctly |
| Long tasks | May abort prematurely | Runs until done |

## Key Differences in Code

### Old (time-based):
```python
no_progress_duration = now - self._last_progress_time
if no_progress_duration > self.no_progress_abort:  # 1200 seconds
    self._trigger_abort("No progress for 20 minutes")
```

### New (progress-based):
```python
stall_status = self.stall_detector.update(current_progress)
if stall_status["should_abort"] and not self.velocity_tracker.is_making_progress():
    self._trigger_abort(f"Stalled for {stall_status['consecutive_unchanged']} measurements")
```

## Test Coverage

27 tests verify the new system:
- Velocity calculation is time-independent
- Stall detection counts measurements, not seconds
- Completion requires multiple signals
- Slow-but-progressing tasks are NOT aborted
- VectorBT sparse update patterns are handled
- Long-running tasks are NOT aborted if making progress

## Integration Path

1. **Keep existing `TaskWatchdog`** as fallback for absolute safety limits
2. **Add `ProgressBasedWatchdog`** as primary monitoring
3. **In `autonomous_optimizer.py`**, use progress-based watchdog:

```python
# Replace:
watchdog = TaskWatchdog(task_id, status, timeout_seconds, ...)

# With:
watchdog = ProgressBasedWatchdog(task_id, status, total_combinations, ...)
```

## Files Created

1. **`/backend/services/progress_watchdog.py`** - Production implementation
2. **`/tests/test_progress_watchdog.py`** - 27 test cases (all passing)
3. **`/docs/progress_based_watchdog_design.py`** - Design document with examples
4. **`/docs/PROGRESS_WATCHDOG_DESIGN.md`** - This document

## Conclusion

The progress-based watchdog system provides robust task monitoring for VectorBT optimizations by:

1. **Eliminating time dependency** - Works on any hardware speed
2. **Using multiple detection methods** - Prevents false positives/negatives
3. **Requiring consensus for abort** - Only aborts truly stuck tasks
4. **Supporting sparse updates** - Handles VectorBT's batch processing

The fundamental guarantee: **If a task is making ANY progress, it will NEVER be aborted.**
