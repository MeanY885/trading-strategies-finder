"""
DYNAMIC THROUGHPUT MONITOR
==========================
Monitors combinations/second and automatically adjusts concurrency to
maximize TOTAL throughput - no hardcoded thresholds.

Key insight: Adding more concurrent tasks doesn't always increase total throughput.
At some point, resource contention causes each task to slow down so much that
total throughput actually DECREASES.

Algorithm:
1. Start conservative (1-2 concurrent tasks)
2. Measure total throughput (sum of all tasks' comb/sec)
3. When adding a task: does total throughput increase or decrease?
4. If decreased → reduce concurrency (we passed the sweet spot)
5. If increased → can try adding more (still scaling well)

This finds the optimal concurrency for ANY system automatically.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from collections import deque
from logging_config import log


@dataclass
class TaskMetrics:
    """Metrics for a single optimization task."""
    task_id: str
    pair: str
    started_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    last_completed: int = 0
    combinations_completed: int = 0
    combinations_total: int = 0

    # Rolling throughput samples (comb/sec for recent intervals)
    throughput_samples: deque = field(default_factory=lambda: deque(maxlen=20))

    def update(self, completed: int, total: int) -> float:
        """Update with new progress. Returns current throughput."""
        now = time.time()
        elapsed_since_last = now - self.last_update

        current_throughput = 0.0
        if elapsed_since_last > 0.5 and completed > self.last_completed:
            # Calculate comb/sec for this interval
            combos_this_interval = completed - self.last_completed
            current_throughput = combos_this_interval / elapsed_since_last
            self.throughput_samples.append(current_throughput)

        self.last_completed = self.combinations_completed
        self.combinations_completed = completed
        self.combinations_total = total
        self.last_update = now

        return current_throughput

    @property
    def avg_throughput(self) -> float:
        """Average comb/sec over recent samples."""
        if not self.throughput_samples:
            return 0.0
        return sum(self.throughput_samples) / len(self.throughput_samples)

    @property
    def current_throughput(self) -> float:
        """Most recent throughput sample."""
        if not self.throughput_samples:
            return 0.0
        return self.throughput_samples[-1]


@dataclass
class ThroughputSnapshot:
    """Snapshot of throughput at a specific concurrency level."""
    timestamp: float
    concurrent_tasks: int
    total_throughput: float  # Sum of all tasks' comb/sec
    per_task_throughput: float  # Average per task


class DynamicThroughputMonitor:
    """
    Monitors throughput and dynamically adjusts concurrency based on
    observed performance - no hardcoded thresholds.
    """

    # How many samples to collect before making decisions
    MIN_SAMPLES_FOR_DECISION = 5

    # Minimum time between concurrency adjustments (seconds)
    ADJUSTMENT_COOLDOWN = 30

    # How much total throughput must drop before reducing concurrency (%)
    THROUGHPUT_DROP_THRESHOLD = 0.15  # 15% drop triggers reduction

    # How much throughput must increase to justify adding more tasks (%)
    THROUGHPUT_GAIN_THRESHOLD = 0.10  # Need 10% gain to add more

    # Absolute limits
    MIN_CONCURRENT = 1
    MAX_CONCURRENT = 16  # Hard ceiling, system will find optimal below this

    def __init__(self):
        self.tasks: Dict[str, TaskMetrics] = {}
        self.current_max_concurrent: int = 2  # Start conservative
        self.last_adjustment_time: float = 0

        # History of throughput at different concurrency levels
        # Key: concurrent_count, Value: list of total_throughput samples
        self.throughput_by_concurrency: Dict[int, deque] = {}

        # Recent snapshots for analysis
        self.snapshots: deque = deque(maxlen=100)

        self._lock = asyncio.Lock()

        log("[Throughput] Dynamic monitor initialized (no static thresholds)")

    def register_task(self, task_id: str, pair: str, total_combinations: int):
        """Register a new optimization task."""
        self.tasks[task_id] = TaskMetrics(
            task_id=task_id,
            pair=pair,
            combinations_total=total_combinations
        )
        log(f"[Throughput] Task registered: {pair} ({total_combinations:,} combinations)")

    def update_task(self, task_id: str, completed: int, total: int) -> Optional[float]:
        """Update task progress. Returns current throughput if available."""
        if task_id in self.tasks:
            return self.tasks[task_id].update(completed, total)
        return None

    def unregister_task(self, task_id: str) -> Optional[Dict]:
        """Remove completed task. Returns summary stats."""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        duration = time.time() - task.started_at
        avg_throughput = task.avg_throughput

        summary = {
            "task_id": task_id,
            "pair": task.pair,
            "duration_sec": duration,
            "combinations": task.combinations_completed,
            "avg_throughput": avg_throughput,
            "concurrent_at_completion": len(self.tasks)
        }

        log(f"[Throughput] Task completed: {task.pair} - "
            f"{task.combinations_completed:,} in {duration:.0f}s "
            f"({avg_throughput:,.0f}/sec)")

        del self.tasks[task_id]
        return summary

    def get_total_throughput(self) -> float:
        """Get combined throughput of all running tasks."""
        return sum(t.avg_throughput for t in self.tasks.values())

    def get_per_task_throughput(self) -> float:
        """Get average throughput per task."""
        if not self.tasks:
            return 0.0
        return self.get_total_throughput() / len(self.tasks)

    def _record_snapshot(self):
        """Record current throughput snapshot."""
        concurrent = len(self.tasks)
        if concurrent == 0:
            return

        total = self.get_total_throughput()
        per_task = self.get_per_task_throughput()

        snapshot = ThroughputSnapshot(
            timestamp=time.time(),
            concurrent_tasks=concurrent,
            total_throughput=total,
            per_task_throughput=per_task
        )
        self.snapshots.append(snapshot)

        # Record in concurrency-indexed history
        if concurrent not in self.throughput_by_concurrency:
            self.throughput_by_concurrency[concurrent] = deque(maxlen=50)
        self.throughput_by_concurrency[concurrent].append(total)

    def _get_avg_throughput_for_concurrency(self, concurrent: int) -> Optional[float]:
        """Get average total throughput observed at a specific concurrency level."""
        if concurrent not in self.throughput_by_concurrency:
            return None
        samples = self.throughput_by_concurrency[concurrent]
        if len(samples) < self.MIN_SAMPLES_FOR_DECISION:
            return None
        return sum(samples) / len(samples)

    async def evaluate_and_adjust(self) -> Dict:
        """
        Evaluate current performance and adjust concurrency if needed.

        Returns dict with recommendation and metrics.
        """
        async with self._lock:
            now = time.time()

            # Record current state
            self._record_snapshot()

            running_count = len(self.tasks)
            total_throughput = self.get_total_throughput()
            per_task = self.get_per_task_throughput()

            result = {
                "current_concurrent": running_count,
                "max_concurrent": self.current_max_concurrent,
                "total_throughput": total_throughput,
                "per_task_throughput": per_task,
                "adjusted": False,
                "reason": "Monitoring"
            }

            # Need tasks running to make decisions
            if running_count == 0:
                result["reason"] = "No tasks running"
                return result

            # Respect cooldown
            if now - self.last_adjustment_time < self.ADJUSTMENT_COOLDOWN:
                result["reason"] = f"Cooldown ({int(self.ADJUSTMENT_COOLDOWN - (now - self.last_adjustment_time))}s remaining)"
                return result

            # Need enough data at current concurrency level
            current_avg = self._get_avg_throughput_for_concurrency(running_count)
            if current_avg is None:
                result["reason"] = f"Collecting data ({len(self.throughput_by_concurrency.get(running_count, []))}/{self.MIN_SAMPLES_FOR_DECISION} samples)"
                return result

            # Compare with other concurrency levels we've observed
            adjustment = self._calculate_adjustment(running_count, current_avg)

            if adjustment != 0:
                new_max = max(self.MIN_CONCURRENT,
                             min(self.MAX_CONCURRENT, self.current_max_concurrent + adjustment))

                if new_max != self.current_max_concurrent:
                    old_max = self.current_max_concurrent
                    self.current_max_concurrent = new_max
                    self.last_adjustment_time = now

                    direction = "↑" if adjustment > 0 else "↓"
                    log(f"[Throughput] {direction} Concurrency adjusted: {old_max} → {new_max}")
                    log(f"[Throughput]   Total throughput: {total_throughput:,.0f}/sec")

                    result["adjusted"] = True
                    result["reason"] = f"{'Increased' if adjustment > 0 else 'Decreased'} based on throughput analysis"
                    result["max_concurrent"] = new_max

            return result

    def _calculate_adjustment(self, current_concurrent: int, current_throughput: float) -> int:
        """
        Calculate whether to increase, decrease, or maintain concurrency.

        Returns: +1 (increase), -1 (decrease), or 0 (maintain)
        """
        # Check if we have data from lower concurrency
        lower_throughput = self._get_avg_throughput_for_concurrency(current_concurrent - 1)

        # Check if we have data from higher concurrency
        higher_throughput = self._get_avg_throughput_for_concurrency(current_concurrent + 1)

        # Case 1: Current throughput is significantly lower than when we had fewer tasks
        # This means we've exceeded optimal concurrency
        if lower_throughput is not None:
            drop_pct = (lower_throughput - current_throughput) / lower_throughput
            if drop_pct > self.THROUGHPUT_DROP_THRESHOLD:
                log(f"[Throughput] Throughput dropped {drop_pct*100:.1f}% vs {current_concurrent-1} tasks - reducing")
                return -1

        # Case 2: We've tried higher concurrency and it was worse
        # Stay at current level
        if higher_throughput is not None:
            if higher_throughput < current_throughput * (1 - self.THROUGHPUT_DROP_THRESHOLD):
                # Higher concurrency was worse, we're at or near optimal
                return 0

        # Case 3: Current is better than lower, and we haven't tried higher yet
        # OR higher was similar/better - try increasing
        if lower_throughput is not None:
            gain_pct = (current_throughput - lower_throughput) / lower_throughput if lower_throughput > 0 else 0
            if gain_pct > self.THROUGHPUT_GAIN_THRESHOLD:
                # Still scaling well, try adding more
                if current_concurrent < self.current_max_concurrent:
                    return 0  # Already at max, can't increase
                if current_concurrent >= self.MAX_CONCURRENT:
                    return 0  # At hard ceiling
                log(f"[Throughput] Throughput improved {gain_pct*100:.1f}% - trying more concurrency")
                return +1

        # Case 4: First run or not enough data - start conservative, gradually increase
        if lower_throughput is None and current_concurrent < 4:
            # Haven't established baseline yet, cautiously increase
            return +1

        return 0

    def get_recommended_max_concurrent(self) -> int:
        """Get the current recommended max concurrent tasks."""
        return self.current_max_concurrent

    def get_status(self) -> Dict:
        """Get full status for API/logging."""
        return {
            "recommended_max_concurrent": self.current_max_concurrent,
            "running_tasks": len(self.tasks),
            "total_throughput_per_sec": self.get_total_throughput(),
            "per_task_throughput": self.get_per_task_throughput(),
            "throughput_history": {
                k: sum(v)/len(v) if v else 0
                for k, v in self.throughput_by_concurrency.items()
            },
            "tasks": {
                tid: {
                    "pair": t.pair,
                    "throughput": t.avg_throughput,
                    "completed": t.combinations_completed,
                    "total": t.combinations_total
                }
                for tid, t in self.tasks.items()
            }
        }


# Global instance
throughput_monitor = DynamicThroughputMonitor()
