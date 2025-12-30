"""
THROUGHPUT MONITOR
==================
Monitors combinations/second across running optimizations and dynamically
adjusts concurrency to maximize total throughput.

Key insight: Running fewer concurrent tasks often gives HIGHER total throughput
because each task gets more resources and runs faster.

Example:
- 4 tasks × 5,000 comb/sec = 20,000 total comb/sec
- 2 tasks × 15,000 comb/sec = 30,000 total comb/sec  ← BETTER

The monitor tracks performance and automatically reduces concurrency when
throughput drops, finding the optimal balance.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, List
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
    combinations_completed: int = 0
    combinations_total: int = 0

    # Rolling window of comb/sec samples
    throughput_samples: deque = field(default_factory=lambda: deque(maxlen=10))

    def update(self, completed: int, total: int):
        """Update with new progress."""
        now = time.time()
        elapsed_since_last = now - self.last_update

        if elapsed_since_last > 0 and completed > self.combinations_completed:
            # Calculate comb/sec for this interval
            combos_this_interval = completed - self.combinations_completed
            comb_per_sec = combos_this_interval / elapsed_since_last
            self.throughput_samples.append(comb_per_sec)

        self.combinations_completed = completed
        self.combinations_total = total
        self.last_update = now

    @property
    def avg_throughput(self) -> float:
        """Average comb/sec over recent samples."""
        if not self.throughput_samples:
            return 0.0
        return sum(self.throughput_samples) / len(self.throughput_samples)

    @property
    def progress_pct(self) -> float:
        """Progress percentage."""
        if self.combinations_total == 0:
            return 0.0
        return (self.combinations_completed / self.combinations_total) * 100


class ThroughputMonitor:
    """
    Monitors throughput across all running optimizations and recommends
    optimal concurrency level.
    """

    # Thresholds for dynamic adjustment
    MIN_THROUGHPUT_PER_TASK = 500    # comb/sec - below this, task is "slow"
    HEALTHY_THROUGHPUT = 2000        # comb/sec - good performance
    EXCELLENT_THROUGHPUT = 5000      # comb/sec - excellent performance

    # Concurrency limits
    MIN_CONCURRENT = 1
    MAX_CONCURRENT = 8

    # How long to wait before adjusting (seconds)
    ADJUSTMENT_COOLDOWN = 60

    # History for analysis
    HISTORY_SIZE = 100

    def __init__(self):
        self.tasks: Dict[str, TaskMetrics] = {}
        self.current_max_concurrent: int = 4  # Start conservative
        self.last_adjustment_time: float = 0
        self.adjustment_history: deque = deque(maxlen=self.HISTORY_SIZE)
        self._lock = asyncio.Lock()

    def register_task(self, task_id: str, pair: str, total_combinations: int):
        """Register a new optimization task."""
        self.tasks[task_id] = TaskMetrics(
            task_id=task_id,
            pair=pair,
            combinations_total=total_combinations
        )
        log(f"[Throughput] Registered task {task_id} ({pair}): {total_combinations:,} combinations")

    def update_task(self, task_id: str, completed: int, total: int):
        """Update task progress."""
        if task_id in self.tasks:
            self.tasks[task_id].update(completed, total)

    def unregister_task(self, task_id: str):
        """Remove completed/cancelled task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            duration = time.time() - task.started_at
            avg_throughput = task.avg_throughput

            log(f"[Throughput] Task {task_id} completed: "
                f"{task.combinations_completed:,} combos in {duration:.1f}s "
                f"(avg {avg_throughput:.0f} comb/sec)")

            # Record in history
            self.adjustment_history.append({
                "task_id": task_id,
                "pair": task.pair,
                "duration": duration,
                "combinations": task.combinations_completed,
                "avg_throughput": avg_throughput,
                "concurrent_at_time": len(self.tasks),
                "timestamp": datetime.now().isoformat()
            })

            del self.tasks[task_id]

    def get_total_throughput(self) -> float:
        """Get combined throughput of all running tasks."""
        return sum(t.avg_throughput for t in self.tasks.values())

    def get_avg_throughput_per_task(self) -> float:
        """Get average throughput per task."""
        if not self.tasks:
            return 0.0
        return self.get_total_throughput() / len(self.tasks)

    def get_slowest_task(self) -> Optional[TaskMetrics]:
        """Get the slowest running task."""
        if not self.tasks:
            return None
        return min(self.tasks.values(), key=lambda t: t.avg_throughput)

    def should_reduce_concurrency(self) -> bool:
        """Check if we should reduce concurrent tasks."""
        if len(self.tasks) <= self.MIN_CONCURRENT:
            return False

        avg_throughput = self.get_avg_throughput_per_task()

        # If average throughput per task is too low, reduce concurrency
        if avg_throughput < self.MIN_THROUGHPUT_PER_TASK:
            return True

        return False

    def should_increase_concurrency(self) -> bool:
        """Check if we can safely increase concurrent tasks."""
        if len(self.tasks) >= self.current_max_concurrent:
            return False

        avg_throughput = self.get_avg_throughput_per_task()

        # Only increase if current tasks are performing excellently
        if avg_throughput > self.EXCELLENT_THROUGHPUT:
            return True

        return False

    async def evaluate_and_adjust(self) -> Dict:
        """
        Evaluate current performance and return adjustment recommendation.

        Returns dict with:
        - recommended_concurrent: Suggested max concurrent
        - reason: Why this recommendation
        - metrics: Current performance metrics
        """
        async with self._lock:
            now = time.time()

            # Don't adjust too frequently
            if now - self.last_adjustment_time < self.ADJUSTMENT_COOLDOWN:
                return {
                    "recommended_concurrent": self.current_max_concurrent,
                    "reason": "Cooldown active",
                    "adjusted": False,
                    "metrics": self._get_current_metrics()
                }

            running_count = len(self.tasks)
            total_throughput = self.get_total_throughput()
            avg_per_task = self.get_avg_throughput_per_task()

            recommendation = self.current_max_concurrent
            reason = "No change needed"
            adjusted = False

            if running_count == 0:
                # No tasks running, nothing to evaluate
                pass
            elif self.should_reduce_concurrency():
                # Performance is poor, reduce concurrency
                recommendation = max(self.MIN_CONCURRENT, self.current_max_concurrent - 1)
                reason = f"Low throughput ({avg_per_task:.0f} comb/sec/task) - reducing concurrency"
                adjusted = True

                log(f"[Throughput] ⚠️ REDUCING concurrency: {self.current_max_concurrent} → {recommendation}")
                log(f"[Throughput]   Reason: {reason}")

            elif self.should_increase_concurrency() and running_count >= self.current_max_concurrent:
                # Performance is excellent, can try increasing
                recommendation = min(self.MAX_CONCURRENT, self.current_max_concurrent + 1)
                reason = f"Excellent throughput ({avg_per_task:.0f} comb/sec/task) - can increase"
                adjusted = True

                log(f"[Throughput] ✅ INCREASING concurrency: {self.current_max_concurrent} → {recommendation}")
                log(f"[Throughput]   Reason: {reason}")

            if adjusted:
                self.current_max_concurrent = recommendation
                self.last_adjustment_time = now

            return {
                "recommended_concurrent": recommendation,
                "reason": reason,
                "adjusted": adjusted,
                "metrics": self._get_current_metrics()
            }

    def _get_current_metrics(self) -> Dict:
        """Get current performance metrics."""
        return {
            "running_tasks": len(self.tasks),
            "current_max_concurrent": self.current_max_concurrent,
            "total_throughput": self.get_total_throughput(),
            "avg_throughput_per_task": self.get_avg_throughput_per_task(),
            "tasks": {
                task_id: {
                    "pair": t.pair,
                    "progress_pct": t.progress_pct,
                    "avg_throughput": t.avg_throughput,
                    "combinations": f"{t.combinations_completed:,}/{t.combinations_total:,}"
                }
                for task_id, t in self.tasks.items()
            }
        }

    def get_status(self) -> Dict:
        """Get full status for API/UI."""
        return {
            "current_max_concurrent": self.current_max_concurrent,
            "running_tasks": len(self.tasks),
            "total_throughput_comb_sec": self.get_total_throughput(),
            "avg_per_task_comb_sec": self.get_avg_throughput_per_task(),
            "thresholds": {
                "min_throughput": self.MIN_THROUGHPUT_PER_TASK,
                "healthy": self.HEALTHY_THROUGHPUT,
                "excellent": self.EXCELLENT_THROUGHPUT
            },
            "tasks": self._get_current_metrics()["tasks"],
            "recent_history": list(self.adjustment_history)[-10:]
        }


# Global instance
throughput_monitor = ThroughputMonitor()
