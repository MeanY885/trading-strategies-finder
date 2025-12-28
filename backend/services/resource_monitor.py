"""
RESOURCE MONITOR
================
Dynamic resource monitoring and adaptive concurrency scaling.
Extracted from main.py for better modularity.
"""
import os
import time
import threading
from typing import Dict, Tuple, List
import psutil
import logging

from config import RESOURCE_THRESHOLDS

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Monitors system resources in real-time and provides adaptive concurrency recommendations.
    Tracks CPU usage, memory availability, and adjusts max workers dynamically.
    """

    def __init__(self):
        self.cpu_cores = os.cpu_count() or 4
        self.mem_total_gb = psutil.virtual_memory().total / (1024**3)

        # Resource thresholds from config
        self.cpu_target_usage = RESOURCE_THRESHOLDS["cpu_target_usage"]
        self.cpu_max_usage = RESOURCE_THRESHOLDS["cpu_max_usage"]
        self.mem_min_available_gb = RESOURCE_THRESHOLDS["mem_min_available_gb"]
        self.mem_per_worker_gb = RESOURCE_THRESHOLDS["mem_per_worker_gb"]
        self.sample_window = RESOURCE_THRESHOLDS["sample_window"]
        self.adjustment_cooldown = RESOURCE_THRESHOLDS["adjustment_cooldown"]

        # Tracking
        self.cpu_samples: List[float] = []
        self.last_adjustment = 0

        # Current state
        self._current_max = self._calculate_initial_max()
        self._lock = threading.Lock()

    def _calculate_initial_max(self) -> int:
        """Calculate initial max workers based on system specs."""
        mem = psutil.virtual_memory()
        available_mem_gb = mem.available / (1024**3)

        # Memory-based limit
        mem_workers = max(1, int((available_mem_gb - self.mem_min_available_gb) / self.mem_per_worker_gb))

        # CPU-based limit - scale with core count
        if self.cpu_cores >= 24:
            cpu_workers = self.cpu_cores - 4
        elif self.cpu_cores >= 16:
            cpu_workers = self.cpu_cores - 3
        elif self.cpu_cores >= 8:
            cpu_workers = self.cpu_cores - 2
        else:
            cpu_workers = max(1, self.cpu_cores - 1)

        # No artificial cap - let the system decide based on actual resources
        optimal = min(cpu_workers, mem_workers)
        return max(2, optimal)

    def get_current_resources(self) -> Dict:
        """Get current system resource state."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()

        return {
            "cpu_cores": self.cpu_cores,
            "cpu_percent": round(cpu_percent, 1),
            "cpu_per_core": [round(x, 1) for x in psutil.cpu_percent(percpu=True, interval=0)],
            "memory_total_gb": round(self.mem_total_gb, 1),
            "memory_available_gb": round(mem.available / (1024**3), 1),
            "memory_used_percent": round(mem.percent, 1),
            "memory_free_gb": round((mem.total - mem.used) / (1024**3), 1),
        }

    def sample_cpu(self) -> None:
        """Take a CPU sample and add to rolling window."""
        cpu = psutil.cpu_percent(interval=0)
        with self._lock:
            self.cpu_samples.append(cpu)
            if len(self.cpu_samples) > self.sample_window:
                self.cpu_samples.pop(0)

    def get_avg_cpu(self) -> float:
        """Get average CPU usage from samples."""
        with self._lock:
            if not self.cpu_samples:
                return 0
            return sum(self.cpu_samples) / len(self.cpu_samples)

    def calculate_optimal_workers(self, current_workers: int) -> int:
        """
        Calculate optimal number of workers based on current resource usage.
        Returns recommended worker count (may be same, higher, or lower).
        """
        resources = self.get_current_resources()
        avg_cpu = self.get_avg_cpu()

        # Memory check - hard limit
        available_mem = resources["memory_available_gb"]
        mem_headroom = available_mem - self.mem_min_available_gb
        mem_max_workers = max(1, int(mem_headroom / self.mem_per_worker_gb) + current_workers)

        # CPU-based scaling
        if avg_cpu < 30 and current_workers > 0:
            # Very underutilized - can add more workers
            cpu_suggested = min(current_workers + 4, self.cpu_cores - 2)
        elif avg_cpu < self.cpu_target_usage:
            # Room to grow
            headroom = self.cpu_target_usage - avg_cpu
            additional = max(1, int(headroom / 10))
            cpu_suggested = current_workers + additional
        elif avg_cpu > self.cpu_max_usage:
            # Overloaded - scale down
            overage = avg_cpu - self.cpu_target_usage
            reduce = max(1, int(overage / 10))
            cpu_suggested = max(2, current_workers - reduce)
        else:
            # In target range
            cpu_suggested = current_workers

        # Apply limits
        optimal = min(cpu_suggested, mem_max_workers, self.cpu_cores - 2)
        optimal = max(2, optimal)

        return optimal

    def should_scale(self, current_running: int) -> Tuple[bool, int, str]:
        """
        Determine if scaling should occur.
        Returns (should_scale: bool, new_target: int, reason: str)
        """
        current_time = time.time()
        if current_time - self.last_adjustment < self.adjustment_cooldown:
            return False, self._current_max, "Cooldown active"

        optimal = self.calculate_optimal_workers(current_running)

        if optimal != self._current_max:
            diff = optimal - self._current_max
            if abs(diff) >= 2 or (optimal > self._current_max and current_running >= self._current_max):
                reason = f"Scaling {'up' if diff > 0 else 'down'}: {self._current_max} → {optimal}"
                return True, optimal, reason

        return False, self._current_max, "No change needed"

    def apply_scaling(self, new_max: int) -> None:
        """Apply new max worker limit."""
        with self._lock:
            old_max = self._current_max
            self._current_max = new_max
            self.last_adjustment = time.time()
        logger.info(f"[ResourceMonitor] Scaled workers: {old_max} → {new_max}")

    @property
    def current_max(self) -> int:
        """Get current max workers."""
        with self._lock:
            return self._current_max

    def get_status(self, current_running: int = 0) -> Dict:
        """Get full resource monitor status."""
        resources = self.get_current_resources()

        return {
            **resources,
            "current_max_workers": self._current_max,
            "current_running": current_running,
            "avg_cpu_usage": round(self.get_avg_cpu(), 1),
            "cpu_target": self.cpu_target_usage,
            "cpu_max_threshold": self.cpu_max_usage,
            "mem_min_available_gb": self.mem_min_available_gb,
            "mem_per_worker_gb": self.mem_per_worker_gb,
            "can_scale_up": current_running >= self._current_max and self.get_avg_cpu() < self.cpu_target_usage,
            "recommended_workers": self.calculate_optimal_workers(current_running),
        }


# Singleton instance
resource_monitor = ResourceMonitor()
