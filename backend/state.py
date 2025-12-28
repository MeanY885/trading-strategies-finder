"""
STATE MANAGEMENT MODULE
=======================
Thread-safe global state management for the BTCGBP ML Optimizer.
Replaces scattered global variables with a centralized, locked state container.
"""
import threading
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import copy

from config import MAX_HISTORY_SIZE, MAX_QUEUE_COMPLETED, MAX_SKIPPED_VALIDATIONS


class ThreadSafeDict:
    """
    A thread-safe dictionary wrapper with locking.
    Provides atomic read/write operations for state management.
    """

    def __init__(self, initial_data: Optional[Dict] = None):
        self._data = initial_data or {}
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def update(self, updates: Dict) -> None:
        with self._lock:
            self._data.update(updates)

    def get_all(self) -> Dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data


@dataclass
class AppState:
    """
    Centralized application state with thread-safe access.
    All state modifications should go through this class.
    """
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # Data loading status
    data_status: Dict = field(default_factory=lambda: {
        "loaded": False,
        "rows": 0,
        "start_date": None,
        "end_date": None,
        "message": "No data loaded",
        "stats": None,
        "progress": 0,
        "fetching": False
    })

    # Manual optimization status
    unified_status: Dict = field(default_factory=lambda: {
        "running": False,
        "progress": 0,
        "message": "Ready",
        "report": None
    })

    # Elite validation status
    elite_status: Dict = field(default_factory=lambda: {
        "running": False,
        "paused": False,
        "current_strategy_id": None,
        "processed": 0,
        "total": 0,
        "message": "Idle",
        "auto_running": False
    })

    # Autonomous optimizer status
    autonomous_status: Dict = field(default_factory=lambda: {
        "auto_running": False,
        "running": False,
        "paused": False,
        "enabled": False,
        "message": "Idle",
        "progress": 0,
        "current_source": None,
        "current_pair": None,
        "current_period": None,
        "current_timeframe": None,
        "current_granularity": None,
        "trial_current": 0,
        "trial_total": 0,
        "data_validation": None,
        "cycle_index": 0,
        "total_combinations": 0,
        "completed_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "last_result": None,
        "last_completed_at": None,
        "best_strategy_found": None,
        "skipped_validations": [],
        "queue_completed": [],
        "queue_current": None,
        "combinations_list": [],
        "parallel_running": [],
        "parallel_count": 0,
        "max_parallel": 4,
    })

    # Running optimizations tracking (for parallel processing)
    running_optimizations: Dict = field(default_factory=dict)

    # History of autonomous runs
    autonomous_history: List = field(default_factory=list)

    # Current optimization status reference (for abort signaling)
    current_optimization_status: Optional[Dict] = None

    # DataFrame storage (for loaded data)
    current_df: Any = None

    # ==========================================================================
    # DATA STATUS METHODS
    # ==========================================================================

    def update_data_status(self, **kwargs) -> None:
        """Thread-safe update of data status."""
        with self._lock:
            self.data_status.update(kwargs)

    def get_data_status(self) -> Dict:
        """Thread-safe get of data status."""
        with self._lock:
            return copy.deepcopy(self.data_status)

    def set_dataframe(self, df: Any) -> None:
        """Store the current dataframe."""
        with self._lock:
            self.current_df = df

    def get_dataframe(self) -> Any:
        """Get the current dataframe."""
        with self._lock:
            return self.current_df

    # ==========================================================================
    # UNIFIED (MANUAL) OPTIMIZATION STATUS METHODS
    # ==========================================================================

    def update_unified_status(self, **kwargs) -> None:
        """Thread-safe update of unified optimization status."""
        with self._lock:
            self.unified_status.update(kwargs)

    def get_unified_status(self) -> Dict:
        """Thread-safe get of unified optimization status."""
        with self._lock:
            return copy.deepcopy(self.unified_status)

    def is_optimization_running(self) -> bool:
        """Check if manual optimization is running."""
        with self._lock:
            return self.unified_status.get("running", False)

    # ==========================================================================
    # ELITE VALIDATION STATUS METHODS
    # ==========================================================================

    def update_elite_status(self, **kwargs) -> None:
        """Thread-safe update of elite validation status."""
        with self._lock:
            self.elite_status.update(kwargs)

    def get_elite_status(self) -> Dict:
        """Thread-safe get of elite validation status."""
        with self._lock:
            return copy.deepcopy(self.elite_status)

    def is_elite_running(self) -> bool:
        """Check if elite validation is running."""
        with self._lock:
            return self.elite_status.get("running", False)

    def is_elite_auto_running(self) -> bool:
        """Check if Elite auto-validation service is running."""
        with self._lock:
            return self.elite_status.get("auto_running", False)

    # ==========================================================================
    # UNIFIED STATUS METHODS
    # ==========================================================================

    def is_unified_running(self) -> bool:
        """Check if unified (manual) optimizer is running."""
        with self._lock:
            return self.unified_status.get("running", False)

    # ==========================================================================
    # AUTONOMOUS OPTIMIZER STATUS METHODS
    # ==========================================================================

    def update_autonomous_status(self, **kwargs) -> None:
        """Thread-safe update of autonomous optimizer status."""
        with self._lock:
            self.autonomous_status.update(kwargs)

    def get_autonomous_status(self) -> Dict:
        """Thread-safe get of autonomous optimizer status."""
        with self._lock:
            return copy.deepcopy(self.autonomous_status)

    def is_autonomous_enabled(self) -> bool:
        """Check if autonomous optimizer is enabled."""
        with self._lock:
            return self.autonomous_status.get("enabled", False)

    def is_autonomous_running(self) -> bool:
        """Check if autonomous optimizer is running."""
        with self._lock:
            return self.autonomous_status.get("auto_running", False)

    def add_skipped_validation(self, entry: Dict) -> None:
        """Add a skipped validation entry (capped at MAX_SKIPPED_VALIDATIONS)."""
        with self._lock:
            self.autonomous_status["skipped_validations"].insert(0, entry)
            if len(self.autonomous_status["skipped_validations"]) > MAX_SKIPPED_VALIDATIONS:
                self.autonomous_status["skipped_validations"] = \
                    self.autonomous_status["skipped_validations"][:MAX_SKIPPED_VALIDATIONS]

    def add_queue_completed(self, entry: Dict) -> None:
        """Add a completed queue entry (capped at MAX_QUEUE_COMPLETED)."""
        with self._lock:
            self.autonomous_status["queue_completed"].insert(0, entry)
            self.autonomous_status["queue_completed"] = \
                self.autonomous_status["queue_completed"][:MAX_QUEUE_COMPLETED]

    # ==========================================================================
    # RUNNING OPTIMIZATIONS TRACKING
    # ==========================================================================

    def add_running_optimization(self, combo_id: str, status: Dict) -> None:
        """Add a running optimization to tracking."""
        with self._lock:
            self.running_optimizations[combo_id] = status
            self.autonomous_status["parallel_running"] = list(self.running_optimizations.values())
            self.autonomous_status["parallel_count"] = len(self.running_optimizations)

    def update_running_optimization(self, combo_id: str, **kwargs) -> None:
        """Update a running optimization's status."""
        with self._lock:
            if combo_id in self.running_optimizations:
                self.running_optimizations[combo_id].update(kwargs)
                self.autonomous_status["parallel_running"] = list(self.running_optimizations.values())

    def remove_running_optimization(self, combo_id: str) -> None:
        """Remove a completed optimization from tracking."""
        with self._lock:
            if combo_id in self.running_optimizations:
                del self.running_optimizations[combo_id]
            self.autonomous_status["parallel_running"] = list(self.running_optimizations.values())
            self.autonomous_status["parallel_count"] = len(self.running_optimizations)

    def get_running_count(self) -> int:
        """Get count of running optimizations."""
        with self._lock:
            return len(self.running_optimizations)

    def clear_running_optimizations(self) -> None:
        """Clear all running optimization tracking."""
        with self._lock:
            self.running_optimizations.clear()
            self.autonomous_status["parallel_running"] = []
            self.autonomous_status["parallel_count"] = 0

    # ==========================================================================
    # HISTORY MANAGEMENT
    # ==========================================================================

    def add_to_history(self, entry: Dict) -> None:
        """Add an entry to autonomous runs history (capped at MAX_HISTORY_SIZE)."""
        with self._lock:
            self.autonomous_history.insert(0, entry)
            if len(self.autonomous_history) > MAX_HISTORY_SIZE:
                self.autonomous_history = self.autonomous_history[:MAX_HISTORY_SIZE]

    def get_history(self) -> List:
        """Get autonomous runs history."""
        with self._lock:
            return copy.deepcopy(self.autonomous_history)

    # ==========================================================================
    # CURRENT OPTIMIZATION STATUS (for abort signaling)
    # ==========================================================================

    def set_current_optimization(self, status: Optional[Dict]) -> None:
        """Set the current optimization status reference."""
        with self._lock:
            self.current_optimization_status = status

    def get_current_optimization(self) -> Optional[Dict]:
        """Get the current optimization status reference."""
        with self._lock:
            return self.current_optimization_status

    def signal_abort(self) -> bool:
        """Signal abort to current optimization. Returns True if successful."""
        with self._lock:
            if self.current_optimization_status is not None:
                self.current_optimization_status["abort"] = True
                return True
            return False

    # ==========================================================================
    # FULL STATE SNAPSHOT
    # ==========================================================================

    def get_full_state(self) -> Dict:
        """Get a complete snapshot of all state for WebSocket broadcast."""
        with self._lock:
            # Get autonomous status but exclude large arrays to keep payload small
            autonomous = copy.deepcopy(self.autonomous_status)
            autonomous.pop("combinations_list", None)
            autonomous.pop("queue_completed", None)

            return {
                "data": copy.deepcopy(self.data_status),
                "optimization": copy.deepcopy(self.unified_status),
                "autonomous": autonomous,
                "elite": copy.deepcopy(self.elite_status),
            }


# =============================================================================
# CONCURRENCY CONFIGURATION
# =============================================================================

class ConcurrencyConfig:
    """Thread-safe concurrency configuration."""

    def __init__(self):
        self._lock = threading.RLock()
        self._config = {
            "max_concurrent": 4,
            "auto_detected": 4,
            "cpu_cores": 4,
            "memory_total_gb": 8.0,
            "memory_available_gb": 4.0,
            "parallel_enabled": True,
            "elite_parallel": True,
            "adaptive_scaling": True,
        }

    def update(self, **kwargs) -> None:
        with self._lock:
            self._config.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._config.get(key, default)

    def get_all(self) -> Dict:
        with self._lock:
            return copy.deepcopy(self._config)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._config[key] = value


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

# Global application state - use this instead of scattered global variables
app_state = AppState()

# Concurrency configuration
concurrency_config = ConcurrencyConfig()


# =============================================================================
# COMPATIBILITY LAYER (for gradual migration)
# =============================================================================

def get_data_status() -> Dict:
    """Compatibility wrapper for data_status global."""
    return app_state.get_data_status()

def get_unified_status() -> Dict:
    """Compatibility wrapper for unified_status global."""
    return app_state.get_unified_status()

def get_elite_status() -> Dict:
    """Compatibility wrapper for elite_validation_status global."""
    return app_state.get_elite_status()

def get_autonomous_status() -> Dict:
    """Compatibility wrapper for autonomous_optimizer_status global."""
    return app_state.get_autonomous_status()
