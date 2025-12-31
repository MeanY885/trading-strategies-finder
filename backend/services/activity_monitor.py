"""
RESOURCE-AWARE ACTIVITY MONITOR
================================
Monitors actual system activity instead of relying on time thresholds.
Determines "is this task actually working" by observing:
- CPU utilization per-thread
- Memory allocation patterns
- I/O activity (network/disk)
- Thread state inspection
- Asyncio task introspection

This replaces time-based watchdog logic with activity-based detection
that can differentiate between:
- Task actively processing (let it run)
- Task blocked on I/O (waiting for network/disk - normal)
- Task idle/stalled (no activity - potential problem)
- Task deadlocked (same stack trace repeatedly)
"""

import asyncio
import sys
import time
import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Set
from collections import deque
from datetime import datetime
from enum import Enum
import logging

import psutil

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Observable states a task can be in."""
    UNKNOWN = "unknown"
    RUNNING = "running"           # Actively consuming CPU
    IO_WAIT = "io_wait"          # Blocked on I/O (normal for network tasks)
    SLEEPING = "sleeping"         # In sleep/await (normal for async)
    BLOCKED = "blocked"          # Blocked on lock/semaphore
    STALLED = "stalled"          # No activity detected
    DEADLOCKED = "deadlocked"    # Same stack trace repeatedly


@dataclass
class CPUSnapshot:
    """CPU utilization snapshot for a process/thread."""
    timestamp: float
    user_time: float      # User CPU time (seconds)
    system_time: float    # System CPU time (seconds)
    cpu_percent: float    # CPU usage percentage


@dataclass
class MemorySnapshot:
    """Memory snapshot for a process."""
    timestamp: float
    rss: int              # Resident Set Size (bytes)
    vms: int              # Virtual Memory Size (bytes)
    rss_delta: int = 0    # Change in RSS since last snapshot


@dataclass
class IOSnapshot:
    """I/O counters snapshot for a process."""
    timestamp: float
    read_bytes: int = 0
    write_bytes: int = 0
    read_count: int = 0
    write_count: int = 0
    read_delta: int = 0   # Change since last snapshot
    write_delta: int = 0


@dataclass
class ThreadSnapshot:
    """Snapshot of thread state and stack."""
    thread_id: int
    timestamp: float
    stack_hash: int       # Hash of stack trace (for deadlock detection)
    stack_trace: str      # Full stack trace
    user_time: float = 0
    system_time: float = 0


@dataclass
class ActivityState:
    """Current activity state of a monitored task."""
    task_id: str
    state: TaskState = TaskState.UNKNOWN
    confidence: float = 0.0  # 0-1, how confident we are in the state

    # Activity indicators
    cpu_active: bool = False
    memory_active: bool = False
    io_active: bool = False
    progress_active: bool = False

    # Metrics
    cpu_percent: float = 0.0
    memory_delta_mb: float = 0.0
    io_rate_kbps: float = 0.0
    progress_percent: float = 0.0

    # Timing
    last_cpu_activity: float = 0.0
    last_memory_activity: float = 0.0
    last_io_activity: float = 0.0
    last_progress_update: float = 0.0

    # Deadlock detection
    stack_unchanged_count: int = 0
    last_stack_hash: int = 0

    # Recommendation
    is_working: bool = False
    recommendation: str = ""


class CPUMonitor:
    """
    Monitor CPU utilization per-process and per-thread.

    Uses psutil to track CPU time accumulation over sampling intervals.
    If CPU time is accumulating, the task is actively computing.
    """

    # Minimum CPU delta to consider "active" (seconds)
    MIN_CPU_DELTA = 0.01  # 10ms of CPU time

    # Sample window size
    SAMPLE_WINDOW = 10

    def __init__(self, pid: int = None):
        """
        Initialize CPU monitor.

        Args:
            pid: Process ID to monitor (default: current process)
        """
        self.pid = pid or psutil.Process().pid
        self._process: Optional[psutil.Process] = None
        self._snapshots: deque = deque(maxlen=self.SAMPLE_WINDOW)
        self._thread_snapshots: Dict[int, deque] = {}
        self._lock = threading.Lock()

    def _get_process(self) -> Optional[psutil.Process]:
        """Get or create psutil Process object."""
        if self._process is None:
            try:
                self._process = psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                return None
        return self._process

    def sample(self) -> Optional[CPUSnapshot]:
        """
        Take a CPU utilization sample.

        Returns:
            CPUSnapshot with current CPU metrics, or None if process not found
        """
        proc = self._get_process()
        if proc is None:
            return None

        try:
            cpu_times = proc.cpu_times()
            cpu_percent = proc.cpu_percent(interval=0)

            snapshot = CPUSnapshot(
                timestamp=time.time(),
                user_time=cpu_times.user,
                system_time=cpu_times.system,
                cpu_percent=cpu_percent
            )

            with self._lock:
                self._snapshots.append(snapshot)

            return snapshot

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def sample_threads(self) -> Dict[int, ThreadSnapshot]:
        """
        Sample CPU times for all threads in the process.

        Returns:
            Dict mapping thread_id to ThreadSnapshot
        """
        proc = self._get_process()
        if proc is None:
            return {}

        snapshots = {}
        try:
            threads = proc.threads()
            for thread in threads:
                snapshot = ThreadSnapshot(
                    thread_id=thread.id,
                    timestamp=time.time(),
                    stack_hash=0,  # Will be set by StackInspector
                    stack_trace="",
                    user_time=thread.user_time,
                    system_time=thread.system_time
                )
                snapshots[thread.id] = snapshot

                # Track per-thread history
                with self._lock:
                    if thread.id not in self._thread_snapshots:
                        self._thread_snapshots[thread.id] = deque(maxlen=self.SAMPLE_WINDOW)
                    self._thread_snapshots[thread.id].append(snapshot)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return snapshots

    def is_cpu_active(self, thread_id: int = None) -> Tuple[bool, float]:
        """
        Check if CPU is being actively used.

        Args:
            thread_id: Specific thread to check (None = whole process)

        Returns:
            Tuple of (is_active, cpu_time_delta_seconds)
        """
        with self._lock:
            if thread_id is not None:
                snapshots = self._thread_snapshots.get(thread_id, deque())
            else:
                snapshots = self._snapshots

            if len(snapshots) < 2:
                return False, 0.0

            latest = snapshots[-1]
            previous = snapshots[-2]

            # Calculate total CPU time delta
            if thread_id is not None:
                delta = (latest.user_time - previous.user_time +
                        latest.system_time - previous.system_time)
            else:
                delta = (latest.user_time - previous.user_time +
                        latest.system_time - previous.system_time)

            time_elapsed = latest.timestamp - previous.timestamp

            # CPU time should be less than or equal to wall time
            # If we're accumulating CPU time, we're working
            is_active = delta >= self.MIN_CPU_DELTA

            return is_active, delta

    def get_cpu_percent(self) -> float:
        """Get average CPU percentage over sample window."""
        with self._lock:
            if not self._snapshots:
                return 0.0
            return sum(s.cpu_percent for s in self._snapshots) / len(self._snapshots)

    def get_thread_cpu_percents(self) -> Dict[int, float]:
        """Get CPU usage per thread."""
        result = {}
        with self._lock:
            for tid, snapshots in self._thread_snapshots.items():
                if len(snapshots) >= 2:
                    latest = snapshots[-1]
                    oldest = snapshots[0]
                    time_diff = latest.timestamp - oldest.timestamp
                    if time_diff > 0:
                        cpu_time = (latest.user_time - oldest.user_time +
                                   latest.system_time - oldest.system_time)
                        result[tid] = (cpu_time / time_diff) * 100
        return result


class MemoryMonitor:
    """
    Monitor memory allocation patterns.

    Active allocation indicates work in progress.
    Memory patterns can distinguish between:
    - Active processing (allocations happening)
    - Idle (stable memory)
    - Memory leak (continuous growth without progress)
    """

    # Minimum memory delta to consider "active" (bytes)
    MIN_MEMORY_DELTA = 1024 * 1024  # 1 MB

    # Sample window size
    SAMPLE_WINDOW = 10

    def __init__(self, pid: int = None):
        self.pid = pid or psutil.Process().pid
        self._process: Optional[psutil.Process] = None
        self._snapshots: deque = deque(maxlen=self.SAMPLE_WINDOW)
        self._lock = threading.Lock()

    def _get_process(self) -> Optional[psutil.Process]:
        if self._process is None:
            try:
                self._process = psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                return None
        return self._process

    def sample(self) -> Optional[MemorySnapshot]:
        """Take a memory utilization sample."""
        proc = self._get_process()
        if proc is None:
            return None

        try:
            mem_info = proc.memory_info()

            with self._lock:
                rss_delta = 0
                if self._snapshots:
                    rss_delta = mem_info.rss - self._snapshots[-1].rss

                snapshot = MemorySnapshot(
                    timestamp=time.time(),
                    rss=mem_info.rss,
                    vms=mem_info.vms,
                    rss_delta=rss_delta
                )
                self._snapshots.append(snapshot)

            return snapshot

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def is_memory_active(self) -> Tuple[bool, int]:
        """
        Check if memory allocation is active.

        Returns:
            Tuple of (is_active, memory_delta_bytes)
        """
        with self._lock:
            if len(self._snapshots) < 2:
                return False, 0

            latest = self._snapshots[-1]
            previous = self._snapshots[-2]

            delta = abs(latest.rss - previous.rss)
            is_active = delta >= self.MIN_MEMORY_DELTA

            return is_active, delta

    def get_memory_trend(self) -> str:
        """
        Analyze memory trend over sample window.

        Returns:
            "growing", "shrinking", "stable", or "unknown"
        """
        with self._lock:
            if len(self._snapshots) < 3:
                return "unknown"

            # Calculate linear regression slope
            times = [s.timestamp - self._snapshots[0].timestamp for s in self._snapshots]
            values = [s.rss for s in self._snapshots]

            n = len(times)
            sum_x = sum(times)
            sum_y = sum(values)
            sum_xy = sum(t * v for t, v in zip(times, values))
            sum_xx = sum(t * t for t in times)

            denominator = n * sum_xx - sum_x * sum_x
            if denominator == 0:
                return "stable"

            slope = (n * sum_xy - sum_x * sum_y) / denominator

            # Slope threshold (bytes per second)
            if abs(slope) < 100_000:  # < 100 KB/s
                return "stable"
            elif slope > 0:
                return "growing"
            else:
                return "shrinking"


class IOMonitor:
    """
    Monitor I/O activity (disk and network).

    Active I/O indicates the task is making progress on data operations
    even if CPU is idle (waiting for network response, database query, etc.)
    """

    # Minimum I/O to consider "active" (bytes)
    MIN_IO_DELTA = 1024  # 1 KB

    # Sample window size
    SAMPLE_WINDOW = 10

    def __init__(self, pid: int = None):
        self.pid = pid or psutil.Process().pid
        self._process: Optional[psutil.Process] = None
        self._snapshots: deque = deque(maxlen=self.SAMPLE_WINDOW)
        self._net_snapshots: deque = deque(maxlen=self.SAMPLE_WINDOW)
        self._lock = threading.Lock()

    def _get_process(self) -> Optional[psutil.Process]:
        if self._process is None:
            try:
                self._process = psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                return None
        return self._process

    def sample(self) -> Optional[IOSnapshot]:
        """Take an I/O activity sample."""
        proc = self._get_process()
        if proc is None:
            return None

        try:
            io_counters = proc.io_counters()

            with self._lock:
                read_delta = 0
                write_delta = 0
                if self._snapshots:
                    prev = self._snapshots[-1]
                    read_delta = io_counters.read_bytes - prev.read_bytes
                    write_delta = io_counters.write_bytes - prev.write_bytes

                snapshot = IOSnapshot(
                    timestamp=time.time(),
                    read_bytes=io_counters.read_bytes,
                    write_bytes=io_counters.write_bytes,
                    read_count=io_counters.read_count,
                    write_count=io_counters.write_count,
                    read_delta=read_delta,
                    write_delta=write_delta
                )
                self._snapshots.append(snapshot)

            return snapshot

        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            # io_counters may not be available on all platforms
            return None

    def is_io_active(self) -> Tuple[bool, int]:
        """
        Check if I/O is active.

        Returns:
            Tuple of (is_active, total_io_delta_bytes)
        """
        with self._lock:
            if len(self._snapshots) < 2:
                return False, 0

            latest = self._snapshots[-1]
            previous = self._snapshots[-2]

            read_delta = latest.read_bytes - previous.read_bytes
            write_delta = latest.write_bytes - previous.write_bytes
            total_delta = read_delta + write_delta

            is_active = total_delta >= self.MIN_IO_DELTA

            return is_active, total_delta

    def get_io_rate(self) -> Tuple[float, float]:
        """
        Get I/O rate in bytes/second.

        Returns:
            Tuple of (read_bytes_per_sec, write_bytes_per_sec)
        """
        with self._lock:
            if len(self._snapshots) < 2:
                return 0.0, 0.0

            oldest = self._snapshots[0]
            latest = self._snapshots[-1]

            time_diff = latest.timestamp - oldest.timestamp
            if time_diff <= 0:
                return 0.0, 0.0

            read_rate = (latest.read_bytes - oldest.read_bytes) / time_diff
            write_rate = (latest.write_bytes - oldest.write_bytes) / time_diff

            return read_rate, write_rate


class StackInspector:
    """
    Inspect thread stacks for deadlock detection.

    Uses sys._current_frames() to get stack traces of all threads.
    If a thread's stack trace remains identical across multiple samples,
    it may be deadlocked (especially if blocked on lock acquisition).

    WARNING: sys._current_frames() has known thread-safety issues in some
    Python versions. Use with caution in heavily threaded applications.
    """

    # Number of identical stacks before considering deadlocked
    DEADLOCK_THRESHOLD = 5

    # Sample window for tracking stacks
    SAMPLE_WINDOW = 10

    def __init__(self):
        self._stack_history: Dict[int, deque] = {}  # thread_id -> stack hashes
        self._full_stacks: Dict[int, str] = {}  # thread_id -> last full stack
        self._lock = threading.Lock()

    def sample_all_threads(self) -> Dict[int, ThreadSnapshot]:
        """
        Sample stack traces for all Python threads.

        Returns:
            Dict mapping thread_id to ThreadSnapshot
        """
        snapshots = {}
        now = time.time()

        try:
            frames = sys._current_frames()

            for thread_id, frame in frames.items():
                # Format stack trace
                stack_lines = traceback.format_stack(frame)
                stack_trace = ''.join(stack_lines)
                stack_hash = hash(stack_trace)

                snapshot = ThreadSnapshot(
                    thread_id=thread_id,
                    timestamp=now,
                    stack_hash=stack_hash,
                    stack_trace=stack_trace
                )
                snapshots[thread_id] = snapshot

                # Track history
                with self._lock:
                    if thread_id not in self._stack_history:
                        self._stack_history[thread_id] = deque(maxlen=self.SAMPLE_WINDOW)
                    self._stack_history[thread_id].append(stack_hash)
                    self._full_stacks[thread_id] = stack_trace

        except Exception as e:
            logger.warning(f"Error sampling thread stacks: {e}")

        return snapshots

    def is_deadlocked(self, thread_id: int) -> Tuple[bool, int]:
        """
        Check if a thread appears deadlocked.

        A thread is considered potentially deadlocked if its stack trace
        has been identical for DEADLOCK_THRESHOLD consecutive samples.

        Returns:
            Tuple of (is_deadlocked, consecutive_identical_count)
        """
        with self._lock:
            history = self._stack_history.get(thread_id)
            if not history or len(history) < 2:
                return False, 0

            # Count consecutive identical hashes from the end
            latest_hash = history[-1]
            count = 0
            for h in reversed(history):
                if h == latest_hash:
                    count += 1
                else:
                    break

            is_deadlocked = count >= self.DEADLOCK_THRESHOLD
            return is_deadlocked, count

    def get_thread_stack(self, thread_id: int) -> Optional[str]:
        """Get the last recorded stack trace for a thread."""
        with self._lock:
            return self._full_stacks.get(thread_id)

    def get_blocking_location(self, thread_id: int) -> Optional[str]:
        """
        Try to identify where a thread is blocked.

        Looks for common blocking patterns in the stack trace.
        """
        stack = self.get_thread_stack(thread_id)
        if not stack:
            return None

        # Common blocking patterns
        blocking_patterns = [
            ("acquire", "Acquiring lock"),
            ("wait", "Waiting on condition/event"),
            ("recv", "Waiting for network data"),
            ("read", "Reading from file/socket"),
            ("write", "Writing to file/socket"),
            ("connect", "Connecting to server"),
            ("accept", "Accepting connection"),
            ("select", "Waiting in select/poll"),
            ("sleep", "Sleeping"),
            ("Queue.get", "Waiting on queue"),
            ("Semaphore", "Waiting on semaphore"),
        ]

        for pattern, description in blocking_patterns:
            if pattern.lower() in stack.lower():
                return description

        return "Unknown blocking location"


class AsyncioInspector:
    """
    Introspect asyncio tasks for stuck coroutine detection.

    Uses asyncio.all_tasks() to get task states and
    tracks how long each task has been in the same state.
    """

    # Time before warning about a stuck task (seconds)
    STUCK_WARNING_THRESHOLD = 60

    # Time before considering task stuck (seconds)
    STUCK_THRESHOLD = 300

    def __init__(self):
        self._task_first_seen: Dict[int, float] = {}  # task_id -> first_seen_time
        self._task_last_state: Dict[int, str] = {}    # task_id -> last_state_repr
        self._task_state_since: Dict[int, float] = {} # task_id -> state_unchanged_since
        self._lock = threading.Lock()

    async def sample_tasks(self) -> Dict[int, dict]:
        """
        Sample all asyncio tasks in the current event loop.

        Returns:
            Dict mapping task_id to task info dict
        """
        task_info = {}
        now = time.time()

        try:
            all_tasks = asyncio.all_tasks()

            for task in all_tasks:
                task_id = id(task)
                task_name = task.get_name()
                task_done = task.done()
                task_cancelled = task.cancelled()

                # Get coroutine state
                coro = task.get_coro()
                coro_state = "unknown"
                if hasattr(coro, 'cr_frame'):
                    if coro.cr_frame is None:
                        coro_state = "finished"
                    else:
                        coro_state = "suspended"
                elif hasattr(coro, 'gi_frame'):  # Generator-based coroutine
                    if coro.gi_frame is None:
                        coro_state = "finished"
                    else:
                        coro_state = "suspended"

                # Create state representation
                state_repr = f"{task_done}:{task_cancelled}:{coro_state}"

                with self._lock:
                    # Track first seen time
                    if task_id not in self._task_first_seen:
                        self._task_first_seen[task_id] = now

                    # Track state changes
                    if task_id not in self._task_last_state:
                        self._task_last_state[task_id] = state_repr
                        self._task_state_since[task_id] = now
                    elif self._task_last_state[task_id] != state_repr:
                        self._task_last_state[task_id] = state_repr
                        self._task_state_since[task_id] = now

                    state_duration = now - self._task_state_since[task_id]
                    lifetime = now - self._task_first_seen[task_id]

                task_info[task_id] = {
                    "name": task_name,
                    "done": task_done,
                    "cancelled": task_cancelled,
                    "coro_state": coro_state,
                    "state_duration": state_duration,
                    "lifetime": lifetime,
                    "is_stuck": state_duration > self.STUCK_THRESHOLD and not task_done,
                    "is_warning": state_duration > self.STUCK_WARNING_THRESHOLD and not task_done,
                }

        except Exception as e:
            logger.warning(f"Error sampling asyncio tasks: {e}")

        return task_info

    async def get_stuck_tasks(self) -> List[dict]:
        """Get list of potentially stuck tasks."""
        tasks = await self.sample_tasks()
        return [info for info in tasks.values() if info.get("is_stuck")]

    def cleanup_finished_tasks(self):
        """Remove tracking for finished tasks."""
        with self._lock:
            # Find tasks that are no longer in all_tasks
            try:
                current_ids = {id(t) for t in asyncio.all_tasks()}
            except RuntimeError:
                return

            to_remove = set(self._task_first_seen.keys()) - current_ids

            for task_id in to_remove:
                self._task_first_seen.pop(task_id, None)
                self._task_last_state.pop(task_id, None)
                self._task_state_since.pop(task_id, None)


class ActivityMonitor:
    """
    Unified activity monitor that combines all detection methods.

    Provides a single "is_working" answer by aggregating signals from:
    - CPU usage
    - Memory allocation
    - I/O activity
    - Thread state
    - Asyncio task state
    - Progress callbacks

    This replaces time-based watchdogs with activity-based detection.
    """

    # Sampling interval (seconds)
    SAMPLE_INTERVAL = 1.0

    # How long without ANY activity before marking stalled
    NO_ACTIVITY_THRESHOLD = 30.0  # 30 seconds

    # Weight factors for activity signals
    WEIGHT_CPU = 0.35
    WEIGHT_MEMORY = 0.15
    WEIGHT_IO = 0.25
    WEIGHT_PROGRESS = 0.25

    # Activity threshold for "is_working"
    ACTIVITY_THRESHOLD = 0.3  # 30% confidence needed

    def __init__(self, pid: int = None):
        """
        Initialize activity monitor.

        Args:
            pid: Process ID to monitor (default: current process)
        """
        self.pid = pid or psutil.Process().pid

        # Component monitors
        self.cpu_monitor = CPUMonitor(self.pid)
        self.memory_monitor = MemoryMonitor(self.pid)
        self.io_monitor = IOMonitor(self.pid)
        self.stack_inspector = StackInspector()
        self.asyncio_inspector = AsyncioInspector()

        # Task tracking
        self._tasks: Dict[str, dict] = {}  # task_id -> task_info
        self._task_activity: Dict[str, ActivityState] = {}  # task_id -> ActivityState
        self._progress_callbacks: Dict[str, Callable[[], float]] = {}

        self._running = False
        self._sample_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        logger.info(f"[ActivityMonitor] Initialized for PID {self.pid}")

    def register_task(
        self,
        task_id: str,
        thread_id: int = None,
        progress_callback: Callable[[], float] = None,
        metadata: dict = None
    ):
        """
        Register a task for monitoring.

        Args:
            task_id: Unique identifier for the task
            thread_id: Thread ID if running in a specific thread
            progress_callback: Function returning progress 0-100
            metadata: Additional task information
        """
        self._tasks[task_id] = {
            "thread_id": thread_id,
            "metadata": metadata or {},
            "registered_at": time.time(),
            "last_activity": time.time(),
        }

        if progress_callback:
            self._progress_callbacks[task_id] = progress_callback

        self._task_activity[task_id] = ActivityState(task_id=task_id)

        logger.debug(f"[ActivityMonitor] Registered task: {task_id}")

    def unregister_task(self, task_id: str):
        """Unregister a task from monitoring."""
        self._tasks.pop(task_id, None)
        self._progress_callbacks.pop(task_id, None)
        self._task_activity.pop(task_id, None)
        logger.debug(f"[ActivityMonitor] Unregistered task: {task_id}")

    def report_progress(self, task_id: str, progress: float):
        """
        Report progress for a task.

        This is the most reliable indicator of work being done.
        """
        if task_id in self._tasks:
            self._tasks[task_id]["last_activity"] = time.time()
            self._tasks[task_id]["last_progress"] = progress

            if task_id in self._task_activity:
                self._task_activity[task_id].progress_percent = progress
                self._task_activity[task_id].last_progress_update = time.time()
                self._task_activity[task_id].progress_active = True

    async def _sample_once(self):
        """Take one sample of all activity indicators."""
        now = time.time()

        # Sample system-wide metrics
        self.cpu_monitor.sample()
        self.cpu_monitor.sample_threads()
        self.memory_monitor.sample()
        self.io_monitor.sample()
        stack_snapshots = self.stack_inspector.sample_all_threads()

        # Sample asyncio tasks
        try:
            await self.asyncio_inspector.sample_tasks()
        except RuntimeError:
            pass  # No event loop in some contexts

        # Update each registered task's activity state
        for task_id, task_info in self._tasks.items():
            activity = self._task_activity.get(task_id)
            if not activity:
                continue

            # CPU activity
            thread_id = task_info.get("thread_id")
            cpu_active, cpu_delta = self.cpu_monitor.is_cpu_active(thread_id)
            activity.cpu_active = cpu_active
            if cpu_active:
                activity.last_cpu_activity = now
            activity.cpu_percent = self.cpu_monitor.get_cpu_percent()

            # Memory activity
            mem_active, mem_delta = self.memory_monitor.is_memory_active()
            activity.memory_active = mem_active
            if mem_active:
                activity.last_memory_activity = now
            activity.memory_delta_mb = mem_delta / (1024 * 1024)

            # I/O activity
            io_active, io_delta = self.io_monitor.is_io_active()
            activity.io_active = io_active
            if io_active:
                activity.last_io_activity = now
            read_rate, write_rate = self.io_monitor.get_io_rate()
            activity.io_rate_kbps = (read_rate + write_rate) / 1024

            # Progress callback
            if task_id in self._progress_callbacks:
                try:
                    progress = self._progress_callbacks[task_id]()
                    prev_progress = activity.progress_percent
                    if progress > prev_progress:
                        activity.progress_active = True
                        activity.last_progress_update = now
                    activity.progress_percent = progress
                except Exception:
                    pass

            # Deadlock detection (if thread_id available)
            if thread_id:
                is_deadlocked, stack_count = self.stack_inspector.is_deadlocked(thread_id)
                activity.stack_unchanged_count = stack_count
                if thread_id in stack_snapshots:
                    activity.last_stack_hash = stack_snapshots[thread_id].stack_hash

            # Calculate overall activity state
            self._calculate_activity_state(activity, now)

    def _calculate_activity_state(self, activity: ActivityState, now: float):
        """Calculate the overall activity state for a task."""
        # Calculate weighted activity score
        signals = []

        if activity.cpu_active:
            signals.append(self.WEIGHT_CPU)
        if activity.memory_active:
            signals.append(self.WEIGHT_MEMORY)
        if activity.io_active:
            signals.append(self.WEIGHT_IO)
        if activity.progress_active:
            signals.append(self.WEIGHT_PROGRESS)
            activity.progress_active = False  # Reset for next interval

        confidence = sum(signals)
        activity.confidence = confidence

        # Determine state
        if confidence >= self.ACTIVITY_THRESHOLD:
            activity.state = TaskState.RUNNING
            activity.is_working = True
            activity.recommendation = "Task is actively working"
        elif activity.io_active:
            activity.state = TaskState.IO_WAIT
            activity.is_working = True
            activity.recommendation = "Task is waiting for I/O (normal)"
        elif activity.stack_unchanged_count >= StackInspector.DEADLOCK_THRESHOLD:
            activity.state = TaskState.DEADLOCKED
            activity.is_working = False
            activity.recommendation = "Task may be deadlocked - check stack trace"
        else:
            # Check time since last activity
            last_activity = max(
                activity.last_cpu_activity,
                activity.last_memory_activity,
                activity.last_io_activity,
                activity.last_progress_update
            )

            if last_activity == 0:
                activity.state = TaskState.UNKNOWN
                activity.is_working = True  # Benefit of the doubt
                activity.recommendation = "Task activity unknown - still monitoring"
            elif now - last_activity > self.NO_ACTIVITY_THRESHOLD:
                activity.state = TaskState.STALLED
                activity.is_working = False
                activity.recommendation = f"No activity for {now - last_activity:.0f}s - task may be stalled"
            else:
                activity.state = TaskState.SLEEPING
                activity.is_working = True
                activity.recommendation = "Task is sleeping/awaiting (normal for async)"

    async def start_monitoring(self):
        """Start the background monitoring loop."""
        if self._running:
            return

        self._running = True
        self._sample_task = asyncio.create_task(self._monitoring_loop())
        logger.info("[ActivityMonitor] Started monitoring loop")

    async def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._running = False
        if self._sample_task:
            self._sample_task.cancel()
            try:
                await self._sample_task
            except asyncio.CancelledError:
                pass
        logger.info("[ActivityMonitor] Stopped monitoring loop")

    async def _monitoring_loop(self):
        """Background loop that samples activity."""
        while self._running:
            try:
                await self._sample_once()
            except Exception as e:
                logger.warning(f"[ActivityMonitor] Sample error: {e}")
            await asyncio.sleep(self.SAMPLE_INTERVAL)

    def is_task_working(self, task_id: str) -> Tuple[bool, str]:
        """
        Check if a task is actively working.

        Returns:
            Tuple of (is_working, reason_string)
        """
        activity = self._task_activity.get(task_id)
        if not activity:
            return True, "Task not registered - assuming working"

        return activity.is_working, activity.recommendation

    def get_task_state(self, task_id: str) -> Optional[ActivityState]:
        """Get the full activity state for a task."""
        return self._task_activity.get(task_id)

    def get_all_states(self) -> Dict[str, ActivityState]:
        """Get activity states for all registered tasks."""
        return dict(self._task_activity)

    def get_status(self) -> dict:
        """Get monitor status for API/logging."""
        return {
            "pid": self.pid,
            "running": self._running,
            "registered_tasks": list(self._tasks.keys()),
            "task_states": {
                tid: {
                    "state": activity.state.value,
                    "is_working": activity.is_working,
                    "cpu_percent": activity.cpu_percent,
                    "memory_delta_mb": activity.memory_delta_mb,
                    "io_rate_kbps": activity.io_rate_kbps,
                    "progress_percent": activity.progress_percent,
                    "recommendation": activity.recommendation,
                }
                for tid, activity in self._task_activity.items()
            },
            "cpu_percent": self.cpu_monitor.get_cpu_percent(),
            "memory_trend": self.memory_monitor.get_memory_trend(),
            "io_rate": self.io_monitor.get_io_rate(),
        }


# =============================================================================
# ACTIVITY-BASED WATCHDOG (replaces time-based TaskWatchdog)
# =============================================================================

class ActivityWatchdog:
    """
    Activity-based watchdog that monitors actual work instead of time.

    Unlike time-based watchdogs that abort after X seconds of "no progress",
    this watchdog monitors actual system activity (CPU, memory, I/O) to
    determine if the task is working.

    A task can take hours if it's actively working. It only gets aborted
    if there's genuinely no activity happening.
    """

    # How often to check activity (seconds)
    CHECK_INTERVAL = 5.0

    # How many consecutive "not working" samples before aborting
    # At 5s intervals, 12 samples = 60 seconds of truly no activity
    ABORT_THRESHOLD = 12

    # Warning threshold (samples)
    WARNING_THRESHOLD = 6

    def __init__(
        self,
        task_id: str,
        status_dict: dict,
        activity_monitor: ActivityMonitor,
        abort_key: str = "abort",
        running_key: str = "running",
        progress_key: str = "progress"
    ):
        """
        Initialize activity-based watchdog.

        Args:
            task_id: Unique identifier for the task
            status_dict: Dictionary containing task status (modified in place)
            activity_monitor: ActivityMonitor instance to use
            abort_key: Key in status_dict to set True when aborting
            running_key: Key in status_dict indicating if task is running
            progress_key: Key in status_dict for progress value
        """
        self.task_id = task_id
        self.status_dict = status_dict
        self.activity_monitor = activity_monitor
        self.abort_key = abort_key
        self.running_key = running_key
        self.progress_key = progress_key

        self._running = False
        self._aborted = False
        self._abort_reason: Optional[str] = None
        self._not_working_count = 0
        self._warning_logged = False

    async def start(self):
        """Start watching the task."""
        self._running = True
        self._aborted = False
        self._not_working_count = 0

        # Register with activity monitor
        self.activity_monitor.register_task(
            task_id=self.task_id,
            progress_callback=lambda: self.status_dict.get(self.progress_key, 0)
        )

        logger.info(f"[ActivityWatchdog] Started monitoring task {self.task_id}")

        while self._running:
            await asyncio.sleep(self.CHECK_INTERVAL)

            # Check if task is still supposed to be running
            if not self.status_dict.get(self.running_key, True):
                logger.info(f"[ActivityWatchdog] Task {self.task_id} marked as not running")
                break

            # Report progress to activity monitor
            progress = self.status_dict.get(self.progress_key, 0)
            self.activity_monitor.report_progress(self.task_id, progress)

            # Check if task is working
            is_working, reason = self.activity_monitor.is_task_working(self.task_id)

            if is_working:
                # Reset counter if working
                self._not_working_count = 0
                self._warning_logged = False
            else:
                # Increment counter if not working
                self._not_working_count += 1

                if self._not_working_count >= self.ABORT_THRESHOLD:
                    self._trigger_abort(reason)
                    break
                elif self._not_working_count >= self.WARNING_THRESHOLD and not self._warning_logged:
                    logger.warning(
                        f"[ActivityWatchdog] Task {self.task_id} showing no activity "
                        f"for {self._not_working_count * self.CHECK_INTERVAL:.0f}s: {reason}"
                    )
                    self._warning_logged = True

        self._running = False
        self.activity_monitor.unregister_task(self.task_id)

    def _trigger_abort(self, reason: str):
        """Trigger abort on the monitored task."""
        self._aborted = True
        self._abort_reason = reason
        self.status_dict[self.abort_key] = True
        logger.error(f"[ActivityWatchdog] ABORT {self.task_id}: {reason}")

    async def stop(self):
        """Stop watching."""
        self._running = False
        self.activity_monitor.unregister_task(self.task_id)
        logger.info(f"[ActivityWatchdog] Stopped monitoring {self.task_id}")

    def is_aborted(self) -> bool:
        """Check if watchdog triggered an abort."""
        return self._aborted

    @property
    def abort_reason(self) -> Optional[str]:
        """Get the reason for abort, if any."""
        return self._abort_reason


# =============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# =============================================================================

def is_thread_consuming_cpu(thread_id: int, min_cpu_percent: float = 1.0) -> bool:
    """
    Quick check if a specific thread is consuming CPU.

    Args:
        thread_id: Native thread ID to check
        min_cpu_percent: Minimum CPU percentage to consider "consuming"

    Returns:
        True if thread is consuming CPU above threshold
    """
    try:
        proc = psutil.Process()
        threads = proc.threads()

        for thread in threads:
            if thread.id == thread_id:
                # Calculate CPU percent from thread times
                # This is a simple check, not a rate calculation
                total_time = thread.user_time + thread.system_time
                return total_time > 0

        return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def get_process_activity_summary(pid: int = None) -> dict:
    """
    Get a quick activity summary for a process.

    Args:
        pid: Process ID (default: current process)

    Returns:
        Dict with activity summary
    """
    pid = pid or psutil.Process().pid

    try:
        proc = psutil.Process(pid)
        cpu_percent = proc.cpu_percent(interval=0.1)
        mem_info = proc.memory_info()

        io_rate = 0.0
        try:
            io_counters = proc.io_counters()
            io_rate = io_counters.read_bytes + io_counters.write_bytes
        except (AttributeError, psutil.AccessDenied):
            pass

        return {
            "pid": pid,
            "cpu_percent": cpu_percent,
            "memory_mb": mem_info.rss / (1024 * 1024),
            "threads": proc.num_threads(),
            "io_bytes": io_rate,
            "is_active": cpu_percent > 1.0,
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        return {"error": str(e)}


# Global activity monitor instance
_activity_monitor: Optional[ActivityMonitor] = None


def get_activity_monitor() -> ActivityMonitor:
    """Get or create the global ActivityMonitor instance."""
    global _activity_monitor
    if _activity_monitor is None:
        _activity_monitor = ActivityMonitor()
    return _activity_monitor


async def init_activity_monitor() -> ActivityMonitor:
    """Initialize and start the global activity monitor."""
    monitor = get_activity_monitor()
    await monitor.start_monitoring()
    return monitor


async def stop_activity_monitor():
    """Stop the global activity monitor."""
    global _activity_monitor
    if _activity_monitor:
        await _activity_monitor.stop_monitoring()
        _activity_monitor = None
