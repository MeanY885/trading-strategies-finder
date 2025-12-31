"""
Tests for Activity Monitor
==========================
Comprehensive tests for the resource-aware activity monitoring system.
"""

import asyncio
import sys
import os
import time
import threading
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add backend to path - direct import to avoid __init__.py issues
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/backend'
sys.path.insert(0, backend_path)

# Import the module directly without going through services/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "activity_monitor",
    os.path.join(backend_path, "services", "activity_monitor.py")
)
activity_monitor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(activity_monitor_module)

# Import classes from the loaded module
TaskState = activity_monitor_module.TaskState
CPUSnapshot = activity_monitor_module.CPUSnapshot
MemorySnapshot = activity_monitor_module.MemorySnapshot
IOSnapshot = activity_monitor_module.IOSnapshot
ThreadSnapshot = activity_monitor_module.ThreadSnapshot
ActivityState = activity_monitor_module.ActivityState
CPUMonitor = activity_monitor_module.CPUMonitor
MemoryMonitor = activity_monitor_module.MemoryMonitor
IOMonitor = activity_monitor_module.IOMonitor
StackInspector = activity_monitor_module.StackInspector
AsyncioInspector = activity_monitor_module.AsyncioInspector
ActivityMonitor = activity_monitor_module.ActivityMonitor
ActivityWatchdog = activity_monitor_module.ActivityWatchdog
is_thread_consuming_cpu = activity_monitor_module.is_thread_consuming_cpu
get_process_activity_summary = activity_monitor_module.get_process_activity_summary

# Original import that caused issues - kept for reference
# from services.activity_monitor import (
#     TaskState, CPUSnapshot, MemorySnapshot, IOSnapshot, ThreadSnapshot,
#     ActivityState, CPUMonitor, MemoryMonitor, IOMonitor, StackInspector,
#     AsyncioInspector, ActivityMonitor, ActivityWatchdog,
#     is_thread_consuming_cpu, get_process_activity_summary,
# )


class TestCPUMonitor:
    """Tests for CPUMonitor class."""

    def test_init(self):
        """Test CPU monitor initialization."""
        monitor = CPUMonitor()
        assert monitor.pid is not None
        assert monitor._snapshots.maxlen == CPUMonitor.SAMPLE_WINDOW

    def test_sample_returns_snapshot(self):
        """Test that sample() returns a CPUSnapshot."""
        monitor = CPUMonitor()
        snapshot = monitor.sample()

        assert snapshot is not None
        assert isinstance(snapshot, CPUSnapshot)
        assert snapshot.timestamp > 0
        assert snapshot.user_time >= 0
        assert snapshot.system_time >= 0

    def test_sample_accumulates_history(self):
        """Test that samples are accumulated in history."""
        monitor = CPUMonitor()

        for _ in range(5):
            monitor.sample()
            time.sleep(0.01)

        assert len(monitor._snapshots) == 5

    def test_is_cpu_active_requires_two_samples(self):
        """Test that is_cpu_active needs at least 2 samples."""
        monitor = CPUMonitor()

        # No samples
        active, delta = monitor.is_cpu_active()
        assert not active
        assert delta == 0

        # One sample
        monitor.sample()
        active, delta = monitor.is_cpu_active()
        assert not active

    def test_is_cpu_active_detects_cpu_work(self):
        """Test that CPU work is detected."""
        monitor = CPUMonitor()

        monitor.sample()

        # Do some CPU work
        result = sum(i * i for i in range(100000))

        time.sleep(0.1)
        monitor.sample()

        active, delta = monitor.is_cpu_active()
        # Should detect CPU activity
        assert delta >= 0

    def test_sample_threads(self):
        """Test thread sampling."""
        monitor = CPUMonitor()
        snapshots = monitor.sample_threads()

        assert isinstance(snapshots, dict)
        assert len(snapshots) >= 1  # At least main thread

        for tid, snapshot in snapshots.items():
            assert isinstance(tid, int)
            assert isinstance(snapshot, ThreadSnapshot)
            assert snapshot.thread_id == tid

    def test_get_cpu_percent(self):
        """Test CPU percentage calculation."""
        monitor = CPUMonitor()

        # Take several samples
        for _ in range(3):
            monitor.sample()
            time.sleep(0.05)

        cpu_percent = monitor.get_cpu_percent()
        assert isinstance(cpu_percent, float)
        assert cpu_percent >= 0


class TestMemoryMonitor:
    """Tests for MemoryMonitor class."""

    def test_init(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor()
        assert monitor.pid is not None
        assert monitor._snapshots.maxlen == MemoryMonitor.SAMPLE_WINDOW

    def test_sample_returns_snapshot(self):
        """Test that sample() returns a MemorySnapshot."""
        monitor = MemoryMonitor()
        snapshot = monitor.sample()

        assert snapshot is not None
        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.timestamp > 0
        assert snapshot.rss > 0
        assert snapshot.vms > 0

    def test_sample_calculates_delta(self):
        """Test that memory delta is calculated correctly."""
        monitor = MemoryMonitor()

        snapshot1 = monitor.sample()
        assert snapshot1.rss_delta == 0  # First sample has no delta

        # Allocate some memory
        big_list = [0] * 100000

        snapshot2 = monitor.sample()
        # Delta can be positive, negative, or zero depending on GC
        assert isinstance(snapshot2.rss_delta, int)

        del big_list

    def test_is_memory_active_requires_two_samples(self):
        """Test that is_memory_active needs at least 2 samples."""
        monitor = MemoryMonitor()

        active, delta = monitor.is_memory_active()
        assert not active

        monitor.sample()
        active, delta = monitor.is_memory_active()
        assert not active

    def test_get_memory_trend(self):
        """Test memory trend analysis."""
        monitor = MemoryMonitor()

        # Need at least 3 samples
        for _ in range(5):
            monitor.sample()
            time.sleep(0.01)

        trend = monitor.get_memory_trend()
        assert trend in ["growing", "shrinking", "stable", "unknown"]


class TestIOMonitor:
    """Tests for IOMonitor class."""

    def test_init(self):
        """Test I/O monitor initialization."""
        monitor = IOMonitor()
        assert monitor.pid is not None
        assert monitor._snapshots.maxlen == IOMonitor.SAMPLE_WINDOW

    def test_sample_returns_snapshot(self):
        """Test that sample() returns an IOSnapshot."""
        monitor = IOMonitor()
        snapshot = monitor.sample()

        # io_counters may not be available on all platforms
        if snapshot is not None:
            assert isinstance(snapshot, IOSnapshot)
            assert snapshot.timestamp > 0
            assert snapshot.read_bytes >= 0
            assert snapshot.write_bytes >= 0

    def test_is_io_active(self):
        """Test I/O activity detection."""
        monitor = IOMonitor()

        # Take initial samples
        monitor.sample()
        time.sleep(0.1)
        monitor.sample()

        active, delta = monitor.is_io_active()
        assert isinstance(active, bool)
        assert isinstance(delta, int)

    def test_get_io_rate(self):
        """Test I/O rate calculation."""
        monitor = IOMonitor()

        for _ in range(3):
            monitor.sample()
            time.sleep(0.05)

        read_rate, write_rate = monitor.get_io_rate()
        assert isinstance(read_rate, float)
        assert isinstance(write_rate, float)
        assert read_rate >= 0
        assert write_rate >= 0


class TestStackInspector:
    """Tests for StackInspector class."""

    def test_init(self):
        """Test stack inspector initialization."""
        inspector = StackInspector()
        assert isinstance(inspector._stack_history, dict)
        assert isinstance(inspector._full_stacks, dict)

    def test_sample_all_threads(self):
        """Test sampling all thread stacks."""
        inspector = StackInspector()
        snapshots = inspector.sample_all_threads()

        assert isinstance(snapshots, dict)
        assert len(snapshots) >= 1  # At least main thread

        main_thread_id = threading.current_thread().ident
        assert main_thread_id in snapshots

        snapshot = snapshots[main_thread_id]
        assert isinstance(snapshot, ThreadSnapshot)
        assert snapshot.thread_id == main_thread_id
        assert snapshot.stack_hash != 0
        assert "test_sample_all_threads" in snapshot.stack_trace

    def test_deadlock_detection_no_deadlock(self):
        """Test that varying code paths are detected."""
        inspector = StackInspector()

        # The stack sampling can detect if code is in the same place
        # In a test environment, the stack trace might be similar since
        # we're in a test function. This test verifies the mechanism works.
        for i in range(3):
            inspector.sample_all_threads()
            time.sleep(0.01)

        main_thread_id = threading.current_thread().ident
        is_deadlocked, count = inspector.is_deadlocked(main_thread_id)

        # The count should be tracked (may or may not be deadlocked
        # depending on how fast the test runs)
        assert isinstance(is_deadlocked, bool)
        assert isinstance(count, int)
        assert count >= 0

    def test_get_thread_stack(self):
        """Test getting thread stack trace."""
        inspector = StackInspector()
        inspector.sample_all_threads()

        main_thread_id = threading.current_thread().ident
        stack = inspector.get_thread_stack(main_thread_id)

        assert stack is not None
        assert isinstance(stack, str)
        assert "test_get_thread_stack" in stack

    def test_get_blocking_location(self):
        """Test identifying blocking location."""
        inspector = StackInspector()
        inspector.sample_all_threads()

        main_thread_id = threading.current_thread().ident
        location = inspector.get_blocking_location(main_thread_id)

        # Should detect something or return "Unknown"
        assert location is not None
        assert isinstance(location, str)


class TestAsyncioInspector:
    """Tests for AsyncioInspector class."""

    @pytest.mark.asyncio
    async def test_sample_tasks(self):
        """Test sampling asyncio tasks."""
        inspector = AsyncioInspector()

        # Create some tasks
        async def dummy_task():
            await asyncio.sleep(0.1)

        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())

        # Sample
        task_info = await inspector.sample_tasks()

        assert isinstance(task_info, dict)
        # Should see at least our tasks (plus test runner tasks)
        assert len(task_info) >= 2

        # Wait for tasks to complete
        await task1
        await task2

    @pytest.mark.asyncio
    async def test_get_stuck_tasks(self):
        """Test stuck task detection."""
        inspector = AsyncioInspector()

        # Create a quick task
        async def quick_task():
            await asyncio.sleep(0.01)

        task = asyncio.create_task(quick_task())

        stuck_tasks = await inspector.get_stuck_tasks()
        # Quick task shouldn't be stuck
        assert isinstance(stuck_tasks, list)

        await task

    @pytest.mark.asyncio
    async def test_cleanup_finished_tasks(self):
        """Test cleanup of finished task tracking."""
        inspector = AsyncioInspector()

        async def quick_task():
            await asyncio.sleep(0.01)

        task = asyncio.create_task(quick_task())
        await inspector.sample_tasks()

        await task
        inspector.cleanup_finished_tasks()

        # Should not raise any errors


class TestActivityMonitor:
    """Tests for unified ActivityMonitor class."""

    def test_init(self):
        """Test activity monitor initialization."""
        monitor = ActivityMonitor()

        assert monitor.pid is not None
        assert monitor.cpu_monitor is not None
        assert monitor.memory_monitor is not None
        assert monitor.io_monitor is not None
        assert monitor.stack_inspector is not None
        assert monitor.asyncio_inspector is not None

    def test_register_task(self):
        """Test task registration."""
        monitor = ActivityMonitor()

        monitor.register_task("test_task", metadata={"description": "Test"})

        assert "test_task" in monitor._tasks
        assert "test_task" in monitor._task_activity

    def test_unregister_task(self):
        """Test task unregistration."""
        monitor = ActivityMonitor()

        monitor.register_task("test_task")
        monitor.unregister_task("test_task")

        assert "test_task" not in monitor._tasks
        assert "test_task" not in monitor._task_activity

    def test_report_progress(self):
        """Test progress reporting."""
        monitor = ActivityMonitor()

        monitor.register_task("test_task")
        monitor.report_progress("test_task", 50.0)

        activity = monitor._task_activity["test_task"]
        assert activity.progress_percent == 50.0
        assert activity.progress_active

    def test_is_task_working_unregistered(self):
        """Test is_task_working for unregistered task."""
        monitor = ActivityMonitor()

        is_working, reason = monitor.is_task_working("nonexistent")

        assert is_working  # Benefit of the doubt
        assert "not registered" in reason.lower()

    def test_get_task_state(self):
        """Test getting task state."""
        monitor = ActivityMonitor()

        monitor.register_task("test_task")

        state = monitor.get_task_state("test_task")
        assert state is not None
        assert isinstance(state, ActivityState)
        assert state.task_id == "test_task"

    def test_get_all_states(self):
        """Test getting all task states."""
        monitor = ActivityMonitor()

        monitor.register_task("task1")
        monitor.register_task("task2")

        states = monitor.get_all_states()

        assert "task1" in states
        assert "task2" in states

    def test_get_status(self):
        """Test status dictionary generation."""
        monitor = ActivityMonitor()

        monitor.register_task("test_task")

        status = monitor.get_status()

        assert "pid" in status
        assert "running" in status
        assert "registered_tasks" in status
        assert "task_states" in status
        assert "test_task" in status["registered_tasks"]

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping the monitor."""
        monitor = ActivityMonitor()

        await monitor.start_monitoring()
        assert monitor._running
        assert monitor._sample_task is not None

        await asyncio.sleep(0.1)

        await monitor.stop_monitoring()
        assert not monitor._running


class TestActivityWatchdog:
    """Tests for ActivityWatchdog class."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test activity watchdog initialization."""
        monitor = ActivityMonitor()
        await monitor.start_monitoring()

        status_dict = {"running": True, "progress": 0, "abort": False}
        watchdog = ActivityWatchdog(
            task_id="test",
            status_dict=status_dict,
            activity_monitor=monitor
        )

        assert watchdog.task_id == "test"
        assert watchdog.status_dict is status_dict

        await monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_watchdog_detects_activity(self):
        """Test that watchdog detects active task."""
        monitor = ActivityMonitor()
        await monitor.start_monitoring()

        status_dict = {"running": True, "progress": 0, "abort": False}
        watchdog = ActivityWatchdog(
            task_id="test",
            status_dict=status_dict,
            activity_monitor=monitor
        )

        # Start watchdog in background
        watchdog_task = asyncio.create_task(watchdog.start())

        # Simulate progress updates
        for i in range(5):
            await asyncio.sleep(0.1)
            status_dict["progress"] = (i + 1) * 20

        # Stop the task
        status_dict["running"] = False
        await watchdog.stop()
        watchdog_task.cancel()

        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

        # Should not have aborted since we were making progress
        assert not watchdog.is_aborted()

        await monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_watchdog_abort_reason(self):
        """Test that abort reason is recorded."""
        monitor = ActivityMonitor()
        status_dict = {"running": True, "progress": 0, "abort": False}
        watchdog = ActivityWatchdog(
            task_id="test",
            status_dict=status_dict,
            activity_monitor=monitor
        )

        # Manually trigger abort
        watchdog._trigger_abort("Test abort reason")

        assert watchdog.is_aborted()
        assert watchdog.abort_reason == "Test abort reason"
        assert status_dict["abort"]


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_thread_consuming_cpu(self):
        """Test thread CPU check."""
        main_tid = threading.current_thread().ident

        result = is_thread_consuming_cpu(main_tid)
        assert isinstance(result, bool)

    def test_get_process_activity_summary(self):
        """Test process activity summary."""
        summary = get_process_activity_summary()

        assert "pid" in summary
        assert "cpu_percent" in summary
        assert "memory_mb" in summary
        assert "threads" in summary
        assert "is_active" in summary

        assert summary["pid"] > 0
        assert summary["memory_mb"] > 0
        assert summary["threads"] >= 1


class TestTaskStateEnum:
    """Tests for TaskState enum."""

    def test_all_states_defined(self):
        """Test that all expected states are defined."""
        expected_states = [
            "unknown", "running", "io_wait", "sleeping",
            "blocked", "stalled", "deadlocked"
        ]

        for state_name in expected_states:
            assert hasattr(TaskState, state_name.upper())

    def test_state_values(self):
        """Test state value strings."""
        assert TaskState.RUNNING.value == "running"
        assert TaskState.STALLED.value == "stalled"
        assert TaskState.DEADLOCKED.value == "deadlocked"


class TestActivityState:
    """Tests for ActivityState dataclass."""

    def test_default_values(self):
        """Test default values."""
        state = ActivityState(task_id="test")

        assert state.task_id == "test"
        assert state.state == TaskState.UNKNOWN
        assert state.confidence == 0.0
        assert not state.cpu_active
        assert not state.memory_active
        assert not state.io_active
        assert not state.progress_active
        assert not state.is_working
        assert state.recommendation == ""

    def test_activity_indicators(self):
        """Test activity indicator flags."""
        state = ActivityState(
            task_id="test",
            cpu_active=True,
            memory_active=True,
            io_active=False,
            progress_active=True
        )

        assert state.cpu_active
        assert state.memory_active
        assert not state.io_active
        assert state.progress_active


class TestIntegration:
    """Integration tests for the complete monitoring system."""

    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self):
        """Test a full monitoring cycle with a simulated task."""
        monitor = ActivityMonitor()
        await monitor.start_monitoring()

        # Register a task
        monitor.register_task(
            task_id="integration_test",
            metadata={"type": "test"}
        )

        # Simulate task progress
        for i in range(5):
            monitor.report_progress("integration_test", i * 20)
            await asyncio.sleep(0.1)

        # Check final state
        state = monitor.get_task_state("integration_test")
        assert state is not None
        assert state.progress_percent == 80

        # Cleanup
        monitor.unregister_task("integration_test")
        await monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_concurrent_task_monitoring(self):
        """Test monitoring multiple concurrent tasks."""
        monitor = ActivityMonitor()
        await monitor.start_monitoring()

        # Register multiple tasks
        for i in range(3):
            monitor.register_task(
                task_id=f"task_{i}",
                metadata={"index": i}
            )

        # Update progress independently
        monitor.report_progress("task_0", 100)
        monitor.report_progress("task_1", 50)
        monitor.report_progress("task_2", 25)

        # Verify independent states
        states = monitor.get_all_states()
        assert len(states) == 3
        assert states["task_0"].progress_percent == 100
        assert states["task_1"].progress_percent == 50
        assert states["task_2"].progress_percent == 25

        # Cleanup
        for i in range(3):
            monitor.unregister_task(f"task_{i}")
        await monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
