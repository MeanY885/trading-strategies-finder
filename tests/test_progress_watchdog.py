"""
Tests for Progress-Based Watchdog System.

These tests verify that the progress-based watchdog:
1. NEVER uses absolute time limits
2. Detects completion through progress signals
3. Handles sparse VectorBT updates correctly
4. Only aborts when truly stuck (not just slow)
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

import sys
import os

# Add backend to path and import directly from file to avoid services/__init__.py
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

# Import directly to avoid services/__init__.py which has database dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "progress_watchdog",
    os.path.join(backend_path, 'services', 'progress_watchdog.py')
)
progress_watchdog_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress_watchdog_module)

ProgressVelocityTracker = progress_watchdog_module.ProgressVelocityTracker
SignalCountStallDetector = progress_watchdog_module.SignalCountStallDetector
CompletionEventDetector = progress_watchdog_module.CompletionEventDetector
ProgressBasedWatchdog = progress_watchdog_module.ProgressBasedWatchdog
WatchdogCoordinator = progress_watchdog_module.WatchdogCoordinator
create_progress_watchdog = progress_watchdog_module.create_progress_watchdog
notify_task_completed = progress_watchdog_module.notify_task_completed


class TestProgressVelocityTracker:
    """Tests for velocity-based progress tracking."""

    def test_velocity_calculation(self):
        """Test that velocity is calculated per-measurement, not per-time."""
        tracker = ProgressVelocityTracker()

        # Make 10 measurements with 1% progress each
        for i in range(10):
            dp = tracker.update(i + 1)
            assert dp.velocity == 1.0  # 1% per measurement

    def test_is_making_progress_with_movement(self):
        """Task making progress should be detected."""
        tracker = ProgressVelocityTracker()

        for i in range(25):
            tracker.update(i * 0.5)  # 0.5% per measurement

        assert tracker.is_making_progress()

    def test_is_making_progress_stalled(self):
        """Stalled task should be detected."""
        tracker = ProgressVelocityTracker()

        # First make some progress
        for i in range(10):
            tracker.update(i)

        # Then stall (same value repeatedly)
        for _ in range(30):
            tracker.update(10)

        assert not tracker.is_making_progress()

    def test_velocity_trend_accelerating(self):
        """Detect accelerating progress."""
        tracker = ProgressVelocityTracker()

        # Increasing velocity: 0.1, 0.2, 0.3, ...
        progress = 0
        for i in range(25):
            progress += (i + 1) * 0.01
            tracker.update(progress)

        trend = tracker.get_trend()
        assert trend in ["accelerating", "steady"]  # Depending on thresholds

    def test_velocity_trend_stalled(self):
        """Detect stalled progress."""
        tracker = ProgressVelocityTracker()

        # Make some initial progress
        for i in range(10):
            tracker.update(i)

        # Then completely stall
        for _ in range(25):
            tracker.update(10)

        assert tracker.get_trend() == "stalled"

    def test_no_time_dependency(self):
        """Verify velocity is NOT affected by wall clock time."""
        tracker = ProgressVelocityTracker()

        # Fast measurements
        for i in range(5):
            tracker.update(i * 2)

        fast_velocity = tracker.get_rolling_velocity()

        # Reset and do same progress with delays
        tracker2 = ProgressVelocityTracker()
        for i in range(5):
            time.sleep(0.01)  # Small delay
            tracker2.update(i * 2)

        slow_velocity = tracker2.get_rolling_velocity()

        # Velocities should be the same (time doesn't matter)
        assert abs(fast_velocity - slow_velocity) < 0.001


class TestSignalCountStallDetector:
    """Tests for signal-count-based stall detection."""

    def test_no_stall_with_progress(self):
        """No stall detected when progress is being made."""
        detector = SignalCountStallDetector()

        for i in range(30):
            result = detector.update(i)

        assert not result["warning"]
        assert not result["should_abort"]

    def test_stall_detected_unchanged(self):
        """Stall detected after many unchanged measurements."""
        detector = SignalCountStallDetector()

        # First make some progress
        for i in range(5):
            detector.update(i)

        # Then send same value repeatedly
        for _ in range(25):
            result = detector.update(5)

        assert result["warning"]
        assert result["consecutive_unchanged"] >= 20

    def test_abort_threshold(self):
        """Abort triggered after extended stall."""
        detector = SignalCountStallDetector()

        # Stall for 55 measurements (above UNCHANGED_ABORT=50)
        for _ in range(55):
            result = detector.update(50.0)

        assert result["should_abort"]

    def test_stall_reset_on_progress(self):
        """Stall counter resets when progress resumes."""
        detector = SignalCountStallDetector()

        # Stall for a while
        for _ in range(30):
            detector.update(10.0)

        # Resume progress
        detector.update(11.0)

        # Counter should reset
        result = detector.update(12.0)
        assert result["consecutive_unchanged"] == 0


class TestCompletionEventDetector:
    """Tests for event-driven completion detection."""

    def test_progress_100_completion(self):
        """Detect completion via 100% progress."""
        detector = CompletionEventDetector()

        assert not detector.check_progress(50.0)
        assert detector.check_progress(100.0)
        assert "progress_100" in detector.signals_received

    def test_result_available_completion(self):
        """Detect completion via result availability."""
        detector = CompletionEventDetector()

        assert not detector.check_result(None)
        assert detector.check_result({"top_10": []})
        assert "result_available" in detector.signals_received

    def test_running_flag_completion(self):
        """Detect completion via running flag."""
        detector = CompletionEventDetector()

        assert not detector.check_running_flag(True)
        assert detector.check_running_flag(False)
        assert "running_false" in detector.signals_received

    def test_combo_count_completion(self):
        """Detect completion via combo count."""
        detector = CompletionEventDetector()

        assert not detector.check_combo_count(1000, 52800)
        assert detector.check_combo_count(52800, 52800)
        assert "all_combos_done" in detector.signals_received

    def test_multiple_signals_for_completion(self):
        """Require multiple signals for confident completion."""
        detector = CompletionEventDetector()

        # Single signal not enough
        detector.check_progress(100)
        assert not detector.is_completed

        # Two signals = completed
        detector.check_result({"top_10": []})
        assert detector.is_completed


class TestProgressBasedWatchdog:
    """Integration tests for the full watchdog."""

    @pytest.mark.asyncio
    async def test_completion_detection(self):
        """Watchdog detects completion correctly."""
        status = {
            "running": True,
            "progress": 0,
            "report": None,
            "abort": False
        }

        watchdog = ProgressBasedWatchdog(
            task_id="test_completion",
            status_dict=status,
            total_combinations=100
        )

        # Start watchdog in background
        task = asyncio.create_task(watchdog.start())

        # Simulate progress
        for i in range(100):
            await asyncio.sleep(0.01)
            status["progress"] = i + 1

        # Signal completion
        status["running"] = False
        status["report"] = {"top_10": []}

        # Wait for watchdog to finish
        await asyncio.wait_for(task, timeout=2.0)

        assert not watchdog.is_aborted

    @pytest.mark.asyncio
    async def test_stall_detection(self):
        """Watchdog detects and reports stalls."""
        status = {
            "running": True,
            "progress": 50.0,  # Stuck at 50%
            "report": None,
            "abort": False
        }

        watchdog = ProgressBasedWatchdog(
            task_id="test_stall",
            status_dict=status,
            total_combinations=100
        )

        # Override thresholds for faster test
        watchdog.stall_detector.UNCHANGED_ABORT = 10
        watchdog.CHECK_INTERVAL = 0.01

        # Start watchdog
        task = asyncio.create_task(watchdog.start())

        # Wait for stall detection (should happen after 10 measurements)
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            await watchdog.stop()

        assert watchdog.is_aborted
        assert "stalled" in watchdog.abort_reason.lower()

    @pytest.mark.asyncio
    async def test_no_abort_with_slow_progress(self):
        """Watchdog does NOT abort slow but progressing tasks."""
        status = {
            "running": True,
            "progress": 0,
            "report": None,
            "abort": False
        }

        watchdog = ProgressBasedWatchdog(
            task_id="test_slow",
            status_dict=status,
            total_combinations=100
        )
        watchdog.CHECK_INTERVAL = 0.01

        # Start watchdog
        task = asyncio.create_task(watchdog.start())

        # Make very slow but consistent progress
        for i in range(50):
            await asyncio.sleep(0.02)  # Slow
            status["progress"] = i * 2  # But progressing

        # Complete the task
        status["running"] = False
        status["progress"] = 100
        status["report"] = {"top_10": []}

        await asyncio.wait_for(task, timeout=2.0)

        assert not watchdog.is_aborted

    @pytest.mark.asyncio
    async def test_sparse_updates_handled(self):
        """Watchdog handles sparse VectorBT-style updates."""
        status = {
            "running": True,
            "progress": 0,
            "report": None,
            "abort": False
        }

        watchdog = ProgressBasedWatchdog(
            task_id="test_sparse",
            status_dict=status,
            total_combinations=52800
        )
        watchdog.CHECK_INTERVAL = 0.01

        # Start watchdog
        task = asyncio.create_task(watchdog.start())

        # Simulate sparse updates (like VectorBT every 10 combos)
        # Many measurements at same value, then jump
        for batch in range(10):
            for _ in range(15):  # 15 measurements at same value
                await asyncio.sleep(0.005)

            # Progress jumps after batch
            status["progress"] = (batch + 1) * 10

        status["running"] = False
        status["progress"] = 100
        status["report"] = {"top_10": []}

        await asyncio.wait_for(task, timeout=5.0)

        # Should NOT abort because progress IS being made (just sparse)
        assert not watchdog.is_aborted


class TestWatchdogCoordinator:
    """Tests for multi-task coordination."""

    def test_registration(self):
        """Test watchdog registration."""
        coordinator = WatchdogCoordinator()
        status = {"running": True, "progress": 0, "abort": False}

        watchdog = ProgressBasedWatchdog("test1", status, 100)
        coordinator.register(watchdog)

        assert "test1" in coordinator.watchdogs

    def test_unregistration(self):
        """Test watchdog unregistration."""
        coordinator = WatchdogCoordinator()
        status = {"running": True, "progress": 0, "abort": False}

        watchdog = ProgressBasedWatchdog("test2", status, 100)
        coordinator.register(watchdog)
        coordinator.unregister("test2")

        assert "test2" not in coordinator.watchdogs
        assert coordinator._completed_count == 1

    @pytest.mark.asyncio
    async def test_heartbeat(self):
        """Test heartbeat triggering."""
        coordinator = WatchdogCoordinator()
        status = {"running": True, "progress": 0, "abort": False}

        watchdog = ProgressBasedWatchdog("test3", status, 100)
        coordinator.register(watchdog)

        # Heartbeat should not raise
        await coordinator.heartbeat()


class TestNoTimeBasedAbort:
    """
    Critical tests ensuring NO time-based abort behavior.

    These tests verify the fundamental requirement that the
    watchdog NEVER uses wall clock time for abort decisions.
    """

    @pytest.mark.asyncio
    async def test_long_running_task_not_aborted(self):
        """A task making progress is NEVER aborted regardless of duration."""
        status = {
            "running": True,
            "progress": 0,
            "report": None,
            "abort": False
        }

        watchdog = ProgressBasedWatchdog(
            task_id="long_runner",
            status_dict=status,
            total_combinations=1000
        )
        watchdog.CHECK_INTERVAL = 0.001  # Very fast checks

        task = asyncio.create_task(watchdog.start())

        # Run for many iterations (simulating long task)
        start = time.time()
        progress = 0
        while progress < 100:
            await asyncio.sleep(0.002)
            progress += 0.1  # Tiny progress
            status["progress"] = progress

        status["running"] = False
        status["report"] = {"top_10": []}

        await asyncio.wait_for(task, timeout=10.0)

        # Task ran for >2 seconds but was NOT aborted
        assert not watchdog.is_aborted
        assert time.time() - start > 2.0  # Verify it ran for a while

    def test_velocity_tracker_ignores_time(self):
        """Velocity tracker only cares about progress, not time."""
        tracker = ProgressVelocityTracker()

        # Measure 1: immediate
        tracker.update(10)

        # Measure 2: after delay
        time.sleep(0.1)
        tracker.update(20)

        # Measure 3: immediate
        tracker.update(30)

        # All velocities should be 10 (progress per measurement)
        # NOT affected by the time.sleep
        velocities = [dp.velocity for dp in tracker.history]
        assert all(v == 10 for v in velocities[1:])  # First is 10 (initial)

    def test_stall_detector_counts_measurements_not_seconds(self):
        """Stall detector counts measurements, not seconds."""
        detector = SignalCountStallDetector()

        # Make measurements with varying delays
        for _ in range(10):
            time.sleep(0.01)  # Some delay
            result = detector.update(50.0)

        # Should report 9 consecutive unchanged (first sets the baseline)
        assert result["consecutive_unchanged"] == 9
        assert result["total_measurements"] == 10


class TestVectorBTScenarios:
    """
    Tests simulating real VectorBT optimization scenarios.
    """

    @pytest.mark.asyncio
    async def test_52800_combination_optimization(self):
        """Simulate full VectorBT optimization with 52,800 combos."""
        status = {
            "running": True,
            "progress": 0,
            "report": None,
            "abort": False,
            "trial_current": 0,
            "trial_total": 52800
        }

        watchdog = ProgressBasedWatchdog(
            task_id="vectorbt_full",
            status_dict=status,
            total_combinations=52800,
            combo_current_key="trial_current",
            combo_total_key="trial_total"
        )
        watchdog.CHECK_INTERVAL = 0.001

        task = asyncio.create_task(watchdog.start())

        # Simulate VectorBT progress (updates every 10 combos)
        for combo in range(0, 5280, 10):  # Simulate 10% of full run
            await asyncio.sleep(0.001)
            status["trial_current"] = combo
            status["progress"] = (combo / 52800) * 100

        # Complete
        status["running"] = False
        status["progress"] = 100
        status["trial_current"] = 52800
        status["report"] = {"top_10": [{"strategy_name": "rsi_extreme"}]}

        await asyncio.wait_for(task, timeout=10.0)

        assert not watchdog.is_aborted

    @pytest.mark.asyncio
    async def test_vectorbt_batch_processing(self):
        """
        Simulate VectorBT batch processing where progress jumps
        between batches (sparse updates).
        """
        status = {
            "running": True,
            "progress": 0,
            "report": None,
            "abort": False
        }

        watchdog = ProgressBasedWatchdog(
            task_id="batch_test",
            status_dict=status,
            total_combinations=1000
        )
        # Increase thresholds to handle batch gaps
        watchdog.stall_detector.UNCHANGED_WARNING = 30
        watchdog.stall_detector.UNCHANGED_ABORT = 60
        watchdog.CHECK_INTERVAL = 0.001

        task = asyncio.create_task(watchdog.start())

        # Simulate batch processing: stuck between batches, then jumps
        for batch in range(10):
            # 20 measurements at same progress (simulating batch processing)
            for _ in range(20):
                await asyncio.sleep(0.001)

            # Progress jumps 10% after batch completes
            status["progress"] = (batch + 1) * 10

        status["running"] = False
        status["progress"] = 100
        status["report"] = {"top_10": []}

        await asyncio.wait_for(task, timeout=10.0)

        # Should NOT abort - progress IS happening, just in batches
        assert not watchdog.is_aborted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
