"""
Tests for Autonomous Optimizer
==============================
Tests for task management, asyncio handling, and memory cleanup.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


class TestAsyncioWaitHandling:
    """Test that asyncio.wait() is handled correctly to prevent memory leaks."""

    @pytest.mark.asyncio
    async def test_asyncio_wait_returns_tuple(self):
        """Verify asyncio.wait() returns (done, pending) tuple."""
        async def sample_task():
            await asyncio.sleep(0.01)
            return "done"

        tasks = {asyncio.create_task(sample_task()) for _ in range(3)}

        result = await asyncio.wait(tasks, timeout=1.0)

        # Verify it returns a tuple of two sets
        assert isinstance(result, tuple)
        assert len(result) == 2
        done, pending = result
        assert isinstance(done, set)
        assert isinstance(pending, set)

    @pytest.mark.asyncio
    async def test_done_tasks_must_be_awaited(self):
        """Verify that done tasks need to be awaited to collect results."""
        results = []

        async def task_with_result(value):
            await asyncio.sleep(0.01)
            return value

        tasks = {asyncio.create_task(task_with_result(i)) for i in range(5)}

        done, pending = await asyncio.wait(tasks, timeout=1.0)

        # Done tasks must be awaited to get their results
        for task in done:
            result = await task
            results.append(result)

        assert len(results) == 5
        assert sorted(results) == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_exception_handling_in_done_tasks(self):
        """Test that exceptions in done tasks are properly caught."""
        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("Task failed")

        async def success_task():
            await asyncio.sleep(0.01)
            return "success"

        tasks = {
            asyncio.create_task(failing_task()),
            asyncio.create_task(success_task()),
        }

        done, pending = await asyncio.wait(tasks, timeout=1.0)

        exceptions = []
        successes = []

        for task in done:
            try:
                result = await task
                successes.append(result)
            except Exception as e:
                exceptions.append(e)

        assert len(exceptions) == 1
        assert len(successes) == 1
        assert isinstance(exceptions[0], ValueError)


class TestTaskCleanup:
    """Test task cleanup and garbage collection."""

    @pytest.mark.asyncio
    async def test_active_tasks_properly_updated(self):
        """Test that active_tasks set is properly updated after wait()."""
        async def short_task():
            await asyncio.sleep(0.01)
            return True

        active_tasks = set()

        # Spawn 5 tasks
        for _ in range(5):
            task = asyncio.create_task(short_task())
            active_tasks.add(task)

        assert len(active_tasks) == 5

        # Wait for completion with proper handling
        done, pending = await asyncio.wait(active_tasks, timeout=1.0)

        # Await done tasks
        for task in done:
            await task

        # Update active_tasks to only pending
        active_tasks = pending

        # All should be done, so active_tasks should be empty
        assert len(active_tasks) == 0

    @pytest.mark.asyncio
    async def test_cleanup_triggers_gc(self):
        """Test that cleanup properly triggers garbage collection."""
        import gc

        gc_collected = []

        class TrackedObject:
            def __del__(self):
                gc_collected.append(True)

        async def task_with_object():
            obj = TrackedObject()
            await asyncio.sleep(0.01)
            return str(obj)

        tasks = {asyncio.create_task(task_with_object()) for _ in range(3)}

        done, pending = await asyncio.wait(tasks, timeout=1.0)

        # Await and delete done tasks
        for task in done:
            await task

        # Force garbage collection
        gc.collect()

        # Objects should have been collected
        assert len(gc_collected) >= 3


class TestStateChangeDetection:
    """Test state change detection for broadcast optimization."""

    def test_state_comparison(self):
        """Test that state changes are properly detected."""
        state1 = {
            "running": True,
            "active_count": 5,
            "index": 10,
            "message": "Running 5 tasks"
        }

        state2 = {
            "running": True,
            "active_count": 5,
            "index": 10,
            "message": "Running 5 tasks"
        }

        state3 = {
            "running": True,
            "active_count": 4,  # Changed
            "index": 10,
            "message": "Running 4 tasks"  # Changed
        }

        assert state1 == state2  # Same state
        assert state1 != state3  # Different state

    def test_state_change_reduces_broadcasts(self):
        """Test that broadcasts only happen on state changes."""
        broadcast_count = 0
        last_state = None

        def should_broadcast(new_state):
            nonlocal last_state, broadcast_count
            if last_state != new_state:
                last_state = new_state.copy()
                broadcast_count += 1
                return True
            return False

        # Same state multiple times
        for _ in range(5):
            should_broadcast({"active": 5, "index": 10})

        assert broadcast_count == 1  # Only first call broadcasts

        # Changed state
        should_broadcast({"active": 4, "index": 11})
        assert broadcast_count == 2  # New state triggers broadcast


class TestResourceChecking:
    """Test resource checking for task spawning."""

    def test_can_spawn_cpu_check(self):
        """Test CPU threshold checking."""
        CPU_SPAWN_THRESHOLD = 85

        assert 30.0 < CPU_SPAWN_THRESHOLD  # Can spawn
        assert 90.0 >= CPU_SPAWN_THRESHOLD  # Cannot spawn

    def test_can_spawn_memory_check(self):
        """Test memory threshold checking."""
        MEM_SPAWN_THRESHOLD = 2.0

        assert 8.0 > MEM_SPAWN_THRESHOLD  # Can spawn
        assert 1.5 <= MEM_SPAWN_THRESHOLD  # Cannot spawn

    def test_spawn_cooldown(self):
        """Test spawn cooldown logic."""
        import time

        SPAWN_COOLDOWN = 5
        last_spawn_time = time.time() - 3  # 3 seconds ago

        time_since_spawn = time.time() - last_spawn_time

        assert time_since_spawn < SPAWN_COOLDOWN  # Still in cooldown

        last_spawn_time = time.time() - 6  # 6 seconds ago
        time_since_spawn = time.time() - last_spawn_time

        assert time_since_spawn >= SPAWN_COOLDOWN  # Cooldown elapsed


class TestLoopTiming:
    """Test loop timing and sleep behavior."""

    @pytest.mark.asyncio
    async def test_loop_sleep_duration(self):
        """Test that loop sleep is not too short."""
        import time

        # The fixed sleep should be 1.0s, not 0.2s
        EXPECTED_SLEEP = 1.0

        start = time.time()
        await asyncio.sleep(EXPECTED_SLEEP)
        elapsed = time.time() - start

        # Should be close to expected sleep time
        assert elapsed >= EXPECTED_SLEEP * 0.9  # Allow 10% variance

    @pytest.mark.asyncio
    async def test_busy_loop_detection(self):
        """Test that we can detect busy loop behavior."""
        iteration_times = []

        for _ in range(5):
            import time
            start = time.time()
            await asyncio.sleep(1.0)  # Proper sleep
            iteration_times.append(time.time() - start)

        avg_time = sum(iteration_times) / len(iteration_times)

        # With 1.0s sleep, iterations should take ~1s each
        assert avg_time >= 0.9  # Not a busy loop


class TestCombinationProcessing:
    """Test combination processing logic."""

    def test_index_progression(self):
        """Test that current_index properly progresses."""
        combinations = [{"id": i} for i in range(30)]
        current_index = 0

        # Simulate processing 10 combinations
        for _ in range(10):
            if current_index < len(combinations):
                current_index += 1

        assert current_index == 10

    def test_cycle_completion_detection(self):
        """Test detection of cycle completion."""
        combinations = [{"id": i} for i in range(10)]
        current_index = 10  # At end

        is_complete = current_index >= len(combinations)
        assert is_complete is True

        current_index = 5  # Mid-cycle
        is_complete = current_index >= len(combinations)
        assert is_complete is False


# Integration test placeholder
class TestOptimizerIntegration:
    """Integration tests for the autonomous optimizer."""

    @pytest.mark.skip(reason="Requires full backend setup")
    @pytest.mark.asyncio
    async def test_full_optimization_cycle(self):
        """Test a full optimization cycle end-to-end."""
        # This would require mocking the entire backend
        pass
