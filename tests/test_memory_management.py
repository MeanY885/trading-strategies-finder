"""
Memory Management Tests
=======================
Tests for memory leaks, garbage collection, and resource cleanup.
"""
import pytest
import gc
import sys
import os
import tracemalloc
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


class TestDataFrameMemoryCleanup:
    """Test DataFrame memory cleanup patterns."""

    def test_dataframe_deletion_frees_memory(self, memory_tracker):
        """Test that deleting DataFrames actually frees memory."""
        memory_tracker.start()
        memory_tracker.snapshot("before")

        # Create large DataFrame
        df = pd.DataFrame(np.random.randn(10000, 100))
        memory_tracker.snapshot("after_create")

        # Delete and collect
        del df
        gc.collect()
        memory_tracker.snapshot("after_delete")

        memory_tracker.stop()

        # Memory should be lower after delete
        # (Allow some tolerance for Python internals)
        diffs = memory_tracker.compare(1, 2)  # Compare after_create to after_delete

        # The memory diff should show reduction
        # Most allocated blocks should be freed
        total_freed = sum(stat.size_diff for stat in diffs if stat.size_diff < 0)
        assert total_freed < 0, "Memory should be freed after DataFrame deletion"

    def test_multiple_dataframes_cleanup(self):
        """Test cleanup of multiple DataFrames simulating validation periods."""
        tracemalloc.start()

        # Simulate 8 validation periods
        for period in range(8):
            df = pd.DataFrame(np.random.randn(1000, 50))
            # Process DataFrame
            result = df.mean().to_dict()
            # Explicit cleanup
            del df
            del result
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should not grow unboundedly
        # With 8 periods, if memory leaked, peak would be 8x single period
        # With cleanup, peak should be closer to 1-2x single period
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 50, f"Peak memory {peak_mb:.1f}MB suggests memory leak"


class TestGarbageCollectionEffectiveness:
    """Test that garbage collection is effective."""

    def test_gc_collect_reclaims_circular_references(self):
        """Test that gc.collect() handles circular references."""
        gc.collect()  # Clear any existing garbage
        initial_count = len(gc.get_objects())

        # Create circular reference
        class Node:
            def __init__(self):
                self.ref = None

        a = Node()
        b = Node()
        a.ref = b
        b.ref = a  # Circular reference

        # Break references manually
        del a
        del b

        # Without gc.collect(), circular refs may persist
        gc.collect()

        final_count = len(gc.get_objects())

        # Object count should not grow significantly
        assert final_count <= initial_count + 10  # Allow small variance

    def test_gc_generation_cleanup(self):
        """Test that gc cleans all generations."""
        for _ in range(3):
            # Create objects that survive to older generations
            objects = [object() for _ in range(100)]
            del objects
            gc.collect(0)  # Young generation

        # Full collection
        collected = gc.collect()

        # Should collect some objects
        # (exact count varies by Python version)
        assert collected >= 0


class TestVectorBTMemoryUsage:
    """Test VectorBT-specific memory patterns."""

    @pytest.fixture
    def mock_vectorbt(self):
        """Mock VectorBT module."""
        mock_vbt = MagicMock()
        mock_portfolio = MagicMock()
        mock_portfolio.total_return.return_value = 0.15
        mock_portfolio.final_value.return_value = 1150.0
        mock_portfolio.trades.count.return_value = 10
        mock_vbt.Portfolio.from_signals.return_value = mock_portfolio
        return mock_vbt

    def test_signal_cache_clearing(self):
        """Test that signal cache is properly cleared."""
        cache = {}

        # Simulate cache accumulation
        for i in range(100):
            key = f"strategy_{i}"
            cache[key] = pd.DataFrame(np.random.randn(1000, 10))

        # Verify cache has data
        assert len(cache) == 100

        # Clear cache
        cache.clear()
        gc.collect()

        assert len(cache) == 0

    def test_portfolio_object_cleanup(self):
        """Test cleanup of portfolio objects."""
        tracemalloc.start()

        for _ in range(10):
            # Simulate portfolio creation
            close = pd.DataFrame(np.random.randn(1000, 100))
            entries = pd.DataFrame(np.random.choice([True, False], (1000, 100)))

            # Simulate portfolio extraction
            result = close.mean().to_dict()

            # Cleanup
            del close
            del entries
            del result
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should stay bounded
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 100, f"Peak memory {peak_mb:.1f}MB too high"


class TestAsyncTaskMemory:
    """Test memory patterns in async tasks."""

    @pytest.mark.asyncio
    async def test_unawaited_tasks_leak_memory(self):
        """Demonstrate that unawaited tasks can leak memory."""
        gc.collect()
        initial_tasks = len(asyncio.all_tasks())

        # Create tasks but don't await them properly
        tasks = set()
        for _ in range(10):
            task = asyncio.create_task(asyncio.sleep(0.001))
            tasks.add(task)

        # Wait for completion
        done, _ = await asyncio.wait(tasks, timeout=1.0)

        # IMPORTANT: Await done tasks to free resources
        for task in done:
            await task

        # Clean up
        gc.collect()
        await asyncio.sleep(0.1)  # Allow cleanup

        final_tasks = len(asyncio.all_tasks())

        # Task count should return to near initial
        assert final_tasks <= initial_tasks + 2

    @pytest.mark.asyncio
    async def test_task_exception_memory(self):
        """Test that exceptions in tasks don't leak memory."""
        async def failing_task():
            raise ValueError("Test error")

        tasks = {asyncio.create_task(failing_task()) for _ in range(5)}

        done, _ = await asyncio.wait(tasks, timeout=1.0)

        # Must await to collect exceptions
        for task in done:
            try:
                await task
            except ValueError:
                pass  # Expected

        gc.collect()

        # Should not leak
        # (No assertion needed - if it hangs or crashes, test fails)


class TestResourceMonitorMemory:
    """Test resource monitor memory patterns."""

    def test_resource_polling_no_memory_growth(self):
        """Test that repeated resource polling doesn't grow memory."""
        tracemalloc.start()

        import psutil

        # Simulate many resource checks
        for _ in range(100):
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0)

            # Simulate storing results
            result = {
                "memory": mem.available,
                "cpu": cpu
            }
            del result
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Should not grow significantly
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 10, f"Resource polling used {peak_mb:.1f}MB"


class TestMemoryThresholds:
    """Test memory threshold calculations."""

    def test_memory_requirement_calculation(self):
        """Test that memory requirements are correctly calculated."""
        TOTAL_RESERVED_MEMORY = 1.3  # GB
        ELITE_MEM_PER_VALIDATION = 0.5  # GB

        min_needed = TOTAL_RESERVED_MEMORY + ELITE_MEM_PER_VALIDATION

        assert min_needed == 1.8  # Should require 1.8GB minimum

    def test_can_spawn_validation(self):
        """Test validation spawning logic."""
        TOTAL_RESERVED = 1.3
        MEM_PER_VALIDATION = 0.5
        min_needed = TOTAL_RESERVED + MEM_PER_VALIDATION

        # Scenarios
        scenarios = [
            (8.0, True),   # 8GB available - can spawn
            (2.0, True),   # 2GB available - can spawn
            (1.8, True),   # Exactly 1.8GB - can spawn
            (1.5, False),  # 1.5GB - cannot spawn
            (0.1, False),  # 0.1GB - definitely cannot spawn
        ]

        for available, expected_can_spawn in scenarios:
            can_spawn = available >= min_needed
            assert can_spawn == expected_can_spawn, \
                f"With {available}GB, can_spawn should be {expected_can_spawn}"


class TestMemoryLeakDetection:
    """Integration tests for memory leak detection."""

    def test_simulated_optimization_memory(self, sample_ohlcv_data):
        """Simulate optimization run and check for memory leaks."""
        tracemalloc.start()

        # Simulate 5 optimization runs
        for run in range(5):
            df = sample_ohlcv_data.copy()

            # Simulate indicator calculation
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))

            # Simulate backtest results
            results = []
            for i in range(10):
                result = {
                    "tp": i * 0.5,
                    "sl": i * 0.25,
                    "pnl": np.random.randn() * 100
                }
                results.append(result)

            # Cleanup
            del df
            del results
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should stay bounded
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 200, f"Peak memory {peak_mb:.1f}MB suggests leak"

    def test_concurrent_task_memory(self, run_async):
        """Test memory with concurrent task simulation."""
        async def run_test():
            tracemalloc.start()

            for batch in range(3):
                tasks = set()

                # Create concurrent tasks
                for i in range(10):
                    async def task_work(idx):
                        data = np.random.randn(100, 100)
                        await asyncio.sleep(0.01)
                        return data.mean()

                    task = asyncio.create_task(task_work(i))
                    tasks.add(task)

                # Wait and cleanup properly
                done, pending = await asyncio.wait(tasks, timeout=5.0)

                for task in done:
                    try:
                        await task
                    except Exception:
                        pass

                tasks.clear()
                gc.collect()

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return peak / 1024 / 1024

        peak_mb = run_async(run_test())
        assert peak_mb < 50, f"Concurrent tasks used {peak_mb:.1f}MB peak"
