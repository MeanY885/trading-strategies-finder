#!/usr/bin/env python3
"""
CONCURRENCY BENCHMARK TOOL
==========================
Tests different concurrent task / workers-per-task configurations
to find optimal settings for your specific hardware.

Usage:
    python backend/tools/benchmark_concurrency.py

Output:
    - Benchmark results for each configuration
    - Recommended settings based on your hardware
    - CSV export for analysis
"""

import os
import sys
import time
import psutil
import asyncio
import statistics
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress vectorbt warnings during benchmark
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    concurrent_tasks: int
    workers_per_task: int
    total_threads: int
    duration_seconds: float
    combinations_tested: int
    combinations_per_second: float
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_memory_gb: float
    peak_memory_gb: float
    efficiency_score: float  # combinations/sec normalized by resource usage


@dataclass
class SystemInfo:
    """System hardware information."""
    cpu_cores: int
    cpu_threads: int
    memory_total_gb: float
    memory_available_gb: float
    cpu_model: str


def get_system_info() -> SystemInfo:
    """Detect system hardware."""
    import platform

    cpu_cores = psutil.cpu_count(logical=False) or 4
    cpu_threads = psutil.cpu_count(logical=True) or 4
    mem = psutil.virtual_memory()

    # Try to get CPU model
    cpu_model = "Unknown"
    try:
        if platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True)
            cpu_model = result.stdout.strip()
        elif platform.system() == "Linux":
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_model = line.split(':')[1].strip()
                        break
    except:
        pass

    return SystemInfo(
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        memory_total_gb=mem.total / (1024**3),
        memory_available_gb=mem.available / (1024**3),
        cpu_model=cpu_model
    )


def cpu_intensive_work(iterations: int) -> float:
    """Simulate CPU-intensive optimization work."""
    import math
    result = 0.0
    for i in range(iterations):
        result += math.sin(i) * math.cos(i) * math.sqrt(abs(i) + 1)
    return result


def worker_task(args: Tuple[int, int]) -> Tuple[int, float]:
    """Worker function for ProcessPoolExecutor."""
    task_id, iterations = args
    start = time.perf_counter()
    result = cpu_intensive_work(iterations)
    duration = time.perf_counter() - start
    return task_id, duration


class ResourceMonitor:
    """Monitor CPU and memory during benchmark."""

    def __init__(self):
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self._running = False

    async def start(self):
        """Start monitoring in background."""
        self._running = True
        while self._running:
            self.cpu_samples.append(psutil.cpu_percent(interval=0.1))
            self.memory_samples.append(psutil.virtual_memory().used / (1024**3))
            await asyncio.sleep(0.2)

    def stop(self):
        """Stop monitoring."""
        self._running = False

    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        if not self.cpu_samples:
            return {"avg_cpu": 0, "peak_cpu": 0, "avg_mem": 0, "peak_mem": 0}

        return {
            "avg_cpu": statistics.mean(self.cpu_samples),
            "peak_cpu": max(self.cpu_samples),
            "avg_mem": statistics.mean(self.memory_samples),
            "peak_mem": max(self.memory_samples),
        }


async def run_benchmark_config(
    concurrent_tasks: int,
    workers_per_task: int,
    work_units: int = 1000,
    iterations_per_unit: int = 50000
) -> BenchmarkResult:
    """
    Run benchmark with specific configuration.

    Args:
        concurrent_tasks: Number of parallel optimization tasks
        workers_per_task: ProcessPoolExecutor workers per task
        work_units: Total work units to process
        iterations_per_unit: CPU iterations per work unit
    """
    total_threads = concurrent_tasks * workers_per_task

    print(f"\n  Testing: {concurrent_tasks} tasks × {workers_per_task} workers = {total_threads} threads")

    # Start resource monitoring
    monitor = ResourceMonitor()
    monitor_task = asyncio.create_task(monitor.start())

    start_time = time.perf_counter()
    completed = 0

    # Simulate concurrent optimization tasks
    async def run_single_task(task_num: int, units_per_task: int):
        nonlocal completed

        work_items = [(i, iterations_per_unit) for i in range(units_per_task)]

        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=workers_per_task) as executor:
            # Submit all work
            futures = [loop.run_in_executor(executor, worker_task, item) for item in work_items]

            # Wait for completion
            for future in asyncio.as_completed(futures):
                await future
                completed += 1

    # Distribute work across concurrent tasks
    units_per_task = work_units // concurrent_tasks
    tasks = [run_single_task(i, units_per_task) for i in range(concurrent_tasks)]

    await asyncio.gather(*tasks)

    duration = time.perf_counter() - start_time

    # Stop monitoring
    monitor.stop()
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

    stats = monitor.get_stats()
    combinations_per_second = completed / duration if duration > 0 else 0

    # Calculate efficiency score
    # Higher is better: throughput normalized by resource usage
    resource_factor = (stats["avg_cpu"] / 100) * (stats["avg_mem"] / 8)  # Normalize
    efficiency = combinations_per_second / max(0.1, resource_factor)

    return BenchmarkResult(
        concurrent_tasks=concurrent_tasks,
        workers_per_task=workers_per_task,
        total_threads=total_threads,
        duration_seconds=duration,
        combinations_tested=completed,
        combinations_per_second=combinations_per_second,
        avg_cpu_percent=stats["avg_cpu"],
        peak_cpu_percent=stats["peak_cpu"],
        avg_memory_gb=stats["avg_mem"],
        peak_memory_gb=stats["peak_mem"],
        efficiency_score=efficiency
    )


def generate_test_configs(system: SystemInfo) -> List[Tuple[int, int]]:
    """Generate configurations to test based on system specs."""
    configs = []
    cpu_threads = system.cpu_threads

    # Strategy: Test various concurrent_tasks × workers combinations
    # that don't exceed total CPU threads

    # Option 1: Many tasks, few workers each (current default approach)
    for tasks in [1, 2, 4, 8, 16]:
        if tasks <= cpu_threads:
            workers = max(2, cpu_threads // tasks)
            if tasks * workers <= cpu_threads * 2:  # Allow some oversubscription
                configs.append((tasks, min(workers, 16)))

    # Option 2: Few tasks, many workers each (user's preferred approach)
    for workers in [4, 8, 12, 16, 24, 32]:
        if workers <= cpu_threads:
            tasks = max(1, cpu_threads // workers)
            for t in [1, 2, max(1, tasks // 2), tasks]:
                if t >= 1 and (t, workers) not in configs:
                    configs.append((t, workers))

    # Remove duplicates and sort
    configs = list(set(configs))
    configs.sort(key=lambda x: (x[0], x[1]))

    # Limit to reasonable number of tests
    return configs[:12]


def print_results_table(results: List[BenchmarkResult], system: SystemInfo):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(f"\nSystem: {system.cpu_model}")
    print(f"Cores: {system.cpu_cores} physical, {system.cpu_threads} logical")
    print(f"Memory: {system.memory_total_gb:.1f} GB total, {system.memory_available_gb:.1f} GB available")
    print()

    # Header
    print(f"{'Tasks':<6} {'Workers':<8} {'Threads':<8} {'Time(s)':<10} {'Comb/sec':<12} "
          f"{'CPU%':<8} {'Mem(GB)':<10} {'Efficiency':<12}")
    print("-" * 100)

    # Sort by combinations per second (throughput)
    sorted_results = sorted(results, key=lambda r: r.combinations_per_second, reverse=True)

    for i, r in enumerate(sorted_results):
        marker = " *BEST*" if i == 0 else ""
        print(f"{r.concurrent_tasks:<6} {r.workers_per_task:<8} {r.total_threads:<8} "
              f"{r.duration_seconds:<10.2f} {r.combinations_per_second:<12.1f} "
              f"{r.avg_cpu_percent:<8.1f} {r.avg_memory_gb:<10.2f} {r.efficiency_score:<12.2f}{marker}")

    print()

    # Recommendations
    best = sorted_results[0]
    most_efficient = max(results, key=lambda r: r.efficiency_score)

    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print(f"\n  FASTEST THROUGHPUT:")
    print(f"    CORES_PER_TASK={best.workers_per_task}")
    print(f"    → {best.concurrent_tasks} concurrent tasks × {best.workers_per_task} workers")
    print(f"    → {best.combinations_per_second:.1f} combinations/sec")

    if most_efficient != best:
        print(f"\n  MOST EFFICIENT (throughput/resources):")
        print(f"    CORES_PER_TASK={most_efficient.workers_per_task}")
        print(f"    → {most_efficient.concurrent_tasks} tasks × {most_efficient.workers_per_task} workers")
        print(f"    → Uses less resources with good throughput")

    print(f"\n  ENVIRONMENT VARIABLES TO SET:")
    print(f"    export CORES_PER_TASK={best.workers_per_task}")
    print(f"    # This will auto-calculate {best.concurrent_tasks} concurrent tasks on this system")
    print()

    # Generate .env snippet
    env_snippet = f"""
# === AUTO-GENERATED OPTIMIZER CONFIG ===
# System: {system.cpu_model}
# Cores: {system.cpu_threads}, RAM: {system.memory_total_gb:.0f}GB
# Benchmark: {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Throughput: {best.combinations_per_second:.0f} combinations/sec

CORES_PER_TASK={best.workers_per_task}
MEMORY_PER_TASK_GB=2.0
RESERVED_CORES=2
RESERVED_MEMORY_GB=2.0
# Result: {best.concurrent_tasks} concurrent tasks × {best.workers_per_task} workers
"""
    print("  .env SNIPPET (copy to your deployment):")
    print("-" * 50)
    print(env_snippet)
    print("-" * 50)

    # Save to file
    env_path = Path(__file__).parent / f"recommended_config_{system.cpu_threads}cores.env"
    with open(env_path, 'w') as f:
        f.write(env_snippet.strip())
    print(f"\n  Saved to: {env_path}")

    return best


def export_csv(results: List[BenchmarkResult], system: SystemInfo, filepath: str):
    """Export results to CSV for further analysis."""
    import csv

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # System info header
        writer.writerow(['System Info'])
        writer.writerow(['CPU Model', system.cpu_model])
        writer.writerow(['Physical Cores', system.cpu_cores])
        writer.writerow(['Logical Threads', system.cpu_threads])
        writer.writerow(['Total Memory GB', f"{system.memory_total_gb:.2f}"])
        writer.writerow(['Available Memory GB', f"{system.memory_available_gb:.2f}"])
        writer.writerow([])

        # Results
        writer.writerow(['Benchmark Results'])
        writer.writerow(['Tasks', 'Workers', 'Total Threads', 'Duration(s)',
                        'Combinations/sec', 'Avg CPU%', 'Peak CPU%',
                        'Avg Mem GB', 'Peak Mem GB', 'Efficiency'])

        for r in results:
            writer.writerow([
                r.concurrent_tasks, r.workers_per_task, r.total_threads,
                f"{r.duration_seconds:.2f}", f"{r.combinations_per_second:.2f}",
                f"{r.avg_cpu_percent:.1f}", f"{r.peak_cpu_percent:.1f}",
                f"{r.avg_memory_gb:.2f}", f"{r.peak_memory_gb:.2f}",
                f"{r.efficiency_score:.2f}"
            ])

    print(f"Results exported to: {filepath}")


async def main():
    """Run the benchmark suite."""
    print("=" * 60)
    print("CONCURRENCY BENCHMARK TOOL")
    print("=" * 60)
    print("\nThis tool tests different concurrent task configurations")
    print("to find the optimal settings for your hardware.\n")

    # Detect system
    system = get_system_info()
    print(f"Detected System:")
    print(f"  CPU: {system.cpu_model}")
    print(f"  Cores: {system.cpu_cores} physical, {system.cpu_threads} logical")
    print(f"  Memory: {system.memory_total_gb:.1f} GB total, {system.memory_available_gb:.1f} GB available")

    # Generate test configurations
    configs = generate_test_configs(system)
    print(f"\nTesting {len(configs)} configurations...")
    print("(Each test simulates optimization workload)\n")

    # Warm-up
    print("Warming up CPU...")
    cpu_intensive_work(100000)

    # Run benchmarks
    results: List[BenchmarkResult] = []

    for i, (tasks, workers) in enumerate(configs):
        print(f"\nBenchmark {i+1}/{len(configs)}")
        try:
            result = await run_benchmark_config(
                concurrent_tasks=tasks,
                workers_per_task=workers,
                work_units=200,  # Reduced for faster benchmarking
                iterations_per_unit=30000
            )
            results.append(result)
            print(f"    Completed: {result.combinations_per_second:.1f} comb/sec, "
                  f"CPU: {result.avg_cpu_percent:.1f}%, Mem: {result.avg_memory_gb:.1f}GB")
        except Exception as e:
            print(f"    Failed: {e}")

        # Brief pause between tests to let system settle
        await asyncio.sleep(1)

    if not results:
        print("\nNo successful benchmarks. Check system resources.")
        return

    # Print results
    best = print_results_table(results, system)

    # Export CSV
    csv_path = Path(__file__).parent / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_csv(results, system, str(csv_path))

    return best


if __name__ == "__main__":
    asyncio.run(main())
