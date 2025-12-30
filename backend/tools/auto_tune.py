#!/usr/bin/env python3
"""
AUTO-TUNE OPTIMIZER SETTINGS
============================
Quick benchmark that runs on container startup to find optimal settings.
Designed to complete in <30 seconds.

Usage:
    # Run and print recommended settings
    python backend/tools/auto_tune.py

    # Run and export to environment (for Docker entrypoint)
    eval $(python backend/tools/auto_tune.py --export)

    # Run with specific test duration
    python backend/tools/auto_tune.py --quick    # ~15 seconds
    python backend/tools/auto_tune.py --full     # ~60 seconds
"""

import os
import sys
import time
import argparse
import psutil
import statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict
import math

# Minimal imports for speed


def get_system_info() -> Dict:
    """Quick system detection."""
    return {
        "cpu_threads": psutil.cpu_count(logical=True) or 4,
        "cpu_cores": psutil.cpu_count(logical=False) or 4,
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
    }


def cpu_work(n: int) -> float:
    """CPU-bound work simulation."""
    result = 0.0
    for i in range(n):
        result += math.sin(i) * math.cos(i)
    return result


def worker(args: Tuple[int, int]) -> float:
    """Worker function."""
    task_id, iterations = args
    start = time.perf_counter()
    cpu_work(iterations)
    return time.perf_counter() - start


def test_config(tasks: int, workers: int, work_units: int = 50, iterations: int = 20000) -> Dict:
    """Test a specific configuration quickly."""
    start = time.perf_counter()
    completed = 0

    work_per_task = work_units // tasks

    for t in range(tasks):
        items = [(i, iterations) for i in range(work_per_task)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(worker, items))
            completed += work_per_task

    duration = time.perf_counter() - start
    throughput = completed / duration if duration > 0 else 0

    return {
        "tasks": tasks,
        "workers": workers,
        "threads": tasks * workers,
        "duration": duration,
        "throughput": throughput,
    }


def find_optimal(system: Dict, mode: str = "quick") -> Dict:
    """Find optimal configuration through quick benchmarking."""
    cpu_threads = system["cpu_threads"]
    mem_gb = system["memory_available_gb"]

    # Adjust test intensity based on mode
    if mode == "quick":
        work_units, iterations = 30, 15000
    elif mode == "full":
        work_units, iterations = 100, 30000
    else:  # default
        work_units, iterations = 50, 20000

    # Generate smart test configs based on system size
    configs = []

    # Test various worker counts
    for workers in [2, 4, 6, 8, 12, 16]:
        if workers <= cpu_threads:
            # Calculate how many concurrent tasks make sense
            max_tasks = max(1, cpu_threads // workers)
            for tasks in [1, 2, max(1, max_tasks // 2), max_tasks]:
                if tasks >= 1 and (tasks, workers) not in [(c[0], c[1]) for c in configs]:
                    # Memory check: ~2GB per task
                    if tasks * 2 <= mem_gb:
                        configs.append((tasks, workers))

    # Remove duplicates and limit
    configs = list(set(configs))[:8]

    results = []
    for tasks, workers in configs:
        try:
            result = test_config(tasks, workers, work_units, iterations)
            results.append(result)
        except Exception:
            pass

    if not results:
        # Fallback to safe defaults
        return {"tasks": 1, "workers": 4, "throughput": 0}

    # Find best throughput
    best = max(results, key=lambda r: r["throughput"])
    return best


def calculate_settings(system: Dict, best: Dict) -> Dict:
    """Calculate final settings from benchmark results."""
    cpu_threads = system["cpu_threads"]
    mem_gb = system["memory_available_gb"]

    cores_per_task = best["workers"]

    # Calculate concurrent tasks based on resources
    cpu_limit = max(1, (cpu_threads - 2) // cores_per_task)
    mem_limit = max(1, int((mem_gb - 2) // 2.0))  # 2GB per task

    max_concurrent = min(cpu_limit, mem_limit)

    return {
        "CORES_PER_TASK": cores_per_task,
        "MEMORY_PER_TASK_GB": 2.0,
        "RESERVED_CORES": 2,
        "RESERVED_MEMORY_GB": 2.0,
        "calculated_concurrent": max_concurrent,
        "calculated_threads": max_concurrent * cores_per_task,
        "throughput": best["throughput"],
    }


def log_banner(msg: str, char: str = "="):
    """Print a banner message for visibility in logs."""
    line = char * 60
    print(line)
    print(f"  {msg}")
    print(line)


def main():
    parser = argparse.ArgumentParser(description="Auto-tune optimizer settings")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (~15s)")
    parser.add_argument("--full", action="store_true", help="Full benchmark (~60s)")
    parser.add_argument("--export", action="store_true", help="Export as shell variables")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    mode = "quick" if args.quick else ("full" if args.full else "default")

    # Detect system
    system = get_system_info()

    if not args.quiet and not args.export:
        log_banner("AUTO-TUNE: Detecting optimal settings")
        print(f"[Auto-Tune] System detected:")
        print(f"[Auto-Tune]   CPU threads: {system['cpu_threads']}")
        print(f"[Auto-Tune]   CPU cores:   {system['cpu_cores']}")
        print(f"[Auto-Tune]   RAM total:   {system['memory_total_gb']:.1f} GB")
        print(f"[Auto-Tune]   RAM available: {system['memory_available_gb']:.1f} GB")
        print(f"[Auto-Tune] Benchmark mode: {mode}")
        print()

    # Run benchmark
    if not args.quiet and not args.export:
        print("[Auto-Tune] Running benchmark...")

    start = time.perf_counter()
    best = find_optimal(system, mode)
    duration = time.perf_counter() - start

    # Calculate settings
    settings = calculate_settings(system, best)

    if args.export:
        # Shell export format for eval
        print(f"export CORES_PER_TASK={settings['CORES_PER_TASK']}")
        print(f"export MEMORY_PER_TASK_GB={settings['MEMORY_PER_TASK_GB']}")
        print(f"export RESERVED_CORES={settings['RESERVED_CORES']}")
        print(f"export RESERVED_MEMORY_GB={settings['RESERVED_MEMORY_GB']}")
    elif args.json:
        import json
        output = {
            "system": system,
            "settings": settings,
            "benchmark_duration": duration,
        }
        print(json.dumps(output, indent=2))
    else:
        print()
        log_banner("AUTO-TUNE: Results", "-")
        print(f"[Auto-Tune] Benchmark duration: {duration:.1f}s")
        print(f"[Auto-Tune] Best configuration found:")
        print(f"[Auto-Tune]   CORES_PER_TASK = {settings['CORES_PER_TASK']}")
        print(f"[Auto-Tune]   Concurrent tasks: {settings['calculated_concurrent']}")
        print(f"[Auto-Tune]   Workers per task: {settings['CORES_PER_TASK']}")
        print(f"[Auto-Tune]   Total threads: {settings['calculated_threads']}")
        print(f"[Auto-Tune]   Throughput: {settings['throughput']:.0f} combinations/sec")
        print()
        print(f"[Auto-Tune] Resource allocation:")
        print(f"[Auto-Tune]   CPU: {settings['calculated_threads']}/{system['cpu_threads']} threads used")
        print(f"[Auto-Tune]   RAM: ~{settings['calculated_concurrent'] * 2:.0f}GB reserved for optimization")
        log_banner("AUTO-TUNE: Complete", "-")

    # Always save to a known location for the app to read
    config_path = Path(__file__).parent.parent / ".auto_tuned"
    with open(config_path, "w") as f:
        f.write(f"CORES_PER_TASK={settings['CORES_PER_TASK']}\n")
        f.write(f"MEMORY_PER_TASK_GB={settings['MEMORY_PER_TASK_GB']}\n")
        f.write(f"# Auto-tuned: {system['cpu_threads']} threads, {system['memory_available_gb']:.1f}GB\n")
        f.write(f"# Throughput: {settings['throughput']:.0f} comb/sec\n")
        f.write(f"# Concurrent tasks: {settings['calculated_concurrent']}\n")
        f.write(f"# Total threads: {settings['calculated_threads']}\n")

    if not args.quiet and not args.export and not args.json:
        print(f"[Auto-Tune] Saved to: {config_path}")

    return settings


if __name__ == "__main__":
    main()
