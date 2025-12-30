"""
Pytest Configuration and Shared Fixtures
=========================================
Common fixtures for testing the BTC/GBP ML Optimizer.
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for backtesting."""
    np.random.seed(42)
    n_bars = 1000

    # Generate realistic price data with trend
    initial_price = 30000
    returns = np.random.normal(0.0001, 0.02, n_bars)  # Small drift, 2% daily vol
    prices = initial_price * np.cumprod(1 + returns)

    # Create OHLCV DataFrame
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='1h')

    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_bars)),
        'high': prices * (1 + np.random.uniform(0.001, 0.02, n_bars)),
        'low': prices * (1 - np.random.uniform(0.001, 0.02, n_bars)),
        'close': prices,
        'volume': np.random.uniform(100, 10000, n_bars)
    }, index=dates)

    # Ensure high >= open, close, low and low <= open, close, high
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def small_ohlcv_data():
    """Small dataset for quick tests."""
    np.random.seed(42)
    n_bars = 100

    initial_price = 30000
    returns = np.random.normal(0.0001, 0.02, n_bars)
    prices = initial_price * np.cumprod(1 + returns)

    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='1h')

    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_bars)),
        'high': prices * (1 + np.random.uniform(0.001, 0.02, n_bars)),
        'low': prices * (1 - np.random.uniform(0.001, 0.02, n_bars)),
        'close': prices,
        'volume': np.random.uniform(100, 10000, n_bars)
    }, index=dates)

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def mock_app_state():
    """Mock application state."""
    mock_state = MagicMock()
    mock_state.is_autonomous_running.return_value = True
    mock_state.is_autonomous_enabled.return_value = True
    mock_state.is_unified_running.return_value = False
    mock_state.get_autonomous_status.return_value = {
        "running": True,
        "enabled": True,
        "paused": False,
        "cycle_index": 0,
        "message": "Testing"
    }
    mock_state.update_autonomous_status = MagicMock()
    mock_state.add_running_optimization = MagicMock()
    mock_state.remove_running_optimization = MagicMock()
    return mock_state


@pytest.fixture
def mock_resource_monitor():
    """Mock resource monitor with healthy resources."""
    mock_monitor = MagicMock()
    mock_monitor.get_current_resources.return_value = {
        "cpu_cores": 8,
        "cpu_percent": 30.0,
        "cpu_per_core": [30.0] * 8,
        "memory_total_gb": 16.0,
        "memory_available_gb": 8.0,
        "memory_used_percent": 50.0,
        "memory_free_gb": 8.0
    }
    return mock_monitor


@pytest.fixture
def mock_ws_manager():
    """Mock WebSocket manager."""
    mock_ws = MagicMock()
    mock_ws.broadcast = AsyncMock()
    return mock_ws


@pytest.fixture
def sample_strategy_result():
    """Sample strategy result for testing."""
    return {
        "strategy": "rsi_oversold",
        "direction": "long",
        "tp_percent": 2.0,
        "sl_percent": 1.0,
        "total_trades": 50,
        "wins": 30,
        "losses": 20,
        "win_rate": 60.0,
        "total_pnl": 150.0,
        "total_return": 15.0,
        "max_drawdown": 5.0,
        "sharpe_ratio": 1.5,
        "composite_score": 75.0
    }


@pytest.fixture
def sample_combinations():
    """Sample optimization combinations for testing."""
    return [
        {
            "pair": "BTCUSDT",
            "period": "3M",
            "timeframe": "1h",
            "granularity": "low"
        },
        {
            "pair": "ETHUSDT",
            "period": "1M",
            "timeframe": "15m",
            "granularity": "medium"
        },
        {
            "pair": "ADAUSDT",
            "period": "6M",
            "timeframe": "4h",
            "granularity": "high"
        }
    ]


# Memory testing helpers
@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""
    import tracemalloc

    class MemoryTracker:
        def __init__(self):
            self.snapshots = []

        def start(self):
            tracemalloc.start()
            self.snapshots = []

        def snapshot(self, label=""):
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append((label, snapshot))
            return snapshot

        def stop(self):
            tracemalloc.stop()

        def get_peak(self):
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                return peak / 1024 / 1024  # MB
            return 0

        def compare(self, idx1=0, idx2=-1):
            if len(self.snapshots) < 2:
                return []
            _, snap1 = self.snapshots[idx1]
            _, snap2 = self.snapshots[idx2]
            return snap2.compare_to(snap1, 'lineno')

    return MemoryTracker()


# Async test helpers
@pytest.fixture
def run_async():
    """Helper to run async functions in sync tests."""
    def _run_async(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    return _run_async
