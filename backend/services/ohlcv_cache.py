"""
OHLCV Data Cache
================
In-memory cache for OHLCV data to avoid redundant Binance API calls.
Enables true parallel optimization by allowing multiple workers to
access the same data simultaneously without refetching.
"""
import threading
import time
import hashlib
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from logging_config import log


@dataclass
class CacheEntry:
    """Single OHLCV cache entry."""
    df: pd.DataFrame
    created_at: float
    hits: int = 0


class OHLCVCache:
    """
    Thread-safe in-memory cache for OHLCV DataFrames.

    Key format: {pair}_{interval}_{months}
    Example: BTCUSDT_15_3 (BTC/USDT, 15min candles, 3 months)

    Features:
    - Thread-safe with RLock
    - TTL-based expiration (default 7 days)
    - Memory limit with LRU eviction
    - Hit statistics for monitoring
    """

    def __init__(self, default_ttl: float = 3600.0, max_entries: int = 100):
        """
        Initialize OHLCV cache.

        Args:
            default_ttl: Time-to-live in seconds (default 7 days)
            max_entries: Maximum cache entries before LRU eviction
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def _make_key(self, pair: str, interval: int, months: float) -> str:
        """Generate cache key from parameters."""
        # Normalize pair (remove slashes, uppercase)
        pair_clean = pair.upper().replace("/", "").replace("-", "")
        return f"{pair_clean}_{interval}_{months}"

    def get(self, pair: str, interval: int, months: float) -> Optional[pd.DataFrame]:
        """
        Get cached OHLCV data if available and not expired.

        Returns:
            DataFrame if cached and valid, None otherwise
        """
        key = self._make_key(pair, interval, months)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            # Check TTL
            if time.time() - entry.created_at > self._default_ttl:
                del self._cache[key]
                self._stats["misses"] += 1
                log(f"[OHLCV Cache] Expired: {key}")
                return None

            entry.hits += 1
            self._stats["hits"] += 1
            log(f"[OHLCV Cache] HIT: {key} ({len(entry.df):,} rows, {entry.hits} hits)")

            # Return a copy to prevent mutation
            return entry.df.copy()

    def set(self, pair: str, interval: int, months: float, df: pd.DataFrame) -> None:
        """
        Store OHLCV data in cache.

        Performs LRU eviction if cache is full.
        """
        if df is None or len(df) == 0:
            return

        key = self._make_key(pair, interval, months)

        with self._lock:
            # LRU eviction if at capacity
            if len(self._cache) >= self._max_entries and key not in self._cache:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                df=df.copy(),
                created_at=time.time(),
                hits=0
            )

            log(f"[OHLCV Cache] SET: {key} ({len(df):,} rows)")

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find entry with oldest access (created_at + hits as proxy)
        # Lower score = older/less used
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at + (self._cache[k].hits * 60)
        )

        del self._cache[oldest_key]
        self._stats["evictions"] += 1
        log(f"[OHLCV Cache] Evicted LRU: {oldest_key}")

    def has(self, pair: str, interval: int, months: float) -> bool:
        """Check if data exists in cache (without counting as hit)."""
        key = self._make_key(pair, interval, months)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if time.time() - entry.created_at > self._default_ttl:
                return False
            return True

    def clear(self) -> int:
        """Clear all cache entries. Returns count cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            log(f"[OHLCV Cache] Cleared {count} entries")
            return count

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            total_rows = sum(e.df.shape[0] for e in self._cache.values())
            memory_mb = sum(e.df.memory_usage(deep=True).sum() for e in self._cache.values()) / (1024 * 1024)

            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "total_rows": total_rows,
                "memory_mb": round(memory_mb, 2),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate_pct": round(hit_rate, 1),
                "evictions": self._stats["evictions"],
                "ttl_seconds": self._default_ttl,
            }

    def get_cached_keys(self) -> list:
        """Get list of currently cached keys."""
        with self._lock:
            return list(self._cache.keys())


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Global cache instance - 7 day TTL, up to 100 datasets
ohlcv_cache = OHLCVCache(default_ttl=604800.0, max_entries=100)


def get_ohlcv_cache() -> OHLCVCache:
    """Get the global OHLCV cache instance."""
    return ohlcv_cache


def clear_ohlcv_cache() -> int:
    """Clear the global OHLCV cache."""
    return ohlcv_cache.clear()


def get_ohlcv_cache_stats() -> Dict:
    """Get stats from the global cache."""
    return ohlcv_cache.get_stats()
