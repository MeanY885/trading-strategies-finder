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


# Use case constants for TTL selection
USE_CASE_OPTIMIZATION = 'optimization'
USE_CASE_VALIDATION = 'validation'


@dataclass
class CacheEntry:
    """Single OHLCV cache entry."""
    df: pd.DataFrame
    created_at: float
    last_accessed: float = 0.0  # Track last access time for proper LRU
    hits: int = 0


class OHLCVCache:
    """
    Thread-safe in-memory cache for OHLCV DataFrames.

    Key format: {pair}_{interval}_{months}
    Example: BTCUSDT_15_3 (BTC/USDT, 15min candles, 3 months)

    Features:
    - Thread-safe with RLock
    - Configurable TTL based on use case (optimization vs validation)
    - Memory limit with LRU eviction
    - Hit statistics for monitoring
    - Symbol-specific cache invalidation
    """

    # Default TTL values for different use cases
    DEFAULT_TTL = 86400  # 24 hours for general/optimization use
    VALIDATION_TTL = 3600  # 1 hour for validation (need fresher data)

    def __init__(self, default_ttl: float = None, max_entries: int = 100):
        """
        Initialize OHLCV cache.

        Args:
            default_ttl: Time-to-live in seconds (default 24 hours).
                        For validation use cases, use get() with use_case='validation'
                        for stricter 1-hour TTL.
            max_entries: Maximum cache entries before LRU eviction
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl if default_ttl is not None else self.DEFAULT_TTL
        self._validation_ttl = self.VALIDATION_TTL
        self._max_entries = max_entries
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
        }

    def _make_key(self, pair: str, interval: int, months: float) -> str:
        """Generate cache key from parameters."""
        # Normalize pair (remove slashes, uppercase)
        pair_clean = pair.upper().replace("/", "").replace("-", "")
        return f"{pair_clean}_{interval}_{months}"

    def _get_ttl_for_use_case(self, use_case: str) -> float:
        """Get the appropriate TTL based on use case."""
        if use_case == USE_CASE_VALIDATION:
            return self._validation_ttl
        return self._default_ttl

    def get(
        self,
        pair: str,
        interval: int,
        months: float,
        use_case: str = USE_CASE_OPTIMIZATION,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get cached OHLCV data if available and not expired.

        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            interval: Candle interval in minutes
            months: Number of months of data
            use_case: Either 'optimization' (24h TTL) or 'validation' (1h TTL).
                     Validation uses stricter TTL to ensure recent market conditions.
            force_refresh: If True, ignore cache and return None to force a refresh

        Returns:
            DataFrame if cached and valid, None otherwise
        """
        key = self._make_key(pair, interval, months)

        # Force refresh bypasses cache entirely
        if force_refresh:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    log(f"[OHLCV Cache] Force refresh requested: {key}")
                self._stats["misses"] += 1
                return None

        ttl = self._get_ttl_for_use_case(use_case)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            # Check TTL based on use case
            age = time.time() - entry.created_at
            if age > ttl:
                del self._cache[key]
                self._stats["misses"] += 1
                ttl_name = "validation" if use_case == USE_CASE_VALIDATION else "default"
                log(f"[OHLCV Cache] Expired ({ttl_name} TTL): {key} (age: {age:.0f}s > {ttl}s)")
                return None

            entry.hits += 1
            entry.last_accessed = time.time()  # Update last access time for LRU
            self._stats["hits"] += 1
            log(f"[OHLCV Cache] HIT: {key} ({len(entry.df):,} rows, {entry.hits} hits, use_case={use_case})")

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

            now = time.time()
            self._cache[key] = CacheEntry(
                df=df.copy(),
                created_at=now,
                last_accessed=now,  # Initialize last_accessed to creation time
                hits=0
            )

            log(f"[OHLCV Cache] SET: {key} ({len(df):,} rows)")

    def _evict_lru(self) -> None:
        """Evict least recently used entry based on last_accessed time."""
        if not self._cache:
            return

        # Find entry with oldest last_accessed time (proper LRU)
        # The entry with the smallest last_accessed is the least recently used
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )

        del self._cache[oldest_key]
        self._stats["evictions"] += 1
        log(f"[OHLCV Cache] Evicted LRU: {oldest_key}")

    def has(
        self,
        pair: str,
        interval: int,
        months: float,
        use_case: str = USE_CASE_OPTIMIZATION
    ) -> bool:
        """Check if data exists in cache (without counting as hit)."""
        key = self._make_key(pair, interval, months)
        ttl = self._get_ttl_for_use_case(use_case)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if time.time() - entry.created_at > ttl:
                return False
            return True

    def invalidate(self, pair: str, interval: int = None, months: float = None) -> int:
        """
        Invalidate cache entries for a specific symbol.

        Args:
            pair: Trading pair to invalidate (e.g., 'BTCUSDT')
            interval: Optional - if provided, only invalidate entries with this interval
            months: Optional - if provided, only invalidate entries with this months value

        Returns:
            Number of entries invalidated
        """
        pair_clean = pair.upper().replace("/", "").replace("-", "")
        invalidated = 0

        with self._lock:
            keys_to_remove = []

            for key in self._cache.keys():
                parts = key.split("_")
                if len(parts) < 3:
                    continue

                key_pair = parts[0]
                key_interval = int(parts[1])
                key_months = float(parts[2])

                # Check if this entry matches the invalidation criteria
                if key_pair != pair_clean:
                    continue

                if interval is not None and key_interval != interval:
                    continue

                if months is not None and key_months != months:
                    continue

                keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]
                invalidated += 1
                log(f"[OHLCV Cache] Invalidated: {key}")

            self._stats["invalidations"] += invalidated

        if invalidated > 0:
            log(f"[OHLCV Cache] Invalidated {invalidated} entries for {pair_clean}")

        return invalidated

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
                "invalidations": self._stats["invalidations"],
                "ttl_seconds": self._default_ttl,
                "validation_ttl_seconds": self._validation_ttl,
            }

    def get_cached_keys(self) -> list:
        """Get list of currently cached keys."""
        with self._lock:
            return list(self._cache.keys())


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Global cache instance - 24 hour default TTL (1 hour for validation), up to 100 datasets
ohlcv_cache = OHLCVCache(default_ttl=OHLCVCache.DEFAULT_TTL, max_entries=100)


def get_ohlcv_cache() -> OHLCVCache:
    """Get the global OHLCV cache instance."""
    return ohlcv_cache


def clear_ohlcv_cache() -> int:
    """Clear the global OHLCV cache."""
    return ohlcv_cache.clear()


def get_ohlcv_cache_stats() -> Dict:
    """Get stats from the global cache."""
    return ohlcv_cache.get_stats()


def invalidate_ohlcv_cache(pair: str, interval: int = None, months: float = None) -> int:
    """
    Invalidate cache entries for a specific symbol.

    Args:
        pair: Trading pair to invalidate (e.g., 'BTCUSDT', 'BTC/USDT')
        interval: Optional - only invalidate entries with this interval
        months: Optional - only invalidate entries with this months value

    Returns:
        Number of entries invalidated

    Example:
        # Invalidate all BTCUSDT cache entries
        invalidate_ohlcv_cache('BTCUSDT')

        # Invalidate only 15-minute BTCUSDT entries
        invalidate_ohlcv_cache('BTCUSDT', interval=15)
    """
    return ohlcv_cache.invalidate(pair, interval, months)
