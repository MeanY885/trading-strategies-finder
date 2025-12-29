"""
Cache Service
=============
Simple TTL cache for database queries to eliminate repeated full table scans.
Dramatically improves UI responsiveness, especially in Docker environments.
"""
import time
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """Single cache entry with value and expiration time."""
    value: Any
    expires_at: float


class TTLCache:
    """
    Thread-safe TTL (Time-To-Live) cache with LRU eviction.

    Usage:
        cache = TTLCache(default_ttl=60, maxsize=1000)  # 60s TTL, max 1000 entries
        cache.set("key", value, ttl=30)   # Override TTL per key
        value = cache.get("key")          # Returns None if expired

    When maxsize is reached, the least recently used entry is evicted.
    """

    def __init__(self, default_ttl: float = 60.0, maxsize: int = 1000):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired. Moves key to end for LRU tracking."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() > entry.expires_at:
                del self._cache[key]
                return None
            # Move to end to mark as most recently used
            self._cache.move_to_end(key)
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value with TTL (uses default if not specified). Evicts LRU entries if at capacity."""
        with self._lock:
            expires_at = time.time() + (ttl if ttl is not None else self._default_ttl)

            # If key exists, update it and move to end
            if key in self._cache:
                self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
                self._cache.move_to_end(key)
                return

            # Evict oldest entries if at capacity
            while len(self._cache) >= self._maxsize:
                # First try to evict expired entries
                self._evict_expired()
                # If still at capacity, evict the oldest (LRU) entry
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    def _evict_expired(self) -> int:
        """Remove expired entries. Returns count evicted. Must be called with lock held."""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now > v.expires_at]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if key existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all entries. Returns count cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def clear_prefix(self, prefix: str) -> int:
        """Clear all entries with matching prefix. Returns count cleared."""
        with self._lock:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: Optional[float] = None) -> Any:
        """
        Get value if cached, otherwise call factory and cache result.
        This is the primary method for cache-through patterns.
        """
        value = self.get(key)
        if value is not None:
            return value

        # Call factory outside lock to avoid blocking
        value = factory()
        self.set(key, value, ttl)
        return value

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            now = time.time()
            total = len(self._cache)
            expired = sum(1 for e in self._cache.values() if now > e.expires_at)
            return {
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
                "maxsize": self._maxsize
            }


# =============================================================================
# GLOBAL CACHE INSTANCES
# =============================================================================

# Strategy data cache (longer TTL - data doesn't change frequently)
# maxsize=500: Strategies can be large objects, limit to prevent memory issues
strategies_cache = TTLCache(default_ttl=300, maxsize=500)  # 5 minutes, max 500 entries

# Counts cache (shorter TTL - want relatively fresh counts)
# maxsize=100: Count queries are small, but we don't need many cached
counts_cache = TTLCache(default_ttl=60, maxsize=100)  # 1 minute, max 100 entries

# Stats cache (short TTL)
# maxsize=50: Stats are typically few distinct queries
stats_cache = TTLCache(default_ttl=60, maxsize=50)  # 1 minute, max 50 entries

# Priority lists cache (medium TTL - rarely changes)
# maxsize=50: Priority lists are few but can be moderately sized
priority_cache = TTLCache(default_ttl=300, maxsize=50)  # 5 minutes, max 50 entries


# =============================================================================
# CACHE KEYS
# =============================================================================

class CacheKeys:
    """Standardized cache key constants."""
    STRATEGIES_ALL = "strategies:all"
    STRATEGIES_PAGINATED = "strategies:page:{limit}:{offset}"
    ELITE_COUNTS = "counts:elite"
    ELITE_STRATEGIES = "elite:strategies"
    DB_STATS = "stats:db"
    PRIORITY_LISTS = "priority:all"
    FILTER_OPTIONS = "strategies:filters"


# =============================================================================
# CACHE INVALIDATION
# =============================================================================

def invalidate_strategy_caches():
    """
    Call when strategies are added, updated, or deleted.
    Clears all caches that depend on strategy data.
    """
    strategies_cache.clear()
    counts_cache.clear()
    stats_cache.clear()


def invalidate_counts_cache():
    """Call when elite status is updated. Also clears elite strategies cache."""
    counts_cache.clear()
    # Also invalidate elite strategies since validation status affects the elite table
    strategies_cache.delete(CacheKeys.ELITE_STRATEGIES)


def invalidate_priority_cache():
    """Call when priority lists are modified."""
    priority_cache.clear()


def invalidate_all_caches():
    """Nuclear option - clear everything."""
    strategies_cache.clear()
    counts_cache.clear()
    stats_cache.clear()
    priority_cache.clear()


# =============================================================================
# CACHE STATS FOR MONITORING
# =============================================================================

def get_all_cache_stats() -> Dict[str, Dict[str, int]]:
    """Get stats for all cache instances."""
    return {
        "strategies": strategies_cache.stats(),
        "counts": counts_cache.stats(),
        "stats": stats_cache.stats(),
        "priority": priority_cache.stats()
    }
