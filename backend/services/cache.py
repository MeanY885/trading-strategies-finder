"""
Cache Service
=============
Simple TTL cache for database queries to eliminate repeated full table scans.
Dramatically improves UI responsiveness, especially in Docker environments.
"""
import time
import threading
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """Single cache entry with value and expiration time."""
    value: Any
    expires_at: float


class TTLCache:
    """
    Thread-safe TTL (Time-To-Live) cache.

    Usage:
        cache = TTLCache(default_ttl=60)  # 60 second default TTL
        cache.set("key", value, ttl=30)   # Override TTL per key
        value = cache.get("key")          # Returns None if expired
    """

    def __init__(self, default_ttl: float = 60.0):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() > entry.expires_at:
                del self._cache[key]
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value with TTL (uses default if not specified)."""
        with self._lock:
            expires_at = time.time() + (ttl if ttl is not None else self._default_ttl)
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

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
                "active_entries": total - expired
            }


# =============================================================================
# GLOBAL CACHE INSTANCES
# =============================================================================

# Strategy data cache (longer TTL - data doesn't change frequently)
strategies_cache = TTLCache(default_ttl=300)  # 5 minutes

# Counts cache (shorter TTL - want relatively fresh counts)
counts_cache = TTLCache(default_ttl=60)  # 1 minute

# Stats cache (short TTL)
stats_cache = TTLCache(default_ttl=60)  # 1 minute

# Priority lists cache (medium TTL - rarely changes)
priority_cache = TTLCache(default_ttl=300)  # 5 minutes


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
