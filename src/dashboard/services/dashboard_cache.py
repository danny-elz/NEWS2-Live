"""
Dashboard Cache Service for Story 3.1
High-performance caching layer for dashboard data
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class DashboardCache:
    """
    High-performance cache for dashboard data
    Implements LRU eviction and TTL-based expiration
    """

    def __init__(self, max_size: int = 100, default_ttl: int = 30):
        """
        Initialize cache

        Args:
            max_size: Maximum number of cached entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Tuple[Any, datetime, int]] = OrderedDict()
        self._hit_count = 0
        self._miss_count = 0
        self._lock = asyncio.Lock()

    def _generate_key(self, **kwargs) -> str:
        """Generate cache key from parameters"""
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        key_string = json.dumps(sorted_params, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value if valid, None otherwise
        """
        async with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None

            value, timestamp, ttl = self._cache[key]

            # Check if expired
            age = (datetime.utcnow() - timestamp).total_seconds()
            if age > ttl:
                del self._cache[key]
                self._miss_count += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hit_count += 1
            return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        async with self._lock:
            # Use provided TTL or default
            ttl = ttl or self.default_ttl

            # Add to cache
            self._cache[key] = (value, datetime.utcnow(), ttl)
            self._cache.move_to_end(key)

            # Evict oldest if over size limit
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    async def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries

        Args:
            pattern: Optional pattern to match keys (prefix match)
        """
        async with self._lock:
            if pattern is None:
                # Clear entire cache
                self._cache.clear()
            else:
                # Remove matching keys
                keys_to_remove = [
                    k for k in self._cache.keys()
                    if k.startswith(pattern)
                ]
                for key in keys_to_remove:
                    del self._cache[key]

    async def get_or_compute(self, key: str, compute_func, ttl: Optional[int] = None) -> Any:
        """
        Get from cache or compute if missing

        Args:
            key: Cache key
            compute_func: Async function to compute value if cache miss
            ttl: Time-to-live in seconds

        Returns:
            Cached or computed value
        """
        # Try to get from cache
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        value = await compute_func()

        # Store in cache
        await self.set(key, value, ttl)

        return value

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 3),
            "total_requests": total_requests
        }

    async def cleanup_expired(self):
        """Remove expired entries from cache"""
        async with self._lock:
            current_time = datetime.utcnow()
            keys_to_remove = []

            for key, (_, timestamp, ttl) in self._cache.items():
                age = (current_time - timestamp).total_seconds()
                if age > ttl:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]

            return len(keys_to_remove)


class DashboardCacheManager:
    """
    Manager for multiple cache instances
    Provides specialized caches for different dashboard components
    """

    def __init__(self):
        # Create specialized caches
        self.ward_cache = DashboardCache(max_size=50, default_ttl=30)
        self.patient_cache = DashboardCache(max_size=200, default_ttl=15)
        self.statistics_cache = DashboardCache(max_size=20, default_ttl=60)
        self.filter_cache = DashboardCache(max_size=100, default_ttl=20)

        # Background cleanup task
        self._cleanup_task = None

    async def start(self):
        """Start background cleanup task"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                # Cleanup all caches
                ward_cleaned = await self.ward_cache.cleanup_expired()
                patient_cleaned = await self.patient_cache.cleanup_expired()
                stats_cleaned = await self.statistics_cache.cleanup_expired()
                filter_cleaned = await self.filter_cache.cleanup_expired()

                total_cleaned = ward_cleaned + patient_cleaned + stats_cleaned + filter_cleaned
                if total_cleaned > 0:
                    logger.info(f"Cleaned {total_cleaned} expired cache entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")

    async def cache_ward_overview(self, ward_id: str, filters: Dict[str, Any], data: Any, ttl: int = 30):
        """Cache ward overview data"""
        key = self.ward_cache._generate_key(ward_id=ward_id, **filters)
        await self.ward_cache.set(key, data, ttl)

    async def get_ward_overview(self, ward_id: str, filters: Dict[str, Any]) -> Optional[Any]:
        """Get cached ward overview data"""
        key = self.ward_cache._generate_key(ward_id=ward_id, **filters)
        return await self.ward_cache.get(key)

    async def cache_patient_tile(self, patient_id: str, data: Any, ttl: int = 15):
        """Cache individual patient tile data"""
        await self.patient_cache.set(patient_id, data, ttl)

    async def get_patient_tile(self, patient_id: str) -> Optional[Any]:
        """Get cached patient tile data"""
        return await self.patient_cache.get(patient_id)

    async def cache_statistics(self, ward_id: str, stats: Any, ttl: int = 60):
        """Cache ward statistics"""
        key = f"stats_{ward_id}"
        await self.statistics_cache.set(key, stats, ttl)

    async def get_statistics(self, ward_id: str) -> Optional[Any]:
        """Get cached ward statistics"""
        key = f"stats_{ward_id}"
        return await self.statistics_cache.get(key)

    async def invalidate_ward(self, ward_id: str):
        """Invalidate all caches for a specific ward"""
        await self.ward_cache.invalidate(ward_id)
        await self.statistics_cache.invalidate(f"stats_{ward_id}")

    async def invalidate_patient(self, patient_id: str):
        """Invalidate cache for a specific patient"""
        await self.patient_cache.invalidate(patient_id)

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        return {
            "ward_cache": self.ward_cache.get_statistics(),
            "patient_cache": self.patient_cache.get_statistics(),
            "statistics_cache": self.statistics_cache.get_statistics(),
            "filter_cache": self.filter_cache.get_statistics(),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global cache manager instance
_cache_manager: Optional[DashboardCacheManager] = None


def get_cache_manager() -> DashboardCacheManager:
    """Get or create cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = DashboardCacheManager()
    return _cache_manager