import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import hashlib
import logging
from collections import OrderedDict

from ..models.patient import Patient


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Patient
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 300  # 5 minutes default
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now(timezone.utc) > (self.created_at + timedelta(seconds=self.ttl_seconds))
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self):
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0


class PatientDataCache:
    """
    High-performance LRU cache for patient data with TTL support.
    
    Optimized for frequently accessed patient information in clinical environments
    where the same patients are monitored repeatedly throughout shifts.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    async def get(self, patient_id: str) -> Optional[Patient]:
        """
        Get patient data from cache.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Patient object if found and not expired, None otherwise
        """
        async with self._lock:
            if patient_id not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[patient_id]
            
            # Check if expired
            if entry.is_expired():
                del self.cache[patient_id]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(patient_id)
            entry.touch()
            self.stats.hits += 1
            
            return entry.data
    
    async def put(self, patient: Patient, ttl: Optional[int] = None) -> None:
        """
        Store patient data in cache.
        
        Args:
            patient: Patient object to cache
            ttl: Time to live in seconds (uses default if not specified)
        """
        async with self._lock:
            patient_id = patient.patient_id
            ttl = ttl or self.default_ttl
            
            # Update existing entry
            if patient_id in self.cache:
                entry = self.cache[patient_id]
                entry.data = patient
                entry.created_at = datetime.now(timezone.utc)
                entry.ttl_seconds = ttl
                self.cache.move_to_end(patient_id)
                return
            
            # Evict least recently used if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats.evictions += 1
            
            # Add new entry
            entry = CacheEntry(
                data=patient,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                ttl_seconds=ttl
            )
            
            self.cache[patient_id] = entry
            self.stats.size = len(self.cache)
    
    async def get_multiple(self, patient_ids: List[str]) -> Dict[str, Optional[Patient]]:
        """
        Get multiple patients from cache efficiently.
        
        Args:
            patient_ids: List of patient identifiers
            
        Returns:
            Dictionary mapping patient_id to Patient (or None if not found)
        """
        results = {}
        
        for patient_id in patient_ids:
            results[patient_id] = await self.get(patient_id)
        
        return results
    
    async def invalidate(self, patient_id: str) -> bool:
        """
        Remove patient from cache.
        
        Args:
            patient_id: Patient identifier to remove
            
        Returns:
            True if patient was in cache, False otherwise
        """
        async with self._lock:
            if patient_id in self.cache:
                del self.cache[patient_id]
                self.stats.size = len(self.cache)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cached data."""
        async with self._lock:
            cleared_count = len(self.cache)
            self.cache.clear()
            self.stats.size = 0
            self.stats.evictions += cleared_count
            self.logger.info(f"Cache cleared: {cleared_count} entries removed")
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = []
            
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            self.stats.size = len(self.cache)
            self.stats.evictions += len(expired_keys)
            
            if expired_keys:
                self.logger.info(f"Cache cleanup: {len(expired_keys)} expired entries removed")
            
            return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        self.stats.size = len(self.cache)
        
        # Estimate memory usage
        if self.cache:
            # Rough estimation based on typical Patient object size
            avg_patient_size_bytes = 500  # Approximate bytes per Patient object
            self.stats.memory_usage_mb = (len(self.cache) * avg_patient_size_bytes) / (1024 * 1024)
        else:
            self.stats.memory_usage_mb = 0.0
        
        return self.stats
    
    def get_cache_info(self) -> Dict:
        """Get detailed cache information."""
        stats = self.get_stats()
        
        return {
            'size': stats.size,
            'max_size': self.max_size,
            'hit_rate': stats.hit_rate,
            'hits': stats.hits,
            'misses': stats.misses,
            'evictions': stats.evictions,
            'memory_usage_mb': stats.memory_usage_mb,
            'fill_ratio': stats.size / self.max_size if self.max_size > 0 else 0.0
        }


class ConnectionPool:
    """
    Simple connection pool simulation for database connections.
    In a real implementation, this would manage actual database connections.
    """
    
    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self.available_connections = list(range(pool_size))
        self.busy_connections = set()
        self._lock = asyncio.Lock()
        self.stats = {
            'total_requests': 0,
            'peak_usage': 0,
            'wait_time_ms': 0.0
        }
    
    async def acquire(self, timeout: float = 5.0) -> int:
        """
        Acquire a connection from the pool.
        
        Args:
            timeout: Maximum time to wait for connection
            
        Returns:
            Connection ID
            
        Raises:
            asyncio.TimeoutError: If no connection available within timeout
        """
        start_time = time.perf_counter()
        
        try:
            async with asyncio.timeout(timeout):
                while True:
                    async with self._lock:
                        if self.available_connections:
                            conn_id = self.available_connections.pop()
                            self.busy_connections.add(conn_id)
                            
                            # Update stats
                            self.stats['total_requests'] += 1
                            self.stats['peak_usage'] = max(
                                self.stats['peak_usage'], 
                                len(self.busy_connections)
                            )
                            
                            wait_time = (time.perf_counter() - start_time) * 1000
                            self.stats['wait_time_ms'] += wait_time
                            
                            return conn_id
                    
                    # Wait briefly before retrying
                    await asyncio.sleep(0.001)
                    
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Could not acquire connection within {timeout}s")
    
    async def release(self, conn_id: int) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            conn_id: Connection ID to release
        """
        async with self._lock:
            if conn_id in self.busy_connections:
                self.busy_connections.remove(conn_id)
                self.available_connections.append(conn_id)
    
    def get_stats(self) -> Dict:
        """Get connection pool statistics."""
        avg_wait_time = (
            self.stats['wait_time_ms'] / self.stats['total_requests'] 
            if self.stats['total_requests'] > 0 else 0.0
        )
        
        return {
            'pool_size': self.pool_size,
            'available': len(self.available_connections),
            'busy': len(self.busy_connections),
            'utilization': len(self.busy_connections) / self.pool_size,
            'total_requests': self.stats['total_requests'],
            'peak_usage': self.stats['peak_usage'],
            'average_wait_time_ms': avg_wait_time
        }


class PerformanceMonitor:
    """
    Performance monitoring utility for NEWS2 calculations.
    Tracks timing, memory usage, and throughput metrics.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.calculation_times = []
        self.memory_samples = []
        self.start_time = time.perf_counter()
        self.total_calculations = 0
        self._lock = asyncio.Lock()
    
    async def record_calculation(self, calculation_time_ms: float, memory_mb: float = 0.0):
        """Record a calculation performance sample."""
        async with self._lock:
            self.calculation_times.append(calculation_time_ms)
            if memory_mb > 0:
                self.memory_samples.append(memory_mb)
            
            # Maintain sliding window
            if len(self.calculation_times) > self.window_size:
                self.calculation_times.pop(0)
            
            if len(self.memory_samples) > self.window_size:
                self.memory_samples.pop(0)
            
            self.total_calculations += 1
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if not self.calculation_times:
            return {
                'total_calculations': 0,
                'avg_calculation_time_ms': 0.0,
                'p95_calculation_time_ms': 0.0,
                'p99_calculation_time_ms': 0.0,
                'max_calculation_time_ms': 0.0,
                'throughput_per_second': 0.0,
                'avg_memory_mb': 0.0,
                'peak_memory_mb': 0.0
            }
        
        times = sorted(self.calculation_times)
        avg_time = sum(times) / len(times)
        p95_time = times[int(len(times) * 0.95)] if len(times) > 20 else max(times)
        p99_time = times[int(len(times) * 0.99)] if len(times) > 100 else max(times)
        max_time = max(times)
        
        # Calculate throughput
        elapsed_time = time.perf_counter() - self.start_time
        throughput = self.total_calculations / elapsed_time if elapsed_time > 0 else 0.0
        
        # Memory stats
        avg_memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0.0
        peak_memory = max(self.memory_samples) if self.memory_samples else 0.0
        
        return {
            'total_calculations': self.total_calculations,
            'avg_calculation_time_ms': avg_time,
            'p95_calculation_time_ms': p95_time,
            'p99_calculation_time_ms': p99_time,
            'max_calculation_time_ms': max_time,
            'throughput_per_second': throughput,
            'avg_memory_mb': avg_memory,
            'peak_memory_mb': peak_memory
        }