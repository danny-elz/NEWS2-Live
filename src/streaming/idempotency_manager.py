"""
Idempotency manager for exactly-once processing semantics in stream processing.
"""

import asyncio
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, Set, Dict, Any
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)


class IdempotencyManager:
    """Manages idempotency for stream processing to ensure exactly-once semantics."""
    
    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 3600):
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.processed_key_prefix = "processed_event:"
        self.sequence_key_prefix = "sequence:"
        self._local_cache: Dict[str, datetime] = {}
        self._cache_cleanup_interval = 300  # 5 minutes
        self._last_cleanup = datetime.now(timezone.utc)
    
    async def is_duplicate(self, event_id: str, patient_id: Optional[str] = None) -> bool:
        """
        Check if event has already been processed.
        
        Args:
            event_id: Unique identifier for the event
            patient_id: Optional patient ID for additional context
            
        Returns:
            True if event is a duplicate, False otherwise
        """
        try:
            # Create composite key for better uniqueness
            key = self._create_processed_key(event_id, patient_id)
            
            # Check local cache first for performance
            if key in self._local_cache:
                cache_time = self._local_cache[key]
                if datetime.now(timezone.utc) - cache_time < timedelta(seconds=self.ttl_seconds):
                    logger.debug(f"Found duplicate in local cache: {event_id}")
                    return True
                else:
                    # Remove expired entry from local cache
                    del self._local_cache[key]
            
            # Check Redis
            exists = await self.redis.exists(key)
            if exists:
                # Update local cache
                self._local_cache[key] = datetime.now(timezone.utc)
                logger.debug(f"Found duplicate in Redis: {event_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate for event {event_id}: {e}")
            # In case of Redis failure, allow processing to continue
            # but log the issue for monitoring
            return False
    
    async def mark_processed(self, event_id: str, patient_id: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark event as processed with TTL.
        
        Args:
            event_id: Unique identifier for the event
            patient_id: Optional patient ID for additional context
            metadata: Optional metadata to store with the processed marker
        """
        try:
            key = self._create_processed_key(event_id, patient_id)
            
            # Prepare value with metadata
            value = {
                'event_id': event_id,
                'patient_id': patient_id,
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            # Store in Redis with TTL
            await self.redis.setex(key, self.ttl_seconds, str(value))
            
            # Update local cache
            self._local_cache[key] = datetime.now(timezone.utc)
            
            logger.debug(f"Marked event as processed: {event_id}")
            
            # Periodic cleanup of local cache
            await self._cleanup_local_cache()
            
        except Exception as e:
            logger.error(f"Error marking event {event_id} as processed: {e}")
            raise
    
    async def is_event_in_sequence(self, event_id: str, patient_id: str, 
                                 event_timestamp: datetime, sequence_window_ms: int = 5000) -> bool:
        """
        Check if event is within acceptable sequence window for ordering.
        
        Args:
            event_id: Unique identifier for the event
            patient_id: Patient ID for sequence tracking
            event_timestamp: Timestamp of the current event
            sequence_window_ms: Maximum allowed time difference in milliseconds
            
        Returns:
            True if event is in acceptable sequence, False if out of order
        """
        try:
            sequence_key = f"{self.sequence_key_prefix}{patient_id}"
            
            # Get last processed timestamp for this patient
            last_timestamp_str = await self.redis.get(sequence_key)
            
            if not last_timestamp_str:
                # First event for this patient
                await self._update_sequence_timestamp(sequence_key, event_timestamp)
                return True
            
            last_timestamp = datetime.fromisoformat(last_timestamp_str.decode())
            time_diff = (event_timestamp - last_timestamp).total_seconds() * 1000
            
            if time_diff >= -sequence_window_ms:
                # Event is within acceptable window (including slightly out of order)
                if event_timestamp > last_timestamp:
                    await self._update_sequence_timestamp(sequence_key, event_timestamp)
                return True
            else:
                # Event is too far out of order
                logger.warning(f"Out of sequence event {event_id} for patient {patient_id}. "
                             f"Time diff: {time_diff}ms")
                return False
                
        except Exception as e:
            logger.error(f"Error checking event sequence for {event_id}: {e}")
            # Allow processing to continue in case of Redis issues
            return True
    
    async def get_processed_count(self, time_window_seconds: int = 3600) -> int:
        """Get count of processed events in the specified time window."""
        try:
            # This is an approximation based on keys with our prefix
            # In production, you might want to use a more sophisticated approach
            pattern = f"{self.processed_key_prefix}*"
            keys = await self.redis.keys(pattern)
            return len(keys) if keys else 0
            
        except Exception as e:
            logger.error(f"Error getting processed count: {e}")
            return -1
    
    async def clear_processed_events(self, pattern: Optional[str] = None) -> int:
        """
        Clear processed event markers (use with caution).
        
        Args:
            pattern: Optional pattern to match specific keys
            
        Returns:
            Number of keys deleted
        """
        try:
            search_pattern = pattern or f"{self.processed_key_prefix}*"
            keys = await self.redis.keys(search_pattern)
            
            if not keys:
                return 0
            
            deleted = await self.redis.delete(*keys)
            self._local_cache.clear()  # Clear local cache too
            
            logger.info(f"Cleared {deleted} processed event markers")
            return deleted
            
        except Exception as e:
            logger.error(f"Error clearing processed events: {e}")
            raise
    
    def _create_processed_key(self, event_id: str, patient_id: Optional[str] = None) -> str:
        """Create a composite key for processed event tracking."""
        if patient_id:
            composite = f"{event_id}:{patient_id}"
        else:
            composite = event_id
        
        # Hash the composite to ensure consistent key length
        key_hash = hashlib.sha256(composite.encode()).hexdigest()
        return f"{self.processed_key_prefix}{key_hash}"
    
    async def _update_sequence_timestamp(self, sequence_key: str, timestamp: datetime) -> None:
        """Update the last processed timestamp for sequence tracking."""
        await self.redis.setex(sequence_key, self.ttl_seconds, timestamp.isoformat())
    
    async def _cleanup_local_cache(self) -> None:
        """Clean up expired entries from local cache."""
        now = datetime.now(timezone.utc)
        
        if (now - self._last_cleanup).seconds < self._cache_cleanup_interval:
            return
        
        expired_keys = []
        cutoff_time = now - timedelta(seconds=self.ttl_seconds)
        
        for key, timestamp in self._local_cache.items():
            if timestamp < cutoff_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._local_cache[key]
        
        self._last_cleanup = now
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries from local cache")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the idempotency manager."""
        try:
            # Test Redis connectivity
            await self.redis.ping()
            
            # Get some basic stats
            processed_count = await self.get_processed_count()
            
            return {
                'status': 'healthy',
                'redis_connected': True,
                'local_cache_size': len(self._local_cache),
                'processed_events_count': processed_count,
                'ttl_seconds': self.ttl_seconds
            }
            
        except Exception as e:
            logger.error(f"Idempotency manager health check failed: {e}")
            return {
                'status': 'unhealthy',
                'redis_connected': False,
                'error': str(e)
            }
    
    async def close(self) -> None:
        """Close Redis connection and clean up resources."""
        try:
            await self.redis.close()
            self._local_cache.clear()
            logger.info("Idempotency manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing idempotency manager: {e}")