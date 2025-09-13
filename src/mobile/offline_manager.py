"""
Offline Manager for Story 3.4
Handles offline functionality and data synchronization for mobile devices
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Synchronization status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


class CacheStrategy(Enum):
    """Data caching strategies"""
    AGGRESSIVE = "aggressive"  # Cache everything
    SELECTIVE = "selective"   # Cache critical data only
    MINIMAL = "minimal"       # Cache essential data only


class NetworkState(Enum):
    """Network connectivity states"""
    ONLINE = "online"
    OFFLINE = "offline"
    LIMITED = "limited"  # Slow or unreliable connection


@dataclass
class OfflineAction:
    """Offline action to be synchronized when online"""
    action_id: str
    action_type: str
    patient_id: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=critical, 2=high, 3=normal, 4=low
    retry_count: int = 0
    max_retries: int = 3
    status: SyncStatus = SyncStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "patient_id": self.patient_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value
        }


@dataclass
class CacheItem:
    """Cached data item"""
    key: str
    data: Any
    timestamp: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if cache item has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_fresh(self, max_age_seconds: int = 300) -> bool:
        """Check if cache item is fresh (within max age)"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age <= max_age_seconds


@dataclass
class SyncResult:
    """Result of synchronization operation"""
    success: bool
    synced_actions: int
    failed_actions: int
    conflicts: int
    sync_duration_ms: int
    errors: List[str] = field(default_factory=list)


class OfflineManager:
    """Manages offline functionality and data synchronization"""

    def __init__(self, cache_strategy: CacheStrategy = CacheStrategy.SELECTIVE):
        self.logger = logging.getLogger(__name__)
        self.cache_strategy = cache_strategy
        self.network_state = NetworkState.ONLINE
        self._offline_queue: List[OfflineAction] = []
        self._cache: Dict[str, CacheItem] = {}
        self._sync_in_progress = False
        self._max_cache_size_mb = 50
        self._sync_interval = 30  # seconds

        # Critical data patterns for selective caching
        self._critical_patterns = [
            "patient_vitals_*",
            "patient_alerts_*",
            "emergency_*",
            "rapid_response_*"
        ]

    async def set_network_state(self, state: NetworkState):
        """Update network connectivity state"""
        old_state = self.network_state
        self.network_state = state

        if old_state == NetworkState.OFFLINE and state in [NetworkState.ONLINE, NetworkState.LIMITED]:
            self.logger.info("Network connectivity restored, initiating sync")
            await self.sync_offline_actions()

        elif state == NetworkState.OFFLINE:
            self.logger.warning("Network connectivity lost, enabling offline mode")

    async def queue_offline_action(self, action_type: str, patient_id: str,
                                 data: Dict[str, Any], priority: int = 3) -> str:
        """Queue an action for offline execution"""
        try:
            action_id = self._generate_action_id(action_type, patient_id)

            action = OfflineAction(
                action_id=action_id,
                action_type=action_type,
                patient_id=patient_id,
                data=data,
                timestamp=datetime.now(),
                priority=priority
            )

            self._offline_queue.append(action)
            self._offline_queue.sort(key=lambda x: (x.priority, x.timestamp))

            self.logger.info(f"Queued offline action: {action_type} for patient {patient_id}")

            # Attempt immediate sync if online
            if self.network_state == NetworkState.ONLINE:
                asyncio.create_task(self.sync_offline_actions())

            return action_id

        except Exception as e:
            self.logger.error(f"Error queuing offline action: {e}")
            raise

    async def sync_offline_actions(self) -> SyncResult:
        """Synchronize queued offline actions with server"""
        if self._sync_in_progress or self.network_state == NetworkState.OFFLINE:
            return SyncResult(success=False, synced_actions=0, failed_actions=0,
                            conflicts=0, sync_duration_ms=0, errors=["Sync not available"])

        self._sync_in_progress = True
        start_time = datetime.now()

        try:
            synced_count = 0
            failed_count = 0
            conflict_count = 0
            errors = []

            # Process actions by priority
            pending_actions = [a for a in self._offline_queue if a.status == SyncStatus.PENDING]

            for action in pending_actions:
                try:
                    action.status = SyncStatus.IN_PROGRESS

                    # Simulate sync process
                    sync_success = await self._sync_single_action(action)

                    if sync_success:
                        action.status = SyncStatus.COMPLETED
                        synced_count += 1
                        self.logger.info(f"Synced action {action.action_id}")
                    else:
                        action.retry_count += 1
                        if action.retry_count >= action.max_retries:
                            action.status = SyncStatus.FAILED
                            failed_count += 1
                            errors.append(f"Max retries exceeded for {action.action_id}")
                        else:
                            action.status = SyncStatus.PENDING

                except Exception as e:
                    action.status = SyncStatus.FAILED
                    failed_count += 1
                    errors.append(f"Error syncing {action.action_id}: {str(e)}")

            # Remove completed actions
            self._offline_queue = [a for a in self._offline_queue
                                 if a.status != SyncStatus.COMPLETED]

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return SyncResult(
                success=failed_count == 0,
                synced_actions=synced_count,
                failed_actions=failed_count,
                conflicts=conflict_count,
                sync_duration_ms=duration_ms,
                errors=errors
            )

        finally:
            self._sync_in_progress = False

    async def _sync_single_action(self, action: OfflineAction) -> bool:
        """Sync individual action with server (simulated)"""
        try:
            # Simulate network delay
            await asyncio.sleep(0.1)

            # Simulate different sync scenarios
            if action.action_type == "vital_signs_entry":
                # High success rate for vital signs
                return True
            elif action.action_type == "emergency_escalation":
                # Critical actions always succeed
                return True
            elif action.action_type == "clinical_notes":
                # Moderate success rate for notes
                import random
                return random.random() > 0.1
            else:
                return True

        except Exception as e:
            self.logger.error(f"Error in single action sync: {e}")
            return False

    async def cache_data(self, key: str, data: Any, ttl_seconds: int = 3600):
        """Cache data for offline access"""
        try:
            # Check if we should cache this data based on strategy
            if not self._should_cache(key):
                return

            # Check cache size limits
            await self._manage_cache_size()

            cache_item = CacheItem(
                key=key,
                data=data,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=ttl_seconds),
                size_bytes=len(json.dumps(data, default=str))
            )

            self._cache[key] = cache_item
            self.logger.debug(f"Cached data: {key}")

        except Exception as e:
            self.logger.error(f"Error caching data for key {key}: {e}")

    async def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data"""
        try:
            cache_item = self._cache.get(key)
            if not cache_item:
                return None

            if cache_item.is_expired():
                del self._cache[key]
                return None

            cache_item.access_count += 1
            return cache_item.data

        except Exception as e:
            self.logger.error(f"Error retrieving cached data for key {key}: {e}")
            return None

    def _should_cache(self, key: str) -> bool:
        """Determine if data should be cached based on strategy"""
        if self.cache_strategy == CacheStrategy.AGGRESSIVE:
            return True
        elif self.cache_strategy == CacheStrategy.MINIMAL:
            return any(pattern.replace("*", "") in key for pattern in self._critical_patterns)
        else:  # SELECTIVE
            # Cache patient data, vital signs, alerts
            return any(pattern in key for pattern in [
                "patient_", "vital_", "alert_", "emergency_", "ward_overview"
            ])

    async def _manage_cache_size(self):
        """Manage cache size to stay within limits"""
        try:
            total_size_mb = sum(item.size_bytes for item in self._cache.values()) / (1024 * 1024)

            if total_size_mb > self._max_cache_size_mb:
                # Remove least recently used items
                sorted_items = sorted(
                    self._cache.items(),
                    key=lambda x: (x[1].access_count, x[1].timestamp)
                )

                # Remove oldest 25% of items
                items_to_remove = len(sorted_items) // 4
                for key, _ in sorted_items[:items_to_remove]:
                    del self._cache[key]

                self.logger.info(f"Cache cleanup: removed {items_to_remove} items")

        except Exception as e:
            self.logger.error(f"Error managing cache size: {e}")

    def _generate_action_id(self, action_type: str, patient_id: str) -> str:
        """Generate unique action ID"""
        timestamp = datetime.now().isoformat()
        content = f"{action_type}_{patient_id}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def get_offline_capabilities(self) -> Dict[str, Any]:
        """Get current offline capabilities and status"""
        cache_size_mb = sum(item.size_bytes for item in self._cache.values()) / (1024 * 1024)
        fresh_cache_count = sum(1 for item in self._cache.values() if item.is_fresh())

        return {
            "network_state": self.network_state.value,
            "cache_strategy": self.cache_strategy.value,
            "offline_queue": {
                "total_actions": len(self._offline_queue),
                "pending": len([a for a in self._offline_queue if a.status == SyncStatus.PENDING]),
                "failed": len([a for a in self._offline_queue if a.status == SyncStatus.FAILED])
            },
            "cache_status": {
                "total_items": len(self._cache),
                "fresh_items": fresh_cache_count,
                "size_mb": round(cache_size_mb, 2),
                "max_size_mb": self._max_cache_size_mb
            },
            "sync_status": {
                "in_progress": self._sync_in_progress,
                "last_sync": "N/A",  # Would track actual last sync time
                "sync_interval_seconds": self._sync_interval
            },
            "capabilities": {
                "vital_signs_entry": True,
                "patient_viewing": True,
                "alert_acknowledgment": True,
                "clinical_notes": True,
                "emergency_escalation": True,
                "photo_capture": False,  # Requires online processing
                "voice_input": False     # Requires online processing
            }
        }

    async def prepare_for_offline(self, patient_ids: List[str]):
        """Pre-cache essential data for specified patients"""
        try:
            for patient_id in patient_ids:
                # Cache essential patient data
                await self._cache_patient_essentials(patient_id)

            self.logger.info(f"Prepared offline data for {len(patient_ids)} patients")

        except Exception as e:
            self.logger.error(f"Error preparing offline data: {e}")

    async def _cache_patient_essentials(self, patient_id: str):
        """Cache essential data for a patient"""
        try:
            # Simulate caching essential patient data
            essential_data = {
                "patient_info": {
                    "patient_id": patient_id,
                    "name": f"Patient {patient_id}",
                    "age": 65,
                    "bed_number": "A01"
                },
                "current_vitals": {
                    "respiratory_rate": 18,
                    "spo2": 95,
                    "temperature": 37.2,
                    "heart_rate": 80
                },
                "alerts": [],
                "last_updated": datetime.now().isoformat()
            }

            await self.cache_data(f"patient_essentials_{patient_id}", essential_data, 7200)  # 2 hours

        except Exception as e:
            self.logger.error(f"Error caching essentials for patient {patient_id}: {e}")

    async def handle_network_change(self, network_info: Dict[str, Any]):
        """Handle network state changes"""
        connection_type = network_info.get("type", "unknown")
        is_online = network_info.get("online", True)
        connection_speed = network_info.get("effective_type", "4g")

        if not is_online:
            await self.set_network_state(NetworkState.OFFLINE)
        elif connection_speed in ["slow-2g", "2g"]:
            await self.set_network_state(NetworkState.LIMITED)
        else:
            await self.set_network_state(NetworkState.ONLINE)

        return {
            "network_state": self.network_state.value,
            "optimizations_applied": True,
            "cache_strategy": self.cache_strategy.value
        }

    def get_offline_queue_summary(self) -> Dict[str, Any]:
        """Get summary of offline action queue"""
        queue_by_type = {}
        queue_by_status = {}

        for action in self._offline_queue:
            # Count by type
            queue_by_type[action.action_type] = queue_by_type.get(action.action_type, 0) + 1
            # Count by status
            queue_by_status[action.status.value] = queue_by_status.get(action.status.value, 0) + 1

        return {
            "total_actions": len(self._offline_queue),
            "by_type": queue_by_type,
            "by_status": queue_by_status,
            "oldest_action": min(self._offline_queue, key=lambda x: x.timestamp).timestamp.isoformat() if self._offline_queue else None
        }