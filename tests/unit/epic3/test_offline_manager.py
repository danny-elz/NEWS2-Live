"""
Unit tests for Offline Manager (Story 3.4)
Tests offline functionality and data synchronization
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.mobile.offline_manager import (
    OfflineManager,
    SyncStatus,
    CacheStrategy,
    NetworkState,
    OfflineAction,
    CacheItem,
    SyncResult
)


class TestOfflineManager:
    """Test suite for OfflineManager"""

    @pytest.fixture
    def offline_manager(self):
        """Create OfflineManager instance"""
        return OfflineManager(CacheStrategy.SELECTIVE)

    @pytest.fixture
    def sample_action_data(self):
        """Create sample action data"""
        return {
            "patient_id": "P001",
            "vital_signs": {
                "respiratory_rate": 18,
                "spo2": 95,
                "temperature": 37.2
            },
            "timestamp": datetime.now().isoformat(),
            "entered_by": "Nurse Johnson"
        }

    @pytest.mark.asyncio
    async def test_queue_offline_action(self, offline_manager, sample_action_data):
        """Test queuing actions for offline execution"""
        action_id = await offline_manager.queue_offline_action(
            "vital_signs_entry",
            "P001",
            sample_action_data,
            priority=1
        )

        assert action_id is not None
        assert len(offline_manager._offline_queue) == 1

        queued_action = offline_manager._offline_queue[0]
        assert queued_action.action_id == action_id
        assert queued_action.action_type == "vital_signs_entry"
        assert queued_action.patient_id == "P001"
        assert queued_action.priority == 1
        assert queued_action.status == SyncStatus.PENDING

    @pytest.mark.asyncio
    async def test_queue_multiple_actions_priority_sorting(self, offline_manager):
        """Test that queued actions are sorted by priority"""
        # Queue actions with different priorities
        await offline_manager.queue_offline_action("low_priority", "P001", {}, priority=4)
        await offline_manager.queue_offline_action("high_priority", "P002", {}, priority=1)
        await offline_manager.queue_offline_action("medium_priority", "P003", {}, priority=2)

        # Check that they are sorted by priority
        priorities = [action.priority for action in offline_manager._offline_queue]
        assert priorities == [1, 2, 4]

        action_types = [action.action_type for action in offline_manager._offline_queue]
        assert action_types == ["high_priority", "medium_priority", "low_priority"]

    @pytest.mark.asyncio
    async def test_sync_offline_actions_success(self, offline_manager, sample_action_data):
        """Test successful synchronization of offline actions"""
        # Queue some actions
        await offline_manager.queue_offline_action("vital_signs_entry", "P001", sample_action_data)
        await offline_manager.queue_offline_action("clinical_notes", "P002", {"note": "test"})

        # Mock successful sync
        with patch.object(offline_manager, '_sync_single_action', return_value=True):
            result = await offline_manager.sync_offline_actions()

            assert result.success
            assert result.synced_actions == 2
            assert result.failed_actions == 0
            assert result.conflicts == 0
            assert len(result.errors) == 0

        # Check that completed actions are removed from queue
        assert len(offline_manager._offline_queue) == 0

    @pytest.mark.asyncio
    async def test_sync_offline_actions_failures(self, offline_manager, sample_action_data):
        """Test synchronization with some failures"""
        # Queue actions
        await offline_manager.queue_offline_action("vital_signs_entry", "P001", sample_action_data)
        await offline_manager.queue_offline_action("emergency_escalation", "P002", {})

        # Mock mixed success/failure
        async def mock_sync(action):
            return action.action_type == "emergency_escalation"  # Only emergency succeeds

        with patch.object(offline_manager, '_sync_single_action', side_effect=mock_sync):
            result = await offline_manager.sync_offline_actions()

            assert not result.success  # Overall failure due to one failed action
            assert result.synced_actions == 1
            assert result.failed_actions == 1

        # Check that failed action is still in queue (with retry count)
        remaining_actions = [a for a in offline_manager._offline_queue if a.status == SyncStatus.PENDING]
        assert len(remaining_actions) == 1
        assert remaining_actions[0].retry_count == 1

    @pytest.mark.asyncio
    async def test_sync_max_retries_exceeded(self, offline_manager, sample_action_data):
        """Test action marked as failed after max retries"""
        # Create an action that has already been retried
        action_id = await offline_manager.queue_offline_action("failing_action", "P001", sample_action_data)

        # Manually set retry count to max
        offline_manager._offline_queue[0].retry_count = 3

        # Mock sync failure
        with patch.object(offline_manager, '_sync_single_action', return_value=False):
            result = await offline_manager.sync_offline_actions()

            assert result.failed_actions == 1
            assert "Max retries exceeded" in " ".join(result.errors)

        # Check that action is marked as failed
        failed_actions = [a for a in offline_manager._offline_queue if a.status == SyncStatus.FAILED]
        assert len(failed_actions) == 1

    @pytest.mark.asyncio
    async def test_network_state_changes(self, offline_manager):
        """Test network state change handling"""
        # Start offline
        await offline_manager.set_network_state(NetworkState.OFFLINE)
        assert offline_manager.network_state == NetworkState.OFFLINE

        # Queue an action while offline
        await offline_manager.queue_offline_action("offline_action", "P001", {})

        # Mock sync to verify it's called when going online
        with patch.object(offline_manager, 'sync_offline_actions', new_callable=AsyncMock) as mock_sync:
            await offline_manager.set_network_state(NetworkState.ONLINE)

            assert offline_manager.network_state == NetworkState.ONLINE
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_data_selective_strategy(self, offline_manager):
        """Test caching with selective strategy"""
        # Should cache patient data
        await offline_manager.cache_data("patient_P001_vitals", {"data": "test"})
        cached_data = await offline_manager.get_cached_data("patient_P001_vitals")
        assert cached_data == {"data": "test"}

        # Should cache vital signs
        await offline_manager.cache_data("vital_signs_P001", {"rr": 18})
        cached_vitals = await offline_manager.get_cached_data("vital_signs_P001")
        assert cached_vitals == {"rr": 18}

        # Should not cache non-critical data (depending on implementation)
        await offline_manager.cache_data("non_critical_data", {"test": "data"})
        # Implementation may or may not cache this based on selective strategy

    @pytest.mark.asyncio
    async def test_cache_data_expiration(self, offline_manager):
        """Test cache data expiration"""
        # Cache data with short TTL
        await offline_manager.cache_data("temp_data", {"test": "data"}, ttl_seconds=1)

        # Should be available immediately
        data = await offline_manager.get_cached_data("temp_data")
        assert data == {"test": "data"}

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        data = await offline_manager.get_cached_data("temp_data")
        assert data is None

    @pytest.mark.asyncio
    async def test_cache_size_management(self, offline_manager):
        """Test cache size management"""
        # Fill cache with data
        for i in range(100):
            large_data = {"data": "x" * 1000}  # 1KB of data
            await offline_manager.cache_data(f"large_item_{i}", large_data)

        # Trigger cache cleanup by adding more data
        await offline_manager._manage_cache_size()

        # Cache should be smaller after cleanup
        assert len(offline_manager._cache) < 100

    @pytest.mark.asyncio
    async def test_get_offline_capabilities(self, offline_manager):
        """Test getting offline capabilities status"""
        # Queue some actions
        await offline_manager.queue_offline_action("action1", "P001", {})
        await offline_manager.queue_offline_action("action2", "P002", {})

        # Cache some data
        await offline_manager.cache_data("test_data", {"test": True})

        capabilities = await offline_manager.get_offline_capabilities()

        assert capabilities["network_state"] == "online"  # Default state
        assert capabilities["cache_strategy"] == "selective"
        assert capabilities["offline_queue"]["total_actions"] == 2
        assert capabilities["offline_queue"]["pending"] == 2
        assert capabilities["cache_status"]["total_items"] >= 1
        assert "capabilities" in capabilities

        # Check specific capabilities
        caps = capabilities["capabilities"]
        assert caps["vital_signs_entry"]
        assert caps["patient_viewing"]
        assert caps["emergency_escalation"]

    @pytest.mark.asyncio
    async def test_prepare_for_offline(self, offline_manager):
        """Test preparing essential data for offline use"""
        patient_ids = ["P001", "P002", "P003"]

        await offline_manager.prepare_for_offline(patient_ids)

        # Check that essential data was cached for each patient
        for patient_id in patient_ids:
            cached_data = await offline_manager.get_cached_data(f"patient_essentials_{patient_id}")
            assert cached_data is not None
            assert cached_data["patient_info"]["patient_id"] == patient_id

    @pytest.mark.asyncio
    async def test_handle_network_change(self, offline_manager):
        """Test handling network state changes"""
        # Test going offline
        network_info = {"online": False, "type": "none"}
        result = await offline_manager.handle_network_change(network_info)

        assert result["network_state"] == "offline"
        assert offline_manager.network_state == NetworkState.OFFLINE

        # Test limited connection
        network_info = {"online": True, "type": "cellular", "effective_type": "2g"}
        result = await offline_manager.handle_network_change(network_info)

        assert result["network_state"] == "limited"
        assert offline_manager.network_state == NetworkState.LIMITED

        # Test good connection
        network_info = {"online": True, "type": "wifi", "effective_type": "4g"}
        result = await offline_manager.handle_network_change(network_info)

        assert result["network_state"] == "online"
        assert offline_manager.network_state == NetworkState.ONLINE

    def test_get_offline_queue_summary(self, offline_manager):
        """Test getting offline queue summary"""
        # Start with empty queue
        summary = offline_manager.get_offline_queue_summary()
        assert summary["total_actions"] == 0
        assert summary["by_type"] == {}
        assert summary["by_status"] == {}
        assert summary["oldest_action"] is None

    @pytest.mark.asyncio
    async def test_sync_not_allowed_when_offline(self, offline_manager):
        """Test that sync is not performed when offline"""
        await offline_manager.set_network_state(NetworkState.OFFLINE)
        await offline_manager.queue_offline_action("test_action", "P001", {})

        result = await offline_manager.sync_offline_actions()

        assert not result.success
        assert "Sync not available" in result.errors[0]
        assert result.synced_actions == 0

    @pytest.mark.asyncio
    async def test_sync_not_allowed_when_in_progress(self, offline_manager):
        """Test that concurrent sync operations are prevented"""
        offline_manager._sync_in_progress = True

        result = await offline_manager.sync_offline_actions()

        assert not result.success
        assert "Sync not available" in result.errors[0]


class TestOfflineAction:
    """Test suite for OfflineAction class"""

    def test_offline_action_creation(self):
        """Test creating offline action"""
        action = OfflineAction(
            action_id="test_123",
            action_type="vital_signs_entry",
            patient_id="P001",
            data={"respiratory_rate": 18},
            timestamp=datetime.now(),
            priority=1
        )

        assert action.action_id == "test_123"
        assert action.action_type == "vital_signs_entry"
        assert action.patient_id == "P001"
        assert action.priority == 1
        assert action.status == SyncStatus.PENDING
        assert action.retry_count == 0

    def test_offline_action_to_dict(self):
        """Test offline action dictionary conversion"""
        now = datetime.now()
        action = OfflineAction(
            action_id="test_123",
            action_type="test_action",
            patient_id="P001",
            data={"test": True},
            timestamp=now
        )

        action_dict = action.to_dict()

        assert action_dict["action_id"] == "test_123"
        assert action_dict["action_type"] == "test_action"
        assert action_dict["patient_id"] == "P001"
        assert action_dict["data"] == {"test": True}
        assert action_dict["timestamp"] == now.isoformat()
        assert action_dict["status"] == "pending"


class TestCacheItem:
    """Test suite for CacheItem class"""

    def test_cache_item_creation(self):
        """Test creating cache item"""
        now = datetime.now()
        item = CacheItem(
            key="test_key",
            data={"test": "data"},
            timestamp=now,
            expires_at=now + timedelta(hours=1),
            size_bytes=1024
        )

        assert item.key == "test_key"
        assert item.data == {"test": "data"}
        assert item.timestamp == now
        assert item.size_bytes == 1024

    def test_cache_item_expiration(self):
        """Test cache item expiration logic"""
        now = datetime.now()

        # Not expired item
        item = CacheItem(
            key="test",
            data="data",
            timestamp=now,
            expires_at=now + timedelta(hours=1)
        )
        assert not item.is_expired()

        # Expired item
        expired_item = CacheItem(
            key="test",
            data="data",
            timestamp=now,
            expires_at=now - timedelta(hours=1)
        )
        assert expired_item.is_expired()

        # Item with no expiration
        no_expire_item = CacheItem(
            key="test",
            data="data",
            timestamp=now
        )
        assert not no_expire_item.is_expired()

    def test_cache_item_freshness(self):
        """Test cache item freshness check"""
        now = datetime.now()

        # Fresh item
        fresh_item = CacheItem(
            key="test",
            data="data",
            timestamp=now
        )
        assert fresh_item.is_fresh(max_age_seconds=3600)

        # Stale item
        stale_item = CacheItem(
            key="test",
            data="data",
            timestamp=now - timedelta(hours=2)
        )
        assert not stale_item.is_fresh(max_age_seconds=3600)


class TestCacheStrategies:
    """Test suite for different cache strategies"""

    def test_aggressive_caching(self):
        """Test aggressive caching strategy"""
        manager = OfflineManager(CacheStrategy.AGGRESSIVE)

        # Should cache everything
        assert manager._should_cache("any_random_key")
        assert manager._should_cache("patient_data")
        assert manager._should_cache("non_critical_info")

    def test_selective_caching(self):
        """Test selective caching strategy"""
        manager = OfflineManager(CacheStrategy.SELECTIVE)

        # Should cache patient-related data
        assert manager._should_cache("patient_P001_data")
        assert manager._should_cache("vital_signs_reading")
        assert manager._should_cache("alert_notification")
        assert manager._should_cache("emergency_response")

        # Should not cache non-essential data
        assert not manager._should_cache("random_metadata")
        assert not manager._should_cache("system_logs")

    def test_minimal_caching(self):
        """Test minimal caching strategy"""
        manager = OfflineManager(CacheStrategy.MINIMAL)

        # Should only cache critical patterns
        assert manager._should_cache("patient_vitals_P001")
        assert manager._should_cache("emergency_alert_123")

        # Should not cache regular patient data
        assert not manager._should_cache("patient_general_info")
        assert not manager._should_cache("vital_signs_routine")